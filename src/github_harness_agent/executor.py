"""Execute repository tasks and verify their patches in independent workspaces."""

import hashlib
import json
import logging
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from fnmatch import fnmatchcase
from math import isfinite
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any, Protocol
from uuid import uuid4

from .activity import observed_call
from .checkpoint import DraftCheckpoint
from .file_edit import EDIT_COMMAND, EDIT_TOOL, InvalidEdit, edit_payload
from .providers import ProviderExhausted
from .test_updates import (
    focused_pytest,
    has_runnable_test,
    passed_pytest_report,
    valid_test_update,
)
from .task_context import (
    CONTEXT_TOOL,
    MESSAGE_BYTES,
    RETRY_MESSAGE_BYTES,
    TOOL_RESULT_BYTES,
    TaskContext,
    compact_messages,
    context_tool_result,
    excerpt,
)
from .worker import (
    Cancelled,
    CommandOutputExceeded,
    DockerRunner,
    ExecutionError,
    LimitExceeded,
    run_process,
)

__all__ = [
    "Cancelled",
    "AttemptRejected",
    "CheckEvidence",
    "DockerRunner",
    "ExecutionError",
    "ExecutionResult",
    "Harness2Executor",
    "LimitExceeded",
    "Limits",
    "NeedsClarification",
    "PlainExecutor",
    "TaskSnapshot",
]

SYSTEM_PROMPT = (
    "Implement the repository task in the pinned checkout. Use run for narrow source reads and "
    "checks. Use edit_file to replace exact text or create a source or test file; act once you "
    "have read enough to locate the change. Keep the patch focused and preserve existing checks "
    "and their configuration. For review repairs, make the requested changes on the current "
    "branch. Add requested coverage in new test files using the repository's naming convention "
    "unless the task specifically authorizes an existing test update. "
    "Inspect git diff and git status, and run the relevant tests before finishing. "
    "If the task has conflicting or materially ambiguous requirements, return only a JSON object "
    "with a clarification key containing one concise question for the maintainer."
)
RUN_TOOL = {
    "type": "function",
    "function": {
        "name": "run",
        "description": "Run an argument vector in the disposable checkout. No shell expansion.",
        "parameters": {
            "type": "object",
            "properties": {"argv": {"type": "array", "items": {"type": "string"}}},
            "required": ["argv"],
            "additionalProperties": False,
        },
    },
}
REVIEW_ROLLOUT_TURNS = 12
REVIEW_EDIT_REMINDER_TURN = 3


class Gateway(Protocol):
    def complete(
        self,
        *,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
        timeout: float | None = None,
        exclude: Sequence[tuple[str, str]] = (),
        provider_offset: int = 0,
        after_provider: str | None = None,
    ): ...


class NeedsClarification(ExecutionError):
    pass


class AttemptRejected(ExecutionError):
    def __init__(self, message, candidate, route_trace):
        super().__init__(message)
        self.candidate = candidate
        self.route_trace = route_trace


class AttemptBudgetRejected(AttemptRejected, LimitExceeded):
    pass


class _RequestBudgetExceeded(LimitExceeded):
    pass


class _CandidateOutputError(ExecutionError):
    pass


class _ProtectedPatchError(_CandidateOutputError):
    repair_feedback = (
        "The candidate changed protected content outside the allowed scope. "
        "Restore protected regions to the pinned baseline and keep any authorized "
        "test update limited to the requested region. Put additional tests in a "
        "separate new test module."
    )


class _VerificationRejected(ExecutionError):
    """A trusted check ran and rejected the candidate."""


class _MissingRegressionTests(_VerificationRejected):
    """A review requires new runnable test coverage."""


@dataclass(frozen=True)
class TaskSnapshot:
    repository: str
    base_sha: str
    instructions: str
    checks: tuple[tuple[str, ...], ...]
    seed: int = 0
    protected_paths: tuple[str, ...] = ()
    repository_guidance: str = ""
    excluded_models: tuple[tuple[str, str], ...] = ()
    provider_offset: int = 0
    observe_model: Callable | None = None
    checkpoint_scope: str = ""
    checkpoints: Mapping[str, DraftCheckpoint] = field(default_factory=dict)
    save_checkpoint: Callable[[DraftCheckpoint], None] | None = None
    context: TaskContext | None = None
    verify_base_first: bool = False
    test_update: str | None = None
    require_new_tests: bool = False


@dataclass(frozen=True)
class Limits:
    wall_time: float = 900
    model_requests: int = 40
    concurrent_probes: int = 2
    patch_bytes: int = 262144
    commands: int = 64
    archive_bytes: int = 67108864
    command_output_bytes: int = 65536
    rollout_requests: int | None = None

    def __post_init__(self):
        if (
            not isfinite(self.wall_time)
            or min(
                self.wall_time,
                self.model_requests,
                self.concurrent_probes,
                self.patch_bytes,
                self.commands,
                self.archive_bytes,
                self.command_output_bytes,
            )
            <= 0
        ):
            raise ValueError("Executor limits must be positive")
        if self.rollout_requests is not None and (
            type(self.rollout_requests) is not int
            or not 1 <= self.rollout_requests <= self.model_requests
        ):
            raise ValueError(
                "rollout_requests must be an integer within the model request budget"
            )


@dataclass(frozen=True)
class CheckEvidence:
    command: tuple[str, ...]
    returncode: int
    output: str


@dataclass(frozen=True)
class RouteEvidence:
    role: str
    route: str
    model: str
    usage: dict[str, int]
    configured_model: str | None = None

    @classmethod
    def from_completion(cls, completion):
        return cls(
            completion.role,
            completion.route,
            completion.model,
            completion.usage,
            getattr(completion, "configured_model", None),
        )


@dataclass(frozen=True)
class CandidateEvidence:
    label: str
    patch_digest: str | None
    trajectory: str
    error: str = ""


@dataclass(frozen=True)
class RefinementEvidence:
    version: str
    seed: int
    repository: str
    instructions: str
    harness_guidance: dict[str, str]
    stages: tuple[dict[str, Any], ...]
    candidates: tuple[CandidateEvidence, ...]
    selected_candidate: str | None = None


@dataclass(frozen=True)
class ExecutionResult:
    patch: bytes
    checks: tuple[CheckEvidence, ...]
    route_trace: tuple[RouteEvidence, ...]
    usage: dict[str, int]
    patch_digest: str
    refinement: RefinementEvidence | None = None


class _Budget:
    def __init__(
        self,
        limits: Limits,
        cancelled: Callable[[], bool],
        task: TaskSnapshot,
    ):
        self.limits = limits
        self.cancelled = cancelled
        self.deadline = time.monotonic() + limits.wall_time
        self.routes: list[RouteEvidence] = []
        self.task = task
        self._calls = Counter()
        self._lock = Lock()
        self._message_limit = MESSAGE_BYTES

    def remaining(self):
        if self.cancelled():
            raise Cancelled("Task cancelled")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise LimitExceeded("Task wall time exceeded")
        return remaining

    def claim(self, kind: str, limit: int):
        with self._lock:
            self.remaining()
            if self._calls[kind] >= limit:
                raise _RequestBudgetExceeded(f"Task {kind} budget exceeded")
            self._calls[kind] += 1

    def complete(self, gateway, role, messages, tools=(), *, after_provider=None):
        options = (
            {"exclude": self.task.excluded_models} if self.task.excluded_models else {}
        )
        if after_provider:
            options["after_provider"] = after_provider
        elif self.task.provider_offset:
            options["provider_offset"] = self.task.provider_offset

        def send(payload):
            self.claim("model request", self.limits.model_requests)
            return observed_call(
                self.task,
                gateway,
                role=role,
                messages=payload,
                tools=tools,
                timeout=self.remaining(),
                **options,
            )

        with self._lock:
            message_limit = self._message_limit
        compact = compact_messages(messages, max_bytes=message_limit, role=role)
        try:
            completion = send(compact)
        except ProviderExhausted as error:
            if "upstream HTTP 413" not in error.failures.values():
                raise
            smaller = compact_messages(
                messages, max_bytes=RETRY_MESSAGE_BYTES, role=role
            )
            if smaller == compact:
                raise
            with self._lock:
                self._message_limit = RETRY_MESSAGE_BYTES
            completion = send(smaller)
        self.remaining()
        with self._lock:
            route = RouteEvidence.from_completion(completion)
            if not self.routes or self.routes[-1].route != route.route:
                logging.getLogger(__name__).info(
                    "%s at %.12s received a %s completion from %s/%s",
                    self.task.repository,
                    self.task.base_sha,
                    role,
                    route.route,
                    route.configured_model,
                )
            self.routes.append(route)
        return completion

    def reject(self, error, candidate, *, latest=False):
        self.remaining()
        with self._lock:
            trace = tuple(self.routes)
        if candidate is None and latest and trace:
            candidate = trace[-1]
        if candidate is None or not candidate.configured_model:
            raise error
        kind = (
            AttemptBudgetRejected
            if isinstance(error, LimitExceeded)
            else AttemptRejected
        )
        raise kind(str(error), candidate, trace) from error


def _combined(result):
    return (result.stdout + result.stderr).decode("utf-8", "replace")


def _test_module(name: str) -> bool:
    return (
        name.startswith("test_")
        and name.endswith(".py")
        or name.endswith(
            ("_test.py", "_test.go", ".test.js", ".test.ts", ".spec.js", ".spec.ts")
        )
    )


def _protected(
    path: str, task: TaskSnapshot, *, new_test: bool, amended_test: bool = False
) -> bool:
    """Preserve existing tests, test infrastructure, and configured protected paths."""
    names = PurePosixPath(path).parts
    test_configs = {
        "conftest.py",
        "sitecustomize.py",
        "usercustomize.py",
        "pyproject.toml",
        "pytest.ini",
        "setup.cfg",
        "setup.py",
        "tox.ini",
        "noxfile.py",
        "Makefile",
        "GNUmakefile",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "justfile",
        "Taskfile.yml",
        "Taskfile.yaml",
        ".gitignore",
        ".gitattributes",
    }
    check_args = {
        str(PurePosixPath(arg))
        for command in task.checks
        for arg in command
        if not arg.startswith("-")
    }
    return (
        bool(set(names) & {".git", ".github", ".circleci"})
        or names[-1] in test_configs
        or (
            not (new_test or amended_test)
            and (
                bool(set(names) & {"tests", "test", "__tests__"})
                or names[-1].startswith("test_")
                or _test_module(names[-1])
            )
        )
        or path in check_args
        or any(fnmatchcase(path, pattern) for pattern in task.protected_paths)
    )


class PlainExecutor:
    def __init__(
        self,
        runner,
        gateway: Gateway,
        limits: Limits = Limits(),
        *,
        verification_runner=None,
    ):
        if limits.rollout_requests is not None and verification_runner is None:
            raise ValueError(
                "rollout_requests requires an explicit verification runner"
            )
        self.runner = runner
        self.verification_runner = (
            runner if verification_runner is None else verification_runner
        )
        self.gateway = gateway
        self.limits = limits

    def execute(
        self,
        task: TaskSnapshot,
        checkout: Path,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> ExecutionResult:
        budget = _Budget(self.limits, cancelled, task)
        archive = self._archive(task, checkout, budget)
        patch, _, actor = self._rollout(
            task,
            archive,
            budget,
            SYSTEM_PROMPT,
            "coding",
            checkpoint_key="plain/coding",
        )
        return self._verify(task, archive, budget, patch, actor=actor)

    def verify_patch(
        self,
        task: TaskSnapshot,
        checkout: Path,
        patch: bytes,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> ExecutionResult:
        budget = _Budget(self.limits, cancelled, task)
        return self._verify(task, self._archive(task, checkout, budget), budget, patch)

    def _archive(self, task, checkout, budget):
        budget.remaining()
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", task.base_sha):
            raise ExecutionError("A full immutable checkout SHA is required")
        if not task.checks or any(not command for command in task.checks):
            raise ExecutionError("At least one trusted required check is required")
        if len(task.repository_guidance) > 3000:
            raise ExecutionError("Repository guidance exceeds the configured size")
        try:
            result = run_process(
                [
                    "git",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "archive",
                    "--format=tar",
                    task.base_sha,
                ],
                budget.remaining(),
                budget.cancelled,
                cwd=checkout,
                cap=self.limits.archive_bytes,
            )
        except CommandOutputExceeded as error:
            raise LimitExceeded(
                f"Pinned checkout exceeds the archive limit ({self.limits.archive_bytes} bytes)"
            ) from error
        if result.returncode:
            raise ExecutionError("Could not archive the pinned checkout")
        return result.stdout

    def _command(self, workspace, budget, argv, *, data=b"", cap=None, checked=True):
        result = workspace.command(
            argv,
            budget.remaining(),
            budget.cancelled,
            data=data,
            cap=cap if cap is not None else self.limits.command_output_bytes,
        )
        if checked and result.returncode:
            raise ExecutionError(f"Worker command failed: {argv[0]}")
        return result

    def _git(self, workspace, budget, *argv, index=None, **kwargs):
        environment = ("env", f"GIT_INDEX_FILE={index}") if index else ()
        return self._command(
            workspace,
            budget,
            (*environment, "git", "-c", "core.hooksPath=/dev/null", *argv),
            **kwargs,
        )

    def _stage_baseline(self, workspace, budget):
        self._git(workspace, budget, "init", "-q", "--template=")
        self._git(workspace, budget, "add", "-A", "-f")
        return self._git(workspace, budget, "write-tree").stdout.strip().decode("ascii")

    def _draft_patch(self, workspace, budget, baseline, *, index=None):
        return self._git(
            workspace,
            budget,
            "diff",
            "--cached",
            "--binary",
            "--full-index",
            "--no-renames",
            "--no-ext-diff",
            "--no-textconv",
            baseline,
            cap=self.limits.patch_bytes,
            index=index,
        ).stdout

    def _restore_draft(self, workspace, task, budget, baseline, draft, guidance):
        if draft.base_sha != task.base_sha or draft.guidance != guidance:
            raise ExecutionError(
                "Saved draft does not match this checkout or candidate guidance"
            )
        if len(draft.patch) > self.limits.patch_bytes:
            raise LimitExceeded("Saved draft exceeded the configured patch size")
        result = self._git(
            workspace,
            budget,
            "apply",
            "--index",
            "--binary",
            data=draft.patch,
            checked=False,
        )
        if result.returncode:
            raise ExecutionError("Saved draft no longer applies to this checkout")
        self._check_patch_paths(workspace, task, budget, baseline)
        tree = self._git(workspace, budget, "write-tree").stdout.strip().decode("ascii")
        self._git(workspace, budget, "read-tree", baseline)
        return tree

    @staticmethod
    def _command_progress(argv, returncode):
        test_tools = {"pytest", "ruff", "mypy", "vitest", "mocha", "jest", "unittest"}
        check = next((arg for arg in argv if arg in test_tools), None)
        executable = PurePosixPath(argv[0]).name
        label = check or (
            executable if executable in {"python", "python3", "git"} else "command"
        )
        status = (
            "exceeded output limit" if returncode is None else f"exited {returncode}"
        )
        return f"{label} {status}", check is not None

    def _rollout(
        self,
        task,
        archive,
        budget,
        prompt,
        role,
        *,
        clarification=True,
        checkpoint_key=None,
        checkpoint_guidance=None,
        seed_draft=None,
    ):
        with self.runner.workspace(
            archive, budget.remaining, budget.cancelled
        ) as workspace:
            baseline = self._stage_baseline(workspace, budget)
            own_draft = task.checkpoints.get(checkpoint_key)
            draft = own_draft or seed_draft
            if draft is not None and draft.key != checkpoint_key:
                raise ExecutionError("Saved draft belongs to a different rollout")
            staged_tree = baseline
            progress = []
            instructions = task.instructions
            if draft is not None and draft.patch:
                staged_tree = self._restore_draft(
                    workspace,
                    task,
                    budget,
                    baseline,
                    draft,
                    checkpoint_guidance,
                )
                progress.extend(draft.progress)
                instructions += (
                    "\n\nResuming a saved draft already applied to this checkout. "
                    "Inspect the existing diff and preserve valid edits while completing the task. "
                    "Earlier command outcomes (rerun required checks before finishing): "
                    + "; ".join(progress)
                )
            checkpoint_index = None
            if checkpoint_key is not None and task.save_checkpoint is not None:
                checkpoint_index = f".git/gha-draft-index-{uuid4().hex}"
                self._git(
                    workspace, budget, "read-tree", staged_tree, index=checkpoint_index
                )
            if task.test_update is not None:
                prompt += (
                    "\nA maintainer authorized this existing test update: "
                    f"{task.test_update}. Edit only the named test body and keep its assertions active."
                )
            if task.require_new_tests:
                prompt += (
                    "\nThe review requires new runnable regression coverage. "
                    "Add a new test module for the requested behavior."
                )
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": instructions},
            ]
            tools = (RUN_TOOL, EDIT_TOOL)
            if task.context is not None:
                tools += (CONTEXT_TOOL,)
            trajectory = []
            actor = None
            last = None
            turns = 0
            limit = budget.limits.rollout_requests
            if task.verify_base_first and limit is None:
                limit = min(REVIEW_ROLLOUT_TURNS, budget.limits.model_requests)
            while limit is None or turns < limit:
                try:
                    last_editor = (
                        own_draft.last_editor if own_draft and not turns else None
                    )
                    completion = budget.complete(
                        self.gateway,
                        role,
                        messages,
                        tools,
                        after_provider=last_editor[0] if last_editor else None,
                    )
                except _RequestBudgetExceeded as error:
                    budget.reject(error, last, latest=True)
                turns += 1
                last = RouteEvidence.from_completion(completion)
                if not completion.tool_calls:
                    if clarification and completion.content:
                        self._check_clarification(completion.content)
                    break
                assistant = {
                    "role": "assistant",
                    "content": completion.content,
                    "tool_calls": list(completion.tool_calls),
                }
                if last.configured_model:
                    assistant["gateway_route"] = [last.route, last.configured_model]
                if provider_context := getattr(completion, "provider_context", None):
                    assistant["provider_context"] = provider_context
                messages.append(assistant)
                ran_command = ran_check = False
                for tool in completion.tool_calls:
                    if (
                        task.context is not None
                        and tool["function"]["name"] == "read_task_context"
                    ):
                        try:
                            budget.claim("context read", 40)
                        except _RequestBudgetExceeded as error:
                            budget.reject(error, last)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool["id"],
                                "content": context_tool_result(task.context, tool),
                            }
                        )
                        continue
                    is_edit = tool["function"]["name"] == "edit_file"
                    touched = False
                    try:
                        budget.claim("command", self.limits.commands)
                        if is_edit:
                            action = "edit_file"
                            data = edit_payload(tool, self.limits.patch_bytes)
                            response = self._command(
                                workspace,
                                budget,
                                EDIT_COMMAND,
                                data=data,
                                checked=False,
                            )
                        else:
                            argv = self._tool_argv(tool)
                            action = json.dumps(argv)
                            response = self._command(
                                workspace, budget, argv, checked=False
                            )
                        touched = not is_edit or response.returncode == 0
                    except InvalidEdit as error:
                        output = str(error)
                    except CommandOutputExceeded:
                        touched = True
                        returncode = None
                        output = (
                            "Command exceeded the output limit. Select a narrower range or "
                            "redirect output to a workspace file and inspect a slice."
                        )
                    except (_RequestBudgetExceeded, _CandidateOutputError) as error:
                        budget.reject(error, last)
                    else:
                        returncode = response.returncode
                        output = excerpt(
                            f"exit {response.returncode}\n{_combined(response)}",
                            TOOL_RESULT_BYTES,
                        )
                    if touched:
                        actor = last
                        ran_command = True
                    if touched and task.save_checkpoint is not None:
                        outcome, is_check = (
                            ("edit_file updated checkout", False)
                            if is_edit
                            else self._command_progress(argv, returncode)
                        )
                        progress = [*progress, outcome][-12:]
                        ran_check |= is_check
                    trajectory.append(f"[ACTION] {action}\n[OBSERVATION] {output}")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool["id"],
                            "content": output,
                        }
                    )
                if ran_command and checkpoint_index is not None:
                    self._git(workspace, budget, "add", "-A", index=checkpoint_index)
                    tree = (
                        self._git(
                            workspace,
                            budget,
                            "write-tree",
                            index=checkpoint_index,
                        )
                        .stdout.strip()
                        .decode("ascii")
                    )
                    if tree != staged_tree or (
                        ran_check and draft is not None and draft.patch
                    ):
                        try:
                            patch = (
                                draft.patch
                                if tree == staged_tree
                                else self._draft_patch(
                                    workspace,
                                    budget,
                                    baseline,
                                    index=checkpoint_index,
                                )
                            )
                            if patch and tree != staged_tree:
                                self._check_patch_paths(
                                    workspace,
                                    task,
                                    budget,
                                    baseline,
                                    index=checkpoint_index,
                                )
                        except _ProtectedPatchError as error:
                            # Keep the previous safe draft and let the model correct the checkout.
                            if task.verify_base_first:
                                feedback = f"[PATCH VALIDATION] {error.repair_feedback}"
                                messages[-1]["content"] += f"\n\n{feedback}"
                                trajectory.append(feedback)
                        except (CommandOutputExceeded, _CandidateOutputError):
                            # Keep the last safe draft while the model fixes transient edits.
                            pass
                        else:
                            editing_route = (
                                (actor.route, actor.configured_model)
                                if tree != staged_tree
                                and actor
                                and actor.configured_model
                                else draft.last_editor
                                if draft is not None
                                else None
                            )
                            draft = DraftCheckpoint(
                                checkpoint_key,
                                task.base_sha,
                                patch,
                                tuple(progress),
                                checkpoint_guidance,
                                editing_route,
                            )
                            budget.remaining()
                            task.save_checkpoint(draft)
                            staged_tree = tree
                if task.verify_base_first and turns == REVIEW_EDIT_REMINDER_TURN:
                    if checkpoint_index is None:
                        review_index = f".git/gha-review-index-{uuid4().hex}"
                        self._git(
                            workspace,
                            budget,
                            "read-tree",
                            staged_tree,
                            index=review_index,
                        )
                        self._git(workspace, budget, "add", "-A", index=review_index)
                        staged_tree = (
                            self._git(
                                workspace, budget, "write-tree", index=review_index
                            )
                            .stdout.strip()
                            .decode("ascii")
                        )
                    if staged_tree == baseline:
                        messages[0]["content"] += (
                            "\n\nIf the checkout still has no draft patch, make a focused "
                            "edit for the requested repair with edit_file, then run the relevant checks."
                        )
            else:
                trajectory.append(
                    "[OBSERVATION] Reached the rollout model-turn limit. Staging the current patch."
                )
            self._git(workspace, budget, "add", "-A")
            try:
                patch = self._draft_patch(workspace, budget, baseline)
                if patch and not clarification:
                    self._check_patch_paths(workspace, task, budget, baseline)
            except (CommandOutputExceeded, _CandidateOutputError) as error:
                budget.reject(error, actor or last)
        return patch, "\n".join(trajectory)[-32768:], actor or last

    @staticmethod
    def _check_clarification(content):
        try:
            reply = json.loads(content)
        except (ValueError, TypeError):
            return
        if isinstance(reply, dict) and isinstance(reply.get("clarification"), str):
            question = reply["clarification"].strip()
            if question:
                raise NeedsClarification(question[:1000])

    @staticmethod
    def _tool_argv(tool):
        try:
            function = tool["function"]
            args = json.loads(function["arguments"])
            argv = args["argv"]
            if (
                function["name"] == "run"
                and isinstance(tool["id"], str)
                and isinstance(argv, list)
                and 0 < len(argv) <= 64
                and all(
                    isinstance(arg, str) and "\0" not in arg and len(arg) <= 8192
                    for arg in argv
                )
                and argv[0]
                and set(args) == {"argv"}
            ):
                return tuple(argv)
        except (ValueError, KeyError, TypeError):
            pass
        raise _CandidateOutputError("Model provided an invalid run tool call")

    def _check_patch_paths(self, workspace, task, budget, baseline, *, index=None):
        result = self._git(
            workspace,
            budget,
            "diff",
            "--cached",
            "--raw",
            "--no-renames",
            "--no-ext-diff",
            "-z",
            baseline,
            "--",
            cap=self.limits.patch_bytes,
            index=index,
        )
        entries = result.stdout.split(b"\0")[:-1]
        if not entries or len(entries) % 2:
            raise _CandidateOutputError("Unexpected patch metadata")
        new_tests = []
        amended_test_path = False
        for header, raw_path in zip(entries[::2], entries[1::2], strict=True):
            fields = header.split()
            if len(fields) != 5 or not header.startswith(b":") or not raw_path:
                raise _CandidateOutputError("Unexpected patch metadata")
            mode, status = fields[1], fields[-1]
            path = raw_path.decode("utf-8", "surrogateescape")
            new_test = (
                status == b"A"
                and mode == b"100644"
                and _test_module(PurePosixPath(path).name)
            )
            amended_test = (
                status == b"M"
                and fields[0] == b":100644"
                and mode == b"100644"
                and task.test_update is not None
                and task.test_update.partition("::")[0] == path
            )
            if _protected(path, task, new_test=new_test, amended_test=amended_test):
                raise _ProtectedPatchError(f"Patch changes protected path: {path}")
            if amended_test:
                baseline_blob, amended_blob = (
                    self._git(
                        workspace,
                        budget,
                        "cat-file",
                        "blob",
                        revision,
                        cap=self.limits.patch_bytes,
                        index=index,
                    ).stdout
                    for revision in (f"{baseline}:{path}", f":{path}")
                )
                if not valid_test_update(task.test_update, baseline_blob, amended_blob):
                    raise _ProtectedPatchError(
                        f"Patch exceeds authorized test update: {path}"
                    )
                amended_test_path = True
            if mode in {b"120000", b"160000"}:
                raise _CandidateOutputError(
                    f"Patch adds or changes a symlink or submodule: {path}"
                )
            if new_test:
                new_tests.append(path)
        return new_tests, amended_test_path

    @staticmethod
    def _require_valid_patch(result, message):
        if result.returncode:
            if result.returncode not in {1, 128} or not result.stderr.startswith(
                b"error:"
            ):
                raise ExecutionError("Worker could not validate the generated patch")
            raise _CandidateOutputError(message)

    def _run_checks(self, workspace, budget, commands):
        checks = []
        for command in commands:
            result = self._command(workspace, budget, command, checked=False)
            checks.append(CheckEvidence(command, result.returncode, _combined(result)))
            if result.returncode:
                break
        return checks

    def _verify(self, task, archive, budget, patch, refinement=None, *, actor=None):
        if not patch:
            budget.reject(
                _CandidateOutputError("Executor returned an empty patch"), actor
            )
        if len(patch) > self.limits.patch_bytes:
            budget.reject(
                CommandOutputExceeded("Patch exceeded the configured size"), actor
            )
        with self.verification_runner.workspace(
            archive, budget.remaining, budget.cancelled
        ) as workspace:
            baseline_tree = self._stage_baseline(workspace, budget)
            try:
                check = self._git(
                    workspace,
                    budget,
                    "apply",
                    "--check",
                    "--index",
                    "--binary",
                    data=patch,
                    checked=False,
                )
                self._require_valid_patch(
                    check, "Invalid patch did not apply to the pinned checkout"
                )
                self._git(workspace, budget, "apply", "--index", "--binary", data=patch)
                new_tests, amended_test = self._check_patch_paths(
                    workspace, task, budget, baseline_tree
                )
            except (_CandidateOutputError, CommandOutputExceeded) as error:
                budget.reject(error, actor)
            if task.require_new_tests and not any(
                has_runnable_test(
                    path,
                    self._git(
                        workspace,
                        budget,
                        "cat-file",
                        "blob",
                        f":{path}",
                        cap=self.limits.patch_bytes,
                    ).stdout,
                )
                for path in new_tests
            ):
                budget.reject(
                    _MissingRegressionTests(
                        "Add new runnable regression tests for the behavior requested in the review."
                    ),
                    actor,
                )
            if new_tests:
                self._git(
                    workspace,
                    budget,
                    "--literal-pathspecs",
                    "rm",
                    "-q",
                    "-f",
                    "--",
                    *new_tests,
                )
            checks = self._run_checks(workspace, budget, task.checks)
        trusted_passed = checks[-1].returncode == 0
        selections = []
        if amended_test:
            selections.append((task.test_update,))
        new_python_tests = tuple(path for path in new_tests if path.endswith(".py"))
        if task.require_new_tests and new_python_tests:
            selections.append(new_python_tests)
        if (new_tests or selections) and trusted_passed:
            with self.verification_runner.workspace(
                archive, budget.remaining, budget.cancelled
            ) as workspace:
                self._stage_baseline(workspace, budget)
                self._git(workspace, budget, "apply", "--index", "--binary", data=patch)
                if new_tests:
                    checks.extend(self._run_checks(workspace, budget, task.checks))
                for selectors in selections:
                    if checks[-1].returncode:
                        break
                    report = f".git/gha-test-update-{uuid4().hex}.xml"
                    focused = focused_pytest(task.checks, selectors, report)
                    if not focused:
                        continue
                    check = self._run_checks(workspace, budget, (focused,))[0]
                    if check.returncode == 0:
                        result = self._command(
                            workspace, budget, ("cat", report), checked=False
                        )
                        if result.returncode or not passed_pytest_report(result.stdout):
                            check = CheckEvidence(
                                focused,
                                1,
                                "The selected test coverage did not pass as executed pytest tests.",
                            )
                    checks.append(check)
        if checks[-1].returncode:
            failed = checks[-1]
            rejection = 0 < failed.returncode < 125
            error_type = _VerificationRejected if rejection else ExecutionError
            error = error_type(f"Required check failed: {failed.command[0]}")
            error.checks = tuple(checks)
            if actor is not None and actor.configured_model and rejection:
                if trusted_passed:
                    budget.reject(error, actor)
                with self.verification_runner.workspace(
                    archive, budget.remaining, budget.cancelled
                ) as baseline:
                    self._stage_baseline(baseline, budget)
                    result = self._command(
                        baseline, budget, failed.command, checked=False
                    )
                if result.returncode == 0:
                    budget.reject(error, actor)
                error_type = (
                    _VerificationRejected
                    if 0 < result.returncode < 125
                    else ExecutionError
                )
                error = error_type(
                    f"Required check also failed on the baseline: {failed.command[0]}"
                )
                error.checks = tuple(checks)
            raise error
        usage = Counter()
        for route in budget.routes:
            usage.update(route.usage)
        return ExecutionResult(
            patch,
            tuple(checks),
            tuple(budget.routes),
            dict(usage),
            hashlib.sha256(patch).hexdigest(),
            refinement,
        )


class Harness2Executor(PlainExecutor):
    def execute(
        self,
        task: TaskSnapshot,
        checkout: Path,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> ExecutionResult:
        from .harness2_adapter import refine, verify_review_repair

        budget = _Budget(self.limits, cancelled, task)
        archive = self._archive(task, checkout, budget)
        try:
            if task.verify_base_first:
                return verify_review_repair(self, task, archive, budget)
            patch, evidence, actor, resumed_base = refine(self, task, archive, budget)
        except _RequestBudgetExceeded as error:
            budget.reject(error, None, latest=True)
        try:
            return self._verify(task, archive, budget, patch, evidence, actor=actor)
        except (
            AttemptRejected,
            _CandidateOutputError,
            _VerificationRejected,
        ) as rejected:
            if isinstance(rejected, LimitExceeded) or resumed_base is None:
                raise
            base_patch, base_actor = resumed_base
            if patch == base_patch:
                raise
            fallback_evidence = replace(
                evidence,
                selected_candidate="base",
                candidates=tuple(
                    replace(
                        candidate, error=f"Verification rejected: {rejected}"[:1000]
                    )
                    if candidate.label == evidence.selected_candidate
                    else candidate
                    for candidate in evidence.candidates
                ),
            )
            try:
                return self._verify(
                    task,
                    archive,
                    budget,
                    base_patch,
                    fallback_evidence,
                    actor=base_actor,
                )
            except (
                AttemptRejected,
                _CandidateOutputError,
                _VerificationRejected,
            ) as base_error:
                if isinstance(base_error, LimitExceeded):
                    raise
                raise rejected from base_error

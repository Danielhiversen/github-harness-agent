"""Execute repository tasks and verify their patches in independent workspaces."""

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from math import isfinite
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any, Protocol

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
    "Implement the repository task. Inspect the pinned checkout with the run tool and edit files "
    "in place. Keep the patch focused and preserve existing checks and their configuration. "
    "When the task requests regression coverage, add new test files using the repository's naming convention. "
    "Finish when the patch is ready. "
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


class Gateway(Protocol):
    def complete(
        self,
        *,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
        timeout: float | None = None,
        exclude: Sequence[tuple[str, str]] = (),
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


@dataclass(frozen=True)
class Limits:
    wall_time: float = 900
    model_requests: int = 40
    concurrent_probes: int = 2
    patch_bytes: int = 262144
    commands: int = 64
    archive_bytes: int = 67108864
    command_output_bytes: int = 65536

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
        excluded_models: tuple[tuple[str, str], ...] = (),
    ):
        self.limits = limits
        self.cancelled = cancelled
        self.deadline = time.monotonic() + limits.wall_time
        self.routes: list[RouteEvidence] = []
        self.excluded_models = excluded_models
        self._calls = Counter()
        self._lock = Lock()

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

    def complete(self, gateway, role, messages, tools=()):
        self.claim("model request", self.limits.model_requests)
        options = {"exclude": self.excluded_models} if self.excluded_models else {}
        completion = gateway.complete(
            role=role,
            messages=messages,
            tools=tools,
            timeout=self.remaining(),
            **options,
        )
        self.remaining()
        with self._lock:
            self.routes.append(RouteEvidence.from_completion(completion))
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


def _protected(path: str, task: TaskSnapshot, *, new_test: bool) -> bool:
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
            not new_test
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
    def __init__(self, runner, gateway: Gateway, limits: Limits = Limits()):
        self.runner = runner
        self.gateway = gateway
        self.limits = limits

    def execute(
        self,
        task: TaskSnapshot,
        checkout: Path,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> ExecutionResult:
        budget = _Budget(self.limits, cancelled, task.excluded_models)
        archive = self._archive(task, checkout, budget)
        patch, _, actor = self._rollout(task, archive, budget, SYSTEM_PROMPT, "coding")
        return self._verify(task, archive, budget, patch, actor=actor)

    def verify_patch(
        self,
        task: TaskSnapshot,
        checkout: Path,
        patch: bytes,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> ExecutionResult:
        budget = _Budget(self.limits, cancelled)
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

    def _git(self, workspace, budget, *argv, **kwargs):
        return self._command(
            workspace,
            budget,
            ("git", "-c", "core.hooksPath=/dev/null", *argv),
            **kwargs,
        )

    def _stage_baseline(self, workspace, budget):
        self._git(workspace, budget, "init", "-q", "--template=")
        self._git(workspace, budget, "add", "-A", "-f")
        return self._git(workspace, budget, "write-tree").stdout.strip().decode("ascii")

    def _rollout(self, task, archive, budget, prompt, role, *, clarification=True):
        with self.runner.workspace(
            archive, budget.remaining, budget.cancelled
        ) as workspace:
            baseline = self._stage_baseline(workspace, budget)
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": task.instructions},
            ]
            trajectory = []
            actor = None
            last = None
            while True:
                try:
                    completion = budget.complete(
                        self.gateway, role, messages, (RUN_TOOL,)
                    )
                except _RequestBudgetExceeded as error:
                    budget.reject(error, last, latest=True)
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
                if provider_context := getattr(completion, "provider_context", None):
                    assistant["provider_context"] = provider_context
                messages.append(assistant)
                for tool in completion.tool_calls:
                    try:
                        budget.claim("command", self.limits.commands)
                        argv = self._tool_argv(tool)
                        response = self._command(workspace, budget, argv, checked=False)
                    except CommandOutputExceeded:
                        output = (
                            "Command exceeded the output limit. Select a narrower range or "
                            "redirect output to a workspace file and inspect a slice."
                        )
                    except (_RequestBudgetExceeded, _CandidateOutputError) as error:
                        budget.reject(error, last)
                    else:
                        output = f"exit {response.returncode}\n{_combined(response)}"
                    actor = last
                    trajectory.append(
                        f"[ACTION] {json.dumps(argv)}\n[OBSERVATION] {output}"
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool["id"],
                            "content": output,
                        }
                    )
            self._git(workspace, budget, "add", "-A")
            try:
                patch = self._git(
                    workspace,
                    budget,
                    "diff",
                    "--cached",
                    "--binary",
                    "--no-renames",
                    "--no-ext-diff",
                    "--no-textconv",
                    baseline,
                    cap=self.limits.patch_bytes,
                ).stdout
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

    def _check_patch_paths(self, workspace, task, budget, baseline):
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
        )
        entries = result.stdout.split(b"\0")[:-1]
        if not entries or len(entries) % 2:
            raise _CandidateOutputError("Unexpected patch metadata")
        new_tests = []
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
            if _protected(path, task, new_test=new_test):
                raise _CandidateOutputError(f"Patch changes protected path: {path}")
            if mode in {b"120000", b"160000"}:
                raise _CandidateOutputError(
                    f"Patch adds or changes a symlink or submodule: {path}"
                )
            if new_test:
                new_tests.append(path)
        return new_tests

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
        with self.runner.workspace(
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
                new_tests = self._check_patch_paths(
                    workspace, task, budget, baseline_tree
                )
            except (_CandidateOutputError, CommandOutputExceeded) as error:
                budget.reject(error, actor)
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
        if new_tests and trusted_passed:
            with self.runner.workspace(
                archive, budget.remaining, budget.cancelled
            ) as workspace:
                self._stage_baseline(workspace, budget)
                self._git(workspace, budget, "apply", "--index", "--binary", data=patch)
                checks.extend(self._run_checks(workspace, budget, task.checks))
        if checks[-1].returncode:
            failed = checks[-1]
            error = ExecutionError(f"Required check failed: {failed.command[0]}")
            error.checks = tuple(checks)
            if (
                actor is not None
                and actor.configured_model
                and 0 < failed.returncode < 125
            ):
                if trusted_passed:
                    budget.reject(error, actor)
                with self.runner.workspace(
                    archive, budget.remaining, budget.cancelled
                ) as baseline:
                    self._stage_baseline(baseline, budget)
                    result = self._command(
                        baseline, budget, failed.command, checked=False
                    )
                if result.returncode == 0:
                    budget.reject(error, actor)
                error = ExecutionError(
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
        from .harness2_adapter import refine

        budget = _Budget(self.limits, cancelled, task.excluded_models)
        archive = self._archive(task, checkout, budget)
        try:
            patch, evidence, actor = refine(self, task, archive, budget)
        except _RequestBudgetExceeded as error:
            budget.reject(error, None, latest=True)
        return self._verify(task, archive, budget, patch, evidence, actor=actor)

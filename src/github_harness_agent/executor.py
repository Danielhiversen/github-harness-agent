"""Execute repository tasks and verify their patches in independent workspaces."""

import hashlib
import json
import logging
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from fnmatch import fnmatchcase
from math import isfinite
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any, Protocol
from uuid import uuid4

from .activity import observed_call
from .check_artifacts import STAT_FILES_COMMAND, check_label
from .check_diagnostics import check_failure_summary, required_review_checks
from .checkpoint import DraftCheckpoint, SolverRecovery
from .file_edit import EDIT_COMMAND, EDIT_TOOL, InvalidEdit, edit_payload
from .repair_context import repair_source_context
from .review_recovery import (
    RECOVERY_EDIT_TOOL,
    TEXT_RECOVERY_SYSTEM,
    ReviewRecovery,
    operator_acceptance_context,
)
from .providers import ProviderExhausted
from .test_updates import (
    focused_pytest,
    has_runnable_test,
    passed_pytest_report,
    regression_assertion_report,
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
REVIEW_EXTRA_TURNS_PER_THREAD = 3
REVIEW_MAX_EXTRA_TURNS = 8


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
        max_output_tokens: int | None = None,
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


REVIEW_ROUND_EXHAUSTED = (
    "Review solver retained verified progress; outstanding checks still fail"
)


class ReviewRoundExhausted(AttemptRejected):
    """A completed proposal round retained a verified partial draft."""


def is_review_round_record(record):
    return (
        record.get("kind") == "review_round_exhausted"
        or record.get("reason") == REVIEW_ROUND_EXHAUSTED
    )


class _RequestBudgetExceeded(LimitExceeded):
    pass


class _CandidateOutputError(ExecutionError):
    pass


@dataclass(frozen=True)
class _CheckArtifact:
    index_entry: bytes
    fingerprint: bytes


class _ProtectedPatchError(_CandidateOutputError):
    repair_feedback = (
        "The candidate changed protected content outside the allowed scope. "
        "Restore protected regions to the pinned baseline and keep any authorized "
        "test update limited to the requested region. Put additional tests in a "
        "separate new test module."
    )


class _TestOnlyPatchError(_ProtectedPatchError):
    repair_feedback = "Restore production code and existing tests. Add only new test files."


class _RegressionScopeError(_ProtectedPatchError):
    repair_feedback = "Change only the reviewed production file and add new regression tests."


class _VerificationRejected(ExecutionError):
    """A trusted check ran and rejected the candidate."""


class _MissingRegressionTests(_VerificationRejected):
    """A review requires new runnable test coverage."""

    missing_regression_tests = True


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
    inline_review_threads: int = 0
    test_update: str | None = None
    require_new_tests: bool = False
    pull_checks: tuple[tuple[str, ...], ...] = ()
    review_recovery: ReviewRecovery | None = None
    repair_failure: tuple["CheckEvidence", ...] = field(default=(), repr=False)
    solver_recovery: SolverRecovery | None = None
    save_solver_recovery: Callable[[SolverRecovery | None, bool], None] | None = None
    retry_after_provider: str | None = None
    regression_base: str | None = None
    regression_archive: bytes | None = field(default=None, repr=False, compare=False)
    regression_edit_path: str | None = None
    regression_head: str | None = None


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
    regression_counterexample: bytes | None = field(default=None, repr=False)


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
        self.regression_counterexample: bytes | None = None
        self.regression_counterexample_archive: bytes | None = None
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
            return self._calls[kind]

    def model_requests_remaining(self):
        with self._lock:
            return self.limits.model_requests - self._calls["model request"]

    def commands_remaining(self):
        with self._lock:
            return self.limits.commands - self._calls["command"]

    def route_usage(self):
        with self._lock:
            routes = tuple(self.routes)
        usage = Counter()
        for route in routes:
            usage.update(route.usage)
        return routes, dict(usage)

    def complete(
        self,
        gateway,
        role,
        messages,
        tools=(),
        *,
        exclude=(),
        after_provider=None,
        retry_oversize=True,
        review_windows=False,
        max_output_tokens=None,
        timeout_cap=None,
    ):
        excluded_models = tuple(dict.fromkeys((*self.task.excluded_models, *exclude)))
        options = {"exclude": excluded_models} if excluded_models else {}
        if after_provider:
            options["after_provider"] = after_provider
        elif self.task.provider_offset and not self.task.review_recovery:
            options["provider_offset"] = self.task.provider_offset
        if max_output_tokens is not None:
            options["max_output_tokens"] = max_output_tokens

        def send(payload):
            request_number = self.claim("model request", self.limits.model_requests)
            request_options = options.copy()
            if request_number == 1 and not after_provider and not (
                self.task.review_recovery and role == "planning"
            ):
                if self.task.review_recovery and self.task.provider_offset:
                    request_options["provider_offset"] = self.task.provider_offset
                if self.task.retry_after_provider:
                    request_options["after_provider"] = self.task.retry_after_provider
            remaining = self.remaining()
            return observed_call(
                self.task,
                gateway,
                role=role,
                messages=payload,
                tools=tools,
                timeout=remaining
                if timeout_cap is None
                else min(remaining, timeout_cap),
                **request_options,
            )

        with self._lock:
            message_limit = self._message_limit
        compact = compact_messages(
            messages, max_bytes=message_limit, role=role, review_windows=review_windows
        )
        try:
            completion = send(compact)
        except ProviderExhausted as error:
            if not retry_oversize or "upstream HTTP 413" not in error.failures.values():
                raise
            smaller = compact_messages(
                messages,
                max_bytes=RETRY_MESSAGE_BYTES,
                role=role,
                review_windows=review_windows,
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

    def exhaust_review_round(self, editor):
        self.remaining()
        with self._lock:
            trace = tuple(self.routes)
        raise ReviewRoundExhausted(REVIEW_ROUND_EXHAUSTED, editor, trace)


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
        task = self._pin_regression(task, checkout, budget)
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
        *,
        regression_counterexample: tuple[str, bytes] | None = None,
    ) -> ExecutionResult:
        budget = _Budget(self.limits, cancelled, task)
        archive = self._archive(task, checkout, budget)
        task = self._pin_regression(task, checkout, budget)
        if regression_counterexample is not None:
            sha, production_patch = regression_counterexample
            if (
                not task.regression_edit_path or not task.regression_base
                or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", sha)
                or not 0 < len(production_patch) <= self.limits.patch_bytes
            ):
                raise ExecutionError("The saved regression counterexample is invalid")
            budget.regression_counterexample = production_patch
            budget.regression_counterexample_archive = self._checkout_archive(
                checkout, budget, sha, "Verified partial checkpoint base", no_replace=True,
            )
        return self._verify(task, archive, budget, patch)

    def _archive(self, task, checkout, budget):
        budget.remaining()
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", task.base_sha):
            raise ExecutionError("A full immutable checkout SHA is required")
        if not task.checks or any(not command for command in task.checks):
            raise ExecutionError("At least one trusted required check is required")
        if len(task.repository_guidance) > 3000:
            raise ExecutionError("Repository guidance exceeds the configured size")
        return self._checkout_archive(
            checkout, budget, task.base_sha, "Pinned checkout",
            no_replace=bool(task.regression_base),
        )

    def _checkout_archive(self, checkout, budget, sha, label, *, no_replace):
        try:
            result = run_process(
                [
                    "git",
                    *(("--no-replace-objects",) if no_replace else ()),
                    "-c",
                    "core.hooksPath=/dev/null",
                    "archive",
                    "--format=tar",
                    sha,
                ],
                budget.remaining(),
                budget.cancelled,
                cwd=checkout,
                cap=self.limits.archive_bytes,
            )
        except CommandOutputExceeded as error:
            raise LimitExceeded(
                f"{label} exceeds the archive limit ({self.limits.archive_bytes} bytes)"
            ) from error
        if result.returncode:
            raise ExecutionError(f"Could not archive the {label.lower()}")
        return result.stdout

    def _pin_regression(self, task, checkout, budget):
        sha = task.regression_base
        if sha is None:
            return task
        mixed = task.regression_edit_path is not None
        if mixed and _protected(task.regression_edit_path, task, new_test=False):
            raise ExecutionError("Regression verification requires an unprotected reviewed production file")
        head = (task.regression_head or task.base_sha) if mixed else task.base_sha
        full_sha = r"[0-9a-f]{40}|[0-9a-f]{64}"
        if (
            not re.fullmatch(full_sha, sha) or not re.fullmatch(full_sha, head)
            or (not mixed and sha == head)
        ):
            raise ExecutionError("Regression verification requires a full SHA for an authorized revision or ancestor")
        for args, expected in (
            (("cat-file", "-t", sha), b"commit\n"),
            (("merge-base", "--is-ancestor", sha, head), b""),
        ):
            result = run_process(
                ("git", "--no-replace-objects", *args),
                budget.remaining(), budget.cancelled, cwd=checkout,
                cap=self.limits.command_output_bytes,
            )
            if result.returncode or result.stdout != expected:
                raise ExecutionError("Regression verification requires a locally available ancestor commit")
        return replace(task, regression_archive=self._checkout_archive(
            checkout, budget, sha, "Regression ancestor", no_replace=True,
        ))

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
        # Give Git tools in the disposable checkout a HEAD for status and diffs.
        self._git(
            workspace,
            budget,
            "-c",
            "user.name=Harness Agent",
            "-c",
            "user.email=workspace@localhost",
            "commit",
            "--allow-empty",
            "--no-gpg-sign",
            "-qm",
            "Pinned workspace baseline",
        )
        return (
            self._git(workspace, budget, "rev-parse", "HEAD^{tree}")
            .stdout.strip()
            .decode("ascii")
        )

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
    def _command_progress(argv, returncode, check):
        executable = PurePosixPath(argv[0]).name
        label = check or (
            executable if executable in {"python", "python3", "git"} else "command"
        )
        status = (
            "exceeded output limit" if returncode is None else f"exited {returncode}"
        )
        return f"{label} {status}", check is not None

    def _checkout_paths(self, workspace, budget, index, *selection):
        result = self._git(
            workspace,
            budget,
            "ls-files",
            *selection,
            "-z",
            index=index,
        )
        return set(result.stdout.split(b"\0")[:-1])

    def _check_snapshot(self, workspace, budget, index):
        try:
            return (
                self._checkout_paths(
                    workspace, budget, index, "--others", "--exclude-standard"
                ),
                self._checkout_paths(workspace, budget, index, "--deleted"),
            )
        except CommandOutputExceeded as error:
            raise _CandidateOutputError(
                "Checkout path inventory exceeded the limit for tracking check output."
            ) from error

    def _artifact_state(self, workspace, budget, paths):
        if not paths:
            return {}
        paths = sorted(paths)
        result = self._command(
            workspace,
            budget,
            STAT_FILES_COMMAND,
            data=b"\0".join(paths) + b"\0",
            cap=self.limits.patch_bytes,
        )
        return {
            path: state
            for path, state in zip(paths, result.stdout.splitlines(), strict=True)
        }

    def _stage_rollout(
        self, workspace, budget, artifacts, *, index=None, tracked=False
    ):
        mode = "-u" if tracked else "-A"
        if not artifacts:
            self._git(workspace, budget, "add", mode, index=index)
            return
        paths = sorted(artifacts)
        self._git(
            workspace,
            budget,
            "add",
            mode,
            "--pathspec-from-file=-",
            "--pathspec-file-nul",
            data=b".\0"
            + b"".join(b":(top,exclude,literal)" + path + b"\0" for path in paths),
            index=index,
        )
        # Restore pre-check entries even if the check itself changed the index.
        self._git(
            workspace,
            budget,
            "update-index",
            "-z",
            "--index-info",
            data=b"".join(
                artifacts[path].index_entry + b"\t" + path + b"\0" for path in paths
            ),
            index=index,
        )

    def _check_changed_entries(self, workspace, budget, before_tree, index):
        result = self._git(
            workspace,
            budget,
            "diff",
            "--cached",
            "--raw",
            "--no-abbrev",
            "--no-renames",
            "--no-ext-diff",
            "-z",
            before_tree,
            index=index,
        )
        pairs = result.stdout.split(b"\0")[:-1]
        entries = {}
        for raw, path in zip(pairs[::2], pairs[1::2], strict=True):
            old_mode, _, old_oid, _, _ = raw.split()
            entries[path] = old_mode.removeprefix(b":") + b" " + old_oid
        return entries

    def _review_tree(self, workspace, budget, baseline, artifacts):
        index = f".git/gha-review-index-{uuid4().hex}"
        self._git(workspace, budget, "read-tree", baseline, index=index)
        self._stage_rollout(workspace, budget, artifacts, index=index)
        tree = self._git(workspace, budget, "write-tree", index=index)
        return index, tree.stdout.strip().decode("ascii")

    def _review_source_progress(
        self, workspace, task, budget, baseline, initial_tree, artifacts
    ):
        try:
            index, tree = self._review_tree(workspace, budget, baseline, artifacts)
            if tree in {baseline, initial_tree}:
                return False
            self._check_patch_paths(workspace, task, budget, baseline, index=index)
            if not self._draft_patch(workspace, budget, baseline, index=index):
                return False
            diff = self._git(
                workspace,
                budget,
                "diff",
                "--cached",
                "--numstat",
                "--no-renames",
                "--no-ext-diff",
                "-z",
                initial_tree,
                index=index,
                cap=self.limits.patch_bytes,
            )
        except (_CandidateOutputError, CommandOutputExceeded):
            return False
        for item in diff.stdout.split(b"\0")[:-1]:
            added, removed, filename = item.split(b"\t", 2)
            path = PurePosixPath(filename.decode("utf-8", "surrogateescape"))
            if (
                added.isdigit()
                and removed.isdigit()
                and (added, removed) != (b"0", b"0")
                and not _test_module(path.name)
                and not {"tests", "test", "__tests__"}.intersection(path.parts)
            ):
                return True
        return False

    def _has_runnable_new_tests(self, workspace, budget, paths, *, index=None):
        return any(
            has_runnable_test(
                path,
                self._git(
                    workspace,
                    budget,
                    "cat-file",
                    "blob",
                    f":{path}",
                    index=index,
                    cap=self.limits.patch_bytes,
                ).stdout,
            )
            for path in paths
        )

    def _review_missing_tests(self, workspace, task, budget, baseline, index):
        try:
            paths, _ = self._check_patch_paths(
                workspace, task, budget, baseline, index=index
            )
            return not self._has_runnable_new_tests(
                workspace, budget, paths, index=index
            )
        except (_CandidateOutputError, CommandOutputExceeded):
            return False

    def _read_recovery_source(self, workspace, budget, task, tree, path):
        if _protected(path, task, new_test=False):
            return None
        lookup = self._git(
            workspace,
            budget,
            "--literal-pathspecs",
            "ls-tree",
            tree,
            "--",
            path,
        )
        if lookup.stdout.split(b" ", 1)[0] not in {b"100644", b"100755"}:
            return None
        try:
            raw = self._git(
                workspace,
                budget,
                "cat-file",
                "blob",
                f"{tree}:{path}",
                cap=512 * 1024,
            ).stdout
            return raw.decode() if b"\0" not in raw else None
        except (CommandOutputExceeded, UnicodeDecodeError):
            return None

    def _legacy_repair_contains_seed(self, workspace, budget, baseline, repair, seed):
        if not seed or repair == seed:
            return True
        if not repair or len(repair) > self.limits.patch_bytes:
            return False
        index = f".git/gha-repair-seed-{uuid4().hex}"
        self._git(workspace, budget, "read-tree", baseline, index=index)
        try:
            applied = self._git(
                workspace,
                budget,
                "apply",
                "--cached",
                "--binary",
                data=repair,
                index=index,
                checked=False,
            )
            return (
                applied.returncode == 0
                and self._git(
                    workspace,
                    budget,
                    "apply",
                    "--cached",
                    "--binary",
                    "--check",
                    "--reverse",
                    data=seed,
                    index=index,
                    checked=False,
                ).returncode
                == 0
            )
        except CommandOutputExceeded:
            return False

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
        scoped_sources=None,
        scoped_action_counts=None,
        scoped_verify=None,
    ):
        with self.runner.workspace(
            archive, budget.remaining, budget.cancelled
        ) as workspace:
            baseline = self._stage_baseline(workspace, budget)
            own_draft = task.checkpoints.get(checkpoint_key)
            seed_digest = None
            upgraded = False
            if checkpoint_key == "harness2/repair":
                seed_patch = seed_draft.patch if seed_draft is not None else b""
                seed_digest = hashlib.sha256(seed_patch).hexdigest()
                if own_draft is not None and own_draft.seed_patch_digest != seed_digest:
                    if (
                        own_draft.seed_patch_digest is None
                        and self._legacy_repair_contains_seed(
                            workspace,
                            budget,
                            baseline,
                            own_draft.patch,
                            seed_patch,
                        )
                    ):
                        own_draft = replace(own_draft, seed_patch_digest=seed_digest)
                        upgraded = True
                    else:
                        own_draft = None
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
                if upgraded and task.save_checkpoint is not None:
                    budget.remaining()
                    task.save_checkpoint(draft)
                progress.extend(draft.progress)
                instructions += (
                    "\n\nResuming a saved draft already applied to this checkout. "
                    "Inspect the existing diff and preserve valid edits while completing the task. "
                    "Earlier command outcomes (rerun required checks before finishing): "
                    + "; ".join(progress)
                )
            initial_tree = staged_tree
            if task.repair_failure:
                source = repair_source_context(
                    task.repair_failure,
                    lambda path: self._read_recovery_source(
                        workspace, budget, task, initial_tree, path
                    ),
                )
                if source:
                    instructions += "\n\n" + source
            acceptance_source = ""
            if task.pull_checks and scoped_sources is None:
                try:
                    acceptance_source = operator_acceptance_context(
                        task.pull_checks,
                        lambda argv, cap: self._command(
                            workspace, budget, argv, cap=cap, checked=False
                        ),
                    )
                except (CommandOutputExceeded, OSError):
                    pass
            recovery = None
            if task.review_recovery is not None and checkpoint_key in {
                "plain/coding",
                "harness2/base",
            }:
                add_test = False
                if (
                    task.require_new_tests
                    and "runnable regression test"
                    in task.review_recovery.previous_failure.casefold()
                ):
                    if initial_tree == baseline:
                        add_test = True
                    else:
                        index, _ = self._review_tree(workspace, budget, baseline, {})
                        new_tests, _ = self._check_patch_paths(
                            workspace,
                            task,
                            budget,
                            baseline,
                            index=index,
                        )
                        add_test = not new_tests
                recovery = task.review_recovery.request(
                    lambda path: self._read_recovery_source(
                        workspace, budget, task, initial_tree, path
                    ),
                    (*task.checks, *task.pull_checks),
                    require_test=add_test,
                    acceptance_source=acceptance_source,
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
            if task.pull_checks:
                instructions += (
                    "\n\nThe operator configured these extra acceptance checks for this PR. "
                    "Run available checks in the coding workspace and use their results to "
                    "complete all outstanding review requests. The independent verifier "
                    "will rerun them before publication.\n"
                    + json.dumps(task.pull_checks)
                )
                if acceptance_source:
                    instructions += "\n\n" + acceptance_source
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": instructions},
            ]
            tools = (RUN_TOOL, EDIT_TOOL)
            if task.context is not None:
                tools += (CONTEXT_TOOL,)
            trajectory = []
            artifact_index = None
            check_tree_index = None
            artifacts = {}
            actor = None
            last = None
            turns = 0
            limit = budget.limits.rollout_requests
            automatic_review = task.verify_base_first and limit is None
            if automatic_review:
                limit = min(
                    REVIEW_ROLLOUT_TURNS, max(1, budget.model_requests_remaining())
                )
            # Keep room for the existing repair candidate when a fresh base draft
            # finishes with missing tests and the shared request budget is small.
            focused_request_minimum = (
                3
                if checkpoint_key == "harness2/base" and initial_tree == baseline
                else 1
            )
            retried_empty_finish = False
            focused_test_requested = False
            focus_next_turn = False
            text_batch_applied = False
            scoped_edits = {}
            scoped_stop = None
            scoped_production = tuple(
                path
                for path in scoped_sources or ()
                if not _protected(path, task, new_test=False)
            )
            scoped_pacing = "inspect" if scoped_production else "ordinary"
            inspection_turns = 0
            scoped_retried = False

            def captured_exact_edits():
                updates = self._capture_scoped_sources(
                    workspace, budget, initial_tree, scoped_sources, text_only=True,
                )
                if any(path not in scoped_edits or scoped_edits[path][0] != source
                       for path, source in updates.items()):
                    raise _CandidateOutputError("Apply scoped source updates through edit_file.")
                return updates

            def valid_recovery_test(path, content):
                return (
                    _test_module(PurePosixPath(path).name)
                    and not _protected(path, task, new_test=True)
                    and has_runnable_test(path, content)
                )

            def log_recovery(mode, outcome, count=0):
                logging.getLogger(__name__).info(
                    "Stalled review recovery mode=%s outcome=%s edit_count=%d",
                    mode,
                    outcome,
                    count,
                )

            def has_model_turn():
                return (
                    limit is None or turns < limit
                ) and budget.model_requests_remaining() > 0

            while limit is None or turns < limit:
                if (text_batch_applied or scoped_sources is not None) and not has_model_turn():
                    break
                recovery_turn = recovery if turns == 0 else None
                focused_test_turn = focus_next_turn
                focus_next_turn = False
                request_messages = messages
                request_tools = tools
                paced_edit = scoped_readback = False
                if recovery_turn:
                    request_tools = (RECOVERY_EDIT_TOOL,)
                    request_messages = [
                        messages[0],
                        {"role": "user", "content": recovery_turn.prompt},
                    ]
                elif focused_test_turn:
                    request_tools = (EDIT_TOOL,)
                    request_messages = [
                        {
                            **messages[0],
                            "content": messages[0]["content"]
                            + "\n\nThis turn is for the review's missing regression tests. "
                            "Use the reviewed behavior and existing repository conventions to "
                            "create a runnable new test file with edit_file. Supply its path "
                            "and complete new_text. Omit old_text to create the file.",
                        },
                        *messages[1:],
                    ]
                elif (
                    scoped_pacing in {"edit", "readback"}
                    and budget.model_requests_remaining() > 1
                ):
                    paced_edit = scoped_pacing == "edit"
                    scoped_readback = not paced_edit
                    scoped_pacing = "ordinary"
                    request_tools = (EDIT_TOOL,) if paced_edit else (RUN_TOOL,)
                    guidance = (
                        "Use edit_file now to make one exact replacement in the reviewed production "
                        f"files {json.dumps(scoped_production)}. Copy old_text from source already inspected. "
                        "Choose the smallest fix for the tracked failure and preserve passing behavior."
                        if paced_edit
                        else "The exact match was stale. Use run to read the affected source lines again. "
                        "Keep this turn to inspection so the next edit can use exact current text."
                    )
                    request_messages = [
                        {
                            **messages[0],
                            "content": messages[0]["content"] + "\n\n" + guidance,
                        },
                        *messages[1:],
                    ]
                native_error = None
                try:
                    last_editor = (
                        own_draft.last_editor if own_draft and not turns else None
                    )
                    completion = budget.complete(
                        self.gateway,
                        role,
                        request_messages,
                        request_tools,
                        after_provider=last_editor[0] if last_editor else None,
                        review_windows=bool(recovery_turn),
                    )
                except _RequestBudgetExceeded as error:
                    if scoped_sources is None:
                        budget.reject(error, last, latest=True)
                    scoped_stop = error
                    break
                except ProviderExhausted as error:
                    if (paced_edit or scoped_readback) and has_model_turn():
                        budget.remaining()
                        continue
                    if not recovery_turn:
                        if text_batch_applied:
                            break
                        if not scoped_edits:
                            raise
                        budget.remaining()
                        scoped_stop = error
                        break
                    native_error = error
                    completion = None
                turns += 1
                calls = ()
                if completion is not None:
                    last = RouteEvidence.from_completion(completion)
                    calls = completion.tool_calls
                text_recovery = False
                recovery_calls = (
                    recovery_turn.normalize_calls(
                        calls, self.limits.patch_bytes, valid_recovery_test
                    )
                    if recovery_turn
                    else ()
                )
                native_accepted = bool(recovery_calls)
                if recovery_turn:
                    log_recovery(
                        "native",
                        "exhausted"
                        if native_error
                        else "accepted"
                        if native_accepted
                        else "invalid",
                        len(calls) if native_accepted else 0,
                    )
                if recovery_turn and not native_accepted:
                    if not has_model_turn():
                        if native_error is not None:
                            raise native_error
                        continue
                    turns += 1
                    try:
                        completion = budget.complete(
                            self.gateway,
                            role,
                            [
                                {"role": "system", "content": TEXT_RECOVERY_SYSTEM},
                                {"role": "user", "content": recovery_turn.text_prompt},
                            ],
                            retry_oversize=False,
                            review_windows=True,
                        )
                    except _RequestBudgetExceeded as error:
                        budget.reject(error, last, latest=True)
                    except ProviderExhausted:
                        log_recovery("text", "exhausted")
                        if not has_model_turn():
                            raise
                        continue
                    last = RouteEvidence.from_completion(completion)
                    if completion.tool_calls:
                        log_recovery("text", "invalid")
                        continue
                    calls = recovery_turn.text_calls(
                        completion.content, self.limits.patch_bytes
                    )
                    recovery_calls = recovery_turn.normalize_calls(
                        calls, self.limits.patch_bytes, valid_recovery_test
                    )
                    if not recovery_calls:
                        log_recovery("text", "invalid")
                        continue
                    log_recovery("text", "accepted", len(calls))
                    text_recovery = True
                if not calls:
                    if clarification and completion.content:
                        self._check_clarification(completion.content)
                    if (paced_edit or scoped_readback) and has_model_turn():
                        continue
                    can_continue = (
                        automatic_review
                        and turns < limit
                        and budget.model_requests_remaining() > 0
                    )
                    if can_continue and focused_test_turn:
                        continue
                    if can_continue and (
                        not retried_empty_finish
                        or (task.require_new_tests and not focused_test_requested)
                    ):
                        index, tree = self._review_tree(
                            workspace, budget, baseline, artifacts
                        )
                        if tree == baseline and not retried_empty_finish:
                            retried_empty_finish = True
                            messages[0]["content"] += (
                                "\n\nThe checkout still has no patch. Make a concrete edit "
                                "for the review request using edit_file before finishing."
                            )
                            continue
                        if (
                            tree != baseline
                            and task.require_new_tests
                            and not focused_test_requested
                            and budget.model_requests_remaining()
                            >= focused_request_minimum
                            and self._review_missing_tests(
                                workspace, task, budget, baseline, index
                            )
                        ):
                            focused_test_requested = focus_next_turn = True
                            continue
                    break
                assistant = {
                    "role": "assistant",
                    "content": None if text_recovery else completion.content,
                    "tool_calls": list(calls),
                }
                # These calls were generated locally from text. Keep the actual
                # route in budget and checkpoint evidence without native affinity.
                if last.configured_model and not text_recovery:
                    assistant["gateway_route"] = [last.route, last.configured_model]
                provider_context = getattr(completion, "provider_context", None)
                if provider_context and not text_recovery:
                    assistant["provider_context"] = provider_context
                messages.append(assistant)
                ran_command = ran_check = False
                inspected = refreshed = stale_edit = False
                successful_text_edits = 0
                for tool in recovery_calls if recovery_turn else calls:
                    name = tool["function"]["name"]
                    if focused_test_turn and name != "edit_file":
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool["id"],
                                "content": "Use edit_file to create a new test module.",
                            }
                        )
                        continue
                    if (
                        task.context is not None
                        and not (paced_edit or scoped_readback)
                        and name == "read_task_context"
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
                    is_edit = name == "edit_file"
                    touched = False
                    checked_command = None
                    before_check = None
                    before_tree = None
                    try:
                        budget.claim("command", self.limits.commands)
                        if paced_edit and (len(calls) != 1 or not is_edit):
                            raise InvalidEdit(
                                "Make exactly one edit_file call on this turn."
                            )
                        if scoped_readback and name != "run":
                            raise InvalidEdit(
                                "Use run to inspect the source on this turn."
                            )
                        if is_edit:
                            action = "edit_file"
                            data = edit_payload(tool, self.limits.patch_bytes)
                            if scoped_sources is not None or focused_test_turn:
                                edit = json.loads(data)
                            if (
                                scoped_sources is not None
                                and edit["path"] not in scoped_sources
                            ):
                                raise InvalidEdit(
                                    "That file is outside the scoped review."
                                )
                            if paced_edit and (
                                edit["path"] not in scoped_production
                                or edit.get("old_text") is None
                            ):
                                raise InvalidEdit(
                                    "Replace exact text in an existing reviewed production file."
                                )
                            if focused_test_turn:
                                path = edit["path"]
                                if (
                                    edit.get("old_text") is not None
                                    or not _test_module(PurePosixPath(path).name)
                                    or _protected(path, task, new_test=True)
                                ):
                                    raise InvalidEdit(
                                        "Create a new allowed test module and omit old_text."
                                    )
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
                            if scoped_verify is not None and argv == ("verify_review",):
                                try:
                                    self._stage_rollout(workspace, budget, artifacts)
                                    output = scoped_verify(captured_exact_edits())
                                except _CandidateOutputError:
                                    output = "Could not verify this draft. Use edit_file for the scoped source updates."
                                messages.append({"role": "tool", "tool_call_id": tool["id"], "content": output})
                                trajectory.append(f"[ACTION] verify_review\n[OBSERVATION] {output}")
                                continue
                            checked_command = check_label(argv, task.checks)
                            if checked_command:
                                if artifact_index is None:
                                    artifact_index = (
                                        f".git/gha-artifact-index-{uuid4().hex}"
                                    )
                                    self._git(
                                        workspace,
                                        budget,
                                        "read-tree",
                                        baseline,
                                        index=artifact_index,
                                    )
                                    check_tree_index = (
                                        f".git/gha-check-tree-{uuid4().hex}"
                                    )
                                    self._git(
                                        workspace,
                                        budget,
                                        "read-tree",
                                        baseline,
                                        index=check_tree_index,
                                    )
                                self._stage_rollout(
                                    workspace, budget, artifacts, index=check_tree_index
                                )
                                before_tree = (
                                    self._git(
                                        workspace,
                                        budget,
                                        "write-tree",
                                        index=check_tree_index,
                                    )
                                    .stdout.strip()
                                    .decode("ascii")
                                )
                                before_check = self._check_snapshot(
                                    workspace, budget, artifact_index
                                )
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
                    if text_recovery and touched and returncode == 0:
                        successful_text_edits += 1
                    if scoped_sources is not None:
                        edit_succeeded = is_edit and touched and returncode == 0
                        if scoped_action_counts is not None:
                            if is_edit:
                                outcome = (
                                    "successful_edits"
                                    if edit_succeeded
                                    else "rejected_edits"
                                )
                                scoped_action_counts[outcome] += 1
                            elif name == "run" and touched:
                                scoped_action_counts["shell_actions"] += 1
                        if edit_succeeded:
                            path = edit["path"]
                            production_edit = path in scoped_production
                            if production_edit:
                                scoped_pacing = "ordinary"
                            earlier = scoped_edits.pop(path, None)
                            expected = earlier[0] if earlier else scoped_sources[path]
                            old = edit.get("old_text")
                            if old and last.configured_model:
                                first = expected.find(old)
                                if first >= 0 and expected.find(old, first + 1) < 0:
                                    scoped_edits[path] = (
                                        expected[:first]
                                        + edit["new_text"]
                                        + expected[first + len(old) :],
                                        turns,
                                        last,
                                    )
                            elif (
                                last.configured_model
                                and old is None
                                and not expected
                                and not production_edit
                            ):
                                scoped_edits[path] = (edit["new_text"], turns, last)
                        elif (
                            is_edit
                            and paced_edit
                            and output.startswith(
                                "exit 1\nold_text must occur exactly once."
                            )
                        ):
                            stale_edit = True
                        elif name == "run" and touched:
                            inspected = True
                            refreshed |= returncode == 0 and checked_command is None
                    if touched:
                        actor = last
                        ran_command = True
                        try:
                            if before_check is not None:
                                untracked_before, missing_before = before_check
                                untracked_after, missing_after = self._check_snapshot(
                                    workspace, budget, artifact_index
                                )
                                created = untracked_after - untracked_before
                                recreated = missing_before - missing_after
                                self._stage_rollout(
                                    workspace,
                                    budget,
                                    artifacts,
                                    index=check_tree_index,
                                    tracked=True,
                                )
                                entries = self._check_changed_entries(
                                    workspace, budget, before_tree, check_tree_index
                                )
                                absent = b"0 " + b"0" * len(baseline)
                                for path in created | recreated:
                                    entries.setdefault(path, absent)
                                entries.update(
                                    (path, artifact.index_entry)
                                    for path, artifact in artifacts.items()
                                )
                                states = self._artifact_state(
                                    workspace, budget, entries
                                )
                                artifacts = {
                                    path: _CheckArtifact(entry, states[path])
                                    for path, entry in entries.items()
                                }
                            elif artifacts:
                                current = self._artifact_state(
                                    workspace, budget, artifacts
                                )
                                artifacts = {
                                    path: artifact
                                    for path, artifact in artifacts.items()
                                    if current[path] == artifact.fingerprint
                                }
                        except (_CandidateOutputError, CommandOutputExceeded) as error:
                            budget.reject(error, last)
                    if touched and task.save_checkpoint is not None:
                        outcome, is_check = (
                            ("edit_file updated checkout", False)
                            if is_edit
                            else self._command_progress(
                                argv, returncode, checked_command
                            )
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
                if text_recovery:
                    text_batch_applied = successful_text_edits == len(calls)
                if scoped_pacing == "inspect" and inspected:
                    inspection_turns += 1
                    if inspection_turns >= 2:
                        scoped_pacing = "edit"
                if (
                    stale_edit
                    and not scoped_retried
                    and budget.model_requests_remaining() >= 3
                ):
                    scoped_retried = True
                    scoped_pacing = "readback"
                elif scoped_readback and refreshed:
                    scoped_pacing = "edit"
                if ran_command and checkpoint_index is not None:
                    self._stage_rollout(
                        workspace, budget, artifacts, index=checkpoint_index
                    )
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
                                seed_digest,
                            )
                            budget.remaining()
                            task.save_checkpoint(draft)
                            staged_tree = tree
                if task.verify_base_first and turns == REVIEW_EDIT_REMINDER_TURN:
                    if checkpoint_index is None:
                        _, staged_tree = self._review_tree(
                            workspace, budget, baseline, artifacts
                        )
                    if staged_tree == baseline:
                        messages[0]["content"] += (
                            "\n\nIf the checkout still has no draft patch, make a focused "
                            "edit for the requested repair with edit_file, then run the relevant checks."
                        )
                if (
                    automatic_review
                    and turns == REVIEW_ROLLOUT_TURNS
                    and task.inline_review_threads > 1
                    and (remaining := budget.model_requests_remaining()) > 0
                    and self._review_source_progress(
                        workspace, task, budget, baseline, initial_tree, artifacts
                    )
                ):
                    limit += min(
                        remaining,
                        REVIEW_MAX_EXTRA_TURNS,
                        REVIEW_EXTRA_TURNS_PER_THREAD
                        * (task.inline_review_threads - 1),
                    )
                    messages[0]["content"] += (
                        "\n\nThe draft now contains a source edit. Use the remaining "
                        "turns to address the outstanding inline review requests."
                    )
                if focused_test_turn and not budget.model_requests_remaining():
                    break
            else:
                trajectory.append(
                    "[OBSERVATION] Reached the rollout model-turn limit. Staging the current patch."
                )
            if scoped_sources is not None and scoped_stop is None and not has_model_turn():
                scoped_stop = _RequestBudgetExceeded(
                    "Interactive review model request limit reached"
                )
            self._stage_rollout(workspace, budget, artifacts)
            try:
                patch = self._draft_patch(workspace, budget, baseline)
                if patch and not clarification:
                    self._check_patch_paths(workspace, task, budget, baseline)
                if scoped_sources is not None:
                    updated = (
                        captured_exact_edits() if scoped_verify is not None else
                        self._capture_scoped_sources(
                            workspace, budget, initial_tree, scoped_sources,
                            text_only=scoped_stop is not None,
                        )
                    )
                    if scoped_stop is not None or (scoped_verify is not None and updated):
                        production = {
                            path for path in updated
                            if not _protected(path, task, new_test=False)
                        }
                        if not production or any(
                            path not in scoped_edits or scoped_edits[path][0] != content
                            for path, content in updated.items()
                        ):
                            if isinstance(scoped_stop, _RequestBudgetExceeded):
                                budget.reject(scoped_stop, last, latest=True)
                            if scoped_stop is not None:
                                raise scoped_stop
                            raise _CandidateOutputError("Interactive review needs an attributed production edit.")
                        _, _, actor = max(
                            (scoped_edits[path] for path in production),
                            key=lambda entry: entry[1],
                        )
                        budget.remaining()
                        if scoped_stop is not None:
                            logging.getLogger(__name__).info(
                                "Harvesting attributed edits in %d scoped production file(s) after %s",
                                len(production),
                                "provider exhaustion"
                                if isinstance(scoped_stop, ProviderExhausted)
                                else "the model request limit",
                            )
                    scoped_sources.update(updated)
            except (CommandOutputExceeded, _CandidateOutputError) as error:
                budget.reject(error, actor or last)
        return patch, "\n".join(trajectory)[-32768:], actor or last

    def _capture_scoped_sources(self, workspace, budget, baseline, sources, *, text_only):
        changed = self._git(
            workspace,
            budget,
            "diff",
            "--cached",
            "--raw",
            "--no-renames",
            "-z",
            baseline,
            "--",
            cap=self.limits.patch_bytes,
        ).stdout.split(b"\0")[:-1]
        if len(changed) % 2:
            raise _CandidateOutputError("Unexpected scoped patch metadata")
        paths = [path.decode("utf-8", "surrogateescape") for path in changed[1::2]]
        if any(path not in sources for path in paths):
            raise _ProtectedPatchError("Interactive edits exceeded the scoped review paths")
        if text_only:
            for header in changed[::2]:
                fields = header.split()
                if len(fields) != 5 or (
                    fields[0] != b":" + fields[1]
                    and (fields[0], fields[1], fields[-1]) != (b":000000", b"100644", b"A")
                ):
                    raise _CandidateOutputError("Interactive edits changed file metadata")
        updated = {}
        for path in paths:
            blob = self._git(
                workspace, budget, "show", ":" + path, checked=False, cap=128 * 1024
            )
            if blob.returncode or b"\x00" in blob.stdout:
                raise _CandidateOutputError("Interactive review requires readable source files")
            updated[path] = blob.stdout.decode("utf-8")
        return updated

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

    def _check_patch_paths(
        self, workspace, task, budget, baseline, *, index=None, new_tests_only=False
    ):
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
            if new_tests_only and not new_test:
                raise _TestOnlyPatchError(f"Test-only review patch changes an out-of-scope path: {path}")
            if task.regression_edit_path and not new_test and not (
                path == task.regression_edit_path and status == b"M"
                and mode in {b"100644", b"100755"} and fields[0] == b":" + mode
            ):
                raise _RegressionScopeError(f"Pinned regression patch changes an out-of-scope path: {path}")
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

    def _run_checks(self, workspace, budget, commands, *, collect=False):
        checks = []
        for command in commands:
            result = self._command(workspace, budget, command, checked=False)
            checks.append(CheckEvidence(command, result.returncode, _combined(result)))
            if result.returncode and (not collect or not 0 < result.returncode < 125):
                break
        return checks

    def _regression_sources(self, workspace, budget, paths):
        return tuple(
            self._command(workspace, budget, ("cat", "--", path), cap=self.limits.patch_bytes).stdout
            for path in paths
        )

    def _verify_test_regression(
        self, task, archive, budget, patch, new_tests, green_report, green_sources, green_output,
        *, counterexample=None,
    ):
        command = (
            ("verify-regression", task.regression_base)
            if counterexample is None else ("verify-regression-counterexample",)
        )
        report = f".git/gha-regression-{uuid4().hex}.xml"
        selectors = tuple(path for path in new_tests if path.endswith(".py"))
        pytest = focused_pytest(task.checks, selectors, report, verbose=True)
        if not (pytest and green_report and task.regression_archive):
            return CheckEvidence(command, 1, "Run the new tests with the configured pytest launcher before checking the ancestor.")
        budget.claim("command", budget.limits.commands)
        with self.verification_runner.workspace(
            task.regression_archive if counterexample is None else archive,
            budget.remaining, budget.cancelled,
        ) as workspace:
            tree = self._stage_baseline(workspace, budget)
            if counterexample is not None:
                self._git(workspace, budget, "apply", "--index", "--binary", data=counterexample)
                partial_tests, amended_test = self._check_patch_paths(workspace, task, budget, tree)
                if partial_tests or amended_test:
                    raise ExecutionError("The verified partial checkpoint must contain production changes only.")
            applied = self._git(
                workspace, budget, "apply", "--index", "--binary", data=patch, checked=False,
            )
            if applied.returncode:
                return CheckEvidence(
                    command, 1, "Use new test paths that can be added unchanged to "
                    + ("the verified ancestor." if counterexample is None else "the verified partial checkpoint."),
                )
            if self._regression_sources(workspace, budget, new_tests) != green_sources:
                return CheckEvidence(command, 1, "The generated test sources must be byte-identical on both revisions.")
            result = self._command(workspace, budget, pytest, checked=False)
            if result.returncode == 1:
                red_report = self._command(workspace, budget, ("cat", report), checked=False)
                if red_report.returncode == 0 and regression_assertion_report(green_report, red_report.stdout):
                    return CheckEvidence(
                        command, 0, "A named test passed on the candidate and failed by assertion on "
                        + ("the verified ancestor." if counterexample is None else "the verified partial checkpoint."),
                    )
        if counterexample is not None:
            detail = (
                "The new tests also pass on the verified partial checkpoint. "
                "Assert the requested behavior that remained missing on that revision."
                if result.returncode == 0 else
                "The verified partial checkpoint did not produce a matched test assertion failure. "
                "Use the same scenario and available public APIs on both revisions."
            )
            return CheckEvidence(command, 1, detail)
        detail = (
            "The new tests also pass on the ancestor. Assert the reviewed behavior that differs on the pinned revision."
            if result.returncode == 0 else
            "Setup, import, and collection failures cannot prove the reviewed behavior. Use APIs available on both revisions."
        )
        if result.returncode == 0 and task.review_recovery and task.review_recovery.test_only:
            try:
                from .review_mock_replay import passing_callback_hint

                if hint := passing_callback_hint(
                    self, task, archive, budget, pytest, patch, new_tests,
                    green_output, result.stdout.decode("utf-8", "replace"),
                ):
                    detail += "\n" + hint
            except Cancelled:
                raise
            except Exception:
                logging.getLogger(__name__).debug("Optional regression callback comparison was unavailable")
        return CheckEvidence(command, 1, "A named assertion must fail on the ancestor. " + detail)

    def _verify(self, task, archive, budget, patch, refinement=None, *, actor=None):
        if not patch:
            budget.reject(
                _CandidateOutputError("Executor returned an empty patch"), actor
            )
        if len(patch) > self.limits.patch_bytes:
            budget.reject(
                CommandOutputExceeded("Patch exceeded the configured size"), actor
            )
        commands = (
            required_review_checks(task) if task.verify_base_first else task.checks
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
                    workspace, task, budget, baseline_tree,
                    new_tests_only=bool(
                        (task.regression_base and not task.regression_edit_path)
                        or (task.review_recovery and task.review_recovery.test_only)
                    ),
                )
            except (_CandidateOutputError, CommandOutputExceeded) as error:
                budget.reject(error, actor)
            missing_tests = task.require_new_tests and not self._has_runnable_new_tests(
                workspace, budget, new_tests
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
            if missing_tests:
                reason = "Add new runnable regression tests for the behavior requested in the review."
                evidence = self._run_checks(
                    workspace,
                    budget,
                    commands if task.verify_base_first else task.pull_checks,
                    collect=task.verify_base_first,
                )
                diagnostics = check_failure_summary(task, evidence)
                if task.verify_base_first and any(
                    not 0 <= check.returncode < 125 for check in evidence
                ):
                    error = ExecutionError(diagnostics)
                    error.checks = tuple(evidence)
                    raise error
                if diagnostics:
                    reason += " " + diagnostics
                error = _MissingRegressionTests(reason)
                error.checks = tuple(evidence)
                budget.reject(error, actor)
            checks = self._run_checks(
                workspace, budget, commands, collect=task.verify_base_first
            )
        trusted_passed = not any(check.returncode for check in checks)
        selections = []
        if amended_test:
            selections.append((task.test_update,))
        new_python_tests = tuple(path for path in new_tests if path.endswith(".py"))
        if task.require_new_tests and new_python_tests:
            selections.append(new_python_tests)
        green_report = b""
        green_sources = ()
        green_output = ""
        historical_patch = patch
        historical_tests = tuple(new_tests) if task.regression_edit_path else new_python_tests
        if (new_tests or selections) and trusted_passed:
            with self.verification_runner.workspace(
                archive, budget.remaining, budget.cancelled
            ) as workspace:
                self._stage_baseline(workspace, budget)
                self._git(workspace, budget, "apply", "--index", "--binary", data=patch)
                if new_tests:
                    checks.extend(
                        self._run_checks(
                            workspace,
                            budget,
                            commands,
                            collect=task.verify_base_first,
                        )
                    )
                for selectors in selections:
                    if any(check.returncode for check in checks):
                        break
                    report = f".git/gha-test-update-{uuid4().hex}.xml"
                    focused = focused_pytest(
                        task.checks, selectors, report, verbose=selectors == new_python_tests
                    )
                    if not focused:
                        continue
                    regression_green = task.regression_base and selectors == new_python_tests
                    focused_workspace = (
                        self.verification_runner.workspace(archive, budget.remaining, budget.cancelled)
                        if regression_green else nullcontext(workspace)
                    )
                    with focused_workspace as selected:
                        if regression_green:
                            tree = self._stage_baseline(selected, budget)
                            self._git(
                                selected, budget, "apply", "--index", "--binary", data=patch,
                            )
                            if task.regression_edit_path:
                                historical_patch = self._git(
                                    selected, budget, "--literal-pathspecs", "diff", "--cached",
                                    "--binary", "--no-ext-diff", "--full-index", tree, "--",
                                    *historical_tests, cap=self.limits.patch_bytes,
                                ).stdout
                            green_sources = self._regression_sources(selected, budget, historical_tests)
                        check = self._run_checks(selected, budget, (focused,))[0]
                        if check.returncode == 0:
                            result = self._command(selected, budget, ("cat", report), checked=False)
                            if result.returncode or not passed_pytest_report(result.stdout):
                                check = CheckEvidence(
                                    focused,
                                    1,
                                    "The selected test coverage did not pass as executed pytest tests.",
                                )
                            elif regression_green:
                                green_report = result.stdout
                                green_output = check.output
                    checks.append(check)
        if task.regression_base and not any(check.returncode for check in checks):
            regression = self._verify_test_regression(
                task, archive, budget, historical_patch, historical_tests, green_report, green_sources, green_output
            )
            checks.append(regression)
            if not regression.returncode and budget.regression_counterexample is not None:
                regression = self._verify_test_regression(
                    task, budget.regression_counterexample_archive or archive, budget,
                    historical_patch, historical_tests, green_report, green_sources, green_output,
                    counterexample=budget.regression_counterexample,
                )
                checks.append(regression)
            if regression.returncode:
                rejection = _VerificationRejected(regression.output)
                rejection.checks = tuple(checks)
                budget.reject(rejection, actor)
        failures = [check for check in checks if check.returncode]
        if failures:
            rejection = all(0 < check.returncode < 125 for check in failures)
            error_type = _VerificationRejected if rejection else ExecutionError
            error = error_type(check_failure_summary(task, checks))
            error.checks = tuple(checks)
            if actor is not None and actor.configured_model and rejection:
                if trusted_passed or any(
                    failed.command in task.pull_checks for failed in failures
                ):
                    budget.reject(error, actor)
                with self.verification_runner.workspace(
                    archive, budget.remaining, budget.cancelled
                ) as baseline:
                    self._stage_baseline(baseline, budget)
                    baseline_checks = self._run_checks(
                        baseline,
                        budget,
                        tuple(failed.command for failed in failures),
                        collect=task.verify_base_first,
                    )
                if any(check.returncode == 0 for check in baseline_checks):
                    budget.reject(error, actor)
                error_type = (
                    _VerificationRejected
                    if all(0 < check.returncode < 125 for check in baseline_checks)
                    else ExecutionError
                )
                error = error_type(
                    check_failure_summary(task, checks, baseline_failed=True)
                )
                error.checks = tuple(checks)
            raise error
        routes, usage = budget.route_usage()
        return ExecutionResult(
            patch,
            tuple(checks),
            routes,
            usage,
            hashlib.sha256(patch).hexdigest(),
            refinement,
            regression_counterexample=budget.regression_counterexample,
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
        task = self._pin_regression(task, checkout, budget)
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

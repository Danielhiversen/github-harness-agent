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

from .worker import Cancelled, DockerRunner, ExecutionError, LimitExceeded, run_process

__all__ = [
    "Cancelled",
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
    "in place. Keep the patch focused and preserve existing checks. Finish when the patch is ready. "
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
    ): ...


class NeedsClarification(ExecutionError):
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
    def __init__(self, limits: Limits, cancelled: Callable[[], bool]):
        self.limits = limits
        self.cancelled = cancelled
        self.deadline = time.monotonic() + limits.wall_time
        self.routes: list[RouteEvidence] = []
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
                raise LimitExceeded(f"Task {kind} budget exceeded")
            self._calls[kind] += 1

    def complete(self, gateway, role, messages, tools=()):
        self.claim("model request", self.limits.model_requests)
        completion = gateway.complete(
            role=role,
            messages=messages,
            tools=tools,
            timeout=self.remaining(),
        )
        self.remaining()
        with self._lock:
            self.routes.append(
                RouteEvidence(
                    completion.role,
                    completion.route,
                    completion.model,
                    completion.usage,
                )
            )
        return completion


def _combined(result):
    return (result.stdout + result.stderr).decode("utf-8", "replace")


def _protected(path: str, task: TaskSnapshot) -> bool:
    """Preserve test/setup files and direct script arguments used by required checks."""
    names = PurePosixPath(path).parts
    test_configs = {
        "conftest.py",
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
        bool(set(names) & {".git", ".github", "tests", "test", "__tests__"})
        or names[-1] in test_configs
        or names[-1].startswith("test_")
        or names[-1].endswith(
            ("_test.go", ".test.js", ".test.ts", ".spec.js", ".spec.ts")
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
        budget = _Budget(self.limits, cancelled)
        archive = self._archive(task, checkout, budget)
        patch, _ = self._rollout(task, archive, budget, SYSTEM_PROMPT, "coding")
        return self._verify(task, archive, budget, patch)

    def _archive(self, task, checkout, budget):
        budget.remaining()
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", task.base_sha):
            raise ExecutionError("A full immutable checkout SHA is required")
        if not task.checks or any(not command for command in task.checks):
            raise ExecutionError("At least one trusted required check is required")
        if len(task.repository_guidance) > 3000:
            raise ExecutionError("Repository guidance exceeds the configured size")
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

    def _rollout(self, task, archive, budget, prompt, role, *, clarification=True):
        with self.runner.workspace(
            archive, budget.remaining, budget.cancelled
        ) as workspace:
            self._git(workspace, budget, "init", "-q", "--template=")
            self._git(workspace, budget, "add", "-A", "-f")
            baseline = (
                self._git(workspace, budget, "write-tree")
                .stdout.strip()
                .decode("ascii")
            )
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": task.instructions},
            ]
            trajectory = []
            while True:
                completion = budget.complete(self.gateway, role, messages, (RUN_TOOL,))
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
                    budget.claim("command", self.limits.commands)
                    argv = self._tool_argv(tool)
                    response = self._command(workspace, budget, argv, checked=False)
                    output = _combined(response)
                    trajectory.append(
                        f"[ACTION] {json.dumps(argv)}\n[OBSERVATION] "
                        f"exit {response.returncode}\n{output}"
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool["id"],
                            "content": f"exit {response.returncode}\n{output}",
                        }
                    )
            self._git(workspace, budget, "add", "-A")
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
                self._check_patch_paths(workspace, task, budget, patch)
        return patch, "\n".join(trajectory)[-32768:]

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
        raise ExecutionError("Model provided an invalid run tool call")

    def _check_patch_paths(self, workspace, task, budget, patch):
        summary = self._git(
            workspace,
            budget,
            "apply",
            "--numstat",
            "-z",
            cap=self.limits.patch_bytes,
            data=patch,
        ).stdout
        for entry in summary.rstrip(b"\0").split(b"\0"):
            parts = entry.split(b"\t", 2)
            if len(parts) != 3 or not parts[-1]:
                raise ExecutionError("Unexpected patch metadata")
            path = parts[-1].decode("utf-8", "surrogateescape")
            if _protected(path, task):
                raise ExecutionError(f"Patch changes protected path: {path}")

    def _verify(self, task, archive, budget, patch, refinement=None):
        if not patch:
            raise ExecutionError("Executor returned an empty patch")
        if len(patch) > self.limits.patch_bytes:
            raise LimitExceeded("Patch exceeded the configured size")
        with self.runner.workspace(
            archive, budget.remaining, budget.cancelled
        ) as workspace:
            self._git(workspace, budget, "init", "-q", "--template=")
            self._git(workspace, budget, "add", "-A", "-f")
            self._check_patch_paths(workspace, task, budget, patch)
            self._git(workspace, budget, "apply", "--index", "--binary", data=patch)
            checks = []
            for command in task.checks:
                result = self._command(workspace, budget, command, checked=False)
                evidence = CheckEvidence(command, result.returncode, _combined(result))
                checks.append(evidence)
                if result.returncode:
                    error = ExecutionError(f"Required check failed: {command[0]}")
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

        budget = _Budget(self.limits, cancelled)
        archive = self._archive(task, checkout, budget)
        patch, evidence = refine(self, task, archive, budget)
        return self._verify(task, archive, budget, patch, evidence)

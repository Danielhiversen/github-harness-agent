"""Use bounded disposable replays to investigate a persistent stock pytest failure."""

import ast
import inspect
import json
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import PurePosixPath
from types import SimpleNamespace

from . import review_guard_observer, review_mock_probe
from .review_definitions import _symbols
from .review_prefix_replay import _interpreter
from .review_solver_context import _STOCK_NODE
from .task_context import excerpt
from .worker import Cancelled, ExecutionError, LimitExceeded

_FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPES = (*_FUNCTION, ast.ClassDef, ast.Lambda)
_CONDITIONS = (
    ast.Name,
    ast.Attribute,
    ast.Subscript,
    ast.Compare,
    ast.BoolOp,
    ast.UnaryOp,
    ast.Call,
)
_MODES = (
    (True, "suppressed the first true result per function activation"),
    (False, "suppressed every true result"),
)
_MAX_GUARDS = 2
_REPLAY_SECONDS = 40
_SINGLE_REPLAY_SECONDS = 15
_PLANNING_RESERVE = 30

_MISSING = {"KeyError", "IndexError", "AttributeError"}


@dataclass(frozen=True)
class _Failure:
    line: int
    kind: str
    digest: str
    comparison: str
    traces: dict | None


def _decode_readout(result):
    if (
        result.returncode not in (0, 1)
        or len(result.stdout) > review_guard_observer.REPORT_BYTES
    ):
        return None
    try:
        payload = json.loads(result.stdout)
    except (ValueError, RecursionError):
        return None
    if (
        type(payload) is not dict
        or payload.keys() != {"status", "observation", "traces"}
        or type(payload["status"]) is not int
        or payload["status"] != result.returncode
    ):
        return None
    observation = payload["observation"]
    if observation is None:
        return result.returncode
    if (
        result.returncode != 1
        or type(observation) is not dict
        or observation.keys() != {"line", "kind", "digest", "comparison"}
    ):
        return None
    line, kind, digest, comparison = (
        observation[key] for key in ("line", "kind", "digest", "comparison")
    )
    if (
        type(line) is not int
        or not 1 <= line <= review_guard_observer.SOURCE_LIMIT
        or type(kind) is not str
        or kind not in {*_MISSING, "AssertionError"}
        or type(digest) is not str
        or not re.fullmatch(r"[a-f0-9]{64}", digest)
        or type(comparison) is not str
        or (
            comparison
            and (
                kind != "AssertionError"
                or not review_guard_observer.SCALAR_COMPARISON.fullmatch(comparison)
            )
        )
    ):
        return None
    traces = review_mock_probe.decode_traces(json.dumps(payload["traces"]))
    return _Failure(line, kind, digest, comparison, traces)


@dataclass(frozen=True)
class _Guard:
    path: str
    method: str
    function: ast.FunctionDef | ast.AsyncFunctionDef
    statement: ast.If
    expression: str


def _parse(source):
    try:
        return ast.parse(source)
    except (ValueError, SyntaxError, RecursionError):
        return None


def _scope_nodes(function):
    pending = list(reversed(function.body))
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, _SCOPES):
            pending.extend(reversed(list(ast.iter_child_nodes(node))))


def _candidates(original, current, fixture):
    targets = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", fixture))
    ranked = []
    for path, source in sorted(current.items()):
        before = original.get(path)
        if not before or before == source or not path.endswith(".py"):
            continue
        if not (module := _parse(source)):
            continue
        functions = [
            (qualified, function)
            for qualified, function in _symbols(module.body)
            if isinstance(function, _FUNCTION) and function.name in targets
        ]
        if not functions:
            continue
        changes = {
            line
            for tag, _, _, start, stop in SequenceMatcher(
                None, before.splitlines(), source.splitlines()
            ).get_opcodes()
            if tag in {"insert", "replace"}
            for line in range(start + 1, stop + 1)
        }
        for qualified, function in functions:
            for node in _scope_nodes(function):
                if not isinstance(node, ast.If) or not isinstance(
                    node.test, _CONDITIONS
                ):
                    continue
                hits = sum(
                    row in changes
                    for row in range(node.lineno, node.test.end_lineno + 1)
                )
                if not hits:
                    continue
                guard = _Guard(
                    path,
                    qualified,
                    function,
                    node,
                    ast.get_source_segment(source, node.test),
                )
                # Prefer guards actually named by attribute calls in the tracked fixture.
                confidence = int(
                    bool(re.search(rf"\.\s*{re.escape(function.name)}\s*\(", fixture))
                )
                ranked.append((confidence, hits, guard))
    ranked.sort(key=lambda item: item[:2], reverse=True)
    return [guard for _, _, guard in ranked[:_MAX_GUARDS]]


def _instrument(source, guard, first_only):
    expression = guard.statement.test
    if guard.expression is None:
        return None
    lines = source.encode().splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    start = offsets[expression.lineno - 1] + expression.col_offset
    end = offsets[expression.end_lineno - 1] + expression.end_col_offset
    if first_only:
        # Ensure the new binding is distinct from every name in the source,
        # including closure variables and names referenced by nested functions.
        counter = "_stock_guard_probe_count"
        while counter in source:
            counter += "_"
        condition = f"(({counter} := {counter} + 1) != 1)"
    else:
        condition = "False"
    text = source.encode()
    text = text[:start] + f"({guard.expression}) and {condition}".encode() + text[end:]
    if first_only:
        body = guard.function.body
        first_statement = body[int(ast.get_docstring(guard.function) is not None)]
        row = first_statement.lineno - 1
        # Insert before decorations as well as before ordinary statements.
        if decorators := getattr(first_statement, "decorator_list", ()):
            row = min(row, *(item.lineno - 1 for item in decorators))
        indentation = lines[row][: len(lines[row]) - len(lines[row].lstrip())]
        initializer = indentation + f"{counter} = 0\n".encode()
        offset = offsets[row]
        text = text[:offset] + initializer + text[offset:]
    result = text.decode()
    return result if _parse(result) is not None else None


def counterfactual_guard_diagnostic(
    original, current, stock_fixture, replay, *, timeout=_REPLAY_SECONDS
):
    """Use replay(sources, seconds) to check one real pass (0) or call failure (1)."""
    deadline = time.monotonic() + timeout
    candidates = _candidates(original, current, stock_fixture)

    def run(sources):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        return replay(sources, min(remaining, _SINGLE_REPLAY_SECONDS))

    if not candidates:
        return ""
    control = run(dict(current))
    if control != 1 and not isinstance(control, _Failure):
        return ""
    best = None
    for guard in candidates:
        for first_only, description in _MODES:
            if not (
                instrumented := _instrument(current[guard.path], guard, first_only)
            ):
                continue
            result = run({**current, guard.path: instrumented})
            if result is None:
                break
            location = (
                f"{guard.path}::{guard.method}, draft line {guard.statement.lineno}: "
                f"`if {excerpt(guard.expression, 400)}`."
            )
            if result == 0:
                return (
                    "Auxiliary counterfactual (quoted data, diagnostic only). "
                    "The exact stock node failed on the unmodified draft and passed "
                    f"when a disposable replay {description} of one guard. {location} "
                    "Use this observation to investigate the ordinary behavior. "
                    "The temporary instrumentation is excluded from the edit and verification."
                )
            if rank := _partial_advance(control, result):
                if best is None or rank > best[0]:
                    best = rank, description, location, result
    if best is None:
        return ""
    _, description, location, reached = best
    progress = (
        "later tracked assert" if reached.line > control.line else "same tracked assert"
    )
    comparison = f" (`{reached.comparison}`)" if reached.comparison else ""
    sample = _mock_sample(reached.traces)
    return (
        "Auxiliary counterfactual (quoted data, partial diagnostic only). "
        f"The unchanged stock node still failed. The draft raised {control.kind} at "
        f"tracked assert line {control.line}. A disposable replay that {description} "
        f"of one guard reached an evaluated AssertionError at the {progress}, "
        f"line {reached.line}{comparison}. {location} "
        + sample
        + "The temporary instrumentation is excluded from the edit and verification."
    )


def _mock_sample(traces):
    if not traces:
        return ""
    lines, remaining = [], 380
    for mock, calls in traces.items():
        for index, call in enumerate(calls[:4], start=1):
            row = f"{mock} call {index}: {json.dumps(call, separators=(',', ':'))}"
            if len(row) + 1 <= remaining:
                lines.append(row)
                remaining -= len(row) + 1
    return "Numeric mock sample (quoted): " + "; ".join(lines) + ". " if lines else ""


def _partial_advance(before, after):
    if (
        not isinstance(before, _Failure)
        or not isinstance(after, _Failure)
        or before.kind not in _MISSING
        or after.kind != "AssertionError"
        or before.digest != after.digest
        or after.line < before.line
        or (after.line == before.line and not after.comparison)
    ):
        return None
    return after.line - before.line, bool(after.comparison)


def stock_guard_diagnostic(
    executor,
    task,
    archive,
    budget,
    command,
    node,
    original,
    current,
    stock_fixture,
    make_patch,
):
    """Run the observer only in fresh offline verification workspaces."""
    node = node.removeprefix("/work/").removeprefix("./")
    base_node, bracket, parameter = node.partition("[")
    test_path = base_node.partition("::")[0]
    runner = executor.verification_runner
    if (
        command not in task.checks
        or command in task.pull_checks
        or len(node) > 512
        or not _STOCK_NODE.fullmatch(base_node)
        or (bracket and (not parameter.endswith("]") or not parameter.isprintable()))
        or (task.test_update or "").partition("::")[0] == test_path
        or getattr(runner, "network", "none") != "none"
    ):
        return ""
    interpreter = _interpreter(command)
    verification_reserve = len(set((*task.checks, *task.pull_checks)))
    if (
        interpreter is None
        or budget.commands_remaining() < verification_reserve + 2
        or budget.remaining() < _PLANNING_RESERVE + 10
    ):
        return ""
    if PurePosixPath(interpreter[0]).name == "uv":
        interpreter = ("env", "UV_OFFLINE=1", *interpreter)

    mock_source = inspect.getsource(review_mock_probe)
    observer = (
        "_mock_probe = {'__name__': '_guard_mock_probe'}\n"
        f"exec({mock_source!r}, _mock_probe)\n"
        "_mock_locals = _mock_probe['_locals_trace']\n"
        + inspect.getsource(review_guard_observer)
    )

    def replay(sources, seconds):
        if budget.commands_remaining() <= verification_reserve:
            return None
        deadline = time.monotonic() + seconds

        def remaining():
            left = min(
                deadline - time.monotonic(), budget.remaining() - _PLANNING_RESERVE
            )
            if left <= 0:
                raise LimitExceeded("Auxiliary guard replay timed out")
            return left

        bounded = SimpleNamespace(remaining=remaining, cancelled=budget.cancelled)
        try:
            with runner.workspace(archive, remaining, budget.cancelled) as ws:
                tree = executor._stage_baseline(ws, bounded)
                patch = make_patch(original, sources)
                if not patch:
                    return None
                executor._git(ws, bounded, "apply", "--index", data=patch)
                executor._check_patch_paths(ws, task, bounded, tree)
                unchanged = executor._git(
                    ws,
                    bounded,
                    "diff",
                    "--cached",
                    "--quiet",
                    tree,
                    "--",
                    test_path,
                    checked=False,
                )
                if unchanged.returncode:
                    return None
                timeout = remaining()
                budget.claim("command", budget.limits.commands)
                result = ws.command(
                    (*interpreter, "-I", "-B", "-c", observer, node),
                    timeout,
                    budget.cancelled,
                    cap=review_guard_observer.REPORT_BYTES,
                )
        except Cancelled:
            raise
        except (ExecutionError, OSError):
            budget.remaining()
            return None
        return _decode_readout(result)

    return counterfactual_guard_diagnostic(original, current, stock_fixture, replay)

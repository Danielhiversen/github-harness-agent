"""Compare a stalled stock pytest case with its passing pinned baseline."""

import ast
import hashlib
import inspect
import json
import re
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import SimpleNamespace

from . import review_counter_trace, review_mock_probe
from .review_prefix_replay import _interpreter
from .review_solver_context import _STOCK_NODE, _tracked_stock_test
from .task_context import excerpt
from .worker import Cancelled, ExecutionError, LimitExceeded

_PLANNING_RESERVE = 30
# Includes startup and staging in a fresh isolated verification workspace.
_REPLAY_SECONDS = 20
_DETAIL_BYTES = 900
_COUNTER_HEADING = "Auxiliary mapping counter changes (quoted advisory data)."
_OBSERVATION_BOUNDARY = re.compile(
    r"(?m)^(?=Auxiliary (?:counterfactual|mapping counter changes|mock calls)\b)"
)


@dataclass(frozen=True)
class CounterCallerLocation:
    path: str
    name: str
    line: int
    source_digest: str


@dataclass(frozen=True)
class CounterCallerSite:
    locations: tuple[CounterCallerLocation, ...]
    excerpts: str


class _ReplayDiagnostic(str):
    def __new__(cls, text, site):
        result = super().__new__(cls, text)
        result.counter_site = site
        return result


def counter_caller_site(report):
    return report.counter_site if isinstance(report, _ReplayDiagnostic) else None


def compact_observations(detail, limit):
    """Budget for each replay report, including both sides of a counter comparison."""
    if len(detail.encode()) <= limit:
        return detail
    reports = [
        part.strip() for part in _OBSERVATION_BOUNDARY.split(detail) if part.strip()
    ]
    if not reports:
        return ""
    allowance = max(0, (limit - len(reports) + 1) // len(reports))
    result = []
    for report in reports:
        if report.startswith(_COUNTER_HEADING) and len(report.encode()) > allowance:
            rows = [
                line
                for line in report.splitlines()
                if line.startswith(("Pinned: ", "Draft: "))
            ]
            if rows and allowance > len(_COUNTER_HEADING) + len(rows):
                row_limit = (allowance - len(_COUNTER_HEADING) - len(rows)) // len(rows)
                result.append(
                    _COUNTER_HEADING
                    + "\n"
                    + "\n".join(excerpt(row, row_limit) for row in rows)
                )
                continue
        result.append(excerpt(report, allowance))
    return "\n".join(result)


def _readout(
    executor,
    task,
    archive,
    budget,
    interpreter,
    node,
    patch,
    expected,
    paths,
    failure_line,
):
    deadline = time.monotonic() + _REPLAY_SECONDS

    def remaining():
        seconds = min(
            deadline - time.monotonic(), budget.remaining() - _PLANNING_RESERVE
        )
        if seconds <= 0:
            raise LimitExceeded("Auxiliary mock replay timed out")
        return seconds

    bounded = SimpleNamespace(remaining=remaining, cancelled=budget.cancelled)
    try:
        with executor.verification_runner.workspace(
            archive, remaining, budget.cancelled
        ) as ws:
            tree = executor._stage_baseline(ws, bounded)
            if patch:
                executor._git(ws, bounded, "apply", "--index", data=patch)
                executor._check_patch_paths(ws, task, bounded, tree)
                test_path = node.split("::", 1)[0]
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
            program = inspect.getsource(review_mock_probe)
            parameters, cap = (node,), review_mock_probe.REPORT_BYTES
            if paths:
                counter_source = inspect.getsource(review_counter_trace)
                program = (
                    "_counter_module = {'__name__': '_stock_counter_probe'}\n"
                    f"exec({counter_source!r}, _counter_module)\n" + program
                )
                parameters = (node, json.dumps(paths), json.dumps(failure_line))
                cap = review_mock_probe.COUNTER_REPORT_BYTES
            result = ws.command(
                (*interpreter, "-I", "-B", "-c", program, *parameters),
                timeout,
                budget.cancelled,
                cap=cap,
            )
    except Cancelled:
        raise
    except (ExecutionError, OSError):
        budget.remaining()
        return None
    if result.returncode != expected or result.stderr:
        return None
    if not paths:
        traces = review_mock_probe.decode_traces(result.stdout)
        return (traces, None) if traces is not None else None
    try:
        envelope = json.loads(result.stdout)
    except (ValueError, RecursionError):
        return None
    if type(envelope) is not dict or envelope.keys() != {"mocks", "counter"}:
        return None
    traces = review_mock_probe.decode_traces(
        json.dumps(envelope["mocks"], separators=(",", ":"))
    )
    counter = envelope["counter"]
    if traces is None or (
        counter is not None
        and not review_counter_trace.valid_report(counter, paths, failure_line)
    ):
        return None
    return traces, counter


def _differences(baseline, candidate):
    def render(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    lines = []
    remaining = _DETAIL_BYTES
    for name in sorted(baseline.keys() & candidate.keys()):
        before, after = baseline[name], candidate[name]
        old, new = render(before), render(after)
        if old == new:
            continue
        row = f"{name} on base {old}\n{name} on current {new}"
        if len(row) > remaining:
            # For a longer sequence, retain the first differing call and total counts.
            for index in range(max(len(before), len(after))):
                old = render(before[index]) if index < len(before) else "absent"
                new = render(after[index]) if index < len(after) else "absent"
                if old != new:
                    break
            row = (
                f"{name}, call {index + 1} on base {old} ({len(before)} calls)\n"
                f"{name}, call {index + 1} on current {new} ({len(after)} calls)"
            )
        if len(row) <= remaining:
            lines.append(row)
            remaining -= len(row) + 1
    if not lines:
        return ""
    return (
        "Auxiliary mock calls (quoted advisory data)\n"
        "The pinned base passed and the current draft failed. "
        "Each call is [method, arguments, keyword arguments].\n" + "\n".join(lines)
    )


def _counter_paths(task, runner, reviewed_paths, test_path):
    if getattr(runner, "network", "none") != "none":
        return ()
    if reviewed_paths is None:
        feedback = getattr(task.review_recovery, "feedback", ())
        reviewed_paths = tuple(path for path, _, _ in feedback)
    return tuple(
        path
        for path in dict.fromkeys(reviewed_paths)
        if path != test_path
        and len(path) <= review_counter_trace.MAX_PATH
        and re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.py", path)
        and str(PurePosixPath(path)) == path
        and ".." not in PurePosixPath(path).parts
    )[:3]


_CALLER_SOURCE_BYTES = 1024


def _caller_source(label, location, sources):
    path, function, row = location
    source = sources.get(path)
    if not source:
        return ""
    lines = source.splitlines(keepends=True)
    if not 1 <= row <= len(lines):
        return ""
    try:
        module = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return ""
    scopes = (
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function.rsplit(".", 1)[-1]
        and node.lineno <= row <= node.end_lineno
    )
    scope = max(scopes, key=lambda node: node.lineno, default=None)
    if scope is None:
        return ""
    nodes = list(ast.walk(scope))
    header = f"\n{label} caller source (quoted): {path}:{row} {function}\n"
    allowance = _CALLER_SOURCE_BYTES // 2 - len(header.encode())

    def source_lines(start, end):
        return "".join(lines[start - 1 : end]).rstrip("\r\n") + "\n"

    calls = [
        item
        for item in nodes
        if isinstance(item, ast.Call)
        and item.lineno <= row <= item.end_lineno <= item.lineno + 5
    ]
    call = max(calls, key=lambda item: item.end_lineno - item.lineno, default=None)
    start, end = (call.lineno, call.end_lineno) if call else (row, row)
    guards = [
        item
        for item in nodes
        if isinstance(item, ast.If)
        and item.lineno < start <= item.end_lineno
        and item.test.end_lineno < start
    ]
    guard = max(guards, key=lambda item: item.lineno, default=None)
    windows = (
        (max(scope.lineno, start - 1), min(scope.end_lineno, end + 1)),
        (start, end),
        (row, row),
    )
    if guard is not None:
        guard_end = guard.body[0].lineno - 1
        for first, last in windows:
            prefix = (
                source_lines(guard.lineno, min(guard_end, first - 1))
                if guard.lineno < first
                else ""
            )
            if guard_end + 1 < first:
                prefix += "# …\n"
            snippet = prefix + source_lines(first, last)
            if len(snippet.encode()) <= allowance:
                return header + snippet
    for first, last in windows:
        snippet = source_lines(first, last)
        if len(snippet.encode()) <= allowance:
            return header + snippet
    return ""


def _divergent_caller_source(old, new, sources):
    for pinned, draft in zip(old, new):
        base_stack, draft_stack = pinned[0], draft[0]
        if pinned[1:] != draft[1:] or base_stack[0][:2] != draft_stack[0][:2]:
            continue
        pair = next(
            (
                (offset, before, after)
                for offset, (before, after) in enumerate(
                    zip(base_stack[1:], draft_stack[1:]), 1
                )
                if before[:2] != after[:2]
            ),
            None,
        )
        if pair is not None:
            offset, before, after = pair
            base = _caller_source("Pinned", before, sources[0])
            attempted = _caller_source("Draft", after, sources[1])
            if not base or not attempted:
                return None
            locations = tuple(
                CounterCallerLocation(
                    path, name, line, hashlib.sha256(source.encode()).hexdigest()
                )
                for path, name, line in draft_stack[offset:]
                if (source := sources[1].get(path))
            )
            return CounterCallerSite(locations, base + attempted)
    return None


def _counter_difference(baseline, candidate, node, reviewed_sources=None):
    if baseline is None or candidate is None or baseline[0] != candidate[0]:
        return ""
    line, failed, windows = candidate
    _, base_failed, base_windows = baseline
    if base_failed is not None or failed is None or failed >= len(base_windows):
        return ""

    def signature(events):
        return tuple(
            (tuple((path, function) for path, function, _ in stack), before, after)
            for stack, before, after in events
        )

    for index in range(failed + 1):
        pinned, draft = base_windows[index], windows[index]
        if pinned is None or draft is None or pinned[0] != draft[0]:
            return ""
        if index < failed and signature(pinned[1]) != signature(draft[1]):
            return ""
    old, new = base_windows[failed][1], windows[failed][1]
    old_sig, new_sig = signature(old), signature(new)
    if old_sig == new_sig:
        return ""
    first = next(
        (
            index
            for index, (left, right) in enumerate(zip(old_sig, new_sig))
            if left != right
        ),
        min(len(old_sig), len(new_sig)),
    )

    def render(label, events):
        if not events:
            return f"{label}: no changes observed within reviewed frames."
        rows = [f"{label}: "]
        remaining = _DETAIL_BYTES // 2 - len(rows[0]) - 1
        selected = events[first : first + 2] or events[-1:]
        for stack, before, after in selected:
            left = "absent" if before is None else str(before)
            right = "absent" if after is None else str(after)
            row = f"{left} -> {right} after "
            for offset, (path, function, line) in enumerate(stack):
                location = f"{' <- ' if offset else ''}{path}:{line} {function}"
                if len(row) + len(location) + 2 > remaining:
                    break
                row += location
            if row.endswith(" after "):
                break
            rows.append(row + "; ")
            remaining -= len(row) + 2
        return "".join(rows).removesuffix("; ")

    rows = [
        render(label, events) for label, events in (("Pinned", old), ("Draft", new))
    ]
    text = (
        _COUNTER_HEADING + " "
        f"The pinned case passed and the draft failed at {node} line {line} (evaluation {failed + 1}). "
        "Locations show where the probe observed each change, with reviewed callers.\n"
        + "\n".join(rows)
    )
    site = (
        _divergent_caller_source(old[first:], new[first:], reviewed_sources)
        if reviewed_sources
        else None
    )
    return _ReplayDiagnostic(text + site.excerpts, site) if site else text


def mock_call_diagnostic(
    executor,
    task,
    archive,
    budget,
    command,
    node,
    patch,
    *,
    reviewed_paths=None,
    failure_line=None,
    reviewed_sources=None,
):
    node = node.removeprefix("/work/").removeprefix("./")
    base_node, bracket, parameter = node.partition("[")
    if (
        not patch
        or (
            failure_line is not None
            and (type(failure_line) is not int or not 1 <= failure_line <= 128 * 1024)
        )
        or command not in task.checks
        or command in task.pull_checks
        or len(node) > 512
        or not _STOCK_NODE.fullmatch(base_node)
        or (bracket and (not parameter.endswith("]") or not parameter.isprintable()))
        or (task.test_update or "").partition("::")[0] == node.split("::", 1)[0]
    ):
        return ""
    interpreter = _interpreter(command)
    reserve = len(set((*task.checks, *task.pull_checks)))
    if (
        interpreter is None
        or budget.commands_remaining() < reserve + 2
        or budget.remaining() < _PLANNING_RESERVE + 2 * _REPLAY_SECONDS
    ):
        return ""
    paths = _counter_paths(
        task, executor.verification_runner, reviewed_paths, base_node.partition("::")[0]
    )
    if paths and _tracked_stock_test(archive, _STOCK_NODE.fullmatch(base_node)) is None:
        paths = ()
    if paths and PurePosixPath(interpreter[0]).name == "uv":
        interpreter = ("env", "UV_OFFLINE=1", *interpreter)
    baseline = _readout(
        executor, task, archive, budget, interpreter, node, b"", 0, paths, failure_line
    )
    if baseline is None or (not any(baseline[0].values()) and baseline[1] is None):
        return ""
    candidate = _readout(
        executor,
        task,
        archive,
        budget,
        interpreter,
        node,
        patch,
        1,
        paths,
        failure_line,
    )
    if candidate is None:
        return ""
    counter = _counter_difference(
        baseline[1], candidate[1], base_node, reviewed_sources
    )
    text = "\n".join(filter(None, (counter, _differences(baseline[0], candidate[0]))))
    site = counter_caller_site(counter)
    return _ReplayDiagnostic(text, site) if site else text

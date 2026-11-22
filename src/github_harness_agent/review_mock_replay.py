"""Compare a stalled stock pytest case with its passing pinned baseline."""

import inspect
import json
import time
from types import SimpleNamespace

from . import review_mock_probe
from .review_prefix_replay import _interpreter
from .review_solver_context import _STOCK_NODE
from .worker import Cancelled, ExecutionError, LimitExceeded

_PLANNING_RESERVE = 30
_REPLAY_SECONDS = 10
_DETAIL_BYTES = 900


def _readout(executor, task, archive, budget, interpreter, node, patch, expected):
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
            result = ws.command(
                (
                    *interpreter,
                    "-I",
                    "-B",
                    "-c",
                    inspect.getsource(review_mock_probe),
                    node,
                ),
                timeout,
                budget.cancelled,
                cap=review_mock_probe.REPORT_BYTES,
            )
    except Cancelled:
        raise
    except (ExecutionError, OSError):
        budget.remaining()
        return None
    if result.returncode != expected or result.stderr:
        return None
    return review_mock_probe.decode_traces(result.stdout)


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


def mock_call_diagnostic(executor, task, archive, budget, command, node, patch):
    node = node.removeprefix("/work/").removeprefix("./")
    base_node, bracket, parameter = node.partition("[")
    if (
        not patch
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
    baseline = _readout(executor, task, archive, budget, interpreter, node, b"", 0)
    if not baseline or not any(baseline.values()):
        return ""
    candidate = _readout(executor, task, archive, budget, interpreter, node, patch, 1)
    return _differences(baseline, candidate) if candidate else ""

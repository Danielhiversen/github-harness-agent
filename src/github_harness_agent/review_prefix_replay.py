"""Optional verification-sandbox replay of a named operator assertion."""

import inspect
import json
import re
from pathlib import PurePosixPath

from . import review_prefixes
from .review_assertions import _rank
from .review_prefixes import PrefixReadout, prefix_improved
from .worker import LimitExceeded


def _interpreter(command):
    i = 0
    if command and PurePosixPath(command[0]).name == "uv" and command[1:2] == ("run",):
        i = 2
        while i < len(command) and command[i] in {
            "--no-sync",
            "--offline",
            "--frozen",
            "--locked",
            "--no-project",
            "--",
        }:
            i += 1
    if i >= len(command):
        return None
    executable = PurePosixPath(command[i])
    if re.fullmatch(r"python(?:[0-9.]+)?", executable.name) and command[
        i + 1 : i + 3
    ] == ("-m", "pytest"):
        return command[: i + 1]
    if executable.name == "pytest":
        if i and len(executable.parts) == 1:
            return (*command[:i], "python")
        if executable.is_absolute():
            return (*command[:i], str(executable.with_name("python")))
    return None


def _candidate_replay(assertions, fixtures, checks, target, before, after):
    number, node = target
    if (
        not 0 <= number < len(checks)
        or target not in before
        or target not in after
        or "[" in node
        or before.keys() != after.keys()
        or any(value != after[key] for key, value in before.items() if key != target)
    ):
        return None
    command = checks[number]
    old = _rank(assertions, command, node, before[target])
    new = _rank(assertions, command, node, after[target])
    interpreter = _interpreter(command)
    if old is None or old != new or interpreter is None:
        return None
    path, name, _, line = new
    if not name.isidentifier():
        return None
    source = fixtures.get(assertions[path][0])
    if source is None:
        return None
    statement = review_prefixes._assertion(source, name)
    if statement is None or statement.lineno != line:
        return None
    return interpreter, path, name, line, review_prefixes._clauses(source, statement)


def _measure(executor, task, archive, budget, patch, replay):
    interpreter, path, name, line, expected = replay
    with executor.verification_runner.workspace(
        archive, budget.remaining, budget.cancelled
    ) as ws:
        tree = executor._stage_baseline(ws, budget)
        if patch:
            executor._git(ws, budget, "apply", "--index", data=patch)
            executor._check_patch_paths(ws, task, budget, tree)
        if budget.remaining() <= 3:
            return None
        budget.claim("command", budget.limits.commands)
        source = inspect.getsource(review_prefixes)
        try:
            result = ws.command(
                (*interpreter, "-I", "-c", source, path, name),
                min(20, budget.remaining() - 3),
                budget.cancelled,
                cap=8192,
            )
        except LimitExceeded:
            budget.remaining()
            return None
    if result.returncode != 1 or not result.stdout.strip():
        return None
    try:
        row = json.loads(result.stdout.splitlines()[-1])
        clauses, values = row["clauses"], row["values"]
        if (
            row["line"] != line
            or not isinstance(clauses, list)
            or not isinstance(values, list)
            or tuple(clauses) != expected
            or len(values) != len(expected)
            or not all(item is None or type(item) is bool for item in values)
        ):
            return None
        return PrefixReadout(line, tuple(clauses), tuple(values))
    except (ValueError, KeyError, TypeError):
        return None


def observe_prefix_progress(
    executor,
    task,
    archive,
    budget,
    assertions,
    fixtures,
    target,
    before,
    after,
    patch,
    candidate,
):
    replay = _candidate_replay(
        assertions, fixtures, task.pull_checks, target, before, after
    )
    reserve = len(set((*task.checks, *task.pull_checks)))
    if (
        replay is None
        or budget.commands_remaining() < reserve + 2
        or budget.remaining() <= 5
    ):
        return ""
    old = _measure(executor, task, archive, budget, patch, replay)
    if old is None or old.passed_prefix() is None or budget.remaining() <= 3:
        return ""
    new = _measure(executor, task, archive, budget, candidate, replay)
    if new is None or not prefix_improved(old, new):
        return ""
    clauses = "\n".join(
        f"- {term}: {'not reached' if value is None else 'passed' if value else 'failed'}"
        for term, value in zip(new.clauses, new.values, strict=True)
    )
    return f"Auxiliary replay observed a longer passing prefix at line {new.line}\n{clauses}"

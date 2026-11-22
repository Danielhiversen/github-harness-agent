"""Recognize conservative, intra-test progress in trusted pytest fixtures."""

import ast
import re
from pathlib import PurePosixPath

from .check_artifacts import check_label

_ASSERTION_FAILURE = re.compile(
    r"^(?P<path>\S[^\n]*?\.py):(?P<line>[1-9][0-9]{0,6}): (?:assert\b|AssertionError\b)"
)
_STRAIGHT_LINE = (
    ast.Assert,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.Delete,
    ast.Expr,
    ast.Import,
    ast.ImportFrom,
    ast.Pass,
)


def _assertions(body, scope=()):
    for node in body:
        if isinstance(node, ast.ClassDef):
            yield from _assertions(node.body, (*scope, node.name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            block = 0
            for statement in node.body:
                if isinstance(statement, ast.Assert):
                    yield "::".join((*scope, node.name)), statement.lineno, block
                elif not isinstance(statement, _STRAIGHT_LINE):
                    block += 1


def operator_assertions(fixtures, operator_root):
    result = {}
    for path, source in fixtures.items():
        try:
            module = ast.parse(source)
        except (SyntaxError, ValueError, RecursionError):
            continue
        result[str(PurePosixPath(operator_root) / path)] = (
            path,
            {(name, line): block for name, line, block in _assertions(module.body)},
        )
    return result


def _rank(assertions, command, node, detail):
    node_path, _, name = node.split("[", 1)[0].partition("::")
    location = _ASSERTION_FAILURE.match(detail)
    if not name or not location:
        return None
    crash_path = location["path"]
    if not crash_path.startswith("/") or not any(
        argument.split("::", 1)[0] == crash_path for argument in command
    ):
        return None
    trusted_file = assertions.get(crash_path)
    if trusted_file is None:
        return None
    path, lines = trusted_file
    if node_path != path and not node_path.endswith("/" + path):
        return None
    line = int(location["line"])
    block = lines.get((name, line))
    return (crash_path, name, block, line) if block is not None else None


def assertion_progress(assertions, operator_commands, target, before, after):
    """One operator case advanced within a block without losing any check passes."""
    if (
        target not in before
        or target not in after
        or target[0] >= len(operator_commands)
    ):
        return False
    if not after.keys() <= before.keys():
        return False
    for key, detail in after.items():
        if key != target and detail == before[key]:
            continue
        if key[0] >= len(operator_commands):
            return False
        # Pytest short reports are associated with node IDs by order. Several
        # instances of the same parametrized function have indistinguishable locations.
        name = key[1].split("[", 1)[0]
        if any(
            other != key and other[0] == key[0] and other[1].split("[", 1)[0] == name
            for other in before
        ):
            return False
        command = operator_commands[key[0]]
        if not command or check_label(tuple(command), ()) != "pytest":
            return False
        previous = _rank(assertions, command, key[1], before[key])
        current = _rank(assertions, command, key[1], detail)
        if previous is None or current is None or previous[:-1] != current[:-1]:
            return False
        if current[-1] < previous[-1] or (
            key == target and current[-1] == previous[-1]
        ):
            return False
    return True

"""Trial one adjacent filter swap grounded in a typed counter replay."""

import ast
import hashlib
from typing import TYPE_CHECKING

from .review_definitions import _symbols

if TYPE_CHECKING:
    from .review_mock_replay import CounterLoopReuse

_FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)
_SOURCE_BYTES = 128_000
_GUARD_LINES = 16


def _reference(node):
    parts = []
    while isinstance(node, ast.Attribute) and len(parts) < 2:
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ()
    path = (node.id, *reversed(parts))
    return path if all(not item.startswith("__") for item in path) else ()


def _predicate(test, *, precondition=False):
    availability = None
    if precondition and isinstance(test, ast.BoolOp):
        if not isinstance(test.op, ast.And) or len(test.values) != 2:
            return None
        availability, test = test.values
    if not isinstance(test, ast.UnaryOp) or not isinstance(test.op, ast.Not):
        return None
    call = test.operand
    if (
        not isinstance(call, ast.Call)
        or not isinstance(call.func, ast.Attribute)
        or call.func.attr.startswith("__")
        or call.keywords
        or not 1 <= len(call.args) <= 3
        or any(not isinstance(arg, (ast.Name, ast.Constant)) for arg in call.args)
    ):
        return None
    receiver = _reference(call.func.value)
    if not receiver or (
        availability is not None and _reference(availability) != receiver
    ):
        return None
    return (
        receiver,
        call.func.attr,
        tuple(ast.dump(arg) for arg in call.args),
        {arg.id for arg in call.args if isinstance(arg, ast.Name)},
    )


def _filter(block):
    if (
        not isinstance(block, ast.If)
        or block.orelse
        or not isinstance(block.body[-1], ast.Continue)
        or len(block.body) > 2
        or not 0 < block.body[0].lineno - block.lineno < _GUARD_LINES
        or block.end_lineno - block.lineno >= _GUARD_LINES
    ):
        return None
    if len(block.body) == 1:
        return (), None
    effect = block.body[0]
    if (
        not isinstance(effect, ast.Expr)
        or not isinstance(effect.value, ast.Call)
        or effect.value.keywords
        or len(effect.value.args) != 1
        or not isinstance(reason := effect.value.args[0], ast.Constant)
        or type(reason.value) is not str
        or not 0 < len(reason.value) <= 160
        or effect.end_lineno >= block.body[-1].lineno
        or not (sink := _reference(effect.value.func))
    ):
        return None
    return sink, effect


def _function(tree, location):
    named = [node for name, node in _symbols(tree.body) if name == location.name]
    if len(named) != 1 or not isinstance(named[0], _FUNCTION):
        return None
    node = named[0]
    return node if node.lineno <= location.line <= node.end_lineno else None


def swap_traced_filters(sources, allowed, observed: "CounterLoopReuse | None"):
    """Return one exact source variant for verification when both guards match."""
    if (
        observed is None
        or observed.entries < 2
        or not 0 <= observed.pinned_mutations < observed.entries <= observed.mutations
    ):
        return None
    owner, mutation = observed.owner, observed.mutation
    if owner.path not in allowed or mutation.path not in allowed:
        return None
    texts, trees = {}, {}
    for location in (owner, mutation):
        source = sources.get(location.path)
        if (
            not source
            or len(source.encode()) > _SOURCE_BYTES
            or hashlib.sha256(source.encode()).hexdigest() != location.source_digest
        ):
            return None
        texts[location.path] = source
    try:
        trees = {path: ast.parse(source) for path, source in texts.items()}
        owner_tree = trees[owner.path]
        function = _function(owner_tree, owner)
        if function is None or _function(trees[mutation.path], mutation) is None:
            return None
        loops = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == observed.variable
            and node.body[0].lineno == observed.body_line
            and node.body[0].lineno <= owner.line <= node.body[-1].end_lineno
        ]
        if len(loops) != 1:
            return None
        loop = loops[0]
        for index, (first, second) in enumerate(zip(loop.body, loop.body[1:])):
            left, right = _filter(first), _filter(second)
            if left is None or right is None or left[1] is None:
                continue
            sink, effect = left
            if not effect.lineno <= owner.line <= effect.end_lineno:
                continue
            if mutation.name == owner.name and mutation.path == owner.path:
                if not effect.lineno <= mutation.line <= effect.end_lineno:
                    return None
            elif mutation.name.rsplit(".", 1)[-1] != sink[-1]:
                return None
            old = _predicate(first.test, precondition=True)
            new = _predicate(second.test)
            if (
                old is None
                or new is None
                or old[0] == new[0]
                or old[1:3] != new[1:3]
                or sink[0] in {observed.variable, *old[3]}
                or (
                    len(sink) > 1
                    and sink[:-1] in {old[0], new[0], _reference(loop.iter)}
                )
                or (
                    right[0]
                    and (right[0] != sink or ast.dump(right[1]) == ast.dump(effect))
                )
            ):
                return None
            lines = texts[owner.path].splitlines(keepends=True)
            start, between, following, end = (
                first.lineno - 1,
                first.end_lineno,
                second.lineno - 1,
                second.end_lineno,
            )
            if any(line.strip() for line in lines[between:following]):
                return None
            revised = "".join(
                (
                    *lines[:start],
                    *lines[following:end],
                    *lines[between:following],
                    *lines[start:between],
                    *lines[end:],
                )
            )
            loop.body[index : index + 2] = (second, first)
            if ast.dump(ast.parse(revised)) != ast.dump(owner_tree):
                return None
            compile(revised, owner.path, "exec")
            return {**sources, owner.path: revised}
    except (SyntaxError, ValueError, RecursionError):
        return None
    return None

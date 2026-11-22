"""Combine a small production edit with its previous behavior under a public input guard."""

import ast
from copy import copy
from textwrap import indent

from .review_definitions import _symbols

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_MAX_SOURCE_BYTES = 48 * 1024
_MAX_BLOCK_LINES = 8
_MAX_BLOCK_CHARS = 800


def _children(node):
    """Stay in the current function's scope."""
    return (
        child
        for child in ast.iter_child_nodes(node)
        if not isinstance(child, (*_FUNCTIONS, ast.ClassDef, ast.Lambda))
    )


def _statements(node, start, stop):
    for _, field in ast.iter_fields(node):
        if not isinstance(field, list):
            continue
        selected = [
            item
            for item in field
            if isinstance(item, ast.stmt)
            and start <= item.lineno
            and item.end_lineno <= stop
        ]
        if selected and selected[0].lineno == start and selected[-1].end_lineno == stop:
            return selected
    for child in _children(node):
        if (
            hasattr(child, "lineno")
            and child.lineno <= start
            and child.end_lineno >= stop
        ):
            block = _statements(child, start, stop)
            if block:
                return block
    return None


def _optional(annotation):
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return any(
            (isinstance(item, ast.Constant) and item.value is None) or _optional(item)
            for item in (annotation.left, annotation.right)
        )
    if isinstance(annotation, ast.Subscript):
        name = ast.unparse(annotation.value).removeprefix("typing.")
        return name == "Optional" or (
            name == "Union"
            and isinstance(annotation.slice, ast.Tuple)
            and any(
                isinstance(item, ast.Constant) and item.value is None
                for item in annotation.slice.elts
            )
        )
    return False


def _guard(function):
    parameters = {
        arg.arg: arg.annotation
        for arg in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
        if arg.arg not in {"self", "cls"} and not arg.arg.startswith("_")
    }
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            parameters.pop(node.id, None)
    for statement in function.body:
        if not isinstance(statement, ast.If) or not isinstance(
            statement.test, ast.Compare
        ):
            continue
        match statement.test:
            case ast.Compare(
                left=ast.Name(id=name),
                ops=[ast.Eq() | ast.NotEq()],
                comparators=[ast.Constant(value=str())],
            ):
                if name in parameters:
                    return ast.unparse(statement.test)
    for name, annotation in parameters.items():
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            annotation = ast.parse(annotation.value, mode="eval").body
        if isinstance(annotation, ast.Name) and annotation.id == "bool":
            return name
        if _optional(annotation):
            return f"{name} is None"
    return None


def _bounded_site(source, alternative):
    before, after = (
        source.splitlines(keepends=True),
        alternative.splitlines(keepends=True),
    )
    start = 0
    for left, right in zip(before, after, strict=False):
        if left != right:
            break
        start += 1
    old_end, new_end = len(before), len(after)
    while min(old_end, new_end) > start and before[old_end - 1] == after[new_end - 1]:
        old_end -= 1
        new_end -= 1
    if not all(0 < end - start <= _MAX_BLOCK_LINES for end in (old_end, new_end)):
        return None
    blocks = "".join(before[start:old_end]), "".join(after[start:new_end])
    if any(len(block) > _MAX_BLOCK_CHARS for block in blocks):
        return None
    return before, start, old_end, new_end, blocks


def _expression(old, new, predicate):
    if type(old) is not type(new) or not isinstance(
        old, (ast.Return, ast.Assign, ast.AnnAssign)
    ):
        return None
    if old.value is None or new.value is None:
        return None
    # Compare the statements with their value replaced, then splice just the expression.
    old_form, new_form = copy(old), copy(new)
    old_form.value = new_form.value = None
    if ast.dump(old_form) != ast.dump(new_form):
        return None
    old_form.value = ast.IfExp(
        ast.parse(predicate, mode="eval").body, new.value, old.value
    )
    return ast.unparse(old_form) + "\n"


def guarded_sources(source, alternative):
    """Yield up to two candidates for one small change inside an existing function."""
    if max(len(source.encode()), len(alternative.encode())) > _MAX_SOURCE_BYTES:
        return
    site = _bounded_site(source, alternative)
    if site is None:
        return
    before, start, old_end, new_end, blocks = site
    try:
        new_defs = dict(_symbols(ast.parse(alternative).body))
        for name, function in _symbols(ast.parse(source).body):
            if (
                not isinstance(function, _FUNCTIONS)
                or not function.lineno <= start < old_end <= function.end_lineno
            ):
                continue
            other = new_defs.get(name)
            if not isinstance(other, _FUNCTIONS) or ast.dump(function.args) != ast.dump(
                other.args
            ):
                return
            old, new = (
                _statements(function, start + 1, old_end),
                _statements(other, start + 1, new_end),
            )
            if not old or not new or old[0].col_offset != new[0].col_offset:
                return
            if any(
                isinstance(item, (*_FUNCTIONS, ast.ClassDef, ast.Global, ast.Nonlocal))
                or (
                    isinstance(item, ast.Constant)
                    and isinstance(item.value, str)
                    and "\n" in item.value
                )
                for statement in (*old, *new)
                for item in ast.walk(statement)
            ):
                return
            prefix = " " * old[0].col_offset
            if any(
                line.strip() and not line.startswith(prefix)
                for block in blocks
                for line in block.splitlines()
            ):
                return
            guard = _guard(function)
            if guard is None:
                return
            for base, edited, rollback, proposed in (
                (*blocks, old, new),
                (*reversed(blocks), new, old),
            ):
                expression = None
                if len(rollback) == len(proposed) == 1 and "#" not in base + edited:
                    expression = _expression(rollback[0], proposed[0], guard)
                replacement = (
                    indent(expression, prefix)
                    if expression
                    else f"{prefix}if {guard}:\n{indent(edited, '    ')}{prefix}else:\n{indent(base, '    ')}"
                )
                result = (
                    "".join(before[:start]) + replacement + "".join(before[old_end:])
                )
                compile(result, "<guarded review edit>", "exec")
                yield result
            return
    except (SyntaxError, ValueError, RecursionError):
        return

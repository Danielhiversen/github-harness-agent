"""Combine a small production edit with its previous behavior under a public input guard."""

import ast
import tokenize
from copy import copy
from io import StringIO
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


def _guards(function, *, before_predicate=None):
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
        elif before_predicate is not None:
            # Several bindings use strings instead of Name(Store) nodes.
            if isinstance(node, ast.alias):
                parameters.pop(node.asname or node.name.split(".", 1)[0], None)
                if node.name == "*":
                    return
            elif isinstance(
                node,
                (
                    *_FUNCTIONS,
                    ast.ClassDef,
                    ast.ExceptHandler,
                    ast.MatchAs,
                    ast.MatchStar,
                ),
            ):
                parameters.pop(node.name, None)
            elif isinstance(node, ast.MatchMapping):
                parameters.pop(node.rest, None)
    for statement in function.body:
        if (
            not isinstance(statement, ast.If)
            or not isinstance(statement.test, ast.Compare)
            or (before_predicate is not None and statement.lineno >= before_predicate)
        ):
            continue
        match statement.test:
            case ast.Compare(
                left=ast.Name(id=name),
                ops=[ast.Eq() | ast.NotEq()],
                comparators=[ast.Constant(value=str())],
            ):
                if name in parameters:
                    yield ast.unparse(statement.test)
    for name, annotation in parameters.items():
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            annotation = ast.parse(annotation.value, mode="eval").body
        if isinstance(annotation, ast.Name) and annotation.id == "bool":
            yield name
        if _optional(annotation):
            yield f"{name} is None"


def _guard(function):
    return next(_guards(function), None)


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


def _predicate(function, start, end):
    pending = list(_children(function))
    matches = []
    while pending:
        node = pending.pop()
        if (
            isinstance(node, ast.If)
            and node.test.lineno <= start <= end <= node.test.end_lineno
        ):
            matches.append(node)
        pending.extend(
            child
            for child in _children(node)
            if hasattr(child, "lineno")
            and child.lineno <= start <= end <= child.end_lineno
        )
    return matches[0] if len(matches) == 1 else None


def _predicate_parts(source, expression):
    raw = source.encode()
    lines = raw.splitlines(keepends=True)
    start = sum(map(len, lines[: expression.lineno - 1])) + expression.col_offset
    end = sum(map(len, lines[: expression.end_lineno - 1])) + expression.end_col_offset
    return raw[:start], raw[start:end], raw[end:]


def _predicate_site(source, alternative, function, other, start, old_end, new_end):
    old = _predicate(function, start, old_end)
    new = _predicate(other, start, new_end)
    if old is None or new is None:
        return None
    prefix, previous, suffix = _predicate_parts(source, old.test)
    new_prefix, edited, new_suffix = _predicate_parts(alternative, new.test)
    return old, new, prefix, previous, edited, suffix, new_prefix, new_suffix


def _guarded_predicates(function, site):
    old, new, prefix, previous, edited, suffix, new_prefix, new_suffix = site
    if (
        prefix != new_prefix
        or suffix != new_suffix
        or ast.dump(old.test) == ast.dump(new.test)
    ):
        return
    if any(
        isinstance(
            item, (ast.NamedExpr, ast.Yield, ast.YieldFrom, ast.Await, ast.Lambda)
        )
        for expression in (old.test, new.test)
        for item in ast.walk(expression)
    ):
        return
    for expression in (previous, edited):
        tokens = tokenize.generate_tokens(
            StringIO("(" + expression.decode() + ")").readline
        )
        if any(token.type == tokenize.COMMENT for token in tokens):
            return
    guards = list(dict.fromkeys(_guards(function, before_predicate=old.lineno)))
    if len(guards) != 1:
        return
    guard = guards[0].encode()
    alternatives = []
    for selected, fallback in ((edited, previous), (previous, edited)):
        expression = (
            b"((" + selected + b") if (" + guard + b") else (" + fallback + b"))"
        )
        result = (prefix + expression + suffix).decode()
        compile(result, "<guarded review predicate>", "exec")
        alternatives.append(result)
    yield from alternatives


def guarded_sources(source, alternative):
    """Yield up to two candidates for one small change inside an existing function."""
    if max(len(source.encode()), len(alternative.encode())) > _MAX_SOURCE_BYTES:
        return
    site = _bounded_site(source, alternative)
    if site is None:
        return
    before, start, old_end, new_end, blocks = site
    try:
        new_symbols = list(_symbols(ast.parse(alternative).body))
        new_defs = dict(new_symbols)
        old_symbols = list(_symbols(ast.parse(source).body))
        for name, function in old_symbols:
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
            predicate = _predicate_site(
                source, alternative, function, other, start + 1, old_end, new_end
            )
            if predicate is not None:
                if any(
                    sum(symbol == name for symbol, _ in entries) != 1
                    for entries in (old_symbols, new_symbols)
                ):
                    return
                yield from _guarded_predicates(function, predicate)
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
    except (SyntaxError, tokenize.TokenError, ValueError, RecursionError):
        return

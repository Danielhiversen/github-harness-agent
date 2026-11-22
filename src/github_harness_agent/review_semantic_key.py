"""Prepare a private trial using the existing sort key for one traced diagnostic."""

import ast
import hashlib

from .review_guard_order import _filter
from .review_mock_replay import CounterLoopReuse

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
_BUILTINS = {
    "all",
    "bool",
    "dict",
    "float",
    "int",
    "object",
    "set",
    "sorted",
    "str",
    "tuple",
    "type",
}
_NAMES = {
    "reported_diagnostic_keys",
    "diagnostic_fields",
    "diagnostic_key",
    "report_diagnostic",
}


def _nodes(body):
    pending = list(reversed(body))
    while pending:
        item = pending.pop()
        yield item
        if not isinstance(item, _SCOPES):
            pending.extend(reversed(list(ast.iter_child_nodes(item))))


def _function(tree, location):
    scope = tree
    for part in location.name.split("."):
        matches = [
            item
            for item in scope.body
            if isinstance(item, _SCOPES) and item.name == part
        ]
        if len(matches) != 1:
            return None
        scope = matches[0]
    if (
        isinstance(scope, _SCOPES[:2])
        and scope.lineno <= location.line <= scope.end_lineno
    ):
        return scope
    return None


def _name(node, name):
    return isinstance(node, ast.Name) and node.id == name


def _binding(node):
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        return node.id
    if isinstance(node, ast.arg):
        return node.arg
    if isinstance(node, ast.alias):
        return node.asname or node.name.split(".")[0]
    if isinstance(
        node,
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.ExceptHandler,
            ast.MatchAs,
            ast.MatchStar,
        ),
    ):
        return node.name
    if isinstance(node, ast.MatchMapping):
        return node.rest
    return None


def _member_call(node, member):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == member
    )


def _bucket_stream(function, loop, sort, record):
    """Prove the bucket comes from exactly the traced record's append stream."""
    rows = sort.args[0]
    if not isinstance(rows, ast.Name):
        return False
    for group in _nodes(function.body):
        if (
            not isinstance(group, ast.For)
            or not isinstance(group.target, ast.Tuple)
            or len(group.target.elts) != 2
            or not all(isinstance(target, ast.Name) for target in group.target.elts)
            or group.target.elts[1].id != rows.id
            or not _member_call(group.iter, "items")
            or group.iter.args
            or group.iter.keywords
            or not isinstance(mapping := group.iter.func.value, ast.Name)
            or group.lineno <= loop.end_lineno
        ):
            continue
        assignment = next(
            (
                stmt
                for stmt in group.body
                if isinstance(stmt, ast.Assign) and sort in ast.walk(stmt.value)
            ),
            None,
        )
        if (
            assignment is None
            or len(assignment.targets) != 1
            or not isinstance(target := assignment.targets[0], ast.Subscript)
            or not _name(target.value, mapping.id)
            or not _name(target.slice, group.target.elts[0].id)
        ):
            continue
        first = [
            stmt.target if isinstance(stmt, ast.AnnAssign) else stmt.targets[0]
            for stmt in function.body
            if (
                isinstance(stmt, ast.AnnAssign)
                or isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
            )
            and isinstance(stmt.value, ast.Dict)
            and not stmt.value.keys
            and stmt.lineno < loop.lineno
            and _name(
                stmt.target if isinstance(stmt, ast.AnnAssign) else stmt.targets[0],
                mapping.id,
            )
        ]
        if len(first) != 1 or any(
            isinstance(node, (ast.Global, ast.Nonlocal)) and mapping.id in node.names
            for node in _nodes(function.body)
        ):
            continue
        appends = [
            call.func.value.func.value
            for call in _nodes(loop.body)
            if _member_call(call, "append")
            and len(call.args) == 1
            and _name(call.args[0], record)
            and not call.keywords
            and _member_call(call.func.value, "setdefault")
            and len(call.func.value.args) == 2
            and not call.func.value.keywords
            and _name(call.func.value.func.value, mapping.id)
            and isinstance(factory := call.func.value.args[1], ast.Call)
            and _name(factory.func, "deque")
            and not factory.args
            and not factory.keywords
        ]
        legal = {mapping, target.value, *first, *appends}
        earlier = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Name) and node.lineno <= sort.lineno
        ]
        rebindings = [
            node
            for node in ast.walk(function)
            if _binding(node) == rows.id
            and getattr(node, "lineno", sort.lineno + 1) <= sort.lineno
        ]
        if rebindings != [group.target.elts[1]]:
            continue
        if appends and all(
            (node.id != mapping.id or node in legal)
            and (
                node.id != rows.id
                or isinstance(node.ctx, ast.Load)
                or node is group.target.elts[1]
            )
            for node in earlier
        ):
            return True
    return False


def _fields(function, loop, variable, *, deque):
    keys = []
    for sort in _nodes(function.body):
        if (
            not isinstance(sort, ast.Call)
            or not _name(sort.func, "sorted")
            or len(sort.args) != 1
            or any(keyword.arg not in {"key", "reverse"} for keyword in sort.keywords)
        ):
            continue
        projections = [
            keyword.value for keyword in sort.keywords if keyword.arg == "key"
        ]
        if not projections:
            continue
        direct = (
            sort is loop.iter
            or isinstance(loop.iter, ast.Name)
            and _name(sort.args[0], loop.iter.id)
            and not any(
                _binding(node) == loop.iter.id and not isinstance(node, ast.arg)
                for node in ast.walk(function)
            )
        )
        if direct or deque and _bucket_stream(function, loop, sort, variable):
            keys.extend(projections)
    if len(keys) != 1 or not isinstance(key := keys[0], ast.Lambda):
        return None
    args = key.args
    if (
        len(args.args) != 1
        or args.posonlyargs
        or args.kwonlyargs
        or args.vararg
        or args.kwarg
        or args.defaults
        or not isinstance(key.body, ast.Tuple)
        or not 1 <= len(key.body.elts) <= 6
    ):
        return None
    fields = []
    for part in key.body.elts:
        convert = (
            isinstance(part, ast.Call)
            and _name(part.func, "str")
            and len(part.args) == 1
            and not part.keywords
        )
        load = part.args[0] if convert else part
        if isinstance(load, ast.Subscript) and _name(load.value, args.args[0].arg):
            literal = load.slice
        elif (
            _member_call(load, "get")
            and _name(load.func.value, args.args[0].arg)
            and len(load.args) == 1
            and not load.keywords
        ):
            literal = load.args[0]
        else:
            return None
        if (
            not isinstance(literal, ast.Constant)
            or type(literal.value) is not str
            or not 0 < len(literal.value) <= 60
        ):
            return None
        fields.append((literal.value, convert))
    return fields if len({name for name, _ in fields}) == len(fields) else None


def _safe_names(tree):
    imports = {
        alias
        for stmt in tree.body
        if isinstance(stmt, ast.ImportFrom) and stmt.level == 0
        for alias in stmt.names
        if not alias.asname
        and (stmt.module, alias.name)
        in {
            ("datetime", "datetime"),
            ("datetime", "UTC"),
            ("collections", "deque"),
        }
    }
    canonical = {alias.name for alias in imports}
    if len(canonical) != len(imports):
        return False, False, False
    clock = {"datetime", "UTC"} <= canonical
    deque = "deque" in canonical
    reserved = _BUILTINS | _NAMES | canonical
    for node in ast.walk(tree):
        if node in imports:
            continue
        if (
            _binding(node) in reserved
            or isinstance(node, (ast.Global, ast.Nonlocal))
            and reserved.intersection(node.names)
            or isinstance(node, ast.alias)
            and node.name == "*"
            or isinstance(node, ast.Name)
            and node.id in _NAMES
        ):
            return False, False, False
    return True, clock, deque


def _guard(indent, newline, effect, variable, fields, clock):
    lines = [
        "report_diagnostic = True",
        f"if type({variable}) is dict and all(type(field_name) is str for field_name in {variable}):",
        "    diagnostic_fields = [",
        *(
            f"        ({variable}.get({key!r}), {convert!r}),"
            for key, convert in fields
        ),
        "    ]",
        "    if all(",
        "        type(field_value) is str or type(field_value) is bool",
        "        or (type(field_value) is int and (not as_text or field_value.bit_length() <= 256))",
        "        or (type(field_value) is float and field_value == field_value)",
        *(
            [
                "        or (as_text and type(field_value) is datetime and field_value.tzinfo is UTC)"
            ]
            if clock
            else []
        ),
        "        for field_value, as_text in diagnostic_fields",
        "    ):",
        "        diagnostic_key = tuple(",
        "            str(value) if as_text else value for value, as_text in diagnostic_fields",
        "        )",
        "        report_diagnostic = diagnostic_key not in reported_diagnostic_keys",
        "        reported_diagnostic_keys.add(diagnostic_key)",
        "if report_diagnostic:",
        "    " + effect,
    ]
    return "".join(indent + line + newline for line in lines)


def guard_traced_counter_once(sources, signal, allowed):
    """Return a candidate only; the caller must independently verify every check."""
    if (
        not isinstance(signal, CounterLoopReuse)
        or signal.entries < 2
        or not 0 <= signal.pinned_mutations < signal.entries <= signal.mutations
    ):
        return None
    owner, mutation = signal.owner, signal.mutation
    trees = {}
    try:
        for site in (owner, mutation):
            source = sources.get(site.path, "")
            if (
                site.path not in allowed
                or not site.path.endswith(".py")
                or not 0 < len(source.encode()) <= 128_000
                or hashlib.sha256(source.encode()).hexdigest() != site.source_digest
            ):
                return None
            trees[site.path] = ast.parse(source)
        tree = trees[owner.path]
        function = _function(tree, owner)
        if function is None or _function(trees[mutation.path], mutation) is None:
            return None
        safe, clock, deque = _safe_names(tree)
        loops = [
            node
            for node in _nodes(function.body)
            if isinstance(node, ast.For)
            and _name(node.target, signal.variable)
            and node.body[0].lineno == signal.body_line
            and node.body[0].lineno <= owner.line <= node.body[-1].end_lineno
        ]
        if not safe or len(loops) != 1:
            return None
        loop = loops[0]
        if any(_binding(node) == signal.variable for node in _nodes(loop.body)):
            return None
        fields = _fields(function, loop, signal.variable, deque=deque)
        if fields is None:
            return None
        effects = [
            (sink, effect)
            for stmt in loop.body
            if (matched := _filter(stmt))
            and (sink := matched[0])
            and (effect := matched[1])
            and effect.lineno == effect.end_lineno == owner.line
        ]
        if len(effects) != 1:
            return None
        sink, effect = effects[0]
        if (owner.path, owner.name) == (mutation.path, mutation.name):
            if mutation.line != effect.lineno:
                return None
        elif mutation.name.rsplit(".", 1)[-1] != sink[-1]:
            return None
        source = sources[owner.path]
        lines = source.splitlines(keepends=True)
        newline = "\r\n" if "\r\n" in source else "\n"
        head = function.body[0]
        if (
            isinstance(head, ast.Expr)
            and isinstance(head.value, ast.Constant)
            and type(head.value.value) is str
        ):
            head = function.body[1]
        before, row = head.lineno - 1, effect.lineno - 1
        start, indent = (
            lines[before][: head.col_offset],
            lines[row][: effect.col_offset],
        )
        if not start or start.strip(" ") or not indent or indent.strip(" "):
            return None
        lines[row] = _guard(
            indent,
            newline,
            lines[row][effect.col_offset :].removesuffix(newline),
            signal.variable,
            fields,
            clock,
        )
        lines.insert(
            before,
            start
            + "reported_diagnostic_keys: set[tuple[object, ...]] = set()"
            + newline,
        )
        revised = "".join(lines)
        compile(revised, owner.path, "exec")
        return {**sources, owner.path: revised}
    except (IndexError, SyntaxError, ValueError, RecursionError):
        return None

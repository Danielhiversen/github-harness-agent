"""Prepare one private queue-round guard from matching pinned and current source."""

import ast
from dataclasses import dataclass

from .review_definitions import _symbols

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
_SOURCE_BYTES = 128 * 1024


@dataclass(frozen=True)
class RoundLimitGuard:
    sources: dict[str, str]
    keyword: str
    factory: str
    advisory: str


def _walk(items):
    todo = list(items)
    while todo:
        node = todo.pop()
        yield node
        if not isinstance(node, _SCOPES):
            todo.extend(ast.iter_child_nodes(node))


def _name(node, name):
    return isinstance(node, ast.Name) and node.id == name


def _number(node, number):
    return (
        isinstance(node, ast.Constant)
        and type(node.value) is int
        and node.value == number
    )


def _unshadowed(tree, names):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.id in names
        ):
            return False
        if (
            isinstance(
                node,
                (
                    ast.ClassDef,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ExceptHandler,
                    ast.MatchAs,
                    ast.MatchStar,
                ),
            )
            and node.name in names
        ):
            return False
        if isinstance(node, ast.arg) and node.arg in names:
            return False
        if isinstance(node, ast.alias) and (
            (node.asname or node.name.split(".")[0]) in names or node.name == "*"
        ):
            return False
        if isinstance(node, (ast.Global, ast.Nonlocal)) and names.intersection(
            node.names
        ):
            return False
    return True


def _method(tree, name):
    matches = [node for qualified, node in _symbols(tree.body) if qualified == name]
    return (
        matches[0]
        if len(matches) == 1 and isinstance(matches[0], ast.FunctionDef)
        else None
    )


def _assignment(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        return node.targets[0], node.value
    if isinstance(node, ast.AnnAssign) and node.value is not None:
        return node.target, node.value
    return None, None


def _member(call, *names):
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in names
    )


def _call(call, name):
    return isinstance(call, ast.Call) and _name(call.func, name)


def _base(node):
    if isinstance(node, ast.Subscript):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _parameters(function):
    args = function.args
    positional = [item.arg for item in (*args.posonlyargs, *args.args)]
    defaults = dict(
        zip(
            positional[len(positional) - len(args.defaults) :],
            args.defaults,
            strict=True,
        )
    )
    defaults.update(
        zip((arg.arg for arg in args.kwonlyargs), args.kw_defaults, strict=True)
    )
    return positional, defaults


def _positive_fixture_value(fixture, call, value):
    def positive(node):
        return (
            isinstance(node, ast.Constant)
            and type(node.value) is int
            and node.value > 0
        )

    if positive(value):
        return True
    if not isinstance(value, ast.Name):
        return False
    owners = []
    for loop in ast.walk(fixture):
        if not isinstance(loop, ast.For) or not isinstance(
            loop.iter, (ast.Tuple, ast.List)
        ):
            continue
        if not 1 <= len(loop.iter.elts) <= 12 or call not in _walk(loop.body):
            continue
        if _name(loop.target, value.id):
            entries = loop.iter.elts
        elif isinstance(loop.target, (ast.Tuple, ast.List)):
            slots = [
                i
                for i, target in enumerate(loop.target.elts)
                if _name(target, value.id)
            ]
            if len(slots) != 1 or any(
                not isinstance(target, ast.Name) for target in loop.target.elts
            ):
                continue
            if not all(
                isinstance(row, (ast.Tuple, ast.List))
                and len(row.elts) == len(loop.target.elts)
                for row in loop.iter.elts
            ):
                continue
            entries = [item.elts[slots[0]] for item in loop.iter.elts]
        else:
            continue
        owners.append(
            all(map(positive, entries))
            and not any(
                _name(node, value.id) and isinstance(node.ctx, (ast.Store, ast.Del))
                for node in _walk(loop.body)
            )
        )
    return owners == [True]


def _bounded_keywords(pinned, current, fixture):
    used = {
        keyword.arg
        for call in ast.walk(fixture)
        if isinstance(call, ast.Call)
        and (
            _name(call.func, current.name)
            or isinstance(call.func, ast.Attribute)
            and call.func.attr == current.name
        )
        for keyword in call.keywords
        if _positive_fixture_value(fixture, call, keyword.value)
    }
    params, defaults = _parameters(current)
    for loop in _walk(pinned.body):
        if not isinstance(loop, ast.For) or not _call(loop.iter, "range"):
            continue
        if len(loop.iter.args) != 1 or loop.iter.keywords:
            continue
        argument = loop.iter.args[0]
        if not isinstance(argument, ast.Name) or argument.id not in used.intersection(
            (*params, *defaults)
        ):
            continue
        key = argument.id
        reads = [node for node in _walk(current.body) if _name(node, key)]
        guards = [
            node.test
            for node in _walk(current.body)
            if isinstance(node, ast.If)
            and not node.orelse
            and len(node.body) == 1
            and isinstance(node.body[0], ast.Raise)
            and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1
        ]
        if len(reads) != 1 or not isinstance(reads[0].ctx, ast.Load):
            continue
        for check in guards:
            left, right = check.left, check.comparators[0]
            direct = left is reads[0]
            minimum = _call(left, "min") and not left.keywords and reads[0] in left.args
            if (direct or minimum) and (
                isinstance(check.ops[0], ast.Lt)
                and _number(right, 1)
                or isinstance(check.ops[0], ast.LtE)
                and _number(right, 0)
            ):
                yield key, loop
                break


def _constructor_arg(call, constructor, param, default):
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(
        kw.arg is None for kw in call.keywords
    ):
        return None
    positional, _ = _parameters(constructor)
    candidates = [kw.value for kw in call.keywords if kw.arg == param]
    if param in positional[1:] and (index := positional.index(param) - 1) < len(
        call.args
    ):
        candidates.append(call.args[index])
    if len(candidates) > 1:
        return None
    return candidates[0] if candidates else default


def _factory(tree, class_name, method_name):
    defs = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(defs) != 1 or defs[0].bases or defs[0].keywords or defs[0].decorator_list:
        return None
    cls = defs[0]
    init = _method(tree, f"{class_name}.__init__")
    method = _method(tree, f"{class_name}.{method_name}")
    if init is None or method is None or init.decorator_list or method.decorator_list:
        return None
    annotations = method.returns
    if (
        not isinstance(annotations, ast.Subscript)
        or not _name(annotations.value, "list")
        or not _name(annotations.slice, class_name)
    ):
        return None
    makers = [
        member
        for member in cls.body
        if isinstance(member, ast.FunctionDef)
        and any(_call(node, class_name) for node in _walk(member.body))
    ]
    if makers != [method]:
        return None
    init_args, defaults = _parameters(init)
    method_args, _ = _parameters(method)
    if (
        not init_args
        or len(method_args) != 1
        or method.args.vararg
        or method.args.kwarg
        or method.args.kwonlyargs
    ):
        return None
    for statement in _walk(method.body):
        if (
            statement is method.body[0]
            and isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and type(statement.value.value) is str
        ):
            continue
        if isinstance(statement, ast.stmt) and not isinstance(
            statement, (ast.If, ast.Return, ast.Assign, ast.AnnAssign)
        ):
            return None
        target, value = _assignment(statement)
        if isinstance(statement, (ast.Assign, ast.AnnAssign)) and (
            value is None or not isinstance(target, ast.Name)
        ):
            return None
        if isinstance(statement, ast.Call) and not any(
            _call(statement, allowed) for allowed in (class_name, "min", "max", "range")
        ):
            return None
        if isinstance(statement, (ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)):
            return None
    products = []
    for node in _walk(method.body):
        if not isinstance(node, ast.Return):
            continue
        value = node.value
        if isinstance(value, ast.ListComp):
            products.append(value.elt)
        elif isinstance(value, ast.List):
            products.extend(value.elts)
        else:
            return None
    calls = [node for node in _walk(method.body) if _call(node, class_name)]
    if not calls or set(calls) != set(products):
        return None
    options = []
    for statement in init.body:
        target, value = _assignment(statement)
        if (
            not isinstance(target, ast.Attribute)
            or not _name(target.value, init_args[0])
            or not isinstance(value, ast.Name)
            or not _number(defaults.get(value.id), 0)
        ):
            continue
        # Reassigning this state anywhere in the class makes its level ambiguous.
        writes = [
            node
            for node in ast.walk(cls)
            if isinstance(node, ast.Attribute)
            and node.attr == target.attr
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ]
        if writes != [target]:
            continue
        evidence = [_constructor_arg(call, init, value.id, None) for call in calls]
        if all(
            isinstance(step, ast.BinOp)
            and isinstance(step.op, ast.Add)
            and isinstance(step.left, ast.Attribute)
            and step.left.attr == target.attr
            and _name(step.left.value, method_args[0])
            and _number(step.right, 1)
            for step in evidence
        ):
            options.append((target.attr, value.id, statement, evidence[0]))
    if len(options) != 1:
        return None
    return init, method, options[0]


def _wrapper(node):
    while (
        isinstance(node, ast.Call)
        and (
            _name(node.func, "deque")
            or _name(node.func, "list")
            or _name(node.func, "reversed")
        )
        and len(node.args) == 1
        and not node.keywords
    ):
        node = node.args[0]
    return node


def _performers(tree, qualified, owner):
    prefix, _, _ = qualified.rpartition(".")
    params, _ = _parameters(owner)
    receiver = params[0] if prefix and params else None
    result = {}
    for call in _walk(owner.body):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
            if receiver is None or not _name(call.func.value, receiver):
                continue
            name = f"{prefix}.{call.func.attr}"
        elif isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
            name = call.func.id
        else:
            continue
        if (helper := _method(tree, name)) is not None and helper is not owner:
            result[call] = name, helper
    return result


def _roots(tree, qualified, owner, class_name, init, param):
    helpers = _performers(tree, qualified, owner)
    roots = {}
    for call, (name, helper) in helpers.items():
        nodes = list(_walk(helper.body))
        constructors = [node for node in nodes if _call(node, class_name)]
        if not constructors:
            continue
        positional, defaults = _parameters(init)
        if not all(
            _number(_constructor_arg(node, init, param, defaults.get(param)), 0)
            for node in constructors
        ):
            return None
        outputs = [node.value for node in nodes if isinstance(node, ast.Return)]
        # Prove explicit constructors can reach a returned collection either as
        # payloads or by appending to the returned local or one of its buckets.
        names = {value.id for value in outputs if isinstance(value, ast.Name)}
        reachable = set()
        for output in outputs:
            if output is not None:
                reachable.update(ast.walk(output))
        for node in nodes:
            target, value = _assignment(node)
            if value is not None and _base(target) in names:
                reachable.update(ast.walk(value))
            if _member(node, "append", "extend", "appendleft") and len(node.args) == 1:
                collection = node.func.value
                if _member(collection, "setdefault"):
                    collection = collection.func.value
                if _base(collection) in names:
                    reachable.update(ast.walk(node.args[0]))
        resolved = [node for node in constructors if node in reachable]
        for statement in nodes:
            target, value = _assignment(statement)
            if (
                isinstance(target, ast.Name)
                and value in constructors
                and any(
                    isinstance(item, ast.Name)
                    and isinstance(item.ctx, ast.Load)
                    and item.id == target.id
                    for item in reachable
                )
            ):
                resolved.append(value)
        if not outputs or set(resolved) != set(constructors):
            return None
        roots[call] = name, constructors[0]
    return roots


def _factory_calls(body):
    return [
        call
        for call in _walk(body)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and not call.args
        and not call.keywords
    ]


def _pins(pinned, round_loop, factory, roots):
    for record_loop in _walk(round_loop.body):
        if not isinstance(record_loop, ast.For) or not isinstance(
            record_loop.target, ast.Name
        ):
            continue
        queue = _base(record_loop.iter)
        seeds = [
            roots[origin]
            for statement in _walk(pinned.body)
            if (assignment := _assignment(statement))[1] is not None
            and _base(assignment[0]) == queue
            and statement.lineno < round_loop.lineno
            and (origin := _wrapper(assignment[1])) in roots
        ]
        if len(seeds) != 1:
            continue
        known = [
            call
            for call in _factory_calls(record_loop.body)
            if call.func.attr == factory
            and _name(call.func.value, record_loop.target.id)
        ]
        if len(known) != 1:
            continue
        grown = [
            write.func.value.id
            for write in _walk(record_loop.body)
            if _member(write, "extend", "appendleft", "append")
            and isinstance(write.func.value, ast.Name)
            and write.args == known
            and not write.keywords
        ]
        if len(grown) != 1:
            continue
        next_layer = grown[0]
        initial = any(
            isinstance(value, ast.List) and not value.elts and _name(target, next_layer)
            for stmt in round_loop.body
            if stmt.lineno < record_loop.lineno
            for target, value in (_assignment(stmt),)
        )
        advanced = any(
            _name(target, queue) and _name(value, next_layer)
            for stmt in round_loop.body
            if stmt.lineno > record_loop.end_lineno
            for target, value in (_assignment(stmt),)
        )
        if initial and advanced:
            yield seeds[0]


def _queue_roots(current, item, call, roots):
    body = [node for node in _walk(current.body) if isinstance(node, ast.stmt)]
    assignments = [
        value
        for statement in body
        if statement.lineno <= call.lineno
        and (pair := _assignment(statement))[1] is not None
        and _name(pair[0], item)
        for value in (pair[1],)
    ]
    if not assignments or not all(
        _member(pop, "pop", "popleft")
        and not pop.args
        and not pop.keywords
        and _base(pop.func.value)
        for pop in assignments
    ):
        return None
    containers = {_base(pop.func.value) for pop in assignments}
    seeded = {
        name: roots[_wrapper(value)]
        for statement in body
        if statement.lineno < call.lineno
        and (pair := _assignment(statement))[1] is not None
        and (name := _base(pair[0])) in containers
        and isinstance(pair[0], ast.Name)
        and (value := pair[1]) is not None
        and _wrapper(value) in roots
    }
    if len(seeded) != 1:
        return None
    # Other popped queues must be local empty worklists filled on expansion.
    for name in containers - seeded.keys():
        starts = [
            value
            for statement in body
            if statement.lineno < call.lineno
            and (pair := _assignment(statement))[1] is not None
            and _name(pair[0], name)
            for value in (pair[1],)
        ]
        if len(starts) != 1 or not _empty_queue(starts[0], seeded.keys()):
            return None
    local_writes = [
        statement
        for statement in body
        if (pair := _assignment(statement))[1] is not None
        and _base(pair[0]) in containers
        and isinstance(pair[0], ast.Name)
    ]
    if len(local_writes) != len(containers):
        return None
    return next(iter(seeded.values())), containers


def _guarded_expansion(current, selected):
    for block in _walk(current.body):
        if not isinstance(block, ast.If) or selected in ast.walk(block.test):
            continue
        for index, statement in enumerate(block.body):
            target, value = _assignment(statement)
            if value is not selected or not isinstance(target, ast.Name):
                continue
            if len(block.body) > index + 1:
                yield block, target.id
    return


def _destination(operation):
    value = operation.func.value
    if _member(value, "setdefault"):
        value = value.func.value
    return _base(value)


def _staged_batches(current, block, grouped_pushes, containers):
    accepted, guards = set(), set()
    in_branch = list(_walk(block.body))
    for stage in {_destination(push) for push in grouped_pushes} - containers - {None}:
        writes = [
            (node, value)
            for node in _walk(current.body)
            if (pair := _assignment(node))[1] is not None and _name(pair[0], stage)
            for value in (pair[1],)
        ]
        if (
            len(writes) != 1
            or writes[0][0] not in in_branch
            or not (isinstance(writes[0][1], ast.Dict) and not writes[0][1].keys)
        ):
            continue
        for group in in_branch:
            if not isinstance(group, ast.For) or not isinstance(group.target, ast.Name):
                continue
            ordered = group.iter
            if not _call(ordered, "sorted") or len(ordered.args) != 1:
                continue
            items = ordered.args[0]
            if not (
                _name(items, stage)
                or _member(items, "keys")
                and _name(items.func.value, stage)
                and not items.args
                and not items.keywords
            ):
                continue
            aliases = [
                (node, target.id)
                for node in group.body
                if (
                    isinstance((pair := _assignment(node))[0], ast.Name)
                    and isinstance(pair[1], ast.Subscript)
                    and _name(pair[1].value, stage)
                    and _name(pair[1].slice, group.target.id)
                )
                for target in (pair[0],)
            ]
            if len(aliases) != 1:
                continue
            assignment, alias = aliases[0]
            if (
                sum(
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Store)
                    and node.id == alias
                    for node in _walk(current.body)
                )
                != 1
            ):
                continue
            group_pushes = {
                node
                for node in _walk(group.body)
                if _member(node, "extend", "extendleft")
                and len(node.args) == 1
                and not node.keywords
                and _name(_wrapper(node.args[0]), alias)
                and _destination(node) in containers
                and node.lineno > assignment.lineno
            }
            if group_pushes:
                accepted.update(group_pushes)
                guards.add(stage)
    return accepted, guards


def _queues_children(current, block, selected, child_list, containers):
    individual, pushes = set(), set()
    for node in _walk(block.body):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and _name(_wrapper(node.iter), child_list)
            and node.lineno > selected.lineno
        ):
            individual.update(
                operation
                for operation in _walk(node.body)
                if _member(operation, "append", "appendleft")
                and len(operation.args) == 1
                and not operation.keywords
                and _name(operation.args[0], node.target.id)
                and _destination(operation)
            )
        if (
            _member(node, "extend", "extendleft")
            and len(node.args) == 1
            and not node.keywords
            and _name(_wrapper(node.args[0]), child_list)
            and _destination(node) in containers
            and node.lineno > selected.lineno
        ):
            pushes.add(node)
    staged, stages = _staged_batches(current, block, individual, containers)
    pushes.update(staged)
    pushes.update(
        node for node in individual if _destination(node) in containers | stages
    )
    if not any(_destination(node) in containers for node in pushes):
        return False
    mapped = containers | stages
    for node in _walk(current.body):
        if _member(
            node, "append", "appendleft", "extend", "extendleft", "insert", "update"
        ):
            if _destination(node) in mapped and node not in pushes:
                return False
        if _member(node, "setdefault") and _base(node.func.value) in mapped:
            if len(node.args) != 2 or node.keywords or not _empty_queue(node.args[1]):
                return False
        target, value = _assignment(node)
        if (
            isinstance(target, ast.Subscript)
            and _base(target) in mapped
            and not _empty_queue(value)
        ):
            return False
    return True


def _empty_queue(value, mapped_roots=()):
    if isinstance(value, ast.DictComp) and len(value.generators) == 1:
        (gen,) = value.generators
        source = gen.iter
        if _member(source, "keys") and not source.args and not source.keywords:
            source = source.func.value
        return (
            isinstance(gen.target, ast.Name)
            and _name(value.key, gen.target.id)
            and not gen.ifs
            and not gen.is_async
            and _base(source) in mapped_roots
            and _empty_queue(value.value)
        )
    return (
        isinstance(value, ast.Dict)
        and not value.keys
        or isinstance(value, ast.List)
        and not value.elts
        or any(_call(value, builtin) for builtin in ("deque", "dict", "list"))
        and not value.args
        and not value.keywords
    )


def _quote(source, node, limit=400):
    text = ast.get_source_segment(source, node) or ""
    return text if len(text.encode()) <= limit else ""


def _source_edit(source, call, guard):
    raw = source.encode()
    rows = raw.splitlines(keepends=True)
    start = sum(map(len, rows[: call.lineno - 1])) + call.col_offset
    end = sum(map(len, rows[: call.end_lineno - 1])) + call.end_col_offset
    return (
        raw[:start] + b"(" + raw[start:end] + guard.encode() + b")" + raw[end:]
    ).decode()


def prepare_round_limit_guard(original, current, allowed, *, owner, operator_fixture):
    """Return at most one snapshot-bound local trial. The caller independently verifies it."""
    path, qualified = owner
    before, after = original.get(path, ""), current.get(path, "")
    if (
        path not in allowed
        or not path.endswith(".py")
        or not operator_fixture
        or not all(
            0 < len(text.encode()) <= _SOURCE_BYTES
            for text in (before, after, operator_fixture)
        )
    ):
        return None
    try:
        old_tree, tree, fixture = map(ast.parse, (before, after, operator_fixture))
        builtins = {"min", "max", "range", "reversed", "list", "dict", "sorted"}
        if not all(_unshadowed(module, builtins) for module in (old_tree, tree)):
            return None
        pinned, working = _method(old_tree, qualified), _method(tree, qualified)
        if pinned is None or working is None:
            return None
        limits = list(_bounded_keywords(pinned, working, fixture))
        if len(limits) != 1:
            return None
        keyword, round_loop = limits[0]
        variants = []
        for call in _factory_calls(working.body):
            expansion = list(_guarded_expansion(working, call))
            if len(expansion) != 1:
                continue
            for class_node in tree.body:
                if not isinstance(class_node, ast.ClassDef):
                    continue
                name, factory = class_node.name, call.func.attr
                derived = _factory(tree, name, factory)
                pinned_factory = _factory(old_tree, name, factory)
                if (
                    derived is None
                    or pinned_factory is None
                    or ast.dump(derived[1]) != ast.dump(pinned_factory[1])
                ):
                    continue
                init, _, (field, param, init_line, advance) = derived
                if derived[2][:2] != pinned_factory[2][:2] or ast.dump(
                    init_line
                ) != ast.dump(pinned_factory[2][2]):
                    continue
                old_roots = _roots(
                    old_tree, qualified, pinned, name, pinned_factory[0], param
                )
                roots = _roots(tree, qualified, working, name, init, param)
                if not old_roots or not roots:
                    continue
                legacy = list(_pins(pinned, round_loop, factory, old_roots))
                queue = _queue_roots(working, call.func.value.id, call, roots)
                if len(legacy) != 1 or queue is None or legacy[0][0] != queue[0][0]:
                    continue
                block, child_list = expansion[0]
                if not _queues_children(working, block, call, child_list, queue[1]):
                    continue
                snippets = (
                    f"pinned round loop: {_quote(before, round_loop.iter)}",
                    f"zero-based constructor: {_quote(after, init_line)}",
                    f"root construction: {_quote(after, queue[0][1])}",
                    f"qualified child factory: {name}.{factory} advances {_quote(after, advance)}",
                    f"current expansion if: {_quote(after, block.test)}",
                )
                guard = f" if {call.func.value.id}.{field} + 1 < {keyword} else []"
                edited = _source_edit(after, call, guard)
                compile(edited, path, "exec")
                variants.append(
                    RoundLimitGuard(
                        {**current, path: edited},
                        keyword,
                        f"{name}.{factory}",
                        "\n".join(snippets),
                    )
                )
        return variants[0] if len(variants) == 1 else None
    except (AttributeError, OverflowError, SyntaxError, ValueError, RecursionError):
        return None

"""Identify an exercised keyword that ceased to bound iteration in a review edit."""

import ast
from textwrap import dedent

from .review_definitions import _symbols

_PUSH = {"append", "appendleft", "extend", "extendleft", "insert", "put"}
_POP = {"pop", "popleft", "get", "popitem"}
_HEAP = {"heappush": "push", "heappop": "pop"}


def _reads(node, name):
    return {
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Name)
        and item.id == name
        and isinstance(item.ctx, ast.Load)
    }


def _statements(function):
    pending = list(function.body)
    while pending:
        item = pending.pop()
        if isinstance(
            item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        yield item
        pending.extend(ast.iter_child_nodes(item))


def _worklist_call(call):
    name = getattr(call.func, "attr", getattr(call.func, "id", ""))
    if name in _HEAP and call.args:
        resource, args, action = call.args[0], call.args[1:], _HEAP[name]
    elif isinstance(call.func, ast.Attribute) and name in _PUSH | _POP:
        resource, args = call.func.value, call.args
        action = "push" if name in _PUSH else "pop"
    else:
        return None
    if key := _binding(resource):
        return key, args, action
    return None


def _binding(node):
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.value.id, node.attr
    return ()


def _numbered(source, first, last, limit=360):
    lines = source.splitlines()
    first, last = max(1, first), min(len(lines), last)
    rows = []
    for number in range(first, min(last, first + 7) + 1):
        row = f"{number} | {lines[number - 1]}\n"
        if len(row.encode()) > limit:
            break
        rows.append(row)
        limit -= len(row.encode())
    return "".join(rows)


def _declare_related(arguments, path, qualified, function, current):
    owner = qualified.rpartition(".")[0]
    receiver = function.args.args[0].arg if owner and function.args.args else None
    definitions = {}
    for filename, source in current.items():
        if not source:
            continue
        try:
            names = list(_symbols(ast.parse(source).body))
        except (SyntaxError, ValueError, RecursionError):
            continue
        for name, node in names:
            if filename == path or "." not in name:
                definitions.setdefault(name, []).append((filename, source, node))
    for argument in arguments:
        for call in ast.walk(argument):
            if not isinstance(call, ast.Call):
                continue
            if isinstance(call.func, ast.Name):
                name = call.func.id
            elif (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == receiver
            ):
                name = f"{owner}.{call.func.attr}"
            else:
                continue
            matches = definitions.get(name, ())
            if len(matches) != 1:
                continue
            filename, source, node = matches[0]
            if isinstance(node, ast.ClassDef):
                row = node.lineno
                declaration = _numbered(
                    source, row, max(row, node.body[0].lineno - 1), limit=120
                )
                fields = [
                    item
                    for item in node.body
                    if isinstance(item, (ast.AnnAssign, ast.Assign))
                ][:4]
                if fields:
                    declaration += _numbered(
                        source, fields[0].lineno, fields[-1].end_lineno, limit=240
                    )
                initializer = next(
                    (
                        item
                        for item in node.body
                        if isinstance(item, ast.FunctionDef) and item.name == "__init__"
                    ),
                    None,
                )
                if initializer and not fields:
                    declaration += _numbered(
                        source, initializer.lineno, initializer.end_lineno, limit=240
                    )
            else:
                declaration = _numbered(
                    source, node.lineno, min(node.end_lineno, node.body[0].lineno + 1)
                )
            if declaration:
                return f"Related construction declaration in {filename} (current):\n{declaration}"
    return ""


def _traversal_windows(path, qualified, function, pinned, current, prior_loop):
    source = current[path]
    items = list(_statements(function))
    assignments = sorted(
        (item for item in items if isinstance(item, (ast.Assign, ast.AnnAssign))),
        key=lambda item: item.lineno,
    )
    bindings = {}
    for item in assignments:
        targets = item.targets if isinstance(item, ast.Assign) else [item.target]
        for target in targets:
            if item.value is not None and (key := _binding(target)):
                bindings.setdefault(key, []).append(item)
    queues = {}
    for call in (item for item in items if isinstance(item, ast.Call)):
        if parsed := _worklist_call(call):
            resource, args, action = parsed
            queues.setdefault(resource, {}).setdefault(action, []).append((call, args))
    active = [
        (resource, operations)
        for resource, operations in queues.items()
        if operations.keys() == {"push", "pop"}
    ]
    if len(active) != 1:
        return ""
    resource, operations = active[0]
    consumer, _ = min(operations["pop"], key=lambda entry: entry[0].lineno)
    producer, arguments = min(
        operations["push"],
        key=lambda entry: (entry[0].lineno < consumer.lineno, entry[0].lineno),
    )
    created = next(
        (item for item in bindings.get(resource, ()) if item.lineno < consumer.lineno),
        None,
    )
    if created is None:
        return ""
    setup = _numbered(source, created.lineno, created.end_lineno)
    seed = next(
        (
            push
            for push, _ in sorted(operations["push"], key=lambda entry: entry[0].lineno)
            if created.end_lineno < push.lineno < consumer.lineno
            and push is not producer
        ),
        None,
    )
    if seed is not None:
        setup += _numbered(source, seed.lineno, seed.end_lineno, limit=150)
    consume = _numbered(source, consumer.lineno - 1, consumer.end_lineno + 1)
    update = _numbered(source, producer.lineno - 1, producer.end_lineno + 1)
    if not all((setup, consume, update)):
        return ""
    values = []
    found_assignment = False
    for argument in arguments[:2]:
        definitions = (
            bindings.get(_binding(argument), ())
            if isinstance(argument, ast.Name)
            else ()
        )
        if len(definitions) == 1 and definitions[0].lineno < producer.lineno:
            assignment = definitions[0]
            values.append(assignment.value)
            if not found_assignment and assignment.end_lineno < producer.lineno - 1:
                update += _numbered(
                    source, assignment.lineno, assignment.end_lineno, limit=200
                )
                found_assignment = True
        values.append(argument)
    declaration = _declare_related(
        (*values, created.value), path, qualified, function, current
    )
    old = _numbered(
        pinned[path], prior_loop.lineno, prior_loop.iter.end_lineno, limit=250
    )
    return (
        f"\nCurrent traversal of {qualified} in {path} (exact source with line numbers):\n"
        f"Worklist initialization:\n{setup}Consumer:\n{consume}Worklist update:\n{update}"
        + declaration
        + (f"Pinned iteration bound (read-only):\n{old}" if old else "")
    )


def lost_iteration_argument(fixture, pinned, current, *, source_windows=False):
    signal = None
    try:
        calls = {
            (
                call.func.attr
                if isinstance(call.func, ast.Attribute)
                else call.func.id,
                kw.arg,
            )
            for call in ast.walk(ast.parse(dedent(fixture)))
            if isinstance(call, ast.Call)
            and isinstance(call.func, (ast.Name, ast.Attribute))
            for kw in call.keywords
            if kw.arg
        }
        for path, source in current.items():
            if not source or source == pinned.get(path) or path not in pinned:
                continue
            old = dict(_symbols(ast.parse(pinned[path]).body))
            for qualified, func in _symbols(ast.parse(source).body):
                prior = old.get(qualified)
                if not isinstance(func, ast.FunctionDef) or not isinstance(
                    prior, ast.FunctionDef
                ):
                    continue
                for param in (*func.args.args, *func.args.kwonlyargs):
                    name = param.arg
                    if (func.name, name) not in calls:
                        continue
                    prior_loop = next(
                        (
                            loop
                            for loop in ast.walk(prior)
                            if isinstance(loop, ast.For) and _reads(loop.iter, name)
                        ),
                        None,
                    )
                    if prior_loop is None:
                        continue
                    reads = _reads(func, name)
                    validation = set().union(
                        *(
                            _reads(item.test, name)
                            for item in func.body
                            if isinstance(item, ast.If)
                            and len(item.body) == 1
                            and isinstance(item.body[0], ast.Raise)
                        )
                    )
                    if reads <= validation:
                        note = (
                            f"The selected test calls {qualified} with the keyword {name}. On the pinned "
                            f"version, {name} controlled a for loop. The current body reads it only in "
                            "input validation or omits it entirely. Check how this public limit affects "
                            "the new traversal and the observed calls in the fixture."
                        )
                        if not source_windows:
                            return note
                        if signal is not None:
                            return signal[0]
                        signal = note, path, qualified, func, prior_loop
        if signal is not None:
            note, path, qualified, function, prior_loop = signal
            return note + _traversal_windows(
                path, qualified, function, pinned, current, prior_loop
            )
    except (SyntaxError, ValueError, RecursionError):
        return signal[0] if signal is not None else ""
    return ""

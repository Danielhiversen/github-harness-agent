"""Identify an exercised keyword that ceased to bound iteration in a review edit."""

import ast
from dataclasses import dataclass
from textwrap import dedent

from .review_definitions import _symbols

_PUSH = {"append", "appendleft", "extend", "extendleft", "insert", "put"}
_POP = {"pop", "popleft", "get", "popitem"}
_HEAP = {"heappush": "push", "heappop": "pop"}
_SUBORDINATE_BYTES = 2 * 1024
_SUBORDINATE_ROWS = 24


@dataclass(frozen=True)
class IterationWindows:
    symbol: tuple[str, str]
    spans: tuple[tuple[int, int], ...]
    note: str
    related: str


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
    if isinstance(node, ast.Subscript):
        resource, key = node.value, node.slice
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"setdefault", "get"}
        and node.args
        and not node.keywords
    ):
        resource, key = node.func.value, node.args[0]
    else:
        return ()
    if isinstance(resource, (ast.Name, ast.Attribute)):
        if base := _binding(resource):
            index = ast.dump(key) if isinstance(key, (ast.Name, ast.Constant)) else "?"
            return (*base, f"[{index}]")
    return ()


def _numbered(
    source,
    first,
    last,
    limit=360,
    *,
    path=None,
    current_ranges=None,
    max_rows=8,
    complete=False,
):
    lines = source.splitlines()
    first, last = max(1, first), min(len(lines), last)
    rows = []
    for number in range(first, min(last, first + max_rows - 1) + 1):
        row = f"{number} | {lines[number - 1]}\n"
        if len(row.encode()) > limit:
            break
        rows.append(row)
        limit -= len(row.encode())
    if complete and len(rows) != last - first + 1:
        return ""
    if rows and path is not None and current_ranges is not None:
        current_ranges.append((path, first, first + len(rows) - 1))
    return "".join(rows)


def _declare_related(
    arguments,
    path,
    qualified,
    function,
    current,
    *,
    current_ranges=None,
    include_result=False,
):
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
            window = {"path": filename, "current_ranges": current_ranges}
            state = ""
            if isinstance(node, ast.ClassDef):
                row = node.lineno
                declaration = _numbered(
                    source, row, max(row, node.body[0].lineno - 1), limit=120, **window
                )
                fields = [
                    item
                    for item in node.body
                    if isinstance(item, (ast.AnnAssign, ast.Assign))
                ][:4]
                if fields:
                    declaration += _numbered(
                        source,
                        fields[0].lineno,
                        fields[-1].end_lineno,
                        limit=240,
                        **window,
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
                        source,
                        initializer.lineno,
                        initializer.end_lineno,
                        limit=240,
                        **window,
                    )
            else:
                declaration = _numbered(
                    source,
                    node.lineno,
                    min(node.end_lineno, node.body[0].lineno + 1),
                    **window,
                )
                if include_result:
                    result = next(
                        (
                            item
                            for item in sorted(
                                _statements(node),
                                key=lambda item: getattr(item, "lineno", 0),
                            )
                            if isinstance(item, (ast.Return, ast.Yield, ast.YieldFrom))
                            and item.value is not None
                        ),
                        None,
                    )
                    if result is not None and result.lineno > node.body[0].lineno + 1:
                        declaration += _numbered(
                            source,
                            result.lineno,
                            result.end_lineno,
                            limit=180,
                            **window,
                        )
                    if result is not None:
                        types = {
                            item.func.id
                            for item in ast.walk(result.value)
                            if isinstance(item, ast.Call)
                            and isinstance(item.func, ast.Name)
                            and len(definitions.get(item.func.id, ())) == 1
                            and isinstance(
                                definitions[item.func.id][0][2], ast.ClassDef
                            )
                        }
                        if len(types) == 1:
                            state_file, state_source, state_type = definitions[
                                types.pop()
                            ][0]
                            rows = _numbered(
                                state_source,
                                state_type.lineno,
                                state_type.end_lineno,
                                limit=230,
                                path=state_file,
                                current_ranges=current_ranges,
                            )
                            if rows:
                                state = f"Candidate returned state declaration in {state_file} (current):\n{rows}"
            if declaration:
                return f"Related construction declaration in {filename} (current):\n{declaration}{state}"
    return ""


def _producer_sources(arguments, bindings, before):
    values = list(arguments[:2])
    assignments = []
    visited = set()
    for value in values:
        names = sorted(
            (
                node
                for node in ast.walk(value)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            ),
            key=lambda node: (node.lineno, node.col_offset),
        )
        for name in names:
            if name.id in visited:
                continue
            visited.add(name.id)
            definitions = bindings.get((name.id,), ())
            if len(definitions) == 1 and definitions[0].lineno < before:
                assignments.append(definitions[0])
                values.append(definitions[0].value)
                if len(assignments) == 3:
                    return values, assignments
    return values, assignments


def _assigned_name(statement):
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
    elif isinstance(statement, ast.AnnAssign):
        target = statement.target
    else:
        return None
    return target.id if isinstance(target, ast.Name) else None


def _consumed_name(consumer, statements):
    return next(
        (
            name
            for item in statements
            if getattr(item, "value", None) is consumer
            and (name := _assigned_name(item)) is not None
        ),
        None,
    )


def _requeues_consumed_token(consumer, producer_args, assignments):
    token = _consumed_name(consumer, assignments)
    return (
        bool(token)
        and len(producer_args) == 1
        and (isinstance(producer_args[0], ast.Name) and producer_args[0].id == token)
    )


def _mapping(key):
    return key[:-1] if key and key[-1].startswith("[") else ()


def _iterating_result(name, statements, after, end):
    if name is None:
        return None
    return min(
        (
            item
            for item in statements
            if isinstance(item, ast.For)
            and after < item.lineno <= end
            and isinstance(item.iter, ast.Name)
            and item.iter.id == name
        ),
        key=lambda item: item.lineno,
        default=None,
    )


def _consumed_expansion(pop, statements, end):
    name = _consumed_name(pop, statements)
    if name is None:
        return None
    candidates = []
    for item in statements:
        if (
            not isinstance(item, (ast.Assign, ast.AnnAssign, ast.For))
            or not pop.lineno < item.lineno <= end
        ):
            continue
        value = item.iter if isinstance(item, ast.For) else item.value
        if value is None:
            continue
        for call in (node for node in ast.walk(value) if isinstance(node, ast.Call)):
            receiver = call.func.value if isinstance(call.func, ast.Attribute) else None
            direct = isinstance(receiver, ast.Name) and receiver.id == name
            if direct or any(
                isinstance(arg, ast.Name) and arg.id == name for arg in call.args
            ):
                child_loop = _iterating_result(
                    _assigned_name(item), statements, item.lineno, end
                )
                iterated = isinstance(item, ast.For) or child_loop is not None
                candidates.append(
                    (not iterated, not direct, item.lineno, item, value, child_loop)
                )
                break
    return min(candidates, key=lambda entry: entry[:3])[3:] if candidates else None


def _subordinate_windows(
    queues, resource, consumer, producer, items, bindings, current_source
):
    enclosing = [
        item
        for item in items
        if isinstance(item, (ast.For, ast.While))
        and item.lineno < consumer.lineno <= producer.lineno <= item.end_lineno
    ]
    end = min((item.end_lineno for item in enclosing), default=float("inf"))
    others = [
        (key, action, call, arguments)
        for key, group in queues.items()
        if key != resource
        for action, calls in group.items()
        for call, arguments in calls
        if consumer.lineno <= call.lineno <= end
    ]
    pops = sorted(
        (entry for entry in others if entry[1] == "pop"),
        key=lambda entry: entry[2].lineno,
    )[:2]
    keys = {entry[0] for entry in pops}
    mappings = {_mapping(key) for key in keys} - {()}
    first_pop = min((entry[2].lineno for entry in pops), default=consumer.lineno)
    pushes = [
        entry for entry in others if entry[1] == "push" and entry[2].lineno >= first_pop
    ]
    matching = [
        entry for entry in pushes if entry[0] in keys or _mapping(entry[0]) in mappings
    ]
    pushes = sorted(
        matching or pushes, key=lambda entry: (entry[0] not in keys, entry[2].lineno)
    )[:2]
    sections, values = [], []
    remaining_bytes, remaining_rows = _SUBORDINATE_BYTES, _SUBORDINATE_ROWS

    def append(label, first, last, cap=None):
        nonlocal remaining_bytes, remaining_rows
        heading = f"{label}:\n"
        room = remaining_bytes - len(heading.encode())
        if room <= 0 or remaining_rows < last - first + 1:
            return False
        rows = current_source(
            first,
            last,
            limit=min(room, cap or room),
            max_rows=remaining_rows,
            complete=True,
        )
        if not rows:
            return False
        sections.append(heading + rows)
        remaining_bytes -= len((heading + rows).encode())
        remaining_rows -= last - first + 1
        return True

    for _, action, call, arguments in (*pops, *pushes):
        label = (
            "Candidate item consumption"
            if action == "pop"
            else "Candidate expansion or scheduling"
        )
        if not append(
            label, call.lineno, call.end_lineno, cap=600 if action == "push" else 300
        ):
            continue
        if action == "push":
            traced, producers = _producer_sources(arguments, bindings, call.lineno)
            values.extend(traced)
            for assignment in producers[:2]:
                if assignment.end_lineno < call.lineno:
                    append(
                        "Candidate input to that update",
                        assignment.lineno,
                        assignment.end_lineno,
                        cap=130,
                    )
    for _, _, popped, _ in pops[:1]:
        if expansion := _consumed_expansion(popped, items, end):
            statement, value, child_loop = expansion
            values.insert(0, value)
            stop = (
                value.end_lineno
                if isinstance(statement, ast.For)
                else statement.end_lineno
            )
            branch = min(
                (
                    item
                    for item in items
                    if isinstance(item, ast.If)
                    and popped.lineno
                    < item.lineno
                    < statement.lineno
                    <= item.end_lineno
                    <= end
                ),
                key=lambda item: item.end_lineno - item.lineno,
                default=None,
            )
            if branch is None or not append(
                "Candidate branch expanding the consumed item",
                branch.lineno,
                branch.end_lineno,
                cap=520,
            ):
                append(
                    "Candidate expansion of the consumed item",
                    statement.lineno,
                    stop,
                    cap=420,
                )
                if branch is not None:
                    append(
                        "Candidate expansion guard",
                        branch.lineno,
                        branch.test.end_lineno,
                        cap=180,
                    )
            if child_loop is not None:
                append(
                    "Candidate loop over produced items",
                    child_loop.lineno,
                    child_loop.iter.end_lineno,
                    cap=250,
                )
    return "".join(sections), values


def _traversal_windows(
    path, qualified, function, pinned, current, prior_loop, current_ranges
):
    source = current[path]
    items = list(_statements(function))
    emitted = []

    def current_source(first, last, limit=360, **options):
        return _numbered(
            source, first, last, limit, path=path, current_ranges=emitted, **options
        )

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
            candidates = bindings.get(resource, ())
            if len(candidates) == 1 and candidates[0].lineno < call.lineno:
                alias = _binding(candidates[0].value)
                if alias and "[" in alias[-1]:
                    resource = alias
            queues.setdefault(resource, {}).setdefault(action, []).append((call, args))
    active = [
        (resource, operations)
        for resource, operations in queues.items()
        if operations.keys() == {"push", "pop"} and resource in bindings
    ]
    if not active:
        return "", ""
    resource, operations = min(
        active, key=lambda group: min(entry[0].lineno for entry in group[1]["pop"])
    )
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
        return "", ""
    multiple = any(
        key != resource and "pop" in group for key, group in queues.items()
    ) or _requeues_consumed_token(consumer, arguments, assignments)
    limit = 210 if multiple else 360
    setup = current_source(created.lineno, created.end_lineno, limit=limit)
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
        setup += current_source(seed.lineno, seed.end_lineno, limit=150)
    consume = current_source(consumer.lineno - 1, consumer.end_lineno + 1, limit=limit)
    update = current_source(producer.lineno - 1, producer.end_lineno + 1, limit=limit)
    if not all((setup, consume, update)):
        return "", ""
    values, expansions = _producer_sources(arguments, bindings, producer.lineno)
    if expansions and expansions[0].end_lineno < producer.lineno - 1 and not multiple:
        assignment = expansions[0]
        update += current_source(assignment.lineno, assignment.end_lineno, limit=200)
    supplementary = ""
    primary_count = len(emitted)
    if multiple:
        supplementary, child_values = _subordinate_windows(
            queues, resource, consumer, producer, items, bindings, current_source
        )
        values = [*child_values, *values]
    declaration = _declare_related(
        (*values, created.value),
        path,
        qualified,
        function,
        current,
        current_ranges=emitted,
        include_result=multiple,
    )
    old = _numbered(
        pinned[path], prior_loop.lineno, prior_loop.iter.end_lineno, limit=250
    )
    if current_ranges is not None:
        current_ranges.extend((*emitted[primary_count:], *emitted[:primary_count]))
    text = (
        f"\nCurrent traversal of {qualified} in {path} (exact source with line numbers):\n"
        + (f"Pinned iteration bound (read-only):\n{old}" if old else "")
        + (
            "Candidate sites across worklists (source evidence, without a proven link):\n"
            if multiple
            else ""
        )
        + supplementary
        + declaration
        + f"Worklist initialization:\n{setup}Consumer:\n{consume}Worklist update:\n{update}"
    )
    return text, declaration


def _iteration_loss(fixture, pinned, current, *, unique):
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
                        if not unique:
                            return note, None
                        if signal is not None:
                            return signal[0], None
                        signal = note, path, qualified, func, prior_loop
        if signal is not None:
            return signal[0], signal[1:]
    except (SyntaxError, ValueError, RecursionError):
        pass
    return (signal[0], None) if signal is not None else ("", None)


def lost_iteration_argument(
    fixture, pinned, current, *, source_windows=False, current_ranges=None
):
    note, signal = _iteration_loss(fixture, pinned, current, unique=source_windows)
    if signal is None:
        return note
    path, qualified, function, prior_loop = signal
    try:
        windows, _ = _traversal_windows(
            path, qualified, function, pinned, current, prior_loop, current_ranges
        )
    except (SyntaxError, ValueError, RecursionError):
        return note
    return note + windows


def iteration_edit_windows(fixture, pinned, current):
    note, signal = _iteration_loss(fixture, pinned, current, unique=True)
    if signal is None:
        return None
    path, qualified, function, prior_loop = signal
    spans = []
    try:
        _, related = _traversal_windows(
            path, qualified, function, pinned, current, prior_loop, spans
        )
    except (SyntaxError, ValueError, RecursionError):
        return None
    writable = tuple(
        (first, last) for filename, first, last in spans if filename == path
    )
    return (
        IterationWindows((path, qualified), writable, note, related)
        if writable
        else None
    )

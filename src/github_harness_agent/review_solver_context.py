"""Select compact Python source excerpts for a single review concern."""

import ast
import re
import tarfile
from io import BytesIO
from pathlib import PurePosixPath

from .check_diagnostics import check_locations
from .review_assertions import _STRAIGHT_LINE

_FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)
_IMPORT = (ast.Import, ast.ImportFrom)
_ASSIGNMENT = (ast.Assign, ast.AnnAssign)
_LOOP = (ast.For, ast.AsyncFor, ast.While)
_CONTROL = (*_LOOP, ast.If, ast.With, ast.AsyncWith, ast.Try, ast.TryStar, ast.Match)
_STOCK_NODE = re.compile(
    r"(?P<path>(?:[\w-][\w.-]*/)*[\w-][\w.-]*\.py)"
    r"(?P<name>(?:::[A-Za-z_]\w*)*::test\w*)"
)
_INTEGER_COMPARISON = re.compile(r"(-?[1-9][0-9]{0,17}|0) +== +([1-9][0-9]{0,17}|0)")
_LOOKUP_PROGRESS = (
    "The tracked pytest comparison now evaluates at the previous lookup failure."
)


def _diagnostic_window(lines, imports, row, limit, *, bounds=None):
    start, stop = max(1, row - 4), min(len(lines), row + 4)
    for index, (first, last) in enumerate(imports):
        if not first <= row <= last:
            continue
        # Import sorting can cover several statements, including multiline imports.
        for prev_first, prev_last in reversed(imports[:index]):
            if any(
                text.strip() and not text.lstrip().startswith("#")
                for text in lines[prev_last : first - 1]
            ):
                break
            first = prev_first
        for next_first, next_last in imports[index + 1 :]:
            if any(
                text.strip() and not text.lstrip().startswith("#")
                for text in lines[last : next_first - 1]
            ):
                break
            last = next_last
        start = min(start, first)
        stop = max(stop, last)
        break
    start, stop = max(start, row - 40), min(stop, row + 40)
    if bounds is not None:
        start, stop = max(start, bounds[0]), min(stop, bounds[1])
    while start <= row <= stop:
        text = f"# Lines {start}-{stop} (diagnostic at line {row})\n" + "".join(
            lines[start - 1 : stop]
        )
        if len(text) <= limit:
            return text, start, stop
        if row - start >= stop - row:
            start += 1
        else:
            stop -= 1
    return "", row, row


def focused_check_excerpt(command, diagnostic, sources, *, max_chars=2800):
    locations = []
    focus_path = None
    for raw_path, row in check_locations(command, diagnostic):
        path = PurePosixPath(raw_path)
        if path.is_absolute():
            if not path.is_relative_to("/work"):
                continue
            path = path.relative_to("/work")
        name = str(path)
        if not sources.get(name) or (focus_path and name != focus_path):
            continue
        focus_path = name
        if row not in locations:
            locations.append(row)
        if len(locations) == 3:
            break
    if not focus_path:
        return None
    source = sources[focus_path]
    lines = source.splitlines(keepends=True)
    locations = [row for row in locations if row <= len(lines)]
    try:
        imports = sorted(
            (node.lineno, node.end_lineno)
            for node in ast.walk(ast.parse(source))
            if isinstance(node, _IMPORT)
        )
    except (SyntaxError, ValueError, RecursionError):
        imports = []
    chunks = []
    stop = start = 0
    for row in locations:
        if start <= row <= stop:
            continue
        chunk, start, stop = _diagnostic_window(
            lines, imports, row, max_chars - sum(map(len, chunks))
        )
        if chunk:
            chunks.append(chunk)
    return (focus_path, "".join(chunks)) if chunks else None


def _tracked_python(snapshot, path):
    try:
        member = snapshot.getmember(path)
        if (
            member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
            or member.size > 128 * 1024
        ):
            return None
        with snapshot.extractfile(member) as source:
            text = source.read().decode()
        return text.splitlines(keepends=True), ast.parse(text).body
    except (KeyError, UnicodeDecodeError, SyntaxError, ValueError, RecursionError):
        return None


def _pytest_fixtures(body):
    for node in body:
        if not isinstance(node, _FUNCTION):
            continue
        for decorator in node.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            symbol = decorator.func if call else decorator
            if not (
                isinstance(symbol, ast.Name) and symbol.id == "fixture"
                or isinstance(symbol, ast.Attribute) and symbol.attr == "fixture"
            ):
                continue
            options = {item.arg: item.value for item in call.keywords} if call else {}
            name = options.get("name")
            name = (
                name.value
                if isinstance(name, ast.Constant) and isinstance(name.value, str)
                else node.name
            )
            autouse = options.get("autouse")
            yield name, node, isinstance(autouse, ast.Constant) and autouse.value is True
            break


def _fixture_arguments(node):
    arguments = node.args
    dependencies = {
        arg.arg for arg in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
    }
    for marker in node.decorator_list:
        if isinstance(marker, ast.Call) and getattr(marker.func, "attr", "") == "usefixtures":
            dependencies.update(
                arg.value
                for arg in marker.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            )
    return dependencies


def _dotted_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return ".".join((node.id, *reversed(parts))) if isinstance(node, ast.Name) else ""


def _reviewed_imports(body, modules):
    bindings = {}
    reviewed_calls = set()
    for node in body:
        if not isinstance(node, _IMPORT) or getattr(node, "level", 0):
            continue
        for alias in node.names:
            if isinstance(node, ast.Import):
                local = alias.asname or alias.name
                qualified = alias.name
                bindings[local.split(".", 1)[0]] = node
            else:
                local = alias.asname or alias.name
                qualified = f"{node.module}.{alias.name}"
                bindings[local] = node
                if alias.name in modules.get(node.module, ()):
                    reviewed_calls.add(local)
            reviewed_calls.update(f"{local}.{name}" for name in modules.get(qualified, ()))
    return bindings, reviewed_calls


def _source_root(path):
    parts = path.parts
    if "src" not in parts[:-1]:
        return PurePosixPath()
    return PurePosixPath(*parts[: len(parts) - parts[::-1].index("src")])


def repository_pytest_example(archive, sources, *, max_bytes=850):
    """Find one complete example that calls a public API of a reviewed module."""
    modules = {}
    for path, source in sources.items():
        source_path = PurePosixPath(path)
        parts = source_path.relative_to(_source_root(source_path)).with_suffix("").parts
        if parts[-1:] == ("__init__",):
            parts = parts[:-1]
        if not parts or not all(part.isidentifier() for part in parts):
            continue
        try:
            declarations = ast.parse(source).body
        except (SyntaxError, ValueError, RecursionError):
            continue
        modules[".".join(parts)] = {
            node.name
            for node in declarations
            if isinstance(node, (*_FUNCTION, ast.ClassDef)) and not node.name.startswith("_")
        }
    if not any(modules.values()):
        return ""
    module_stems = {module.rsplit(".", 1)[-1] for module in modules}

    try:
        with tarfile.open(fileobj=BytesIO(archive), mode="r:") as snapshot:
            candidates = []
            for member in snapshot:
                path = PurePosixPath(member.name)
                if (
                    member.type in (tarfile.REGTYPE, tarfile.AREGTYPE)
                    and member.size <= 128 * 1024
                    and _STOCK_NODE.fullmatch(member.name + "::test_example")
                    and (
                        path.name == "conftest.py"
                        or path.name.startswith("test_")
                        or path.name.endswith("_test.py")
                    )
                ):
                    stem = path.stem.removeprefix("test_").removesuffix("_test")
                    priority = stem in module_stems
                    candidates.append((not priority, member.name))
            for _, path in sorted(candidates)[:64]:
                tracked = _tracked_python(snapshot, path)
                if tracked is None:
                    continue
                lines, body = tracked
                imports, reviewed_calls = _reviewed_imports(body, modules)
                if not reviewed_calls:
                    continue
                fixtures = {node for _, node, _ in _pytest_fixtures(body)}
                for node in body:
                    if not isinstance(node, _FUNCTION) or not (
                        node.name.startswith("test_") or node in fixtures
                    ):
                        continue
                    if not any(
                        _dotted_name(call.func) in reviewed_calls
                        for call in ast.walk(node) if isinstance(call, ast.Call)
                    ):
                        continue
                    selected = {imports[name] for name in _references([node]) & imports.keys()}
                    excerpt = f"# {path}\n" + "".join(
                        "".join(lines[_start(item) - 1 : item.end_lineno])
                        for item in sorted(selected, key=lambda item: item.lineno) + [node]
                    )
                    if len(excerpt.encode()) <= max_bytes:
                        return excerpt.rstrip()
    except (tarfile.TarError, OSError, EOFError):
        pass
    return ""


def imported_model_declarations(archive, sources, *, max_bytes=850):
    """Describe imported classes from the pinned checkout without importing code."""
    imports = {}
    for path, source in sources.items():
        path = PurePosixPath(path)
        try:
            body = ast.parse(source).body
        except (SyntaxError, ValueError, RecursionError):
            continue
        for node in body:
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.level > len(path.parents):
                continue
            roots = (path.parents[node.level - 1],) if node.level else (
                _source_root(path), PurePosixPath()
            )
            module_parts = node.module.split(".")
            for root in roots:
                module = root.joinpath(*module_parts)
                imports.setdefault(str(module.with_suffix(".py")), set()).update(
                    alias.name for alias in node.names if alias.name[:1].isupper()
                )
    chunks = []
    try:
        with tarfile.open(fileobj=BytesIO(archive), mode="r:") as snapshot:
            relevant = (path for path, names in imports.items() if names)
            for path in tuple(relevant)[:24]:
                tracked = _tracked_python(snapshot, path)
                if not tracked:
                    continue
                _, body = tracked
                for node in body:
                    if not isinstance(node, ast.ClassDef) or node.name not in imports[path]:
                        continue
                    fields = [
                        f"    {ast.unparse(item)}"
                        for item in node.body
                        if isinstance(item, ast.AnnAssign)
                    ]
                    fields += [
                        f"    def __init__({ast.unparse(item.args)})"
                        for item in node.body
                        if isinstance(item, ast.FunctionDef) and item.name == "__init__"
                    ]
                    if fields:
                        chunk = "\n".join((f"# {path}", f"class {node.name}:", *fields, ""))
                        if len(chunk.encode()) <= max_bytes:
                            chunks.append(chunk)
                            max_bytes -= len(chunk.encode())
    except (tarfile.TarError, OSError, EOFError, RecursionError):
        pass
    return "".join(chunks).rstrip()


def stock_pytest_excerpt(archive, command, node, diagnostic, *, failure_window_only=False):
    """Read a pytest node and local fixture declarations from the pinned Git archive."""
    target = node.split("[", 1)[0].removeprefix("/work/").removeprefix("./")
    match = _STOCK_NODE.fullmatch(target) if len(target) <= 512 else None
    if match is None:
        return ""
    path = match["path"]
    *owners, name = match["name"].split("::")[1:]
    try:
        with tarfile.open(fileobj=BytesIO(archive), mode="r:") as snapshot:
            module = _tracked_python(snapshot, path)
            if module is None:
                return ""
            lines, body = module
            scope = body
            for owner in owners:
                declaration = next(
                    (item for item in scope if isinstance(item, ast.ClassDef) and item.name == owner),
                    None,
                )
                if declaration is None:
                    return ""
                scope = declaration.body
            test = next(
                (item for item in scope if isinstance(item, _FUNCTION) and item.name == name),
                None,
            )
            if test is None:
                return ""
            scopes = [(path, lines, scope)]
            if scope is not body:
                scopes.append((path, lines, body))
            parents = list(PurePosixPath(path).parents)
            for parent in dict.fromkeys((*parents[:3], *parents[-1:])):
                config_path = str(parent / "conftest.py")
                if config_path != path and (config := _tracked_python(snapshot, config_path)):
                    scopes.append((config_path, *config))
    except (tarfile.TarError, OSError, EOFError):
        return ""

    def definition(label, source, item, cap):
        start = _start(item)
        text = "".join(source[start - 1 : item.end_lineno])
        if len(text) > cap:
            text = text[:cap] + "\n# Definition excerpt truncated\n"
        return f"# {label}:{start}\n{text}\n"

    chunks = ["# Read-only repository test context\n"]
    row = next(
        (
            row for reported, row in check_locations(command, diagnostic)
            if reported.removeprefix("/work/").removeprefix("./") == path
            and test.lineno <= row <= test.end_lineno
        ),
        None,
    )
    if row is not None:
        window, _, _ = _diagnostic_window(
            lines, (), row, 650 if failure_window_only else 1000,
            bounds=(test.lineno, test.end_lineno),
        )
        if window:
            chunk = f"# {path}\n{window}\n"
            if failure_window_only:
                return chunk
            chunks.append(chunk)
    if failure_window_only:
        return ""
    chunks.append(definition(path, lines, test, 2700))
    visible = {}
    wanted = _fixture_arguments(test)
    for label, source, scope in scopes:
        for fixture, item, autouse in _pytest_fixtures(scope):
            visible.setdefault(fixture, (label, source, item))
            if autouse:
                wanted.add(fixture)
    shown = set()
    # Follow a bounded chain of dependencies across the visible fixture scopes.
    for _ in range(4):
        selected = (wanted - shown) & visible.keys()
        if not selected:
            break
        shown.update(selected)
        for fixture in sorted(selected):
            label, source, item = visible[fixture]
            wanted.update(_fixture_arguments(item))
            chunks.append(definition(label, source, item, 850))
            if sum(map(len, chunks)) >= 6000:
                return "".join(chunks)[:6000]
    return "".join(chunks)


def _words(text):
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return set(re.findall(r"[A-Za-z][A-Za-z0-9]*", separated.lower()))


def _stock_frame(match, detail):
    prefix = rf"^(?:/work/|\./)?{re.escape(match['path'])}:([1-9][0-9]{{0,6}}): "
    kind = r"(assert|[A-Za-z]{1,48}(?:Error|Exception))\b"
    frame = re.match(prefix + kind, detail)
    if not frame:
        # pytest --tb=short separates the frame from the exception. Require
        # the selected function and an adjacent error, never a helper's frame.
        function = re.escape(match["name"].rsplit("::", 1)[-1])
        frame = re.match(prefix + rf"in {function}\nE[ \t]+" + kind, detail)
    if frame:
        return int(frame[1]), "AssertionError" if frame[2] == "assert" else frame[2]
    return None


def _stock_node_match(node):
    selected = node.split("[", 1)[0].removeprefix("/work/").removeprefix("./")
    return _STOCK_NODE.fullmatch(selected) if len(selected) <= 512 else None


def stock_failure_site(node, detail):
    """Describe a traceback in the selected test without logging assertion data."""
    match = _stock_node_match(node)
    frame = _stock_frame(match, detail) if match else None
    return f"line {frame[0]} ({frame[1]})" if frame else ""


def stock_assertion_failure_line(node, detail):
    """Return the traceback line only when the exact selected pytest assertion failed."""
    match = _stock_node_match(node)
    frame = _stock_frame(match, detail) if match else None
    return frame[0] if frame and frame[1] == "AssertionError" else None


def _tracked_stock_test(archive, match):
    try:
        with tarfile.open(fileobj=BytesIO(archive), mode="r:") as snapshot:
            tracked = _tracked_python(snapshot, match["path"])
    except (tarfile.TarError, OSError, EOFError):
        return None
    if tracked is None:
        return None
    scope = tracked[1]
    *owners, name = match["name"].split("::")[1:]
    for owner in owners:
        classes = [
            item
            for item in scope
            if isinstance(item, ast.ClassDef) and item.name == owner
        ]
        if len(classes) != 1:
            return None
        scope = classes[0].body
    tests = [
        item for item in scope if isinstance(item, _FUNCTION) and item.name == name
    ]
    return tests[0] if len(tests) == 1 else None


def _rewritten_assertion(detail):
    """Read one pytest-rewritten expression, from line or structured traceback output."""
    lines = detail.splitlines()
    expressions = set(re.findall(r"^E[ \t]+assert[ \t]+([^\n]+)$", detail, re.M))
    if lines and (first := re.match(r"^.+\.py:[0-9]+: assert[ \t]+(.+)$", lines[0])):
        expressions.add(first[1])
    return expressions.pop() if len(expressions) == 1 else None


def stock_integer_assertion_failure_line(archive, node, detail):
    """Identify a failing integer equality in a pinned, selected pytest test."""
    match = _stock_node_match(node)
    frame = _stock_frame(match, detail) if match else None
    expression = _INTEGER_COMPARISON.fullmatch(_rewritten_assertion(detail) or "")
    if (
        frame is None
        or frame[1] != "AssertionError"
        or expression is None
        or expression[1] == expression[2]
        or (test := _tracked_stock_test(archive, match)) is None
    ):
        return None
    for item in ast.walk(test):
        if (
            isinstance(item, ast.Assert)
            and item.lineno == frame[0]
            and isinstance(item.test, ast.Compare)
            and len(item.test.ops) == 1
            and isinstance(item.test.ops[0], ast.Eq)
        ):
            return frame[0]
    return None


def _straight_line_suites(body):
    yield body
    for statement in body:
        if isinstance(statement, ast.If):
            yield from _straight_line_suites(statement.body)
            yield from _straight_line_suites(statement.orelse)
        elif (
            isinstance(statement, ast.For)
            and isinstance(statement.iter, (ast.Tuple, ast.List))
            and len(statement.iter.elts) == 1
            and not isinstance(statement.iter.elts[0], ast.Starred)
        ):
            # Both failures must come from the same iteration. Dynamic and
            # multi-item loops can hide the old failure behind an earlier iteration.
            yield from _straight_line_suites(statement.body)


def _integer_approach(assertion, before, after):
    comparison = assertion.test
    if not (
        isinstance(comparison, ast.Compare)
        and len(comparison.ops) == 1
        and isinstance(comparison.ops[0], ast.Eq)
        and not isinstance(comparison.left, ast.Constant)
        and isinstance(expected := comparison.comparators[0], ast.Constant)
        and type(expected.value) is int
        and expected.value >= 0
    ):
        return False
    old, new = (
        _INTEGER_COMPARISON.fullmatch(before),
        _INTEGER_COMPARISON.fullmatch(after),
    )
    if (
        old is None
        or new is None
        or int(old[2]) != expected.value
        or int(new[2]) != expected.value
    ):
        return False
    low, high = sorted((int(old[1]), expected.value))
    return low < int(new[1]) < high


def stock_assertion_progress(archive, node, before, after, *, changed_paths=()):
    """Report bounded progress on an unchanged, archive-backed pytest assertion."""
    match = _stock_node_match(node)
    if match is None or match["path"] in changed_paths:
        return ""
    old, new = _stock_frame(match, before), _stock_frame(match, after)
    if old is None or new is None or new[1] != "AssertionError":
        return ""
    lookup = old[1] in {"KeyError", "IndexError", "AttributeError"} and old[0] == new[0]
    old_expression, new_expression = (
        _rewritten_assertion(before),
        _rewritten_assertion(after),
    )
    if not new_expression or (
        not lookup
        and (old[1] != "AssertionError" or not old_expression or old[0] > new[0])
    ):
        return ""
    test = _tracked_stock_test(archive, match)
    if test is None:
        return ""
    if lookup:
        if any(
            isinstance(item, ast.Assert)
            and item.lineno == old[0]
            and isinstance(item.test, ast.Compare)
            and any(
                isinstance(expr, (ast.Subscript, ast.Attribute))
                for expr in ast.walk(item.test)
            )
            for item in ast.walk(test)
        ):
            return _LOOKUP_PROGRESS
        return ""

    sites = {}
    for suite in _straight_line_suites(test.body):
        for index, statement in enumerate(suite):
            if isinstance(statement, ast.Assert) and statement.lineno in (
                old[0],
                new[0],
            ):
                sites.setdefault(statement.lineno, []).append((suite, index))
    previous, current = sites.get(old[0], []), sites.get(new[0], [])
    if len(previous) != 1 or len(current) != 1:
        return ""
    old_suite, start = previous[0]
    new_suite, stop = current[0]
    if old[0] == new[0] and _integer_approach(
        old_suite[start], old_expression, new_expression
    ):
        return (
            "The tracked pytest integer comparison moved closer to its expected value."
        )
    if (
        old_suite is new_suite
        and start < stop
        and all(
            isinstance(statement, _STRAIGHT_LINE)
            for statement in old_suite[start + 1 : stop]
        )
    ):
        return "The tracked pytest case reached a later assertion in the same straight-line suite."
    return ""


def stock_assertion_evaluated(archive, node, before, after):
    """A tracked comparison now evaluates both sides instead of raising during lookup."""
    return stock_assertion_progress(archive, node, before, after) == _LOOKUP_PROGRESS


def stock_missing_count_after_reorder(archive, node, before, after, expected_line):
    """Describe the observed loss of a literal mapping entry at one pinned count assertion."""
    selected = _stock_node_match(node)
    if selected is None or (
        _stock_frame(selected, before) != (expected_line, "AssertionError")
        or _stock_frame(selected, after) != (expected_line, "KeyError")
    ):
        return ""
    comparison = _INTEGER_COMPARISON.fullmatch(_rewritten_assertion(before) or "")
    test = _tracked_stock_test(archive, selected)
    if comparison is None or test is None:
        return ""
    assertions = [
        node.test
        for node in ast.walk(test)
        if isinstance(node, ast.Assert) and node.lineno == expected_line
    ]
    if len(assertions) != 1:
        return ""
    assertion = assertions[0]
    if (
        not isinstance(assertion, ast.Compare)
        or len(assertion.ops) != 1
        or not isinstance(assertion.ops[0], ast.Eq)
        or not isinstance(assertion.left, ast.Subscript)
        or not isinstance(assertion.left.slice, ast.Constant)
        or type(assertion.left.slice.value) not in {str, int}
        or not isinstance(expected := assertion.comparators[0], ast.Constant)
        or type(expected.value) is not int
    ):
        return ""
    recorded, wanted = map(int, comparison.groups())
    if wanted != expected.value or recorded == wanted:
        return ""
    observed_keys = set(
        re.findall(
            r"(?:^E[ \t]+|^.+\.py:[0-9]+: )KeyError: ([^\n]{1,160})$", after, re.M
        )
    )
    if len(observed_keys) != 1:
        return ""
    try:
        missing = ast.literal_eval(observed_keys.pop())
    except (ValueError, SyntaxError, RecursionError):
        return ""
    key = assertion.left.slice.value
    if type(missing) is not type(key) or missing != key:
        return ""
    return (
        f"The unchanged tracked assertion observed a count of {recorded} instead of {wanted}. "
        "Exchanging the adjacent filters in an isolated trial raised KeyError for the required "
        "mapping entry at that same assertion. The trial was discarded."
    )


def _start(node):
    return min(
        [
            node.lineno,
            *(decorator.lineno for decorator in getattr(node, "decorator_list", ())),
        ]
    )


def _references(nodes):
    roots = []
    for node in nodes:
        if isinstance(node, ast.ClassDef):
            roots.extend((*node.bases, *node.keywords, *node.decorator_list))
        else:
            roots.append(node)
    return {
        name
        for node in roots
        for child in ast.walk(node)
        if (name := getattr(child, "id", None) or getattr(child, "attr", None))
        and not isinstance(getattr(child, "ctx", None), ast.Store)
    }


def _bindings(node):
    if isinstance(node, _IMPORT):
        return {alias.asname or alias.name.split(".", 1)[0] for alias in node.names}
    if isinstance(node, _ASSIGNMENT):
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        return {
            child.id
            for target in targets
            for child in ast.walk(target)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
        }
    return {getattr(node, "name", None)}


def _definitions(root):
    pending = [(root, None)]
    while pending:
        node, owner = pending.pop()
        if isinstance(node, ast.ClassDef):
            yield node, None
            owner = node
        elif isinstance(node, _FUNCTION):
            yield node, owner
        pending.extend(
            (child, owner) for child in reversed(list(ast.iter_child_nodes(node)))
        )


def _consumers(active, candidates):
    bindings = _bindings(active)
    for node, _ in candidates:
        if not isinstance(node, _FUNCTION) or node is active:
            continue
        if isinstance(active, _ASSIGNMENT):
            references = _references((node,))
        else:
            references = _references(
                child.func for child in ast.walk(node) if isinstance(child, ast.Call)
            )
        if bindings & references:
            yield node


def _site_ranges(consumer, active):
    parents = {
        child: node
        for node in ast.walk(consumer)
        for child in ast.iter_child_nodes(node)
    }
    bindings = _bindings(active)
    ranges = {}
    for node in parents:
        if isinstance(active, _ASSIGNMENT):
            matches = isinstance(
                node, (ast.Name, ast.Attribute)
            ) and bindings & _references((node,))
        else:
            matches = isinstance(node, ast.Call) and bindings & _references(
                (node.func,)
            )
        if not matches:
            continue
        current = node
        while current is not consumer and not isinstance(current, ast.stmt):
            current = parents[current]
        if current is consumer:
            continue
        owner = parents.get(current)
        surrounding = current
        nearby = []
        if owner is not None:
            for _, siblings in ast.iter_fields(owner):
                if isinstance(siblings, list) and current in siblings:
                    start = siblings.index(current) + 1
                    nearby = [
                        item
                        for item in siblings[start : start + 8]
                        if isinstance(item, _CONTROL)
                    ]
                    break
        if isinstance(current, _CONTROL):
            block = current
        elif nearby:
            block = next(
                (item for item in nearby if isinstance(item, _LOOP)), nearby[0]
            )
        else:
            block = None
            while (surrounding := parents.get(surrounding)) not in {None, consumer}:
                if isinstance(surrounding, _CONTROL):
                    block = surrounding
                    break
        call = (current.lineno, current.end_lineno)
        if block is None:
            contextual = (call[0], min(consumer.end_lineno, call[1] + 6))
        else:
            contextual = (min(call[0], block.lineno), max(call[1], block.end_lineno))
        ranges[call, contextual] = (
            2 if isinstance(block, _LOOP) else int(block is not None)
        )
    return sorted(ranges, key=lambda pair: (-ranges[pair], pair[0][0]))


def _caller_excerpt(lines, consumer, active, limit):
    """Show the full caller when possible, otherwise show exact call-site lines."""

    def snippet(start, end):
        return f"# Lines {start}-{end}\n" + "".join(lines[start - 1 : end]) + "\n"

    complete = snippet(_start(consumer), consumer.end_lineno)
    if len(complete) <= limit:
        return complete
    ranges = _site_ranges(consumer, active)
    if not ranges:
        return ""
    call, contextual = ranges[0]
    header = snippet(
        _start(consumer), max(consumer.lineno, _start(consumer.body[0]) - 1)
    )
    whole = snippet(*contextual)
    if len(header) + len(whole) <= limit:
        return header + whole
    # Preserve a body around the actual use instead of character-clipping the
    # ends of the caller. Grow right first to retain subsequent controlled work.
    start, end = call
    for row in range(end + 1, contextual[1] + 1):
        if len(header) + len(snippet(start, row)) > limit:
            break
        end = row
    for row in range(start - 1, contextual[0] - 1, -1):
        if len(header) + len(snippet(row, end)) > limit:
            break
        start = row
    result = header + snippet(start, end)
    return result if len(result) <= limit else ""


def _self_attribute(value):
    return (
        isinstance(value, ast.Attribute)
        and isinstance(value.value, ast.Name)
        and value.value.id == "self"
    )


def _attribute_statements(method):
    for node in ast.walk(method):
        if isinstance(node, ast.Assign) and any(map(_self_attribute, node.targets)):
            yield node
        elif isinstance(node, ast.AnnAssign) and _self_attribute(node.target):
            yield node
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value.func
            if (
                isinstance(call, ast.Attribute)
                and call.attr in {"append", "extend"}
                and _self_attribute(call.value)
            ):
                yield node


def _dependency_inventory(lines, module, active, caller, shown, limit):
    """Summarize missing APIs for local class types referenced by this edit."""
    direct = _references((active, caller))
    related = _references(
        node
        for node in module.body
        if isinstance(node, _FUNCTION) and node.name in direct
    )
    classes = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name in direct | related
    ]
    classes.sort(key=lambda node: (node.name not in direct, node.lineno))
    heading = "# Referenced class APIs (selected source lines)\n"
    sections = []
    shown_lines = {line.strip() for line in shown.splitlines()}
    remaining = limit - len(heading) - 1
    for cls in classes[:3]:
        label = f"# {cls.name}\n"
        allowance = min(remaining, max(320, limit // len(classes))) - len(label)
        methods = [node for node in cls.body if isinstance(node, _FUNCTION)]
        methods.sort(key=lambda method: (method.name != "__init__", method.lineno))
        ranges = [
            (node.lineno, node.end_lineno)
            for node in cls.body
            if isinstance(node, _ASSIGNMENT)
        ]
        ranges.extend(
            (node.lineno, node.end_lineno)
            for method in methods
            for node in _attribute_statements(method)
        )
        ranges.extend(
            (method.lineno, max(method.lineno, _start(method.body[0]) - 1))
            for method in sorted(
                methods, key=lambda node: (node.name not in direct, node.lineno)
            )
            if method.name == "__init__"
            or method.name in direct
            or not method.name.startswith("_")
        )
        kept = []
        for start, end in ranges:
            text = "".join(lines[start - 1 : end])
            if len(text) > min(allowance, 220) or all(
                row.strip() in shown_lines for row in text.splitlines() if row.strip()
            ):
                continue
            kept.append(text)
            allowance -= len(text)
            shown_lines.update(row.strip() for row in text.splitlines())
        if kept:
            section = label + "".join(kept)
            sections.append(section)
            remaining -= len(section)
    return heading + "".join(sections) + "\n" if sections else ""


def focused_source_excerpt(
    source, target, *, review_note="", anchor=None, max_chars=3000
):
    """Prioritize the active symbol and definitions it refers to within a text budget."""
    if max_chars <= 0:
        return ""
    lines = source.splitlines(keepends=True)
    try:
        module = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        offset = max(0, min(len(lines), anchor or 1) - 5)
        return "".join(lines[offset:])[:max_chars]

    targets = _words(target)
    notes = _words(review_note)
    candidates = list(_definitions(module))
    candidates.extend(
        (node, None) for node in module.body if isinstance(node, _ASSIGNMENT)
    )
    if not candidates:
        return source[:max_chars]

    def rank(candidate):
        node, owner = candidate
        names = set().union(*(_words(name) for name in _bindings(node) if name))
        scope = _words(owner.name) if owner else set()
        at_anchor = anchor is not None and _start(node) <= anchor <= node.end_lineno
        anchored_definition = at_anchor and not isinstance(node, ast.ClassDef)
        start = _start(node)
        end = node.lineno if isinstance(node, ast.ClassDef) else node.end_lineno
        return (
            anchored_definition,
            -(end - start) if anchored_definition else 0,
            8 * len(names & targets)
            + 3 * len((names | scope) & notes)
            + 5 * at_anchor
            + min(4, len(_words("".join(lines[start - 1 : end])) & targets)),
            at_anchor,
            isinstance(node, _FUNCTION),
            -(node.end_lineno - node.lineno),
        )

    active, owner = max(candidates, key=rank)
    consumers = list(_consumers(active, candidates)) if anchor is not None else []
    seen = set()
    excerpts = []
    remaining = max_chars

    def append(node, *, end=None, clip_to=None):
        nonlocal remaining
        if node in seen:
            return
        start = _start(node)
        end = end if end is not None else node.end_lineno
        heading = f"# Line {start}\n"
        text = "".join(lines[start - 1 : end])
        if clip_to and len(heading) + len(text) + 1 > clip_to and clip_to > 90:
            cap = clip_to - len(heading) - 1
            if anchor is not None and start < anchor <= end:
                middle = "".join(lines[max(start - 1, anchor - 4) : anchor + 3])
                text = text[: cap // 3] + "\n# …\n" + middle[: cap - cap // 3 - 5]
            else:
                head = cap // 2
                text = text[:head] + "\n# …\n" + text[-(cap - head - 5) :]
        chunk = heading + text + "\n"
        if len(chunk) <= remaining:
            excerpts.append(chunk)
            seen.add(node)
            remaining -= len(chunk)

    def declaration(node):
        if not isinstance(node, ast.ClassDef):
            append(node)
            return
        header_end = max(node.lineno, _start(node.body[0]) - 1)
        append(node, end=header_end)
        for child in node.body:
            if isinstance(child, _FUNCTION) and child.name == "__init__":
                append(child, clip_to=min(max_chars // 3, remaining))
        for child in node.body:
            if isinstance(child, _ASSIGNMENT):
                append(child)

    if isinstance(active, ast.ClassDef):
        declaration(active)
    else:
        append(active, clip_to=max_chars * (45 if consumers else 66) // 100)
    if owner is not None:
        declaration(owner)
    if consumers and max_chars >= 2400:
        primary = _caller_excerpt(lines, consumers[0], active, remaining)
        inventory = _dependency_inventory(
            lines,
            module,
            active,
            consumers[0],
            "".join(excerpts) + primary,
            min(1100, remaining // 5),
        )
        if inventory:
            excerpts.append(inventory)
            remaining -= len(inventory)
    for index, node in enumerate(consumers[:2]):
        limit = remaining if index == 0 else min(max_chars // 4, remaining)
        chunk = _caller_excerpt(lines, node, active, limit)
        if chunk:
            excerpts.append(chunk)
            seen.add(node)
            remaining -= len(chunk)
        elif isinstance(active, _ASSIGNMENT):
            append(node, clip_to=limit)

    related = [active, *seen]
    for _ in range(2):
        names = _references(related)
        available = [
            *module.body,
            *(
                child
                for node, _ in candidates
                if isinstance(node, ast.ClassDef) and node in seen
                for child in node.body
            ),
        ]
        matches = [
            node for node in available if node not in seen and _bindings(node) & names
        ]
        for node in matches:
            declaration(node)
        related = list(seen)
    return "".join(excerpts)

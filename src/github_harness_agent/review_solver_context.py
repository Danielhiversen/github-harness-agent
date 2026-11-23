"""Select compact Python source excerpts for a single review concern."""

import ast
import builtins
import re
import symtable
import tarfile
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath

from .check_diagnostics import check_locations
from .review_assertions import _STRAIGHT_LINE
from .review_definitions import _symbols

_FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)
_IMPORT = (ast.Import, ast.ImportFrom)
_ASSIGNMENT = (ast.Assign, ast.AnnAssign)
_LOOP = (ast.For, ast.AsyncFor, ast.While)
_CONTROL = (*_LOOP, ast.If, ast.With, ast.AsyncWith, ast.Try, ast.TryStar, ast.Match)
_STOCK_NODE = re.compile(
    r"(?P<path>(?:[\w-][\w.-]*/)*[\w-][\w.-]*\.py)"
    r"(?P<name>(?:::[A-Za-z_]\w*)*::test\w*)"
)
_GENERATED_TEST_STEM = re.compile(r"test_review_regressions(?:_\d+)?")
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


_NATIVE_TEST_FRAME = re.compile(r'^File "([^"\n]{1,512}\.py)", line ([1-9][0-9]{0,6})(?:, in .*)?$')
_TEST_MISSING_NAME = re.compile(
    r"^(?:E\s+|[^\n]{1,512}\.py:[1-9][0-9]{0,6}:\s*|(?:FAILED|ERROR) \S+ - )?"
    r"NameError:\s+name (['\"])([^\W\d]\w{0,100})\1 is not defined\b"
)
_NAME_ERROR_CRASH_FRAME = re.compile(r":\s*NameError(?:\s|:|$)")
_ARGUMENT_TYPEERROR = re.compile(
    r"^(?:E\s+|[^\n]{1,512}\.py:[1-9][0-9]{0,6}:\s*)?"
    r"TypeError:\s+([A-Za-z_]\w{0,100})(?:\.([A-Za-z_]\w{0,100}))?\(\) "
    r"(?:got an unexpected keyword argument|missing [1-9][0-9]* required "
    r"(?:positional|keyword-only) arguments?)(?=[:\s]|$)"
)


def _reviewed_argument_signature(sources, owner, method):
    matches = []
    for path, source in sources.items():
        try:
            declarations = ast.parse(source).body
            for declaration in declarations:
                if not isinstance(declaration, ast.ClassDef) or declaration.name != owner:
                    continue
                for node in declaration.body:
                    if isinstance(node, _FUNCTION) and node.name == method:
                        signature = f"{owner}.{method}({ast.unparse(node.args)})"
                        if node.returns is not None:
                            signature += f" -> {ast.unparse(node.returns)}"
                        matches.append(f"{path}:{node.lineno}\n{signature}")
        except (SyntaxError, ValueError, RecursionError):
            continue
    return matches[0] if len(matches) == 1 else ""


def test_failure_focus(sources, diagnostic, *, max_bytes=660):
    """Return at most one small excerpt from reviewed source named by the test failure."""
    diagnostic_lines = diagnostic.splitlines()
    for line in reversed(diagnostic_lines):
        match = _ARGUMENT_TYPEERROR.match(line.lstrip())
        if match:
            signature = _reviewed_argument_signature(sources, match[1], match[2] or "__init__")
            if signature and len(signature.encode()) <= max_bytes:
                return signature
    for line in reversed(diagnostic_lines):
        native = _NATIVE_TEST_FRAME.fullmatch(line.lstrip())
        frames = ((native[1], int(native[2])),) if native else check_locations(("pytest",), line)
        for raw_path, row in frames:
            path = raw_path.removeprefix("/work/").removeprefix("./")
            if path not in sources:
                continue
            lines = sources[path].splitlines(keepends=True)
            if row > len(lines):
                continue
            for radius in (2, 1, 0):
                context = "".join(lines[max(0, row - 1 - radius) : row + radius]).rstrip("\n")
                focus = f"{path}:{row}\n{context}"
                if context.strip() and len(focus.encode()) <= max_bytes:
                    return focus
    return ""


def test_name_error_bindings(path, draft, diagnostic, *, max_bytes=600):
    """Quote exact existing bindings for a NameError in the generated draft."""
    details = diagnostic.splitlines()
    missing = {
        match[2] for line in details if (match := _TEST_MISSING_NAME.match(line.lstrip()))
    }
    if len(missing) != 1:
        return ""
    name = missing.pop()
    locations = []
    for line in details:
        if frame := _NATIVE_TEST_FRAME.fullmatch(line.lstrip()):
            locations.append((frame[1], int(frame[2]), True))
        else:
            locations.extend(
                (file, row, bool(_NAME_ERROR_CRASH_FRAME.search(line)))
                for file, row in check_locations(("pytest",), line)
            )
    if not locations:
        return ""
    raw_path, row, crash = locations[-1]
    if not crash or raw_path.removeprefix("/work/").removeprefix("./") != path:
        return ""
    lines = draft.splitlines()
    if row > len(lines):
        return ""
    try:
        module = ast.parse(draft)
    except (SyntaxError, ValueError, RecursionError):
        return ""

    enclosing = [
        node.body for _, node in _symbols(module.body, nested=True)
        if isinstance(node, _FUNCTION) and node.lineno <= row <= node.end_lineno
    ]
    heading = f"Current draft bindings for missing name {name!r} (quoted source):"
    remaining = max_bytes - len(heading.encode()) - 1
    selected = []
    for statements in (*reversed(enclosing), module.body):
        for node in statements:
            if not isinstance(node, (*_IMPORT, *_ASSIGNMENT, *_FUNCTION, ast.ClassDef)):
                continue
            aliases = (
                {alias.name.rsplit(".", 1)[-1] for alias in node.names}
                if isinstance(node, _IMPORT) else set()
            )
            if name not in _bindings(node) | aliases:
                continue
            end = (
                max(node.lineno, _start(node.body[0]) - 1)
                if isinstance(node, (*_FUNCTION, ast.ClassDef)) else node.end_lineno
            )
            snippet = "\n".join(lines[_start(node) - 1:end])
            size = len(snippet.encode()) + 1
            if snippet and snippet not in selected and size <= remaining:
                selected.append(snippet)
                remaining -= size
    return "\n".join((heading, *selected)) if selected else ""


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


def _pytest_fixtures(body, *, symbols=None):
    for node in body:
        if not isinstance(node, _FUNCTION):
            continue
        for decorator in node.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            symbol = decorator.func if call else decorator
            looks_like_fixture = (
                isinstance(symbol, ast.Name) and symbol.id == "fixture"
                or isinstance(symbol, ast.Attribute) and symbol.attr == "fixture"
            )
            if not (looks_like_fixture if symbols is None else _dotted_name(symbol) in symbols):
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
    reviewed_calls = {}
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
                    reviewed_calls[local] = modules[node.module][alias.name]
            reviewed_calls.update(
                {f"{local}.{name}": returned for name, returned in modules.get(qualified, {}).items()}
            )
    return bindings, reviewed_calls


def _module_pytest_fixtures(body, imports):
    bindings = Counter(name for item in body for name in _bindings(item))
    symbols = set()
    for local, declaration in imports.items():
        if bindings[local] != 1:
            continue
        for alias in declaration.names:
            if (alias.asname or alias.name) != local:
                continue
            if isinstance(declaration, ast.Import) and alias.name == "pytest":
                symbols.add(f"{local}.fixture")
            elif getattr(declaration, "module", None) == "pytest" and alias.name == "fixture":
                symbols.add(local)
    fixtures = {}
    nodes = set()
    autouse = set()
    for name, item, automatic in _pytest_fixtures(body, symbols=symbols):
        nodes.add(item)
        if automatic:
            autouse.add(name)
        fixtures[name] = item if name not in fixtures and bindings[item.name] == 1 else None
    return fixtures, nodes, autouse


def _local_fixture_setup(node, fixtures, *, autouse=()):
    wanted = (_fixture_arguments(node) | set(autouse)) & fixtures.keys()
    selected = set()
    for _ in range(4):
        if not wanted:
            return selected
        for name in wanted:
            fixture = fixtures[name]
            if fixture is None:
                return None
            selected.add(fixture)
        deps = {
            dep for fixture in selected for dep in _fixture_arguments(fixture)
        }
        wanted = {
            name for name in deps & fixtures.keys() if fixtures[name] not in selected
        }
    return None if wanted else selected


def _source_root(path):
    parts = path.parts
    if "src" not in parts[:-1]:
        return PurePosixPath()
    return PurePosixPath(*parts[: len(parts) - parts[::-1].index("src")])


@dataclass(frozen=True)
class _ReturnedValue:
    methods: dict[str, bool | tuple[bool, ...]]
    mapping: bool | tuple[bool, ...] = False
    class_name: str = ""


@dataclass(frozen=True)
class _UsageOrigin:
    rows: tuple[str, ...]
    methods: dict[str, bool | tuple[bool, ...]]
    mapping: bool = False
    class_name: str = ""


def _mapping_result(annotation):
    def is_mapping(item):
        item = item.value if isinstance(item, ast.Subscript) else item
        return _dotted_name(item) in {"dict", "Dict", "typing.Dict", "Mapping", "typing.Mapping", "collections.abc.Mapping"}

    if (
        isinstance(annotation, ast.Subscript)
        and _dotted_name(annotation.value) in {"tuple", "Tuple", "typing.Tuple"}
        and isinstance(annotation.slice, ast.Tuple)
        and not any(isinstance(item, ast.Constant) and item.value is Ellipsis for item in annotation.slice.elts)
    ):
        return tuple(is_mapping(item) for item in annotation.slice.elts)
    return is_mapping(annotation)


def _reviewed_calls(declarations):
    classes = {
        node.name: {
            method.name: _mapping_result(method.returns) for method in node.body
            if isinstance(method, _FUNCTION) and not method.name.startswith("_")
        }
        for node in declarations
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    }
    functions = {}
    for node in declarations:
        if not isinstance(node, _FUNCTION) or node.name.startswith("_"):
            continue
        result = getattr(node.returns, "id", "")
        if result not in classes:
            returns = [item.value for item in node.body if isinstance(item, ast.Return)]
            if len(returns) == 1 and isinstance(returns[0], ast.Call):
                result = _dotted_name(returns[0].func)
        functions[node.name] = _ReturnedValue(classes.get(result, {}), _mapping_result(node.returns))
    return {
        **{name: _ReturnedValue(methods, class_name=name) for name, methods in classes.items()},
        **functions,
    }


def _shape_operand(node):
    if isinstance(node, ast.Starred):
        return "*" + _shape_operand(node.value)
    name = _dotted_name(node)
    return name if len(name) <= 72 and name else "..."


def _shape_targets(node):
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, (ast.Tuple, ast.List)) and all(
        isinstance(item, ast.Name) for item in node.elts
    ):
        return tuple(item.id for item in node.elts)
    return ()


def _rebound_names(node):
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del)):
            yield item.id
        elif isinstance(item, ast.alias):
            yield item.asname or item.name.split(".", 1)[0]
        elif isinstance(item, (*_FUNCTION, ast.ClassDef, ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            if item.name:
                yield item.name
        elif isinstance(item, ast.MatchMapping) and item.rest:
            yield item.rest


def _usage_shapes(node, reviewed_calls, *, contract_only=False):
    """Read calls and public attributes from straight-line code and context bodies."""
    bound = _fixture_arguments(node)
    bound.update(arg.arg for arg in (node.args.vararg, node.args.kwarg) if arg is not None)
    bound.update(
        name for item in ast.walk(node) if isinstance(item, (ast.Global, ast.Nonlocal))
        for name in item.names
    )
    local_names = set(_rebound_names(node)) | bound
    yield from _usage_block(node.body, reviewed_calls, local_names, {}, bound, contract_only)


def _usage_block(statements, reviewed_calls, local_names, origins, bound, contract_only):
    for statement in statements:
        rebound = Counter(_rebound_names(statement))
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            context_names = set().union(*(_rebound_names(item) for item in statement.items))
            nested_origins = {name: origin for name, origin in origins.items() if name not in context_names}
            yield from _usage_block(
                statement.body, reviewed_calls, local_names, nested_origins, bound | context_names,
                contract_only,
            )
            # A suppressing exit can leave a previous value in place. A fresh name with a
            # single proven mapping assignment either has that value or cannot be read.
            for name, writes in rebound.items():
                origins.pop(name, None)
                origin = nested_origins.get(name)
                if writes == 1 and name not in bound and name not in context_names and origin and origin.mapping:
                    origins[name] = origin
            bound.update(rebound)
            continue
        target = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
        elif isinstance(statement, ast.AnnAssign):
            target = statement.target
        names = _shape_targets(target)
        expression = getattr(statement, "value", None)
        awaited = isinstance(expression, ast.Await)
        if awaited:
            expression = expression.value
        rows = ()
        returned = _ReturnedValue({})
        if isinstance(expression, ast.Call):
            called = _dotted_name(expression.func)
            receiver, _, method = called.rpartition(".")
            origin = origins.get(receiver)
            if (
                contract_only and isinstance(expression.func, ast.Attribute)
                and isinstance(expression.func.value, ast.Call)
            ):
                factory = _dotted_name(expression.func.value.func)
                if factory.split(".", 1)[0] not in local_names:
                    origin = reviewed_calls.get(factory)
                method = expression.func.attr
            if called in reviewed_calls and called.split(".", 1)[0] not in local_names:
                returned = reviewed_calls[called]
                if contract_only:
                    called = ""
            elif origin and method in origin.methods:
                mapping = origin.methods[method]
                returned = _ReturnedValue({}, mapping)
                if contract_only:
                    returns_mapping = mapping is True or (isinstance(mapping, tuple) and any(mapping))
                    called = f"{origin.class_name}.{method}" if origin.class_name and returns_mapping else ""
            else:
                called = ""
            if called:
                args = ["..." if contract_only else _shape_operand(arg) for arg in expression.args]
                args.extend(
                    f"{arg.arg}={'...' if contract_only else _shape_operand(arg.value)}" if arg.arg else "**..."
                    for arg in expression.keywords
                )
                invocation = f"{called}({', '.join(args)})"
                if awaited and not contract_only:
                    invocation = "await " + invocation
                if names and not contract_only:
                    invocation = f"{', '.join(names)} = {invocation}"
                if len(invocation.encode()) <= 210:
                    setup = origins[receiver].rows if receiver in origins else ()
                    rows = (*setup[-1:], invocation)
                    yield called, rows
        value = statement.test if isinstance(statement, ast.Assert) else expression
        if value is not None:
            for item in ast.walk(value):
                if (
                    isinstance(item, ast.Subscript) and isinstance(item.ctx, ast.Load)
                    and isinstance(item.value, ast.Name) and not item.value.id.startswith("_")
                    and len(item.value.id) <= 72 and (origin := origins.get(item.value.id))
                    and origin.mapping and isinstance(item.slice, ast.Constant)
                    and isinstance(key := item.slice.value, str)
                    and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,47}", key)
                ):
                    receiver = "result" if contract_only else item.value.id
                    lookup = f"{receiver}[{key!r}]"
                    yield lookup, (*origin.rows, lookup)
                if contract_only or not isinstance(item, ast.Attribute):
                    continue
                attribute = _dotted_name(item)
                owner, _, member = attribute.partition(".")
                if (
                    owner in origins and not any(part.startswith("_") for part in member.split("."))
                    and member not in origins[owner].methods and len(attribute.encode()) <= 160
                ):
                    yield attribute, (*origins[owner].rows, attribute)
        for name in rebound:
            origins.pop(name, None)
        bound.update(rebound)
        if rows or (contract_only and returned.class_name):
            single = isinstance(target, ast.Name)
            if single and isinstance(returned.mapping, bool):
                mapping = (returned.mapping,)
            elif not single and isinstance(returned.mapping, tuple) and len(returned.mapping) == len(names):
                mapping = returned.mapping
            else:
                mapping = (False,) * len(names)
            for name, is_mapping in zip(names, mapping):
                origins[name] = _UsageOrigin(
                    rows, returned.methods if single else {}, is_mapping,
                    returned.class_name if single else "",
                )


def _python_module(path):
    source_path = PurePosixPath(path)
    parts = source_path.relative_to(_source_root(source_path)).with_suffix("").parts
    if parts[-1:] == ("__init__",):
        parts = parts[:-1]
    return ".".join(parts) if parts and all(part.isidentifier() for part in parts) else ""


def _reviewed_modules(sources):
    modules = {}
    for path, source in sources.items():
        if not (name := _python_module(path)):
            continue
        try:
            declarations = ast.parse(source).body
        except (SyntaxError, ValueError, RecursionError):
            continue
        modules[name] = _reviewed_calls(declarations)
    return modules


def production_protocol_example(preview, sources, query, *, max_bytes=1150):
    """Quote a dependency call and nearby handling from a helper's production entry point."""
    try:
        setup = ast.parse(preview).body
    except (SyntaxError, ValueError, RecursionError):
        return ""
    bindings = {}
    for node in setup:
        if isinstance(node, ast.ImportFrom) and not node.level:
            bindings.update({a.asname or a.name: f"{node.module}.{a.name}" for a in node.names})
        elif isinstance(node, ast.Import):
            bindings.update({a.asname or a.name: a.name for a in node.names})
    used = set()
    for helper in setup:
        if not isinstance(helper, _FUNCTION) or helper.name.startswith("test_"):
            continue
        local = set(_fixture_arguments(helper)) | set(_rebound_names(helper))
        for call in ast.walk(helper):
            if isinstance(call, ast.Call) and (name := _dotted_name(call.func)):
                for alias, target in bindings.items():
                    if alias.split(".")[0] not in local and (name == alias or name.startswith(alias + ".")):
                        used.add(target + name[len(alias):])

    if not used:
        return ""
    best = ""
    best_rank = None
    for path, source in sources.items():
        file = PurePosixPath(path)
        if (file.suffix != ".py" or "tests" in file.parts or "test" in file.parts or file.name == "conftest.py"
                or file.name.startswith("test_") or file.name.endswith("_test.py")):
            continue
        module = _python_module(path)
        if not any(name.startswith(module + ".") for name in used):
            continue
        try:
            declarations = ast.parse(source).body
        except (SyntaxError, ValueError, RecursionError):
            continue
        lines = source.splitlines()
        for entry in declarations:
            if not isinstance(entry, (*_FUNCTION, ast.ClassDef)) or module + "." + entry.name not in used:
                continue
            methods = entry.body if isinstance(entry, ast.ClassDef) else [entry]
            for method in methods:
                if not isinstance(method, _FUNCTION):
                    continue
                parameters = _fixture_arguments(method) - {"self", "cls"}
                for call in ast.walk(method):
                    if not isinstance(call, ast.Call) or not (name := _dotted_name(call.func)):
                        continue
                    parts = name.split(".")
                    if not (parts[0] in parameters or parts[0] == "self" and len(parts) >= 3):
                        continue
                    # Stay inside the implementing method and keep its original
                    # indentation, call arguments and consuming expressions visible.
                    end = min(call.end_lineno + 8, method.end_lineno)
                    if end - call.lineno > 18:
                        continue
                    scope = f"{entry.name}.{method.name}" if entry is not method else entry.name
                    snippet = f"{path}:{call.lineno} ({scope})\n" + "\n".join(
                        lines[call.lineno - 1:end]
                    )
                    if len(snippet.encode()) > max_bytes:
                        continue
                    tokens = set(re.findall(r"[a-z]{3,}", snippet.casefold()))
                    rank = (len(query & tokens), len(parts), -call.lineno)
                    if best_rank is None or rank > best_rank:
                        best_rank, best = rank, snippet
    return best


def operator_api_contract(fixtures, sources, *, max_bytes=700):
    """Read public method shapes and dictionary keys without copying test code."""
    modules = _reviewed_modules(sources)
    calls = {}
    for source in tuple(fixtures.values())[:4]:
        if len(source.encode()) > 128 * 1024:
            continue
        try:
            body = ast.parse(source).body
        except (SyntaxError, ValueError, RecursionError):
            continue
        _, reviewed_calls = _reviewed_imports(body, modules)
        constructors = {
            name: value for name, value in reviewed_calls.items() if value.class_name
        }
        for node in body:
            if not isinstance(node, _FUNCTION) or not node.body:
                continue
            last = node.body[-1]
            if not isinstance(last, ast.Return) or not isinstance(last.value, ast.Call):
                continue
            constructor = _dotted_name(last.value.func)
            bound = set(_rebound_names(node)) | _fixture_arguments(node)
            if (
                constructor in constructors and constructor.split(".", 1)[0] not in bound
                and sum(isinstance(item, ast.Return) for item in ast.walk(node)) == 1
            ):
                reviewed_calls[node.name] = constructors[constructor]
        for node in body:
            if not isinstance(node, _FUNCTION) or not node.name.startswith("test_"):
                continue
            for _, rows in _usage_shapes(node, reviewed_calls, contract_only=True):
                if rows:
                    calls.setdefault(rows[0], {}).update(dict.fromkeys(rows[1:]))
    lines = []
    for call, lookups in calls.items():
        for line in (call, *lookups):
            size = len(line.encode()) + 1
            if size > max_bytes:
                return "\n".join(lines)
            lines.append(line)
            max_bytes -= size
    return "\n".join(lines)


def _mock_patch_imports(body, imports):
    names = Counter(name for item in body for name in _bindings(item))
    patches = {}
    for local, node in imports.items():
        if names[local] != 1:
            continue
        for alias in node.names:
            name = alias.asname or alias.name
            if isinstance(node, ast.Import) and name.split(".", 1)[0] == local:
                suffix = {"unittest": ".mock.patch", "unittest.mock": ".patch"}.get(alias.name)
            elif getattr(node, "module", None) == "unittest" and alias.name == "mock" and name == local:
                suffix = ".patch"
            elif getattr(node, "module", None) == "unittest.mock" and alias.name == "patch" and name == local:
                suffix = ""
            else:
                continue
            if suffix is not None:
                binding = name + suffix
                patches[binding] = patches[binding + ".object"] = node
    return patches


def _mock_bindings(body):
    names = Counter(name for item in body for name in _bindings(item))
    bindings = {}
    for item in body:
        if isinstance(item, (*_FUNCTION, ast.ClassDef)) and names[item.name] == 1:
            bindings[item.name] = item
        elif isinstance(item, _ASSIGNMENT):
            targets = item.targets if isinstance(item, ast.Assign) else (item.target,)
            if len(targets) == 1 and isinstance(target := targets[0], ast.Name) and names[target.id] == 1:
                bindings[target.id] = item.value
    return bindings


def _mock_payload(node, bindings, depth=0, *, max_depth=3):
    if depth >= max_depth:
        return "..."
    if isinstance(node, ast.Name):
        node = bindings.get(node.id)
    if isinstance(node, ast.Dict):
        entries = [
            f"{key.value!r}: {_mock_payload(value, bindings, depth + 1, max_depth=max_depth)}"
            for key, value in zip(node.keys[:5], node.values[:5])
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,47}", key.value)
        ]
        if len(entries) < len(node.keys):
            entries.append("...")
        return "{" + ", ".join(entries) + "}"
    if isinstance(node, (ast.Tuple, ast.List)):
        first = _mock_payload(node.elts[0], bindings, depth + 1, max_depth=max_depth) if node.elts else ""
        if len(node.elts) > 1 and first != "...":
            first += ", ..."
        return f"[{first}]" if isinstance(node, ast.List) else f"({first},)" if first else "()"
    return "..."


def _mock_value(node, bindings):
    if isinstance(node, ast.Lambda):
        names = ", ".join(arg.arg for arg in (*node.args.posonlyargs, *node.args.args))
        return f"lambda {names}: {_mock_payload(node.body, bindings)}"
    shape = _mock_payload(node, bindings)
    return _shape_operand(node) if shape == "..." or len(shape.encode()) > 150 else shape


def _fake_method_shapes(node, bindings):
    if isinstance(node, ast.ClassDef):
        methods = [
            (f"{node.name}.{item.name}", item)
            for item in node.body if isinstance(item, _FUNCTION)
        ][:4]
    else:
        methods = [(node.name, node)] if isinstance(node, _FUNCTION) else []
    for name, method in methods:
        if isinstance(last := method.body[-1], ast.Return):
            payload = _mock_payload(last.value, {**bindings, **_mock_bindings(method.body[:-1])})
            yield f"# {name}(...) returns {payload}"
        else:
            yield f"# {name}(...)"


def _mock_actions(statements):
    for statement in statements:
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            for item in statement.items:
                if isinstance(item.context_expr, ast.Call):
                    yield item.context_expr, item.optional_vars, statement.body
            yield from _mock_actions(statement.body)
        elif isinstance(statement, (ast.Expr, *_ASSIGNMENT)) and isinstance(statement.value, ast.Call):
            yield statement.value, None, ()


def _mock_return_shapes(alias, statements, bindings):
    if not isinstance(alias, ast.Name) or any(alias.id in _rebound_names(item) for item in statements):
        return
    for index, statement in enumerate(statements):
        if not isinstance(statement, _ASSIGNMENT):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
        if len(targets) != 1:
            continue
        name = _dotted_name(targets[0])
        if not name.startswith(alias.id + ".") or not name.endswith((".return_value", ".side_effect")):
            continue
        payload = _mock_payload(statement.value, {**bindings, **_mock_bindings(statements[:index])})
        row = f"{name} = {payload}"
        if payload != "..." and len(row.encode()) <= 250:
            yield row


def _mock_setup_shapes(node, setup, module_bindings, patches, fixture_names):
    found = {}
    for scope in (*sorted(setup, key=lambda item: item.lineno), node):
        arguments = _fixture_arguments(scope)
        rebound = set(_rebound_names(scope))
        setters = {
            name: declaration for name, declaration in patches.items()
            if name.split(".", 1)[0] not in arguments | rebound
        }
        for name, helpers in (
            ("monkeypatch", ("monkeypatch.setattr",)),
            ("mocker", ("mocker.patch", "mocker.patch.object")),
        ):
            if name in arguments and name not in rebound | fixture_names:
                setters.update(dict.fromkeys(helpers))
        if not setters:
            continue
        actions = [
            (item, None, ()) for item in scope.decorator_list if isinstance(item, ast.Call)
        ]
        actions.extend(_mock_actions(scope.body))
        for call, alias, block in actions:
            if (name := _dotted_name(call.func)) not in setters or not call.args:
                continue
            count = 2 if name.endswith(".object") or (
                name == "monkeypatch.setattr" and not isinstance(call.args[0], ast.Constant)
            ) else 1
            target = [
                repr(arg.value) if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,119}", arg.value) else _dotted_name(arg)
                for arg in call.args[:count]
            ]
            if len(target) != count or not all(target):
                continue
            bindings = {
                **{key: item for key, item in module_bindings.items() if key not in arguments | rebound},
                **_mock_bindings([item for item in scope.body if item.lineno < call.lineno]),
            }
            values = call.args[count:]
            args = [*target, *(_mock_value(item, bindings) for item in values)]
            args.extend(
                f"{kw.arg}={_mock_value(kw.value, bindings)}" if kw.arg else "**..."
                for kw in call.keywords
            )
            row = f"{name}({', '.join(args)})"
            if call in scope.decorator_list:
                row = "@" + row
            elif block:
                suffix = f" as {alias.id}" if isinstance(alias, ast.Name) else ""
                row = f"with {row}{suffix}:"
            if len(row.encode()) > 250:
                continue
            if declaration := setters[name]:
                found.setdefault(ast.unparse(declaration))
            found.setdefault(row)
            for returned in _mock_return_shapes(alias, block, bindings):
                found.setdefault(returned)
            for value in (*values, *(kw.value for kw in call.keywords)):
                for method in _fake_method_shapes(bindings.get(_dotted_name(value)), bindings):
                    if len(method.encode()) <= 250:
                        found.setdefault(method)
            if len(found) >= 5:
                return tuple(found)[:5]
    return tuple(found)


def _usage_relevance(symbol, query_words):
    return len(_words(symbol.replace("_", " ")) & query_words)


def _temporary_path(node):
    parts = []
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and len(parts) < 3:
        part = node.right
        if not (
            isinstance(part, ast.Constant) and isinstance(part.value, str)
            and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,47}", part.value)
        ):
            return ""
        parts.append(repr(part.value))
        node = node.left
    if parts and isinstance(node, ast.Name) and node.id == "tmp_path":
        return " / ".join(("tmp_path", *reversed(parts)))
    return ""


def _temporary_file_shapes(setup, module_bindings, imports, fixture_names):
    if "tmp_path" in fixture_names:
        return
    for fixture in sorted(setup, key=lambda item: item.lineno):
        arguments = _fixture_arguments(fixture)
        rebound = set(_rebound_names(fixture))
        if "tmp_path" not in arguments or "tmp_path" in rebound:
            continue
        bound = arguments | rebound
        for call, _, _ in _mock_actions(fixture.body):
            if not isinstance(call.func, ast.Attribute) or call.func.attr not in ("write_text", "write_bytes"):
                continue
            bindings = {
                **{name: item for name, item in module_bindings.items() if name not in bound},
                **_mock_bindings([item for item in fixture.body if item.lineno < call.lineno]),
            }
            receiver = call.func.value
            location = bindings.get(receiver.id) if isinstance(receiver, ast.Name) else receiver
            if not (path := _temporary_path(location)):
                continue
            used_imports = set()

            def render_args(expression, depth):
                return ", ".join((
                    *(render(item, depth) for item in expression.args),
                    *(f"{kw.arg}={render(kw.value, depth)}" if kw.arg else "**..." for kw in expression.keywords),
                ))

            def render(item, depth):
                value = bindings.get(item.id, item) if isinstance(item, ast.Name) else item
                if depth and isinstance(value, ast.Call):
                    name = _dotted_name(value.func)
                    root = name.split(".", 1)[0]
                    if root in imports and root not in bound:
                        used_imports.add(imports[root])
                        return f"{name}({render_args(value, depth - 1)})"
                shape = _mock_payload(value, bindings, max_depth=5)
                return _shape_operand(item) if shape == "..." else shape

            target = receiver.id if isinstance(receiver, ast.Name) else f"({path})"
            write = f"{target}.{call.func.attr}({render_args(call, 2)})"
            if len(write.encode()) <= 300:
                yield (
                    f"# Module-local fixture {fixture.name}({', '.join(sorted(arguments))})",
                    *(ast.unparse(item) for item in sorted(used_imports, key=lambda item: item.lineno)),
                    *((f"{target} = {path}",) if isinstance(receiver, ast.Name) else ()),
                    write,
                )


def _extended_context_path(path):
    if path.is_absolute() or any(part.startswith(".") for part in path.parts):
        return False
    if _GENERATED_TEST_STEM.fullmatch(path.stem):
        return False
    return not _words(str(path)).intersection({
        "operator", "secret", "secrets", "credential", "credentials",
        "settings", "config", "configuration", "script", "scripts",
    })


def _complete_pytest_nodes(body, node, setup, *, references=None):
    bindings = {}
    for item in body:
        if isinstance(item, (*_IMPORT, *_ASSIGNMENT, *_FUNCTION)) and (
            not isinstance(item, _FUNCTION) or not item.name.startswith("test_")
        ):
            for name in _bindings(item):
                bindings.setdefault(name, []).append(item)
    selected = {node, *setup}
    for _ in range(8):
        needed = (
            references(selected) if references else _references(selected, include_attributes=False)
        ) & bindings.keys()
        if any(len(bindings[name]) != 1 for name in needed):
            return ()
        extra = {bindings[name][0] for name in needed} - selected
        if not extra:
            return sorted(selected, key=lambda item: item.lineno)
        selected.update(extra)
    return ()


_PRIOR_SETUP_IMPORTS = {
    "collections", "dataclasses", "datetime", "decimal", "enum", "functools", "io",
    "json", "math", "pathlib", "pytest", "re", "textwrap", "types", "typing",
    "unittest.mock", "yaml",
}
_PRIOR_SETUP_FIXTURES = {"tmp_path", "tmp_path_factory", "monkeypatch", "request", "capsys", "capfd"}
_PRIOR_SENSITIVE = {"credential", "credentials", "environ", "getenv", "operator",
                    "password", "secret", "secrets", "token", "apikey"}


def _prior_sensitive(spelling):
    words = _words((spelling or "").replace("_", " "))
    return bool(words & _PRIOR_SENSITIVE or {"api", "key"} <= words)


def _prior_setup_import(node, modules):
    if isinstance(node, ast.Import):
        names = [item.name for item in node.names]
    elif not node.level and node.module and all(item.name != "*" for item in node.names):
        names = [node.module] if node.module in modules else [
            f"{node.module}.{item.name}" for item in node.names
        ]
    else:
        return False
    return all(name in modules and not _prior_sensitive(name) for name in names) and not any(
        _prior_sensitive(item.name) or _prior_sensitive(item.asname) for item in node.names
    )


def _prior_setup_source(
    items, fixtures, modules, *, sample=None, reference_helpers=False, fixture_placeholders=False,
):
    available = _PRIOR_SETUP_FIXTURES | {name for name, fixture in fixtures.items() if fixture in items}
    fixture_nodes = set(fixtures.values())
    for item in items:
        if isinstance(item, _IMPORT):
            if not _prior_setup_import(item, modules):
                return ""
            continue
        if not isinstance(item, _FUNCTION) or (
            (item in fixture_nodes or item is sample) and _fixture_arguments(item) - available
        ):
            return ""
        borrowed = (reference_helpers or fixture_placeholders) and item is not sample
        token_arguments = {
            node.arg for node in ast.walk(item) if isinstance(node, ast.arg)
            and "token" in _words(node.arg.replace("_", " "))
            and not (_words(node.arg.replace("_", " ")) & (_PRIOR_SENSITIVE - {"token"}))
        } if borrowed else set()
        for node in ast.walk(item):
            if borrowed and isinstance(node, _IMPORT) and _prior_setup_import(node, modules):
                continue
            spelling = (
                getattr(node, "id", "") or getattr(node, "attr", "")
                or getattr(node, "name", "") or getattr(node, "arg", "")
            )
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if re.search(r"(?m)^\s*assert\b", node.value):
                    return ""
                spelling = node.value
            inert_fixture_keyword = (
                borrowed and isinstance(node, ast.keyword)
                and isinstance(node.value, ast.Constant) and node.value.value == "fixture"
            )
            if (
                isinstance(node, (ast.Global, *_IMPORT))
                or (item is not sample and (isinstance(node, ast.Assert) or (spelling or "").startswith("assert")))
                or (isinstance(node, _FUNCTION) and node is not sample and node.name.startswith("test_"))
                or spelling in {"raises", "fail", "eval", "exec", "__import__"}
                or (
                    _prior_sensitive(spelling) and not inert_fixture_keyword and not (
                        isinstance(node, (ast.arg, ast.Name)) and spelling in token_arguments
                    )
                )
            ):
                return ""
    source = "\n\n".join(ast.unparse(item) for item in items)
    try:
        module = symtable.symtable(source, "<prior pytest setup>", "exec")
    except (SyntaxError, ValueError, RecursionError):
        return ""
    globals_in_source = set(vars(builtins)) | {
        symbol.get_name() for symbol in module.get_symbols()
        if symbol.is_assigned() or symbol.is_imported()
    }
    if _global_references(module) - globals_in_source:
        return ""
    return source


def _global_references(module):
    names = set()
    pending = [module]
    while pending:
        table = pending.pop()
        names.update(
            symbol.get_name()
            for symbol in table.get_symbols()
            if symbol.is_referenced() and symbol.is_global()
        )
        pending.extend(table.get_children())
    return names


def _lexical_references(items):
    return _global_references(symtable.symtable(
        "\n\n".join(ast.unparse(item) for item in items), "<pytest fixture assembly>", "exec",
    ))


def _prior_reference_modules(archive, ancestor_archive, roots, imports):
    names = set()
    for node in imports:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif not node.level and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    candidates = {
        name for name in names - _PRIOR_SETUP_IMPORTS
        if name.split(".", 1)[0] in roots and not _prior_sensitive(name)
        and all(part.isidentifier() for part in name.split("."))
    }
    if len(candidates) > 24:
        return set()
    try:
        with (
            tarfile.open(fileobj=BytesIO(archive), mode="r:") as current,
            tarfile.open(fileobj=BytesIO(ancestor_archive), mode="r:") as ancestor,
        ):
            allowed = set(_PRIOR_SETUP_IMPORTS)
            for name in candidates:
                module = name.replace(".", "/")
                if any(
                    _tracked_python(current, path) is not None and _tracked_python(ancestor, path) is not None
                    for prefix in ("", "src/")
                    for path in (f"{prefix}{module}.py", f"{prefix}{module}/__init__.py")
                ):
                    allowed.add(name)
            return allowed
    except (tarfile.TarError, OSError, EOFError):
        return set()


def _prior_reference_header(body, fixture_nodes, modules):
    """Accept only inert declarations and known pytest fixture registration on import."""
    for item in body:
        if isinstance(item, _IMPORT):
            if not _prior_setup_import(item, modules):
                return False
        elif isinstance(item, _FUNCTION):
            if item.type_params:
                return False
            if item.decorator_list and (item not in fixture_nodes or len(item.decorator_list) != 1):
                return False
            decorator = item.decorator_list[0] if item.decorator_list else None
            if isinstance(decorator, ast.Call) and (
                decorator.args or any(not isinstance(keyword.value, ast.Constant) for keyword in decorator.keywords)
            ):
                return False
            for expression in (*item.args.defaults, *item.args.kw_defaults):
                if expression is not None and not isinstance(expression, ast.Constant):
                    return False
            if any(
                isinstance(node, ast.Call)
                for expression in (item.args, item.returns) if expression is not None
                for node in ast.walk(expression)
            ):
                return False
        elif not (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant)):
            return False
        if _bindings(item) & {"pytest_plugins", "__file__", "__spec__", "__package__"}:
            return False
    return True


@dataclass(frozen=True)
class OperatorTestAssembly:
    declarations: tuple
    fixtures: dict
    modules: frozenset
    preview: str

    def assemble(self, response, *, max_bytes):
        if not isinstance(response, str) or not 0 < len(response.encode()) <= 6000:
            raise ValueError("Expected one complete Python test function")
        response = response.strip()
        if wrapped := re.fullmatch(r"```(?:python|py)?[ \t]*\n(.*?)\n```", response, re.S):
            response = wrapped[1]
        try:
            body = ast.parse(response).body
            if (
                len(body) == 1 and isinstance(test := body[0], ast.FunctionDef)
                and test.name.startswith("test_") and not test.decorator_list
                and any(isinstance(node, ast.Assert) for node in ast.walk(test))
            ):
                setup = _local_fixture_setup(test, self.fixtures)
                items = None if setup is None else _complete_pytest_nodes(
                    self.declarations, test, setup, references=_lexical_references,
                )
                if (
                    items and any(item is not test and isinstance(item, _FUNCTION) for item in items)
                    and _prior_reference_header(items, set(self.fixtures.values()), self.modules)
                    and (source := _prior_setup_source(
                        items, self.fixtures, self.modules, sample=test, fixture_placeholders=True,
                    ))
                    and len(source.encode()) < max_bytes
                ):
                    return source + "\n"
        except (SyntaxError, ValueError, RecursionError):
            pass
        raise ValueError("Expected one new test function with an assertion and approved fixture references")


def operator_test_assembly(fixtures, archive, ancestor_archive, sources, *, max_bytes=6200):
    """Select one bounded, closed setup API from authorized operator pytest source."""
    roots = {name.split(".", 1)[0] for name in _reviewed_modules(sources)}
    for source in tuple(fixtures.values())[:4]:
        if len(source.encode()) > 128 * 1024:
            continue
        try:
            body = ast.parse(source).body
            imports, _ = _reviewed_imports(body, {})
            local, fixture_nodes, _ = _module_pytest_fixtures(body, imports)
            helpers = [
                item for item in body if isinstance(item, _FUNCTION) and not item.name.startswith("test_")
            ][:24]
            if not helpers:
                continue
            modules = frozenset(_prior_reference_modules(
                archive, ancestor_archive, roots,
                [item for item in body if isinstance(item, _IMPORT)]
                + [node for helper in helpers for node in ast.walk(helper) if isinstance(node, _IMPORT)],
            ))
            approved = {item for item in body if isinstance(item, _IMPORT) and _prior_setup_import(item, modules)}
            for helper in helpers:
                dependencies = _local_fixture_setup(helper, local) if helper in fixture_nodes else ()
                if dependencies is None or not (items := _complete_pytest_nodes(
                    body, helper, dependencies, references=_lexical_references,
                )):
                    continue
                selected = sorted(approved.union(items), key=lambda item: item.lineno)
                if (
                    _prior_reference_header(selected, fixture_nodes, modules)
                    and (preview := _prior_setup_source(
                        selected, local, modules, fixture_placeholders=True,
                    ))
                    and len(preview.encode()) <= max_bytes
                ):
                    approved.update(items)
            declarations = tuple(sorted(approved, key=lambda item: item.lineno))
            if any(isinstance(item, _FUNCTION) for item in declarations):
                return OperatorTestAssembly(
                    declarations, {name: item for name, item in local.items() if item in approved},
                    modules, "\n\n".join(ast.unparse(item) for item in declarations),
                )
        except (SyntaxError, ValueError, RecursionError):
            continue
    return None


def _prior_reference_seed(archive, ancestor_archive, path, body, items, test, fixtures, fixture_nodes, modules):
    definitions = sorted(item.name for item in items if isinstance(item, _FUNCTION) and item is not test)
    if not definitions:
        return ""
    imports = {
        item for item in body if isinstance(item, _IMPORT)
    } | {
        node for item in items for node in ast.walk(item) if isinstance(node, _IMPORT)
    }
    roots = {name.split(".", 1)[0] for name in modules}
    allowed = _prior_reference_modules(archive, ancestor_archive, roots, imports)
    if not _prior_reference_header(body, fixture_nodes, allowed) or not _prior_setup_source(
        items, fixtures, allowed, sample=test, reference_helpers=True,
    ):
        return ""
    used = _references((test,), include_attributes=False)
    borrowed_names = {"_gha_find_spec", "_GhaPriorPath", "_gha_prior_spec"}
    if borrowed_names & (used | set(_rebound_names(test))):
        return ""
    stem = PurePosixPath(path).stem
    filename = PurePosixPath(path).name
    header = (
        "# These named fixtures are in the unchanged sibling present on both revisions.\n"
        "# Keep the sibling check when adapting their setup to the new test.\n"
        "from importlib.util import find_spec as _gha_find_spec\n"
        "from pathlib import Path as _GhaPriorPath\n"
        f"_gha_prior_spec = _gha_find_spec({stem!r})\n"
        "if (_gha_prior_spec is None or _gha_prior_spec.origin is None\n"
        "        or _GhaPriorPath(_gha_prior_spec.origin).resolve()\n"
        f"        != _GhaPriorPath(__file__).with_name({filename!r}).resolve()):\n"
        "    raise ImportError('Expected the pinned sibling pytest module')\n"
        f"from {stem} import {', '.join(definitions)}"
    )
    return "\n\n".join((
        header,
        *(ast.unparse(item) for item in items if isinstance(item, _IMPORT) and _bindings(item) & used),
        ast.unparse(test),
    ))


def _prior_generated_module(archive, ancestor_archive, new_path, require_identical):
    match = re.fullmatch(r"tests/test_review_regressions_([1-9]\d?)\.py", new_path)
    if not match or not 2 <= int(match[1]) <= 32:
        return None
    previous = int(match[1]) - 1
    suffix = f"_{previous}" if previous > 1 else ""
    path = f"tests/test_review_regressions{suffix}.py"
    try:
        with tarfile.open(fileobj=BytesIO(ancestor_archive), mode="r:") as ancestor:
            if require_identical:
                pinned = _tracked_python(ancestor, path)
                if pinned is None:
                    return None
            elif path in ancestor.getnames():
                return None
        with tarfile.open(fileobj=BytesIO(archive), mode="r:") as snapshot:
            tracked = _tracked_python(snapshot, path)
    except (tarfile.TarError, OSError, EOFError):
        return None
    if tracked is None:
        return None
    # The decoded UTF-8 lines retain every newline and all whitespace.
    if require_identical and tracked[0] != pinned[0]:
        return None
    return path, tracked[1]


def prior_generated_fixture_context(
    archive, ancestor_archive, new_path, sources, *, max_bytes=1700, require_identical=False,
):
    """Quote setup from the immediate generated sibling under the selected ancestor policy."""
    if not (previous := _prior_generated_module(archive, ancestor_archive, new_path, require_identical)):
        return ""
    path, body = previous
    imports, _ = _reviewed_imports(body, {})
    fixtures, fixture_nodes, _ = _module_pytest_fixtures(body, imports)
    factories = {
        item.name: item for item in body if isinstance(item, _FUNCTION)
        and (
            re.match(r"^_?(?:make|build|create|prepare|setup|sample|fake|fixture|write|seed|mock)_", item.name)
            or (
                "tmp_path" in _fixture_arguments(item)
                and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and call.func.attr in {"write_text", "write_bytes"} for call in ast.walk(item))
            )
        )
        and any(isinstance(statement, ast.Return) for statement in item.body)
    }
    members = [node for item in body if isinstance(item, ast.ClassDef) and item.name.startswith("Test")
               for node in item.body]
    seeds = set()
    for node in (*body, *members):
        if not isinstance(node, _FUNCTION) or not node.name.startswith("test_"):
            continue
        seeds.update(_local_fixture_setup(node, fixtures) or ())
        bound = set(_rebound_names(node)) | _fixture_arguments(node)
        seeds.update(
            factories[name] for call in ast.walk(node) if isinstance(call, ast.Call)
            and (name := _dotted_name(call.func)) in factories and name not in bound
        )
    heading = (
        f"# Prior generated pytest setup from {path}. Adapt it to the current review.\n"
        "# Keep the new module self-contained for the pinned ancestor comparison.\n"
    )
    modules = _PRIOR_SETUP_IMPORTS | _reviewed_modules(sources).keys()
    kept = set()
    result = ""
    for item in sorted(seeds, key=lambda node: node.lineno)[:32]:
        dependencies = _local_fixture_setup(item, fixtures) if item in fixture_nodes else ()
        if dependencies is None or not (selected := _complete_pytest_nodes(body, item, dependencies)):
            continue
        candidates = sorted(kept.union(selected), key=lambda node: node.lineno)
        if (source := _prior_setup_source(candidates, fixtures, modules)) and len((heading + source).encode()) <= max_bytes:
            kept.update(selected)
            result = heading + source
    return result


def prior_generated_test_seed(archive, ancestor_archive, new_path, sources, query, *, max_bytes=2500):
    """Project one related, self-contained test from the unchanged predecessor for verification."""
    if not (previous := _prior_generated_module(archive, ancestor_archive, new_path, True)):
        return None
    path, body = previous
    modules = _reviewed_modules(sources)
    imports, reviewed = _reviewed_imports(body, modules)
    fixtures, fixture_nodes, autouse = _module_pytest_fixtures(body, imports)
    query_words = _words(query.replace("_", " "))
    best, rank = None, (0, 0)
    borrowed = []
    for test in body:
        if (
            not isinstance(test, ast.FunctionDef) or not test.name.startswith("test_")
            or test.decorator_list or not any(isinstance(node, ast.Assert) for node in ast.walk(test))
        ):
            continue
        bound = set(_rebound_names(test)) | _fixture_arguments(test)
        calls = {
            name for node in ast.walk(test) if isinstance(node, ast.Call)
            and (name := _dotted_name(node.func)) in reviewed and name.split(".", 1)[0] not in bound
        }
        calls.update(symbol for symbol, rows in _usage_shapes(test, reviewed) if rows[-1].endswith(")"))
        matches = _usage_relevance(" ".join((test.name, *sorted(calls))), query_words)
        if not calls or not matches or (matches, len(calls)) <= rank:
            continue
        setup = _local_fixture_setup(test, fixtures, autouse=autouse)
        if setup is None or not (items := _complete_pytest_nodes(body, test, setup)):
            continue
        source = _prior_setup_source(items, fixtures, _PRIOR_SETUP_IMPORTS | modules.keys(), sample=test)
        if source and len(source.encode()) < max_bytes:
            best, rank = (path, test.name, source + "\n"), (matches, len(calls))
        elif not source:
            borrowed.append(((matches, len(calls)), items, test))
    for candidate_rank, items, test in sorted(borrowed, key=lambda item: item[0], reverse=True)[:4]:
        if candidate_rank <= rank:
            break
        source = _prior_reference_seed(
            archive, ancestor_archive, path, body, items, test, fixtures, fixture_nodes, modules,
        )
        if source and len(source.encode()) < max_bytes:
            return path, test.name, source + "\n"
    return best


def _invoked_anchored_apis(node, setup, imports, fixtures, preferred_apis):
    """Resolve exact imported calls and receivers built by the example's local fixtures."""
    if not preferred_apis:
        return set()
    owners = {name.rpartition(".")[0] for name in preferred_apis}
    aliases = {}
    for name, item in imports.items():
        for alias in item.names:
            if isinstance(item, ast.Import) and name == (alias.asname or alias.name.split(".")[0]):
                aliases[name] = alias.name if alias.asname else name
            elif isinstance(item, ast.ImportFrom) and name == (alias.asname or alias.name):
                aliases[name] = f"{item.module}.{alias.name}"
    hits = set()

    def walk(function, known_fixtures):
        values = {name: known_fixtures[name] for name in _fixture_arguments(function) & known_fixtures.keys()}
        shadowed = set(_rebound_names(function)) | _fixture_arguments(function)

        def qualify(expression):
            root, dot, tail = _dotted_name(expression).partition(".")
            return aliases.get(root, "") + dot + tail if root not in shadowed and root in aliases else ""

        def origin(expression):
            if isinstance(expression, ast.Name):
                return values.get(expression.id, "")
            if isinstance(expression, ast.Call):
                name = qualify(expression.func)
                return name if name in owners else ""
            return ""

        returns = []
        for statement in function.body:
            for child in ast.walk(statement):
                if isinstance(child, ast.Call):
                    name = qualify(child.func)
                    if isinstance(child.func, ast.Attribute) and (receiver := origin(child.func.value)):
                        name = f"{receiver}.{child.func.attr}"
                    if name in preferred_apis:
                        hits.add(name)
            if isinstance(statement, ast.Return):
                returns.append(origin(statement.value))
            else:
                targets = tuple(_rebound_names(statement))
                expression = getattr(statement, "value", None)
                assigned = (
                    origin(expression)
                    if len(targets) == 1 and isinstance(statement, _ASSIGNMENT) else ""
                )
                for target in targets:
                    values.pop(target, None)
                    if assigned:
                        values[target] = assigned
        return returns[0] if len(returns) == 1 else ""

    origins = {}
    relevant = {name: fixture for name, fixture in fixtures.items() if fixture in setup}
    for _ in range(min(4, len(relevant))):
        proved = {name: kind for name, fixture in relevant.items() if (kind := walk(fixture, origins))}
        if proved == origins:
            break
        origins = proved
    walk(node, origins)
    return hits


def repository_pytest_context(
    archive, sources, query="", *, max_bytes=850, usage_bytes=820, ranked=False,
    preferred_apis=(), exclude_generated=False,
):
    """Return a small example plus bounded tracked API and fixture setup shapes."""
    modules = _reviewed_modules(sources)
    if not any(modules.values()):
        return "", ""
    module_stems = {module.rsplit(".", 1)[-1] for module in modules}
    query_words = _words(query.replace("_", " "))
    example = ""
    example_rank = (-1, -1, -1)
    shapes = []
    setup_usage = ()
    setup_rank = (-1, False)

    try:
        with tarfile.open(fileobj=BytesIO(archive), mode="r:") as snapshot:
            candidates = []
            for member in snapshot:
                path = PurePosixPath(member.name)
                if (
                    member.type in (tarfile.REGTYPE, tarfile.AREGTYPE)
                    and member.size <= 128 * 1024
                    and _STOCK_NODE.fullmatch(member.name + "::test_example")
                    and (not ranked or _extended_context_path(path))
                    and (not exclude_generated or not _GENERATED_TEST_STEM.fullmatch(path.stem))
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
                fixtures, fixture_nodes, autouse = _module_pytest_fixtures(body, imports)
                unambiguous_fixtures = set(fixtures.values()) - {None}
                if usage_bytes:
                    module_bindings = _mock_bindings(body)
                    patches = _mock_patch_imports(body, imports)
                members = tuple(
                    node
                    for declaration in body
                    if isinstance(declaration, ast.ClassDef) and declaration.name.startswith("Test")
                    for node in declaration.body
                    if isinstance(node, _FUNCTION) and node.name.startswith("test_")
                )
                for node in (*body, *members):
                    if not isinstance(node, _FUNCTION) or not (
                        node.name.startswith("test_") or node in fixture_nodes
                    ):
                        continue
                    if usage_bytes:
                        grounded = tuple(_usage_shapes(node, reviewed_calls))
                        for symbol, rows in grounded:
                            matches = _usage_relevance(symbol, query_words)
                            shapes.append((matches, "." in symbol or "[" in symbol, path, rows))
                        invocations = [(symbol, rows) for symbol, rows in grounded if rows[-1].endswith(")")]
                        if invocations and node.name.startswith("test_"):
                            symbol, invocation = max(invocations, key=lambda entry: (
                                _usage_relevance(entry[0], query_words), "." in entry[0]
                            ))
                            setup = _local_fixture_setup(node, fixtures)
                            if setup is not None:
                                mock_rows = _mock_setup_shapes(node, setup, module_bindings, patches, fixtures.keys())
                                file_rows = tuple(_temporary_file_shapes(setup, module_bindings, imports, fixtures.keys()))
                                variants = [
                                    *(("mock and temporary-file", (*mock_rows, *rows)) for rows in file_rows if mock_rows),
                                    ("mock", mock_rows),
                                    *(("temporary-file", rows) for rows in file_rows),
                                ]
                                for kind, rows in variants:
                                    matched = _usage_relevance(" ".join((symbol, node.name, *rows)), query_words)
                                    rank = matched, "." in symbol
                                    context = (f"# {path}::{node.name} tracked {kind} setup and API usage", *rows, *invocation)
                                    if (
                                        rows and rank > setup_rank
                                        and len("\n".join(context).encode()) + 1 <= usage_bytes
                                    ):
                                        setup_rank = rank
                                        setup_usage = context
                    if (example and not ranked) or node in members:
                        continue
                    if ranked and not node.name.startswith("test_"):
                        continue
                    setup = _local_fixture_setup(node, fixtures, autouse=autouse if ranked else ())
                    if setup is None:
                        continue
                    called = {
                        name for item in ({node, *setup} if ranked else {node})
                        for call in ast.walk(item) if isinstance(call, ast.Call)
                        if (name := _dotted_name(call.func)) in reviewed_calls
                    }
                    anchored_calls = _invoked_anchored_apis(node, setup, imports, fixtures, preferred_apis)
                    if not (called or anchored_calls):
                        continue
                    if node in fixture_nodes and node not in unambiguous_fixtures:
                        continue
                    references = _references([node, *setup], include_attributes=False)
                    selected = {imports[name] for name in references & imports.keys()}
                    items = (
                        *sorted(selected, key=lambda item: item.lineno),
                        *sorted(setup - {node}, key=lambda item: item.lineno),
                        node,
                    )
                    if ranked:
                        grounded = [symbol for symbol, _ in _usage_shapes(node, reviewed_calls)]
                        rank = (
                            len(anchored_calls),
                            _usage_relevance(" ".join((*sorted(called), *grounded, node.name)), query_words),
                            len(setup),
                        )
                        if rank <= example_rank or not (items := _complete_pytest_nodes(body, node, setup)):
                            continue
                    excerpt = f"# {path}\n"
                    if PurePosixPath(path).name != "conftest.py" and (setup or node in fixture_nodes):
                        excerpt += "# Module-local pytest fixture setup. Define it explicitly if reusing it.\n"
                    excerpt += "".join(
                        "".join(lines[_start(item) - 1 : item.end_lineno])
                        for item in items
                    )
                    if len(excerpt.encode()) <= max_bytes:
                        example = excerpt.rstrip()
                        if ranked:
                            example_rank = rank
                        elif not usage_bytes:
                            return example, ""
    except (tarfile.TarError, OSError, EOFError):
        pass
    usage = ["\n".join(setup_usage)] if setup_usage else []
    shown = set(setup_usage)
    if setup_usage:
        usage_bytes -= len(usage[0].encode()) + 1
    for _, _, path, rows in sorted(shapes, key=lambda item: item[:2], reverse=True):
        additions = [row for row in rows if row not in shown]
        if not additions:
            continue
        chunk = "\n".join((f"# {path}", *additions))
        if len(chunk.encode()) + 1 <= usage_bytes:
            usage.append(chunk)
            shown.update(additions)
            usage_bytes -= len(chunk.encode()) + 1
        if len(usage) >= 4:
            break
    return example, "\n".join(usage)


def repository_pytest_example(archive, sources, *, max_bytes=850):
    example, _ = repository_pytest_context(archive, sources, max_bytes=max_bytes, usage_bytes=0)
    return example


def _direct_review_caller(name, node, anchored):
    owner, _, _ = name.rpartition(".")
    receivers = node.args.posonlyargs + node.args.args
    receiver = receivers[0].arg if receivers and receivers[0].arg in {"self", "cls"} else ""
    targets = {
        (f"{receiver}.{leaf}" if owner else leaf)
        for symbol in anchored
        for enclosing, _, leaf in [symbol.rpartition(".")]
        if enclosing == owner and (not owner or receiver)
    }
    return any(
        _dotted_name(call.func) in targets
        for statement in node.body for call in ast.walk(statement)
        if isinstance(call, ast.Call)
    )


def extended_pytest_context(archive, sources, feedback, *, max_bytes=16 * 1024):
    """Quote a tracked example and exact reviewed functions as optional context."""
    sources = {
        path: source for path, source in sources.items()
        if _extended_context_path(PurePosixPath(path))
    }
    query = " ".join(body for _, _, body in feedback)
    query_words = _words(query)
    definitions = []
    preferred_apis = set()
    for path, source in sources.items():
        try:
            module = ast.parse(source)
        except (SyntaxError, ValueError, RecursionError):
            continue
        lines = source.splitlines(keepends=True)
        anchors = {row for reviewed_path, row, _ in feedback if reviewed_path == path}
        functions = [(name, node) for name, node in _symbols(module.body) if isinstance(node, _FUNCTION)]
        anchored = {
            name for name, node in functions if any(_start(node) <= row <= node.end_lineno for row in anchors)
        }
        for name, node in functions:
            start = _start(node)
            public = not any(part.startswith("_") for part in name.split("."))
            caller = public and _direct_review_caller(name, node, anchored)
            if public and (name in anchored or caller) and (namespace := _python_module(path)):
                preferred_apis.add(f"{namespace}.{name}")
            relevance = _usage_relevance(name, query_words)
            if name in anchored or caller or relevance:
                code = f"# {path}::{name}\n" + "".join(lines[start - 1 : node.end_lineno])
                definitions.append((name in anchored, caller, relevance, code.rstrip()))
    example, _ = repository_pytest_context(
        archive, sources, query, max_bytes=2 * max_bytes // 3, usage_bytes=0,
        ranked=True, preferred_apis=preferred_apis,
    )
    sections = [example] if example else []
    remaining = max_bytes - len(example.encode())
    ranked = sorted(definitions, reverse=True)
    selected = [item for item in ranked if item[0] or item[1]] or ranked[:1]
    for _, _, _, code in selected[:3]:
        size = len(code.encode()) + 2
        if size <= remaining:
            sections.append(code)
            remaining -= size
    return "\n\n".join(sections)


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


def _references(nodes, *, include_attributes=True):
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
        if (name := getattr(child, "id", None) or (include_attributes and getattr(child, "attr", None)))
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

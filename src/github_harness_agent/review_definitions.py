"""Apply bounded, atomic replacements of named existing Python definitions."""

import ast
import json
import re
import textwrap
from collections import defaultdict
from copy import deepcopy
from pathlib import PurePosixPath

CAUSAL_EDIT_GUIDANCE = "Reason through the observed passing and failing executions before producing the scoped edit."
DEFINITION_ENDING = "Respond with complete named Python definitions only."
LOCALIZED_METHOD_HEADING = "Exact current method localized by the offline replay:"
DEFINITION_FORMAT = """Use only definition edit blocks. Begin each with an exact header:
REPLACE relative/path.py::Class.method
Then include the complete replacement definition, optionally in a Python code fence.
Use the exact dotted lexical name for a nested symbol. Keep the same name and kind.
Optionally add imports separately with IMPORTS relative/path.py and raw import statements.
Both methods and functions may be unindented. Keep unrelated code and preserve behavior."""

NATIVE_DEFINITION_TOOL = {
    "type": "function",
    "function": {
        "name": "replace_python_definition",
        "description": "Replace one existing Python function or method by exact path and qualified name.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Exact editable relative Python path.",
                },
                "symbol": {
                    "type": "string",
                    "description": "Exact dotted lexical name, such as function, Class.method, or Class.method.inner.",
                },
                "definition": {
                    "type": "string",
                    "description": "Complete replacement Python definition.",
                },
            },
            "required": ["path", "symbol", "definition"],
            "additionalProperties": False,
        },
    },
}

_FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)
_DEF = (*_FUNCTION, ast.ClassDef)
_IMPORT = (ast.Import, ast.ImportFrom)
_LIMIT = 256 * 1024
_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_HEADER = re.compile(
    rf"^(?:(REPLACE) (\S+\.py)::({_IDENT}(?:\.{_IDENT})*)|(IMPORTS) (\S+\.py))[ \t]*$",
    re.MULTILINE,
)
_FENCE = re.compile(r"^```(?:python|py|text)?[ \t]*\n(.*?)\n```[ \t]*$", re.S | re.M)


def _unfence(text):
    stripped = textwrap.dedent(text.strip("\n")).rstrip()
    match = _FENCE.fullmatch(stripped)
    return textwrap.dedent(match[1]).rstrip() if match else stripped


def _parse(text, label):
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        raise ValueError(f"{label}: complete valid Python is required") from None


def _symbols(body, prefix="", *, nested=False, _local=False):
    for node in body:
        if isinstance(node, _DEF):
            qualified = prefix + node.name
            yield qualified, node
            if nested or isinstance(node, ast.ClassDef):
                yield from _symbols(
                    node.body, qualified + ".", nested=nested,
                    _local=_local or isinstance(node, _FUNCTION),
                )
        elif _local and isinstance(node, (ast.stmt, ast.ExceptHandler, ast.match_case)):
            yield from _symbols(ast.iter_child_nodes(node), prefix, nested=nested, _local=True)


def _definition_index(module):
    index = defaultdict(list)
    for name, node in _symbols(module.body, nested=True):
        index[name].append(node)
    return index


def _qualified_scope(index, name):
    parts = name.split(".")
    scopes = []
    for level in range(1, len(parts) + 1):
        matches = index.get(".".join(parts[:level]), ())
        if len(matches) != 1:
            return ()
        scopes.append(matches[0])
    return scopes


def _definition_start(node):
    return min([node.lineno, *(entry.lineno for entry in node.decorator_list)])


def visible_nested_definitions(source, shown, anchor):
    """Find exact nested functions already shown inside the reviewed function."""
    try:
        index = _definition_index(_parse(source, "Reviewed source"))
    except ValueError:
        return ()
    lines = source.splitlines()
    visible = "\n" + shown.replace("\r\n", "\n") + "\n"
    names = []
    for name in index:
        scopes = _qualified_scope(index, name)
        if not scopes or not isinstance(scopes[-1], _FUNCTION):
            continue
        if not any(
            isinstance(owner, _FUNCTION) and _definition_start(owner) <= anchor <= owner.end_lineno
            for owner in scopes[:-1]
        ):
            continue
        node = scopes[-1]
        snippet = "\n".join(lines[_definition_start(node) - 1:node.end_lineno])
        if f"\n{snippet}\n" in visible and len(name.encode()) <= 120:
            names.append(name)
        if len(names) == 3:
            break
    return tuple(names)


def reviewed_nested_helper(source, anchor):
    """Return one uniquely shown nested helper and calls in its short reviewed owner."""
    try:
        module = _parse(source, "Reviewed source")
    except ValueError:
        return None
    owners = [
        (name, node) for name, node in _symbols(module.body)
        if isinstance(node, _FUNCTION) and _definition_start(node) <= anchor <= node.end_lineno
    ]
    if len(owners) != 1:
        return None
    owner_name, owner = owners[0]
    lines = source.splitlines()
    shown = "\n".join(lines[_definition_start(owner) - 1:owner.end_lineno])
    if owner.end_lineno - _definition_start(owner) >= 90 or len(shown.encode()) > 3600:
        return None
    nested = [node for node in ast.walk(owner) if node is not owner and isinstance(node, _FUNCTION)]
    if len(nested) != 1:
        return None
    helper = nested[0]
    name = f"{owner_name}.{helper.name}"
    if visible_nested_definitions(source, shown, anchor) != (name,):
        return None
    calls = [
        ast.get_source_segment(source, call)
        for call in ast.walk(owner)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == helper.name
        and owner.body[0].lineno <= call.lineno
        and not _definition_start(helper) <= call.lineno <= helper.end_lineno
    ]
    if not calls or len(calls) > 3 or not all(calls) or sum(len(call.encode()) for call in calls) > 900:
        return None
    body = "\n".join(lines[_definition_start(helper) - 1:helper.end_lineno])
    return name, textwrap.dedent(body), tuple(calls)


def apply_scoped_raw_definition(response, sources, symbol):
    """Bind plain Python only to a previously selected, existing nested function."""
    path, name = symbol
    if not isinstance(response, str) or len(response.encode()) > 4096 or path not in sources:
        raise ValueError("Expected one complete Python helper within the edit limit")
    body = _unfence(response.replace("\r\n", "\n"))
    nodes = _parse(body, path).body
    scopes = _qualified_scope(_definition_index(_parse(sources[path], path)), name)
    if (
        not scopes or not any(isinstance(owner, _FUNCTION) for owner in scopes[:-1])
        or len(nodes) != 1 or not isinstance(nodes[0], _FUNCTION)
        or _definition_start(nodes[0]) != 1 or nodes[0].end_lineno != len(body.splitlines())
    ):
        raise ValueError("Expected only the selected complete nested Python function")
    return apply_definitions(
        f"REPLACE {path}::{name}\n{body}", sources, only=symbol, single_function=True,
    )


def _imports(nodes):
    if not nodes or any(
        not isinstance(node, _IMPORT)
        or (isinstance(node, ast.ImportFrom) and node.module == "__future__")
        for node in nodes
    ):
        raise ValueError("IMPORTS must contain ordinary import statements only")
    return nodes


def _edits(text, documents):
    matches = list(_HEADER.finditer(text))
    if matches:
        if text[: matches[0].start()].strip():
            raise ValueError(
                "Start the response with the exact REPLACE or IMPORTS header"
            )
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = _unfence(text[match.end() : end])
            replacement, path, name, _, import_path = match.groups()
            if replacement:
                yield path, name, body
            else:
                yield import_path, None, body
        return

    # A free provider may output a usable bare definition instead of a header.
    if text.count("```") == 2 and (fence := _FENCE.search(text)):
        text = textwrap.dedent(fence[1]).rstrip()
    nodes = _parse(text, "Unlabeled response").body
    if len(nodes) != 1 or not isinstance(nodes[0], _DEF):
        raise ValueError("Return a single definition or use qualified REPLACE headers")
    node = nodes[0]
    owners = [
        (path, name)
        for path, module in documents.items()
        for name, old in _symbols(module.body)
        if type(old) is type(node) and old.name == node.name
    ]
    if len(owners) != 1:
        raise ValueError(
            "Unlabeled definition is missing or ambiguous; use a qualified header"
        )
    yield (*owners[0], text)


def _import_index(module):
    body = module.body
    if ast.get_docstring(module) is not None:
        index = body[0].end_lineno
        body = body[1:]
    else:
        # Keep shebang, encoding cookies and comments above the first statement.
        first = body[0]
        index = (
            min(
                [
                    first.lineno,
                    *(d.lineno for d in getattr(first, "decorator_list", ())),
                ]
            )
            - 1
        )
    for node in body:
        if not isinstance(node, _IMPORT):
            break
        index = node.end_lineno
    return index


def _documents(sources):
    documents = {}
    for path, source in sources.items():
        parts = PurePosixPath(path)
        if (
            source
            and not parts.is_absolute()
            and ".." not in parts.parts
            and parts.suffix == ".py"
        ):
            module = _parse(source, path)
            if module.body:
                documents[path] = module
    return documents


def apply_definitions(response, sources, *, only=None, single_function=False):
    """Return a fresh source mapping or reject the proposal without partial edits."""
    if (
        not isinstance(response, str)
        or not response.strip()
        or len(response.encode()) > 80_000
    ):
        raise ValueError("Definition response is empty or exceeds the edit limit")
    text = _unfence(response.replace("\r\n", "\n"))
    if single_function and not text.startswith("REPLACE "):
        raise ValueError("Start with a qualified REPLACE header")
    documents = _documents(sources)
    replacements = _edits(text, documents)
    if only is not None or single_function:
        replacements = tuple(replacements)
        if only is not None and (len(replacements) != 1 or replacements[0][:2] != only):
            raise ValueError("Expected one replacement of the localized definition")
        if single_function and (len(replacements) != 1 or not replacements[0][1]):
            raise ValueError("Expected one complete replacement function or method")
    return _apply_definition_edits(
        sources, documents, replacements, functions_only=single_function
    )


def _unique_object(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("Duplicate native edit argument")
    return value


def apply_native_definition(calls, sources):
    """Validate a single native function call against the supplied source snapshot."""
    try:
        if len(calls) != 1:
            raise ValueError
        function = calls[0]["function"]
        if (
            calls[0]["type"] != "function"
            or function["name"] != "replace_python_definition"
        ):
            raise ValueError
        arguments = function["arguments"]
        if not isinstance(arguments, str) or len(arguments.encode()) > 16_384:
            raise ValueError
        edit = json.loads(arguments, object_pairs_hook=_unique_object)
        if type(edit) is not dict or edit.keys() != {"path", "symbol", "definition"}:
            raise ValueError
        if not all(isinstance(value, str) for value in edit.values()):
            raise ValueError
        path, symbol, body = (edit[key] for key in ("path", "symbol", "definition"))
        if (
            not _HEADER.fullmatch(f"REPLACE {path}::{symbol}")
            or len(body.encode()) > 10_240
        ):
            raise ValueError
    except (IndexError, KeyError, TypeError, ValueError, RecursionError):
        raise ValueError("Expected one valid native definition replacement") from None

    body = _unfence(body.replace("\r\n", "\n"))
    documents = _documents(sources)
    current = documents.get(path)
    scopes = _qualified_scope(_definition_index(current), symbol) if current else ()
    if not scopes or not isinstance(scopes[-1], _FUNCTION):
        raise ValueError(
            "Native edit must name an existing function in the editable source list"
        )
    return _apply_definition_edits(sources, documents, ((path, symbol, body),))


def _apply_definition_edits(sources, documents, replacements, *, functions_only=False):
    edits = defaultdict(list)
    added_imports = defaultdict(dict)
    indexes = {}
    nested_paths = set()
    spliced_bodies = defaultdict(dict)
    for path, name, body in replacements:
        if path not in documents:
            raise ValueError("File is outside the editable Python source list")
        module = documents[path]
        nodes = _parse(body, path).body
        if name is None:
            existing = {
                ast.dump(node) for node in module.body if isinstance(node, _IMPORT)
            }
            for node in _imports(nodes):
                identity = ast.dump(node)
                if identity not in existing:
                    added_imports[path][identity] = (
                        ast.get_source_segment(body, node) + "\n"
                    )
            continue

        if path not in indexes:
            indexes[path] = _definition_index(module)
        scopes = _qualified_scope(indexes[path], name)
        if not scopes:
            raise ValueError(
                "Qualified definition is missing or ambiguous in the named file"
            )
        old = scopes[-1]
        is_nested = any(isinstance(owner, _FUNCTION) for owner in scopes[:-1])
        if functions_only and len(nodes) > 1:
            # Some providers repeat imports before a single requested test.
            # Treat only imports already present in the module as context.
            existing_imports = {ast.dump(node) for node in module.body if isinstance(node, _IMPORT)}
            if all(isinstance(node, _IMPORT) and ast.dump(node) in existing_imports for node in nodes[:-1]):
                definition = nodes[-1]
                if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = min([definition.lineno, *(item.lineno for item in definition.decorator_list)])
                    body = "\n".join(body.splitlines()[start - 1:definition.end_lineno])
                    nodes = _parse(body, path).body
        if (
            len(nodes) != 1
            or type(nodes[0]) is not type(old)
            or nodes[0].name != old.name
        ):
            raise ValueError(
                "Replacement must be one complete definition with the same name and kind"
            )
        if (functions_only or is_nested) and not isinstance(old, _FUNCTION):
            raise ValueError("Expected a function or method, including for class-based pytest tests")
        lines = sources[path].splitlines(keepends=True)
        if is_nested:
            if "\f" in body or "\r" in body:
                raise ValueError("Nested replacement requires valid indentation")
            nested_paths.add(path)
            new = nodes[0]
            if _nested_signature(old) != _nested_signature(new):
                if not _only_omitted_annotations(old, new):
                    raise ValueError("Nested replacement must preserve its signature and decorators")
                start, suite = _splice_nested_body(lines, old, body, new)
                edits[path].append((start, old.end_lineno, suite))
                spliced_bodies[path][name] = _definition_field(new, "body")
                continue
        start = _definition_start(old) - 1
        margin = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
        edits[path].append(
            (start, old.end_lineno, textwrap.indent(body + "\n", margin))
        )

    for path, imports in added_imports.items():
        index = _import_index(documents[path])
        edits[path].append((index, index, "".join(imports.values())))

    result = dict(sources)
    for path, replacements in edits.items():
        lines = sources[path].splitlines(keepends=True)
        output = []
        cursor = 0
        for start, end, replacement in sorted(replacements):
            if start < cursor:
                raise ValueError("Definition replacements overlap")
            output.extend((*lines[cursor:start], replacement))
            cursor = end
        revised = "".join((*output, *lines[cursor:]))
        if len(revised.encode()) > _LIMIT:
            raise ValueError("Edited file exceeds the source limit")
        parsed = _parse(revised, path)
        if path in spliced_bodies:
            index = _definition_index(parsed)
            for name, expected in spliced_bodies[path].items():
                scopes = _qualified_scope(index, name)
                if not scopes or _definition_field(scopes[-1], "body") != expected:
                    raise ValueError("Nested replacement body cannot be spliced into the original scope")
        if path in nested_paths:
            try:
                compile(revised, path, "exec", dont_inherit=True)
            except (SyntaxError, ValueError, RecursionError):
                raise ValueError("Nested replacement must produce valid Python in its enclosing scope") from None
        result[path] = revised
    if result == sources:
        raise ValueError("No definition replacements changed the checkout")
    return result


def _definition_field(node, field):
    value = getattr(node, field, None)
    if isinstance(value, list):
        return tuple(ast.dump(item) for item in value)
    return ast.dump(value) if value is not None else None


def _nested_signature(node):
    return tuple(
        _definition_field(node, field) for field in ("args", "returns", "decorator_list", "type_params")
    )


def _only_omitted_annotations(old, new):
    filled = deepcopy(new.args)
    old_args = (node for node in ast.iter_child_nodes(old.args) if isinstance(node, ast.arg))
    new_args = (node for node in ast.iter_child_nodes(filled) if isinstance(node, ast.arg))
    for original, proposed in zip(old_args, new_args):
        if proposed.annotation is None:
            proposed.annotation = original.annotation
    return (
        ast.dump(old.args) == ast.dump(filled)
        and (new.returns is None or _definition_field(old, "returns") == _definition_field(new, "returns"))
        and all(
            _definition_field(old, field) == _definition_field(new, field)
            for field in ("decorator_list", "type_params")
        )
    )


def _suite_start(lines, node):
    first = node.body[0]
    if isinstance(first, _DEF) and first.decorator_list:
        row = _definition_start(first) - 1
        return row, lines[row][:len(lines[row]) - len(lines[row].lstrip(" \t"))]
    row = first.lineno - 1
    return row, lines[row].encode()[:first.col_offset].decode()


def _splice_nested_body(lines, old, reply, new):
    """Keep the old declaration and move the proposed suite into its indentation."""
    start, prefix = _suite_start(lines, old)
    replacement = reply.splitlines(keepends=True)
    proposed_start, proposed_prefix = _suite_start(replacement, new)
    if proposed_prefix.strip():
        replacement[proposed_start] = replacement[proposed_start][len(proposed_prefix):]
    suite = textwrap.dedent("".join(replacement[proposed_start:])).rstrip("\n")
    if prefix.strip():
        if "\n" not in suite:
            suite = prefix + suite
        else:
            declaration = lines[old.lineno - 1]
            margin = declaration[:len(declaration) - len(declaration.lstrip(" \t"))]
            indent = margin + ("\t" if margin.endswith("\t") else "    ")
            suite = prefix.rstrip(" \t") + "\n" + textwrap.indent(suite, indent)
    else:
        suite = textwrap.indent(suite, prefix)
    newline = "\r\n" if lines[start].endswith("\r\n") else "\n"
    return start, suite.replace("\n", newline) + newline

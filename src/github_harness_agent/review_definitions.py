"""Apply bounded, atomic replacements of named existing Python definitions."""

import ast
import json
import re
import textwrap
from collections import defaultdict
from pathlib import PurePosixPath

DEFINITION_ENDING = "Respond with complete named Python definitions only."
LOCALIZED_METHOD_HEADING = "Exact current method localized by the offline replay:"
DEFINITION_FORMAT = """Use only definition edit blocks. Begin each with an exact header:
REPLACE relative/path.py::Class.method
Then include the complete replacement definition, optionally in a Python code fence.
For a module-level symbol, use relative/path.py::function. Keep the same name and kind.
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
                    "description": "Existing function name or Class.method.",
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

_DEF = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
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


def _symbols(body, prefix=""):
    for node in body:
        if isinstance(node, _DEF):
            qualified = prefix + node.name
            yield qualified, node
            if isinstance(node, ast.ClassDef):
                yield from _symbols(node.body, qualified + ".")


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


def apply_definitions(response, sources, *, only=None):
    """Return a fresh source mapping or reject the proposal without partial edits."""
    if (
        not isinstance(response, str)
        or not response.strip()
        or len(response.encode()) > 80_000
    ):
        raise ValueError("Definition response is empty or exceeds the edit limit")
    text = _unfence(response.replace("\r\n", "\n"))
    documents = _documents(sources)
    replacements = _edits(text, documents)
    if only is not None:
        replacements = tuple(replacements)
        if len(replacements) != 1 or replacements[0][:2] != only:
            raise ValueError("Expected one replacement of the localized definition")
    return _apply_definition_edits(sources, documents, replacements)


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
    if current is None or not any(
        name == symbol and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for name, node in _symbols(current.body)
    ):
        raise ValueError(
            "Native edit must name an existing function in the editable source list"
        )
    return _apply_definition_edits(sources, documents, ((path, symbol, body),))


def _apply_definition_edits(sources, documents, replacements):
    edits = defaultdict(list)
    added_imports = defaultdict(dict)
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

        matches = [node for symbol, node in _symbols(module.body) if symbol == name]
        if len(matches) != 1:
            raise ValueError(
                "Qualified definition is missing or ambiguous in the named file"
            )
        old = matches[0]
        if (
            len(nodes) != 1
            or type(nodes[0]) is not type(old)
            or nodes[0].name != old.name
        ):
            raise ValueError(
                "Replacement must be one complete definition with the same name and kind"
            )
        lines = sources[path].splitlines(keepends=True)
        start = min([old.lineno, *(item.lineno for item in old.decorator_list)]) - 1
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
        _parse(revised, path)
        result[path] = revised
    if result == sources:
        raise ValueError("No definition replacements changed the checkout")
    return result

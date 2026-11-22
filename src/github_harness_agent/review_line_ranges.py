"""Apply bounded line-range edits to one existing Python definition."""

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from .review_definitions import _symbols, _unique_object

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_MAX_EDITS = 6
_MAX_FILE = 256 * 1024
_MAX_DEFINITION = 16 * 1024
_MAX_REPLY = 20 * 1024
_LINE_BREAK = re.compile(r"[\v\f\x1c-\x1e\x85\u2028\u2029]")
_INVALID_LINE = re.compile(r"[\x00-\x08\x0a-\x1f\x85\u2028\u2029]")


@dataclass(frozen=True)
class LineRangeSnapshot:
    symbol: tuple[str, str]
    digest: str
    first_body_line: int
    lines: tuple[str, ...]


@dataclass
class _Scope:
    tree: ast.Module
    node: ast.FunctionDef | ast.AsyncFunctionDef
    rows: list[str]
    start: int
    indent: str
    newline: str


def _scope(source, qualified):
    if (
        not isinstance(source, str)
        or not source
        or len(source.encode()) > _MAX_FILE
        or _LINE_BREAK.search(source)
    ):
        return None
    newline = "\r\n" if "\r\n" in source else "\n"
    if "\r" in source.replace(newline, "") or (
        newline == "\r\n" and "\n" in source.replace(newline, "")
    ):
        return None
    tree = ast.parse(source)
    symbols = list(_symbols(tree.body))
    parts = qualified.split(".")
    # A duplicate class makes its members ambiguous even when a leaf is unique.
    for count in range(1, len(parts) + 1):
        owner = ".".join(parts[:count])
        matches = [node for name, node in symbols if name == owner]
        if len(matches) != 1:
            return None
    node = matches[0]
    if not isinstance(node, _FUNCTIONS):
        return None
    start = min([node.lineno, *(item.lineno for item in node.decorator_list)]) - 1
    rows = source.splitlines(keepends=True)
    first = node.body[0]
    indent = rows[first.lineno - 1].encode()[: first.col_offset]
    if first.lineno <= node.lineno or indent.strip(b" \t"):
        return None
    if (
        sum(len(row.encode()) for row in rows[start : node.end_lineno])
        > _MAX_DEFINITION
    ):
        return None
    return _Scope(tree, node, rows, start, indent.decode(), newline)


def _snapshot(sources, symbol, scope):
    payload = json.dumps(
        [symbol, sorted(sources.items())], separators=(",", ":"), ensure_ascii=False
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return LineRangeSnapshot(
        symbol,
        digest,
        scope.node.body[0].lineno - scope.start,
        tuple(
            row.removesuffix(scope.newline)
            for row in scope.rows[scope.start : scope.node.end_lineno]
        ),
    )


def _valid_symbol(symbol):
    if (
        type(symbol) is not tuple
        or len(symbol) != 2
        or any(type(part) is not str for part in symbol)
    ):
        return False
    path, _ = symbol
    parts = PurePosixPath(path)
    return (
        not parts.is_absolute()
        and ".." not in parts.parts
        and parts.suffix == ".py"
        and str(parts) == path
    )


def prepare_line_range_edit(sources, symbol):
    """Return numbered complete source for a unique, independently editable body."""
    if not _valid_symbol(symbol):
        return None
    path, qualified = symbol
    try:
        scope = _scope(sources.get(path), qualified)
        return _snapshot(sources, symbol, scope) if scope else None
    except (ValueError, SyntaxError, RecursionError):
        return None


def line_range_prompt(snapshot):
    """Render the editor contract and exact line numbering for this snapshot."""
    path, qualified = snapshot.symbol
    numbered = "\n".join(
        f"{row} | {line}" for row, line in enumerate(snapshot.lines, 1)
    )
    return (
        f"Edit only {path}::{qualified} using numbered lines from this exact current definition.\n"
        "Return only one JSON object in this format.\n"
        f'{{"snapshot":"{snapshot.digest}","edits":['
        f'{{"start":{snapshot.first_body_line},"end":{snapshot.first_body_line},"lines":["replacement line"]}}'
        "]}\n"
        f"Use one to {_MAX_EDITS} non-overlapping, inclusive ranges within lines "
        f"{snapshot.first_body_line} through {len(snapshot.lines)}. Numbers refer to the unedited definition. "
        "Preserve exact indentation. Use an empty lines array to delete a range. To insert, "
        "replace an anchor line with itself and the inserted lines in their desired order.\n"
        "Do not include numbering, newline characters, or other output in replacement strings.\n"
        f"Exact current definition (quoted data):\n{numbered}\n"
    )


def _edits(response, snapshot, indent):
    if (
        not isinstance(response, str)
        or not response.strip()
        or len(response.encode()) > _MAX_REPLY
    ):
        raise ValueError("Line-range response is empty or exceeds the edit limit")
    try:
        payload = json.loads(response, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError):
        raise ValueError("Expected one exact JSON line-range edit object") from None
    if type(payload) is not dict or payload.keys() != {"snapshot", "edits"}:
        raise ValueError("Expected snapshot and edits only")
    if payload["snapshot"] != snapshot.digest:
        raise ValueError("Line-range response does not identify this snapshot")
    records = payload["edits"]
    if type(records) is not list or not 1 <= len(records) <= _MAX_EDITS:
        raise ValueError("Expected one to six line-range edits")
    edits = []
    for record in records:
        if type(record) is not dict or record.keys() != {"start", "end", "lines"}:
            raise ValueError("Each range requires start, end, and lines")
        start, end, lines = record["start"], record["end"], record["lines"]
        if (
            type(start) is not int
            or type(end) is not int
            or not snapshot.first_body_line <= start <= end <= len(snapshot.lines)
            or type(lines) is not list
            or any(
                type(line) is not str
                or _INVALID_LINE.search(line)
                or (line.strip() and not line.startswith(indent))
                for line in lines
            )
        ):
            raise ValueError("Invalid body line range or replacement text")
        edits.append((start, end, lines))
    edits.sort(key=lambda edit: edit[0])
    if any(previous[1] >= current[0] for previous, current in zip(edits, edits[1:])):
        raise ValueError("Line-range edits overlap")
    return edits


def apply_line_range_edit(response, sources, snapshot):
    """Validate a proposal and return an independent source mapping."""
    if not isinstance(snapshot, LineRangeSnapshot) or not _valid_symbol(
        snapshot.symbol
    ):
        raise ValueError("Expected a valid line-range snapshot")
    path, qualified = snapshot.symbol
    try:
        scope = _scope(sources.get(path), qualified)
        fresh = _snapshot(sources, snapshot.symbol, scope) if scope else None
    except (ValueError, SyntaxError, RecursionError):
        fresh = None
    if fresh != snapshot:
        raise ValueError("Current checkout does not match the line-range snapshot")
    rows = scope.rows.copy()
    for start, end, replacement in reversed(_edits(response, snapshot, scope.indent)):
        first, last = scope.start + start - 1, scope.start + end
        inserted = [line + scope.newline for line in replacement]
        if (
            inserted
            and last == len(rows)
            and not rows[last - 1].endswith(scope.newline)
        ):
            inserted[-1] = replacement[-1]
        rows[first:last] = inserted
    revised = "".join(rows)
    if revised == sources[path]:
        raise ValueError("Line ranges did not change the definition")
    try:
        check = _scope(revised, qualified)
        if check is None or check.node.end_lineno > scope.node.end_lineno + len(
            rows
        ) - len(scope.rows):
            raise ValueError(
                "Line ranges changed the scope or size of the selected definition"
            )
        # Compare the entire module with only the selected body removed.
        scope.node.body = check.node.body = []
        if ast.dump(scope.tree) != ast.dump(check.tree):
            raise ValueError(
                "Line ranges changed the signature or code outside the selected body"
            )
        compile(revised, path, "exec")
    except (SyntaxError, RecursionError) as error:
        raise ValueError("Line ranges did not produce valid Python") from error
    return {**sources, path: revised}

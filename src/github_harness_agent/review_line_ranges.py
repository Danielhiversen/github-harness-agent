"""Apply bounded line-range edits to one existing Python definition."""

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from .review_definitions import _symbols, _unique_object

LINE_RANGE_ENDING = "Respond with the single line-range JSON object only."
_NATIVE_TOOL_NAME = "replace_python_line_ranges"
NATIVE_LINE_RANGE_TOOL = {
    "type": "function",
    "function": {
        "name": _NATIVE_TOOL_NAME,
        "description": "Edit only the displayed numbered rows of the current function body.",
        "parameters": {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array", "minItems": 1, "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "integer", "minimum": 1},
                            "end": {"type": "integer", "minimum": 1},
                            "lines": {"type": "array", "maxItems": 24, "items": {"type": "string"}},
                        },
                        "required": ["start", "end", "lines"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["edits"],
            "additionalProperties": False,
        },
    },
}

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


def prepare_window_line_range_edit(sources, symbol, source_spans):
    """Bind independently localized source rows to the current definition snapshot."""
    if not _valid_symbol(symbol):
        return None
    try:
        scope = _scope(sources.get(symbol[0]), symbol[1])
    except (ValueError, SyntaxError, RecursionError):
        return None
    if scope is None:
        return None
    snapshot = _snapshot(sources, symbol, scope)
    first = snapshot.first_body_line
    headers = frozenset(range(1, first))
    allowed = frozenset()
    for start, end in source_spans:
        rows = frozenset(
            range(
                max(first, start - scope.start),
                min(len(snapshot.lines), end - scope.start) + 1,
            )
        )
        trial = allowed | rows | headers
        if (
            len(trial) <= 26
            and sum(len(snapshot.lines[row - 1].encode()) for row in trial) <= 2300
        ):
            allowed |= rows
    return (snapshot, allowed | headers, allowed) if allowed else None


def prepare_loop_line_range_edit(sources, symbol, body_line, variable, caller_line):
    """Show only entry, verified loop and mutation windows of one definition."""
    if not _valid_symbol(symbol):
        return None
    try:
        scope = _scope(sources.get(symbol[0]), symbol[1])
    except (ValueError, SyntaxError, RecursionError):
        return None
    if scope is None:
        return None
    loops = [
        item for item in ast.walk(scope.node)
        if isinstance(item, ast.For)
        and isinstance(item.target, ast.Name)
        and item.target.id == variable
        and item.body[0].lineno == body_line
        and body_line <= caller_line <= item.body[-1].end_lineno
    ]
    if len(loops) != 1 or body_line - loops[0].lineno > 5:
        return None
    snapshot = _snapshot(sources, symbol, scope)
    first = snapshot.first_body_line
    stop = len(snapshot.lines)
    points = (
        (first, first + 2),
        (loops[0].lineno - scope.start - 1, body_line - scope.start + 2),
        (caller_line - scope.start - 2, caller_line - scope.start + 2),
    )
    allowed = frozenset(
        row for start, end in points
        for row in range(max(first, start), min(stop, end) + 1)
    )
    visible = allowed | frozenset(range(1, first))
    if len(visible) > 26 or sum(len(snapshot.lines[row - 1].encode()) for row in visible) > 2300:
        return None
    return snapshot, visible, allowed


def line_range_prompt(snapshot, *, visible_rows=None, native=False):
    """Render the editor contract and exact line numbering for this snapshot."""
    path, qualified = snapshot.symbol
    first = snapshot.first_body_line
    indent = re.match(r"[ \t]*", snapshot.lines[first - 1])[0]
    example_line = json.dumps(indent + "pass")
    numbered = "\n".join(
        f"{row} | {line}" for row, line in enumerate(snapshot.lines, 1)
        if visible_rows is None or row in visible_rows
    )
    sparse = (
        "Use only rows shown below in the definition body, with at most three edit ranges. "
        "Do not span gaps between shown rows. Limit old and replacement lines to 24 in total.\n"
        if visible_rows is not None else ""
    )
    instruction = (
        "Supply only the edits field in the tool arguments, with exact indentation.\n"
        if native else "Use this JSON format. Supply the changed lines with their indentation.\n"
    )
    example = '{"edits":[' if native else f'{{"snapshot":"{snapshot.digest}","edits":['
    return (
        f"Edit only {path}::{qualified} using numbered lines from this exact current definition.\n"
        + instruction + example
        + f'{{"start":{first},"end":{first},"lines":[{example_line}]}}'
        + "]}\n"
        + f"Use one to {3 if native else _MAX_EDITS} non-overlapping, inclusive ranges within lines "
        f"{snapshot.first_body_line} through {len(snapshot.lines)}. Numbers refer to the unedited definition. "
        "Preserve exact indentation. Use an empty lines array to delete a range. To insert, "
        "replace an anchor line with itself and the inserted lines in their desired order.\n"
        "Do not include numbering, newline characters, or other output in replacement strings.\n"
        f"{sparse}Exact current definition (quoted data):\n{numbered}\n"
    )


def selectable_line_tool(snapshots):
    function = NATIVE_LINE_RANGE_TOOL["function"]
    schema = function["parameters"]
    properties = {**schema["properties"], "snapshot": {"type": "string", "enum": [s.digest for s in snapshots]}}
    return {"type": "function", "function": {
        **function, "parameters": {**schema, "properties": properties, "required": ["snapshot", "edits"]},
    }}


def apply_native_line_range_edit(calls, sources, snapshot, *, allowed_rows, alternatives=()):
    """Bind a single edits-only tool response to trusted numbered source."""
    try:
        if not isinstance(snapshot, LineRangeSnapshot) or not allowed_rows or len(calls) != 1:
            raise ValueError
        call = calls[0]
        function = call["function"]
        arguments = function["arguments"]
        if (
            call["type"] != "function" or function["name"] != _NATIVE_TOOL_NAME
            or not isinstance(arguments, str) or len(arguments.encode()) > _MAX_REPLY
        ):
            raise ValueError
        payload = json.loads(arguments, object_pairs_hook=_unique_object)
        if alternatives:
            selected = select_line_snapshot(arguments, (snapshot, *alternatives))
            if selected is not snapshot:
                allowed_rows = frozenset(range(selected.first_body_line, len(selected.lines) + 1))
            snapshot = selected
        elif type(payload) is not dict or payload.keys() != {"edits"}:
            raise ValueError
    except (IndexError, KeyError, TypeError, ValueError, RecursionError):
        raise ValueError("Expected one numbered editing tool call with edits only") from None
    response = json.dumps({"snapshot": snapshot.digest, "edits": payload["edits"]}, ensure_ascii=False)
    return apply_line_range_edit(response, sources, snapshot, allowed_rows=allowed_rows)


def _line_payload(response):
    if (
        not isinstance(response, str)
        or not response.strip()
        or len(response.encode()) > _MAX_REPLY
    ):
        raise ValueError("Line-range response is empty or exceeds the edit limit")
    rows = response.strip().splitlines()
    if len(rows) >= 3 and rows[0] in ("```", "```json") and rows[-1] == "```":
        response = "\n".join(rows[1:-1])
    try:
        payload = json.loads(response, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError):
        raise ValueError("Expected one exact JSON line-range edit object") from None
    if type(payload) is not dict or payload.keys() != {"snapshot", "edits"}:
        raise ValueError("Expected snapshot and edits only")
    return payload


def _single_hunk(response, path, original):
    rows = response.splitlines()
    if rows[:1] == [f"diff --git a/{path} b/{path}"]:
        rows = rows[1:]
        if rows and re.fullmatch(r"index [0-9a-f]+\.\.[0-9a-f]+(?: 100644)?", rows[0]):
            rows = rows[1:]
    if len(rows) < 4 or rows[:2] not in ([f"--- a/{path}", f"+++ b/{path}"], [f"--- {path}", f"+++ {path}"]):
        raise ValueError("Expected a single unified hunk for the offered file")
    header = re.fullmatch(r"@@ -([0-9]+)(?:,([0-9]+))? \+([0-9]+)(?:,([0-9]+))? @@(?: .*)?", rows[2])
    if not header or any(not row or row[0] not in " +-" for row in rows[3:]):
        raise ValueError("Expected an exact unified hunk with old and new line counts")
    start, old_count, new_start, new_count = (int(value or 1) for value in header.groups())
    old = [row[1:] for row in rows[3:] if row[0] in " -"]
    new = [row[1:] for row in rows[3:] if row[0] in " +"]
    if (not old or start < 1 or new_start != (start if new else start - 1)
            or (len(old), len(new)) != (old_count, new_count) or original[start - 1:start - 1 + len(old)] != old):
        raise ValueError("Unified hunk does not match the exact current file lines")
    return start - 1, old, new


def apply_offered_line_reply(response, sources, snapshots, *, block_pattern):
    """Bind a textual free-model edit to one of the actually offered full definitions."""
    if not isinstance(response, str) or not 0 < len(response.encode()) <= _MAX_REPLY:
        raise ValueError("Expected a bounded scoped edit")
    text = response.replace("\r\n", "\n").strip("\n")
    fenced = re.fullmatch(r"```(?:json|diff|patch|text)?[ \t]*\n(.*)\n```", text, re.S | re.I)
    if fenced:
        text = fenced[1]
    path = snapshots[0].symbol[0]
    original = sources[path].splitlines()
    if text.lstrip().startswith("{"):
        try:
            payload = json.loads(text, object_pairs_hook=_unique_object)
        except (ValueError, RecursionError):
            raise ValueError("Expected one exact JSON line-range edit object") from None
        if type(payload) is not dict or "edits" not in payload or payload.keys() - {"snapshot", "symbol", "path", "edits"}:
            raise ValueError("Expected edits and an offered snapshot or symbol")
        matches = list(snapshots)
        if "path" in payload and payload["path"] != path:
            raise ValueError("The supplied path must match the offered file")
        if "symbol" in payload:
            matches = [s for s in matches if payload["symbol"] in (s.symbol[1], "::".join(s.symbol))]
        if "snapshot" in payload:
            prefix = payload["snapshot"]
            if isinstance(prefix, str) and re.fullmatch("[0-9a-f]{12,64}", prefix):
                matches = [s for s in matches if s.digest.startswith(prefix)]
            else:
                matches = [s for s in matches if prefix in (s.symbol[1], "::".join(s.symbol))]
        if len(matches) != 1 or not ({"symbol", "snapshot"} & payload.keys()):
            raise ValueError("The reply must identify exactly one offered definition")
        selected = matches[0]
        edits = payload["edits"]
    else:
        if text.startswith(("diff --git ", "--- ")):
            start, old, new = _single_hunk(text, path, original)
        else:
            if text.startswith("<<<<<<< SEARCH\n"):
                text = path + "\n" + text
            block = block_pattern.fullmatch(text + "\n")
            if not block or block[1] != path or any(
                row.startswith(("<<<<<<<", "=======", ">>>>>>>")) for row in (block[2] + block[3]).splitlines()
            ):
                raise ValueError("Expected one exact SEARCH/REPLACE block for the offered file")
            old, new = block[2].splitlines(), block[3].splitlines()
            if not old:
                raise ValueError("The SEARCH must contain an exact old source line")
            haystack = "\n" + "\n".join(original) + "\n"
            needle = "\n" + "\n".join(old) + "\n"
            match = haystack.find(needle)
            if match < 0 or haystack.find(needle, match + 1) >= 0:
                raise ValueError("The SEARCH must match exactly one location in the current file")
            start = haystack.count("\n", 0, match)
        common = min(len(old), len(new))
        prefix = next((i for i in range(common) if old[i] != new[i]), common)
        suffix = next((i for i in range(common - prefix) if old[-1 - i] != new[-1 - i]), common - prefix)
        start += prefix
        old, new = old[prefix:len(old) - suffix], new[prefix:len(new) - suffix]
        if not old and not new:
            raise ValueError("The alternate format did not change the checkout")
        # Bind insertions to a real on-snapshot preceding row, just like the native format.
        if not old:
            if start < 1:
                raise ValueError("Insertions need an existing body row as their anchor")
            start -= 1
            old = [original[start]]
            new = old + new
        matches = []
        for snapshot in snapshots:
            scope = _scope(sources[path], snapshot.symbol[1])
            if scope is None:
                continue
            first = start + 1 - scope.start
            last = first + len(old) - 1
            if snapshot.first_body_line <= first <= last <= len(snapshot.lines):
                matches.append((snapshot, [{"start": first, "end": last, "lines": new}]))
        if len(matches) != 1:
            raise ValueError("The change must lie in the body of exactly one offered definition")
        selected, edits = matches[0]
    normalized = json.dumps({"snapshot": selected.digest, "edits": edits})
    allowed = frozenset(range(selected.first_body_line, len(selected.lines) + 1))
    return apply_line_range_edit(normalized, sources, selected, allowed_rows=allowed)


def select_line_snapshot(response, snapshots):
    digest = _line_payload(response)["snapshot"]
    for snapshot in snapshots:
        if digest == snapshot.digest:
            return snapshot
    raise ValueError("The reply must identify exactly one offered line-range snapshot")


def _edits(response, snapshot, indent):
    payload = _line_payload(response)
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
        ):
            raise ValueError(
                f"Use numbered definition rows {snapshot.first_body_line} through "
                f"{len(snapshot.lines)} for start and end. The numbers are relative to this definition."
            )
        if type(lines) is not list or any(
            type(line) is not str or _INVALID_LINE.search(line) for line in lines
        ):
            raise ValueError("Use a lines array with one string per source line and no line breaks")
        if any(line.strip() and not line.startswith(indent) for line in lines):
            raise ValueError("Preserve the body indentation shown after the numbering on each nonblank line")
        edits.append((start, end, lines))
    edits.sort(key=lambda edit: edit[0])
    if any(previous[1] >= current[0] for previous, current in zip(edits, edits[1:])):
        raise ValueError("Line-range edits overlap")
    return edits


def apply_line_range_edit(response, sources, snapshot, *, allowed_rows=None):
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
    edits = _edits(response, snapshot, scope.indent)
    if allowed_rows is not None and (
        len(edits) > 3
        or sum(end - start + 1 + len(lines) for start, end, lines in edits) > 24
        or any(not allowed_rows.issuperset(range(start, end + 1)) for start, end, _ in edits)
    ):
        raise ValueError("Line ranges exceeded the shown scoped edit windows")
    rows = scope.rows.copy()
    for start, end, replacement in reversed(edits):
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

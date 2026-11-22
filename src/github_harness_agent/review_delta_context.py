"""Evidence for reconciling a tracked Python regression with review checks."""

import ast
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import PurePosixPath

from .review_definitions import _symbols
from .review_solver_context import _start
from .task_context import excerpt

_FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)
_DELTA_LIMIT = 4600
_EXAMPLE_LIMIT = 3300
_REGION_LIMIT = 2200


def _parse(source):
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return None


def _definitions(sources):
    definitions = {}
    for path, source in sources.items():
        if not path.endswith(".py") or not (module := _parse(source)):
            continue
        lines = source.splitlines(keepends=True)
        symbols = list(_symbols(module.body))
        counts = Counter(name for name, _ in symbols)
        for name, node in symbols:
            if counts[name] == 1 and isinstance(node, _FUNCTION):
                start = _start(node)
                definitions[f"{path}::{name}"] = (
                    start,
                    lines[start - 1 : node.end_lineno],
                )
    return definitions


def _rows(lines, start, end, allowance, prefix=""):
    rows = []
    remaining = allowance
    for line in lines[start:end]:
        row = prefix + line
        if len(row) > remaining:
            break
        rows.append(row)
        remaining -= len(row)
    return "".join(rows)


def _delta_rows(lines, start, end, allowance, prefix):
    full = _rows(lines, start, end, allowance, prefix)
    if len(full.splitlines()) == end - start:
        return full
    marker = f"{prefix}# ... changed lines omitted ...\n"
    half = (allowance - len(marker)) // 2
    head = _rows(lines, start, end, half, prefix)
    tail_start = end
    remaining = half
    while tail_start > start and len(lines[tail_start - 1]) + len(prefix) <= remaining:
        tail_start -= 1
        remaining -= len(lines[tail_start]) + len(prefix)
    return head + marker + _rows(lines, tail_start, end, half, prefix)


def _changes(before, after):
    old_start, old = before
    new_start, new = after
    matcher = SequenceMatcher(None, old, new)
    hunks, regions = [], []
    remaining = 1900
    for tag, i, j, k, end in matcher.get_opcodes():
        if tag == "equal":
            continue
        header = f"@@ baseline line {old_start + i}, draft line {new_start + k} @@\n"
        allowance = min(520, (remaining - len(header) - 1) // 2)
        minimum, maximum = max(0, k - 3), min(len(new), max(end, k + 1) + 3)
        regions.append((minimum, maximum))
        if end - k > 10:
            regions.append((end - 4, maximum))
        if allowance < 80:
            break
        removed = _delta_rows(old, i, j, allowance, "-")
        added = _delta_rows(new, k, end, allowance, "+")
        if not (removed or added):
            continue
        context = _rows(
            new,
            max(0, k - 2),
            k,
            min(160, remaining - len(header + removed + added) - 1),
            " ",
        )
        hunk = header + context + removed + added + "\n"
        hunks.append(hunk)
        remaining -= len(hunk)
    return "".join(hunks), regions


def _search_source(start, lines, regions):
    if sum(map(len, lines)) <= _REGION_LIMIT:
        return "".join(lines)
    sections = []
    remaining = _REGION_LIMIT
    if regions and regions[0][0] > 3:
        regions = [(0, 3), *regions]
    for first, end in regions or [(0, 12), (max(12, len(lines) - 10), len(lines))]:
        heading = f"# Exact current lines starting at {start + first}\n"
        body = _rows(lines, first, end, min(750, remaining - len(heading) - 1))
        if body:
            sections.append(heading + body)
            remaining -= len(sections[-1])
    return "".join(sections)


def _call_names(tree):
    return {
        call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, (ast.Attribute, ast.Name))
    }


def _test_excerpt(lines, node, names):
    first = _start(node) - 1
    body = _rows(lines, first, node.end_lineno, 1550)
    if len(body.splitlines()) == node.end_lineno - first:
        return body
    selected = {first, node.lineno - 1}
    statements = sorted(
        (
            stmt
            for stmt in ast.walk(node)
            if isinstance(stmt, (ast.Assert, ast.Assign, ast.AnnAssign, ast.Expr))
        ),
        key=lambda stmt: stmt.lineno,
    )
    relevant = [
        stmt
        for stmt in statements
        if isinstance(stmt, ast.Assert) or _call_names(stmt) & names
    ]
    # Retain assignments supplying the selected calls and expected results.
    needed = set()
    for stmt in reversed(statements):
        bindings = {
            child.id
            for child in ast.walk(stmt)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
        }
        if stmt in relevant or bindings & needed:
            needed.update(
                child.id
                for child in ast.walk(stmt)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
            )
            selected.update(range(stmt.lineno - 1, stmt.end_lineno))
    groups = []
    remaining = 1550
    for row in sorted(selected):
        text = lines[row]
        if len(text) <= remaining:
            groups.append(text)
            remaining -= len(text)
    return "".join(groups)


def _operator_examples(fixtures, names):
    candidates = []
    for path, source in fixtures.items():
        if not (module := _parse(source)):
            continue
        lines = source.splitlines(keepends=True)
        symbols = {
            node.name: node
            for _, node in _symbols(module.body)
            if isinstance(node, _FUNCTION) and not node.name.startswith("test_")
        }
        for qualified, node in _symbols(module.body):
            if not isinstance(node, _FUNCTION) or not node.name.startswith("test_"):
                continue
            calls = _call_names(node)
            direct = calls & names
            helpers = [
                helper
                for name in sorted(calls & symbols.keys())
                if _call_names(helper := symbols[name]) & names
            ]
            if direct or helpers:
                body = _test_excerpt(lines, node, names | symbols.keys())
                body += "".join(
                    _test_excerpt(lines, helper, names) for helper in helpers[:1]
                )
                candidates.append((len(direct), f"{path}::{qualified}\n{body}\n"))
    remaining = _EXAMPLE_LIMIT
    selected = []
    # Keep the archive order for equally relevant examples.
    for _, example in sorted(candidates, key=lambda item: item[0], reverse=True):
        if len(example) <= remaining:
            selected.append(example)
            remaining -= len(example)
    return "".join(selected)


def stock_delta_context(original, current, stock_fixture, fixtures, notes):
    """Return bounded production deltas, operator evidence and exact draft regions."""
    baseline = _definitions(original)
    draft = _definitions(
        {path: current[path] for path in original.keys() & current.keys()}
    )
    names = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", stock_fixture))

    def leaf(key):
        return key.split("::", 1)[1].rsplit(".", 1)[-1]

    all_symbols = baseline.keys() | draft.keys()
    names &= {leaf(key) for key in all_symbols}
    ranked = sorted(all_symbols, key=lambda key: (leaf(key) not in names, key))

    inventory, changes, changed_names = {}, [], set()
    remaining = _DELTA_LIMIT
    for key in ranked:
        before, after = baseline.get(key, (1, [])), draft.get(key, (1, []))
        delta, regions = _changes(before, after) if before[1] != after[1] else ("", [])
        if delta:
            changed_names.add(leaf(key))
            section = f"{key}\n{delta}"
            if len(section) <= remaining:
                changes.append(section)
                remaining -= len(section)
        if key in draft:
            inventory[key] = _search_source(*after, regions)
    examples = _operator_examples(fixtures, names | changed_names)
    note_limit = 3300 // max(1, len(notes))
    constraints = "\nReview notes (quoted):\n" + "\n".join(
        f"{path}:{line}: {excerpt(body, note_limit)}" for path, line, body in notes
    )
    if examples:
        constraints += "\nOperator pytest examples (quoted, read-only):\n" + examples
    return "".join(changes), constraints, inventory


def resolve_symbol(label, inventory):
    """Accept unique qualified Python spellings of an inventory ID."""
    label = label.strip()
    if label in inventory:
        return label
    path, separator, name = label.partition("::")
    if not separator:
        name = path
    matches = []
    for key in inventory:
        filename, symbol = key.split("::", 1)
        if separator and path != filename:
            continue
        module = ".".join(PurePosixPath(filename).with_suffix("").parts)
        qualified = module + "." + symbol
        if name == symbol or qualified.endswith("." + name) or name == qualified:
            matches.append(key)
    return matches[0] if len(matches) == 1 else None

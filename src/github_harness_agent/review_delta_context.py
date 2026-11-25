"""Evidence for reconciling a tracked Python regression with review checks."""

import ast
import re
import tarfile
from collections import Counter
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import PurePosixPath

from .review_definitions import _symbols
from .review_solver_context import _start, _tracked_python
from .task_context import excerpt

_FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)
_DELTA_LIMIT = 4600
_EXAMPLE_LIMIT = 3300
STOCK_SOURCE_LIMIT = 16 * 1024
_REGION_LIMIT = 2200
_PRIVATE_SOURCE_LIMIT = 3200


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


def _changes(before, after, *, labels=("baseline", "draft"), limit=1900):
    old_start, old = before
    new_start, new = after
    matcher = SequenceMatcher(None, old, new)
    hunks, regions = [], []
    remaining = limit
    for tag, i, j, k, end in matcher.get_opcodes():
        if tag == "equal":
            continue
        header = f"@@ {labels[0]} line {old_start + i}, {labels[1]} line {new_start + k} @@\n"
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


def reviewed_ancestor_delta(archive, sources, feedback, *, max_bytes=900):
    """Ground new tests in changed public definitions on both reviewed revisions."""
    scripts = {"bin", "script", "scripts", "tool", "tools"}
    try:
        with tarfile.open(fileobj=BytesIO(archive), mode="r:*") as snapshot:
            previous = {
                path: "".join(tracked[0])
                for path in sources
                if not scripts.intersection(PurePosixPath(path).parts)
                and (tracked := _tracked_python(snapshot, path)) is not None
            }
    except (OSError, tarfile.TarError):
        return ""
    before = _definitions(previous)
    current = _definitions({path: sources[path] for path in previous})

    def tokens(value):
        return set(re.findall(r"[a-z][a-z0-9]{2,}", value.casefold()))

    candidates = []
    for key in before.keys() & current.keys():
        path, _, name = key.partition("::")
        first, lines = current[key]
        if any(part.startswith("_") for part in name.split(".")) or lines == before[key][1]:
            continue
        relevant = [(line, body) for reviewed_path, line, body in feedback if reviewed_path == path]
        query = tokens(" ".join(body for _, body in relevant))
        score = (
            any(first <= row < first + len(lines) for row, _ in relevant),
            len(query & tokens(name)),
        )
        if any(score):
            candidates.append((score, key))
    sections = []
    remaining = max_bytes
    for _, key in sorted(candidates, reverse=True)[:2]:
        delta, _ = _changes(
            before[key], current[key], labels=("ancestor", "current"),
            limit=remaining - len(key.encode()) - 2,
        )
        section = f"{key}\n{delta}"
        if delta and (size := len(section.encode())) <= remaining:
            sections.append(section)
            remaining -= size
    return "".join(sections)


def _search_source(
    key, start, lines, regions, *, limit=STOCK_SOURCE_LIMIT, label="current"
):
    if len(key) + 1 + sum(map(len, lines)) <= limit:
        return "".join(lines)
    sections = []
    remaining = min(limit, _REGION_LIMIT)
    if regions and regions[0][0] > 3:
        regions = [(0, 3), *regions]
    for first, end in regions or [(0, 12), (max(12, len(lines) - 10), len(lines))]:
        heading = f"# Exact {label} lines starting at {start + first}\n"
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


def _selected_operator_calls(fixtures, node_id):
    path, _, name = node_id.split("[", 1)[0].partition("::")
    matching = [
        path_hint
        for path_hint in fixtures
        if path == path_hint or path.endswith("/" + path_hint)
    ]
    if not name or not matching:
        return set()
    module = _parse(fixtures[max(matching, key=len)])
    if module is None:
        return set()
    qualified = name.replace("::", ".")
    selected = [
        node
        for symbol, node in _symbols(module.body)
        if symbol == qualified and isinstance(node, _FUNCTION)
    ]
    if len(selected) != 1:
        return set()
    calls = _call_names(selected[0])
    helpers = [
        node
        for node in module.body
        if isinstance(node, _FUNCTION)
        and node.name in calls
        and node is not selected[0]
    ]
    aliases = {
        alias.asname: alias.name
        for item in module.body
        if isinstance(item, ast.ImportFrom)
        for alias in item.names
        if alias.asname
    }
    calls -= {helper.name for helper in helpers}
    calls.update(callee for helper in helpers[:4] for callee in _call_names(helper))
    return {aliases.get(callee, callee) for callee in calls}


def private_operator_source(verified, tentative, fixtures, node_id):
    """Show exact source pairs only for changed, unambiguous fixture callees."""
    calls = _selected_operator_calls(fixtures, node_id)
    if not calls:
        return ""
    previous, current = _definitions(verified), _definitions(tentative)
    leaves = {
        key: key.split("::", 1)[1].rsplit(".", 1)[-1]
        for key in previous.keys() | current.keys()
    }
    counts = Counter(leaves.values())
    symbols = sorted(
        key
        for key in previous.keys() & current.keys()
        if leaves[key] in calls
        and counts[leaves[key]] == 1
        and previous[key][1] != current[key][1]
    )[:2]
    sections = []
    for key in symbols:
        target_label = f"Tentative edit target (exact source): {key}\n"
        donor_label = f"Previously verified donor (read-only): {key}\n"
        allowance = (
            _PRIVATE_SOURCE_LIMIT // len(symbols) - len(target_label + donor_label) - 2
        ) // 2
        if allowance < 180:
            continue
        before, after = previous[key], current[key]
        _, current_regions = _changes(before, after)
        _, donor_regions = _changes(after, before)
        target = _search_source(key, *after, current_regions, limit=allowance)
        donor = _search_source(
            key, *before, donor_regions, limit=allowance, label="verified donor"
        )
        if target and donor:
            sections.append(target_label + target + "\n" + donor_label + donor + "\n")
    return "".join(sections)


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


def _operator_examples(fixtures, names, *, first_limit=None):
    candidates = []
    contracts = []
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
                calls = [
                    ast.unparse(call)
                    for call in ast.walk(node)
                    if isinstance(call, ast.Call)
                    and isinstance(call.func, (ast.Name, ast.Attribute))
                    and (call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id)
                    in names
                ]
                assertions = [
                    ast.unparse(stmt.test) for stmt in ast.walk(node)
                    if isinstance(stmt, ast.Assert)
                ]
                if calls and assertions:
                    contracts.append(
                        f"{qualified}: {excerpt(calls[0], 160)} -> {excerpt(assertions[-1], 180)}"
                    )
                body = _test_excerpt(lines, node, names | symbols.keys())
                body += "".join(
                    _test_excerpt(lines, helper, names) for helper in helpers[:1]
                )
                candidates.append((len(direct), f"{path}::{qualified}\n{body}\n"))
    remaining = _EXAMPLE_LIMIT
    selected = []
    # Keep the archive order for equally relevant examples.
    for _, example in sorted(candidates, key=lambda item: item[0], reverse=True):
        if first_limit is not None:
            if len(example.encode()) <= first_limit:
                return example
            continue
        if len(example) <= remaining:
            selected.append(example)
            remaining -= len(example)
    if first_limit is not None:
        return ""
    overview = excerpt("\n".join(contracts), 2300) if contracts else ""
    return (overview + "\n\n" if overview else "") + "".join(selected)


def interactive_stock_focus(original, current, stock_fixture, fixtures):
    """Find a shared public call and one operator example for a scoped tool session."""
    before, after = _definitions(original), _definitions(current)
    called = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", stock_fixture))
    matching = {
        key: name for key in after
        if (name := key.split("::", 1)[1].rsplit(".", 1)[-1]) in called
    }
    changed = [
        key for key in matching
        if key in before and before[key][1] != after[key][1]
    ]
    hint = changed if changed else matching
    symbol = tuple(next(iter(hint)).split("::", 1)) if len(hint) == 1 else None
    example = _operator_examples(fixtures, set(matching.values()), first_limit=900)
    return symbol, example


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
            inventory[key] = _search_source(key, *after, regions)
    # A draft for multiple review threads can change unrelated public APIs.
    # Prefer positive examples of the entry points the failing stock case uses.
    examples = _operator_examples(fixtures, names or changed_names)
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

"""Select compact Python source excerpts for a single review concern."""

import ast
import re
from pathlib import PurePosixPath

from .check_diagnostics import check_locations

_FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)
_IMPORT = (ast.Import, ast.ImportFrom)
_ASSIGNMENT = (ast.Assign, ast.AnnAssign)
_LOOP = (ast.For, ast.AsyncFor, ast.While)
_CONTROL = (*_LOOP, ast.If, ast.With, ast.AsyncWith, ast.Try, ast.TryStar, ast.Match)


def _diagnostic_window(lines, imports, row, limit):
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


def _words(text):
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return set(re.findall(r"[A-Za-z][A-Za-z0-9]*", separated.lower()))


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

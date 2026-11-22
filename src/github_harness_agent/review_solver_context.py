"""Select compact Python source excerpts for a single review concern."""

import ast
import re

_FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)
_IMPORT = (ast.Import, ast.ImportFrom)
_ASSIGNMENT = (ast.Assign, ast.AnnAssign)


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
    }


def _bindings(node):
    if isinstance(node, _IMPORT):
        return {alias.asname or alias.name.split(".", 1)[0] for alias in node.names}
    if isinstance(node, _ASSIGNMENT):
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        return _references(targets)
    return {getattr(node, "name", None)}


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
    candidates = []
    for node in module.body:
        if isinstance(node, ast.ClassDef):
            candidates.append((node, None))
            candidates.extend(
                (child, node) for child in node.body if isinstance(child, _FUNCTION)
            )
        elif isinstance(node, _FUNCTION):
            candidates.append((node, None))
    if not candidates:
        return source[:max_chars]

    def rank(candidate):
        node, owner = candidate
        names = _words(node.name)
        scope = _words(owner.name) if owner else set()
        at_anchor = anchor is not None and _start(node) <= anchor <= node.end_lineno
        end = node.lineno if isinstance(node, ast.ClassDef) else node.end_lineno
        return (
            8 * len(names & targets)
            + 3 * len((names | scope) & notes)
            + 5 * at_anchor
            + min(4, len(_words("".join(lines[_start(node) - 1 : end])) & targets)),
            at_anchor,
            isinstance(node, _FUNCTION),
            -(node.end_lineno - node.lineno),
        )

    active, owner = max(candidates, key=rank)
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
        append(active, clip_to=max_chars * 2 // 3)
    if owner is not None:
        declaration(owner)

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

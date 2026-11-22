"""Compose small exact line edits independently made against one reviewed draft."""

from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass(frozen=True)
class LineEdit:
    path: str
    start: int
    original: tuple[str, ...]
    replacement: tuple[str, ...]

    @property
    def stop(self):
        return self.start + len(self.original)


def small_line_edits(base, proposal, allowed):
    if base.keys() != proposal.keys():
        return ()
    changed = [path for path in base if base[path] != proposal[path]]
    if not 0 < len(changed) <= 2 or any(
        path not in allowed or not path.endswith(".py") for path in changed
    ):
        return ()
    edits = []
    for path in changed:
        if max(len(base[path].encode()), len(proposal[path].encode())) > 64 * 1024:
            return ()
        before = base[path].splitlines(keepends=True)
        after = proposal[path].splitlines(keepends=True)
        if max(len(before), len(after)) > 1536:
            return ()
        for op, start, stop, first, last in SequenceMatcher(
            None, before, after, autojunk=False
        ).get_opcodes():
            if op != "equal":
                edits.append(
                    LineEdit(
                        path, start, tuple(before[start:stop]), tuple(after[first:last])
                    )
                )
    lines = [line for edit in edits for line in (*edit.original, *edit.replacement)]
    if (
        len(edits) > 4
        or len(lines) > 24
        or sum(len(line.encode()) for line in lines) > 4096
    ):
        return ()
    return tuple(edits)


def compose_line_edits(base, first, second):
    if not first or not second:
        return None
    by_path = {}
    for edit in (*first, *second):
        by_path.setdefault(edit.path, []).append(edit)
    merged = base.copy()
    for path, edits in by_path.items():
        if path not in base:
            return None
        lines = base[path].splitlines(keepends=True)
        edits.sort(key=lambda edit: (edit.start, edit.stop))
        for previous, following in zip(edits, edits[1:], strict=False):
            if previous.stop > following.start or (
                previous.stop == following.start
                and (not previous.original or not following.original)
            ):
                return None
        for edit in reversed(edits):
            if (
                not 0 <= edit.start <= edit.stop <= len(lines)
                or tuple(lines[edit.start : edit.stop]) != edit.original
            ):
                return None
            lines[edit.start : edit.stop] = edit.replacement
        source = "".join(lines)
        try:
            compile(source, path, "exec")
        except (SyntaxError, ValueError, RecursionError):
            return None
        merged[path] = source
    return merged

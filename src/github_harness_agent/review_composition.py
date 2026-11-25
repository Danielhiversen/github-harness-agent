"""Apply and combine small exact line edits against one reviewed draft."""

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


def small_line_edits(base, proposal, allowed, *, max_edits=4):
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
        len(edits) > max_edits
        or len(lines) > 24
        or sum(len(line.encode()) for line in lines) > 4096
    ):
        return ()
    return tuple(edits)


def compose_line_edits(base, first, second):
    if not first or not second:
        return None
    return _apply_line_edits(base, (*first, *second))


def _apply_line_edits(base, edits):
    by_path = {}
    for edit in edits:
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


def individual_line_edits(base, proposal, allowed, *, limit):
    """Yield independently applicable hunks from one small multi-hunk proposal."""
    edits = small_line_edits(base, proposal, allowed, max_edits=6)
    if len(edits) < 2 or limit < 1:
        return
    for edit in edits:
        candidate = _apply_line_edits(base, (edit,))
        if candidate is not None:
            yield candidate
            limit -= 1
            if not limit:
                return


def selective_reverts(base, proposal, allowed, locations, *, static_locations=()):
    """Yield at most two exact subsets of a single small multi-hunk edit."""
    edits = small_line_edits(base, proposal, allowed, max_edits=6)
    if len(edits) < 3:
        return

    shifts = {}
    ranked = []
    for index, edit in enumerate(edits):
        # Tracebacks refer to the provider draft, where earlier hunks can shift rows.
        shift = shifts.get(edit.path, 0)
        start = edit.start + shift + 1
        stop = start + max(1, len(edit.replacement))
        static_hits = sum(
            start <= row < stop for path, row in static_locations if path == edit.path
        )
        hits = sum(start <= row < stop for path, row in locations if path == edit.path)
        ranked.append(
            (-static_hits, -hits, -(len(edit.original) + len(edit.replacement)), index)
        )
        shifts[edit.path] = shift + len(edit.replacement) - len(edit.original)

    yielded = 0
    for *_, omitted in sorted(ranked):
        candidate = _apply_line_edits(base, (*edits[:omitted], *edits[omitted + 1 :]))
        if candidate is not None:
            yield candidate
            yielded += 1
            if yielded == 2:
                return

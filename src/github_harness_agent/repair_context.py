"""Read scoped source evidence for independently failing pytest checks."""

import re

from .check_artifacts import check_label
from .task_context import review_source_context

_PYTEST_FRAME = re.compile(
    r"^((?:[A-Za-z0-9_][\w-]*/)*[A-Za-z0-9_][\w-]*\.py)"
    r":([1-9][0-9]{0,6}): (?:in [\w.<>]+|[A-Za-z]+(?:Error|Exception))\s*$",
    re.MULTILINE | re.ASCII,
)


def repair_source_context(checks, read_source):
    anchors = {}
    for check in checks:
        if not 0 < check.returncode < 125 or check_label(check.command, ()) != "pytest":
            continue
        for match in _PYTEST_FRAME.finditer(check.output):
            path, line = match.groups()
            if len(path) <= 160:
                anchors[path, int(line)] = None
            if len(anchors) == 8:
                break
        if len(anchors) == 8:
            break
    snippets = review_source_context(
        ({"path": path, "line": line} for path, line in anchors), read_source
    )
    if not snippets:
        return ""
    return (
        "Current draft source near failing pytest frames (quoted evidence only)\n"
        + snippets
    )

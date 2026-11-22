"""Bound task and tool context, with read-only access to omitted evidence."""

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from .github import GitHubError, validate_repository
from .review_definitions import DEFINITION_ENDING
from .review_line_ranges import LINE_RANGE_ENDING

MESSAGE_BYTES = 57344
RETRY_MESSAGE_BYTES = 24576
TOOL_RESULT_BYTES = 6000
PAGE_CHARS = 1800
HISTORY_PAGES = 12
HISTORY_ROWS = 4
_ELLIPSIS = "\n[excerpt shortened]\n"
_THREAD_PAGE = """
query TaskThreadPage($thread: ID!, $cursor: String) {
  node(id: $thread) {
    ... on PullRequestReviewThread {
      comments(first: 4, after: $cursor) {
        nodes { body state publishedAt author { login } }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
"""

CONTEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "read_task_context",
        "description": "Read one page of this task's read-only context. Read section 'sections' to list available sections.",
        "parameters": {
            "type": "object",
            "properties": {
                "section": {"type": "string"},
                "page": {"type": "integer", "minimum": 1},
            },
            "required": ["section", "page"],
            "additionalProperties": False,
        },
    },
}

HistoryLoader = Callable[[int], tuple[str, bool]]


def wire_size(value: Any) -> int:
    """Count conservatively, including JSON escapes used in provider messages."""
    return len(json.dumps(value, ensure_ascii=True, separators=(",", ":")))


def excerpt(text: str, max_bytes: int) -> str:
    data = text.encode()
    if len(data) <= max_bytes:
        return text
    available = max(0, max_bytes - len(_ELLIPSIS))
    if not available:
        return data[:max_bytes].decode(errors="ignore")
    first = available // 2
    return (
        data[:first].decode(errors="ignore")
        + _ELLIPSIS
        + data[-(available - first) :].decode(errors="ignore")
    )


@dataclass(frozen=True)
class TaskContext:
    instructions: str
    _sections: Mapping[str, str]
    _history: Mapping[str, HistoryLoader]
    _cache: dict[tuple[str, int], tuple[str, bool]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def read(self, section: str, page: int) -> dict:
        def response(content: str, more: bool = False) -> dict:
            return {"content": content, "next_page": page + 1 if more else None}

        if type(page) is not int or page < 1 or not isinstance(section, str):
            return response("Provide a listed section and a positive integer page.")
        if section in self._sections:
            text = self._sections[section]
            start = (page - 1) * PAGE_CHARS
            end = start + PAGE_CHARS
            return response(text[start:end], end < len(text))
        if section not in self._history:
            return response(
                "Unknown section. Read 'sections' to list available sections."
            )
        if page > HISTORY_PAGES:
            return response("This remote history section reached its page limit.")

        key = (section, page)
        with self._lock:
            if key not in self._cache:
                try:
                    self._cache[key] = self._history[section](page)
                except (GitHubError, KeyError, TypeError, ValueError):
                    return response("History page is unavailable.")
            content, more = self._cache[key]
        return response(content, more and page < HISTORY_PAGES)


def _comment_summary(rows) -> str:
    chunks = []
    for row in rows:
        if not isinstance(row, dict) or row.get("state", "SUBMITTED") not in {
            "SUBMITTED",
            "COMMENTED",
            "CHANGES_REQUESTED",
            "APPROVED",
            "DISMISSED",
        }:
            continue
        body = row.get("body")
        if isinstance(body, str) and body:
            actor = row.get("author") or row.get("user") or {}
            name = excerpt(str(actor.get("login") or "commenter"), 80)
            chunks.append(f"{name}\n{excerpt(body, 800)}")
    return "\n\n".join(chunks) if chunks else "No comments on this page."


@dataclass
class _ThreadHistory:
    github: Any
    thread: str
    _cursor: str | None = None
    _pages: list[tuple[str, bool]] = field(default_factory=list)

    def __call__(self, page: int) -> tuple[str, bool]:
        while len(self._pages) < page:
            if self._pages and not self._pages[-1][1]:
                return "No comments on this page.", False
            payload = self.github.graphql(
                _THREAD_PAGE, {"thread": self.thread, "cursor": self._cursor}
            )
            connection = payload["node"]["comments"]
            info = connection["pageInfo"]
            after = info.get("endCursor")
            more = bool(
                info.get("hasNextPage")
                and isinstance(after, str)
                and after != self._cursor
            )
            self._pages.append((_comment_summary(connection["nodes"]), more))
            self._cursor = after
        return self._pages[page - 1]


def github_history(
    github, repo: str, issue: int, pull: int | None, feedback: Sequence[Mapping]
) -> dict:
    """Bind history reads to this job's repository, discussion and review nodes."""
    repo = validate_repository(repo)
    if (
        type(issue) is not int
        or issue <= 0
        or (pull is not None and (type(pull) is not int or pull <= 0))
    ):
        raise ValueError("Invalid task discussion number")

    def comments(path: str) -> HistoryLoader:
        def fetch(page: int) -> tuple[str, bool]:
            rows = github.rest(
                "GET",
                path,
                params={"per_page": HISTORY_ROWS, "page": page},
            )
            if not isinstance(rows, list):
                raise ValueError("Invalid comment page")
            return _comment_summary(rows[:HISTORY_ROWS]), len(rows) == HISTORY_ROWS

        return fetch

    history = {"issue-comments": comments(f"/repos/{repo}/issues/{issue}/comments")}
    if pull is not None:
        if pull != issue:
            history["pull-comments"] = comments(f"/repos/{repo}/issues/{pull}/comments")
        history["pull-reviews"] = comments(f"/repos/{repo}/pulls/{pull}/reviews")
    seen = set()
    for index, item in enumerate(feedback, 1):
        thread = item.get("thread")
        if (
            pull is not None
            and item.get("reply_to")
            and isinstance(thread, str)
            and thread
            and thread not in seen
        ):
            seen.add(thread)
            history[f"review-thread-{index}"] = _ThreadHistory(github, thread)
    return history


# Only inspect tracked source files a reviewer explicitly identified. Load them
# from the pinned Git tree so links or paths in a review cannot escape checkout.
_REVIEW_FILE = re.compile(r"`([\w./-]+\.py)(?:::(?:[\w.]+::)*[\w.]+)?`")


def review_source_context(feedback, read_source, test_selector=None):
    previews = {}
    files = {}
    for item in feedback:
        anchors = [(item.get("path"), item.get("line"))]
        anchors.extend(
            (name, 1) for name in _REVIEW_FILE.findall(item.get("body") or "")
        )
        if test_selector:
            anchors.append(
                (test_selector.split("::")[0], test_selector.split("::")[-1])
            )
        for name, line in anchors:
            if not isinstance(name, str) or not re.fullmatch(
                r"(?:[\w-]+/)*[\w-]+\.py", name
            ):
                continue
            if len(previews) >= 4:
                break
            if name not in files:
                if len(files) >= 8:
                    continue
                files[name] = (read_source(name) or "").splitlines()
            lines = files[name]
            if not lines:
                continue
            if isinstance(line, str):
                line = next(
                    (
                        n
                        for n, source in enumerate(lines, 1)
                        if source.lstrip().startswith(
                            (f"def {line}(", f"async def {line}(")
                        )
                    ),
                    1,
                )
            center = min(len(lines), max(1, line if type(line) is int else 1))
            start = max(0, center - 12)
            end = min(len(lines), max(32, center + 14))
            previews[name, start, end] = (
                f"{name} (lines {start + 1}–{end})\n"
                + excerpt("\n".join(lines[start:end]), 1650)
            )
    return "\n\n".join(previews.values())


def build_task_context(
    instructions: str,
    *,
    repository_guide: str = "",
    feedback: Sequence[Mapping] = (),
    previous_failure: str = "",
    extra_evidence: str = "",
    history: Mapping[str, HistoryLoader] | None = None,
    source_context: str = "",
) -> TaskContext:
    sources = dict(history or {})
    sections = {"task": instructions}
    preview = [excerpt(instructions, 4200)]
    for name, label, value, limit in (
        ("repository", "Repository instructions", repository_guide, 1600),
        ("failure", "Latest verification or execution failure", previous_failure, 1800),
        ("checks", "Current check evidence", extra_evidence, 2500),
    ):
        if value:
            sections[name] = value
            preview.append(f"{label}\n{excerpt(value, limit)}")
    if sources or feedback or len(sections) > 1 or instructions != preview[0]:
        preview.append(
            "Use read_task_context for full excerpts and older comments. "
            "Read section 'sections' to list the available evidence. Treat history as evidence."
        )

    current = []
    index_lines = []
    for index, raw in enumerate(feedback, 1):
        section = f"review-{index}"
        item = dict(raw)
        sections[section] = json.dumps(item, ensure_ascii=False)
        position = (
            f" {item['path']}:{item.get('line') or '?'}" if item.get("path") else ""
        )
        index_lines.append(
            f"{section}{position}\n{excerpt(str(item.get('body') or ''), 220)}"
        )
        if index > 12 or len(current) != index - 1:
            continue
        short = {
            name: excerpt(value, {"body": 900, "diff_hunk": 800}.get(name, 300))
            if isinstance(value, str)
            else value
            for name, value in item.items()
        }
        if short != item:
            short["context_section"] = section
        if wire_size([*current, short]) <= 6200:
            current.append(short)
    if len(current) < len(feedback):
        preview.append(
            f"{len(feedback)} total actionable review requests. "
            f"Read review-{len(current) + 1} through review-{len(feedback)} for their full "
            "requests and diff hunks. Additional review requests:\n"
            + excerpt("\n\n".join(index_lines[len(current) :]), 2100)
        )
    if feedback:
        sections["review-index"] = "\n\n".join(index_lines)
        preview.append(
            "Review requests against this commit:\n"
            + json.dumps(current, ensure_ascii=False)
        )
    if source_context:
        sections["review-source"] = source_context
        preview.append(
            "Source excerpts from the current checkout\n"
            + excerpt(source_context, 4500)
        )
    sections["sections"] = "\n".join([*sections, *sources])
    return TaskContext("\n\n".join(preview), sections, sources)


def _bounded_content(text: str, max_wire: int) -> str:
    if max_wire < 2:
        return ""
    size = max_wire
    while wire_size(text) > max_wire:
        text = excerpt(text, size)
        size //= 2
    return text


def context_tool_result(context: TaskContext, call: Mapping[str, Any]) -> str:
    try:
        request = json.loads(call["function"]["arguments"])
        if not isinstance(request, dict) or request.keys() != {"section", "page"}:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        return "Provide a section and positive integer page to read_task_context."
    return json.dumps(
        context.read(request["section"], request["page"]), ensure_ascii=False
    )


def _editor_content(text: str, max_wire: int) -> str:
    try:
        payload = json.loads(text)
    except ValueError:
        return _bounded_content(text, max_wire)
    if not isinstance(payload, dict) or not isinstance(payload.get("evidence"), dict):
        return _bounded_content(text, max_wire)
    evidence = payload["evidence"]
    compact = {
        key: _bounded_content(
            str(payload[key]), max_wire // (3 if key == "task" else 7)
        )
        for key in ("task", "mission", "current_guidance")
        if key in payload
    }
    compact.update(
        seed=payload.get("seed"), evidence={}, evidence_omitted=len(evidence)
    )
    for name, value in reversed(evidence.items()):
        compact["evidence"][excerpt(name, 240)] = _bounded_content(str(value), 1800)
        compact["evidence_omitted"] -= 1
        if wire_size(json.dumps(compact, ensure_ascii=False)) > max_wire:
            compact["evidence"].pop(excerpt(name, 240))
            compact["evidence_omitted"] += 1
            break
    return json.dumps(compact, ensure_ascii=False)


def _review_file_windows(source: str, anchors: list[int], budget: int) -> str:
    if wire_size(source) <= budget or re.match(r"^Lines \d+–\d+\n", source):
        return _bounded_content(source, budget)
    lines = source.splitlines(keepends=True)
    windows = []
    per_anchor = budget // len(anchors) - 40
    for line in anchors:
        if not 1 <= line <= len(lines):
            continue
        start, end = max(0, line - 7), min(len(lines), line + 6)
        before = _bounded_content("".join(lines[start : line - 1]), per_anchor // 3)
        anchor = _bounded_content(lines[line - 1], per_anchor // 3)
        after = _bounded_content("".join(lines[line:end]), per_anchor // 3)
        windows.append(f"Lines {start + 1}–{end}\n{before}{anchor}{after}")
    return "\n\n".join(windows) if windows else _bounded_content(source, budget)


def _review_content(text: str, max_wire: int) -> str:
    if wire_size(text) <= max_wire:
        return text
    prefix, marker, sources = text.partition("\nCurrent reviewed files\n")
    if not marker or "\nOutstanding review\n" not in prefix:
        return _bounded_content(text, max_wire)
    sources, acceptance_marker, acceptance = sources.partition(
        "\n\nOperator acceptance test source "
    )
    sources, span_marker, spans = sources.partition("\n\nReviewed source spans\n")
    anchors: dict[str, list[int]] = {}
    for path, line in re.findall(r"(?m)^- ([^\n:]+):(\d{1,9}) ", prefix):
        positions = anchors.setdefault(path, [])
        if len(positions) < 3 and int(line) not in positions:
            positions.append(int(line))
    if not anchors:
        return _bounded_content(text, max_wire)
    pieces = re.split(
        r"(?:^|\n\n)(" + "|".join(re.escape(path) for path in anchors) + r")\n",
        sources,
    )
    if len(pieces) < 3 or pieces[0]:
        return _bounded_content(text, max_wire)

    review, rejection_marker, rejection = prefix.partition(
        "\n\nMost recent rejection\n"
    )
    rejection, checks_marker, checks = rejection.partition(
        "\n\nChecks still required\n"
    )
    diagnosis = _bounded_content(rejection_marker + rejection, max_wire // 8)
    check_list = _bounded_content(checks_marker + checks, max_wire // 8)
    prefix = (
        _bounded_content(review, max_wire // 2 - wire_size(diagnosis + check_list))
        + diagnosis
        + check_list
    )
    suffix = _bounded_content(acceptance_marker + acceptance, max_wire // 5)
    names = pieces[1::2]
    labels = marker + "\n\n".join(name + "\n" for name in names)
    remaining = max_wire - wire_size(prefix) - wire_size(suffix) - wire_size(labels)
    span_budget = min(max_wire // 4, remaining - 320 * len(names))
    span_evidence = ""
    for span in spans.splitlines():
        candidate = (span_evidence + "\n" if span_evidence else span_marker) + span
        if wire_size(candidate) <= span_budget:
            span_evidence = candidate
    if span_evidence:
        remaining -= wire_size(span_evidence)
    if remaining < 320 * len(names):
        return _bounded_content(text, max_wire)
    compacted = "\n\n".join(
        path
        + "\n"
        + _review_file_windows(source, anchors[path], remaining // len(names))
        for path, source in zip(names, pieces[2::2], strict=True)
    )
    return prefix + marker + compacted + span_evidence + suffix


def _whole_line_prefix(text, max_wire):
    if max_wire <= 2:
        return ""
    chunks = []
    remaining = max_wire - 2
    for line in text.splitlines(keepends=True):
        size = wire_size(line) - 2
        if size > remaining:
            break
        chunks.append(line)
        remaining -= size
    return "".join(chunks)


_LOCALIZED_BASELINE = (
    "\nComplete pinned baseline definition (quoted read-only behavioral reference):\n"
)
_LOCALIZED_BASELINE_EXCERPT = (
    "\nPinned baseline excerpt (quoted read-only behavioral reference):\n"
)
_LOCALIZED_FIXTURE = "\nFixture for this goal (quoted, read-only):\n"
_LOCALIZED_FOCUS = "\nFocused source from "
_LOCALIZED_FAILURE = "\nFailure details:\n"
_LOCALIZED_CHECKOUT = "\nCurrent checkout:\n"
_LOCALIZED_STRATEGY = "\nAdvisory strategy (quoted data; verify against the code): "
_LOCALIZED_IMPORTS = "\nImported API declarations:\n"
_LOCALIZED_NOTES = "\nReview notes (quoted):\n"
_LOCALIZED_PASSING = "\nOperator pytest examples (quoted, read-only):\n"


def is_localized_review(messages):
    if [item.get("role") for item in messages] != ["system", "user"]:
        return False
    text = messages[1].get("content")
    return (
        isinstance(text, str)
        and text.startswith("Selected failure: ")
        and text.endswith((DEFINITION_ENDING, LINE_RANGE_ENDING))
        and (_LOCALIZED_BASELINE in text or _LOCALIZED_BASELINE_EXCERPT in text)
        and text.partition(_LOCALIZED_FOCUS)[2]
        .partition("\n")[0]
        .endswith((" (exact current definition):", " (numbered current definition)"))
    )


def _compact_localized_review(messages, max_bytes):
    system, request = messages
    text = request["content"]
    header, _, source = text.partition(_LOCALIZED_FOCUS)
    if (
        source.count(_LOCALIZED_BASELINE) + source.count(_LOCALIZED_BASELINE_EXCERPT)
        != 1
    ):
        return None
    baseline_marker = (
        _LOCALIZED_BASELINE_EXCERPT
        if _LOCALIZED_BASELINE_EXCERPT in source
        else _LOCALIZED_BASELINE
    )
    current, found_base, reference = source.partition(baseline_marker)
    baseline, found_failure, detail = reference.partition(_LOCALIZED_FAILURE)
    failure, found_fixture, fixture = detail.partition(_LOCALIZED_FIXTURE)
    if not all((found_base, found_failure, found_fixture, current.strip())):
        return None
    checkout, found_checkout, footer = fixture.partition(_LOCALIZED_CHECKOUT)
    if not found_checkout:
        return None
    fixture, _, context = checkout.partition(_LOCALIZED_IMPORTS)
    imports, notes_marker, context = context.partition(_LOCALIZED_NOTES)
    notes, passing_marker, passing = context.partition(_LOCALIZED_PASSING)
    header, strategy_marker, strategy = header.partition(_LOCALIZED_STRATEGY)
    # Keep a recognizable layout so budget-level compaction can be followed
    # by a smaller provider-specific compaction without losing the method.
    layout = (
        (strategy_marker, strategy, 8),
        (_LOCALIZED_BASELINE_EXCERPT, baseline, 10),
        (_LOCALIZED_FAILURE, failure, 20),
        (_LOCALIZED_FIXTURE, fixture, 25),
        (_LOCALIZED_IMPORTS, imports, 2),
        (_LOCALIZED_NOTES, notes if notes_marker else "", 10),
        (_LOCALIZED_PASSING, passing if passing_marker else "", 25),
    )

    def payload(excerpts):
        advisory, *evidence = excerpts
        return [
            system,
            {
                **request,
                "content": (
                    header
                    + advisory
                    + _LOCALIZED_FOCUS
                    + current
                    + "".join(evidence)
                    + _LOCALIZED_CHECKOUT
                    + footer
                ),
            },
        ]

    markers = [heading for heading, _, _ in layout]
    spare = max_bytes - wire_size(payload(markers))
    if spare < 0:
        return None
    excerpts = [
        heading + _bounded_content(body, spare * share // 100)
        for heading, body, share in layout
    ]
    compact = payload(excerpts)
    return compact if wire_size(compact) <= max_bytes else None


def compact_focused_review(messages, *, max_bytes):
    """Retain the active review goal and focused source while dropping bulk context."""
    if [item.get("role") for item in messages] != ["system", "user"] or wire_size(
        messages
    ) <= max_bytes:
        return None
    if is_localized_review(messages):
        return _compact_localized_review(messages, max_bytes)
    system, request = messages
    text = request.get("content")
    if not isinstance(text, str) or not text.startswith("Selected failure: "):
        return None
    context, checkout_marker, _ = text.partition("\nCurrent checkout:\n")
    goal, _, _ = context.partition("\nOther reviewer feedback:\n")
    endings = ("Respond with exact SEARCH/REPLACE blocks only.", DEFINITION_ENDING)
    ending = next((value for value in endings if text.endswith(value)), None)
    if not checkout_marker or "\nFocused source from " not in goal or ending is None:
        return None
    test_path = next(
        (
            line
            for line in reversed(context.splitlines())
            if line.startswith("New test file: ")
        ),
        "",
    )
    footer = f"\n{test_path}\n{ending}"
    envelope = [system, {**request, "content": footer}]
    budget = max_bytes - wire_size(envelope)
    if budget <= 0:
        return None
    lead, marker, fixture = goal.partition("\nFixture for this goal:\n")
    header, focus_marker, source = lead.partition("\nFocused source from ")
    source, diagnostic_marker, diagnostic = source.partition("\nFailure details:\n")
    if diagnostic_marker and marker:
        fixture, _, _ = fixture.partition("\nImported API declarations:\n")
        selected = _whole_line_prefix(marker + fixture, budget // 4)
        summary = _bounded_content(header, budget // 8)
        details = _bounded_content(diagnostic_marker + diagnostic, budget // 6)
        source_budget = budget - wire_size(summary + details + selected)
        focused = _whole_line_prefix(focus_marker + source, source_budget)
        kept = summary + focused + details + selected
    else:
        kept = _bounded_content(lead, budget)
        kept += _bounded_content(marker + fixture, budget - wire_size(kept))
    compact = [system, {**request, "content": kept + footer}]
    return compact if wire_size(compact) <= max_bytes else None


def compact_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    max_bytes: int = MESSAGE_BYTES,
    role: str = "",
    review_windows: bool = False,
) -> list[Mapping[str, Any]]:
    """Keep a bounded prompt and a suffix of complete, unchanged tool turns."""
    if wire_size(messages) <= max_bytes:
        return list(messages)
    if is_localized_review(messages):
        return _compact_localized_review(messages, max_bytes) or list(messages)

    prefix = []
    index = 0
    while index < min(2, len(messages)) and messages[index].get("role") in {
        "system",
        "user",
    }:
        original = messages[index]
        content = original.get("content") or ""
        size = max_bytes * 2 // 3 if original["role"] == "user" else max_bytes // 8
        compact = _bounded_content
        if original["role"] == "user":
            if role in {"proposal", "composition"}:
                compact = _editor_content
            elif review_windows:
                compact = _review_content
        compacted = compact(content, size)
        prefix.append({**original, "content": compacted})
        index += 1

    groups: list[list[Mapping[str, Any]]] = []
    for message in messages[index:]:
        if message.get("role") == "tool" and groups and groups[-1][0].get("tool_calls"):
            groups[-1].append(message)
        else:
            groups.append([message])
    retained: list[Mapping[str, Any]] = []
    kept_groups = 0
    for group in reversed(groups):
        calls = group[0].get("tool_calls")
        if calls and [call["id"] for call in calls] != [
            msg.get("tool_call_id") for msg in group[1:]
        ]:
            continue
        if group[0].get("role") == "tool":
            continue
        if wire_size([*prefix, *group, *retained]) > max_bytes - 1200:
            break
        retained[:0] = group
        kept_groups += 1
    omitted = len(groups) - kept_groups
    if omitted and prefix and prefix[-1]["role"] == "user":
        failure = next(
            (
                message["content"]
                for message in reversed(messages[index:])
                if message.get("role") == "tool"
                and isinstance(message.get("content"), str)
                and re.match(r"^exit [1-9][0-9]*\b", message["content"])
            ),
            "",
        )
        summary = (
            f"\n\n{omitted} earlier conversation groups were omitted to stay within the request limit. "
            "Workspace edits remain available. Use bounded checkout reads to recover relevant details."
        )
        if failure:
            summary += "\nLatest failed command output excerpt:\n" + _bounded_content(
                failure, 640
            )
        prefix[-1]["content"] += summary
    return [*prefix, *retained]

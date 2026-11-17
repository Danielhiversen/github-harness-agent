"""Collect current review feedback and recover replies after interrupted writes."""

import hashlib
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from .github import GitHubError, validate_repository
from .publish import ClosedPull, Journal, StaleHead


_REVIEW_STATES = {"COMMENTED", "CHANGES_REQUESTED"}
_PAGE = "pageInfo { hasNextPage endCursor }"
_COMMENT = """
    fullDatabaseId author { login __typename } body lastEditedAt publishedAt state diffHunk
    replyTo { fullDatabaseId }
    pullRequestReview { state submittedAt author { login __typename } }
"""
_REVIEW_BODIES = """
query ReviewBodies($owner: String!, $repo: String!, $pull: Int!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $pull) {
      state headRefOid
      reviews(first: 100, after: $cursor) {
        nodes { fullDatabaseId author { login __typename } body lastEditedAt submittedAt state }
        PAGE
      }
    }
  }
}
""".replace("PAGE", _PAGE)
_REVIEW_THREADS = """
query ReviewThreads($owner: String!, $repo: String!, $pull: Int!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $pull) {
      state headRefOid
      reviewThreads(first: 50, after: $cursor) {
        nodes {
          id path line originalLine isResolved isOutdated diffSide
          comments(first: 20) { nodes { COMMENT } PAGE }
        }
        PAGE
      }
    }
  }
}
""".replace("COMMENT", _COMMENT).replace("PAGE", _PAGE)
_THREAD_COMMENTS = """
query ThreadComments($thread: ID!, $cursor: String!) {
  node(id: $thread) {
    ... on PullRequestReviewThread {
      isResolved
      comments(first: 100, after: $cursor) { nodes { COMMENT } PAGE }
    }
  }
}
""".replace("COMMENT", _COMMENT).replace("PAGE", _PAGE)
_REPLY_TARGET = """
query ReplyTarget($thread: ID!) {
  node(id: $thread) { ... on PullRequestReviewThread { isResolved } }
}
"""


@dataclass(frozen=True)
class Feedback:
    key: str
    thread: str
    reply_to: int | None
    body: str
    author: str
    path: str | None = None
    line: int | None = None
    outdated: bool = False
    original_line: int | None = None
    diff_hunk: str | None = None
    diff_side: str | None = None


ReviewItem = Feedback


@dataclass(frozen=True)
class ReviewBatch:
    head: str
    items: tuple[Feedback, ...]
    retry_at: datetime | None = None


def map_feedback(item: Feedback, source: str | None) -> Feedback | None:
    """Map an outdated anchor to a unique line in the pinned file."""
    if not item.outdated:
        return item
    if source is None:
        return None
    if item.original_line is None:
        return item if item.line is None else None
    if not item.diff_hunk or item.diff_side not in {"LEFT", "RIGHT"}:
        return None
    hunk_line = None
    context = {}
    excluded = "+" if item.diff_side == "LEFT" else "-"
    for line in item.diff_hunk.split("\n"):
        header = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if header:
            hunk_line = int(header[1 if item.diff_side == "LEFT" else 2])
            continue
        if (
            hunk_line is not None
            and line[:1] in {" ", "+", "-"}
            and line[0] != excluded
        ):
            context[hunk_line] = line[1:].removesuffix("\r")
            hunk_line += 1
    anchor = context.get(item.original_line)
    if anchor is None or not anchor.strip():
        return None

    lines = [line.removesuffix("\r") for line in source.split("\n")]
    candidates = []
    for index, content in enumerate(lines):
        if content != anchor:
            continue
        score = sum(
            0 <= index + delta < len(lines)
            and lines[index + delta] == context.get(item.original_line + delta)
            for delta in (-3, -2, -1, 1, 2, 3)
        )
        candidates.append((score, index + 1))
    if not candidates:
        return None
    best = max(score for score, _ in candidates)
    matches = [line for score, line in candidates if score == best]
    return replace(item, line=matches[0]) if len(matches) == 1 else None


def _digest(*values) -> str:
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode()
    ).hexdigest()


def _author(record: dict, bot_login: str) -> str | None:
    author = record.get("author") or {}
    login = author.get("login") or ""
    if (
        not login
        or login.casefold() == bot_login.casefold()
        or login.casefold().endswith("[bot]")
        or author.get("__typename") == "Bot"
    ):
        return None
    return login


def _review_submitted(review: dict | None) -> bool:
    return bool(
        review and review["state"] in _REVIEW_STATES and review.get("submittedAt")
    )


def _next(connection: dict, previous: set[str]) -> str | None:
    page = connection["pageInfo"]
    if not page["hasNextPage"]:
        return None
    cursor = page["endCursor"]
    if not cursor or cursor in previous:
        raise GitHubError("GitHub review pagination repeated or omitted a cursor")
    previous.add(cursor)
    return cursor


def _threads(github, connection):
    for thread in connection["nodes"]:
        if not thread or thread["isResolved"]:
            continue
        comments = thread["comments"]
        items = [item for item in comments["nodes"] if item]
        visited = set()
        cursor = _next(comments, visited)
        while cursor is not None:
            result = github.graphql(
                _THREAD_COMMENTS, {"thread": thread["id"], "cursor": cursor}
            )
            node = result["node"]
            if node is None or node["isResolved"]:
                items = []
                break
            comments = node["comments"]
            items.extend(item for item in comments["nodes"] if item)
            cursor = _next(comments, visited)
        if items and not items[0].get("replyTo"):
            yield thread, items


def collect_reviews(
    github,
    *,
    repo: str,
    pull: int,
    bot_login: str,
    authorized: Callable[[str], bool],
    known_threads: Iterable[str] = (),
    seen: Iterable[str] = (),
    debounce_seconds: int = 20,
    now: datetime | None = None,
) -> ReviewBatch:
    owner, name = validate_repository(repo).split("/")
    params = {"owner": owner, "repo": name, "pull": pull}
    previous_keys = set(seen)
    known = set(known_threads)
    now = now or datetime.now(timezone.utc)
    head = None
    items = []
    retry_at = None
    reviewers = {}

    def allowed(login):
        if login not in reviewers:
            reviewers[login] = authorized(login)
        return reviewers[login]

    def pages(query, field):
        nonlocal head
        cursor = None
        visited = set()
        while True:
            result = github.graphql(query, {**params, "cursor": cursor})
            request = (result.get("repository") or {}).get("pullRequest")
            if request is None or request["state"] != "OPEN":
                raise ClosedPull("pull request is closed or unavailable")
            if head is not None and request["headRefOid"] != head:
                raise StaleHead("pull request head changed while collecting reviews")
            head = request["headRefOid"]
            connection = request[field]
            yield connection
            cursor = _next(connection, visited)
            if cursor is None:
                return

    for page in pages(_REVIEW_BODIES, "reviews"):
        for review in page["nodes"]:
            if not _review_submitted(review):
                continue
            login = _author(review, bot_login)
            body = review["body"].strip()
            if not login or not body or not allowed(login):
                continue
            thread = f"review:{review['fullDatabaseId']}"
            key = f"{thread}:{_digest(body, review.get('lastEditedAt'))}"
            if key not in previous_keys:
                items.append(Feedback(key, thread, None, body, login))

    for page in pages(_REVIEW_THREADS, "reviewThreads"):
        for thread, comments in _threads(github, page):
            root = int(comments[0]["fullDatabaseId"])
            for comment in comments:
                login = _author(comment, bot_login)
                body = comment["body"].strip()
                if (
                    comment["state"] != "SUBMITTED"
                    or not login
                    or not body
                    or not allowed(login)
                ):
                    continue
                identifier = comment["fullDatabaseId"]
                edited_at = comment.get("lastEditedAt")
                key = f"comment:{identifier}:{_digest(body, edited_at)}"
                if key in previous_keys:
                    continue
                review = comment.get("pullRequestReview")
                if review is None:
                    if thread["id"] not in known or not comment.get("replyTo"):
                        continue
                elif not _review_submitted(review):
                    continue
                if comment.get("replyTo"):
                    published = edited_at or comment.get("publishedAt")
                    if not published:
                        continue
                    due = datetime.fromisoformat(published.replace("Z", "+00:00"))
                    due += timedelta(seconds=debounce_seconds)
                    if due > now:
                        retry_at = due if retry_at is None else min(retry_at, due)
                        continue
                items.append(
                    Feedback(
                        key,
                        thread["id"],
                        root,
                        body,
                        login,
                        thread["path"],
                        thread["line"],
                        thread["isOutdated"],
                        thread["originalLine"],
                        comments[0]["diffHunk"],
                        thread.get("diffSide"),
                    )
                )

    return ReviewBatch(head, tuple(items), retry_at)


def reply_to_review(
    github,
    journal: Journal,
    authorize: Callable[[], None],
    *,
    repo: str,
    pull: int,
    item: Feedback,
    commit: str | None,
    checks: Iterable[str],
    bot_login: str,
    question: str | None = None,
    expected_head: str | None = None,
) -> int | None:
    repo = validate_repository(repo)
    if bool(commit) == bool(question):
        raise ValueError(
            "a review reply requires a verified commit or a clarification question"
        )
    attempt = _digest(repo, pull, item.thread, commit or item.key)
    key = f"reply:{attempt}"
    marker = f"<!-- github-harness-agent:{attempt} -->"
    if item.reply_to is None:
        path = f"/repos/{repo}/issues/{pull}/comments"
        post_path = path
    else:
        path = f"/repos/{repo}/pulls/{pull}/comments"
        post_path = f"{path}/{item.reply_to}/replies"
    body = question
    if commit:
        evidence = "\n".join(f"- {check}" for check in checks)
        if not evidence:
            raise ValueError("a review commit reply requires verification evidence")
        body = f"Addressed this feedback in `{commit}`.\n\nChecks\n\n{evidence}"
    if item.reply_to is None:
        review_id = item.thread.removeprefix("review:")
        body = f"Following up on review #{review_id}.\n\n{body}"
    intent = {"path": post_path, "body": f"{body}\n\n{marker}"}

    def lookup():
        for comment in github.paginate(path):
            login = (comment.get("user") or {}).get("login", "")
            if (
                login.casefold() == bot_login.casefold()
                and comment.get("in_reply_to_id") == item.reply_to
                and comment["body"].rstrip().endswith(marker)
            ):
                return comment
        return None

    authorize()
    current = github.rest("GET", f"/repos/{repo}/pulls/{pull}")
    if current["state"] != "open":
        raise ClosedPull("pull request is closed")
    head = expected_head or commit
    if head is not None and current["head"]["sha"] != head:
        raise StaleHead("pull request head changed before replying")
    journal.begin(key, intent)
    entry = journal.get(key)
    if "result" in entry:
        return entry["result"]["id"]
    comment = lookup()
    if comment is None:
        if item.reply_to is not None:
            node = github.graphql(_REPLY_TARGET, {"thread": item.thread}).get("node")
            if not node or node["isResolved"]:
                journal.finish(key, {"id": None})
                return None
        authorize()
        try:
            comment = github.rest("POST", post_path, json={"body": intent["body"]})
        except GitHubError:
            comment = lookup()
            if comment is None:
                raise
    journal.finish(key, {"id": comment["id"]})
    return comment["id"]

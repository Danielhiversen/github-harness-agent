"""Collect current review feedback and recover replies after interrupted writes."""

import hashlib
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from .github import GitHubError, validate_repository
from .publish import ClosedPull, Journal, StaleHead


_REVIEW_STATES = {"APPROVED", "COMMENTED", "CHANGES_REQUESTED"}
_PAGE = "pageInfo { hasNextPage endCursor }"
_COMMENT = """
    fullDatabaseId author { login __typename } body lastEditedAt publishedAt state diffHunk
    replyTo { fullDatabaseId }
    pullRequestReview { fullDatabaseId state submittedAt author { login __typename } }
"""
_REVIEW_BODIES = """
query ReviewBodies($owner: String!, $repo: String!, $pull: Int!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $pull) {
      state headRefOid
      reviews(first: 100, after: $cursor) {
        nodes {
          fullDatabaseId author { login __typename } body lastEditedAt submittedAt state commit { oid }
        }
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
    test_update_head: str | None = None
    revision_at: float | None = None


ReviewItem = Feedback


@dataclass(frozen=True)
class ReviewBatch:
    head: str
    items: tuple[Feedback, ...]
    retry_at: datetime | None = None
    handled: tuple[Feedback, ...] = ()


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


def _review_time(value: str | None) -> datetime | None:
    if value:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return None


def _review_revision(record):
    return _review_time(record.get("lastEditedAt") or record.get("submittedAt"))


def _review_commit(record):
    return (record.get("commit") or {}).get("oid")


_ADDRESSING_REVIEW = re.compile(
    r"^(?:addressed|addressing|implemented|fixed)\s+(?:the\s+)?(?:review feedback|this feedback)\b",
    re.IGNORECASE,
)
_REVIEW_REFERENCE = re.compile(
    r"(?:pullrequestreview-|\breview\s+#)([1-9]\d*)\b",
    re.IGNORECASE,
)
_INLINE_CONTEXT = re.compile(r"\bI (?:found|noticed|spotted)\b", re.IGNORECASE)
_BODY_REQUEST = re.compile(
    r"\b(?:please|should|must|can you|could you|add|remove|fix|change|use|make|preserve|update|handle|reject)\b",
    re.IGNORECASE,
)


def _prior_review_replies(conversation, reviews, candidates, bot_login):
    """Match replies by review reference or a verified account's follow-up review."""
    replies = {}
    for comment in conversation:
        login = (comment.get("user") or {}).get("login", "")
        body = comment.get("body") or ""
        if login.casefold() != bot_login.casefold() or not any(
            _ADDRESSING_REVIEW.match(line.strip()) for line in body.splitlines()
        ):
            continue
        posted = _review_time(comment.get("created_at"))
        for identifier in _REVIEW_REFERENCE.findall(body):
            if posted and (identifier not in replies or posted > replies[identifier]):
                replies[identifier] = posted

    for review in reviews:
        author = (review.get("author") or {}).get("login", "")
        posted = _review_revision(review)
        commit = _review_commit(review)
        if (
            author.casefold() != bot_login.casefold()
            or not posted
            or not commit
            or not _ADDRESSING_REVIEW.match((review.get("body") or "").strip())
        ):
            continue
        previous = [
            (when, identifier)
            for identifier, original in candidates.items()
            if (when := _review_revision(original))
            and when < posted
            and _review_commit(original)
            and _review_commit(original) != commit
        ]
        if previous:
            _, identifier = max(previous)
            if identifier not in replies or posted > replies[identifier]:
                replies[identifier] = posted
    return replies


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
    adopted_at: str | None = None,
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
    history = []
    body_items = {}
    inline_items = []
    current_threads = set()
    cutoff = _review_time(adopted_at)

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
            if review:
                history.append(review)
            if not _review_submitted(review):
                continue
            login = _author(review, bot_login)
            body = review["body"].strip()
            if not login or not body or not allowed(login):
                continue
            if review["state"] == "APPROVED" and not _BODY_REQUEST.search(body):
                continue
            thread = f"review:{review['fullDatabaseId']}"
            key = f"{thread}:{_digest(body, review.get('lastEditedAt'))}"
            if key not in previous_keys:
                test_update_head = (
                    head
                    if adopted_at and (review.get("commit") or {}).get("oid") == head
                    else None
                )
                body_items[str(review["fullDatabaseId"])] = Feedback(
                    key,
                    thread,
                    None,
                    review["body"],
                    login,
                    test_update_head=test_update_head,
                    revision_at=_review_revision(review).timestamp(),
                )

    for page in pages(_REVIEW_THREADS, "reviewThreads"):
        for thread, comments in _threads(github, page):
            root = int(comments[0]["fullDatabaseId"])
            root_review = comments[0].get("pullRequestReview") or {}
            review_id = str(root_review.get("fullDatabaseId") or "")
            root_login = _author(comments[0], bot_login)
            if (
                cutoff
                and root_login
                and allowed(root_login)
                and _review_submitted(root_review)
            ):
                known.add(thread["id"])
                if review_id and not thread["isOutdated"]:
                    current_threads.add(review_id)
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
                comment_id = comment["fullDatabaseId"]
                edited_at = comment.get("lastEditedAt")
                key = f"comment:{comment_id}:{_digest(body, edited_at)}"
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
                when = _review_time(edited_at or comment.get("publishedAt"))
                inline_items.append(
                    (
                        Feedback(
                            key,
                            thread["id"],
                            root,
                            comment["body"],
                            login,
                            thread["path"],
                            thread["line"],
                            thread["isOutdated"],
                            thread["originalLine"],
                            comments[0]["diffHunk"],
                            thread.get("diffSide"),
                            revision_at=when.timestamp() if when else None,
                        ),
                        review_id,
                        when,
                        bool(comment.get("replyTo")),
                    )
                )

    conversation = list(github.paginate(f"/repos/{repo}/issues/{pull}/comments"))
    for comment in conversation:
        author = comment.get("user") or {}
        login = _author(
            {"author": {"login": author.get("login"), "__typename": author.get("type")}},
            bot_login,
        )
        body = (comment.get("body") or "").strip()
        revision = comment.get("updated_at") or comment.get("created_at")
        when = _review_time(revision)
        if not login or not body or not when or (cutoff and when <= cutoff):
            continue
        thread = f"conversation:{comment['id']}"
        key = f"{thread}:{_digest(body, revision)}"
        if key not in previous_keys and allowed(login):
            items.append(Feedback(key, thread, None, comment["body"], login, revision_at=when.timestamp()))

    if cutoff is None:
        return ReviewBatch(
            head,
            tuple(body_items.values()) + tuple(row[0] for row in inline_items) + tuple(items),
            retry_at,
        )

    requests = {
        str(review["fullDatabaseId"]): review
        for review in history
        if _review_submitted(review)
        and (login := _author(review, bot_login))
        and allowed(login)
    }
    current_reviews = {}
    for review in history:
        if (
            review["state"] not in _REVIEW_STATES
            or _review_commit(review) != head
        ):
            continue
        if login := _author(review, bot_login):
            revision = _review_revision(review)
            if revision and (
                login not in current_reviews or revision > current_reviews[login]
            ):
                current_reviews[login] = revision
    have_history = any(
        identifier in body_items
        and (when := _review_revision(review))
        and when <= cutoff
        for identifier, review in requests.items()
    ) or any(
        item.outdated and when and when <= cutoff for item, _, when, _ in inline_items
    )
    replied = (
        _prior_review_replies(conversation, history, requests, bot_login)
        if have_history
        else {}
    )

    def addressed(identifier, revision, login):
        if not revision or revision > cutoff:
            return False
        if identifier in replied and replied[identifier] >= revision:
            return True
        original = requests.get(identifier)
        return (
            identifier not in current_threads
            and original is not None
            and _review_commit(original) not in {None, head}
            and login in current_reviews
            and current_reviews[login] > revision
        )

    handled = []
    for identifier, item in body_items.items():
        if addressed(identifier, _review_revision(requests[identifier]), item.author):
            handled.append(item)
        elif (
            identifier in current_threads
            and len(item.body) <= 200
            and _INLINE_CONTEXT.search(item.body)
            and not _BODY_REQUEST.search(item.body)
        ):
            handled.append(item)
        else:
            items.append(item)
    for item, identifier, when, is_reply in inline_items:
        if item.outdated and not is_reply and addressed(identifier, when, item.author):
            handled.append(item)
        else:
            items.append(item)
    return ReviewBatch(head, tuple(items), retry_at, tuple(handled))


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
        kind, _, number = item.thread.partition(":")
        reference = f"PR comment #{number}" if kind == "conversation" else f"review #{number}"
        body = f"Following up on {reference}.\n\n{body}"
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

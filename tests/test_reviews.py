from datetime import datetime, timezone

import pytest


def connection(nodes, next_page=None):
    return {
        "nodes": nodes,
        "pageInfo": {"hasNextPage": bool(next_page), "endCursor": next_page},
    }


def review(number, *, body="", login="maintainer", state="COMMENTED", edited=None):
    return {
        "fullDatabaseId": str(number),
        "body": body,
        "author": {"login": login},
        "state": state,
        "submittedAt": "2026-01-01T00:00:00Z",
        "lastEditedAt": edited,
    }


def comment(
    number, *, body="Fix the sentence", login="maintainer", parent=None, created=None
):
    return {
        "fullDatabaseId": str(number),
        "body": body,
        "author": {"login": login},
        "state": "SUBMITTED",
        "publishedAt": created or "2026-01-01T00:00:00Z",
        "lastEditedAt": None,
        "diffHunk": "@@ -1 +1 @@\n-old\n+new",
        "replyTo": {"fullDatabaseId": str(parent)} if parent else None,
        "pullRequestReview": review(7) if parent is None else None,
    }


def thread(id_, comments, *, resolved=False):
    return {
        "id": id_,
        "path": "readme.md",
        "line": None,
        "originalLine": 1,
        "isResolved": resolved,
        "isOutdated": True,
        "comments": connection(comments),
    }


class ReviewGitHub:
    def __init__(self):
        self.reviews = [
            review(1, body="Include the example."),
            review(2, body="LGTM", state="APPROVED"),
        ]
        self.pages = [
            [thread("A", [comment(3), comment(5, login="test-app[bot]", parent=3)])],
            [thread("B", [comment(4)]), thread("C", [comment(6)], resolved=True)],
        ]
        self.state = "OPEN"
        self.head = "a" * 40
        self.comments = []
        self.fail_next = True

    def graphql(self, query, variables):
        if "ReplyTarget" in query:
            thread_ = next(
                item
                for page in self.pages
                for item in page
                if item["id"] == variables["thread"]
            )
            return {"node": {"isResolved": thread_["isResolved"]}}
        if "ReviewBodies" in query:
            result = {"reviews": connection(self.reviews)}
        else:
            page = int(variables["cursor"] or "0")
            result = {
                "reviewThreads": connection(
                    self.pages[page], "1" if page == 0 else None
                ),
            }
        return {
            "repository": {
                "pullRequest": {"headRefOid": self.head, "state": self.state, **result}
            }
        }

    def rest(self, method, path, *, json=None):
        from github_harness_agent.github import GitHubError

        if method == "GET":
            return {"state": self.state.lower(), "head": {"sha": self.head}}
        answer = {
            "id": 100 + len(self.comments),
            "body": json["body"],
            "user": {"login": "test-app[bot]"},
            "in_reply_to_id": 3,
        }
        self.comments.append(answer)
        if self.fail_next:
            self.fail_next = False
            raise GitHubError("connection lost during reply")
        return answer

    def paginate(self, path, *, params=None):
        return iter(self.comments)


def collect(github, **kwargs):
    from github_harness_agent.reviews import collect_reviews

    return collect_reviews(
        github,
        repo="test/repo",
        pull=6,
        bot_login="test-app[bot]",
        authorized=lambda login: login == "maintainer",
        **kwargs,
    )


def test_collects_new_maintainer_conversation_comments_and_edits():
    github = ReviewGitHub()
    github.reviews = []
    github.pages = [[], []]
    created = "2026-01-02T00:00:00Z"
    github.comments = [
        {"id": 101, "user": {"login": "maintainer"}, "body": "Fix the catalog.", "created_at": created},
        {"id": 102, "user": {"login": "visitor"}, "body": "Alter settings.", "created_at": created},
        {"id": 103, "user": {"login": "test-app[bot]"}, "body": "Reply.", "created_at": created},
        {"id": 104, "user": {"login": "maintainer"}, "body": "Old note.", "created_at": "2025-12-31T00:00:00Z"},
    ]
    options = {"adopted_at": "2026-01-01T00:00:00Z"}
    (item,) = collect(github, **options).items
    assert item.thread == "conversation:101" and item.reply_to is None
    assert collect(github, seen=[item.key], **options).items == ()
    github.comments[0].update(body="Fix the catalog and add a test.", updated_at="2026-01-03T00:00:00Z")
    (edited,) = collect(github, seen=[item.key], **options).items
    assert edited.thread == item.thread and edited.key != item.key


def test_approval_can_include_a_review_request():
    github = ReviewGitHub()
    github.reviews = [review(19, body="Please normalize the continent label.", state="APPROVED")]
    github.pages = [[], []]
    (item,) = collect(github).items
    assert item.thread == "review:19"


def test_collects_body_and_all_unresolved_threads_and_filters_bot_comments():
    github = ReviewGitHub()
    batch = collect(github)
    assert batch.head == github.head
    assert [item.thread for item in batch.items] == ["review:1", "A", "B"]
    assert batch.items[1].original_line == 1
    assert batch.items[1].outdated


def test_new_review_and_edited_body_remain_pending_after_previous_batch():
    github = ReviewGitHub()
    initial = collect(github)
    github.reviews.extend(
        [review(7, body="Add a test."), review(11, body="Update the example.")]
    )
    github.reviews[0] = review(
        1, body="Include both examples.", edited="2026-01-01T01:00:00Z"
    )
    revised = collect(github, seen=[item.key for item in initial.items])
    assert [item.body for item in revised.items] == [
        "Include both examples.",
        "Add a test.",
        "Update the example.",
    ]


def test_adopted_pull_distinguishes_handled_history_from_current_inline_feedback():
    github = ReviewGitHub()
    first = dict(
        review(10, body="Use the provider's first-seen time."), commit={"oid": "b" * 40}
    )
    reply = dict(
        review(
            11,
            login="test-app[bot]",
            body="Addressing review feedback: the cap now uses first seen.",
        ),
        submittedAt="2026-01-02T00:00:00Z",
        commit={"oid": "c" * 40},
    )
    older = dict(
        review(12, body="Retain timestamp collisions."),
        submittedAt="2026-01-03T00:00:00Z",
        commit={"oid": "c" * 40},
    )
    latest = dict(
        review(
            13,
            body="The cap is covered. I found a compatibility issue with existing archives.",
        ),
        submittedAt="2026-01-04T00:00:00Z",
        commit={"oid": github.head},
    )
    historic_body = dict(
        review(14, body="Retain unaddressed review text."),
        submittedAt="2026-01-05T00:00:00Z",
        commit={"oid": github.head},
    )
    github.reviews = [first, reply, older, latest, historic_body]
    old_inline = dict(comment(20), pullRequestReview=first)
    latest_inline = dict(
        comment(21, body="Preserve legacy equivalence."), pullRequestReview=latest
    )
    open_inline = dict(comment(22), pullRequestReview=first)
    github.pages = [
        [thread("outdated", [old_inline])],
        [
            dict(thread("latest", [latest_inline]), isOutdated=False),
            dict(thread("still-open", [open_inline]), isOutdated=False),
        ],
    ]
    batch = collect(github, adopted_at="2026-01-06T00:00:00Z")
    assert [item.thread for item in batch.items] == [
        "review:14",
        "latest",
        "still-open",
    ]
    assert {item.thread for item in batch.handled} == {
        "review:10",
        "review:12",
        "review:13",
        "outdated",
    }


def test_only_current_maintainer_review_body_can_authorize_a_test_update():
    from dataclasses import asdict

    from github_harness_agent.test_updates import authorized_test_update

    github = ReviewGitHub()
    directive = "Allow test update: tests/test_examples.py::test_example"
    github.reviews = [
        dict(review(11, body=f"    {directive}"), commit={"oid": github.head}),
        dict(review(12, body=directive), commit={"oid": "b" * 40}),
        dict(review(13, body=directive, login="visitor"), commit={"oid": github.head}),
    ]
    github.pages = [[thread("inline", [comment(20, body=directive)])], []]
    batch = collect(github, adopted_at="2025-12-31T00:00:00Z")
    authorized = [
        asdict(item) for item in batch.items if item.test_update_head == batch.head
    ]
    assert [item["thread"] for item in authorized] == ["review:11"]
    assert authorized_test_update(authorized) is None
    github.reviews[0]["body"] = directive
    batch = collect(github, adopted_at="2025-12-31T00:00:00Z")
    assert (
        authorized_test_update(
            asdict(item) for item in batch.items if item.test_update_head == batch.head
        )
        == "tests/test_examples.py::test_example"
    )


def test_adopted_pull_uses_exact_agent_reply_with_review_revision_time():
    github = ReviewGitHub()
    github.reviews = [
        review(10, body="Include the example."),
        review(11, body="Document the result.", edited="2026-01-03T00:00:00Z"),
    ]
    github.pages = [[], []]
    github.comments = [
        {
            "body": f"Following up on review #{number}.\n\nAddressed this feedback in `abcd`.",
            "user": {"login": "test-app[bot]"},
            "created_at": "2026-01-02T00:00:00Z",
        }
        for number in (10, 11)
    ]
    batch = collect(github, adopted_at="2026-01-04T00:00:00Z")
    assert [item.thread for item in batch.handled] == ["review:10"]
    assert [item.thread for item in batch.items] == ["review:11"]


def test_known_thread_followup_waits_for_debounce():
    github = ReviewGitHub()
    github.pages[0][0]["comments"]["nodes"].append(
        comment(9, parent=3, created="2026-01-02T00:00:00Z")
    )
    recent = collect(
        github, known_threads=["A"], now=datetime(2026, 1, 2, tzinfo=timezone.utc)
    )
    assert len(recent.items) == 3
    assert recent.retry_at == datetime(2026, 1, 2, 0, 0, 20, tzinfo=timezone.utc)
    settled = collect(github, known_threads=["A"], now=recent.retry_at)
    assert [item.reply_to for item in settled.items] == [None, 3, 3, 4]
    assert settled.items[2].revision_at == datetime(2026, 1, 2, tzinfo=timezone.utc).timestamp()
    assert settled.items[2].revision_at > settled.items[3].revision_at


def test_recovers_reply_after_crash_without_duplicate_comment():
    from github_harness_agent.publish import Journal
    from github_harness_agent.reviews import reply_to_review

    github = ReviewGitHub()
    item = collect(github).items[1]
    data = {}
    first = reply_to_review(
        github,
        Journal(data, lambda: None),
        lambda: None,
        repo="test/repo",
        pull=6,
        item=item,
        commit=github.head,
        checks=("pytest",),
        bot_login="test-app[bot]",
    )
    second = reply_to_review(
        github,
        Journal(data, lambda: None),
        lambda: None,
        repo="test/repo",
        pull=6,
        item=item,
        commit=github.head,
        checks=("pytest",),
        bot_login="test-app[bot]",
    )
    assert first == second == 100
    assert len(github.comments) == 1


def test_closed_pull_stops_review_processing():
    from github_harness_agent.publish import ClosedPull

    github = ReviewGitHub()
    github.state = "CLOSED"
    with pytest.raises(ClosedPull):
        collect(github)


@pytest.mark.parametrize("location", ["review", "inline", "conversation"])
def test_clear_acknowledgments_are_ignored_without_hiding_requests(location):
    github = ReviewGitHub()
    github.reviews = []
    github.pages = [[], []]
    if location == "review":
        github.reviews = [
            review(1, body="Thanks!"),
            review(2, body="Thanks! Please fix the example."),
        ]
    elif location == "inline":
        github.pages[0] = [
            thread("A", [comment(1, body="Thanks!")]),
            thread("B", [comment(2, body="Thanks! Please fix the example.")]),
        ]
    else:
        github.comments = [
            {
                "id": index,
                "body": body,
                "user": {"login": "maintainer"},
                "created_at": "2026-01-02T00:00:00Z",
            }
            for index, body in enumerate(
                ["Thanks!", "Thanks! Please fix the example."], 1
            )
        ]
    batch = collect(github)
    assert [item.body for item in batch.items] == ["Thanks! Please fix the example."]
    assert [item.body for item in batch.handled] == ["Thanks!"]


def test_two_items_in_one_thread_get_distinct_retry_safe_replies():
    from dataclasses import replace
    from github_harness_agent.publish import Journal
    from github_harness_agent.reviews import reply_to_review

    github = ReviewGitHub()
    first = collect(github).items[1]
    second = replace(first, key="second-revision", body="Also fix the example.")
    journal = Journal({}, lambda: None)
    for item in (first, second, first, second):
        reply_to_review(
            github,
            journal,
            lambda: None,
            repo="test/repo",
            pull=6,
            item=item,
            commit=github.head,
            checks=("pytest tests/test_example.py (exit 0)",),
            bot_login="test-app[bot]",
        )
    assert len(github.comments) == 2
    assert first.body in github.comments[0]["body"]
    assert second.body in github.comments[1]["body"]
    assert (
        f"https://github.com/test/repo/commit/{github.head}"
        in github.comments[0]["body"]
    )


@pytest.mark.parametrize(
    "saved_journal", [False, "pending", "pending-unposted", "complete"]
)
def test_previous_release_reply_is_recovered_without_duplicate(saved_journal):
    from dataclasses import replace
    from github_harness_agent.publish import Journal
    from github_harness_agent.reviews import _digest, reply_to_review

    github = ReviewGitHub()
    item = collect(github).items[1]
    legacy = _digest("test/repo", 6, item.thread, github.head)
    body = f"Addressed this feedback in `{github.head}`.\n\n<!-- github-harness-agent:{legacy} -->"
    github.comments = [
        {
            "id": 100,
            "body": body,
            "user": {"login": "test-app[bot]"},
            "in_reply_to_id": item.reply_to,
        }
    ]
    if saved_journal == "pending-unposted":
        github.comments = []
    data = {}
    journal = Journal(data, lambda: None)
    if saved_journal:
        journal.begin(
            f"reply:{legacy}",
            {
                "path": f"/repos/test/repo/pulls/6/comments/{item.reply_to}/replies",
                "body": body,
            },
        )
        if saved_journal == "complete":
            journal.finish(f"reply:{legacy}", {"id": 100})
    for _ in range(2):
        assert (
            reply_to_review(
                github,
                journal,
                lambda: None,
                repo="test/repo",
                pull=6,
                item=item,
                commit=github.head,
                checks=("pytest",),
                bot_login="test-app[bot]",
            )
            == 100
        )
    assert len(github.comments) == 1
    # A recovered legacy reply belongs to one revision, never every item in
    # the same thread at the same commit.
    next_item = replace(item, key="another-revision", body="Fix another case.")
    assert (
        reply_to_review(
            github,
            journal,
            lambda: None,
            repo="test/repo",
            pull=6,
            item=next_item,
            commit=github.head,
            checks=("pytest",),
            bot_login="test-app[bot]",
        )
        == 101
    )
    assert len(github.comments) == 2


@pytest.mark.parametrize("stage", ["reply", "question", "repair"])
def test_review_decisions_survive_issue_to_pull_adoption(tmp_path, stage):
    from github_harness_agent.store import Store

    store = Store(tmp_path / "store.db")
    original = {
        "stage": stage,
        "pull": 34,
        "branch": "agent/issue-12",
        "review_head": "head",
        "feedback": [{"key": "active"}],
        "question": "Which example?",
        "review_outcomes": {
            "done": {"status": "addressed", "commit": "head"},
            "active": {"status": "clarification", "question": "Which example?"},
        },
        "active_review_key": "active",
        "deferred_review_keys": ["later"],
        "github_operations": {"reply:old": {"result": {"id": 100}}},
    }
    store.enqueue("org/repo", 12, original)
    assert store.adopt_pull(
        "org/repo",
        34,
        {
            "stage": "watch",
            "branch": original["branch"],
            "author": "bot",
            "linked_issues": [12],
        },
        "agent/",
        head_sha="head",
    )
    adopted = store.claim(pull=("org/repo", 34))["data"]
    for key in (
        "stage",
        "feedback",
        "review_head",
        "review_outcomes",
        "active_review_key",
        "deferred_review_keys",
        "github_operations",
    ):
        assert adopted[key] == original[key]
    if stage == "question":
        assert adopted["question"] == original["question"]


def test_review_diff_evidence_has_a_byte_budget_and_hunk_locations():
    from github_harness_agent.controller import review_change_evidence

    evidence = review_change_evidence(
        b"diff --git a/example.py b/example.py\n@@ -1 +1 @@\n-secret\n+fix\n"
        + ("@@ " + "é" * 10000).encode()
    )
    assert evidence[:2] == ["diff --git a/example.py b/example.py", "@@ -1 +1 @@"]
    assert sum(len(line.encode()) + 1 for line in evidence) <= 4000
    assert all("secret" not in line and "+fix" not in line for line in evidence)


@pytest.mark.parametrize("recovery", ["journal", "pending", "remote"])
def test_legacy_controller_restart_recovers_posted_reply_without_clarification(
    tmp_path, recovery
):
    import dataclasses
    from github_harness_agent.controller import Controller
    from github_harness_agent.reviews import Feedback, _digest
    from github_harness_agent.store import Store
    from test_intake import pull

    request = pull(20, "feature")
    item = Feedback("old-revision", "old-thread", 44, "Fix the example", "maintainer")
    head = request["head"]["sha"]
    legacy = _digest("owner/repo", 20, item.thread, head)
    body = f"Addressed this feedback in `{head}`.\n\n<!-- github-harness-agent:{legacy} -->"
    posted = {
        "id": 100,
        "body": body,
        "user": {"login": "agent-user"},
        "in_reply_to_id": 44,
    }
    data = {
        "stage": "reply",
        "pull": 20,
        "branch": "feature",
        "author": "agent-user",
        "linked_issues": [],
        "title": "PR",
        "body": "",
        "base": "release/2",
        "seen": [],
        "threads": [],
        "round": 1,
        "feedback": [dataclasses.asdict(item)],
        "review_head": "previous-head",
        "last_commit": head,
        "last_checks": ["pytest (exit 0)"],
    }
    if recovery in {"journal", "pending"}:
        entry = {
            "intent": {
                "path": "/repos/owner/repo/pulls/20/comments/44/replies",
                "body": body,
            }
        }
        if recovery == "journal":
            entry["result"] = {"id": 100}
        data["github_operations"] = {f"reply:{legacy}": entry}

    class API:
        def rest(self, method, path, **kwargs):
            assert method == "GET", "restart must not post a duplicate clarification"
            return request

        def paginate(self, path, **kwargs):
            assert path == "/repos/owner/repo/pulls/20/comments"
            return [] if recovery == "journal" else [posted]

    database = tmp_path / "job.db"
    Store(database).adopt_pull("owner/repo", 20, data, "agent/")
    restarted = Store(database)
    Controller(
        restarted, API(), None, {"checks": [["test"]], "mode": "canary"}, "agent-user"
    ).run(restarted.claim())
    with restarted.tx() as connection_:
        import json

        saved = json.loads(connection_.execute("SELECT data FROM tasks").fetchone()[0])
    assert saved["seen"] == [item.key]
    assert saved["stage"] == "watch"
    assert saved["review_outcomes"][item.key] == {
        "status": "recovered",
        "commit": head,
        "replied": True,
    }


def test_thread_resolved_before_reply_suppresses_comment():
    from github_harness_agent.publish import Journal
    from github_harness_agent.reviews import reply_to_review

    github = ReviewGitHub()
    item = collect(github).items[1]
    github.pages[0][0]["isResolved"] = True
    reply_to_review(
        github,
        Journal({}, lambda: None),
        lambda: None,
        repo="test/repo",
        pull=6,
        item=item,
        commit=github.head,
        checks=("pytest",),
        bot_login="test-app[bot]",
    )
    assert github.comments == []


@pytest.mark.parametrize(
    ("source", "line"),
    [
        ("new heading\nbefore\nfix me\nafter", 3),
        ("fix me\nunrelated\nbefore\nfix me\nafter", 4),
        ("before\nfix me\nafter\nbefore\nfix me\nafter", None),
        ("before\nchanged entirely\nafter", None),
        (None, None),
    ],
)
def test_maps_outdated_comments_by_diff_context(source, line):
    from github_harness_agent.reviews import ReviewItem, map_feedback

    original = ReviewItem(
        "key",
        "thread",
        3,
        "Explain this",
        "maintainer",
        "sample.txt",
        None,
        True,
        20,
        "@@ -19,3 +19,3 @@\n before\n-old\n+fix me\n after",
        "RIGHT",
    )
    mapped = map_feedback(original, source)
    assert (mapped.line if mapped else None) == line

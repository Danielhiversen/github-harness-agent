import json
import sqlite3

import pytest

from github_harness_agent.intake import adopt_open_pulls, reconcile
from github_harness_agent.store import Store


class FakeRepo:
    role = "write"
    blocked = True

    def paginate(self, path, params=None):
        if path.endswith("/issues"):
            return [
                {
                    "number": 4,
                    "state": "open",
                    "title": "fixture",
                    "labels": [{"name": "agent:ready"}],
                    "assignees": [{"login": "agent-user"}],
                }
            ]
        if path.endswith("/events"):
            return [
                {
                    "event": "labeled",
                    "label": {"name": "agent:ready"},
                    "actor": {"login": "maintainer"},
                }
            ]
        return [{"state": "open" if self.blocked else "closed"}]

    def rest(self, method, path):
        return (
            {"permission": self.role}
            if path.endswith("permission")
            else {"object": {"sha": "abc"}}
        )


def test_reconciliation_recovers_missed_delivery_and_waits_for_dependencies(tmp_path):
    db, api = Store(tmp_path / "db"), FakeRepo()

    def sync():
        reconcile(api, db, "owner/repo", {"base": "main", "bot_login": "agent-user"})

    sync()
    assert db.claim() is None
    api.blocked = False
    sync()
    sync()
    assert db.claim()["data"]["base_sha"] == "abc"
    assert db.claim() is None


def pull(number, branch, *, body="", author="agent-user", head_repo="owner/repo"):
    return {
        "number": number,
        "state": "open",
        "title": f"PR {number}",
        "body": body,
        "head": {"ref": branch, "sha": "a" * 40, "repo": {"full_name": head_repo}},
        "base": {
            "ref": "release/2",
            "sha": "b" * 40,
            "repo": {"full_name": "owner/repo"},
        },
        "user": {"login": author},
    }


class PullListing:
    def __init__(self, requests):
        self.requests = requests

    def paginate(self, path, params=None):
        assert path == "/repos/owner/repo/pulls"
        assert params == {"state": "open", "sort": "updated", "direction": "desc"}
        return iter(self.requests)


def test_adopts_verified_open_pull_without_conflating_pull_and_issue_numbers(tmp_path):
    store = Store(tmp_path / "db", lambda: 100)
    store.enqueue("owner/repo", 5, {"stage": "implement"})
    store.enqueue("owner/repo", 25, {"stage": "implement"})
    api = PullListing(
        [
            pull(
                20, "past/branch", body="Closes #5\n\n`Fixes #7`\n\nFixes other/repo#9"
            ),
            pull(25, "work/unlinked"),
            pull(26, "elsewhere", head_repo="contributor/repo"),
            pull(27, "another-author", author="someone-else"),
            dict(pull(28, "closed"), state="closed"),
        ]
    )
    assert adopt_open_pulls(api, store, "owner/repo", {}, "Agent-User") == 2
    assert adopt_open_pulls(api, store, "owner/repo", {}, "Agent-User") == 0
    store.enqueue("owner/repo", 5, {"stage": "implement", "branch": "agent/issue-5"})
    with store.tx() as db:
        jobs = db.execute(
            "SELECT kind,issue,state,data FROM tasks ORDER BY kind,issue"
        ).fetchall()
    assert [(job["kind"], job["issue"], job["state"]) for job in jobs] == [
        ("issue", 5, "superseded"),
        ("issue", 25, "queued"),
        ("pull", 20, "queued"),
        ("pull", 25, "queued"),
    ]
    linked = json.loads(jobs[2]["data"])
    assert linked["branch"] == "past/branch"
    assert linked["base"] == "release/2"
    assert linked["linked_issues"] == [5]
    assert json.loads(jobs[3]["data"])["linked_issues"] == []


def test_legacy_issue_database_keeps_job_and_lease_during_adoption_migration(tmp_path):
    path = tmp_path / "db"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE tasks(repo TEXT,issue INTEGER,state TEXT,generation INTEGER,
            token TEXT,expires REAL,due REAL,data TEXT,attempts INTEGER,PRIMARY KEY(repo,issue))""")
        db.execute(
            "INSERT INTO tasks VALUES('owner/repo',25,'running',4,'existing',500,0,?,3)",
            ('{"stage":"implement"}',),
        )
    store = Store(path, lambda: 100)
    store.check({"token": "existing"})
    listing = PullListing([pull(25, "unlinked/feature")])
    assert adopt_open_pulls(listing, store, "owner/repo", {}, "agent-user") == 1
    adopted = store.claim()
    assert adopted["kind"] == "pull" and adopted["issue"] == 25
    store.save(adopted, "queued")
    store.check({"token": "existing"})


def test_adoption_closes_and_reopens_existing_job_without_resetting_review_history(
    tmp_path,
):
    store = Store(tmp_path / "db", lambda: 100)
    api = PullListing([pull(25, "unlinked/feature")])
    adopt_open_pulls(api, store, "owner/repo", {}, "agent-user")
    job = store.claim()
    job["data"]["seen"] = ["review:123"]
    store.save(job, "paused")
    api.requests = []
    assert adopt_open_pulls(api, store, "owner/repo", {}, "agent-user") == 0
    assert store.claim() is None
    api.requests = [pull(25, "unlinked/feature")]
    assert adopt_open_pulls(api, store, "owner/repo", {}, "agent-user") == 0
    assert store.claim()["data"]["seen"] == ["review:123"]


@pytest.mark.parametrize("same_head", [True, False])
def test_adoption_preserves_cooldown_and_only_resumes_matching_review_attempt(
    tmp_path, same_head
):
    store = Store(tmp_path / "db", lambda: 100)
    repo = "owner/repo"
    github = PullListing([pull(20, "existing/agent")])
    current_head = github.requests[0]["head"]["sha"]
    stored_head = current_head if same_head else "f" * 40
    feedback = [{"key": "active-review"}]
    excluded = [{"candidate": {"route": "groq-free", "configured_model": "fixture"}}]
    store.enqueue(
        repo,
        5,
        {
            "stage": "repair",
            "pull": 20,
            "review_head": stored_head,
            "feedback": feedback,
            "provider_retry_count": 2,
            "provider_failures": {"groq-free": "upstream timeout"},
            "last_error": "provider routes exhausted",
            "retry_at": 600,
            "model_attempt_failures": excluded,
            "prepared": {"sha": "b" * 40},
        },
    )
    store.save(store.claim(), "queued", delay=500)
    assert adopt_open_pulls(github, store, repo, {}, "agent-user") == 1
    assert store.claim() is None
    with store.tx() as db:
        record = db.execute(
            "SELECT * FROM tasks WHERE kind='pull' AND issue=20"
        ).fetchone()
    assert record["due"] == 600
    data = json.loads(record["data"])
    assert data["provider_retry_count"] == 2
    assert "prepared" not in data
    if same_head:
        assert data["stage"] == "repair"
        assert data["review_head"] == stored_head
        assert data["feedback"] == feedback
        assert data["model_attempt_failures"] == excluded
    else:
        assert data["stage"] == "watch"
        assert not {"feedback", "review_head", "model_attempt_failures"} & data.keys()


def test_assignment_is_required_and_label_is_only_an_optional_gate(tmp_path):
    from github_harness_agent.intake import eligible

    api = FakeRepo()
    api.blocked = False
    issue = {
        "number": 4,
        "state": "open",
        "labels": [],
        "assignees": [{"login": "Agent-User"}],
    }
    assert eligible(api, "owner/repo", issue, "agent-user")
    assert not eligible(api, "owner/repo", issue, "agent-user", "agent:ready")
    issue.update(
        labels=[{"name": "agent:ready"}], assignees=[{"login": "someone-else"}]
    )
    assert not eligible(api, "owner/repo", issue, "agent-user")


@pytest.mark.parametrize("leased", [False, True])
def test_issue_edit_fences_old_work_and_retains_unverified_draft(tmp_path, leased):
    from github_harness_agent.store import LeaseLost

    store = Store(tmp_path / "db")
    store.enqueue(
        "owner/repo",
        4,
        {
            "stage": "implement",
            "title": "old",
            "body": "old requirements",
            "prepared": {
                "patch": "cGF0Y2g=",
                "sha": "old-sha",
                "sync": {
                    "origin": None,
                    "parent": "base",
                    "target": {"repo": "owner/repo", "ref": "main", "sha": "base"},
                    "identity": ["Bot", "bot@example.test"],
                    "previous_base": None,
                },
            },
        },
    )
    old = store.claim() if leased else None
    assert store.refresh_issue(
        "owner/repo", 4, {"title": "new", "body": "new requirements"}
    )
    if old:
        with pytest.raises(LeaseLost):
            store.save(old)
    current = store.claim()
    assert current["data"]["title"] == "new"
    assert current["data"]["body"] == "new requirements"
    assert "prepared" not in current["data"]
    assert current["data"]["revision_prepared_draft"]["patch"] == "cGF0Y2g="
    assert current["data"]["issue_revision_previous"]["body"] == "old requirements"


def test_reconciliation_cancels_a_lease_when_assignment_disappears(tmp_path):
    from github_harness_agent.store import LeaseLost

    store, api = Store(tmp_path / "db"), FakeRepo()
    api.blocked = False
    config = {"base": "main", "bot_login": "agent-user"}
    reconcile(api, store, "owner/repo", config)
    leased = store.claim()
    original = api.paginate
    api.paginate = (
        lambda path, params=None: []
        if path.endswith("/issues")
        else original(path, params)
    )
    api.rest = lambda method, path: {
        "number": 4,
        "state": "open",
        "assignees": [],
        "title": "fixture",
    }
    reconcile(api, store, "owner/repo", config)
    with pytest.raises(LeaseLost):
        store.check(leased)


def test_dedicated_assignee_can_differ_from_publishing_login(tmp_path):
    store, api = Store(tmp_path / "db"), FakeRepo()
    api.blocked = False
    reconcile(
        api,
        store,
        "owner/repo",
        {"base": "main", "bot_login": "publishing-app[bot]", "assignee": "agent-user"},
    )
    assert store.claim()["issue"] == 4


def test_optional_label_keeps_maintainer_provenance_gate():
    from github_harness_agent.intake import eligible

    api = FakeRepo()
    api.blocked = False
    issue = {
        "number": 4,
        "state": "open",
        "labels": [{"name": "agent:ready"}],
        "assignees": [{"login": "agent-user"}],
    }
    api.role = "read"
    assert not eligible(api, "owner/repo", issue, "agent-user", "agent:ready")
    api.role = "maintain"
    assert eligible(api, "owner/repo", issue, "agent-user", "agent:ready")


def test_issue_edits_do_not_unpause_existing_pull_review(tmp_path):
    store = Store(tmp_path / "db")
    store.enqueue(
        "owner/repo", 4, {"stage": "repair", "pull": 21, "title": "old", "body": ""}
    )
    store.save(store.claim(), "paused")
    assert not store.refresh_issue("owner/repo", 4, {"title": "new", "body": "new"})
    assert store.claim() is None


def test_stale_scan_cannot_cancel_refreshed_or_newly_queued_work(tmp_path):
    store = Store(tmp_path / "db")
    store.enqueue("owner/repo", 4, {"stage": "implement", "title": "old", "body": ""})
    scan = store.issue_scan_snapshot("owner/repo")
    store.refresh_issue("owner/repo", 4, {"title": "new", "body": "new"})
    store.enqueue("owner/repo", 5, {"stage": "implement"})
    store.cancel_missing_issues("owner/repo", set(), scan)
    assert {store.claim()["issue"], store.claim()["issue"]} == {4, 5}


def test_reassigned_issue_survives_a_lagging_filtered_list(tmp_path):
    store, api = Store(tmp_path / "db"), FakeRepo()
    api.blocked = False
    config = {"base": "main", "bot_login": "agent-user"}
    reconcile(api, store, "owner/repo", config)
    leased = store.claim()
    original = api.paginate
    current = original("/repos/owner/repo/issues")[0]
    api.paginate = (
        lambda path, params=None: []
        if path.endswith("/issues")
        else original(path, params)
    )
    api.rest = lambda method, path: current
    reconcile(api, store, "owner/repo", config)
    store.check(leased)


def test_adoption_preserves_assignment_authority_only_for_generated_issue_pull(
    tmp_path,
):
    store = Store(tmp_path / "db")
    store.enqueue(
        "owner/repo",
        5,
        {
            "stage": "watch",
            "pull": 20,
            "title": "old",
            "body": "",
            "branch": "agent/issue-5",
        },
    )
    adopt_open_pulls(
        PullListing([pull(20, "agent/issue-5"), pull(21, "standalone")]),
        store,
        "owner/repo",
        {},
        "agent-user",
    )
    with store.tx() as db:
        rows = {
            row["issue"]: json.loads(row["data"])
            for row in db.execute("SELECT issue,data FROM tasks WHERE kind='pull'")
        }
    assert rows[20]["assignment_issues"] == [5]
    assert "assignment_issues" not in rows[21]

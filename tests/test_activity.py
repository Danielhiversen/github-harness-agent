import pytest

from github_harness_agent.activity import control, status
from github_harness_agent.store import LeaseLost, Store


def test_lease_fenced_model_activity_and_scoped_retry(tmp_path):
    now = [100.0]
    store = Store(tmp_path / "jobs.db", lambda: now[0])
    repo = "org/weather"
    store.enqueue(
        repo,
        4,
        {
            "pull": 20,
            "stage": "repair",
            "round": 3,
            "draft_checkpoints": {"drafts": {"base": {}}},
        },
    )
    store.enqueue(repo, 20, {"pull": 24, "stage": "watch"})
    job = store.claim(ttl=5)
    store.observe_model(job, "a", "attempt", "coding", "first", "m1")
    store.observe_model(job, "b", "attempt", "probe", "second", "m2")
    (row,) = status(store, repo, 20)
    assert [c["provider"] for c in row["active_calls"]] == ["first", "second"]
    with pytest.raises(ValueError, match="running"):
        control(store, repo, 20, "retry")
    store.observe_model(job, "a", "success", "coding", "first", "m1")
    now[0] += 6
    (row,) = status(store, repo, 20)
    assert row["last_successful_call"]["model"] == "m1" and not row["active_calls"]
    with pytest.raises(LeaseLost):
        store.observe_model(job, "b", "success", "probe", "second", "m2")
    job = store.claim()
    store.save(job, "queued", delay=500)
    control(store, repo, 20, "pause")
    (row,) = status(store, repo, 20)
    assert row["blocker"] == "operator paused this PR"
    control(store, repo, 20, "resume")
    control(store, repo, 20, "retry")
    (row,) = status(store, repo, 20)
    assert row["issue"] == 4 and row["drafts"] == ["base"] and row["next_retry_at"]
    with store.tx() as db:
        assert (
            db.execute(
                "select json_extract(data,'$.allowed_rounds') from tasks where issue=4"
            ).fetchone()[0]
            is None
        )
        assert (
            db.execute("select generation from tasks where issue=20").fetchone()[0] == 1
        )


def test_adopted_owner_wins_after_closed_issue_reopen(tmp_path):
    store = Store(tmp_path / "jobs.db")
    repo = "org/weather"
    store.enqueue(repo, 2, {"pull": 20, "stage": "watch", "branch": "old"})
    job = store.claim()
    store.save(job, "complete")
    store.adopt_pull(
        repo,
        20,
        {
            "pull": 20,
            "branch": "new",
            "linked_issues": [],
            "author": "agent",
            "stage": "watch",
            "round": 3,
        },
        "agent/",
    )
    control(store, repo, 20, "pause")
    (row,) = status(store, repo, 20)
    assert row["state"] == "paused" and row["kind"] == "pull"
    with store.tx() as db:
        db.execute(
            "update tasks set data=json_set(data,'$.pause_reason','review repair round limit reached') where kind='pull'"
        )
    control(store, repo, 20, "resume")
    with store.tx() as db:
        assert (
            db.execute(
                "select json_extract(data,'$.allowed_rounds') from tasks where kind='pull'"
            ).fetchone()[0]
            == 4
        )
        assert (
            db.execute("select state from tasks where kind='issue'").fetchone()[0]
            == "complete"
        )

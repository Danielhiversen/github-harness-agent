import dataclasses
import json

import pytest

from github_harness_agent import controller
from github_harness_agent.controller import Controller, draft_scope, git
from github_harness_agent.rebase import TargetBase
from github_harness_agent.reviews import Feedback, ReviewBatch
from github_harness_agent.store import Store

from test_intake import pull


@pytest.mark.parametrize("scenario", ["unchanged", "new-feedback", "moved-head"])
def test_resumed_unprepared_repair_refreshes_feedback_and_fences_changed_draft_scope(
    tmp_path, monkeypatch, scenario
):
    changed = scenario == "new-feedback"
    moved = scenario == "moved-head"
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-q")
    (source / "value").write_text("baseline\n")
    git(source, "add", "value")
    git(
        source,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.com",
        "commit",
        "-qm",
        "base",
    )
    head = git(source, "rev-parse", "HEAD").decode().strip()
    request = pull(19, "feature/existing")
    live_head = "b" * 40 if moved else head
    request["head"]["sha"] = live_head
    request["base"]["sha"] = head
    repo = "owner/repo"
    retained = Feedback("old", "thread-1", 1, "Still outstanding.", "maintainer")
    new = Feedback(
        "new", "thread-2", 2, "Arrived after the previous batch.", "maintainer"
    )
    closed = Feedback("closed", "thread-3", 3, "Already resolved.", "maintainer")
    prior = (retained, closed) if changed else (retained,)
    active = (retained, new) if changed else prior
    config = {"mode": "shadow", "checks": [["test"]]}
    store = Store(tmp_path / "job.db")
    data = {
        "pull": 19,
        "branch": "feature/existing",
        "author": "agent-user",
        "linked_issues": [],
        "stage": "repair",
        "title": "Existing PR",
        "body": "",
        "base": "release/2",
        "base_sha": head,
        "adopted_at": "2026-01-01T00:00:00Z",
        "review_head": head,
        "seen": [],
        "threads": [],
        "feedback": [dataclasses.asdict(item) for item in prior],
        "round": 0,
    }
    target = TargetBase(repo, "release/2", head)
    scope, keys = draft_scope(
        {"repo": repo, "kind": "pull", "issue": 19, "data": data},
        "repair",
        head,
        target,
        config,
    )
    data["draft_checkpoints"] = {
        "scope": scope,
        "head": head,
        "parent": head,
        "drafts": {},
        "reviews": keys,
    }
    store.adopt_pull(repo, 19, data, "agent/")

    class API:
        def rest(self, method, path):
            assert method == "GET"
            if path.endswith("/pulls/19"):
                return request
            assert "/git/ref/heads/" in path
            return {"object": {"sha": head}}

        def fetch(self, name, checkout, sha):
            git(checkout, "fetch", str(source), sha)

    class AtExecutor(Exception):
        pass

    class Executor:
        def execute(self, task, path, *, cancelled):
            assert not moved
            assert "Still outstanding." in task.instructions
            assert "Already resolved." not in task.instructions
            assert ("Arrived after the previous batch." in task.instructions) == changed
            assert (task.checkpoint_scope != scope) == changed
            raise AtExecutor

    monkeypatch.setattr(
        controller,
        "collect_reviews",
        lambda *args, **kwargs: ReviewBatch(live_head, active),
    )
    worker = Controller(store, API(), Executor(), config, "agent-user")
    if moved:
        worker.run(store.claim())
    else:
        with pytest.raises(AtExecutor):
            worker.run(store.claim())
    with store.tx() as db:
        row = db.execute("SELECT state,data FROM tasks").fetchone()
    saved = json.loads(row["data"])
    if moved:
        assert row["state"] == "paused"
        assert saved["stage"] == "watch"
        assert saved["retired_draft_checkpoints"]["head"] == head
        return
    assert [item["key"] for item in saved["feedback"]] == [item.key for item in active]
    if changed:
        assert "draft_checkpoints" not in saved
        assert saved["retired_draft_checkpoints"]["scope"] == scope
    else:
        assert saved["draft_checkpoints"]["scope"] == scope

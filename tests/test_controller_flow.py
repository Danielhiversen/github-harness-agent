"""Integration between durable jobs, verified Git objects and the PR writer."""

import json
from types import SimpleNamespace

from github_harness_agent import controller
from github_harness_agent.controller import Controller, git
from github_harness_agent.executor import CheckEvidence
from github_harness_agent.intake import reconcile
from github_harness_agent.reviews import Feedback, ReviewBatch
from github_harness_agent.store import Store

from test_publish import FakeGitHub


def test_issue_and_review_update_the_same_pull(tmp_path, monkeypatch):
    repo = tmp_path / "project"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "value").write_text("zero\n")
    git(repo, "add", "value")
    git(
        repo,
        "-c",
        "user.name=T",
        "-c",
        "user.email=t@example.test",
        "commit",
        "-qm",
        "baseline",
    )
    base = git(repo, "rev-parse", "HEAD").decode().strip()
    issue = dict(
        number=3, title="change value", state="open", labels=[{"name": "agent:ready"}]
    )

    class API(FakeGitHub):
        def rest(self, method, path, **kw):
            if path.endswith("/permission"):
                return {"permission": "maintain"}
            if path.endswith("/issues/3"):
                return issue
            if path.endswith("/heads/main"):
                return {"object": {"sha": base}}
            if path.endswith("/pulls/21"):
                return self.pulls[0]
            result = super().rest(method, path, **kw)
            if path.endswith("/pulls"):
                result["head"].update(ref="agent/issue-3", repo={"full_name": "a/b"})
            return result

        def graphql(self, query, variables):
            self.pulls[0]["draft"] = False

        def paginate(self, path, params=None):
            if path.endswith("/issues"):
                return [issue]
            if path.endswith("/events"):
                return [
                    dict(
                        event="labeled",
                        label={"name": "agent:ready"},
                        actor={"login": "owner"},
                    )
                ]
            if path.endswith("/blocked_by"):
                return []
            return super().paginate(path, params=params)

        def fetch(self, name, path, sha):
            git(path, "fetch", str(repo), sha)

        def push(self, name, path, branch, expected_head, commit):
            super().push(name, path, branch, expected_head, commit)
            git(repo, "fetch", str(path), commit)
            git(repo, "update-ref", "refs/heads/" + branch, commit)
            if self.pulls:
                self.pulls[0]["head"]["sha"] = commit

    class Executor:
        def execute(self, task, path, cancelled):
            assert not cancelled()
            is_review = "Review requests" in task.instructions
            if is_review:
                feedback = json.loads(
                    task.instructions.split("Review requests against this commit:\n")[1]
                )
                assert [item["line"] for item in feedback] == [1]
                assert [item["key"] for item in feedback] == ["review-revision"]
            (path / "value").write_text("two\n" if is_review else "one\n")
            patch = git(path, "diff")
            return SimpleNamespace(
                patch=patch,
                checks=(CheckEvidence(("test",), 0, "ok"),),
                route_trace=(),
                usage={},
                refinement=None,
            )

    clock = [100]
    db = Store(tmp_path / "db", lambda: clock[0])
    api = API(None)
    cfg = {"checks": [["test"]], "base": "main", "mode": "canary"}
    reconcile(api, db, "a/b", cfg)
    run = Controller(db, api, Executor(), cfg, "app[bot]").run
    run(db.claim())
    assert len(api.pulls) == 1 and not api.pulls[0]["draft"]
    first = api.branch
    item = Feedback(
        "review-revision",
        "thread",
        4,
        "Use two",
        "owner",
        "value",
        None,
        True,
        5,
        "@@ -5 +5 @@\n-zero\n+one",
        "RIGHT",
    )
    ambiguous = Feedback(
        "ambiguous-revision",
        "other-thread",
        5,
        "Fix this",
        "owner",
        "other",
        None,
        True,
        5,
        "@@ -5 +5 @@\n-before\n+after",
        "RIGHT",
    )
    monkeypatch.setattr(
        controller,
        "collect_reviews",
        lambda *a, **kw: ReviewBatch(first, (item, ambiguous)),
    )
    replies = []
    monkeypatch.setattr(
        controller, "reply_to_review", lambda *a, **kw: replies.append(kw)
    )
    clock[0] += 46
    run(db.claim())
    assert git(repo, "show", api.branch + ":value") == b"two\n"
    assert git(repo, "rev-parse", api.branch + "^").decode().strip() == first
    assert len(api.pulls) == 1 and replies[1]["commit"] == api.branch
    assert replies[0]["commit"] is None and replies[0]["item"] == ambiguous
    assert replies[0]["question"] and replies[0]["expected_head"] == first
    with db.tx() as conn:
        state = json.loads(conn.execute("SELECT data FROM tasks").fetchone()[0])
    assert (
        state["seen"] == ["ambiguous-revision", "review-revision"]
        and state["stage"] == "watch"
    )

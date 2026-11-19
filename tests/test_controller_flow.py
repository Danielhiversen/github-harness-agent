"""Integration between durable jobs, verified Git objects and the PR writer."""

import json
from types import SimpleNamespace

import pytest

from github_harness_agent import controller
from github_harness_agent.controller import Controller, git
from github_harness_agent.executor import AttemptRejected, CheckEvidence, RouteEvidence
from github_harness_agent.intake import adopt_open_pulls, reconcile
from github_harness_agent.reviews import Feedback, ReviewBatch
from github_harness_agent.store import Store

from test_intake import pull
from test_publish import FakeGitHub


@pytest.mark.parametrize("scenario", ["stable", "moving", "conflict", "fallback"])
def test_issue_and_review_update_the_same_pull(tmp_path, monkeypatch, scenario):
    move_target = scenario in {"moving", "conflict"}
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
    target = {"main": base, "release": base}
    move_at_push = scenario == "moving"

    def commit_file(parent, name, contents, author="T"):
        git(repo, "checkout", "--detach", parent)
        (repo / name).write_text(contents)
        git(repo, "add", name)
        git(
            repo,
            "-c",
            f"user.name={author}",
            "-c",
            "user.email=t@example.test",
            "commit",
            "-qm",
            f"Add {name}",
        )
        return git(repo, "rev-parse", "HEAD").decode().strip()

    issue = dict(
        number=3, title="change value", state="open", labels=[{"name": "agent:ready"}]
    )

    class API(FakeGitHub):
        def rest(self, method, path, **kw):
            if path.endswith("/permission"):
                return {"permission": "maintain"}
            if path.endswith("/issues/3"):
                return issue
            if path.rsplit("/", 1)[-1] in target:
                return {"object": {"sha": target[path.rsplit("/", 1)[-1]]}}
            if path.endswith("/pulls/21"):
                return self.pulls[0]
            result = super().rest(method, path, **kw)
            if method == "POST" and path.endswith("/git/refs"):
                git(repo, "update-ref", kw["json"]["ref"], kw["json"]["sha"])
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

        def push(self, name, path, branch, expected_head, commit, **kwargs):
            nonlocal move_at_push
            if self.pulls and move_at_push:
                move_at_push = False
                target["release"] = commit_file(
                    target["release"], "last-minute", "moved before push\n"
                )
            if guard := kwargs.get("before_write"):
                guard()
            super().push(name, path, branch, expected_head, commit)
            git(repo, "fetch", str(path), commit)
            git(repo, "update-ref", "refs/heads/" + branch, commit, expected_head)
            if self.pulls:
                self.pulls[0]["head"]["sha"] = commit

    class Executor:
        def execute(self, task, path, cancelled):
            assert not cancelled()
            is_review = "Review requests" in task.instructions
            assert task.verify_base_first is is_review
            if scenario == "fallback":
                if is_review:
                    assert task.excluded_models == ()
                elif not task.excluded_models:
                    candidate = RouteEvidence(
                        "coding", "fixture", "actual", {}, "configured"
                    )
                    raise AttemptRejected("empty patch", candidate, (candidate,))
                else:
                    assert task.excluded_models == (("fixture", "configured"),)
                    assert task.provider_offset == 1
            if is_review:
                feedback = json.loads(
                    task.instructions.split("Review requests against this commit:\n")[1]
                )
                assert [item["line"] for item in feedback] == [1]
                assert [item["key"] for item in feedback] == ["review-revision"]
                if move_target:
                    assert (path / "before").read_text() == "before edit\n"
                    assert (path / "human").read_text() == "human work\n"
                    target["release"] = commit_file(
                        target["release"], "during", "during edit\n"
                    )
            elif move_target:
                assert (
                    path / "main-before"
                ).read_text() == "main moved before editing\n"
                target["main"] = commit_file(
                    target["main"], "main-during", "main moved during editing\n"
                )
            (path / "value").write_text("two\n" if is_review else "one\n")
            patch = git(path, "diff")
            return SimpleNamespace(
                patch=patch,
                checks=(CheckEvidence(("test",), 0, "ok"),),
                route_trace=(),
                usage={},
                refinement=None,
            )

        def verify_patch(self, task, path, patch, cancelled):
            assert not cancelled()
            if "Review requests" in task.instructions:
                assert git(path, "show", task.base_sha + ":during") == b"during edit\n"
                assert b"+two" in patch
            else:
                assert (
                    git(path, "show", task.base_sha + ":main-during")
                    == b"main moved during editing\n"
                )
                assert b"+one" in patch
            return SimpleNamespace(
                patch=patch, checks=(CheckEvidence(("retest",), 0, "ok"),)
            )

    clock = [100]
    db = Store(tmp_path / "db", lambda: clock[0])
    api = API(None)
    cfg = {"checks": [["test"]], "base": "main", "mode": "canary"}
    reconcile(api, db, "a/b", cfg)
    if move_target:
        target["main"] = commit_file(base, "main-before", "main moved before editing\n")
    run = Controller(db, api, Executor(), cfg, "app[bot]").run
    run(db.claim())
    if scenario == "fallback":
        assert not api.pulls and api.branch is None
        retry = db.claim()
        retry["data"]["provider_retry_count"] = 1
        recorded = retry["data"]["model_attempt_failures"][0]
        assert recorded["candidate"]["model"] == "actual"
        assert recorded["reason"] == "empty patch"
        Controller(db, api, Executor(), cfg, "app[bot]").run(retry)
        with db.tx() as conn:
            after = json.loads(conn.execute("SELECT data FROM tasks").fetchone()[0])
        assert "model_attempt_failures" not in after
        assert after["last_evidence"]["model_attempt_failures"] == [recorded]
    assert len(api.pulls) == 1 and not api.pulls[0]["draft"]
    first = api.branch
    if move_target:
        git(repo, "merge-base", "--is-ancestor", target["main"], first)
        target["release"] = commit_file(base, "before", "before edit\n")
        first = commit_file(first, "human", "human work\n", author="Human Maintainer")
        git(repo, "update-ref", "refs/heads/agent/issue-3", first)
        api.branch = first
        api.pulls[0]["head"]["sha"] = first
        api.pulls[0]["base"].update(ref="release", repo={"full_name": "a/b"})
        if scenario == "conflict":
            target["release"] = commit_file(
                target["release"], "value", "conflicting base edit\n"
            )
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
    lag = True

    def reply(*args, **kw):
        nonlocal lag
        if kw["commit"] and lag:
            lag = False
            raise controller.StaleHead("PR view still reports pre-push head")
        replies.append(kw)

    monkeypatch.setattr(controller, "reply_to_review", reply)
    clock[0] += 46
    run(db.claim())
    if scenario == "conflict":
        assert not replies
        assert api.branch == first
        with db.tx() as conn:
            stored = conn.execute("SELECT data,state FROM tasks").fetchone()
        assert stored["state"] == "paused"
        assert "could not be rebased" in json.loads(stored["data"])["pause_reason"]
        return
    if scenario == "moving":
        assert api.branch == first
        clock[0] += 6
        retry = db.claim()
        assert retry["data"]["stage"] == "repair"
        assert retry["data"]["prepared"]["sync"]["target"]["sha"] != target["release"]
        run(retry)
    assert len(replies) == 1
    clock[0] += 16
    retry = db.claim()
    assert retry["data"]["stage"] == "reply"
    run(retry)
    assert git(repo, "show", api.branch + ":value") == b"two\n"
    if move_target:
        git(repo, "merge-base", "--is-ancestor", target["release"], api.branch)
        assert git(repo, "show", api.branch + ":human") == b"human work\n"
        assert (
            git(repo, "log", "-1", "--format=%an", api.branch + "^").strip()
            == b"Human Maintainer"
        )
        assert replies[1]["checks"] == ["retest (exit 0)"]
        files = git(repo, "ls-tree", "--name-only", api.branch)
        assert b"last-minute" in files and b"main-during" not in files
    else:
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


@pytest.mark.parametrize("revoke", [False, True])
def test_adopted_job_repairs_only_the_existing_branch_without_issue_eligibility(
    tmp_path, monkeypatch, revoke
):
    repo = tmp_path / "repository"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "value").write_text("existing work\n")
    git(repo, "add", "value")
    git(
        repo,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.com",
        "commit",
        "-qm",
        "base",
    )
    base = git(repo, "rev-parse", "HEAD").decode().strip()
    git(repo, "branch", "past/feature", base)
    request = pull(25, "past/feature", body="Existing work.")
    request["base"]["sha"] = request["head"]["sha"] = base

    class API:
        def paginate(self, path, *, params):
            assert path == "/repos/owner/repo/pulls"
            return [request]

        def rest(self, method, path):
            assert method == "GET"
            if path.endswith("/pulls/25"):
                return request
            if path.endswith("/git/ref/heads/release/2"):
                return {"object": {"sha": base}}
            assert path.endswith("/git/ref/heads/past/feature")
            return {"object": {"sha": request["head"]["sha"]}}

        def fetch(self, name, checkout, sha):
            git(checkout, "fetch", str(repo), sha)

        def push(
            self,
            name,
            checkout,
            branch,
            expected_head,
            commit,
            *,
            before_write,
            **kwargs,
        ):
            assert branch == "past/feature"
            before_write()
            git(repo, "fetch", str(checkout), commit)
            git(repo, "update-ref", "refs/heads/past/feature", commit, expected_head)
            request["head"]["sha"] = commit

    class Executor:
        def execute(self, task, path, *, cancelled):
            assert "implementation is already present" in task.instructions
            assert "Outstanding fix" in task.instructions
            assert (path / "value").read_text() == "existing work\n"
            (path / "value").write_text("reviewed work\n")
            if revoke:
                request["user"]["login"] = "unowned"
            return SimpleNamespace(
                patch=git(path, "diff"),
                checks=(CheckEvidence(("test",), 0, "ok"),),
                route_trace=(),
                usage={},
                refinement=None,
            )

    review = Feedback("review-key", "thread", 42, "Outstanding fix", "reviewer")
    monkeypatch.setattr(
        controller,
        "collect_reviews",
        lambda *args, **kwargs: ReviewBatch(base, (review,)),
    )
    replies = []
    monkeypatch.setattr(
        controller, "reply_to_review", lambda *args, **kw: replies.append(kw)
    )
    store = Store(tmp_path / "db", lambda: 100)
    api = API()
    config = {"base": "configured-but-unused", "checks": [["test"]], "mode": "canary"}
    assert adopt_open_pulls(api, store, "owner/repo", config, "agent-user") == 1
    Controller(store, api, Executor(), config, "agent-user").run(store.claim())
    if revoke:
        assert not replies
        assert request["head"]["sha"] == base
        with store.tx() as db:
            assert db.execute("SELECT state FROM tasks").fetchone()[0] == "complete"
    else:
        assert replies[0]["pull"] == 25
        assert git(repo, "show", "past/feature:value") == b"reviewed work\n"
        assert b"Address review feedback on PR #25" in git(
            repo, "log", "-1", "past/feature"
        )

import json
from types import SimpleNamespace

import pytest

from github_harness_agent import controller
from github_harness_agent.controller import Controller, git
from github_harness_agent.executor import CheckEvidence, NeedsClarification
from github_harness_agent.reviews import Feedback, ReviewBatch
from github_harness_agent.store import Store

from test_ci import status
from test_intake import pull


@pytest.mark.parametrize(
    "scenario",
    ["ci", "mixed", "stale", "retired", "clarification", "round-limit", "resume", "review-after-limit"],
)
def test_current_agent_ci_failure_can_update_existing_pr_once(
    tmp_path, monkeypatch, scenario
):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "value").write_text("existing pushed code\n")
    git(repo, "add", "value")
    git(
        repo,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.com",
        "commit",
        "-qm",
        "existing",
    )
    agent_commit = git(repo, "rev-parse", "HEAD").decode().strip()
    git(repo, "branch", "past/feature", agent_commit)
    request = pull(20, "past/feature")
    request["base"]["sha"] = agent_commit
    request["head"]["sha"] = agent_commit if scenario != "stale" else "b" * 40
    reads = []
    executions = []
    pushes = []
    replies = []
    current_status = "failure"

    class API:
        def rest(self, method, path, *, params=None):
            assert method == "GET"
            if path.endswith("/pulls/20"):
                return request
            if path.endswith("/check-runs"):
                assert path.endswith(f"/{request['head']['sha']}/check-runs")
                reads.append(path)
                return {"total_count": 0, "check_runs": []}
            if path.endswith("/git/ref/heads/past/feature"):
                return {"object": {"sha": request["head"]["sha"]}}
            assert path.endswith("/git/ref/heads/release/2")
            return {"object": {"sha": agent_commit}}

        def paginate(self, path, *, params=None):
            assert path.endswith(f"/{request['head']['sha']}/statuses")
            reads.append(path)
            return [
                status(1, "ci/circleci", current_status, summary="Failed regression")
            ]

        def fetch(self, name, path, sha):
            git(path, "fetch", str(repo), sha)

        def push(
            self, name, path, branch, expected_head, commit, *, before_write, **kwargs
        ):
            assert branch == "past/feature"
            before_write()
            git(repo, "fetch", str(path), commit)
            git(repo, "update-ref", "refs/heads/past/feature", commit, expected_head)
            request["head"]["sha"] = commit
            pushes.append(commit)

    class Executor:
        def execute(self, task, path, *, cancelled):
            nonlocal current_status
            assert (
                "ci/circleci" in task.instructions
                and "Failed regression" in task.instructions
            )
            assert ("Human review" in task.instructions) == (scenario in {"mixed", "review-after-limit"})
            executions.append(task)
            if scenario == "clarification":
                raise NeedsClarification("What is the hidden failure?")
            (path / "value").write_text("CI fix\n")
            if scenario == "retired":
                current_status = "success"
            return SimpleNamespace(
                patch=git(path, "diff"),
                checks=(CheckEvidence(("test",), 0, "ok"),),
                route_trace=(),
                usage={},
                refinement=None,
            )

    human = Feedback("human-key", "review-thread", 44, "Human review", "maintainer")

    def collect_reviews(*args, **kwargs):
        feedback = (
            (human,) if scenario in {"mixed", "review-after-limit"} and human.key not in kwargs["seen"] else ()
        )
        return ReviewBatch(request["head"]["sha"], feedback)

    monkeypatch.setattr(controller, "collect_reviews", collect_reviews)
    monkeypatch.setattr(
        controller, "reply_to_review", lambda *args, **kwargs: replies.append(kwargs)
    )
    now = [100]
    store = Store(tmp_path / "job.db", lambda: now[0])
    data = {
        "stage": "watch",
        "pull": 20,
        "branch": "past/feature",
        "author": "agent-user",
        "linked_issues": [],
        "title": "Existing PR",
        "body": "",
        "base": "release/2",
        "base_sha": agent_commit,
        "last_commit": agent_commit,
        "threads": [],
        "seen": [],
        "round": 3 if scenario in {"round-limit", "resume", "review-after-limit"} else 0,
        "adopted_at": "2026-01-01T00:00:00Z",
    }
    if scenario == "resume":
        data["allowed_rounds"] = 4
    store.adopt_pull("owner/repo", 20, data, "agent/")
    config = {"base": "main", "checks": [["test"]], "mode": "canary"}
    run = Controller(store, API(), Executor(), config, "agent-user").run
    run(store.claim())
    if scenario == "stale":
        assert not reads and not executions and not pushes and not replies
    elif scenario in {"retired", "clarification"}:
        assert len(executions) == 1 and not pushes and not replies
    elif scenario == "round-limit":
        with store.tx() as db:
            job = db.execute("SELECT state,data FROM tasks").fetchone()
        assert job["state"] == "queued"
        assert json.loads(job["data"])["stage"] == "watch"
        assert not executions and not pushes
    else:
        assert len(executions) == len(pushes) == 1
        assert git(repo, "show", "past/feature:value") == b"CI fix\n"
        assert [reply["item"] for reply in replies] == (
            [human] if scenario in {"mixed", "review-after-limit"} else []
        )
        with store.tx() as db:
            persisted = json.loads(db.execute("SELECT data FROM tasks").fetchone()[0])
        assert [item["name"] for item in persisted["last_evidence"]["ci_failures"]] == [
            "ci/circleci"
        ]
        now[0] += 46
        run(store.claim())
        # The same context also reports failure on the agent's repair commit.
        assert len(executions) == len(pushes) == 1

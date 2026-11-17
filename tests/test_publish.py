from copy import deepcopy
from pathlib import Path

import pytest


class ControllerStopped(BaseException):
    pass


class FakeGitHub:
    def __init__(self, failure, *, stopped=False):
        self.failure = failure
        self.stopped = stopped
        self.branch = None
        self.pulls = []
        self.pushes = 0
        self.refs_created = 0

    def fail_once(self, operation):
        from github_harness_agent.github import GitHubError

        if self.failure == operation:
            self.failure = None
            if self.stopped:
                raise ControllerStopped
            raise GitHubError("connection lost")

    def rest(self, method, path, *, json=None):
        from github_harness_agent.github import GitHubError

        if method == "GET" and "/git/ref/" in path:
            if self.branch is None:
                raise GitHubError("missing", status=404)
            return {"object": {"sha": self.branch}}
        if method == "POST" and path.endswith("/git/refs"):
            assert self.branch is None
            self.refs_created += 1
            self.branch = json["sha"]
            self.fail_once("ref")
            return {}
        if method == "POST" and path.endswith("/pulls"):
            assert json["draft"] is True
            assert "Closes #3" in json["body"]
            result = {
                "number": 21,
                "html_url": "https://github.com/a/b/pull/21",
                "node_id": "PR_21",
                "head": {"sha": self.branch},
                "base": {"ref": "main"},
                "draft": True,
                "state": "open",
            }
            self.pulls.append(result)
            self.fail_once("pull")
            return result
        if method == "GET" and path.endswith("/pulls/21"):
            return self.pulls[0]
        raise AssertionError((method, path))

    def paginate(self, path, *, params):
        assert params == {"state": "all", "head": "a:agent/issue-3", "base": "main"}
        return iter(self.pulls)

    def push(self, repo, checkout, branch, expected_head, commit):
        assert self.branch == expected_head
        self.pushes += 1
        self.branch = commit
        self.fail_once("push")

    def graphql(self, query, variables):
        assert "markPullRequestReadyForReview" in query
        assert variables == {"id": "PR_21"}
        self.pulls[0]["draft"] = False
        self.fail_once("ready")
        return {}


@pytest.mark.parametrize("failure", ["ref", "push", "pull"])
def test_publish_reconciles_connection_loss_after_each_write(failure):
    from github_harness_agent.publish import Journal, Publisher, PublishRequest

    github = FakeGitHub(failure)
    data = {}
    request = PublishRequest(
        repo="a/b",
        issue=3,
        base="main",
        base_sha="a" * 40,
        branch="agent/issue-3",
        checkout=Path("/unused"),
        commit="b" * 40,
        title="Fix item",
        summary="Fix item.",
        checks=("uv run pytest",),
    )
    result = Publisher(github, Journal(data, lambda: None), lambda: None).publish(
        request
    )
    same = Publisher(github, Journal(data, lambda: None), lambda: None).publish(request)
    assert same == result
    assert len(github.pulls) == 1
    assert result.number == 21
    assert github.branch == "b" * 40


@pytest.mark.parametrize("failure", ["ref", "push", "pull"])
def test_publish_replays_journal_after_interruption(failure):
    from github_harness_agent.publish import Journal, Publisher, PublishRequest

    github = FakeGitHub(failure, stopped=True)
    request = PublishRequest(
        repo="a/b",
        issue=3,
        base="main",
        base_sha="a" * 40,
        branch="agent/issue-3",
        checkout=Path("/unused"),
        commit="b" * 40,
        title="Fix item",
        summary="Fix item.",
        checks=("uv run pytest",),
    )
    persisted = {}

    def publisher():
        data = deepcopy(persisted)

        def checkpoint():
            persisted.update(deepcopy(data))

        return Publisher(github, Journal(data, checkpoint), lambda: None)

    with pytest.raises(ControllerStopped):
        publisher().publish(request)
    result = publisher().publish(request)
    assert result.number == 21
    assert github.refs_created == github.pushes == len(github.pulls) == 1


def test_publish_preserves_human_commit_after_initial_pr():
    from github_harness_agent.publish import (
        Journal,
        Publisher,
        PublishRequest,
        StaleHead,
    )

    github = FakeGitHub(None)
    data = {}
    request = PublishRequest(
        repo="a/b",
        issue=3,
        base="main",
        base_sha="a" * 40,
        branch="agent/issue-3",
        checkout=Path("/unused"),
        commit="b" * 40,
        title="Fix item",
        summary="Fix item.",
        checks=("uv run pytest",),
    )
    publisher = Publisher(github, Journal(data, lambda: None), lambda: None)
    publisher.publish(request)
    github.branch = "c" * 40
    with pytest.raises(StaleHead):
        publisher.publish(request)
    assert github.branch == "c" * 40


def test_recovery_checkpoints_ready_transition_after_interruption():
    from github_harness_agent.publish import Journal, Publisher, PublishRequest

    github = FakeGitHub("ready", stopped=True)
    data = {}
    request = PublishRequest(
        repo="a/b",
        issue=3,
        base="main",
        base_sha="a" * 40,
        branch="agent/issue-3",
        checkout=Path("/unused"),
        commit="b" * 40,
        title="Fix item",
        summary="Fix item.",
        checks=("uv run pytest",),
        ready=True,
    )
    journal = Journal(data, lambda: None)
    with pytest.raises(ControllerStopped):
        Publisher(github, journal, lambda: None).publish(request)
    restarted = Journal(deepcopy(data), lambda: None)
    result = Publisher(github, restarted, lambda: None).publish(request)
    assert result.draft is False
    assert restarted.get("ready:a/b:21")["result"] == {"number": 21}

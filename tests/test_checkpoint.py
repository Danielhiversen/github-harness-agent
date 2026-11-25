import pytest

from github_harness_agent.checkpoint import DraftCheckpoint, PrivateRecovery, SolverRecovery
from github_harness_agent.controller import (
    _retire_drafts,
    draft_scope,
    restore_drafts,
    restore_solver_recovery,
)
from github_harness_agent.rebase import BOT_IDENTITY, RebaseConflict, TargetBase, git
from github_harness_agent.store import Store


def commit(path, message):
    git(path, "add", "-A")
    git(
        path,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.test",
        "commit",
        "-qm",
        message,
    )
    return git(path, "rev-parse", "HEAD").decode().strip()


def test_solver_recovery_admits_only_matching_scope_head_parent_and_bounded_counts():
    state = SolverRecovery("a" * 64, (("b" * 64, 2, 1),))
    envelope = {
        "scope": "review",
        "head": "head",
        "parent": "parent",
        "solver_recovery": state.to_data(),
    }
    assert restore_solver_recovery(envelope, "review", "head", "parent") == state
    assert restore_solver_recovery(envelope, "new review", "head", "parent") is None
    assert restore_solver_recovery(envelope, "review", "new head", "parent") is None
    assert restore_solver_recovery(envelope, "review", "head", "new parent") is None
    envelope["solver_recovery"]["goals"][0][1] = 3
    assert restore_solver_recovery(envelope, "review", "head", "parent") is None


def test_retiring_recovery_only_envelope_does_not_signal_a_saved_draft():
    recovery = SolverRecovery("a" * 64, (("b" * 64, 0, 2),))
    data = {"draft_checkpoints": {"drafts": {}, "solver_recovery": recovery.to_data()}}
    assert not _retire_drafts(data)
    assert not data


def test_invalid_private_alternate_keeps_independent_recovery_counters():
    primary = SolverRecovery("a" * 64, (("b" * 64, 0, 2),))
    alternate = PrivateRecovery(
        "c" * 64, "d" * 64, "e" * 64, b"patch", ("1" * 64, "2" * 64), ("2" * 64,)
    )
    value = SolverRecovery(primary.identity, primary.goals, alternate).to_data()
    assert SolverRecovery.from_data(value).alternate == alternate
    assert (
        SolverRecovery.from_data(
            SolverRecovery(primary.identity, (), alternate).to_data()
        ).alternate
        == alternate
    )

    for key, invalid in (
        ("cases", ["3" * 64]),
        ("patch_b64", "===?"),
        ("patch_b64", "A" * (350_000)),
        ("static", [["f" * 64, True]]),
    ):
        corrupted = {**value, "alternate": {**value["alternate"], key: invalid}}
        assert SolverRecovery.from_data(corrupted) == primary

    drafts = {"primary": {"patch": "verified primary"}}
    data = {"draft_checkpoints": {"drafts": drafts, "solver_recovery": value}}
    assert _retire_drafts(data)
    assert data["retired_draft_checkpoints"] == {
        "drafts": drafts,
        "solver_recovery": primary.to_data(),
    }


@pytest.mark.parametrize("conflict", [False, True])
def test_recovering_draft_replays_on_new_base_or_reports_conflict(tmp_path, conflict):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-q")
    (source / "draft.txt").write_text("base\n")
    (source / "context.txt").write_text("original\n")
    original = commit(source, "base")
    (source / "draft.txt").write_text("draft\n")
    (source / "added.txt").write_text("new file\n")
    git(source, "add", "-A")
    patch = git(source, "diff", "--cached", "--binary", "--full-index", original)
    git(source, "reset", "--hard", original)
    changed_file = "draft.txt" if conflict else "context.txt"
    (source / changed_file).write_text("human update\n")
    latest = commit(source, "target advanced")

    class LocalRepository:
        def fetch(self, _repository, path, sha):
            git(path, "fetch", "-q", str(source), sha)

    api = LocalRepository()
    checkout = tmp_path / "worker"
    checkout.mkdir()
    git(checkout, "init", "-q")
    api.fetch("org/repo", checkout, latest)
    git(checkout, "checkout", "--detach", latest)
    draft = DraftCheckpoint("plain/coding", original, patch, ("pytest exited 0",))
    envelope = {
        "scope": "current review",
        "head": None,
        "parent": original,
        "identity": BOT_IDENTITY,
        "target": {"repo": "org/repo", "ref": "main", "sha": original},
        "drafts": {draft.key: draft.to_data()},
    }
    args = (api, checkout, "org/repo", None, latest, "current review", envelope)
    if conflict:
        with pytest.raises(RebaseConflict, match="saved draft conflicts"):
            restore_drafts(*args)
    else:
        restored = restore_drafts(*args)[draft.key]
        assert restored.base_sha == latest
        assert restored.progress == ()
        git(checkout, "apply", "--index", "--binary", "-", data=restored.patch)
        assert (checkout / "draft.txt").read_text() == "draft\n"
        assert (checkout / "added.txt").read_text() == "new file\n"
        assert (checkout / "context.txt").read_text() == "human update\n"


def test_unrelated_review_head_or_feedback_revision_cannot_adopt_saved_draft():
    job = {
        "repo": "org/repo",
        "kind": "pull",
        "issue": 4,
        "data": {
            "title": "",
            "body": "",
            "pull": 4,
            "feedback": [{"key": "review-revision-1"}],
        },
    }
    first_target = TargetBase("org/repo", "main", "a")
    scope, _ = draft_scope(job, "repair", "head-1", first_target, {})
    assert (
        draft_scope(job, "repair", "head-1", TargetBase("org/repo", "main", "b"), {})[0]
        == scope
    )
    assert draft_scope(job, "repair", "head-2", first_target, {})[0] != scope
    job["data"]["feedback"][0]["key"] = "review-revision-2"
    updated, _ = draft_scope(job, "repair", "head-1", first_target, {})
    assert updated != scope
    assert (
        restore_drafts(None, None, "org/repo", "head-1", "b", updated, {"scope": scope})
        == {}
    )


def test_matching_repair_draft_can_follow_issue_job_into_pr_adoption(tmp_path):
    store = Store(tmp_path / "controller.db")
    issue = {
        "stage": "repair",
        "pull": 34,
        "review_head": "review-head",
        "branch": "agent/issue-12",
        "title": "Issue title",
        "body": "Issue content",
        "feedback": [],
        "ci_feedback": [{"key": "ci-revision-1"}],
    }
    target = TargetBase("org/repo", "main", "target-a")
    issue_job = {"repo": "org/repo", "kind": "issue", "issue": 12, "data": issue}
    scope, _ = draft_scope(issue_job, "repair", issue["review_head"], target, {})
    draft = DraftCheckpoint("plain/coding", "original", b"verified fixture patch")
    issue["draft_checkpoints"] = {
        "scope": scope,
        "head": issue["review_head"],
        "parent": "original",
        "drafts": {draft.key: draft.to_data()},
    }
    store.enqueue("org/repo", 12, issue)
    assert store.adopt_pull(
        "org/repo",
        34,
        {
            "stage": "watch",
            "pull": 34,
            "branch": issue["branch"],
            "author": "bot",
            "linked_issues": [12],
            "title": "PR title",
            "body": "PR description",
        },
        "agent/",
        head_sha=issue["review_head"],
    )
    adopted = store.claim(pull=("org/repo", 34))
    assert adopted["data"]["stage"] == "repair"
    assert draft_scope(adopted, "repair", issue["review_head"], target, {})[0] == scope
    assert (
        restore_drafts(
            None,
            None,
            "org/repo",
            issue["review_head"],
            "original",
            scope,
            adopted["data"]["draft_checkpoints"],
        )[draft.key]
        == draft
    )
    adopted["data"]["ci_feedback"][0]["key"] = "ci-revision-2"
    assert draft_scope(adopted, "repair", issue["review_head"], target, {})[0] != scope
    with store.tx() as db:
        assert (
            db.execute(
                "SELECT json_extract(data, '$.draft_checkpoints') FROM tasks WHERE kind='issue'"
            ).fetchone()[0]
            is None
        )


def test_private_tradeoff_serialization_and_retirement():
    from github_harness_agent.checkpoint import PrivateCaseTradeoff
    from github_harness_agent.controller import _retired_draft_envelope

    private = PrivateCaseTradeoff("c"*64, "d"*64, "e"*64, b"private fixture bytes")
    state = SolverRecovery("a"*64, (), tradeoff=private)
    assert SolverRecovery.from_data(state.to_data()) == state
    assert "private fixture bytes" not in repr(state)
    assert "solver_recovery" not in _retired_draft_envelope({"solver_recovery": state.to_data()})
    for malformed in ({"patch_b64": "%%%!"}, {"control": private.origin}, {"identity": "wrong"}):
        data = state.to_data()
        data["tradeoff"].update(malformed)
        data["goals"] = [["b"*64, 1, 0]]
        restored = SolverRecovery.from_data(data)
        assert restored.tradeoff is None and restored.goals == (("b"*64, 1, 0),)

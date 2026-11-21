import pytest

from github_harness_agent.rebase import RebaseConflict, TargetBase, branch_parent, git


def test_rebase_begins_after_base_tree_was_adopted_in_a_merge(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-q", "-b", "main")
    git(source, "config", "user.name", "Contributor")
    git(source, "config", "user.email", "contributor@example.com")
    git(source, "commit", "--allow-empty", "-qm", "root")

    def commit(filename, contents):
        (source / filename).write_text(contents)
        git(source, "add", filename)
        git(source, "commit", "-qm", contents)
        return git(source, "rev-parse", "HEAD").decode().strip()

    git(source, "switch", "-qc", "issue")
    commit("client", "pre-integration client")
    git(source, "switch", "-q", "main")
    fork = commit("client", "integrated client")
    target = commit("offline-check", "fixture from latest main")
    git(source, "switch", "-q", "issue")
    git(
        source, "merge", "--no-ff", "-X", "theirs", "-m", "keep integrated client", fork
    )
    tip = commit("review-fix", "preserve cached availability")

    class Transport:
        def fetch(self, repo, path, sha):
            git(path, "fetch", str(source), sha)

    work = tmp_path / "work"
    work.mkdir()
    git(work, "init", "-q")
    result = branch_parent(
        Transport(), work, "fixture", tip, TargetBase("fixture", "main", target)
    )
    assert git(work, "rev-parse", result + "^").decode().strip() == target
    assert git(work, "show", result + ":client") == b"integrated client"
    assert git(work, "show", result + ":review-fix") == b"preserve cached availability"
    assert git(work, "show", result + ":offline-check") == b"fixture from latest main"


def _commit(source, message, files):
    for name, content in files.items():
        (source / name).write_text(content)
    git(source, "add", "-A")
    git(source, "commit", "-qm", message)
    return git(source, "rev-parse", "HEAD").decode().strip()


def _diverged(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-q", "-b", "main")
    git(source, "config", "user.name", "Contributor")
    git(source, "config", "user.email", "contributor@example.com")
    return source


def _integrate(tmp_path, source, tip, target, folder):
    class Transport:
        def fetch(self, repo, path, sha):
            git(path, "fetch", str(source), sha)

    work = tmp_path / folder
    work.mkdir()
    git(work, "init", "-q")
    return work, branch_parent(
        Transport(), work, "fixture", tip, TargetBase("fixture", "main", target)
    )


def test_replay_conflict_retains_tip_merge_resolution_and_target_changes(tmp_path):
    source = _diverged(tmp_path)
    _commit(source, "start", {"logic.py": "state = 0\n"})
    git(source, "switch", "-qc", "task")
    _commit(
        source,
        "early task edit",
        {"logic.py": "state = 1\n", "task.txt": "earlier independent work\n"},
    )
    git(source, "switch", "-q", "main")
    integrated = _commit(source, "base integration", {"logic.py": "state = 2\n"})
    target = _commit(source, "target continues", {"target.txt": "new target\n"})
    git(source, "switch", "-q", "task")
    git(source, "merge", "--no-ff", "-X", "theirs", "-m", "settle edits", integrated)
    tip = _commit(source, "finish task", {"task.txt": "task-specific change\n"})
    work, result = _integrate(tmp_path, source, tip, target, "work")
    _, reproduced = _integrate(tmp_path, source, tip, target, "reproduced")

    assert result == reproduced
    assert git(work, "show", "-s", "--format=%P", result).strip().decode() == target
    assert git(work, "show", result + ":logic.py") == b"state = 2\n"
    assert git(work, "show", result + ":target.txt") == b"new target\n"
    assert git(work, "show", result + ":task.txt") == b"task-specific change\n"


def test_conflict_fallback_preserves_target_security_and_test_discovery(tmp_path):
    source = _diverged(tmp_path)

    def config(paths):
        return f"[tool.pytest.ini_options]\ntestpaths = {paths}\n"

    _commit(
        source,
        "start",
        {".gitignore": "private/\n.local\n", "pyproject.toml": config('["checks"]')},
    )
    git(source, "switch", "-qc", "task")
    tip = _commit(
        source,
        "task adds discovery and ignores",
        {
            ".gitignore": ".local\n*.tmp\n",
            "pyproject.toml": config('["checks", "component_checks"]'),
        },
    )
    git(source, "switch", "-q", "main")
    target_ignores = "# Sensitive\nprivate/\n*.tmp\n.local\ncache/\n"
    target_config = config('["checks", "component_checks", "integration_checks"]')
    target = _commit(
        source,
        "target expands security policy and discovery",
        {".gitignore": target_ignores, "pyproject.toml": target_config},
    )

    work, result = _integrate(tmp_path, source, tip, target, "work")
    assert git(work, "rev-parse", result + "^").strip().decode() == target
    assert git(work, "diff", target, result) == b""
    assert git(work, "show", result + ":.gitignore").decode() == target_ignores
    assert git(work, "show", result + ":pyproject.toml").decode() == target_config


@pytest.mark.parametrize(
    "name, baseline, branch, main",
    [
        ("logic.py", "state = 0\n", "state = 1\n", "state = 2\n"),
        (".gitignore", "private/\n", "!private/\n", "private/\ncache/\n"),
        (
            "pyproject.toml",
            '[tool.pytest.ini_options]\ntestpaths = ["checks"]\n',
            '[tool.pytest.ini_options]\ntestpaths = ["checks", "branch"]\n',
            '[tool.pytest.ini_options]\ntestpaths = ["checks", "main"]\n',
        ),
    ],
)
def test_fallback_pauses_on_ambiguous_net_tip_conflict(
    tmp_path, name, baseline, branch, main
):
    source = _diverged(tmp_path)
    _commit(source, "start", {name: baseline})
    git(source, "switch", "-qc", "task")
    tip = _commit(source, "task edits", {name: branch})
    git(source, "switch", "-q", "main")
    target = _commit(source, "target edits", {name: main})

    with pytest.raises(RebaseConflict, match="unresolved rebase conflicts"):
        _integrate(tmp_path, source, tip, target, "work")
    work = tmp_path / "work"
    assert git(work, "rev-parse", "HEAD").strip().decode() == tip
    assert git(work, "ls-files", "--unmerged") == b""

from github_harness_agent.rebase import TargetBase, branch_parent, git


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

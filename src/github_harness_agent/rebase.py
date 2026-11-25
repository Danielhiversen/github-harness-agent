"""Prepare trusted local history against an immutable target commit."""

import os
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path


BOT_IDENTITY = ("GitHub Harness Agent", "agent@users.noreply.github.com")


class RebaseConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class TargetBase:
    repo: str
    ref: str
    sha: str


def git(path, *args, data=None, commit_date=None):
    env = {
        name: os.environ[name]
        for name in ("PATH", "HOME", "TMPDIR")
        if name in os.environ
    }
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_EDITOR": "true",
            "GIT_SEQUENCE_EDITOR": "true",
        }
    )
    if commit_date is not None:
        env.update(GIT_AUTHOR_DATE=commit_date, GIT_COMMITTER_DATE=commit_date)
    return subprocess.run(
        ["git", "-C", str(path), "-c", f"core.hooksPath={os.devnull}", *args],
        input=data,
        check=True,
        capture_output=True,
        env=env,
    ).stdout


def _ordered_subset(required, available):
    remaining = iter(available)
    return all(item in remaining for item in required)


def _target_preserves_testpaths(base, target, origin):
    def parse(content):
        config = tomllib.loads(content)
        values = config["tool"]["pytest"]["ini_options"].pop("testpaths")
        if (
            not isinstance(values, list)
            or not all(isinstance(value, str) and value for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError("unexpected pytest testpaths")
        return config, values

    def without_testpaths(content):
        stripped, count = re.subn(
            r"(?m)^[ \t]*testpaths[ \t]*=[ \t]*\[[^#\r\n]*\][ \t]*$",
            "",
            content,
        )
        return stripped if count == 1 else None

    before = without_testpaths(base)
    if before is None or before != without_testpaths(origin):
        return False
    try:
        (old_config, old_paths), (_, current_paths), (task_config, task_paths) = map(
            parse, (base, target, origin)
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return (
        old_config == task_config
        and _ordered_subset(old_paths, task_paths)
        and _ordered_subset(task_paths, current_paths)
    )


def _target_preserves_ignores(base, target, origin):
    def rules(content):
        patterns, comments = set(), set()
        for line in content.split("\n"):
            if not line:
                continue
            if line.startswith("#"):
                comments.add(line)
            elif re.fullmatch(r"[A-Za-z0-9_./*?-]+", line):
                patterns.add(line)
            else:
                raise ValueError("ignore rule requires ordering or escaping")
        return patterns, comments

    try:
        base_rules, base_comments = rules(base)
        target_rules, target_comments = rules(target)
        task_rules, task_comments = rules(origin)
    except ValueError:
        return False
    # Keep the target's broader exclusions when it already includes the task's.
    return (
        task_rules - base_rules <= target_rules
        and task_comments - base_comments <= target_comments
    )


def _only_content_conflicts(messages, conflicts):
    records = messages.rstrip(b"\0").split(b"\0")
    affected = set()
    cursor = 0
    while cursor < len(records):
        if not records[cursor].isdigit():
            return False
        count = int(records[cursor])
        kind = cursor + 1 + count
        if kind + 1 >= len(records):
            return False
        if records[kind] == b"CONFLICT (contents)" and count == 1:
            affected.add(records[cursor + 1])
        elif records[kind] != b"Auto-merging":
            return False
        cursor = kind + 2
    return bool(affected) and affected == conflicts.keys()


def _resolve_net_conflicts(checkout, output, commits):
    staged, separator, messages = output.partition(b"\0\0")
    tree, *entries = staged.split(b"\0")
    protected = {
        b".gitignore": _target_preserves_ignores,
        b"pyproject.toml": _target_preserves_testpaths,
    }
    conflicts = {}
    for entry in entries:
        metadata, _, path = entry.partition(b"\t")
        fields = metadata.split()
        if len(fields) != 3 or path not in protected or fields[0] != b"100644":
            raise RebaseConflict("the pull request tip has unresolved rebase conflicts")
        _, oid, stage = fields
        conflicts.setdefault(path, {})[stage] = oid
    if not separator or not _only_content_conflicts(messages, conflicts):
        raise RebaseConflict("the pull request tip has unresolved rebase conflicts")

    resolved = []
    for path, stages in conflicts.items():
        name = path.decode("ascii")
        oids = [
            git(checkout, "rev-parse", f"{commit}:{name}").strip() for commit in commits
        ]
        if stages != dict(zip((b"1", b"2", b"3"), oids, strict=True)):
            raise RebaseConflict("could not match a conflict to its original files")
        if any(
            int(git(checkout, "cat-file", "-s", oid.decode("ascii"))) > 262144
            for oid in oids
        ):
            raise RebaseConflict("a conflicting configuration file is too large")
        try:
            contents = [
                git(checkout, "cat-file", "blob", oid.decode("ascii")).decode()
                for oid in oids
            ]
        except UnicodeDecodeError:
            raise RebaseConflict(
                "a conflicting configuration file is not UTF-8"
            ) from None
        if not protected[path](*contents):
            raise RebaseConflict("the pull request tip has unresolved rebase conflicts")
        resolved.append((name, oids[1].decode("ascii")))

    git(checkout, "read-tree", "--reset", tree.decode("ascii"))
    for path, target_blob in resolved:
        git(checkout, "update-index", "--cacheinfo", "100644", target_blob, path)
    return git(checkout, "write-tree").decode("ascii").strip()


def _integrate_tip(checkout, origin, target, fork_tree, identity):
    bases = git(checkout, "merge-base", "--all", target, origin).decode().splitlines()
    if len(bases) != 1:
        raise RebaseConflict("could not identify a unique common ancestor for the tip")
    ancestor = bases[0]
    if git(checkout, "rev-parse", f"{ancestor}^{{tree}}").decode().strip() != fork_tree:
        raise RebaseConflict("the tip and replay have different base trees")
    # Repository merge attributes can silently accept a conflicting file. Only
    # use this fallback when Git can apply the ordinary three-way merge rules.
    for commit in (ancestor, target, origin):
        paths = git(checkout, "ls-tree", "-r", "--name-only", "-z", commit)
        if any(
            path.rsplit(b"/", 1)[-1] == b".gitattributes" for path in paths.split(b"\0")
        ):
            raise RebaseConflict(
                "tip integration cannot use repository merge attributes"
            )
    try:
        tree = (
            git(
                checkout,
                "-c",
                f"core.attributesFile={os.devnull}",
                "merge-tree",
                "--write-tree",
                "-z",
                target,
                origin,
            )
            .rstrip(b"\0")
            .decode("ascii")
        )
    except subprocess.CalledProcessError as error:
        if error.returncode != 1:
            raise
        tree = _resolve_net_conflicts(
            checkout, error.stdout, (ancestor, target, origin)
        )

    date = git(checkout, "show", "-s", "--format=%aI", origin).decode().strip()
    return (
        git(
            checkout,
            "-c",
            f"user.name={identity[0]}",
            "-c",
            f"user.email={identity[1]}",
            "commit-tree",
            "--no-gpg-sign",
            tree,
            "-p",
            target,
            "-m",
            "Integrate the pull request tip onto the current base",
            commit_date=date,
        )
        .decode()
        .strip()
    )


def branch_parent(
    github,
    checkout: Path,
    repo: str,
    origin: str | None,
    target: TargetBase,
    identity: tuple[str, str] = BOT_IDENTITY,
    previous_base: TargetBase | None = None,
) -> str:
    github.fetch(target.repo, checkout, target.sha)
    if origin is None or origin == target.sha:
        return target.sha
    github.fetch(repo, checkout, origin)
    try:
        fork = git(checkout, "merge-base", target.sha, origin).decode().strip()
        if fork == target.sha:
            return origin
        if fork == origin:
            return target.sha
        if previous_base is not None and previous_base.sha != target.sha:
            github.fetch(previous_base.repo, checkout, previous_base.sha)
            old_fork = (
                git(checkout, "merge-base", previous_base.sha, origin).decode().strip()
            )
            shared = git(checkout, "merge-base", old_fork, fork).decode().strip()
            if shared == fork:
                fork = old_fork
            elif shared != old_fork:
                raise RebaseConflict(
                    "could not separate task commits from the previous PR base"
                )
        # A task branch may have merged a squash-integrated base and explicitly
        # retained exactly the base tree. Replaying commits before that merge
        # would resurrect changes the merge already settled.
        fork_tree = git(checkout, "rev-parse", f"{fork}^{{tree}}").decode().strip()
        merges = git(
            checkout,
            "log",
            "--first-parent",
            "--merges",
            "--format=%H %T %P",
            f"{fork}..{origin}",
        )
        for line in merges.decode().splitlines():
            merge, tree, *parents = line.split()
            if tree == fork_tree and fork in parents[1:]:
                fork = merge
                break
        git(checkout, "checkout", "--detach", "--force", origin)
        try:
            git(
                checkout,
                "-c",
                f"user.name={identity[0]}",
                "-c",
                f"user.email={identity[1]}",
                "-c",
                "rerere.enabled=false",
                "rebase",
                "--rebase-merges=rebase-cousins",
                "--reapply-cherry-picks",
                "--empty=keep",
                "--committer-date-is-author-date",
                "--no-update-refs",
                "--no-fork-point",
                "--onto",
                target.sha,
                fork,
            )
        except subprocess.CalledProcessError as error:
            if error.returncode != 1 or not git(
                checkout, "ls-files", "--unmerged", "-z"
            ):
                raise
            git(checkout, "rebase", "--abort")
            # Replay can resurrect conflicts already resolved in a branch merge.
            try:
                return _integrate_tip(checkout, origin, target.sha, fork_tree, identity)
            except RebaseConflict as conflict:
                raise RebaseConflict(
                    "task branch could not be rebased onto the current PR base: "
                    f"{conflict}"
                ) from None
        return git(checkout, "rev-parse", "HEAD").decode().strip()
    except subprocess.CalledProcessError:
        raise RebaseConflict(
            "task branch could not be rebased onto the current PR base"
        ) from None


def transplant_patch(checkout: Path, parent: str, patch: bytes) -> bytes:
    try:
        git(checkout, "read-tree", parent)
        git(checkout, "apply", "--cached", "--3way", "--binary", "-", data=patch)
    except subprocess.CalledProcessError:
        raise RebaseConflict(
            "verified patch conflicts with the current PR base"
        ) from None
    updated = git(
        checkout,
        "diff",
        "--cached",
        "--binary",
        "--full-index",
        "--no-renames",
        "--no-ext-diff",
        "--no-textconv",
        parent,
    )
    if not updated:
        raise RebaseConflict(
            "verified patch has no changes after rebasing onto the current PR base"
        )
    return updated

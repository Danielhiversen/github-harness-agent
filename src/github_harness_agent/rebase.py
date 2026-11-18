"""Prepare trusted local history against an immutable target commit."""

import os
import subprocess
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


def git(path, *args, data=None):
    env = {
        name: os.environ[name]
        for name in ("PATH", "HOME", "TMPDIR")
        if name in os.environ
    }
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_EDITOR": "true",
            "GIT_SEQUENCE_EDITOR": "true",
        }
    )
    return subprocess.run(
        ["git", "-C", str(path), "-c", f"core.hooksPath={os.devnull}", *args],
        input=data,
        check=True,
        capture_output=True,
        env=env,
    ).stdout


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
        git(checkout, "checkout", "--detach", "--force", origin)
        git(
            checkout,
            "-c",
            f"user.name={identity[0]}",
            "-c",
            f"user.email={identity[1]}",
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

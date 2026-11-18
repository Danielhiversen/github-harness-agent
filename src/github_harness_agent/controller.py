"""Run one leased implementation or review pass and checkpoint external writes."""

import base64
import dataclasses
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from .executor import NeedsClarification, TaskSnapshot
from .intake import eligible, maintainer
from .publish import BranchUpdate, Journal, Publisher, PublishRequest, StaleHead
from .reviews import ReviewItem, collect_reviews, map_feedback, reply_to_review
from .store import LeaseLost


def git(path, *args, data=None):
    return subprocess.run(
        ["git", "-C", str(path), *args], input=data, check=True, capture_output=True
    ).stdout


def prepare_commit(path, parent, patch, issue):
    # Stage patch directly in the index. No checked-out project code runs here.
    git(path, "read-tree", parent)
    git(path, "apply", "--cached", "--binary", "-", data=patch)
    tree = git(path, "write-tree").decode().strip()
    return (
        git(
            path,
            "-c",
            "user.name=GitHub Harness Agent",
            "-c",
            "user.email=agent@users.noreply.github.com",
            "commit-tree",
            tree,
            "-p",
            parent,
            "-m",
            f"Address issue #{issue}",
        )
        .decode()
        .strip()
    )


def review_source(path: Path, parent: str, review_path: str) -> str | None:
    if "\0" in review_path:
        return None
    object_name = f"{parent}:{review_path}"
    try:
        size = int(git(path, "cat-file", "-s", object_name))
        if size > 1_048_576:
            return None
        blob = git(path, "cat-file", "blob", object_name)
        return None if b"\0" in blob else blob.decode()
    except (subprocess.CalledProcessError, UnicodeDecodeError):
        return None


def acknowledge_feedback(data, item):
    if item.key not in data["seen"]:
        data["seen"].append(item.key)
    if item.thread and item.thread not in data["threads"]:
        data["threads"].append(item.thread)


class Controller:
    def __init__(self, store, api, executor, config, bot_login):
        self.store, self.api, self.executor = store, api, executor
        self.config, self.bot_login = config, bot_login

    def run(self, job):
        data = job["data"]
        repo, issue = job["repo"], job["issue"]
        api, store, cfg = self.api, self.store, self.config
        label = cfg.get("label", "agent:ready")
        branch = cfg.get("branch_prefix", "agent/") + f"issue-{issue}"

        def authorize():
            store.check(job)
            current = api.rest("GET", f"/repos/{repo}/issues/{issue}")
            if not eligible(api, repo, current, label):
                store.cancel(repo, issue)
                raise LeaseLost()

        authorize()
        journal = Journal(data, lambda: store.save(job))
        publisher = Publisher(api, journal, authorize)
        stage = data["stage"]
        parent = data["base_sha"]
        instructions = f"{data['title']}\n\n{data['body']}"
        checks = tuple(tuple(check) for check in cfg["checks"])
        if not checks:
            raise ValueError("Configure at least one trusted check")
        if stage != "implement":
            pr = api.rest("GET", f"/repos/{repo}/pulls/{data['pull']}")
            if pr["state"] != "open":
                store.save(job, "complete")
                return
            if stage == "watch":
                batch = collect_reviews(
                    api,
                    repo=repo,
                    pull=data["pull"],
                    bot_login=self.bot_login,
                    authorized=lambda user: maintainer(api, repo, user),
                    known_threads=data["threads"],
                    seen=data["seen"],
                )
                if not batch.items:
                    store.save(job, "queued", delay=45)
                    return
                if data["round"] >= 3:
                    store.save(job, "paused")
                    return
                data["feedback"] = [dataclasses.asdict(item) for item in batch.items]
                data["review_head"] = batch.head
                data["stage"] = stage = "repair"
                store.save(job)
            parent = data["review_head"]

        if stage not in {"reply", "question"}:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                git(path, "init", "-q")
                api.fetch(repo, path, parent)
                git(path, "checkout", "--detach", parent)
                guides = git(path, "ls-tree", "--name-only", parent).splitlines()
                if b"AGENTS.md" in guides:
                    instructions += (
                        "\nRepository instructions:\n"
                        + git(path, "show", parent + ":AGENTS.md").decode()
                    )
                if "prepared" not in data:
                    if stage == "repair":
                        mapped = []
                        for raw in data["feedback"]:
                            item = ReviewItem(**raw)
                            if item.outdated and item.path:
                                source = review_source(path, parent, item.path)
                                current = map_feedback(item, source)
                            else:
                                current = item
                            if current is not None:
                                mapped.append(dataclasses.asdict(current))
                                continue
                            if cfg.get("mode", "shadow") != "shadow":
                                try:
                                    reply_to_review(
                                        api,
                                        journal,
                                        authorize,
                                        repo=repo,
                                        pull=data["pull"],
                                        item=item,
                                        commit=None,
                                        checks=(),
                                        bot_login=self.bot_login,
                                        question="Could you point me to the line or section on the "
                                        "current pull request that this request applies to?",
                                        expected_head=parent,
                                    )
                                except StaleHead:
                                    data["stage"] = "watch"
                                    store.save(job, "queued")
                                    return
                                acknowledge_feedback(data, item)
                        data["feedback"] = mapped
                        if not mapped:
                            data["stage"] = "watch"
                            if cfg.get("mode", "shadow") == "shadow":
                                store.save(job, "shadow_complete")
                            else:
                                store.save(job, "queued", delay=45)
                            return
                        instructions += (
                            "\nReview requests against this commit:\n"
                            + json.dumps(mapped)
                        )
                    task = TaskSnapshot(
                        repo,
                        parent,
                        instructions,
                        checks,
                        protected_paths=tuple(cfg.get("protected_paths", ())),
                        repository_guidance=cfg.get("reviewed_guidance", ""),
                    )

                    def cancelled():
                        try:
                            store.check(job)
                        except LeaseLost:
                            return True
                        return False

                    try:
                        result = self.executor.execute(task, path, cancelled=cancelled)
                    except NeedsClarification as err:
                        if stage != "repair":
                            raise
                        if cfg.get("mode", "shadow") == "shadow":
                            store.save(job, "shadow_complete")
                            return
                        data.update(stage="question", question=str(err))
                        store.save(job, "queued")
                        return
                    payload = base64.b64encode(result.patch).decode()
                    # Preserve the verified commit object for exact crash recovery.
                    commit = prepare_commit(path, parent, result.patch, issue)
                    data["prepared"] = {
                        "patch": payload,
                        "sha": commit,
                        "object": base64.b64encode(
                            git(path, "cat-file", "commit", commit)
                        ).decode(),
                        "checks": [
                            " ".join(check.command) + f" (exit {check.returncode})"
                            for check in result.checks
                        ],
                        "trace": [
                            dataclasses.asdict(route) for route in result.route_trace
                        ],
                        "usage": result.usage,
                        "refinement": dataclasses.asdict(result.refinement)
                        if result.refinement
                        else None,
                        "digest": hashlib.sha256(result.patch).hexdigest(),
                    }
                    store.save(job)
                prepared = data["prepared"]
                patch = base64.b64decode(prepared["patch"])
                git(path, "read-tree", parent)
                git(path, "apply", "--cached", "--binary", "-", data=patch)
                git(path, "write-tree")
                sha = (
                    git(
                        path,
                        "hash-object",
                        "-w",
                        "-t",
                        "commit",
                        "--stdin",
                        data=base64.b64decode(prepared["object"]),
                    )
                    .decode()
                    .strip()
                )
                if sha != prepared["sha"]:
                    raise ValueError("Stored commit digest mismatch")
                authorize()
                if cfg.get("mode", "shadow") == "shadow":
                    store.save(job, "shadow_complete")
                    return
                if stage == "implement":
                    published = publisher.publish(
                        PublishRequest(
                            repo,
                            issue,
                            data["base"],
                            parent,
                            branch,
                            path,
                            sha,
                            data["title"],
                            f"Implements issue #{issue}.",
                            tuple(prepared["checks"]),
                            ready=True,
                        )
                    )
                    data.update(pull=published.number, stage="watch")
                else:
                    remote_head = pr["head"]["sha"]
                    if remote_head not in {parent, sha}:
                        data.pop("prepared")
                        data["stage"] = "watch"
                        store.save(job, "queued")
                        return
                    try:
                        publisher.advance(
                            BranchUpdate(
                                repo, branch, parent, path, sha, pull=data["pull"]
                            )
                        )
                    except StaleHead:
                        del data["prepared"]
                        data["stage"] = "watch"
                        store.save(job, "queued")
                        return
                    data.update(stage="reply", round=data["round"] + 1)
                data["last_commit"] = sha
                data["last_checks"] = prepared["checks"]
                data["last_evidence"] = {
                    key: prepared[key]
                    for key in ("trace", "usage", "refinement", "digest")
                }
                del data["prepared"]
                store.save(job)
        if data["stage"] in {"reply", "question"}:
            for raw in data["feedback"]:
                item = ReviewItem(**raw)
                question = data.get("question")
                try:
                    reply_to_review(
                        api,
                        journal,
                        authorize,
                        repo=repo,
                        pull=data["pull"],
                        item=item,
                        commit=None if question else data["last_commit"],
                        checks=data["last_checks"],
                        bot_login=self.bot_login,
                        question=question,
                        expected_head=data["review_head"] if question else None,
                    )
                except StaleHead:
                    if not question:
                        ref = api.rest("GET", f"/repos/{repo}/git/ref/heads/{branch}")
                        if ref["object"]["sha"] == data["last_commit"]:
                            # GitHub can serve the previous PR head briefly after a push.
                            # Keep the verified reply pending while the PR view catches up.
                            store.save(job, "queued", delay=15)
                            return
                    data.pop("question", None)
                    data["stage"] = "watch"
                    store.save(job, "queued")
                    return
                acknowledge_feedback(data, item)
            data.pop("question", None)
            data["stage"] = "watch"
        store.save(job, "queued", delay=45)

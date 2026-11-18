"""Run one leased implementation or review pass and checkpoint external writes."""

import base64
import dataclasses
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from .executor import AttemptRejected, ExecutionError, NeedsClarification, TaskSnapshot
from .intake import adoptable_pull, eligible, maintainer
from .publish import (
    BranchUpdate,
    ClosedPull,
    Journal,
    Publisher,
    PublishRequest,
    StaleHead,
    TargetMoved,
)
from .rebase import (
    BOT_IDENTITY,
    RebaseConflict,
    TargetBase,
    branch_parent,
    git,
    transplant_patch,
)
from .reviews import ReviewItem, collect_reviews, map_feedback, reply_to_review
from .store import LeaseLost


def prepare_commit(path, parent, patch, issue, identity=BOT_IDENTITY, *, adopted=False):
    # Stage patch directly in the index. No checked-out project code runs here.
    git(path, "read-tree", parent)
    git(path, "apply", "--cached", "--binary", "-", data=patch)
    tree = git(path, "write-tree").decode().strip()
    return (
        git(
            path,
            "-c",
            f"user.name={identity[0]}",
            "-c",
            f"user.email={identity[1]}",
            "commit-tree",
            tree,
            "-p",
            parent,
            "-m",
            f"Address review feedback on PR #{issue}"
            if adopted
            else f"Address issue #{issue}",
        )
        .decode()
        .strip()
    )


def verified_commit(
    path,
    parent,
    patch,
    issue,
    checks,
    origin,
    target,
    identity,
    previous_base,
    *,
    adopted=False,
):
    sha = prepare_commit(path, parent, patch, issue, identity, adopted=adopted)
    return {
        "patch": base64.b64encode(patch).decode(),
        "sha": sha,
        "object": base64.b64encode(git(path, "cat-file", "commit", sha)).decode(),
        "checks": [
            " ".join(check.command) + f" (exit {check.returncode})" for check in checks
        ],
        "digest": hashlib.sha256(patch).hexdigest(),
        "sync": {
            "target": dataclasses.asdict(target),
            "origin": origin,
            "parent": parent,
            "identity": list(identity),
            "previous_base": dataclasses.asdict(previous_base)
            if previous_base
            else None,
        },
    }


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
        try:
            self._run(job)
        except TargetMoved:
            self.store.save(job, "queued", delay=5)
        except RebaseConflict as error:
            job["data"]["pause_reason"] = str(error)
            self.store.save(job, "paused")
        except ClosedPull:
            self.store.save(job, "complete")

    def _run(self, job):
        data = job["data"]
        repo, issue = job["repo"], job["issue"]
        api, store, cfg = self.api, self.store, self.config
        adopted = job.get("kind") == "pull"
        label = cfg.get("label", "agent:ready")
        branch = (
            data.get("branch") or cfg.get("branch_prefix", "agent/") + f"issue-{issue}"
        )

        def read_pull():
            result = api.rest("GET", f"/repos/{repo}/pulls/{data['pull']}")
            if result["state"] != "open":
                raise ClosedPull("pull request is closed")
            if adopted and not adoptable_pull(result, repo, self.bot_login, branch):
                raise ClosedPull(
                    "pull request no longer matches its adopted branch and author"
                )
            return result

        def authorize():
            store.check(job)
            if adopted:
                read_pull()
            else:
                current = api.rest("GET", f"/repos/{repo}/issues/{issue}")
                if not eligible(api, repo, current, label):
                    store.cancel(repo, issue)
                    raise LeaseLost()
            store.check(job)

        def cancelled():
            try:
                store.check(job)
            except LeaseLost:
                return True
            return False

        authorize()
        journal = Journal(data, lambda: store.save(job))
        publisher = Publisher(api, journal, authorize)
        stage = data["stage"]
        if adopted and (stage == "implement" or data["pull"] != issue):
            raise ValueError("adopted jobs can only review their existing pull request")
        origin = None
        instructions = f"{data['title']}\n\n{data['body']}"
        checks = tuple(tuple(check) for check in cfg["checks"])
        if not checks:
            raise ValueError("Configure at least one trusted check")
        if stage != "implement":
            pr = read_pull()
            if adopted:
                instructions = (
                    "The pull request implementation is already present. Treat its description as "
                    "background. Change only what the outstanding review requests require, and add "
                    "the requested regression tests.\n\n"
                    f"{pr['title']}\n\n{pr.get('body') or ''}"
                )
            if stage == "watch":
                batch = collect_reviews(
                    api,
                    repo=repo,
                    pull=data["pull"],
                    bot_login=self.bot_login,
                    authorized=lambda user: maintainer(api, repo, user),
                    known_threads=data["threads"],
                    seen=data["seen"],
                    adopted_at=data["adopted_at"] if adopted else None,
                )
                for item in batch.handled:
                    acknowledge_feedback(data, item)
                if not batch.items:
                    store.save(job, "queued", delay=45)
                    return
                if data["round"] >= 3:
                    store.save(job, "paused")
                    return
                data["feedback"] = [dataclasses.asdict(item) for item in batch.items]
                data["review_head"] = batch.head
                data["stage"] = stage = "repair"
                data.pop("model_attempt_failures", None)
                store.save(job)
            origin = data["review_head"]

        if stage not in {"reply", "question"}:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                git(path, "init", "-q", "--template=")
                pull = data["pull"] if stage == "repair" else None
                latest = publisher.target(repo, data["base"], pull)
                identity = getattr(api, "commit_identity", None) or BOT_IDENTITY
                prepared = data.get("prepared")
                previous_base = None
                if pull is not None:
                    previous_base = (
                        TargetBase(**data["last_target"])
                        if data.get("last_target")
                        else (TargetBase(repo, data["base"], data["base_sha"]))
                    )
                if prepared is None:
                    target = latest
                    parent = branch_parent(
                        api, path, repo, origin, target, identity, previous_base
                    )
                elif sync := prepared.get("sync"):
                    target = TargetBase(**sync["target"])
                    identity = tuple(sync["identity"])
                    previous_base = (
                        TargetBase(**sync["previous_base"])
                        if sync.get("previous_base")
                        else None
                    )
                    if sync["origin"] != origin:
                        raise RebaseConflict(
                            "stored patch has a different original task branch"
                        )
                    parent = branch_parent(
                        api, path, repo, origin, target, identity, previous_base
                    )
                    if parent != sync["parent"]:
                        raise RebaseConflict(
                            "could not reproduce the verified rebased parent"
                        )
                else:
                    # Continue jobs prepared before syncing was introduced.
                    target = TargetBase(repo, data["base"], data["base_sha"])
                    parent = origin or target.sha
                    api.fetch(repo, path, parent)
                git(path, "checkout", "--detach", "--force", parent)
                guides = git(path, "ls-tree", "--name-only", parent).splitlines()
                if b"AGENTS.md" in guides:
                    instructions += (
                        "\nRepository instructions:\n"
                        + git(path, "show", parent + ":AGENTS.md").decode()
                    )
                if prepared is not None and stage == "repair":
                    instructions += (
                        "\nReview requests against this commit:\n"
                        + json.dumps(data["feedback"])
                    )
                if "prepared" not in data:
                    if stage == "repair":
                        mapped = []
                        for raw in data["feedback"]:
                            item = ReviewItem(**raw)
                            if item.path and (item.outdated or parent != origin):
                                source = review_source(path, parent, item.path)
                                current = map_feedback(
                                    dataclasses.replace(item, outdated=True), source
                                )
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
                                        expected_head=origin,
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
                    failures = data.get("model_attempt_failures", [])
                    excluded = tuple(
                        (
                            record["candidate"]["route"],
                            record["candidate"]["configured_model"],
                        )
                        for record in failures
                    )
                    task = TaskSnapshot(
                        repo,
                        parent,
                        instructions,
                        checks,
                        protected_paths=tuple(cfg.get("protected_paths", ())),
                        repository_guidance=cfg.get("reviewed_guidance", ""),
                        excluded_models=excluded,
                    )

                    try:
                        if publisher.target(repo, data["base"], pull) != target:
                            raise TargetMoved("target ref changed before editing")
                        if (
                            pull is not None
                            and publisher.branch_head(repo, branch) != origin
                        ):
                            data["stage"] = "watch"
                            store.save(job, "queued")
                            return
                        result = self.executor.execute(task, path, cancelled=cancelled)
                    except AttemptRejected as err:
                        candidate = err.candidate
                        if (candidate.route, candidate.configured_model) in excluded:
                            data["pause_reason"] = (
                                "gateway selected an excluded coding candidate"
                            )
                            store.save(job, "paused")
                            return
                        failures.append(
                            {
                                "candidate": dataclasses.asdict(candidate),
                                "reason": str(err),
                                "trace": [
                                    dataclasses.asdict(route)
                                    for route in err.route_trace
                                ],
                            }
                        )
                        data["model_attempt_failures"] = failures
                        store.save(job, "queued")
                        return
                    except NeedsClarification as err:
                        if stage != "repair":
                            raise
                        if cfg.get("mode", "shadow") == "shadow":
                            store.save(job, "shadow_complete")
                            return
                        data.update(stage="question", question=str(err))
                        store.save(job, "queued")
                        return
                    data["prepared"] = {
                        **verified_commit(
                            path,
                            parent,
                            result.patch,
                            issue,
                            result.checks,
                            origin,
                            target,
                            identity,
                            previous_base,
                            adopted=adopted,
                        ),
                        "trace": [
                            dataclasses.asdict(route) for route in result.route_trace
                        ],
                        "model_attempt_failures": data.pop(
                            "model_attempt_failures", []
                        ),
                        "usage": result.usage,
                        "refinement": dataclasses.asdict(result.refinement)
                        if result.refinement
                        else None,
                    }
                    store.save(job)
                prepared = data["prepared"]
                latest = publisher.target(repo, data["base"], pull)
                published_head = publisher.branch_head(repo, branch)
                if pull is not None and published_head not in {origin, prepared["sha"]}:
                    del data["prepared"]
                    data["stage"] = "watch"
                    store.save(job, "queued")
                    return
                if latest != target and published_head != prepared["sha"]:
                    parent = branch_parent(
                        api, path, repo, origin, latest, identity, previous_base
                    )
                    patch = transplant_patch(
                        path, parent, base64.b64decode(prepared["patch"])
                    )
                    verify_task = TaskSnapshot(
                        repo,
                        parent,
                        instructions,
                        checks,
                        protected_paths=tuple(cfg.get("protected_paths", ())),
                        repository_guidance=cfg.get("reviewed_guidance", ""),
                    )

                    try:
                        verified = self.executor.verify_patch(
                            verify_task,
                            path,
                            patch,
                            cancelled=cancelled,
                        )
                    except ExecutionError as error:
                        store.check(job)
                        raise RebaseConflict(
                            "required verification failed after rebasing"
                        ) from error
                    target = latest
                    prepared.update(
                        verified_commit(
                            path,
                            parent,
                            verified.patch,
                            issue,
                            verified.checks,
                            origin,
                            target,
                            identity,
                            previous_base,
                            adopted=adopted,
                        )
                    )
                    store.save(job)
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
                            target.ref,
                            parent,
                            branch,
                            path,
                            sha,
                            data["title"],
                            f"Implements issue #{issue}.",
                            tuple(prepared["checks"]),
                            ready=True,
                            target=target,
                        )
                    )
                    data.update(pull=published.number, stage="watch")
                else:
                    try:
                        publisher.advance(
                            BranchUpdate(
                                repo,
                                branch,
                                origin,
                                path,
                                sha,
                                pull=data["pull"],
                                target=target,
                                verified_parent=parent,
                            )
                        )
                    except StaleHead:
                        del data["prepared"]
                        data["stage"] = "watch"
                        store.save(job, "queued")
                        return
                    data.update(stage="reply", round=data["round"] + 1)
                data["last_commit"] = sha
                data["last_target"] = dataclasses.asdict(target)
                data["last_checks"] = prepared["checks"]
                data["last_evidence"] = {
                    key: prepared[key]
                    for key in (
                        "trace",
                        "usage",
                        "refinement",
                        "digest",
                        "model_attempt_failures",
                    )
                    if key in prepared
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

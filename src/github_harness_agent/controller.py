"""Run one leased implementation or review pass and checkpoint external writes."""

import base64
import dataclasses
import hashlib
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from threading import Lock

from .checkpoint import DraftCheckpoint
from .ci import CI_BATCH_SIZE, format_ci_feedback, read_ci_report
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
from .test_updates import authorized_test_update, requests_new_tests
from .store import LeaseLost
from .task_context import build_task_context, github_history, review_source_context


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


def draft_scope(job, stage, origin, target, config):
    data = job["data"]
    reviews = (
        sorted(item["key"] for item in data.get("feedback", ()))
        if stage == "repair"
        else []
    )
    ci_keys = (
        sorted(item["key"] for item in data.get("ci_feedback", ()))
        if stage == "repair"
        else []
    )
    owner = ("pull", data["pull"]) if stage == "repair" else (job["kind"], job["issue"])
    identity = [
        job["repo"],
        owner,
        stage,
        origin,
        target.repo,
        target.ref,
        reviews,
        ci_keys,
        (data["title"], data["body"]) if stage == "implement" else (),
        config.get("reviewed_guidance", ""),
        config.get("protected_paths", ()),
        config.get("checks", ()),
    ]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest(), reviews


def restore_drafts(api, checkout, repo, origin, parent, scope, envelope):
    if envelope.get("scope") != scope:
        return {}
    try:
        old_parent = envelope["parent"]
        drafts = {
            key: DraftCheckpoint.from_data(value)
            for key, value in envelope["drafts"].items()
        }
        if envelope["head"] != origin or any(
            key != draft.key or draft.base_sha != old_parent
            for key, draft in drafts.items()
        ):
            raise ValueError("inconsistent draft metadata")
        if old_parent == parent or not drafts:
            return drafts
        old_target = TargetBase(**envelope["target"])
        old_base = (
            TargetBase(**envelope["previous_base"])
            if envelope.get("previous_base")
            else None
        )
        recreated = branch_parent(
            api,
            checkout,
            repo,
            origin,
            old_target,
            tuple(envelope["identity"]),
            old_base,
        )
        if recreated != old_parent:
            raise RebaseConflict("could not reproduce the saved draft checkout")
        try:
            return {
                key: dataclasses.replace(
                    draft,
                    base_sha=parent,
                    patch=transplant_patch(checkout, parent, draft.patch),
                    progress=(),
                )
                for key, draft in drafts.items()
            }
        except RebaseConflict as error:
            raise RebaseConflict(
                "saved draft conflicts with the current target base"
            ) from error
        finally:
            git(checkout, "checkout", "--detach", "--force", parent)
    except (KeyError, TypeError, ValueError) as error:
        raise RebaseConflict("saved draft metadata could not be recovered") from error


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
        pull_checks = cfg.get("pull_checks", {}).get(str(data.get("pull")), [])
        if pull_checks:
            if not isinstance(pull_checks, list) or not all(
                isinstance(cmd, list)
                and cmd
                and all(isinstance(arg, str) and arg for arg in cmd)
                for cmd in pull_checks
            ):
                raise ValueError(
                    "Each pull check must be a nonempty command argument list"
                )
            cfg = {**cfg, "checks": [*cfg["checks"], *pull_checks]}
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
        resumed_repair = stage == "repair"
        if adopted and (stage == "implement" or data["pull"] != issue):
            raise ValueError("adopted jobs can only review their existing pull request")
        origin = None
        instructions = f"{data['title']}\n\n{data['body']}"
        checks = tuple(tuple(check) for check in cfg["checks"])
        if not checks:
            raise ValueError("Configure at least one trusted check")

        def review_batch():
            try:
                return collect_reviews(
                    api,
                    repo=repo,
                    pull=data["pull"],
                    bot_login=self.bot_login,
                    authorized=lambda user: maintainer(api, repo, user),
                    known_threads=data["threads"],
                    seen=data["seen"],
                    adopted_at=data["adopted_at"] if adopted else None,
                )
            except StaleHead as error:
                raise TargetMoved(
                    "pull request head changed while collecting feedback"
                ) from error

        def retire_ci_failures(head):
            selected = data.get("ci_feedback")
            if not selected:
                return False
            report = read_ci_report(api, repo, head)
            active = {failure.key for failure in report.failures}
            retired = {
                failure["key"]
                for failure in selected
                if failure["source"] in report.sources and failure["key"] not in active
            }
            if not retired:
                return False
            data["ci_feedback"] = [
                failure for failure in selected if failure["key"] not in retired
            ]
            data["ci_seen"] = [
                key for key in data.get("ci_seen", []) if key not in retired
            ]
            data.pop("prepared", None)
            data.pop("draft_checkpoints", None)
            return True

        if stage != "implement":
            pr = read_pull()
            if adopted:
                instructions = (
                    "The pull request implementation is already present. Treat its description as "
                    "background. Change only what the outstanding review requests or CI failures "
                    "require, and add the requested regression tests.\n\n"
                    f"{pr['title']}\n\n{pr.get('body') or ''}"
                )
            if stage == "watch":
                data.pop("draft_checkpoints", None)
                batch = review_batch()
                for item in batch.handled:
                    acknowledge_feedback(data, item)
                head = data.get("last_commit")
                ci_feedback = []
                if (
                    head
                    and head == batch.head == pr["head"]["sha"]
                    and adoptable_pull(pr, repo, self.bot_login, branch)
                    and publisher.branch_head(repo, branch) == head
                ):
                    seen = {
                        key
                        for key in data.get("ci_seen", ())
                        if key.startswith(f"{head}:")
                    }
                    attempted = () if batch.items else data.get("ci_attempted", ())
                    report = read_ci_report(api, repo, head)
                    ci_feedback = [
                        failure
                        for failure in report.failures
                        if failure.key not in seen and failure.identity not in attempted
                    ][: max(0, min(CI_BATCH_SIZE, 128 - len(seen)))]
                if not batch.items and not ci_feedback:
                    store.save(job, "queued", delay=45)
                    return
                if data["round"] >= data.get("allowed_rounds", 3):
                    data["pause_reason"] = "review repair round limit reached"
                    store.save(job, "paused")
                    return
                data["feedback"] = [dataclasses.asdict(item) for item in batch.items]
                data["ci_feedback"] = [dataclasses.asdict(item) for item in ci_feedback]
                if ci_feedback:
                    data["ci_seen"] = sorted(seen | {item.key for item in ci_feedback})
                data["review_head"] = batch.head
                data["stage"] = stage = "repair"
                data.pop("model_attempt_failures", None)
                data.pop("prior_model_attempt_failures", None)
                store.save(job)
            elif stage == "repair" and data.get("ci_feedback"):
                head = data["review_head"]
                current_branch = publisher.branch_head(repo, branch)
                pushed = data.get("prepared", {}).get("sha")
                if not pushed or current_branch != pushed:
                    if head == data.get("last_commit") == pr["head"][
                        "sha"
                    ] == current_branch and adoptable_pull(
                        pr, repo, self.bot_login, branch
                    ):
                        if retire_ci_failures(head) and not (
                            data["ci_feedback"] or data["feedback"]
                        ):
                            data["stage"] = "watch"
                            store.save(job, "queued", delay=45)
                            return
                    else:
                        data.pop("ci_feedback", None)
                        data.pop("prepared", None)
                        data.pop("draft_checkpoints", None)
                        data["stage"] = "watch"
                        store.save(job, "queued")
                        return
            if resumed_repair and "prepared" not in data:
                batch = review_batch()
                if batch.head != data["review_head"]:
                    data["stage"] = "watch"
                    data.pop("ci_feedback", None)
                    if "draft_checkpoints" in data:
                        data["retired_draft_checkpoints"] = data.pop(
                            "draft_checkpoints"
                        )
                        raise RebaseConflict(
                            "Pull request head changed. The previous draft was retired. "
                            "Use resume to remap outstanding feedback on the current head."
                        )
                    store.save(job, "queued")
                    return
                if batch.head != pr["head"]["sha"]:
                    raise TargetMoved("pull request head views do not agree yet")
                previous_keys = {item["key"] for item in data.get("feedback", ())}
                current_keys = {item.key for item in batch.items}
                if current_keys != previous_keys:
                    if "draft_checkpoints" in data:
                        data["retired_draft_checkpoints"] = data.pop(
                            "draft_checkpoints"
                        )
                    data.pop("model_attempt_failures", None)
                    data.pop("prior_model_attempt_failures", None)
                for item in batch.handled:
                    acknowledge_feedback(data, item)
                data["feedback"] = [dataclasses.asdict(item) for item in batch.items]
                if not data["feedback"] and not data.get("ci_feedback"):
                    data["stage"] = "watch"
                    store.save(job, "queued", delay=45)
                    return
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
                guide = (
                    git(path, "show", parent + ":AGENTS.md").decode(errors="replace")
                    if b"AGENTS.md" in guides
                    else ""
                )

                def current_context():
                    feedback = data.get("feedback", ()) if stage == "repair" else ()
                    failures = (
                        data.get("model_attempt_failures")
                        or data.get("prior_model_attempt_failures")
                        or ()
                    )
                    previous = (
                        failures[-1].get("reason")
                        if failures
                        else data.get("last_error")
                    )
                    return build_task_context(
                        instructions,
                        repository_guide=guide,
                        feedback=feedback,
                        previous_failure=previous or "",
                        extra_evidence=format_ci_feedback(data.get("ci_feedback", ())),
                        history=github_history(api, repo, issue, pull, feedback),
                        source_context=review_source_context(
                            feedback,
                            lambda name: review_source(path, parent, name),
                            test_selector=current_test_update(),
                        ),
                    )

                def current_test_update():
                    if not adopted or stage != "repair" or pr["head"]["sha"] != origin:
                        return None
                    return authorized_test_update(
                        item
                        for item in data.get("feedback", ())
                        if item.get("test_update_head") == origin
                    )

                def needs_new_tests():
                    return stage == "repair" and any(
                        requests_new_tests(item["body"])
                        for item in data.get("feedback", ())
                    )

                if (
                    prepared is not None
                    and (
                        (needs_new_tests() and not prepared.get("verified_new_tests"))
                        or prepared.get("pull_checks", []) != pull_checks
                    )
                    and publisher.branch_head(repo, branch) == origin
                ):
                    data.pop("prepared")
                    store.save(job, "queued")
                    return

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
                        if not mapped and not data.get("ci_feedback"):
                            data["stage"] = "watch"
                            if cfg.get("mode", "shadow") == "shadow":
                                store.save(job, "shadow_complete")
                            else:
                                store.save(job, "queued", delay=45)
                            return
                    failures = data.get("model_attempt_failures", [])
                    excluded = tuple(
                        (
                            record["candidate"]["route"],
                            record["candidate"]["configured_model"],
                        )
                        for record in failures
                    )
                    scope, review_keys = draft_scope(job, stage, origin, target, cfg)
                    existing = data.get("draft_checkpoints", {})
                    if existing and existing.get("scope") != scope:
                        data.pop("draft_checkpoints")
                        store.save(job)
                    try:
                        checkpoints = restore_drafts(
                            api, path, repo, origin, parent, scope, existing
                        )
                    except RebaseConflict as error:
                        data["retired_draft_checkpoints"] = data.pop(
                            "draft_checkpoints"
                        )
                        raise RebaseConflict(
                            f"{error}. The previous draft was retired. Use resume to start from the current checkout."
                        ) from error
                    saved = {
                        "scope": scope,
                        "head": origin,
                        "reviews": review_keys,
                        "ci_keys": sorted(
                            item["key"] for item in data.get("ci_feedback", ())
                        ),
                        "target": dataclasses.asdict(target),
                        "parent": parent,
                        "identity": identity,
                        "previous_base": dataclasses.asdict(previous_base)
                        if previous_base
                        else None,
                        "drafts": {
                            key: draft.to_data() for key, draft in checkpoints.items()
                        },
                    }
                    checkpoint_lock = Lock()
                    retry_offset = data.get("provider_retry_count", 0)
                    if (
                        checkpoints
                        and job.get("state") == "running"
                        and not any(draft.last_editor for draft in checkpoints.values())
                    ):
                        retry_offset = max(retry_offset, 1)

                    def save_checkpoint(draft):
                        with checkpoint_lock:
                            if (
                                data["stage"] != stage
                                or draft.base_sha != parent
                                or data.get("review_head") != origin
                                and stage == "repair"
                            ):
                                raise LeaseLost()
                            if draft.patch:
                                saved["drafts"][draft.key] = draft.to_data()
                            else:
                                saved["drafts"].pop(draft.key, None)
                            if saved["drafts"]:
                                data["draft_checkpoints"] = saved
                            else:
                                data.pop("draft_checkpoints", None)
                            store.save(job)

                    context = current_context()
                    task = TaskSnapshot(
                        repo,
                        parent,
                        context.instructions,
                        checks,
                        protected_paths=tuple(cfg.get("protected_paths", ())),
                        repository_guidance=cfg.get("reviewed_guidance", ""),
                        excluded_models=excluded,
                        provider_offset=retry_offset,
                        checkpoint_scope=scope,
                        checkpoints=checkpoints,
                        save_checkpoint=save_checkpoint,
                        observe_model=lambda *event: store.observe_model(job, *event),
                        context=context,
                        verify_base_first=stage == "repair"
                        and bool(data.get("feedback") or data.get("ci_feedback")),
                        inline_review_threads=len(
                            {
                                item["thread"]
                                for item in data.get("feedback", ())
                                if item.get("thread")
                                and item.get("path")
                                and item.get("reply_to")
                            }
                        )
                        if stage == "repair"
                        else 0,
                        test_update=current_test_update(),
                        require_new_tests=needs_new_tests(),
                        pull_checks=tuple(map(tuple, pull_checks)),
                    )

                    try:
                        if publisher.target(repo, data["base"], pull) != target:
                            raise TargetMoved("target ref changed before editing")
                        if (
                            pull is not None
                            and publisher.branch_head(repo, branch) != origin
                        ):
                            if checkpoints:
                                data["retired_draft_checkpoints"] = data.pop(
                                    "draft_checkpoints"
                                )
                                data["stage"] = "watch"
                                raise RebaseConflict(
                                    "Pull request head changed. The previous draft was retired. "
                                    "Use resume to remap outstanding feedback on the current head."
                                )
                            data["stage"] = "watch"
                            store.save(job, "queued")
                            return
                        logging.getLogger(__name__).info(
                            "Starting %s %s #%s at %.12s (provider retry %s)",
                            repo,
                            job["kind"],
                            issue,
                            parent,
                            task.provider_offset,
                        )
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
                            data.pop("draft_checkpoints", None)
                            store.save(job, "shadow_complete")
                            return
                        if not data["feedback"] and data.get("ci_feedback"):
                            data.pop("ci_feedback")
                            data.pop("draft_checkpoints", None)
                            data["stage"] = "watch"
                            data["ci_diagnostic"] = (
                                "CI evidence needs maintainer investigation"
                            )
                            store.save(job, "queued", delay=45)
                            return
                        data.update(stage="question", question=str(err))
                        data.pop("draft_checkpoints", None)
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
                        "prior_model_attempt_failures": data.pop(
                            "prior_model_attempt_failures", []
                        ),
                        "usage": result.usage,
                        "verified_new_tests": task.require_new_tests,
                        "pull_checks": pull_checks,
                        "refinement": dataclasses.asdict(result.refinement)
                        if result.refinement
                        else None,
                    }
                    data.pop("draft_checkpoints", None)
                    store.save(job)
                prepared = data["prepared"]
                latest = publisher.target(repo, data["base"], pull)
                published_head = publisher.branch_head(repo, branch)
                if pull is not None and published_head not in {origin, prepared["sha"]}:
                    del data["prepared"]
                    data["stage"] = "watch"
                    store.save(job, "queued")
                    return
                if (
                    pull is not None
                    and published_head == origin
                    and retire_ci_failures(origin)
                ):
                    if not (data["ci_feedback"] or data["feedback"]):
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
                    context = current_context()
                    verify_task = TaskSnapshot(
                        repo,
                        parent,
                        context.instructions,
                        checks,
                        protected_paths=tuple(cfg.get("protected_paths", ())),
                        repository_guidance=cfg.get("reviewed_guidance", ""),
                        test_update=current_test_update(),
                        require_new_tests=needs_new_tests(),
                        pull_checks=tuple(map(tuple, pull_checks)),
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
                ci_evidence = []
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
                    human_feedback = bool(data["feedback"])
                    attempted = [] if human_feedback else data.get("ci_attempted", [])
                    ci_evidence = data.pop("ci_feedback", [])
                    addressed = [item["identity"] for item in ci_evidence]
                    data["ci_attempted"] = list(
                        dict.fromkeys([*attempted, *addressed])
                    )[-32:]
                    data.pop("ci_diagnostic", None)
                    data.update(
                        stage="reply" if human_feedback else "watch",
                        round=data["round"] + 1,
                    )
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
                        "prior_model_attempt_failures",
                    )
                    if key in prepared
                }
                if ci_evidence:
                    data["last_evidence"]["ci_failures"] = ci_evidence
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

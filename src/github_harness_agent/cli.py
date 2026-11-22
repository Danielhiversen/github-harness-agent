import argparse
import json
import logging
import os
import re
import time
import tomllib
from math import isfinite
from pathlib import Path

import uvicorn

from .activity import control, status
from .intake import adopt_open_pulls, reconcile
from .store import PROVIDER_EXHAUSTED, LeaseLost, Store
from .webhooks import create_app


_EXCLUDED_MODELS = "all eligible models excluded"
_PROVIDER_REASONS = {
    _EXCLUDED_MODELS,
    "account or model ineligible",
    "authenticated gateway request failed",
    "deadline exceeded",
    "free eligibility unverified",
    "initial coding response did not use a tool",
    "invalid or missing credentials",
    "malformed request or upstream response",
    "missing endpoint or credentials",
    "model unavailable",
    "rate limited",
    "request in progress",
    "temporarily unavailable",
    "tool capability unverified",
    "upstream connection failed",
    "upstream timeout",
}


def _provider_failure_map(failures, aliases):
    if not isinstance(failures, dict):
        return {}
    safe = {}
    for alias in sorted(aliases):
        if alias not in failures:
            continue
        reason = failures[alias]
        recognized = isinstance(reason, str) and (
            reason in _PROVIDER_REASONS
            or re.fullmatch(r"upstream HTTP [1-5][0-9]{2}", reason)
        )
        safe[alias] = reason if recognized else "unclassified failure"
        if len(safe) == 32:
            break
    return safe


def _defer_provider(store, job, error, aliases):
    data = job["data"]
    previous = data.get("provider_retry_count", 0)
    count = 1 + previous if type(previous) is int and previous > 0 else 1
    now = store.clock()
    delay = min(3600, 30 * 2 ** min(count - 1, 7))
    try:
        cooldown = float(error.retry_at) if type(error.retry_at) in {int, float} else 0
    except OverflowError:
        cooldown = 0
    if isfinite(cooldown):
        delay = max(delay, cooldown - now)
    data.update(
        last_error=PROVIDER_EXHAUSTED,
        provider_retry_count=count,
        provider_failures=_provider_failure_map(error.failures, aliases),
        retry_at=now + delay,
    )
    if error.failures == {"gateway": _EXCLUDED_MODELS}:
        failures = data.pop("model_attempt_failures", [])
        if failures:
            data["prior_model_attempt_failures"] = [
                {
                    "candidate": {
                        field: value[:160]
                        for field, value in attempt.get("candidate", {}).items()
                        if field in {"role", "route", "model", "configured_model"}
                        and isinstance(value, str)
                    },
                    "reason": str(attempt.get("reason", "")).split(":", 1)[0][:160],
                }
                for attempt in failures[-16:]
            ]
    store.save(job, "queued", delay=delay)


def _configured_limit(repo, config, key, default, maximum):
    value = config.get(key, default)
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{repo} {key} must be an integer between 1 and {maximum}")
    return value


def run_worker(store, config, once=False, *, pull: tuple[str, int] | None = None):
    from .controller import Controller
    from .executor import DockerRunner, Harness2Executor, Limits, PlainExecutor
    from .github import GitHubApp, GitHubNamedAccount
    from .providers import ModelGatewayClient, ProviderExhausted, load_manifest

    repositories = config["repositories"]
    if pull is not None:
        repo, number = pull
        if not once or type(number) is not int or not 0 < number <= 2**63 - 1:
            raise ValueError(
                "targeted execution requires once and a positive pull number"
            )
        if repo not in repositories:
            raise ValueError(f"pull repository is not configured: {repo}")
        repositories = {repo: repositories[repo]}
    defaults = Limits()
    limits = {}
    for repo, cfg in repositories.items():
        model_requests = _configured_limit(
            repo, cfg, "model_requests", defaults.model_requests, 256
        )
        rollout_requests = None
        if "rollout_requests" in cfg:
            image = cfg.get("verification_image")
            if not isinstance(image, str) or not image.strip() or image.startswith("-"):
                raise ValueError(
                    f"{repo} rollout_requests requires an explicit verification_image"
                )
            rollout_requests = _configured_limit(
                repo, cfg, "rollout_requests", None, model_requests
            )
        limits[repo] = Limits(
            wall_time=_configured_limit(repo, cfg, "wall_time", defaults.wall_time, 3600),
            model_requests=model_requests,
            concurrent_probes=_configured_limit(repo, cfg, "concurrent_probes", defaults.concurrent_probes, 6),
            commands=_configured_limit(repo, cfg, "commands", defaults.commands, 512),
            rollout_requests=rollout_requests,
        )
    auth_mode = config.get("github_auth", "app")
    app = account = None
    if auth_mode == "app":
        app = GitHubApp(
            os.environ["GITHUB_APP_ID"],
            Path(os.environ["GITHUB_APP_PRIVATE_KEY_FILE"]).read_text(),
        )
    elif auth_mode == "named-account":
        account = GitHubNamedAccount(
            config["bot_login"],
            os.environ.get("GH_CONFIG_DIR", ""),
            gh=config.get("github_cli", "gh"),
        )
    else:
        raise ValueError("github_auth must be app or named-account")
    gateway = ModelGatewayClient.from_env()
    manifest = load_manifest()
    aliases = {
        "gateway",
        *manifest["web"],
        *manifest["catalog"],
        *manifest.get("superagent", {}).get("providers", {}),
    }
    controllers = {}
    for repo, cfg in repositories.items():
        if cfg.get("mode", "shadow") not in {"shadow", "canary"}:
            raise ValueError("mode must be shadow or canary")
        if cfg.get("backend", "plain") not in {"plain", "harness2"}:
            raise ValueError("backend must be plain or harness2")
        api = app.installation(cfg["installation_id"]) if app is not None else account
        backend = (
            Harness2Executor if cfg.get("backend") == "harness2" else PlainExecutor
        )
        runtime = cfg.get("worker_runtime", "runsc")
        executor = backend(
            DockerRunner(cfg["worker_image"], runtime=runtime),
            gateway,
            limits[repo],
            verification_runner=(
                DockerRunner(cfg["verification_image"], runtime=runtime)
                if "verification_image" in cfg
                else None
            ),
        )
        controllers[repo] = Controller(store, api, executor, cfg, config["bot_login"])
    store.recover_provider_pauses(repositories, pull=pull)
    next_sync = 0
    online = set()
    while True:
        events = store.events() if pull is None else ()
        if pull is None and (events or time.monotonic() >= next_sync):
            failed = set()
            for repo, controller in controllers.items():
                try:
                    if account is not None:
                        count = adopt_open_pulls(
                            controller.api,
                            store,
                            repo,
                            controller.config,
                            config["bot_login"],
                        )
                        if count:
                            logging.getLogger(__name__).info(
                                "Adopted %s open PR(s) in %s", count, repo
                            )
                    reconcile(controller.api, store, repo, controller.config)
                    if repo not in online:
                        logging.getLogger(__name__).info(
                            "Listening to %s as %s for label %s",
                            repo,
                            config["bot_login"],
                            controller.config.get("label", "agent:ready"),
                        )
                        online.add(repo)
                except Exception as err:
                    logging.error(
                        "Reconciliation failed for %s (%s)", repo, type(err).__name__
                    )
                    failed.add(repo)
                    online.discard(repo)
            for event in events:
                payload = json.loads(event["payload"])
                if payload["repository"]["full_name"] not in failed:
                    store.ack(event["id"])
            next_sync = time.monotonic() + 60
        job = store.claim(ttl=1800, pull=pull)
        if job:
            before = (job["data"]["stage"], job["data"].get("last_commit"))
            try:
                controllers[job["repo"]].run(job)
            except LeaseLost:
                pass
            except ProviderExhausted as err:
                try:
                    _defer_provider(store, job, err, aliases)
                except LeaseLost:
                    pass
            except Exception as err:
                logging.error(
                    "Task %s#%s failed (%s)",
                    job["repo"],
                    job["issue"],
                    type(err).__name__,
                )
                job["data"]["last_error"] = type(err).__name__
                count = job["data"].get("consecutive_failures", 0) + 1
                job["data"]["consecutive_failures"] = count
                delay = max(30, min(3600, 2 ** min(count, 10)))
                try:
                    store.save(job, "paused" if count >= 5 else "queued", delay)
                except LeaseLost:
                    pass
            else:
                after = (job["data"]["stage"], job["data"].get("last_commit"))
                store.clear_failures(
                    job,
                    progressed=(
                        after[0] in {"watch", "reply"}
                        or after[1] != before[1]
                        or "prepared" in job["data"]
                    ),
                )
                if after != before:
                    logging.getLogger(__name__).info(
                        "%s %s #%s entered %s (commit %s)",
                        job["repo"],
                        job["kind"],
                        job["issue"],
                        after[0],
                        after[1] or "pending",
                    )
        if once:
            if pull is not None and not job:
                logging.getLogger(__name__).info("No due pull job for %s #%s", *pull)
            return
        if not job:
            time.sleep(3)


def main():
    logging.basicConfig()
    logging.getLogger("github_harness_agent").setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", help="Configured repository for a targeted once run")
    parser.add_argument(
        "--pull", type=int, help="Existing pull job number for a targeted once run"
    )
    parser.add_argument(
        "command",
        choices=[
            "serve",
            "work",
            "once",
            "status",
            "retry",
            "resume",
            "pause",
            "fence",
        ],
    )
    args = parser.parse_args()
    pull = None
    targeted = args.repo is not None or args.pull is not None
    if args.command in {"pause", "retry", "resume"} or targeted:
        if (
            not args.repo
            or args.pull is None
            and args.command != "status"
            or (args.pull is not None and not 0 < args.pull <= 2**63 - 1)
            or args.command not in {"once", "status", "pause", "resume", "retry"}
        ):
            parser.error(
                "targeted execution requires once, --repo and a positive --pull number; "
                "status and per-PR controls also support targeting"
            )
        if args.pull is not None:
            pull = (args.repo, args.pull)
    with args.config.open("rb") as f:
        config = tomllib.load(f)
    if args.repo is not None and args.repo not in config["repositories"]:
        parser.error(f"pull repository is not configured: {args.repo}")
    store = Store(config["database"])
    if args.command == "serve":
        app = create_app(
            store,
            os.environ["GITHUB_WEBHOOK_SECRET"].encode(),
            set(config["repositories"]),
        )
        uvicorn.run(
            app, host=config.get("host", "127.0.0.1"), port=config.get("port", 8080)
        )
    elif args.command in {"status", "pause", "retry", "resume"}:
        if args.command != "status":
            try:
                control(store, args.repo, args.pull, args.command)
            except ValueError as error:
                parser.error(str(error))
        rows = list(status(store, args.repo, args.pull))
        if args.pull and not rows:
            parser.error("Pull request has no tracked job")
        for row in rows:
            print(json.dumps(row))
    elif args.command == "fence":
        store.fence()
    else:
        run_worker(store, config, once=args.command == "once", pull=pull)

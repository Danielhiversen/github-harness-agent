import argparse
import json
import logging
import os
import time
import tomllib
from pathlib import Path

import uvicorn

from .intake import reconcile
from .store import LeaseLost, Store
from .webhooks import create_app


def run_worker(store, config, once=False):
    from .controller import Controller
    from .executor import DockerRunner, Harness2Executor, PlainExecutor
    from .github import GitHubApp
    from .providers import ModelGatewayClient, ProviderExhausted

    app = GitHubApp(
        os.environ["GITHUB_APP_ID"],
        Path(os.environ["GITHUB_APP_PRIVATE_KEY_FILE"]).read_text(),
    )
    gateway = ModelGatewayClient.from_env()
    controllers = {}
    for repo, cfg in config["repositories"].items():
        if cfg.get("mode", "shadow") not in {"shadow", "canary"}:
            raise ValueError("mode must be shadow or canary")
        if cfg.get("backend", "plain") not in {"plain", "harness2"}:
            raise ValueError("backend must be plain or harness2")
        api = app.installation(cfg["installation_id"])
        backend = (
            Harness2Executor if cfg.get("backend") == "harness2" else PlainExecutor
        )
        executor = backend(DockerRunner(cfg["worker_image"]), gateway)
        controllers[repo] = Controller(store, api, executor, cfg, config["bot_login"])
    next_sync = 0
    while True:
        events = store.events()
        if events or time.monotonic() >= next_sync:
            failed = set()
            for repo, controller in controllers.items():
                try:
                    reconcile(controller.api, store, repo, controller.config)
                except Exception as err:
                    logging.error(
                        "Reconciliation failed for %s (%s)", repo, type(err).__name__
                    )
                    failed.add(repo)
            for event in events:
                payload = json.loads(event["payload"])
                if payload["repository"]["full_name"] not in failed:
                    store.ack(event["id"])
            next_sync = time.monotonic() + 60
        job = store.claim(ttl=1800)
        if job:
            try:
                controllers[job["repo"]].run(job)
            except LeaseLost:
                pass
            except ProviderExhausted as err:
                job["data"]["last_error"] = "provider routes exhausted"
                job["data"]["retry_at"] = err.retry_at
                try:
                    if err.retry_at is None:
                        store.save(job, "paused")
                    else:
                        store.save(job, "queued", max(1, err.retry_at - time.time()))
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
                with store.tx() as db:
                    db.execute(
                        """UPDATE tasks SET data=json_remove(data,'$.consecutive_failures')
                        WHERE repo=? AND issue=? AND generation=? AND token IS NULL""",
                        (job["repo"], job["issue"], job["generation"]),
                    )
        if once:
            return
        if not job:
            time.sleep(3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("command", choices=["serve", "work", "once", "status", "fence"])
    args = parser.parse_args()
    with args.config.open("rb") as f:
        config = tomllib.load(f)
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
    elif args.command == "status":
        with store.tx() as db:
            for row in db.execute(
                "SELECT repo,issue,state,generation,attempts FROM tasks"
            ):
                print(json.dumps(dict(row)))
    elif args.command == "fence":
        with store.tx() as db:
            db.execute("""UPDATE tasks SET generation=generation+1,token=NULL,state='paused'
                WHERE state IN ('running','queued')""")
    else:
        run_worker(store, config, once=args.command == "once")

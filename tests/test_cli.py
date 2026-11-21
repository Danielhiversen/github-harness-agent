import json
import sys

import pytest

from github_harness_agent import cli, executor, github, providers
from github_harness_agent import controller
from github_harness_agent.store import LeaseLost, Store


def test_worker_uses_named_account_without_app_credentials(monkeypatch, tmp_path):
    api = object()
    created = []
    scans = []

    def get_named_account(login, config_dir, *, gh):
        assert (login, config_dir, gh) == (
            "example-agent",
            str(tmp_path),
            "/opt/bin/gh",
        )
        return api

    def make_runner(image, *, runtime):
        created.append((image, runtime))

    def reconcile(client, store, repo, config):
        assert client is api
        assert repo == "example/project"
        scans.append("issues")

    def adopt(client, store, repo, config, login):
        assert client is api and (repo, login) == ("example/project", "example-agent")
        scans.append("pulls")

    monkeypatch.delenv("GITHUB_APP_ID", raising=False)
    monkeypatch.delenv("GITHUB_APP_PRIVATE_KEY_FILE", raising=False)
    monkeypatch.setenv("GH_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(github, "GitHubNamedAccount", get_named_account)
    monkeypatch.setattr(executor, "DockerRunner", make_runner)
    monkeypatch.setattr(providers.ModelGatewayClient, "from_env", lambda: object())
    monkeypatch.setattr(cli, "reconcile", reconcile)
    monkeypatch.setattr(cli, "adopt_open_pulls", adopt)
    cli.run_worker(
        Store(tmp_path / "agent.db"),
        {
            "github_auth": "named-account",
            "github_cli": "/opt/bin/gh",
            "bot_login": "example-agent",
            "repositories": {
                "example/project": {
                    "worker_image": "agent:test",
                    "worker_runtime": "runc",
                }
            },
        },
        once=True,
    )
    assert created == [("agent:test", "runc")]
    assert scans == ["pulls", "issues"]


def test_targeted_once_uses_repo_limits_and_runs_only_exact_eligible_pull(
    monkeypatch, tmp_path
):
    job_store = Store(tmp_path / "agent.db", lambda: 100)
    repo = "example/project"
    job_store.enqueue(repo, 20, {"stage": "implement"})
    for number, name in [(21, repo), (20, "example/other"), (20, repo)]:
        job_store.adopt_pull(
            name,
            number,
            {
                "stage": "watch",
                "pull": number,
                "author": "example-agent",
                "branch": f"existing/{number}",
                "linked_issues": [],
            },
            "agent/",
        )
    run = []

    class Controller:
        def __init__(self, store, api, backend, config, bot_login):
            assert backend.limits.model_requests == 120
            assert backend.limits.wall_time == 2700
            assert backend.limits.commands == 200
            assert backend.limits.rollout_requests == 24
            assert backend.runner.image == "worker:test"
            assert backend.verification_runner.image == "verify:test"
            assert (
                backend.runner.runtime == backend.verification_runner.runtime == "runc"
            )
            self.store = store

        def run(self, job):
            run.append((job["repo"], job["kind"], job["issue"]))
            self.store.save(job, "queued", delay=30)

    def unexpected_reconciliation(*args):
        pytest.fail("targeted once must only claim the existing job")

    monkeypatch.setattr(github, "GitHubNamedAccount", lambda *args, **kw: object())
    monkeypatch.setattr(providers.ModelGatewayClient, "from_env", lambda: object())
    monkeypatch.setattr(controller, "Controller", Controller)
    monkeypatch.setattr(cli, "adopt_open_pulls", unexpected_reconciliation)
    monkeypatch.setattr(cli, "reconcile", unexpected_reconciliation)
    config = {
        "github_auth": "named-account",
        "bot_login": "example-agent",
        "repositories": {
            repo: {
                "worker_image": "worker:test",
                "verification_image": "verify:test",
                "worker_runtime": "runc",
                "model_requests": 120,
                "wall_time": 2700,
                "commands": 200,
                "rollout_requests": 24,
            },
            "example/other": {"worker_image": "worker:test", "model_requests": False},
        },
    }
    job_store.receive("delivery", "issues", {"repository": {"full_name": repo}})
    cli.run_worker(job_store, config, once=True, pull=(repo, 20))
    cli.run_worker(job_store, config, once=True, pull=(repo, 20))
    assert run == [(repo, "pull", 20)]
    assert len(job_store.events()) == 1
    with job_store.tx() as db:
        assert db.execute("SELECT SUM(attempts) FROM tasks").fetchone()[0] == 1


@pytest.mark.parametrize(
    ("key", "invalid"),
    [
        ("model_requests", True),
        ("model_requests", 0),
        ("model_requests", 257),
        ("wall_time", 3601),
        ("commands", 4.5),
        ("commands", -1),
        ("commands", 513),
    ],
)
def test_rejects_invalid_repository_budgets_before_using_credentials(
    tmp_path, key, invalid
):
    config = {
        "github_auth": "named-account",
        "bot_login": "example-agent",
        "repositories": {
            "example/project": {"worker_image": "worker:test", key: invalid}
        },
    }
    with pytest.raises(ValueError, match=key):
        cli.run_worker(Store(tmp_path / "agent.db"), config, once=True)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"rollout_requests": 4}, "verification_image"),
        (
            {"rollout_requests": 0, "verification_image": "checks:test"},
            "rollout_requests",
        ),
        (
            {"rollout_requests": True, "verification_image": "checks:test"},
            "rollout_requests",
        ),
        (
            {"rollout_requests": 5, "verification_image": "checks:test"},
            "rollout_requests",
        ),
    ],
)
def test_rollout_budget_requires_a_verification_image_and_fits_shared_budget(
    tmp_path, override, message
):
    repo = {"worker_image": "worker:test", "model_requests": 4}
    repo.update(override)
    config = {
        "github_auth": "named-account",
        "bot_login": "example-agent",
        "repositories": {"example/project": repo},
    }
    with pytest.raises(ValueError, match=message):
        cli.run_worker(Store(tmp_path / "agent.db"), config, once=True)


def test_targeted_claim_respects_existing_lease_and_fences_expired_token(tmp_path):
    clock = [100]
    store = Store(tmp_path / "agent.db", lambda: clock[0])
    repo = "example/project"
    store.enqueue(repo, 5, {"stage": "implement"})
    store.adopt_pull(
        repo,
        20,
        {
            "stage": "watch",
            "pull": 20,
            "author": "example-agent",
            "branch": "existing",
            "linked_issues": [],
        },
        "agent/",
    )
    first = store.claim(ttl=10, pull=(repo, 20))
    assert first["kind"] == "pull"
    assert store.claim(pull=(repo, 20)) is None
    assert store.claim()["kind"] == "issue"

    clock[0] += 11
    recovered = store.claim(pull=(repo, 20))
    assert recovered["issue"] == 20 and recovered["token"] != first["token"]
    with pytest.raises(LeaseLost):
        store.check(first)
    store.save(recovered, "paused")
    assert store.claim(pull=(repo, 20)) is None


@pytest.mark.parametrize(
    "arguments",
    [
        ["once", "--repo", "example/project", "--pull", "20"],
        ["work", "--repo", "example/project", "--pull", "20"],
        ["once", "--pull", "20"],
        ["once", "--repo", "unknown/repo", "--pull", "20"],
    ],
)
def test_targeted_cli_requires_once_and_an_exact_configured_pull(
    monkeypatch, tmp_path, arguments
):
    config = tmp_path / "agent.toml"
    database = tmp_path / "queue.db"
    config.write_text(f'database = "{database}"\n[repositories."example/project"]\n')
    monkeypatch.setattr(
        sys, "argv", ["github-harness-agent", "--config", str(config), *arguments]
    )
    called = []
    monkeypatch.setattr(
        cli, "run_worker", lambda store, cfg, **options: called.append(options)
    )

    if arguments[0] == "once" and "example/project" in arguments:
        cli.main()
        assert called == [{"once": True, "pull": ("example/project", 20)}]
    else:
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2
        assert not called and not database.exists()


def _provider_test_worker(monkeypatch, tmp_path, handler):
    repo = "example/project"
    now = [100]
    store = Store(tmp_path / "provider.db", lambda: now[0])
    config = {
        "github_auth": "named-account",
        "bot_login": "example-agent",
        "repositories": {repo: {"worker_image": "worker:test"}},
    }

    class Controller:
        def __init__(self, store, api, backend, config, bot_login):
            self.api, self.config = api, config

        def run(self, job):
            handler(store, job)

    monkeypatch.setattr(github, "GitHubNamedAccount", lambda *args, **kwargs: object())
    monkeypatch.setattr(providers.ModelGatewayClient, "from_env", lambda: object())
    monkeypatch.setattr(controller, "Controller", Controller)
    monkeypatch.setattr(cli, "adopt_open_pulls", lambda *args: 0)
    monkeypatch.setattr(cli, "reconcile", lambda *args: None)
    return repo, now, store, config


def _seed_retry_pull(store, repo, number):
    store.adopt_pull(
        repo,
        number,
        {
            "stage": "repair",
            "pull": number,
            "branch": f"existing/{number}",
            "author": "example-agent",
            "linked_issues": [],
        },
        "agent/",
    )


def _stored_job(store, kind, number):
    with store.tx() as db:
        row = db.execute(
            "SELECT * FROM tasks WHERE kind=? AND issue=?", (kind, number)
        ).fetchone()
    result = dict(row)
    result["data"] = json.loads(result["data"])
    return result


def test_provider_cooldown_and_backoff_are_durable_and_only_clear_on_progress(
    monkeypatch, tmp_path
):
    outcomes = iter(
        [
            providers.ProviderExhausted(
                150,
                {
                    "kilo-free": "rate limited",
                    "groq-free": "secret or response text",
                    "secret-route-token": "rate limited",
                },
            ),
            providers.ProviderExhausted(
                None, {"gateway": "authenticated gateway request failed"}
            ),
            "retry-candidate",
            "watch",
        ]
    )
    contexts = []

    def run(store, job):
        contexts.append(job["data"].get("provider_retry_count", 0))
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        if outcome == "watch":
            job["data"]["stage"] = "watch"
        store.save(job, "queued")

    repo, now, store, config = _provider_test_worker(monkeypatch, tmp_path, run)
    _seed_retry_pull(store, repo, 20)

    def work():
        cli.run_worker(store, config, once=True, pull=(repo, 20))

    work()
    record = _stored_job(store, "pull", 20)
    assert record["state"] == "queued" and record["due"] == 150
    assert record["data"]["provider_failures"] == {
        "kilo-free": "rate limited",
        "groq-free": "unclassified failure",
    }
    work()
    assert contexts == [0]
    now[0] = 150
    work()
    record = _stored_job(store, "pull", 20)
    assert record["state"] == "queued" and record["due"] == 210
    assert (
        record["data"]["retry_at"] == 210
        and record["data"]["provider_retry_count"] == 2
    )
    now[0] = 210
    work()
    assert _stored_job(store, "pull", 20)["data"]["provider_retry_count"] == 2
    work()
    final = _stored_job(store, "pull", 20)["data"]
    assert (
        not {"last_error", "retry_at", "provider_failures", "provider_retry_count"}
        & final.keys()
    )
    assert contexts == [0, 1, 2, 2]


def test_all_excluded_retries_after_resetting_only_active_candidate_exclusions(
    monkeypatch, tmp_path
):
    def run(store, job):
        job["data"]["provider_retry_count"] = 100
        job["data"]["model_attempt_failures"] = [
            {
                "candidate": {
                    "route": "kilo-free",
                    "configured_model": f"test-{number}",
                },
                "reason": "empty patch",
                "trace": ["large trace"],
            }
            for number in range(20)
        ]
        raise providers.ProviderExhausted(
            101, {"gateway": "all eligible models excluded"}
        )

    repo, now, store, config = _provider_test_worker(monkeypatch, tmp_path, run)
    _seed_retry_pull(store, repo, 20)
    cli.run_worker(store, config, once=True, pull=(repo, 20))
    record = _stored_job(store, "pull", 20)
    assert record["state"] == "queued" and record["due"] == now[0] + 3600
    assert "model_attempt_failures" not in record["data"]
    assert record["data"]["prior_model_attempt_failures"] == [
        {
            "candidate": {"route": "kilo-free", "configured_model": f"test-{number}"},
            "reason": "empty patch",
        }
        for number in range(4, 20)
    ]


def test_worker_recovers_only_legacy_provider_pauses_within_lease_scope(
    monkeypatch, tmp_path
):
    seen = []

    def run(store, job):
        seen.append((job["kind"], job["issue"], job["data"]["provider_retry_count"]))
        store.save(job, "queued", delay=30)

    repo, now, store, config = _provider_test_worker(monkeypatch, tmp_path, run)
    legacy = {"last_error": "provider routes exhausted", "retry_at": None}
    store.enqueue(repo, 5, {"stage": "implement", **legacy})
    store.save(store.claim(), "paused")
    for number, state, override in [
        (20, "paused", {}),
        (21, "paused", {"pause_reason": "manual fence"}),
        (22, "paused", {"pause_reason": "requires rebase"}),
        (23, "paused", {"round": 3, "stage": "watch"}),
        (24, "complete", {}),
        (25, "paused", {"retry_at": 150}),
    ]:
        _seed_retry_pull(store, repo, number)
        job = store.claim(pull=(repo, number))
        job["data"].update(legacy | override)
        store.save(job, state)

    cli.run_worker(store, config, once=True, pull=(repo, 20))
    assert seen == [("pull", 20, 1)]
    assert _stored_job(store, "issue", 5)["state"] == "paused"
    cli.run_worker(store, config, once=True)
    assert seen == [("pull", 20, 1), ("issue", 5, 1)]
    assert _stored_job(store, "pull", 20)["due"] == now[0] + 30
    assert all(
        _stored_job(store, "pull", number)["state"] == "paused"
        for number in (21, 22, 23, 25)
    )
    assert _stored_job(store, "pull", 24)["state"] == "complete"


def test_fence_keeps_provider_pauses_manual_on_future_worker_startup(tmp_path):
    store = Store(tmp_path / "provider.db", lambda: 100)
    repo = "example/project"
    store.enqueue(
        repo,
        5,
        {
            "stage": "implement",
            "last_error": "provider routes exhausted",
            "retry_at": None,
        },
    )
    store.save(store.claim(), "paused")
    store.fence()
    assert store.recover_provider_pauses([repo]) == 0
    assert _stored_job(store, "issue", 5)["data"]["pause_reason"] == "manual fence"


def test_retry_metadata_cleanup_does_not_clear_a_newer_lease(tmp_path):
    store = Store(tmp_path / "provider.db", lambda: 100)
    repo = "example/project"
    store.enqueue(repo, 5, {"stage": "implement"})
    first = store.claim()
    store.save(first, "queued")
    next_attempt = store.claim()
    next_attempt["data"]["provider_retry_count"] = 1
    store.save(next_attempt, "queued", delay=30)

    store.clear_failures(first, progressed=True)
    assert _stored_job(store, "issue", 5)["data"]["provider_retry_count"] == 1

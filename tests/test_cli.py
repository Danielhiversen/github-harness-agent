from github_harness_agent import cli, executor, github, providers
from github_harness_agent.store import Store


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

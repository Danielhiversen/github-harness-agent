import base64
import json
from datetime import datetime, timezone
from subprocess import CalledProcessError, CompletedProcess

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from github_harness_agent.github import GitHubError, GitHubNamedAccount


@pytest.fixture(scope="module")
def private_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def test_installation_auth_reuses_token_across_paginated_results(private_key):
    from github_harness_agent.github import GitHubApp

    token_requests = []

    def handle(request):
        if request.url.path == "/app/installations/42/access_tokens":
            token_requests.append(request.headers["authorization"])
            return httpx.Response(
                201,
                json={
                    "token": "installation-token",
                    "expires_at": "2030-01-01T00:00:00Z",
                },
            )
        assert request.headers["authorization"] == "Bearer installation-token"
        if request.url.params.get("page") == "2":
            return httpx.Response(200, json=[{"number": 2}])
        return httpx.Response(
            200,
            json=[{"number": 1}],
            headers={
                "Link": '<https://api.github.com/repos/a/b/issues?page=2>; rel="next"'
            },
        )

    app = GitHubApp(
        14,
        private_key,
        client=httpx.Client(transport=httpx.MockTransport(handle)),
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert list(app.installation(42).paginate("/repos/a/b/issues")) == [
        {"number": 1},
        {"number": 2},
    ]
    assert len(token_requests) == 1
    jwt = token_requests[0].removeprefix("Bearer ")
    header, claims, signature = jwt.split(".")
    envelope = json.loads(base64.urlsafe_b64decode(header + "=="))
    payload = json.loads(base64.urlsafe_b64decode(claims + "=="))
    assert envelope["alg"] == "RS256"
    assert payload == {"iat": 1767225540, "exp": 1767226140, "iss": "14"}
    public_key = serialization.load_pem_private_key(
        private_key, password=None
    ).public_key()
    public_key.verify(
        base64.urlsafe_b64decode(signature + "=="),
        f"{header}.{claims}".encode(),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )


def test_pagination_rejects_off_origin_next_link(private_key):
    from github_harness_agent.github import GitHubApp, GitHubError

    def handle(request):
        assert request.url.host == "api.github.com"
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(
                201,
                json={
                    "token": "installation-token",
                    "expires_at": "2030-01-01T00:00:00Z",
                },
            )
        return httpx.Response(
            200,
            json=[],
            headers={"Link": '<https://other.example/page>; rel="next"'},
        )

    app = GitHubApp(
        14, private_key, client=httpx.Client(transport=httpx.MockTransport(handle))
    )
    with pytest.raises(GitHubError, match="origin"):
        list(app.installation(42).paginate("/repos/a/b/issues"))


def test_named_account_scopes_the_credential_and_verifies_it_before_rest(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("GH_TOKEN", "wrong-account-environment-token")
    calls = []

    def read_credential(argv, **kwargs):
        assert argv == [
            "gh",
            "auth",
            "token",
            "--hostname",
            "github.com",
            "--user",
            "example-agent",
        ]
        assert kwargs["env"]["GH_CONFIG_DIR"] == str(tmp_path)
        assert "GH_TOKEN" not in kwargs["env"]
        assert kwargs["capture_output"]
        return CompletedProcess(argv, 0, "fixture-token\n", "")

    def handle(request):
        calls.append(request.url.path)
        assert request.headers["Authorization"] == "Bearer fixture-token"
        if request.url.path == "/user":
            return httpx.Response(200, json={"login": "Example-Agent", "id": 12345})
        return httpx.Response(200, json={"number": 4})

    monkeypatch.setattr("github_harness_agent.github.subprocess.run", read_credential)
    api = GitHubNamedAccount(
        "example-agent",
        str(tmp_path),
        client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    assert api.commit_identity == (
        "Example-Agent",
        "12345+Example-Agent@users.noreply.github.com",
    )
    assert api.rest("GET", "/repos/example/project/issues/4") == {"number": 4}
    assert calls == ["/user", "/repos/example/project/issues/4"]


def test_named_account_rejects_wrong_github_login(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "github_harness_agent.github.subprocess.run",
        lambda argv, **kwargs: CompletedProcess(argv, 0, "fixture-token\n", ""),
    )

    def handle(request):
        assert request.url.path == "/user"
        return httpx.Response(200, json={"login": "unexpected-account"})

    with pytest.raises(GitHubError, match="does not match bot_login"):
        GitHubNamedAccount(
            "example-agent",
            str(tmp_path),
            client=httpx.Client(transport=httpx.MockTransport(handle)),
        )


def test_failed_named_credential_command_hides_its_output(
    monkeypatch, tmp_path, capsys
):
    def read_credential(argv, **kwargs):
        raise CalledProcessError(
            1, argv, output="fixture-token", stderr="fixture-token"
        )

    monkeypatch.setattr("github_harness_agent.github.subprocess.run", read_credential)
    with pytest.raises(
        GitHubError, match="Could not load the configured GitHub account credential"
    ) as error:
        GitHubNamedAccount("example-agent", str(tmp_path))
    assert "fixture-token" not in str(error.value)
    assert capsys.readouterr() == ("", "")

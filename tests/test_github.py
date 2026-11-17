import base64
import json
from datetime import datetime, timezone

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


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

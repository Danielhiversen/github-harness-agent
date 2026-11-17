"""Controller-only GitHub App authentication, API calls and Git transport."""

import base64
import json
import os
import re
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


class GitHubError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _base64url(value: bytes) -> bytes:
    return base64.urlsafe_b64encode(value).rstrip(b"=")


def _sha(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("expected a full Git commit SHA")
    return value


def validate_repository(repo: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("expected an owner/repository name")
    if repo.split("/", 1)[1] in {".", ".."}:
        raise ValueError("expected an owner/repository name")
    return repo


class GitHubApp:
    def __init__(
        self,
        app_id,
        private_key: str | bytes,
        *,
        client: httpx.Client | None = None,
        api_url: str = "https://api.github.com",
        clock=_utcnow,
    ):
        self.app_id = str(app_id)
        key = private_key.encode() if isinstance(private_key, str) else private_key
        self._key = serialization.load_pem_private_key(key, password=None)
        if not isinstance(self._key, rsa.RSAPrivateKey):
            raise ValueError("GitHub App authentication requires an RSA private key")
        self._http = client or httpx.Client(timeout=20, trust_env=False)
        self._api_url = api_url.rstrip("/")
        self._clock = clock
        self._installations: dict[int, GitHubClient] = {}
        self._lock = threading.Lock()

    def installation(self, installation_id: int) -> "GitHubClient":
        if type(installation_id) is not int or installation_id <= 0:
            raise ValueError("invalid GitHub installation ID")
        with self._lock:
            client = self._installations.get(installation_id)
            if client is None:
                client = GitHubClient(self, installation_id)
                self._installations[installation_id] = client
            return client

    def _jwt(self) -> str:
        now = int(self._clock().timestamp())
        claims = {"iat": now - 60, "exp": now + 540, "iss": self.app_id}
        encoded = b".".join(
            (
                _base64url(b'{"alg":"RS256","typ":"JWT"}'),
                _base64url(json.dumps(claims, separators=(",", ":")).encode()),
            )
        )
        signature = self._key.sign(encoded, padding.PKCS1v15(), hashes.SHA256())
        return (encoded + b"." + _base64url(signature)).decode()

    def _request(self, token, method, path, *, params=None, json=None):
        url = urljoin(self._api_url + "/", path)
        origin = urlsplit(self._api_url)
        destination = urlsplit(url)
        if (destination.scheme, destination.netloc) != (origin.scheme, origin.netloc):
            raise GitHubError("GitHub API link points outside the configured origin")
        try:
            response = self._http.request(
                method,
                url,
                params=params,
                json=json,
                follow_redirects=False,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "User-Agent": "github-harness-agent",
                },
            )
        except httpx.HTTPError:
            raise GitHubError(f"GitHub {method} request failed in transit") from None
        if not response.is_success:
            raise GitHubError(
                f"GitHub {method} returned HTTP {response.status_code}",
                status=response.status_code,
            )
        return response


def _payload(response):
    if response.status_code == 204:
        return None
    try:
        return response.json()
    except ValueError:
        raise GitHubError("GitHub returned an invalid JSON response") from None


class GitHubClient:
    def __init__(self, app: GitHubApp, installation_id: int):
        self._app = app
        self._installation_id = installation_id
        self._token = ""
        self._expires = datetime.min.replace(tzinfo=timezone.utc)
        self._lock = threading.Lock()

    def _access_token(self) -> str:
        with self._lock:
            if self._expires <= self._app._clock() + timedelta(minutes=1):
                payload = _payload(
                    self._app._request(
                        self._app._jwt(),
                        "POST",
                        f"/app/installations/{self._installation_id}/access_tokens",
                    )
                )
                try:
                    token = payload["token"]
                    expires = datetime.fromisoformat(
                        payload["expires_at"].replace("Z", "+00:00")
                    )
                    if (
                        not isinstance(token, str)
                        or not token
                        or expires.tzinfo is None
                    ):
                        raise ValueError
                except (KeyError, TypeError, AttributeError, ValueError):
                    raise GitHubError(
                        "GitHub returned an invalid installation token"
                    ) from None
                self._token = token
                self._expires = expires
            return self._token

    def _request(self, method, path, *, params=None, json=None):
        token = self._access_token()
        try:
            return self._app._request(token, method, path, params=params, json=json)
        except GitHubError as error:
            if error.status != 401:
                raise
        with self._lock:
            if token == self._token:
                self._expires = datetime.min.replace(tzinfo=timezone.utc)
        return self._app._request(
            self._access_token(), method, path, params=params, json=json
        )

    def rest(self, method, path, *, params=None, json=None):
        return _payload(self._request(method, path, params=params, json=json))

    def paginate(self, path, *, params=None):
        params = {"per_page": 100, **(params or {})}
        visited = set()
        while path:
            response = self._request("GET", path, params=params)
            payload = _payload(response)
            if not isinstance(payload, list):
                raise GitHubError("GitHub returned an invalid collection")
            yield from payload
            path = response.links.get("next", {}).get("url")
            if path and path in visited:
                raise GitHubError("GitHub pagination repeated a page")
            visited.add(path)
            params = None

    def graphql(self, query, variables):
        result = self.rest(
            "POST", "/graphql", json={"query": query, "variables": variables}
        )
        if not isinstance(result, dict) or result.get("errors") or "data" not in result:
            raise GitHubError("GitHub GraphQL operation failed")
        return result["data"]

    def fetch(self, repo: str, checkout: Path, sha: str) -> None:
        url = f"https://github.com/{validate_repository(repo)}.git"
        _git(checkout, "fetch", "--no-tags", url, _sha(sha), token=self._access_token())

    def push(
        self,
        repo: str,
        checkout: Path,
        branch: str,
        expected_head: str,
        commit: str,
    ) -> None:
        url = f"https://github.com/{validate_repository(repo)}.git"
        _sha(commit)
        _sha(expected_head)
        ref = f"refs/heads/{branch}"
        _git(checkout, "check-ref-format", ref)
        parent_line = _git(checkout, "rev-list", "--parents", "-n", "1", commit).split()
        if parent_line != [commit, expected_head]:
            raise GitHubError(
                "verified commit must directly descend from the expected head"
            )
        _git(
            checkout,
            "push",
            f"--force-with-lease={ref}:{expected_head}",
            url,
            f"{commit}:{ref}",
            token=self._access_token(),
        )


def _git(checkout: Path, *args: str, token: str | None = None) -> str:
    # The controller owns this checkout. Keep the token out of URLs, argv and on-disk config.
    config = {
        "core.hooksPath": os.devnull,
        "credential.helper": "",
        "http.followRedirects": "false",
        "protocol.allow": "never",
        "protocol.https.allow": "always",
    }
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        config["http.https://github.com/.extraHeader"] = f"Authorization: Basic {basic}"
    env = {
        name: os.environ[name]
        for name in ("PATH", "HOME", "TMPDIR")
        if name in os.environ
    }
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_CONFIG_COUNT": str(len(config)),
        }
    )
    for index, (key, value) in enumerate(config.items()):
        env[f"GIT_CONFIG_KEY_{index}"] = key
        env[f"GIT_CONFIG_VALUE_{index}"] = value
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        raise GitHubError(f"Git transport failed during {args[0]}") from None
    return result.stdout.strip()

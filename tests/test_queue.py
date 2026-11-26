import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from github_harness_agent.store import LeaseLost, Store
from github_harness_agent.webhooks import create_app


def test_restart_redelivery_and_fencing(tmp_path):
    now = [0]
    store = Store(tmp_path / "db", lambda: now[0])
    repo = "acme/widgets"
    app = TestClient(create_app(store, b"secret", {repo}))
    body = json.dumps({"repository": {"full_name": repo}}).encode()
    headers = {"x-github-event": "issues", "x-github-delivery": "1"}
    assert app.post("/webhook", content=body, headers=headers).status_code == 401
    headers["x-hub-signature-256"] = (
        "sha256=" + hmac.new(b"secret", body, hashlib.sha256).hexdigest()
    )
    for _ in range(2):
        assert app.post("/webhook", content=body, headers=headers).status_code == 202
    assert len(store.events()) == 1
    store.ack("1")
    for _ in range(2):
        store.enqueue(repo, 1, {"base": "abc"})
    original = store.claim(ttl=2)
    assert store.claim() is None
    now[0] += 3
    resumed = Store(store.path, lambda: now[0]).claim()
    assert resumed["token"] != original["token"]
    with pytest.raises(LeaseLost):
        store.save(original)
    store.cancel(repo, 1)
    with pytest.raises(LeaseLost):
        store.check(resumed)


def test_recover_legacy_review_limits_only_requeues_watchers(tmp_path):
    store = Store(tmp_path / "db", lambda: 100)
    for repo, number, reason in (
        ("acme/widgets", 1, "review repair round limit reached"),
        ("acme/widgets", 2, "operator paused this PR"),
        ("acme/other", 3, "review repair round limit reached"),
    ):
        store.enqueue(repo, number, {"stage": "watch", "pause_reason": reason})
        store.save(store.claim(), "paused")

    assert store.recover_review_limit_pauses(("acme/widgets",)) == 1
    job = store.claim()
    assert job["issue"] == 1 and "pause_reason" not in job["data"]
    assert store.claim() is None


def test_only_pull_request_conversation_webhooks_enter_review_inbox(tmp_path):
    repo = "acme/widgets"
    store = Store(tmp_path / "db")
    app = TestClient(create_app(store, b"secret", {repo}))
    for number, issue in enumerate(({"number": 1}, {"number": 2, "pull_request": {"url": "pr"}})):
        body = json.dumps({"repository": {"full_name": repo}, "issue": issue}).encode()
        signature = hmac.new(b"secret", body, hashlib.sha256).hexdigest()
        response = app.post("/webhook", content=body, headers={
            "x-github-event": "issue_comment",
            "x-github-delivery": str(number),
            "x-hub-signature-256": "sha256=" + signature,
        })
        assert response.status_code == 202
    assert [event["id"] for event in store.events()] == ["1"]

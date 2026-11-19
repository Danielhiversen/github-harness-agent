import hashlib
import hmac
import json

from fastapi import FastAPI, HTTPException, Request

from .ci import webhook_ci_failure
from .store import Store


def create_app(store: Store, secret: bytes, repositories: set[str]):
    if not secret:
        raise ValueError("A webhook secret is required")
    app = FastAPI()

    @app.post("/webhook", status_code=202)
    async def webhook(request: Request):
        body = await request.body()
        expected = "sha256=" + hmac.new(secret, body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(
            expected, request.headers.get("x-hub-signature-256", "")
        ):
            raise HTTPException(401)
        event = request.headers.get("x-github-event")
        delivery = request.headers.get("x-github-delivery")
        if not delivery or not event:
            raise HTTPException(400)
        try:
            payload = json.loads(body)
            repo = payload.get("repository", {}).get("full_name")
        except (ValueError, AttributeError):
            raise HTTPException(400)
        failure = webhook_ci_failure(event, payload) if repo in repositories else None
        if failure:
            store.receive(
                delivery, event, {"repository": {"full_name": repo}}, ci_failure=failure
            )
        elif repo in repositories and event in {
            "issues",
            "pull_request",
            "pull_request_review",
            "pull_request_review_comment",
        }:
            store.receive(delivery, event, payload)
        return {"received": True}

    return app

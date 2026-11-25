import hashlib
import hmac
import json

from fastapi.testclient import TestClient

from github_harness_agent.store import Store
from github_harness_agent.webhooks import create_app


def run(
    id_,
    sha,
    name,
    *,
    conclusion="failure",
    status="completed",
    app=7,
    summary="tests failed",
):
    return {
        "id": id_,
        "head_sha": sha,
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "app": {"id": app, "slug": f"app-{app}"},
        "output": {"title": "Failure", "summary": summary},
    }


def status(id_, context, state, *, summary="Test failed"):
    return {"id": id_, "context": context, "state": state, "description": summary}


def test_ci_uses_latest_result_per_context_and_check_app_across_pages():
    from github_harness_agent.ci import read_ci_report

    head = "a" * 40

    class API:
        def rest(self, method, path, *, params):
            assert method == "GET" and path == f"/repos/a/b/commits/{head}/check-runs"
            assert params["filter"] == "all"
            pages = [
                [
                    run(8, head, "test", status="queued", conclusion=None),
                    run(
                        6,
                        head,
                        "test",
                        app=9,
                        summary="Authorization: Bearer ghp_hiddentoken "
                        "https://secret.tld/failure?key=hidden " + "x" * 3000,
                    ),
                    run(1, "b" * 40, "different commit"),
                ],
                [run(7, head, "test"), run(5, head, "obsolete", conclusion="stale")],
            ]
            return {"total_count": 5, "check_runs": pages[params["page"] - 1]}

        def paginate(self, path):
            assert path == f"/repos/a/b/commits/{head}/statuses"
            yield status(4, "ci/x", "success")
            yield status(2, "ci/x", "failure")
            yield status(6, "ci/circle", "error")
            yield status(7, "ci/pending", "pending")
            yield status(3, "ci/pending", "failure")

    failures = read_ci_report(API(), "a/b", head).failures
    assert {(item.source, item.name, item.revision) for item in failures} == {
        ("check", "test", 6),
        ("status", "ci/circle", 6),
    }
    output = next(item.summary for item in failures if item.source == "check")
    assert len(output) <= 480
    assert "hiddentoken" not in output
    assert "secret.tld" not in output


def test_ci_webhook_key_is_stable_across_reruns_but_scoped_to_commit():
    from github_harness_agent.ci import webhook_ci_failure

    sha = "a" * 40
    first = webhook_ci_failure(
        "check_run", {"action": "completed", "check_run": run(8, sha, "test")}
    )
    second = webhook_ci_failure(
        "check_run", {"action": "completed", "check_run": run(9, sha, "test")}
    )
    new_sha = webhook_ci_failure(
        "check_run", {"action": "completed", "check_run": run(9, "b" * 40, "test")}
    )
    assert first == second
    assert new_sha != first
    assert (
        webhook_ci_failure(
            "check_run",
            {
                "action": "completed",
                "check_run": run(9, sha, "test", conclusion="success"),
            },
        )
        is None
    )
    assert (
        webhook_ci_failure("status", {"sha": sha, **status(4, "circleci", "pending")})
        is None
    )


def test_ci_webhook_wakeup_only_advances_matching_unseen_watch_job(tmp_path):
    from github_harness_agent.ci import webhook_ci_failure

    sha = "a" * 40
    failure = webhook_ci_failure(
        "status", {"sha": sha, **status(1, "circleci", "failure")}
    )
    store = Store(tmp_path / "ci.db", lambda: 100)
    for pull, stage, head, seen, state in [
        (20, "watch", sha, [], "queued"),
        (21, "watch", "b" * 40, [], "queued"),
        (22, "repair", sha, [], "queued"),
        (23, "watch", sha, [failure[1]], "queued"),
        (24, "watch", sha, [], "paused"),
    ]:
        store.adopt_pull(
            "a/b",
            pull,
            {
                "stage": stage,
                "pull": pull,
                "branch": f"existing/{pull}",
                "author": "bot",
                "linked_issues": [],
                "last_commit": head,
                "ci_seen": seen,
            },
            "agent/",
        )
        job = store.claim(pull=("a/b", pull))
        store.save(job, state, delay=240)
    payload = {
        "repository": {"full_name": "a/b"},
        "sha": sha,
        **status(1, "circleci", "failure"),
    }
    content = json.dumps(payload).encode()
    headers = {
        "x-github-event": "status",
        "x-github-delivery": "one",
        "x-hub-signature-256": "sha256="
        + hmac.new(b"fixture", content, hashlib.sha256).hexdigest(),
    }
    client = TestClient(create_app(store, b"fixture", {"a/b"}))
    for _ in range(2):
        assert (
            client.post("/webhook", content=content, headers=headers).status_code == 202
        )
    assert len(store.events()) == 1
    assert json.loads(store.events()[0]["payload"]) == {
        "repository": {"full_name": "a/b"}
    }
    assert store.claim()["issue"] == 20
    assert store.claim() is None


def test_status_provider_still_reports_failures_without_checks_permission():
    from github_harness_agent.ci import read_ci_report
    from github_harness_agent.github import GitHubError

    class API:
        def rest(self, *args, **kwargs):
            raise GitHubError("checks not available", status=403)

        def paginate(self, *args, **kwargs):
            return iter([status(1, "circleci", "failure")])

    report = read_ci_report(API(), "a/b", "a" * 40)
    assert report.sources == {"status"}
    assert [failure.name for failure in report.failures] == ["circleci"]

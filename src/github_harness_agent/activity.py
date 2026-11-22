"""Model activity and operator controls for one leased pull request."""

from datetime import datetime, timezone
import json
from uuid import uuid4

from .providers import ModelGatewayClient

ACTIVITY_TABLE = """
CREATE TABLE IF NOT EXISTS model_activity(
 repo TEXT, kind TEXT, issue INTEGER, slot TEXT, token TEXT,
 role TEXT, provider TEXT, model TEXT, observed REAL,
 PRIMARY KEY(repo,kind,issue,slot))
"""


def observed_call(task, gateway, **request):
    observe = task.observe_model
    if observe is None:
        return gateway.complete(**request)
    call = uuid4().hex
    role = request["role"]
    observe(call, "start", role, None, None)
    try:
        if isinstance(gateway, ModelGatewayClient):
            request["on_attempt"] = lambda provider, model: observe(
                call, "attempt", role, provider, model
            )
        result = gateway.complete(**request)
    except BaseException:
        observe(call, "end", role, None, None)
        raise
    observe(
        call, "success", role, result.route, result.model or result.configured_model
    )
    return result


def _iso(value):
    return (
        datetime.fromtimestamp(value, timezone.utc).isoformat()
        if value is not None
        else None
    )


def _select(db, repo, pull):
    query = "SELECT * FROM tasks"
    params = []
    if repo is not None:
        query += " WHERE repo=?"
        params.append(repo)
    if pull is not None:
        query += """ AND ((kind='pull' AND issue=?) OR
          (kind='issue' AND json_extract(data,'$.pull')=? AND state!='superseded'))"""
        params.extend([pull, pull])
    rows = db.execute(query + " ORDER BY repo, kind, issue", params).fetchall()
    if pull is not None and any(row["kind"] == "pull" for row in rows):
        return [row for row in rows if row["kind"] == "pull"]
    return rows


def status(store, repo=None, pull=None):
    with store.tx() as db:
        for row in _select(db, repo, pull):
            data = json.loads(row["data"])
            key = (row["repo"], row["kind"], row["issue"])
            models = db.execute(
                """SELECT * FROM model_activity
                WHERE repo=? AND kind=? AND issue=? AND
                    (slot='last_success' OR (token=? AND ?>?)) ORDER BY observed""",
                (*key, row["token"], row["expires"], store.clock()),
            ).fetchall()
            active = []
            success = None
            for model in models:
                detail = {k: model[k] for k in ("provider", "model", "role")}
                detail["at"] = _iso(model["observed"])
                if model["slot"] == "last_success":
                    success = detail
                else:
                    active.append(detail)
            waiting = row["state"] == "paused" or (
                row["state"] == "queued" and row["due"] > store.clock()
            )
            last_error = data.get("last_error")
            yield {
                **{
                    k: row[k]
                    for k in (
                        "repo",
                        "kind",
                        "issue",
                        "state",
                        "generation",
                        "attempts",
                    )
                },
                "pull": data.get(
                    "pull", row["issue"] if row["kind"] == "pull" else None
                ),
                "branch": data.get("branch"),
                "stage": data.get("stage"),
                "last_commit": data.get("last_commit"),
                "active_calls": active,
                "last_successful_call": success,
                "next_retry_at": _iso(row["due"]) if row["state"] == "queued" else None,
                "last_error": last_error,
                "blocker": data.get("pause_reason")
                or (last_error if waiting else None)
                or data.get("ci_diagnostic"),
                "ci_failures": [
                    {k: failure.get(k) for k in ("source", "name", "result")}
                    for failure in data.get("ci_feedback", ())
                ],
                "provider_failures": data.get("provider_failures", {}),
                "drafts": sorted(
                    (data.get("draft_checkpoints") or {}).get("drafts", {})
                ),
            }


def control(store, repo, pull, command):
    expected = {"retry": "queued", "resume": "paused", "pause": "queued"}[command]
    with store.tx() as db:
        rows = _select(db, repo, pull)
        if len(rows) != 1:
            raise ValueError("Expected a single tracked job for this pull request")
        (row,) = rows
        if row["state"] != expected:
            raise ValueError(
                f"{command} requires a {expected} job (current state {row['state']})"
            )
        data = json.loads(row["data"])
        if command == "pause":
            data["pause_reason"] = "operator paused this PR"
        if command == "resume":
            if data.get(
                "pause_reason"
            ) == "review repair round limit reached" and data.get(
                "round", 0
            ) >= data.get("allowed_rounds", 3):
                data["allowed_rounds"] = data["round"] + 1
            data.pop("pause_reason", None)
            data.pop("consecutive_failures", None)
        db.execute(
            """UPDATE tasks SET data=?,state=?,due=?,generation=generation+1,token=NULL
            WHERE repo=? AND kind=? AND issue=?""",
            (
                json.dumps(data),
                "paused" if command == "pause" else "queued",
                store.clock(),
                repo,
                row["kind"],
                row["issue"],
            ),
        )

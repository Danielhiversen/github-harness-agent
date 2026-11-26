"""Durable webhook inbox and single-host task queue."""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager

from .activity import ACTIVITY_TABLE


class LeaseLost(RuntimeError):
    pass


PROVIDER_EXHAUSTED = "provider routes exhausted"
_LEGACY_PROVIDER_PAUSE = """
state='paused' AND json_extract(data,'$.last_error')=?
AND json_type(data,'$.retry_at')='null'
AND json_extract(data,'$.pause_reason') IS NULL
AND json_extract(data,'$.prepared') IS NULL
AND COALESCE(json_extract(data,'$.consecutive_failures'),0)=0
AND COALESCE(json_extract(data,'$.round'),0)<3
AND json_extract(data,'$.stage') IN ('implement','repair')
"""


_TASK_TABLE = """
CREATE TABLE IF NOT EXISTS tasks(
    repo TEXT, issue INTEGER, state TEXT, generation INTEGER,
    token TEXT, expires REAL, due REAL, data TEXT, attempts INTEGER,
    kind TEXT NOT NULL DEFAULT 'issue' CHECK(kind IN ('issue', 'pull')),
    PRIMARY KEY(repo,kind,issue));
"""


class Store:
    def __init__(self, path, clock=time.time):
        self.path = str(path)
        self.clock = clock
        with self.tx() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS deliveries(
                    id TEXT PRIMARY KEY, payload TEXT, event TEXT, received REAL);
            """)
            db.execute(ACTIVITY_TABLE)
            columns = db.execute("PRAGMA table_info(tasks)").fetchall()
            if columns and not any(column["name"] == "kind" for column in columns):
                db.execute("ALTER TABLE tasks RENAME TO issue_tasks")
                db.execute(_TASK_TABLE)
                db.execute("INSERT INTO tasks SELECT *, 'issue' FROM issue_tasks")
                db.execute("DROP TABLE issue_tasks")
            else:
                db.execute(_TASK_TABLE)

    @contextmanager
    def tx(self):
        db = sqlite3.connect(self.path, timeout=20, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        finally:
            db.close()

    def receive(self, delivery, event, payload, *, ci_failure=None):
        with self.tx() as db:
            now = self.clock()
            inserted = bool(
                db.execute(
                    "INSERT OR IGNORE INTO deliveries VALUES(?,?,?,?)",
                    (delivery, json.dumps(payload), event, now),
                ).rowcount
            )
            if inserted and ci_failure is not None:
                sha, key = ci_failure
                db.execute(
                    """UPDATE tasks SET due=MIN(due,?) WHERE repo=? AND state='queued'
                       AND json_extract(data,'$.stage')='watch'
                       AND json_extract(data,'$.last_commit')=?
                       AND NOT EXISTS (SELECT 1 FROM json_each(tasks.data,'$.ci_seen') WHERE value=?)
                       AND NOT EXISTS (SELECT 1 FROM json_each(tasks.data,'$.ci_attempted') WHERE value=?)""",
                    (
                        now,
                        payload["repository"]["full_name"],
                        sha,
                        key,
                        key.removeprefix(f"{sha}:"),
                    ),
                )
            return inserted

    def events(self):
        with self.tx() as db:
            return db.execute(
                "SELECT * FROM deliveries WHERE payload IS NOT NULL ORDER BY received LIMIT 100"
            ).fetchall()

    def ack(self, delivery):
        with self.tx() as db:
            db.execute("UPDATE deliveries SET payload=NULL WHERE id=?", (delivery,))

    def enqueue(self, repo, issue, data):
        with self.tx() as db:
            if db.execute(
                """SELECT 1 FROM tasks WHERE repo=? AND kind='pull'
                   AND state NOT IN ('complete','cancelled')
                   AND (json_extract(data,'$.branch')=?
                     OR EXISTS (SELECT 1 FROM json_each(data,'$.linked_issues') WHERE value=?))""",
                (repo, data.get("branch"), issue),
            ).fetchone():
                return
            db.execute(
                """INSERT INTO tasks VALUES(?,?,'queued',1,NULL,0,?,?,0,'issue')
                ON CONFLICT(repo,kind,issue) DO UPDATE SET state='queued',generation=generation+1,
                data=CASE WHEN json_extract(tasks.data,'$.pull') IS NOT NULL
                     OR json_extract(tasks.data,'$.github_operations') IS NOT NULL
                     THEN tasks.data ELSE excluded.data END,
                due=excluded.due,attempts=0 WHERE state IN ('cancelled','superseded')""",
                (repo, issue, self.clock(), json.dumps(data)),
            )

    def cancel(self, repo, issue):
        with self.tx() as db:
            db.execute(
                """UPDATE tasks SET state='cancelled',token=NULL,generation=generation+1
                WHERE repo=? AND issue=? AND kind='issue' AND state!='cancelled'""",
                (repo, issue),
            )

    def adopt_pull(self, repo, pull, data, branch_prefix, *, head_sha=None):
        """Assign at most one active job to this PR's exact source branch."""
        with self.tx() as db:
            now = self.clock()
            existing = db.execute(
                "SELECT * FROM tasks WHERE repo=? AND kind='pull' AND issue=?",
                (repo, pull),
            ).fetchone()
            active_issue_jobs = db.execute(
                """SELECT issue,state,expires,due,data FROM tasks WHERE repo=? AND kind='issue'
                   AND state NOT IN ('complete','superseded')""",
                (repo,),
            )
            replacements = []
            state = "queued"
            due = now
            for job in active_issue_jobs:
                original = json.loads(job["data"])
                owned_pull = original.get("pull")
                same_branch = data["branch"] == original.get(
                    "branch", f"{branch_prefix}issue-{job['issue']}"
                )
                if owned_pull is not None and owned_pull != pull:
                    continue
                if (
                    owned_pull != pull
                    and not same_branch
                    and job["issue"] not in data["linked_issues"]
                ):
                    continue
                if job["state"] == "running" and job["expires"] > now:
                    return False
                replacements.append(job["issue"])
                if owned_pull == pull and existing is None:
                    db.execute(
                        """INSERT OR REPLACE INTO model_activity
                        SELECT repo,'pull',?,slot,token,role,provider,model,observed FROM model_activity
                        WHERE repo=? AND kind='issue' AND issue=? AND slot='last_success'""",
                        (pull, repo, job["issue"]),
                    )
                    for key in (
                        "allowed_rounds",
                        "seen",
                        "threads",
                        "round",
                        "last_commit",
                        "last_checks",
                        "last_target",
                        "last_evidence",
                        "github_operations",
                        "last_error",
                        "retry_at",
                        "provider_retry_count",
                        "provider_failures",
                        "prior_model_attempt_failures",
                        "ci_seen",
                        "ci_attempted",
                    ):
                        if key in original:
                            data[key] = original[key]
                    if original["stage"] == "reply":
                        for key in ("stage", "feedback", "review_head"):
                            data[key] = original[key]
                    if (
                        original["stage"] == "repair"
                        and head_sha
                        and original.get("review_head") == head_sha
                        and "feedback" in original
                    ):
                        for key in (
                            "stage",
                            "feedback",
                            "review_head",
                            "model_attempt_failures",
                            "ci_feedback",
                            "draft_checkpoints",
                        ):
                            if key in original:
                                data[key] = original[key]
                    if original.get("last_error") == PROVIDER_EXHAUSTED:
                        due = job["due"]
                    if job["state"] == "paused":
                        state = "paused"
                        if "pause_reason" in original:
                            data["pause_reason"] = original["pause_reason"]
            for issue in replacements:
                db.execute(
                    """UPDATE tasks SET state='superseded', token=NULL, generation=generation+1,
                       data=json_remove(data,'$.draft_checkpoints')
                       WHERE repo=? AND kind='issue' AND issue=?""",
                    (repo, issue),
                )
            if existing is not None:
                if existing["state"] == "running":
                    return False
                previous = json.loads(existing["data"])
                if (
                    previous["branch"] != data["branch"]
                    or previous["author"] != data["author"]
                ):
                    return False
                previous["linked_issues"] = data["linked_issues"]
                reopened = existing["state"] in {"complete", "cancelled"}
                if reopened:
                    previous["stage"] = "watch"
                    for key in (
                        "prepared",
                        "draft_checkpoints",
                        "model_attempt_failures",
                        "feedback",
                        "question",
                    ):
                        previous.pop(key, None)
                db.execute(
                    """UPDATE tasks SET data=?,state=?,due=?,generation=generation+?
                       WHERE repo=? AND kind='pull' AND issue=?""",
                    (
                        json.dumps(previous),
                        "queued" if reopened else existing["state"],
                        now if reopened else existing["due"],
                        int(reopened),
                        repo,
                        pull,
                    ),
                )
                return False
            db.execute(
                """INSERT INTO tasks VALUES(?,?,?,1,NULL,0,?,?,0,'pull')""",
                (repo, pull, state, due, json.dumps(data)),
            )
            return True

    def complete_missing_pulls(self, repo, open_pulls):
        with self.tx() as db:
            for row in db.execute(
                "SELECT issue FROM tasks WHERE repo=? AND kind='pull' AND state!='complete'",
                (repo,),
            ).fetchall():
                if row["issue"] not in open_pulls:
                    db.execute(
                        """UPDATE tasks SET state='complete',token=NULL,generation=generation+1
                           WHERE repo=? AND kind='pull' AND issue=?""",
                        (repo, row["issue"]),
                    )

    def recover_review_limit_pauses(self, repositories, *, pull=None):
        """Old round-limit pauses must watch for fresh review feedback again."""
        repositories = tuple(repositories)
        if not repositories:
            return 0
        selection = " AND repo IN (" + ", ".join("?" for _ in repositories) + ")"
        parameters = repositories
        if pull is not None:
            selection += " AND repo=? AND kind='pull' AND issue=?"
            parameters += pull
        with self.tx() as db:
            return db.execute(
                """UPDATE tasks SET state='queued',token=NULL,generation=generation+1,due=?,
                   data=json_remove(data,'$.pause_reason')
                   WHERE state='paused' AND json_extract(data,'$.stage')='watch'
                   AND json_extract(data,'$.pause_reason')='review repair round limit reached'"""
                + selection,
                (self.clock(), *parameters),
            ).rowcount

    def recover_provider_pauses(self, repositories, *, pull=None):
        """Recover only records paused by the old provider-exhaustion handler."""
        repositories = tuple(repositories)
        if not repositories:
            return 0
        selection = " AND repo IN (" + ", ".join("?" for _ in repositories) + ")"
        parameters = repositories
        if pull is not None:
            selection += " AND repo=? AND kind='pull' AND issue=?"
            parameters += pull
        with self.tx() as db:
            now = self.clock()
            return db.execute(
                """UPDATE tasks SET state='queued',token=NULL,generation=generation+1,due=?,
                   data=json_set(data,'$.provider_retry_count',1,'$.retry_at',?) WHERE """
                + _LEGACY_PROVIDER_PAUSE
                + selection,
                (now, now, PROVIDER_EXHAUSTED, *parameters),
            ).rowcount

    def fence(self):
        with self.tx() as db:
            db.execute(
                """UPDATE tasks SET generation=generation+1,token=NULL,state='paused',
                   data=json_set(data,'$.pause_reason','manual fence')
                   WHERE state IN ('running','queued') OR ("""
                + _LEGACY_PROVIDER_PAUSE
                + ")",
                (PROVIDER_EXHAUSTED,),
            )

    def clear_failures(self, job, *, progressed=False):
        with self.tx() as db:
            db.execute(
                """UPDATE tasks SET data=CASE WHEN ? OR state IN ('complete','shadow_complete')
                     THEN json_remove(data,'$.consecutive_failures','$.last_error','$.retry_at',
                         '$.provider_retry_count','$.provider_failures')
                     ELSE json_remove(data,'$.consecutive_failures') END
                   WHERE repo=? AND issue=? AND kind=? AND generation=?
                     AND attempts=? AND token IS NULL""",
                (
                    progressed,
                    job["repo"],
                    job["issue"],
                    job["kind"],
                    job["generation"],
                    job["attempts"] + 1,
                ),
            )

    def claim(self, ttl=900, *, pull: tuple[str, int] | None = None, repo_ttls=None):
        now = self.clock()
        selection = ""
        parameters = (now, now)
        if pull is not None:
            selection = " AND repo=? AND kind='pull' AND issue=?"
            parameters += pull
        with self.tx() as db:
            row = db.execute(
                """SELECT * FROM tasks WHERE ((state='queued' AND due<=?)
                OR (state='running' AND expires<=?))"""
                + selection
                + " ORDER BY due,repo,kind,issue LIMIT 1",
                parameters,
            ).fetchone()
            if row is None:
                return None
            db.execute(
                """DELETE FROM model_activity WHERE repo=? AND kind=? AND issue=?
                AND slot!='last_success'""",
                (row["repo"], row["kind"], row["issue"]),
            )
            token = uuid.uuid4().hex
            if repo_ttls is not None:
                ttl = repo_ttls.get(row["repo"], ttl)
            db.execute(
                """UPDATE tasks SET token=?,state='running',expires=?,attempts=attempts+1
                WHERE repo=? AND issue=? AND kind=?""",
                (token, now + ttl, row["repo"], row["issue"], row["kind"]),
            )
            return {**dict(row), "data": json.loads(row["data"]), "token": token}

    def observe_model(self, job, call, phase, role, provider, model):
        key = (job["repo"], job["kind"], job["issue"])
        with self.tx() as db:
            now = self.clock()
            if not db.execute(
                "SELECT 1 FROM tasks WHERE token=? AND expires>?", (job["token"], now)
            ).fetchone():
                raise LeaseLost()
            if phase in {"end", "success"}:
                db.execute(
                    "DELETE FROM model_activity WHERE repo=? AND kind=? AND issue=? AND slot=?",
                    (*key, call),
                )
            if phase != "end":
                slot = "last_success" if phase == "success" else call
                db.execute(
                    "INSERT OR REPLACE INTO model_activity VALUES(?,?,?,?,?,?,?,?,?)",
                    (*key, slot, job["token"], role, provider, model, now),
                )

    def check(self, job):
        with self.tx() as db:
            if not db.execute(
                "SELECT 1 FROM tasks WHERE token=? AND expires>?",
                (job["token"], self.clock()),
            ).fetchone():
                raise LeaseLost()

    def save(self, job, state="running", delay=0):
        with self.tx() as db:
            now = self.clock()
            changed = db.execute(
                """UPDATE tasks SET data=?,state=?,due=?, token=?
                WHERE token=? AND expires>?""",
                (
                    json.dumps(job["data"]),
                    state,
                    now + delay,
                    job["token"] if state == "running" else None,
                    job["token"],
                    now,
                ),
            ).rowcount
            if not changed:
                raise LeaseLost()

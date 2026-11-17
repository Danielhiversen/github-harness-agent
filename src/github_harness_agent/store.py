"""Durable webhook inbox and single-host task queue."""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager


class LeaseLost(RuntimeError):
    pass


class Store:
    def __init__(self, path, clock=time.time):
        self.path = str(path)
        self.clock = clock
        with self.tx() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS deliveries(
                    id TEXT PRIMARY KEY, payload TEXT, event TEXT, received REAL);
                CREATE TABLE IF NOT EXISTS tasks(
                    repo TEXT, issue INTEGER, state TEXT, generation INTEGER,
                    token TEXT, expires REAL, due REAL, data TEXT, attempts INTEGER,
                    PRIMARY KEY(repo,issue));
            """)

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

    def receive(self, delivery, event, payload):
        with self.tx() as db:
            return bool(
                db.execute(
                    "INSERT OR IGNORE INTO deliveries VALUES(?,?,?,?)",
                    (delivery, json.dumps(payload), event, self.clock()),
                ).rowcount
            )

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
            db.execute(
                """INSERT INTO tasks VALUES(?,?,'queued',1,NULL,0,0,?,0)
                ON CONFLICT(repo,issue) DO UPDATE SET state='queued',generation=generation+1,
                data=CASE WHEN json_extract(tasks.data,'$.pull') IS NOT NULL
                     OR json_extract(tasks.data,'$.github_operations') IS NOT NULL
                     THEN tasks.data ELSE excluded.data END,
                due=0,attempts=0 WHERE state='cancelled'""",
                (repo, issue, json.dumps(data)),
            )

    def cancel(self, repo, issue):
        with self.tx() as db:
            db.execute(
                """UPDATE tasks SET state='cancelled',token=NULL,generation=generation+1
                WHERE repo=? AND issue=? AND state!='cancelled'""",
                (repo, issue),
            )

    def claim(self, ttl=900):
        now = self.clock()
        with self.tx() as db:
            row = db.execute(
                """SELECT * FROM tasks WHERE (state='queued' AND due<=?)
                OR (state='running' AND expires<=?) ORDER BY due,repo,issue LIMIT 1""",
                (now, now),
            ).fetchone()
            if row is None:
                return None
            token = uuid.uuid4().hex
            db.execute(
                """UPDATE tasks SET token=?,state='running',expires=?,attempts=attempts+1
                WHERE repo=? AND issue=?""",
                (token, now + ttl, row["repo"], row["issue"]),
            )
            return {**dict(row), "data": json.loads(row["data"]), "token": token}

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

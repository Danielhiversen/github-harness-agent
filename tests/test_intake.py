from github_harness_agent.intake import reconcile
from github_harness_agent.store import Store


class FakeRepo:
    role = "write"
    blocked = True

    def paginate(self, path, params=None):
        if path.endswith("/issues"):
            return [
                {
                    "number": 4,
                    "state": "open",
                    "title": "fixture",
                    "labels": [{"name": "agent:ready"}],
                }
            ]
        if path.endswith("/events"):
            return [
                {
                    "event": "labeled",
                    "label": {"name": "agent:ready"},
                    "actor": {"login": "maintainer"},
                }
            ]
        return [{"state": "open" if self.blocked else "closed"}]

    def rest(self, method, path):
        return (
            {"permission": self.role}
            if path.endswith("permission")
            else {"object": {"sha": "abc"}}
        )


def test_reconciliation_recovers_missed_delivery_and_waits_for_dependencies(tmp_path):
    db, api = Store(tmp_path / "db"), FakeRepo()

    def sync():
        reconcile(api, db, "owner/repo", {"base": "main"})

    sync()
    assert db.claim() is None
    api.blocked, api.role = False, "read"
    sync()
    assert db.claim() is None
    api.role = "maintain"
    sync()
    sync()
    assert db.claim()["data"]["base_sha"] == "abc"
    assert db.claim() is None

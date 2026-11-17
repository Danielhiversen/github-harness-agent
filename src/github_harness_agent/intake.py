"""Map current GitHub state to one durable task per issue."""

from urllib.parse import quote


def maintainer(api, repo, login):
    data = api.rest(
        "GET", f"/repos/{repo}/collaborators/{quote(login, safe='')}/permission"
    )
    return data["permission"] in {"admin", "maintain", "write"}


def eligible(api, repo, issue, label):
    if issue["state"] != "open" or label not in {x["name"] for x in issue["labels"]}:
        return False
    number = issue["number"]
    events = api.paginate(f"/repos/{repo}/issues/{number}/events")
    actor = None
    for event in events:
        if event.get("label", {}).get("name") == label:
            if event["event"] == "labeled":
                actor = (event.get("actor") or {}).get("login")
            elif event["event"] == "unlabeled":
                actor = None
    if not actor or not maintainer(api, repo, actor):
        return False
    return all(
        dep["state"] == "closed"
        for dep in api.paginate(
            f"/repos/{repo}/issues/{number}/dependencies/blocked_by"
        )
    )


def reconcile(api, store, repo, config):
    label = config.get("label", "agent:ready")
    for issue in api.paginate(
        f"/repos/{repo}/issues", params={"state": "open", "labels": label}
    ):
        if "pull_request" in issue or not eligible(api, repo, issue, label):
            continue
        base = config["base"]
        sha = api.rest("GET", f"/repos/{repo}/git/ref/heads/{quote(base, safe='/')}")[
            "object"
        ]["sha"]
        store.enqueue(
            repo,
            issue["number"],
            {
                "base": base,
                "base_sha": sha,
                "title": issue["title"],
                "body": issue.get("body") or "",
                "round": 0,
                "seen": [],
                "threads": [],
                "stage": "implement",
            },
        )

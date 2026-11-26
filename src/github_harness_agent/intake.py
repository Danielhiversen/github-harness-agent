"""Map current GitHub state to durable issue and review jobs."""

import re
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import quote

from .github import GitHubError, validate_repository


_CLOSING_REFERENCE = re.compile(
    r"\b(?:clos(?:e|es|ed)|fix(?:es|ed)?|resolv(?:e|es|ed))\s*:?\s+"
    r"(?:(?P<repo>[\w.-]+/[\w.-]+))?#(?P<number>[1-9]\d*)\b",
    re.IGNORECASE,
)


def _linked_issues(body: str, repo: str) -> list[int]:
    body = re.sub(
        r"<!--.*?-->|(?ms:^[ \t]*(```+|~~~+).*?^[ \t]*\1[ \t]*$)|`[^`]*`",
        "",
        body,
        flags=re.S,
    )
    return sorted(
        {
            int(match["number"])
            for match in _CLOSING_REFERENCE.finditer(body)
            if match["repo"] is None or match["repo"].casefold() == repo.casefold()
        }
    )


def _same_repository(ref, repo):
    return (
        (ref.get("repo") or {}).get("full_name") or ""
    ).casefold() == repo.casefold()


def adoptable_pull(request, repo, bot_login, branch=None):
    head = request.get("head") or {}
    base = request.get("base") or {}
    author = (request.get("user") or {}).get("login") or ""
    return (
        request["state"] == "open"
        and author.casefold() == bot_login.casefold()
        and _same_repository(head, repo)
        and _same_repository(base, repo)
        and bool(head.get("sha"))
        and bool(base.get("sha"))
        and bool(head.get("ref"))
        and bool(base.get("ref"))
        and head["ref"] != base["ref"]
        and (branch is None or head["ref"] == branch)
    )


def adopt_open_pulls(api, store, repo, config, bot_login):
    """Adopt existing pull requests from the authenticated account after listing all pages."""
    validate_repository(repo)
    requests = list(
        api.paginate(
            f"/repos/{repo}/pulls",
            params={"state": "open", "sort": "updated", "direction": "desc"},
        )
    )
    branches = Counter(
        request["head"]["ref"]
        for request in requests
        if request["state"] == "open"
        and _same_repository(request.get("head") or {}, repo)
    )
    open_pulls = set()
    adopted = 0
    branch_prefix = config.get("branch_prefix", "agent/")
    for request in requests:
        if not adoptable_pull(request, repo, bot_login):
            continue
        head = request["head"]
        if branches[head["ref"]] != 1:
            continue
        number = request["number"]
        open_pulls.add(number)
        base = request["base"]
        body = request.get("body") or ""
        state = {
            "pull": number,
            "branch": head["ref"],
            "author": bot_login.casefold(),
            "base": base["ref"],
            "base_sha": base["sha"],
            "title": request["title"],
            "body": body,
            "linked_issues": _linked_issues(body, repo),
            "adopted_at": datetime.fromtimestamp(
                store.clock(), timezone.utc
            ).isoformat(),
            "round": 0,
            "seen": [],
            "threads": [],
            "stage": "watch",
        }
        adopted += store.adopt_pull(
            repo, number, state, branch_prefix, head_sha=head["sha"]
        )
    store.complete_missing_pulls(repo, open_pulls)
    return adopted


def maintainer(api, repo, login):
    data = api.rest(
        "GET", f"/repos/{repo}/collaborators/{quote(login, safe='')}/permission"
    )
    return data["permission"] in {"admin", "maintain", "write"}


def eligible(api, repo, issue, bot_login, label=None):
    """Assignment to the configured account is the authority to work an issue."""
    if issue["state"] != "open" or not bot_login or bot_login.casefold() not in {
        (assignee.get("login") or "").casefold()
        for assignee in issue.get("assignees", ())
    }:
        return False
    if label and label not in {x["name"] for x in issue.get("labels", ())}:
        return False
    if label:
        actor = None
        for event in api.paginate(f"/repos/{repo}/issues/{issue['number']}/events"):
            if event.get("label", {}).get("name") == label:
                if event["event"] == "labeled":
                    actor = (event.get("actor") or {}).get("login")
                elif event["event"] == "unlabeled":
                    actor = None
        if not actor or not maintainer(api, repo, actor):
            return False
    return all(
        dep["state"] == "closed"
        for dep in api.paginate(f"/repos/{repo}/issues/{issue['number']}/dependencies/blocked_by")
    )


def reconcile(api, store, repo, config):
    label = config.get("label")
    bot_login = config.get("assignee") or config.get("bot_login")
    params = {"state": "open", "assignee": bot_login}
    if label:
        params["labels"] = label
    scan = store.issue_scan_snapshot(repo)
    eligible_issues = set()
    for issue in api.paginate(
        f"/repos/{repo}/issues", params=params
    ):
        if "pull_request" in issue or not eligible(api, repo, issue, bot_login, label):
            continue
        eligible_issues.add(issue["number"])
        store.refresh_issue(repo, issue["number"], issue)
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
                "branch": config.get("branch_prefix", "agent/")
                + f"issue-{issue['number']}",
                "body": issue.get("body") or "",
                "round": 0,
                "seen": [],
                "threads": [],
                "stage": "implement",
            },
        )
    # A filtered list can lag a reassignment. Confirm missing jobs individually
    # before fencing them, and never infer cancellation from an incomplete scan.
    for number in scan.keys() - eligible_issues:
        try:
            current = api.rest("GET", f"/repos/{repo}/issues/{number}")
        except GitHubError as error:
            if error.status != 404:
                raise
        else:
            if eligible(api, repo, current, bot_login, label):
                eligible_issues.add(number)
                store.refresh_issue(repo, number, current)
    store.cancel_missing_issues(repo, eligible_issues, scan)

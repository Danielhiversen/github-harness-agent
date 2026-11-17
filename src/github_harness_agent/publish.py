"""Recoverable GitHub writes for the verified issue and review commits."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from .github import GitHubError, validate_repository


class StaleHead(RuntimeError):
    pass


class ClosedPull(RuntimeError):
    pass


class Journal:
    """Persist a write-ahead record in the current fenced job snapshot."""

    def __init__(self, data: dict, checkpoint: Callable[[], None]):
        self._entries = data.setdefault("github_operations", {})
        self._checkpoint = checkpoint

    def get(self, key: str) -> dict | None:
        return self._entries.get(key)

    def begin(self, key: str, intent: dict) -> None:
        previous = self._entries.get(key)
        if previous is None:
            self._entries[key] = {"intent": intent}
            self._checkpoint()
        elif previous["intent"] != intent:
            raise RuntimeError("GitHub write conflicts with its persisted intent")

    def finish(self, key: str, result: dict) -> None:
        entry = self._entries[key]
        if entry.get("result") != result:
            entry["result"] = result
            self._checkpoint()


@dataclass(frozen=True)
class BranchUpdate:
    repo: str
    branch: str
    expected_head: str
    checkout: Path
    commit: str
    pull: int | None = None


@dataclass(frozen=True)
class PublishRequest:
    repo: str
    issue: int
    base: str
    base_sha: str
    branch: str
    checkout: Path
    commit: str
    title: str
    summary: str
    checks: tuple[str, ...]
    limitations: tuple[str, ...] = ()
    ready: bool = False


@dataclass(frozen=True)
class PublishedPull:
    number: int
    url: str
    head: str
    draft: bool


def _ready_query() -> str:
    return """
    mutation Ready($id: ID!) {
      markPullRequestReadyForReview(input: {pullRequestId: $id}) {
        pullRequest { number isDraft headRefOid state }
      }
    }
    """


class Publisher:
    def __init__(self, github, journal: Journal, authorize: Callable[[], None]):
        self.github = github
        self.journal = journal
        self.authorize = authorize

    def _head(self, repo: str, branch: str) -> str | None:
        path = f"/repos/{validate_repository(repo)}/git/ref/heads/{quote(branch, safe='/')}"
        try:
            remote = self.github.rest("GET", path)
        except GitHubError as error:
            if error.status == 404:
                return None
            raise
        return remote["object"]["sha"]

    def _ensure_branch(self, request: PublishRequest) -> None:
        key = f"ref:{request.repo}:{request.branch}"
        entry = self.journal.get(key)
        intent = {"ref": f"refs/heads/{request.branch}", "sha": request.base_sha}
        head = self._head(request.repo, request.branch)
        if head is not None:
            if entry is None:
                raise StaleHead(
                    "issue branch already exists without a matching journal"
                )
            self.journal.begin(key, intent)
            self.journal.finish(key, {"sha": request.base_sha})
            return
        if entry and entry.get("result"):
            raise StaleHead("issue branch was deleted")
        self.journal.begin(key, intent)
        self.authorize()
        try:
            self.github.rest("POST", f"/repos/{request.repo}/git/refs", json=intent)
        except GitHubError:
            if self._head(request.repo, request.branch) != request.base_sha:
                raise
        self.journal.finish(key, {"sha": request.base_sha})

    def advance(self, update: BranchUpdate) -> str:
        self.authorize()
        repo = validate_repository(update.repo)
        key = f"push:{repo}:{update.branch}:{update.commit}"
        head = self._head(repo, update.branch)
        if update.pull is not None:
            pull = self.github.rest("GET", f"/repos/{repo}/pulls/{update.pull}")
            if pull["state"] != "open":
                raise ClosedPull("pull request is closed")
            if pull["head"]["ref"] != update.branch or pull["head"]["sha"] != head:
                raise StaleHead("pull request head changed")
            if pull["head"]["repo"]["full_name"].casefold() != repo.casefold():
                raise StaleHead("pull request head repository changed")
        entry = self.journal.get(key)
        if head == update.commit and entry is not None:
            self.journal.finish(key, {"sha": update.commit})
            return update.commit
        if head != update.expected_head or (entry and entry.get("result")):
            raise StaleHead("branch no longer has the verified expected head")
        self.journal.begin(
            key, {"expected": update.expected_head, "sha": update.commit}
        )
        self.authorize()
        try:
            self.github.push(
                repo,
                update.checkout,
                update.branch,
                update.expected_head,
                update.commit,
            )
        except GitHubError:
            head = self._head(repo, update.branch)
            if head != update.commit:
                if head != update.expected_head:
                    raise StaleHead("branch changed during the push") from None
                raise
        self.journal.finish(key, {"sha": update.commit})
        return update.commit

    def publish(self, request: PublishRequest) -> PublishedPull:
        self.authorize()
        repo = validate_repository(request.repo)
        if request.issue <= 0 or request.branch == request.base:
            raise ValueError("an issue pull request requires a separate branch")
        if request.ready and not request.checks:
            raise ValueError("a ready pull request requires verification evidence")
        self._ensure_branch(request)
        self.advance(
            BranchUpdate(
                repo,
                request.branch,
                request.base_sha,
                request.checkout,
                request.commit,
            )
        )
        pull = self._ensure_pull(request)
        if pull["state"] != "open":
            raise ClosedPull("pull request is closed")
        if pull["head"]["sha"] != request.commit:
            raise StaleHead("pull request does not point to the verified commit")
        ready = self.journal.get(f"ready:{repo}:{pull['number']}")
        if request.ready and (pull["draft"] or (ready and "result" not in ready)):
            pull = self._mark_ready(request, pull)
        return PublishedPull(
            pull["number"], pull["html_url"], request.commit, pull["draft"]
        )

    def _ensure_pull(self, request: PublishRequest) -> dict:
        owner = request.repo.split("/", 1)[0]
        key = f"pull:{request.repo}:{request.branch}:{request.base}"
        path = f"/repos/{request.repo}/pulls"
        params = {
            "state": "all",
            "head": f"{owner}:{request.branch}",
            "base": request.base,
        }

        def lookup():
            matches = list(self.github.paginate(path, params=params))
            if len(matches) > 1:
                raise RuntimeError(
                    "multiple pull requests match the issue branch and base"
                )
            return matches[0] if matches else None

        pull = lookup()
        entry = self.journal.get(key)
        if pull is not None:
            if entry is None or (
                entry.get("result") and entry["result"]["number"] != pull["number"]
            ):
                raise RuntimeError(
                    "matching pull request exists without matching publish intent"
                )
        else:
            if entry and entry.get("result"):
                raise ClosedPull("previously published pull request is unavailable")
            body = f"{request.summary.strip()}\n\nCloses #{request.issue}"
            if request.checks:
                body += "\n\n## Checks\n\n" + "\n".join(
                    f"- {check}" for check in request.checks
                )
            if request.limitations:
                body += "\n\n## Limitations\n\n" + "\n".join(
                    f"- {limitation}" for limitation in request.limitations
                )
            intent = {
                "title": request.title,
                "body": body,
                "head": request.branch,
                "base": request.base,
                "draft": True,
            }
            self.journal.begin(key, intent)
            self.authorize()
            if self._head(request.repo, request.branch) != request.commit:
                raise StaleHead("branch changed before pull request creation")
            try:
                pull = self.github.rest("POST", path, json=intent)
            except GitHubError:
                pull = lookup()
                if pull is None:
                    raise
        self.journal.finish(key, {"number": pull["number"]})
        return pull

    def _mark_ready(self, request: PublishRequest, pull: dict) -> dict:
        number = pull["number"]
        key = f"ready:{request.repo}:{number}"
        path = f"/repos/{request.repo}/pulls/{number}"
        self.journal.begin(key, {"node_id": pull["node_id"], "sha": request.commit})
        self.authorize()
        current = self.github.rest("GET", path)
        if current["state"] != "open":
            raise ClosedPull("pull request is closed")
        if current["head"]["sha"] != request.commit:
            raise StaleHead("pull request head changed before marking ready")
        if current["draft"]:
            try:
                self.github.graphql(_ready_query(), {"id": pull["node_id"]})
            except GitHubError:
                current = self.github.rest("GET", path)
                if current["draft"]:
                    raise
            else:
                current = self.github.rest("GET", path)
        if current["state"] != "open":
            raise ClosedPull("pull request closed while marking ready")
        if current["head"]["sha"] != request.commit:
            raise StaleHead("pull request head changed while marking ready")
        if current["draft"]:
            raise GitHubError("pull request readiness could not be confirmed")
        self.journal.finish(key, {"number": number})
        return current

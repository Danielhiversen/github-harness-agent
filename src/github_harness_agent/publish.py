"""Recoverable GitHub writes for the verified issue and review commits."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from .github import GitHubError, validate_repository
from .rebase import TargetBase


class StaleHead(RuntimeError):
    pass


class TargetMoved(RuntimeError):
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

    def replace_pending(self, key: str, intent: dict) -> None:
        entry = self._entries[key]
        if "result" in entry:
            raise RuntimeError("completed GitHub write cannot be replaced")
        entry["intent"] = intent
        self._checkpoint()


@dataclass(frozen=True)
class BranchUpdate:
    repo: str
    branch: str
    expected_head: str
    checkout: Path
    commit: str
    pull: int | None = None
    target: TargetBase | None = None
    verified_parent: str | None = None


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
    target: TargetBase | None = None


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

    def branch_head(self, repo: str, branch: str) -> str | None:
        return self._head(repo, branch)

    def target(self, repo: str, ref: str, pull: int | None = None) -> TargetBase:
        if pull is not None:
            request = self.github.rest(
                "GET", f"/repos/{validate_repository(repo)}/pulls/{pull}"
            )
            if request["state"] != "open":
                raise ClosedPull("pull request is closed")
            base = request["base"]
            ref = base["ref"]
            repo = (base.get("repo") or {}).get("full_name", repo)
        sha = self._head(repo, ref)
        if sha is None:
            raise TargetMoved("target ref is unavailable")
        return TargetBase(repo, ref, sha)

    def _check_target(
        self, target: TargetBase | None, repo: str, pull: int | None = None
    ) -> None:
        if target is not None and self.target(repo, target.ref, pull) != target:
            raise TargetMoved("target ref changed before the GitHub write")

    def _ensure_branch(self, request: PublishRequest) -> str:
        key = f"ref:{request.repo}:{request.branch}"
        entry = self.journal.get(key)
        intent = {"ref": f"refs/heads/{request.branch}", "sha": request.base_sha}
        head = self._head(request.repo, request.branch)
        if head is not None:
            if entry is None:
                raise StaleHead(
                    "issue branch already exists without a matching journal"
                )
            original = entry["intent"]["sha"]
            self.journal.finish(key, {"sha": original})
            if head != original:
                # A revised issue can reuse a branch pushed before a crash, but only
                # when that exact commit has this job's durable push intent.
                pushed = self.journal.get(f"push:{request.repo}:{request.branch}:{head}")
                if pushed and pushed.get("intent", {}).get("sha") == head:
                    return head
            return original
        if entry and entry.get("result"):
            raise StaleHead("issue branch was deleted")
        if entry and entry["intent"] != intent and request.target is not None:
            self.journal.replace_pending(key, intent)
        else:
            self.journal.begin(key, intent)
        self.authorize()
        self._check_target(request.target, request.repo)
        try:
            self.github.rest("POST", f"/repos/{request.repo}/git/refs", json=intent)
        except GitHubError:
            if self._head(request.repo, request.branch) != request.base_sha:
                raise
        self.journal.finish(key, {"sha": request.base_sha})
        return request.base_sha

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
        intent = {"expected": update.expected_head, "sha": update.commit}
        kwargs = {}
        if update.target is not None:
            intent["base"] = update.target.sha
            kwargs = {
                "verified_parent": update.verified_parent,
                "target_base": update.target.sha,
                "before_write": lambda: self._check_target(
                    update.target, repo, update.pull
                ),
            }
        self.journal.begin(key, intent)
        self.authorize()
        try:
            self.github.push(
                repo,
                update.checkout,
                update.branch,
                update.expected_head,
                update.commit,
                **kwargs,
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
        expected_head = self._ensure_branch(request)
        self.advance(
            BranchUpdate(
                repo,
                request.branch,
                expected_head,
                request.checkout,
                request.commit,
                target=request.target,
                verified_parent=request.base_sha,
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
            if entry and entry["intent"] != intent:
                self.journal.replace_pending(key, intent)
            else:
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
        intent = {"node_id": pull["node_id"], "sha": request.commit}
        entry = self.journal.get(key)
        if entry and "result" not in entry and entry["intent"] != intent:
            self.journal.replace_pending(key, intent)
        else:
            self.journal.begin(key, intent)
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

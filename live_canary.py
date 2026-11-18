"""One-shot qualification, restricted to the labeled public fixture."""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from github_harness_agent.controller import Controller
from github_harness_agent.executor import PlainExecutor, DockerRunner, Limits
from github_harness_agent.github import GitHubApp, GitHubClient
from github_harness_agent.intake import reconcile
from github_harness_agent.providers import ProviderRouter, load_manifest, ProviderExhausted
from github_harness_agent.store import Store


class ActionsInstallation:
    # The runner already holds a short-lived installation token issued by GitHub.
    _clock = staticmethod(lambda: datetime.now(timezone.utc))
    _request = GitHubApp._request
    _api_url = "https://api.github.com"
    _http = httpx.Client(timeout=20, trust_env=False)


class InstallationClient(GitHubClient):
    def _access_token(self):
        return os.environ["CANARY_INSTALLATION_TOKEN"]


repo = "Danielhiversen/github-harness-agent"
assert os.environ["GITHUB_REPOSITORY"] == repo
api = InstallationClient(ActionsInstallation(), 0)
db = Store(Path(os.environ["RUNNER_TEMP"])/"canary.db")
check = "from pathlib import Path; s=Path('docs/queue-status.md').read_text(); assert 'uv run github-harness-agent --config agent.toml status' in s"
config = dict(label="agent:live-canary", base="dhh/implement-github-agent", mode=os.environ["CANARY_MODE"],
              branch_prefix="canary/", checks=[["python", "-c", check]])
router = ProviderRouter(load_manifest(), [dict(model="kilo-free/kilo-auto/free",
    url="https://ai-gateway.kilo.ai/api/v1", headers={"Authorization":"Bearer anonymous"},
    free_verified=True,tools=True)])
executor = PlainExecutor(DockerRunner("canary-worker",runtime="runc"),router,Limits(wall_time=300,model_requests=40))
controller = Controller(db,api,executor,config,"github-actions[bot]")
reconcile(api,db,repo,config)
deadline = time.monotonic()+900
attempts=0
while time.monotonic()<deadline:
    job=db.claim(ttl=600)
    if not job:
        time.sleep(4)
        continue
    assert job["issue"]==2
    try:
        controller.run(job)
    except ProviderExhausted as err:
        print("Provider failure",err.failures,flush=True)
        attempts+=1
        if attempts>=4: raise
        db.save(job,"queued",delay=10)
        continue
    d=job["data"]
    if config["mode"]=="shadow":
        print("Verified shadow patch",d["prepared"]["digest"],d["prepared"]["checks"],d["prepared"]["trace"],flush=True)
        break
    print(json.dumps({k:d.get(k) for k in ("stage","pull","round","last_commit","last_checks")}),flush=True)
    print("Route evidence:",json.dumps(d.get("last_evidence",{}).get("trace",[])),flush=True)
    if d["round"] and d["stage"]=="watch" and d["seen"]:
        break
else:
    raise RuntimeError("Review repair timed out")

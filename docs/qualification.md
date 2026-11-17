# Qualifying the replacement

Start with a disposable repository and GitHub App installation. Exercise issue → verified PR → submitted review → repaired commit, human push during execution, app revocation, duplicate webhooks, and a service restart after each external write. Verify one PR and preserved human commits through the GitHub API. Run the opt-in worker isolation test on the deployment host. Keep the incumbent in service throughout qualification.

## Reproduce a matched replay

Prepare an evaluation directory outside workers. Select at least twenty completed issues and recover the issue text, repository instructions and base SHA as they existed when each issue became ready. Separate hidden checks, later reviews and the eventual PR patch from the model context. Pin identical allowed provider deployments, model budgets, container images, timeout and seed. Archive retries and fallback metadata from the gateway. Each trusted adapter receives the same JSON request on stdin and emits a verified unified patch to stdout. The incumbent wrapper must run a nonpublishing task in its own isolated checkout. Keep all credentials and host paths outside published evaluation results.

Use a JSON spec following this structure, with absolute paths to executables available to the evaluation controller. The example includes one case to illustrate the schema.

```json
{
  "timeout_seconds": 1000,
  "commands": {
    "incumbent": ["/opt/replay/incumbent-adapter"],
    "plain": ["/opt/agent/.venv/bin/python", "-m", "github_harness_agent.replay_worker", "plain"],
    "harness2": ["/opt/agent/.venv/bin/python", "-m", "github_harness_agent.replay_worker", "harness2"]
  },
  "issues": [{
    "id": "case-01",
    "input": {
      "repository": "example/project",
      "checkout": "/opt/replay/frozen-project",
      "base_sha": "FULL_COMMIT_SHA",
      "instructions": "Original issue and repository instructions",
      "worker_image": "project-replay-image",
      "checks": [["python", "-m", "unittest", "discover"]],
      "seed": 42,
      "limits": {"model_requests": 40, "wall_time": 900}
    }
  }]
}
```

Run the commands from the installed controller environment.

```sh
uv run python -m github_harness_agent.qualify replay results --spec replay.json
uv run python -m github_harness_agent.qualify report results
```

Between commands, give evaluators the anonymized patches and scorecard. Keep the ledger private until scoring ends. Run the held-out checks for each patch in a disposable evaluation environment, measure actual human correction minutes, and record acceptance, review rounds, quota and fallback. Include failed and timed-out runs. Report uses the complete matched sample to compare acceptance and median correction time against the incumbent. Review per-issue differences, quota and latency and extend the sample if results are inconclusive.

## Live gates

1. Record a passing basic exchange per enabled route and a tool round trip for each coding route. Record retired models, eligibility and quota behavior. A declared free price in a manifest cannot establish billing eligibility.
2. Observe ten live shadow issues, then assign ten exclusive canary issues. Include a successful review repair, a provider cooldown, a restart and rollback rehearsal. Record duplicate writes, lost human commits, unauthorized writes and paid fallback counts. Any nonzero count blocks cutover.
3. Compare quality, reviewer time, latency and quota. Select the plain or Harness2 backend per repository. Require maintainers to approve the evidence and routing change before transferring new issues. Finish existing PRs with their original owner.

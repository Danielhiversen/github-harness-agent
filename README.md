# github-harness-agent
Python GitHub App for taking a maintainer-labeled issue through a verified change, pull request and review repairs. Start in shadow mode. A single-host SQLite controller tracks task ownership and recovers pending GitHub writes. It can run a plain coding agent or one step of sequential Harness2 refinement.

## Run locally

1. Install Python 3.12 and uv. Run `uv sync --locked` and `uv run pytest -q`. The locked environment includes the exact upstream Harness2 revision.
2. Prepare a Linux worker with Docker, the runsc runtime and a trusted image containing the target project's test dependencies. Follow the [worker guide](docs/workers.md).
3. Set up the [model gateway](docs/model-gateway.md). Run a live smoke test on every route enabled for the pilot. Use only account/model combinations with confirmed free access.
4. Create a GitHub App and install it on one test repository. Grant repository metadata read, issues read/write, contents read/write and pull requests read/write. Subscribe to Issues, Pull request, Pull request review and Pull request review comment events. Configure its webhook URL to reach the controller's HTTPS reverse proxy at `/webhook`.
5. Copy `agent.example.toml` to `agent.toml`. Supply your repository name, installation ID, App bot login, branch convention, trusted worker image and exact check argv lists. Set `GITHUB_APP_ID`, `GITHUB_APP_PRIVATE_KEY_FILE`, `GITHUB_WEBHOOK_SECRET`, `MODEL_GATEWAY_URL` and `MODEL_GATEWAY_TOKEN` in the appropriate services. Restrict access to the database and its directory because it contains issue text, patches and review history.
6. Run the two commands below as separate processes from the same directory. Apply the configured label to a test issue from a maintainer account. Set issue dependencies using GitHub's native blocked-by relationship. Cancellation and publication recheck current issue eligibility. The polling fallback also picks up missed webhook events.

```sh
uv run github-harness-agent --config agent.toml serve
uv run github-harness-agent --config agent.toml work
```

Use `status` to inspect jobs and `once` to process a single iteration. Shadow mode saves the patch and evidence in the database without a GitHub write. For a canary, set the mode to `canary`, select a separate pilot label and database, and restart both processes. The incumbent and replacement must have exclusive ownership of their respective issues. A verified patch becomes a draft PR and passes to review after local required checks succeed. Branch protection and merge approval remain with the repository.

The controller revisits submitted review bodies and unresolved threads, checks the reviewer and current head, and records replies. It pauses after three repair rounds, persistent failure or exhausted routes without a retry time. Inspect paused jobs before intervention. To roll back, stop the worker, run `uv run github-harness-agent --config agent.toml fence`, reconcile each open branch and PR, then assign outstanding issues to one owner before restoring incumbent intake.

Before creating an issue branch, the controller fetches the current target branch. It rebases existing task commits against the PR's current target before editing and refreshes the target again before pushing. If the target advanced during execution, the controller rebases the patch and reruns the required checks. A conflicting rebase pauses the task. Pushing with an expected branch head prevents a concurrent contributor's commit from being overwritten.

## Use a dedicated GitHub account

The worker can use an existing GitHub CLI login for a dedicated account with access to the configured repositories. Set the top-level `github_auth` to `"named-account"` and `bot_login` to the account's GitHub login. `installation_id` is required only in App mode. Set `GH_CONFIG_DIR` in the controller service to the dedicated, existing GitHub CLI configuration directory. Place `gh` on the service's `PATH` or set `github_cli` to its absolute executable path. The worker selects the configured account on `github.com` and checks `/user` before polling or leasing work. Set the repository's `worker_runtime` to the Docker runtime installed on the worker host if it differs from `runsc`. [GitHub CLI account selection](https://cli.github.com/manual/gh_auth_token)

Run `work` for polling. For signed webhooks, separately configure an authorized repository webhook and run `serve` with `GITHUB_WEBHOOK_SECRET`. The named-account credential stays in controller process memory after startup. Restart the controller after replacing the GitHub CLI credential.

## Qualify a replacement

Use the [qualification guide](docs/qualification.md) and [implementation plan](docs/superpowers/plans/2026-11-17-github-agent.md) for matched replay, shadow and canary gates. Offline tests exercise recovery, provider contracts and both executors. Live GitHub, provider, isolation and reviewer-time evidence are additional cutover requirements.

# github-harness-agent
Python GitHub App for taking a maintainer-labeled issue through a verified change, pull request and review repairs. Start in shadow mode. A single-host SQLite controller tracks task ownership and recovers pending GitHub writes. It can run a plain coding agent or one step of sequential Harness2 refinement.

## Run locally

1. Install Python 3.12 and uv. Run `uv sync --locked` and `uv run pytest -q`. The locked environment includes the exact upstream Harness2 revision.
2. Prepare a Linux worker with Docker, the runsc runtime and a trusted image containing the target project's test dependencies. Follow the [worker guide](docs/workers.md).
3. Set up the [model gateway](docs/model-gateway.md). Run a live smoke test on every route enabled for the pilot. Use only account/model combinations with confirmed free access.
4. Create a GitHub App and install it on one test repository. Grant repository metadata, checks and commit statuses read access, plus issues, contents and pull requests read/write access. Subscribe to Issues, Pull request, Pull request review, Pull request review comment, Check run and Status events. Configure its webhook URL to reach the controller's HTTPS reverse proxy at `/webhook`.
5. Copy `agent.example.toml` to `agent.toml`. Supply your repository name, installation ID, App bot login, branch convention, trusted worker image and exact check argv lists. Set `GITHUB_APP_ID`, `GITHUB_APP_PRIVATE_KEY_FILE`, `GITHUB_WEBHOOK_SECRET`, `MODEL_GATEWAY_URL` and `MODEL_GATEWAY_TOKEN` in the appropriate services. Restrict access to the database and its directory because it contains issue text, patches and review history.
6. Run the two commands below as separate processes from the same directory. Apply the configured label to a test issue from a maintainer account. Set issue dependencies using GitHub's native blocked-by relationship. Cancellation and publication recheck current issue eligibility. The polling fallback also picks up missed webhook events.

```sh
uv run github-harness-agent --config agent.toml serve
uv run github-harness-agent --config agent.toml work
```

Use `status` to inspect jobs and `once` to process a single iteration. Shadow mode saves the patch and evidence in the database without a GitHub write. For a canary, set the mode to `canary`, select a separate pilot label and database, and restart both processes. The incumbent and replacement must have exclusive ownership of their respective issues. A verified patch becomes a draft PR and passes to review after local required checks succeed. Branch protection and merge approval remain with the repository.

Set `model_requests` and `commands` in a repository's configuration to cap model calls and agent tool commands per coding attempt. They default to 40 and 64 and accept integers from 1 to 256 and 1 to 512, respectively.

Set `verification_image` to run required checks in a separate trusted image with the same worker runtime. The verification image needs Git, tar, and the tools and fixtures used by the required checks. The model's tool workspace uses `worker_image`.

With an explicit `verification_image`, set `rollout_requests` to cap model turns in each rollout. It accepts integers from 1 through `model_requests`. The executor stages the current patch at the cap and qualifies it through the usual required checks. Harness2 proposals, composition, and all rollouts still share `model_requests`.

To process only an existing adopted PR job, run `uv run github-harness-agent --config agent.toml once --repo example/project --pull 20`. The job must be due or have an expired lease.

The controller revisits submitted review bodies and unresolved threads, checks the reviewer and current head, and records replies. Provider failures advance to the next eligible free provider. When all routes fail, the job stays queued for a later attempt with backoff and respects provider cooldowns. On startup the worker recovers jobs that older releases paused for provider exhaustion. Other persistent failures and the three-round review limit can still pause a job. Inspect paused jobs before intervention. To roll back, stop the worker, run `uv run github-harness-agent --config agent.toml fence`, reconcile each open branch and PR, then assign outstanding issues to one owner before restoring incumbent intake.

### Monitor or resume a pull request

```sh
uv run github-harness-agent --config agent.toml status --repo example/project --pull 20
uv run github-harness-agent --config agent.toml retry --repo example/project --pull 20
uv run github-harness-agent --config agent.toml pause --repo example/project --pull 20
uv run github-harness-agent --config agent.toml resume --repo example/project --pull 20
```

Status includes the active model calls, last successful model and timestamp, next scheduled poll or retry, blocker, and saved draft names. A call can briefly show an empty provider while the gateway selects a route. Retry advances a queued job to the next worker cycle. Pause applies to a queued job. Resume releases an individual paused job, allowing one more repair round when its limit was reached. These controls preserve the saved draft, branch checks and active worker lease.

The worker saves bounded patches after editing commands. A replacement worker restores the matching draft, including its last command results, and checks it in a fresh workspace before publication. Harness2 keeps each candidate's draft separately. If the selected refinement of a recovered draft fails verification, the worker can verify the completed base candidate against the same checks before publishing. Reviewers can supply long threads and CI diagnostics. The initial prompt contains a bounded excerpt and the worker can request individual pages when it needs more detail.

After a push the controller checks both GitHub Check Runs and individual commit statuses, including CircleCI reports. A new failure on the agent's recorded PR head can schedule a repair without a human review. The controller verifies that the failure and branch head are still current before pushing. Repeated failures are bounded by the repair policy.

Before creating an issue branch, the controller fetches the current target branch. It rebases existing task commits against the PR's current target before editing and refreshes the target again before pushing. If the target advanced during execution, the controller rebases the patch and reruns the required checks. A conflicting rebase pauses the task. Pushing with an expected branch head prevents a concurrent contributor's commit from being overwritten.

## Use a dedicated GitHub account

The worker can use an existing GitHub CLI login for a dedicated account with access to the configured repositories. Set the top-level `github_auth` to `"named-account"` and `bot_login` to the account's GitHub login. `installation_id` is required only in App mode. Set `GH_CONFIG_DIR` in the controller service to the dedicated, existing GitHub CLI configuration directory. Place `gh` on the service's `PATH` or set `github_cli` to its absolute executable path. The worker selects the configured account on `github.com` and checks `/user` before polling or leasing work. Set the repository's `worker_runtime` to the Docker runtime installed on the worker host if it differs from `runsc`. [GitHub CLI account selection](https://cli.github.com/manual/gh_auth_token)

Run `work` for polling. For signed webhooks, separately configure an authorized repository webhook and run `serve` with `GITHUB_WEBHOOK_SECRET`. The named-account credential stays in controller process memory after startup. Restart the controller after replacing the GitHub CLI credential.

In named-account mode, each scan also discovers open pull requests authored by that account with a head branch in the configured repository. The controller adopts each PR for review handling on its existing branch, including PRs created before deployment and PRs without a linked issue. Adoption runs before labeled issue intake. The issue label continues to select new implementation tasks. A review repair uses the PR's current target, rechecks branch ownership and head, and replies on the original PR after verification.

### Repair an outdated test in a review

The review worker reads the reported code from the current checkout. A free model can replace a unique text fragment or create a file using `edit_file`. For actionable review feedback, the worker verifies the first coding draft in a fresh sandbox. A failed check can trigger one more coding pass with the draft and failing test names. Successful drafts proceed through the branch and target checks before the agent pushes to the existing PR.

If an existing Python test enforces behavior that the review changes, a maintainer can authorize a change to that test's body. Include the selector on its own line in the submitted GitHub review body, for example:

```text
Allow test update: tests/test_training_rows.py::TestBuildDataset::test_saved_run
```

The selector can also name a module-level test. The permission applies to the named test on the reviewed commit of an adopted PR. The verifier checks the scope of the amendment and runs the configured checks.

When new coverage is part of the requested fix, say so explicitly in the review, for example `Add regression tests for both boundaries.` The worker asks the model for a new test module and checks for it before publishing. For a configured pytest launcher, it runs the requested tests separately and checks the pytest report. A rejected draft goes back to the model with the reason for rejection.

During review repair, the worker snapshots the draft before running a recognized check. If the check creates, overwrites, or recreates a deleted file, the worker preserves the pre-check draft for that path. A subsequent edit by the model can still be committed. This tracking covers configured commands and common test runners invoked directly or through supported wrappers.

## Qualify a replacement

Use the [qualification guide](docs/qualification.md) and [implementation plan](docs/superpowers/plans/2026-11-17-github-agent.md) for matched replay, shadow and canary gates. Offline tests exercise recovery, provider contracts and both executors. Live GitHub, provider, isolation and reviewer-time evidence are additional cutover requirements.

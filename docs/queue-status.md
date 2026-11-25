# Agent queue status

Run the following command to inspect the live agent queue:

```sh
uv run github-harness-agent --config agent.toml status
```

The status command is read-only and can run while the worker is active. The operator sees the current state of queued and in-progress agent runs for the configured repository.

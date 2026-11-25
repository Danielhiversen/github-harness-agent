# Model gateway

Run the gateway on the host that holds the model credentials. Configure executor clients with `MODEL_GATEWAY_URL` and the shared `MODEL_GATEWAY_TOKEN`. Use a private network or a TLS reverse proxy when executors connect from another host.

Set `MODEL_GATEWAY_DEPLOYMENTS` on the gateway to the path of a private JSON file. Store the file outside the checkout and use owner-only permissions. List the model deployments that the gateway may use. For example, a local Ollama deployment can use this configuration.

```json
[
  {
    "model": "ollama-local/qwen2.5-coder:7b",
    "url": "http://127.0.0.1:11434/v1",
    "headers": {},
    "free_verified": true,
    "tools": false
  }
]
```

Use the model IDs in `provider_manifest.v1.json`. For a cloud route, set `url` to the API base and set `headers` to the provider's authentication headers. The gateway appends `/chat/completions` to OpenAI-compatible bases. For the `google` alias, use the Gemini API version base, such as `https://generativelanguage.googleapis.com/v1beta`, with an `x-goog-api-key` header. It appends the model-specific `generateContent` endpoint.

Set `free_verified` to true after verifying zero-cost model access for the exact endpoint and account. This includes confirming trial credits or free-plan eligibility where applicable. Set `tools` to true for a coding candidate and confirm the tool round trip with the smoke command. For a verified Kilo endpoint that permits requests without credentials, explicitly set `allow_keyless` to true and use empty `headers`. Cloud routes otherwise require their provider credentials. Explicitly configured eligible models are the only candidates for a request.

Generate a deployment template outside the checkout with one row per model. Use `--profile superagent` for the 11 active Super Agent models across Kilo, Ollama and NVIDIA. The default `--profile all` also covers the web profile and reusable catalog aliases. Fill in the provider endpoint and headers on each row, then verify access individually. Template rows start disabled. Paid-capable OpenRouter routers stay recorded in the inventory and are rejected by the gateway.

```sh
uv run python -m github_harness_agent.providers template --profile superagent
uv run python -m github_harness_agent.providers template --profile all
```

Set an integer `quality_priority` per model to choose the order. Higher scores run first, followed by unranked candidates. Ties and unranked choices preserve the web profile default and order, then deployment-file order. Templates place the active Super Agent default, NVIDIA's `poolside/laguna-xs-2.1`, first as a starting preference. Use matched coding replays to adjust priorities by verified patch success, confirmation of structured tool use, and latency. Provider claims and model size alone cannot establish the best route for this checkout. Set `attempt_timeout` in seconds to override the 15-second per-attempt default. Each attempt also respects the remaining request deadline. The executor supplies its remaining task time so slower providers can be reached during a longer fallback sequence.

For a loopback Ollama route configured with `local_model_digest` and `local_max_tokens`, optionally set `local_input_bytes` to an integer between 8192 and 57344. A value such as 12288 limits the serialized message history for that local attempt. Recovery prompts retain excerpts around reviewed source lines, and other providers receive the original history on fallback. Tool definitions are outside the message budget.

After a transport error, bad status, malformed response or unusable initial coding response, the gateway tries the best eligible model on another provider account. After trying each account, it cycles through their remaining models in rank order. A valid initial coding clarification can still be returned to the maintainer. Requests sharing an endpoint and credentials share a rate-limit cooldown and allow one concurrent request. A model with a transport or server failure cools down briefly while healthy routes serve subsequent requests. Gateway fallback applies to a single model exchange. Task retries carry failed model exclusions. If the entire provider chain is unavailable, the controller preserves the job, applies backoff, and rotates the preferred starting provider on its next attempt.

Each completion records the requested `configured_model` and the provider-reported `model`, which can differ when a router selects a backend. After a successful fallback, subsequent tool turns in that rollout prefer the working model and retain their conversation and checkout. A fresh rollout starts from the configured ranking and the task's retry position. Task retries can exclude an exact `(route, configured_model)` pair from further exchanges.

```sh
uv run python -m github_harness_agent.providers smoke
uv run python -m github_harness_agent.providers serve
```

The smoke command tries each configured model individually and reports failed credentials, unavailable models, quota exhaustion, or tool failures by model. Run it before enabling a deployment. Save the output as cutover evidence.

To check the importer against the repository fixtures, run the following command and compare its output with the versioned manifest.

```sh
uv run python -m github_harness_agent.providers import \
  tests/fixtures/incumbent_provider_profile.json \
  tests/fixtures/incumbent_provider_catalog.json \
  tests/fixtures/incumbent_provider_superagent.json
```

The incumbent snapshot covers the six web routes, eight reusable catalog aliases, and three active Super Agent routes observed on November 18, 2026. The web profile uses `allowedModels` to order subagent choices. The active Super Agent profile defines a default and contains no ordered fallback setting.

## Initial live validation

On November 17, 2026, `ollama-local/qwen2.5-coder:7b` passed basic chat through the gateway adapter. It returned no structured tool calls on two prompts through the OpenAI-compatible endpoint and one prompt through the native Ollama endpoint. The incumbent Ollama server did not have `gemma4:26b` installed. Cloud routes still require deployment-specific live validation.

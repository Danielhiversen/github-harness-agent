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

Set `free_verified` to true after verifying zero-cost model access for the exact endpoint and account. This includes confirming trial credits or free-plan eligibility where applicable. Set `tools` to true for a coding candidate and confirm the tool round trip with the smoke command. Only explicitly deployed, eligible models can receive requests. The gateway tries the incumbent default first, follows the ordered web profile candidate list, then tries any additional deployments in file order. It skips models cooling down after a rate-limit response.

```sh
uv run python -m github_harness_agent.providers smoke
uv run python -m github_harness_agent.providers serve
```

The smoke command tries each configured model individually and reports failed credentials, unavailable models, quota exhaustion, or tool failures by model. Run it before enabling a deployment. Save the output as cutover evidence.

To check the importer against the repository fixtures, run the following command and compare its output with the versioned manifest.

```sh
uv run python -m github_harness_agent.providers import \
  tests/fixtures/incumbent_provider_profile.json \
  tests/fixtures/incumbent_provider_catalog.json
```

The incumbent snapshot covers the six web routes and eight reusable catalog aliases observed on November 17, 2026. The incumbent profile uses `allowedModels` to order subagent choices. The gateway uses that order for fallbacks. Some catalog entries are paid-capable routing aliases. The manifest preserves those IDs for parity and the gateway rejects those deployments.

## Initial live validation

On November 17, 2026, `ollama-local/qwen2.5-coder:7b` passed basic chat through the gateway adapter. It returned no structured tool calls on two prompts through the OpenAI-compatible endpoint and one prompt through the native Ollama endpoint. The incumbent Ollama server did not have `gemma4:26b` installed. Cloud routes still require deployment-specific live validation.

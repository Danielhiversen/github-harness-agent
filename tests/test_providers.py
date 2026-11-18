import asyncio
import json
from pathlib import Path

import httpx
import pytest

from github_harness_agent.providers import (
    ModelGatewayClient,
    ProviderExhausted,
    ProviderRouter,
    create_model_gateway,
    import_profiles,
    load_manifest,
)


FIXTURES = Path(__file__).parent / "fixtures"
PROFILE = json.loads((FIXTURES / "incumbent_provider_profile.json").read_text())
CATALOG = json.loads((FIXTURES / "incumbent_provider_catalog.json").read_text())
SUPERAGENT = json.loads((FIXTURES / "incumbent_provider_superagent.json").read_text())
KILO = "kilo-free/nvidia/nemotron-3.5-super-vl:free"
OLLAMA = "ollama-local/gemma4:26b"
GOOGLE = "google/gemma-4-31b-it"


def deployment(*models):
    return [
        {
            "model": model,
            "url": f"https://{model.split('/')[0]}.example.invalid",
            "headers": {"Authorization": "Bearer fake"},
            "free_verified": True,
            "tools": True,
        }
        for model in models
    ]


def routed(config, handler):
    return ProviderRouter(
        load_manifest(),
        config,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def openai_response():
    return {
        "model": "actual-model",
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3},
    }


def test_import_preserves_exact_parity_without_copying_deployment_values():
    profile = json.loads(json.dumps(PROFILE))
    for route in profile[0]["config"]["providers"].values():
        route.update(
            baseURL="https://private.invalid", headers={"Authorization": "fake"}
        )
    imported = import_profiles(profile, CATALOG, SUPERAGENT)
    assert imported == load_manifest()
    for key, original in [("web", PROFILE), ("catalog", CATALOG)]:
        expected = original[0]["config"]["providers"]
        assert list(imported[key]) == list(expected)
        for alias, provider in expected.items():
            assert [model["id"] for model in imported[key][alias]["models"]] == [
                model["id"] for model in provider["models"]
            ]
    assert imported["default"] == PROFILE[1]["config"]
    assert imported["fallback"] == PROFILE[2]["config"]["allowedModels"]


def test_429_falls_back_and_reports_resolved_model_and_usage():
    def handler(req):
        if req.url.host.startswith("kilo"):
            return httpx.Response(429, headers={"Retry-After": "60"})
        return httpx.Response(200, json=openai_response())

    result = routed(deployment(OLLAMA, KILO), handler).complete(
        role="compose", messages=[]
    )
    assert (result.route, result.model, result.role) == (
        "ollama-local",
        "actual-model",
        "compose",
    )
    assert result.usage == {"input_tokens": 2, "output_tokens": 3}


def test_exhaustion_honors_cooldown_and_pauses_unverified_routes():
    config = deployment(KILO, OLLAMA)
    config[1]["free_verified"] = False
    calls = []

    def handler(req):
        calls.append(req.url.host)
        return httpx.Response(429, headers={"Retry-After": "60"})

    router = routed(config, handler)
    for _ in range(2):
        with pytest.raises(ProviderExhausted) as error:
            router.complete(role="plain", messages=[])
        assert error.value.retry_at is not None
        assert error.value.failures["ollama-local"] == "free eligibility unverified"
    assert calls == ["kilo-free.example.invalid"]


@pytest.mark.parametrize("first", ["timeout", "not found", "malformed"])
def test_provider_failures_use_next_eligible_model(first):
    def handler(req):
        if req.url.host.startswith("kilo"):
            if first == "timeout":
                raise httpx.ReadTimeout("fixture")
            if first == "not found":
                return httpx.Response(404)
            return httpx.Response(200, json={"choices": [{"message": {"content": {}}}]})
        return httpx.Response(200, json=openai_response())

    assert (
        routed(deployment(KILO, OLLAMA), handler)
        .complete(role="plain", messages=[])
        .route
        == "ollama-local"
    )


def test_gemini_tool_round_trip_replays_provider_signature():
    tools = [
        {
            "type": "function",
            "function": {"name": "echo", "parameters": {"type": "object"}},
        }
    ]
    messages = [{"role": "user", "content": "Echo."}]

    def handler(req):
        payload = json.loads(req.content)
        assert req.url.path.endswith("/models/gemma-4-31b-it:generateContent")
        assert payload["tools"][0]["functionDeclarations"][0]["name"] == "echo"
        if len(payload["contents"]) == 1:
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {
                                            "id": "call-1",
                                            "name": "echo",
                                            "args": {"text": "hi"},
                                        },
                                        "thoughtSignature": "fixture",
                                    }
                                ]
                            }
                        }
                    ]
                },
            )
        assert payload["contents"][1]["parts"][0]["thoughtSignature"] == "fixture"
        response = payload["contents"][2]["parts"][0]["functionResponse"]
        assert response == {
            "id": "call-1",
            "name": "echo",
            "response": {"output": "hi"},
        }
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": "done"}]}}]}
        )

    router = routed(deployment(GOOGLE), handler)
    call = router.complete(role="code", messages=messages, tools=tools)
    assert json.loads(call.tool_calls[0]["function"]["arguments"]) == {"text": "hi"}
    messages += [
        {
            "role": "assistant",
            "content": call.content,
            "tool_calls": call.tool_calls,
            "provider_context": call.provider_context,
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "hi"},
    ]
    assert (
        router.complete(role="code", messages=messages, tools=tools).content == "done"
    )


def test_gateway_requires_token_and_uses_provider_route():
    router = routed(
        deployment(KILO, OLLAMA),
        lambda req: httpx.Response(200, json=openai_response()),
    )
    app = create_model_gateway(router, "gateway-token")
    payload = {"role": "plain", "messages": []}

    async def dispatch(request):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
            response = await client.send(request)
            return httpx.Response(
                response.status_code, headers=response.headers, content=response.content
            )

    with httpx.Client(
        transport=httpx.MockTransport(lambda req: asyncio.run(dispatch(req)))
    ) as http:
        url = "https://gateway.example.invalid"
        assert http.post(f"{url}/complete", json=payload).status_code == 401
        client = ModelGatewayClient(url, "gateway-token", client=http)
        result = client.complete(**payload, exclude=[tuple(KILO.split("/", 1))])
        assert result.route == "ollama-local"
        assert result.configured_model == "gemma4:26b"
        assert result.model == "actual-model"


@pytest.mark.parametrize(
    "alias",
    sorted(
        set(PROFILE[0]["config"]["providers"]) | set(CATALOG[0]["config"]["providers"])
    ),
)
def test_alias_uses_its_chat_contract(alias):
    provider = (
        PROFILE[0]["config"]["providers"].get(alias)
        or CATALOG[0]["config"]["providers"][alias]
    )
    model = provider["models"][0]["id"]

    def handler(request):
        body = json.loads(request.content)
        if alias == "google":
            assert request.url.path == f"/models/{model}:generateContent"
            assert body["contents"] == [{"role": "user", "parts": [{"text": "test"}]}]
            return httpx.Response(
                200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
            )
        assert request.url.path == "/chat/completions"
        assert body["model"] == model
        if alias.startswith("openrouter"):
            assert body["provider"]["max_price"] == {"prompt": 0, "completion": 0}
        return httpx.Response(200, json=openai_response())

    router = routed(deployment(f"{alias}/{model}"), handler)
    assert (
        router.complete(
            role="plain", messages=[{"role": "user", "content": "test"}]
        ).content
        == "ok"
    )


def test_paid_catalog_routers_cannot_be_enabled():
    with pytest.raises(ValueError, match="outside the free allowlist"):
        routed(
            deployment("openrouter/openrouter/auto"),
            lambda _: pytest.fail("upstream call"),
        )


def test_chat_tool_round_trip():
    tools = [
        {
            "type": "function",
            "function": {"name": "echo", "parameters": {"type": "object"}},
        }
    ]
    messages = [{"role": "user", "content": "Echo."}]
    call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "echo", "arguments": '{"text": "hi"}'},
    }

    def handler(request):
        payload = json.loads(request.content)
        if len(payload["messages"]) == 1:
            return httpx.Response(
                200, json={"choices": [{"message": {"tool_calls": [call]}}]}
            )
        assert payload["messages"][1]["tool_calls"] == [call]
        assert payload["messages"][2]["tool_call_id"] == "call-1"
        return httpx.Response(200, json=openai_response())

    router = routed(deployment(KILO), handler)
    result = router.complete(role="coding", messages=messages, tools=tools)
    messages.extend(
        [
            {
                "role": "assistant",
                "content": result.content,
                "tool_calls": result.tool_calls,
                "provider_context": result.provider_context,
            },
            {"role": "tool", "content": "hi", "tool_call_id": "call-1"},
        ]
    )
    assert (
        router.complete(role="coding", messages=messages, tools=tools).content == "ok"
    )


def test_superagent_profile_preserves_active_models_and_default():
    active = import_profiles(PROFILE, CATALOG, SUPERAGENT)["superagent"]
    assert active == load_manifest()["superagent"]
    assert list(active["providers"]) == ["kilo-free", "ollama-local", "nvidia-free"]
    assert sum(len(r["models"]) for r in active["providers"].values()) == 11
    assert active["default"] == {
        "provider": "nvidia-free",
        "model": "poolside/laguna-xs-2.1",
    }
    assert active["fallback"] == []


def test_explicit_quality_priority_precedes_incumbent_default():
    config = deployment(KILO, OLLAMA)
    config[0]["quality_priority"] = 1
    config[1]["quality_priority"] = 10
    calls = []

    def handle(request):
        calls.append(request.url.host)
        if request.url.host.startswith("ollama"):
            return httpx.Response(502)
        return httpx.Response(200, json=openai_response())

    assert (
        routed(config, handle).complete(role="plain", messages=[]).route == "kilo-free"
    )
    assert calls == ["ollama-local.example.invalid", "kilo-free.example.invalid"]


def test_keyless_kilo_uses_only_an_explicitly_verified_route():
    config = deployment(KILO)
    config[0].update(headers={}, allow_keyless=True)

    def handle(request):
        assert "authorization" not in request.headers
        return httpx.Response(200, json=openai_response())

    assert (
        routed(config, handle).complete(role="plain", messages=[]).route == "kilo-free"
    )
    config[0]["free_verified"] = False
    with pytest.raises(ProviderExhausted):
        routed(config, lambda _: pytest.fail("unverified upstream call")).complete(
            role="plain", messages=[]
        )


@pytest.mark.parametrize("status", [429, 503])
def test_provider_cooldown_skips_sibling_models_for_the_same_account(status):
    config = deployment(KILO, "kilo-free/reflection/beam-free", OLLAMA)
    calls = []

    def handle(request):
        calls.append(request.url.host)
        if request.url.host.startswith("kilo"):
            return httpx.Response(status, headers={"Retry-After": "60"})
        return httpx.Response(200, json=openai_response())

    assert (
        routed(config, handle).complete(role="plain", messages=[]).route
        == "ollama-local"
    )
    assert calls == ["kilo-free.example.invalid", "ollama-local.example.invalid"]


@pytest.mark.parametrize(
    "first_choice",
    [
        {"message": {"content": "The work is done."}},
        {"message": {"content": "truncated"}, "finish_reason": "length"},
        {
            "message": {
                "tool_calls": [
                    {"id": "bad", "function": {"name": "run", "arguments": "{}"}}
                ]
            }
        },
        {
            "message": {
                "tool_calls": [
                    {
                        "id": "bad",
                        "function": {"name": "run", "arguments": '{"argv":[3]}'},
                    }
                ]
            }
        },
    ],
)
def test_unusable_initial_coding_completion_falls_back(first_choice):
    from github_harness_agent.executor import RUN_TOOL

    def handle(request):
        if request.url.host.startswith("kilo"):
            return httpx.Response(200, json={"choices": [first_choice]})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "ok",
                                    "function": {
                                        "name": "run",
                                        "arguments": '{"argv":["ls"]}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
        )

    result = routed(deployment(KILO, OLLAMA), handle).complete(
        role="coding",
        messages=[{"role": "user", "content": "Inspect this repo."}],
        tools=[RUN_TOOL],
    )
    assert result.route == "ollama-local"
    assert json.loads(result.tool_calls[0]["function"]["arguments"]) == {"argv": ["ls"]}


def test_initial_coding_completion_accepts_intentional_clarification():
    from github_harness_agent.executor import RUN_TOOL

    content = '{"clarification": "Which format should the output use?"}'
    router = routed(
        deployment(KILO),
        lambda _: httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
            },
        ),
    )
    assert (
        router.complete(role="coding", messages=[], tools=[RUN_TOOL]).content == content
    )


def test_concurrent_requests_skip_models_sharing_an_active_account():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    started, release = Event(), Event()

    def handle(request):
        if request.url.host.startswith("kilo"):
            started.set()
            assert release.wait(3)
        return httpx.Response(200, json=openai_response())

    router = routed(deployment(KILO, "kilo-free/reflection/beam-free", OLLAMA), handle)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(router.complete, role="plain", messages=[])
        try:
            assert started.wait(3)
            assert router.complete(role="plain", messages=[]).route == "ollama-local"
        finally:
            release.set()
        assert first.result().route == "kilo-free"


def test_temporary_failure_defers_unhealthy_model_between_requests():
    calls = []

    def handle(request):
        calls.append(request.url.host)
        if request.url.host.startswith("kilo"):
            return httpx.Response(502)
        return httpx.Response(200, json=openai_response())

    router = routed(deployment(KILO, OLLAMA), handle)
    for _ in range(2):
        assert router.complete(role="plain", messages=[]).route == "ollama-local"
    assert calls == [
        "kilo-free.example.invalid",
        "ollama-local.example.invalid",
        "ollama-local.example.invalid",
    ]


def test_superagent_template_covers_all_active_models_without_authorizing_requests(
    monkeypatch, capsys
):
    from github_harness_agent.providers import main

    monkeypatch.setattr(
        "sys.argv", ["providers", "template", "--profile", "superagent"]
    )
    assert main() == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["model"] == "nvidia-free/poolside/laguna-xs-2.1"
    expected = {
        f"{alias}/{model['id']}"
        for alias, provider in SUPERAGENT[0]["config"]["providers"].items()
        for model in provider["models"]
    }
    assert {row["model"] for row in rows} == expected
    assert all(
        not row["free_verified"] and not row["tools"] and not row["headers"]
        for row in rows
    )

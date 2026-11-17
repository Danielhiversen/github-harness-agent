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
    imported = import_profiles(profile, CATALOG)
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
        deployment(OLLAMA), lambda req: httpx.Response(200, json=openai_response())
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
        assert client.complete(**payload).route == "ollama-local"


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

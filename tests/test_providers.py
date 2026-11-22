import asyncio
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from github_harness_agent import providers
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
OPENROUTER = "openrouter-free/baidu/ernie-5.5-flash-preview:free"
OPENROUTER_SIBLING = "openrouter/thinkingmachines/inkling-2-small:free"
GROQ = "groq-free/qwen/qwen3.8-27b"


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


@pytest.mark.parametrize(
    "statuses,oversized,elapsed,expected_routes",
    [
        ([413], True, 1, ["kilo-free", "kilo-free"]),
        ([413, 413], True, 1, ["kilo-free", "kilo-free", "ollama-local"]),
        ([413], False, 1, ["kilo-free", "ollama-local"]),
        ([413], True, 20, ["kilo-free", "ollama-local"]),
        ([429], True, 1, ["kilo-free", "ollama-local"]),
        ([401], True, 1, ["kilo-free", "ollama-local"]),
    ],
)
def test_focused_review_retries_an_oversized_413_once_before_fallback(
    monkeypatch, statuses, oversized, elapsed, expected_routes
):
    calls = []
    responses = iter(statuses)
    now = 0
    monkeypatch.setattr(
        providers, "time", SimpleNamespace(monotonic=lambda: now, time=time.time)
    )

    def handle(request):
        nonlocal now
        calls.append(
            (
                request.url.host.split(".")[0],
                json.loads(request.content),
                request.extensions["timeout"]["read"],
            )
        )
        now += elapsed
        return httpx.Response(next(responses, 200), json=openai_response())

    router = routed(deployment(KILO, OLLAMA), handle)
    focused = (
        "Selected failure: test_edit\nReviewer feedback: preserve ordered values\n"
        "Focused source from module.py:\ndef edit():\n    return current_value\n"
        "Failure details:\nwrong result\nLast proposal feedback for this goal: try again\n"
        "\nFixture for this goal:\nassert edit()\n"
        "\nOther reviewer feedback:\n"
        + ("noise " * 5000 if oversized else "noise")
        + "\nCurrent checkout:\nmodule.py\nRespond with exact SEARCH/REPLACE blocks only."
    )
    router.complete(
        role="coding",
        messages=[
            {"role": "system", "content": "Edit source."},
            {"role": "user", "content": focused},
        ],
        max_output_tokens=3072,
    )
    assert [alias for alias, _, _ in calls] == expected_routes
    if expected_routes[0] == expected_routes[1]:
        _, payload, timeout = calls[1]
        assert payload["max_tokens"] == 1536
        assert len(json.dumps(payload)) <= 14 * 1024
        content = payload["messages"][1]["content"]
        assert "preserve ordered values" in content
        assert "return current_value" in content
        assert "try again" in content
        assert "Respond with exact SEARCH/REPLACE blocks only." in content
        assert timeout == calls[0][2] - elapsed


def test_compact_groq_reasoning_leaves_output_for_the_actual_edit():
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=openai_response())

    router = routed(deployment("groq-free/openai/gpt-oss-120b"), handle)
    request = (
        "Selected failure: test_case\nFocused source from module.py:\nvalue = value + 1\n"
        "Other reviewer feedback:\n"
        + ("other " * 5000)
        + "\nCurrent checkout:\nmodule.py\nRespond with exact SEARCH/REPLACE blocks only."
    )
    router.complete(
        role="coding",
        messages=[
            {"role": "system", "content": "Edit."},
            {"role": "user", "content": request},
        ],
        max_output_tokens=3072,
    )
    assert len(calls) == 1
    assert len(json.dumps(calls[0])) <= 14 * 1024
    assert calls[0]["max_tokens"] == 2600
    assert calls[0]["reasoning_effort"] == "low"


@pytest.mark.parametrize("use_tool", [False, True])
def test_nemotron_reserves_output_for_focused_reviews(use_tool):
    name = "nvidia-free/nvidia/nemotron-3-ultra-550b-a55b"
    router = routed(deployment(name), lambda req: httpx.Response(200, json=openai_response()))
    route = router.routes[0]
    messages = [
        {"role": "system", "content": "Edit the module."},
        {"role": "user", "content": "Selected failure: test_index_order"},
    ]
    tools = [{"type": "function", "function": {"name": "edit"}}] if use_tool else []
    body = providers._chat_payload(route, messages, tools, 3072)
    expected = {"enable_thinking": use_tool, "force_nonempty_content": True}
    if use_tool:
        expected["medium_effort"] = True
    assert body["chat_template_kwargs"] == expected
    messages[1]["content"] = "A general programming question"
    assert "chat_template_kwargs" not in providers._chat_payload(route, messages, tools, 3072)


def test_nemotron_planner_has_its_own_reasoning_budget():
    payloads = []
    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=openai_response())

    router = routed(deployment("nvidia-free/nvidia/nemotron-3-ultra-550b-a55b"), handle)
    router.complete(
        role="planning",
        messages=[{"role": "system", "content": "Plan."},
                  {"role": "user", "content": "Selected failure: test_order"}],
        max_output_tokens=4096,
    )
    assert payloads[0]["chat_template_kwargs"] == {
        "enable_thinking": True, "medium_effort": True, "force_nonempty_content": True,
    }
    assert payloads[0]["max_tokens"] == 4096


@pytest.mark.parametrize("named", [False, True])
def test_nemotron_extended_deadline_is_for_named_repair_only(named):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=openai_response())

    config = deployment("nvidia-free/nvidia/nemotron-3-ultra-550b-a55b")
    config[0]["attempt_timeout"] = 110
    router = routed(config, handle)
    content = "Selected failure: test_index_order\n"
    if named:
        content += providers.DEFINITION_ENDING
    router.complete(
        role="coding",
        messages=[{"role": "system", "content": "Edit."}, {"role": "user", "content": content}],
        max_output_tokens=6144,
        timeout=115,
    )
    payload = json.loads(requests[0].content)
    assert payload["chat_template_kwargs"]["enable_thinking"] == named
    assert payload["max_tokens"] == 6144
    timeout = requests[0].extensions["timeout"]["read"]
    limit = 110 if named else 60
    assert limit - 3 < timeout <= limit


def quality_order():
    config = deployment(OPENROUTER, OPENROUTER_SIBLING, KILO, OLLAMA)
    config[1]["url"] = config[0]["url"]
    for priority, item in enumerate(reversed(config), start=1):
        item["quality_priority"] = priority
    return config


def test_attempts_distinct_accounts_before_a_sibling_and_logs_only_route(caplog):
    calls = []

    def handle(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == OPENROUTER_SIBLING.split("/", 1)[1]:
            return httpx.Response(200, json=openai_response())
        return httpx.Response(404, text="private-upstream-response")

    router = routed(quality_order(), handle)
    with caplog.at_level(logging.INFO, logger=providers.__name__):
        result = router.complete(
            role="plain", messages=[{"role": "user", "content": "private-task-body"}]
        )
    assert result.route == "openrouter"
    assert calls == [
        model.split("/", 1)[1]
        for model in (OPENROUTER, KILO, OLLAMA, OPENROUTER_SIBLING)
    ]
    assert "provider=openrouter-free" in caplog.text
    assert "configured_model=thinkingmachines/inkling-2-small:free" in caplog.text
    assert "accepted" in caplog.text
    assert "private-task-body" not in caplog.text
    assert "private-upstream-response" not in caplog.text
    assert "Bearer fake" not in caplog.text
    assert "example.invalid" not in caplog.text


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (0, "openrouter-free"),
        (1, "kilo-free"),
        (2, "ollama-local"),
        (3, "openrouter-free"),
    ],
)
def test_offset_rotates_three_accounts_without_removing_primary(offset, expected):
    router = routed(
        quality_order(), lambda _: httpx.Response(200, json=openai_response())
    )
    assert (
        router.complete(role="plain", messages=[], provider_offset=offset).route
        == expected
    )


def test_exclusions_and_retry_offset_do_not_skip_the_next_fresh_account():
    excluded = [tuple(model.split("/", 1)) for model in (OPENROUTER, KILO)]
    groq = deployment(GROQ)[0]
    groq["quality_priority"] = 2
    calls = []

    def handle(request):
        calls.append(request.url.host)
        if not request.url.host.startswith("openrouter"):
            return httpx.Response(404)
        return httpx.Response(200, json=openai_response())

    result = routed([*quality_order(), groq], handle).complete(
        role="plain", messages=[], exclude=excluded, provider_offset=1
    )
    assert result.route == "openrouter"
    assert calls == [
        "groq-free.example.invalid",
        "ollama-local.example.invalid",
        "openrouter-free.example.invalid",
    ]


def test_rotation_counts_only_accounts_available_after_cooldown():
    calls = []

    def handle(request):
        calls.append(request.url.host)
        if request.url.host.startswith("openrouter"):
            return httpx.Response(429, headers={"Retry-After": "60"})
        return httpx.Response(200, json=openai_response())

    router = routed(quality_order(), handle)
    assert router.complete(role="plain", messages=[]).route == "kilo-free"
    assert (
        router.complete(role="plain", messages=[], provider_offset=1).route
        == "ollama-local"
    )
    assert calls == [
        "openrouter-free.example.invalid",
        "kilo-free.example.invalid",
        "ollama-local.example.invalid",
    ]


def test_model_cooldown_does_not_hide_its_healthy_sibling_on_a_new_request():
    def handle(request):
        model = json.loads(request.content)["model"]
        if model == OPENROUTER.split("/", 1)[1]:
            raise httpx.ReadTimeout("fixture")
        return httpx.Response(200, json=openai_response())

    router = routed(quality_order(), handle)
    assert router.complete(role="plain", messages=[]).route == "kilo-free"
    assert router.complete(role="plain", messages=[]).route == "openrouter"


def test_tool_continuation_prefers_exact_previous_model_then_distinct_provider():
    attempts = []

    def handle(request):
        body = json.loads(request.content)
        assert all("gateway_route" not in message for message in body["messages"])
        attempts.append(body["model"])
        if body["model"] == OPENROUTER_SIBLING.split("/", 1)[1]:
            return httpx.Response(404)
        return httpx.Response(200, json=openai_response())

    router = routed(quality_order(), handle)
    assistant = {
        "role": "assistant",
        "gateway_route": OPENROUTER_SIBLING.split("/", 1),
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "run", "arguments": '{"argv":["ls"]}'},
            }
        ],
    }
    messages = [assistant, {"role": "tool", "tool_call_id": "call-1", "content": "src"}]
    assert router.complete(role="coding", messages=messages).route == "kilo-free"
    assert attempts == [model.split("/", 1)[1] for model in (OPENROUTER_SIBLING, KILO)]
    assert (
        router.complete(role="proposal", messages=messages).route == "openrouter-free"
    )
    assert (
        router.complete(
            role="coding",
            messages=[*messages, {"role": "user", "content": "New task"}, messages[-1]],
        ).route
        == "openrouter-free"
    )
    malformed = {**assistant, "gateway_route": ["openrouter", None]}
    assert (
        router.complete(
            role="coding", messages=[*messages, malformed, messages[-1]]
        ).route
        == "openrouter-free"
    )


def test_exhaustion_identifies_all_excluded_only_if_their_routes_were_eligible():
    config = deployment(KILO, OLLAMA)
    config[1]["free_verified"] = False
    router = routed(config, lambda _: pytest.fail("excluded route reached upstream"))
    with pytest.raises(ProviderExhausted) as error:
        router.complete(role="plain", messages=[], exclude=[tuple(KILO.split("/", 1))])
    assert error.value.failures == {"gateway": "all eligible models excluded"}
    config[0]["free_verified"] = False
    router = routed(config, lambda _: pytest.fail("unverified route reached upstream"))
    with pytest.raises(ProviderExhausted) as error:
        router.complete(role="plain", messages=[], exclude=[tuple(KILO.split("/", 1))])
    assert error.value.failures == {
        "kilo-free": "free eligibility unverified",
        "ollama-local": "free eligibility unverified",
    }


@pytest.mark.parametrize(
    ("timeout", "expected"), [(None, [40, 20]), (120, [40, 40, 40])]
)
def test_explicit_long_deadline_can_reach_fallback_past_sixty_seconds(
    monkeypatch, timeout, expected
):
    clock = [0.0]
    monkeypatch.setattr(
        providers, "time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time)
    )
    config = deployment(OPENROUTER, KILO, OLLAMA)
    for priority, item in enumerate(reversed(config), start=1):
        item.update(quality_priority=priority, attempt_timeout=40)
    attempts = []

    def handle(request):
        limit = request.extensions["timeout"]["read"]
        attempts.append(limit)
        if request.url.host.startswith("ollama"):
            return httpx.Response(200, json=openai_response())
        clock[0] += limit
        raise httpx.ReadTimeout("fixture")

    router = routed(config, handle)
    if timeout is None:
        with pytest.raises(ProviderExhausted):
            router.complete(role="plain", messages=[])
    else:
        assert (
            router.complete(role="plain", messages=[], timeout=timeout).route
            == "ollama-local"
        )
    assert attempts == expected


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
    upstream_payloads = []

    def upstream(req):
        upstream_payloads.append(json.loads(req.content))
        return httpx.Response(200, json=openai_response())

    router = routed(
        deployment(KILO, OLLAMA),
        upstream,
    )
    app = create_model_gateway(router, "gateway-token")
    payload = {"role": "plain", "messages": []}
    requests = []

    async def dispatch(request):
        requests.append(request)
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
        assert "max_output_tokens" not in json.loads(requests[-1].content)
        local = next(route for route in router.routes if route.alias == "ollama-local")
        assert upstream_payloads[-1]["max_tokens"] == local.max_tokens
        assert client.complete(**payload, max_output_tokens=512).route == "kilo-free"
        assert upstream_payloads[-1]["max_tokens"] == 512
        assert (
            client.complete(**payload, provider_offset=1, timeout=900).route
            == "ollama-local"
        )
        assert json.loads(requests[-1].content)["provider_offset"] == 1
        assert 890 < json.loads(requests[-1].content)["timeout"] < 900
        assert requests[-1].extensions["timeout"]["read"] == 900
        assert client.complete(**payload, timeout=1200).route == "kilo-free"
        assert json.loads(requests[-1].content)["timeout"] < 900
        assert requests[-1].extensions["timeout"]["read"] == 900
        attempts = []
        result = client.complete(
            **payload,
            on_attempt=lambda *attempt: attempts.append(attempt),
            max_output_tokens=900,
        )
        assert attempts == [tuple(KILO.split("/", 1))]
        assert result.route == attempts[0][0]
        assert upstream_payloads[-1]["max_tokens"] == 900
        assert http.post(f"{url}/complete/observed", json=payload).status_code == 401
        with pytest.raises(ProviderExhausted) as failure:
            client.complete(
                **payload,
                on_attempt=lambda *unused: None,
                exclude=[tuple(m.split("/", 1)) for m in (OLLAMA, KILO)],
            )
        assert failure.value.failures == {"gateway": "all eligible models excluded"}
        auth = {"Authorization": "Bearer gateway-token"}
        for invalid in (
            {"provider_offset": -1},
            {"provider_offset": True},
            {"timeout": 901},
            {"max_output_tokens": 0},
            {"max_output_tokens": True},
            {"max_output_tokens": "512"},
        ):
            assert (
                http.post(
                    f"{url}/complete", json=payload | invalid, headers=auth
                ).status_code
                == 422
            )


@pytest.mark.parametrize("max_output_tokens", [None, 3072, 100_000])
def test_gemini_output_limit_is_bounded_by_the_route(max_output_tokens):
    payloads = []

    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": "done"}]}}]}
        )

    router = routed(deployment(GOOGLE), handle)
    route_max = router.routes[0].max_tokens
    result = router.complete(
        role="plain", messages=[], max_output_tokens=max_output_tokens
    )
    assert result.content == "done"
    assert payloads[0]["generationConfig"]["maxOutputTokens"] == min(
        max_output_tokens or route_max, route_max
    )


def test_gateway_preserves_requested_limit_when_observed_endpoint_is_unavailable():
    requests = []

    def dispatch(request):
        requests.append((request.url.path, json.loads(request.content)))
        if request.url.path.endswith("/observed"):
            return httpx.Response(404)
        return httpx.Response(
            200,
            json={
                "route": "fixture",
                "model": "fixture-model",
                "role": "coding",
                "content": "done",
                "tool_calls": [],
                "usage": {},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(dispatch)) as http:
        client = ModelGatewayClient(
            "https://gateway.example.invalid", "token", client=http
        )
        result = client.complete(
            role="coding",
            messages=[],
            max_output_tokens=512,
            on_attempt=lambda *_: None,
        )
    assert result.content == "done"
    assert [(path, body["max_output_tokens"]) for path, body in requests] == [
        ("/complete/observed", 512),
        ("/complete", 512),
    ]


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


@pytest.mark.parametrize("after_provider", [None, "openrouter-free"])
def test_failed_alternate_rechecks_top_model_released_by_concurrent_request(
    after_provider,
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    started, release = Event(), Event()
    attempts = []
    primary, alternate = (model.split("/", 1)[1] for model in (OPENROUTER, KILO))

    def handle(request):
        payload = json.loads(request.content)
        if payload["messages"]:
            assert payload["model"] == primary
            started.set()
            assert release.wait(3)
        else:
            attempts.append(payload["model"])
            if payload["model"] == alternate:
                release.set()
                assert first.result(timeout=3).route == "openrouter-free"
                raise httpx.ReadTimeout("fixture")
        return httpx.Response(200, json=openai_response())

    router = routed(quality_order(), handle)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(
            router.complete,
            role="plain",
            messages=[{"role": "user", "content": "Concurrent request"}],
        )
        try:
            assert started.wait(3)
            result = router.complete(
                role="plain", messages=[], after_provider=after_provider
            )
        finally:
            release.set()

    assert result.route == "openrouter-free"
    assert attempts == [alternate, primary]


@pytest.mark.parametrize("use_alternate", [True, False])
def test_recovered_model_preempts_untried_siblings(monkeypatch, use_alternate):
    clock = [0.0]
    monkeypatch.setattr(
        providers,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], time=lambda: clock[0]),
    )
    models = [
        KILO,
        "kilo-free/reflection/beam-free",
        "kilo-free/google/gemma-4-31b-it:free",
    ]
    if use_alternate:
        models.append(OLLAMA)
    primary, sibling, *_ = (model.split("/", 1)[1] for model in models)
    alternate = OLLAMA.split("/", 1)[1]
    config = deployment(*models)
    for priority, item in enumerate(reversed(config), start=1):
        item["quality_priority"] = priority

    attempts = []

    def handle(request):
        model = json.loads(request.content)["model"]
        attempts.append(model)
        if clock[0] == 0:
            raise httpx.ReadTimeout("fixture")
        if model == alternate:
            return httpx.Response(404)
        if model == sibling:
            clock[0] = 16
            return httpx.Response(404)
        return httpx.Response(200, json=openai_response())

    router = routed(config, handle)
    with pytest.raises(ProviderExhausted):
        router.complete(
            role="plain",
            messages=[],
            exclude=[tuple(model.split("/", 1)) for model in models[1:]],
        )
    assert attempts == [primary]

    attempts.clear()
    clock[0] = 1
    router.complete(role="plain", messages=[], after_provider="kilo-free")
    expected = [alternate] if use_alternate else []
    assert attempts == [*expected, sibling, primary]


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


def test_resumed_editor_account_tried_last_and_remains_eligible():
    attempts = []

    def upstream(request):
        alias = request.url.host.split(".")[0]
        attempts.append(alias)
        code = 200 if alias == "openrouter-free" else 404
        return httpx.Response(code, json=openai_response())

    router = routed(quality_order(), upstream)
    result = router.complete(
        role="coding", messages=[], after_provider="openrouter-free"
    )
    assert result.route == "openrouter-free"
    assert attempts == ["kilo-free", "ollama-local", "openrouter-free"]


def test_resumed_editor_returns_to_top_model_before_alternate_siblings():
    models = (KILO, OPENROUTER, OPENROUTER_SIBLING, OLLAMA)
    config = deployment(*models)
    config[2]["url"] = config[1]["url"]
    for priority, route in enumerate(reversed(config), start=1):
        route["quality_priority"] = priority
    previous, alternate, sibling, local = (model.split("/", 1)[1] for model in models)
    attempts = []

    def upstream(request):
        model = json.loads(request.content)["model"]
        attempts.append(model)
        status = 200 if model in {previous, sibling} else 404
        return httpx.Response(status, json=openai_response())

    router = routed(config, upstream)
    result = router.complete(role="coding", messages=[], after_provider="kilo-free")

    assert result.route == "kilo-free"
    assert attempts == [alternate, local, previous]


LOCAL_ALIAS = "ollama-local/harness-review-coder:30b"
LOCAL_MODEL = "harness-review-coder:30b"
LOCAL_URL = "http://127.0.0.1:11434/v1"
LOCAL_DIGEST = "a9" * 32


def pinned_local(**overrides):
    return {
        "model": LOCAL_ALIAS,
        "url": LOCAL_URL,
        "headers": {},
        "free_verified": True,
        "tools": True,
        "local_max_tokens": 4096,
        "local_model_digest": f"sha256:{LOCAL_DIGEST}",
        "attempt_timeout": 480,
        **overrides,
    }


def local_inventory(**overrides):
    return {
        "models": [
            {"name": LOCAL_MODEL, "digest": LOCAL_DIGEST, "size": 300000, **overrides}
        ]
    }


def mock_async_local(monkeypatch, handler):
    client = httpx.AsyncClient
    monkeypatch.setattr(
        providers.httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs),
    )


@pytest.mark.parametrize("route_cap,requested", [(4096, None), (2048, 3072)])
def test_pinned_unlisted_ollama_model_uses_loopback_inventory_and_completion(
    monkeypatch,
    route_cap,
    requested,
):
    calls = []

    async def handle(request):
        calls.append(request.method)
        if request.method == "GET":
            assert str(request.url) == "http://127.0.0.1:11434/api/tags"
            return httpx.Response(200, json=local_inventory())
        assert str(request.url) == f"{LOCAL_URL}/chat/completions"
        assert "authorization" not in request.headers
        body = json.loads(request.content)
        assert body["model"] == LOCAL_MODEL
        assert body["max_tokens"] == route_cap
        assert 110 < request.extensions["timeout"]["read"] <= 120
        return httpx.Response(200, json={**openai_response(), "model": LOCAL_MODEL})

    mock_async_local(monkeypatch, handle)
    router = ProviderRouter(load_manifest(), [pinned_local(local_max_tokens=route_cap)])
    assert calls == []
    result = router.complete(
        role="plain", messages=[], timeout=120, max_output_tokens=requested
    )
    assert (result.route, result.model) == ("ollama-local", LOCAL_MODEL)
    assert calls == ["GET", "POST"]
    assert LOCAL_MODEL not in {
        item["id"] for item in load_manifest()["web"]["ollama-local"]["models"]
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"url": "http://localhost:11434/v1"},
        {"url": "http://127.0.0.1:11434/v1?model=cloud"},
        {"url": "http://127.0.0.1:11435/v1"},
        {"headers": {"Authorization": "Bearer test"}},
        {"model": "kilo-free/harness-review-coder:30b"},
        {"model": "ollama-local/namespace/coder:30b"},
        {"model": "ollama-local/harness-review-coder:cloud"},
        {"model": "ollama-local/harness-review-coder:30b-cloud"},
        {"model": "ollama-local/harness-review-coder-cloud:30b"},
        {"model": "ollama-local/harness-review-coder"},
        {"local_max_tokens": 0},
        {"local_max_tokens": True},
        {"local_max_tokens": 8193},
        {"local_model_digest": "short"},
        {"local_model_digest": None},
        {"free_verified": False},
        {"attempt_timeout": 481},
    ],
)
def test_unlisted_local_model_requires_explicit_bounded_loopback_settings(overrides):
    with pytest.raises(ValueError):
        ProviderRouter(load_manifest(), [pinned_local(**overrides)])


@pytest.mark.parametrize(
    "inventory",
    [
        {"models": []},
        local_inventory(digest="1" * 64),
        local_inventory(remote_model="remote"),
        local_inventory(remote_host="https://example.invalid"),
        local_inventory(size=0),
    ],
)
def test_ollama_inventory_pin_failure_skips_local_completion(monkeypatch, inventory):
    async def handle(request):
        assert request.method == "GET"
        return httpx.Response(200, json=inventory)

    mock_async_local(monkeypatch, handle)
    router = ProviderRouter(load_manifest(), [pinned_local()])
    with pytest.raises(ProviderExhausted) as error:
        router.complete(role="plain", messages=[])
    assert error.value.failures == {"ollama-local": "model unavailable"}


def test_pinned_ollama_absolute_deadline_cancels_local_request(monkeypatch):
    cancelled = []

    async def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json=local_inventory())
        try:
            await asyncio.sleep(15)
        finally:
            cancelled.append(True)

    mock_async_local(monkeypatch, handle)
    router = ProviderRouter(load_manifest(), [pinned_local()])
    with pytest.raises(ProviderExhausted) as error:
        router.complete(role="plain", messages=[], timeout=0.1)
    assert cancelled
    assert error.value.failures == {"ollama-local": "upstream timeout"}


@pytest.mark.parametrize("model", [KILO, OLLAMA])
def test_long_attempt_timeout_remains_unavailable_to_unpinned_routes(model):
    config = deployment(model)
    config[0]["attempt_timeout"] = 480
    with pytest.raises(ValueError, match="at most 60"):
        ProviderRouter(load_manifest(), config)


@pytest.mark.parametrize("cap", [8191, 57345, True, None])
def test_pinned_ollama_input_limit_is_bounded(cap):
    with pytest.raises(ValueError, match="local_input_bytes"):
        ProviderRouter(load_manifest(), [pinned_local(local_input_bytes=cap)])


@pytest.mark.parametrize("model", [KILO, OLLAMA])
def test_ollama_input_limit_requires_a_pinned_route(model):
    config = deployment(model)
    config[0]["local_input_bytes"] = 12288
    with pytest.raises(ValueError, match="local_input_bytes"):
        ProviderRouter(load_manifest(), config)


@pytest.mark.parametrize("settings", [{}, {"local_input_bytes": 12288}])
def test_pinned_local_compaction_keeps_complete_tool_turns_and_cloud_history(
    monkeypatch,
    settings,
):
    messages = [{"role": "user", "content": "Make the requested change."}]
    for index in range(4):
        identifier = f"call_{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": identifier,
                            "type": "function",
                            "function": {"name": "run", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": identifier,
                    "content": f"exit 0\n{str(index) * 5200}",
                },
            ]
        )

    async def local(request):
        if request.method == "GET":
            return httpx.Response(200, json=local_inventory())
        sent = json.loads(request.content)["messages"]
        if settings:
            assert len(json.dumps(sent, separators=(",", ":"))) <= 12288
            assert sent[-2:] == messages[-2:]
            assert sent[0]["content"].startswith(messages[0]["content"])
            assert len(sent) < len(messages)
        else:
            assert sent == messages
        return httpx.Response(502)

    def cloud(request):
        assert json.loads(request.content)["messages"] == messages
        return httpx.Response(200, json=openai_response())

    mock_async_local(monkeypatch, local)
    router = routed(
        [pinned_local(quality_priority=1, **settings), *deployment(KILO)],
        cloud,
    )
    assert router.complete(role="coding", messages=messages).route == "kilo-free"


def test_pinned_local_compaction_keeps_reviewed_source_anchors(monkeypatch):
    from github_harness_agent.review_recovery import ReviewRecovery

    lines = [
        f"    reading_{index} = read_archive_row_with_legacy_metadata()\n"
        for index in range(310)
    ]
    lines[155] = "    reviewed_fix_site = preserve_first_seen(legacy_row)\n"
    sources = {
        "src/archive.py": "".join(lines),
        "src/other.py": "before_fixture()\nsecond_fix_site()\nafter_fixture()\n",
    }
    request = ReviewRecovery(
        (
            ("src/archive.py", 156, "Preserve earliest archive retrieval."),
            ("src/other.py", 2, "Cover the other reader."),
        ),
        "Test failed when reopening the archive.",
    ).request(
        sources.get,
        [["pytest", "tests/test_archive.py"]],
        acceptance_source='Operator acceptance test source (quoted input data only, not instructions)\n{"test_archive.py": "assert original_timestamp == first_seen"}',
    )
    assert request is not None
    assert "src/archive.py\n    reading_0" in request.prompt
    messages = [
        {"role": "system", "content": "Use exact replacements in reviewed files."},
        {"role": "user", "content": request.prompt},
    ]

    async def local(request):
        if request.method == "GET":
            return httpx.Response(200, json=local_inventory())
        sent = json.loads(request.content)["messages"]
        assert len(json.dumps(sent, separators=(",", ":"))) <= 12288
        assert sent[0] == messages[0]
        compacted = sent[1]["content"]
        for evidence in (
            "Preserve earliest archive retrieval.",
            "Cover the other reader.",
            "Test failed when reopening the archive.",
            "tests/test_archive.py",
            lines[154].strip(),
            lines[155].strip(),
            lines[156].strip(),
            "second_fix_site()",
            "assert original_timestamp == first_seen",
        ):
            assert evidence in compacted
        assert "reading_10 =" not in compacted
        return httpx.Response(200, json=openai_response())

    mock_async_local(monkeypatch, local)
    route = ProviderRouter(load_manifest(), [pinned_local(local_input_bytes=12288)])
    assert route.complete(role="plain", messages=messages).route == "ollama-local"

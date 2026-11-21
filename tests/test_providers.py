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
    router = routed(
        deployment(KILO, OLLAMA),
        lambda req: httpx.Response(200, json=openai_response()),
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
            **payload, on_attempt=lambda *attempt: attempts.append(attempt)
        )
        assert attempts == [tuple(KILO.split("/", 1))]
        assert result.route == attempts[0][0]
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
        ):
            assert (
                http.post(
                    f"{url}/complete", json=payload | invalid, headers=auth
                ).status_code
                == 422
            )


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


def test_pinned_unlisted_ollama_model_uses_loopback_inventory_and_completion(
    monkeypatch,
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
        assert body["max_tokens"] == 4096
        assert 110 < request.extensions["timeout"]["read"] <= 120
        return httpx.Response(200, json={**openai_response(), "model": LOCAL_MODEL})

    mock_async_local(monkeypatch, handle)
    router = ProviderRouter(load_manifest(), [pinned_local()])
    assert calls == []
    result = router.complete(role="plain", messages=[], timeout=120)
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

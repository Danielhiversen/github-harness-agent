"""Import public model settings and route bounded requests through a credentialed gateway."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from email.utils import parsedate_to_datetime
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import yaml
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

PUBLIC_ALIASES = frozenset(
    {
        "kilo-free",
        "ollama-local",
        "openrouter-free",
        "nvidia-free",
        "routeway-free",
        "groq-free",
        "openrouter",
        "nvidia",
        "google",
        "zai",
        "mistral",
        "huggingface",
    }
)
MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,159}")
NON_FREE_ROUTERS = frozenset(
    {
        "openrouter/auto",
        "openrouter/auto-beta",
        "openrouter/fusion",
        "openrouter/pareto-code",
        "typesafe/jev-router",
        "auto",
    }
)


def import_profiles(
    profile: list[dict], catalog: list[dict], superagent: list[dict] | None = None
) -> dict:
    """Whitelist known Cordis plugins and fields. Never copy credential configuration."""

    def plugin(entries: list[dict], name: str) -> dict:
        return next(
            (entry["config"] for entry in entries if entry.get("id") == name), {}
        )

    def model_id(value: str) -> str:
        if not MODEL_ID.fullmatch(value) or "://" in value:
            raise ValueError("Invalid public model ID")
        return value

    def routes(entries: list[dict]) -> dict:
        public = {}
        for alias, provider in (
            plugin(entries, "llm-pi-ai").get("providers", {}).items()
        ):
            if alias not in PUBLIC_ALIASES:
                continue
            models = []
            for model in provider["models"]:
                models.append(
                    {
                        "id": model_id(model["id"]),
                        "input": [
                            kind for kind in model["input"] if kind in {"text", "image"}
                        ],
                        "context_window": int(model["contextWindow"]),
                        "max_tokens": int(model["maxTokens"]),
                        **(
                            {"reasoning": str(model["reasoning"]).lower() == "true"}
                            if "reasoning" in model
                            else {}
                        ),
                    }
                )
            public[alias] = {
                "protocol": "gemini" if alias == "google" else "chat",
                "models": models,
            }
        return public

    def selection(entry: dict, providers: dict) -> dict:
        alias, model = entry["provider"], model_id(entry["model"])
        if alias not in providers or model not in {
            item["id"] for item in providers[alias]["models"]
        }:
            raise ValueError("Configured choice references an unknown model")
        return {"provider": alias, "model": model}

    def policy(entries: list[dict], providers: dict) -> dict:
        return {
            "default": selection(plugin(entries, "agent-default-model"), providers),
            "fallback": [
                selection(entry, providers)
                for entry in plugin(entries, "subagent-model-selection-settings").get(
                    "allowedModels", []
                )
            ],
        }

    web = routes(profile)
    manifest = {
        "version": 1,
        "web": web,
        "catalog": routes(catalog),
        **policy(profile, web),
    }
    if superagent is not None:
        active = routes(superagent)
        manifest["superagent"] = {"providers": active, **policy(superagent, active)}
    return manifest


def load_manifest() -> dict:
    return json.loads(
        files(__package__).joinpath("provider_manifest.v1.json").read_text()
    )


@dataclass(frozen=True)
class Completion:
    content: str | None
    tool_calls: tuple[dict[str, Any], ...]
    route: str
    model: str
    role: str
    usage: dict[str, int]
    provider_context: dict[str, Any] | None = None
    configured_model: str | None = None


class ProviderExhausted(RuntimeError):
    def __init__(self, retry_at: float | None, failures: dict[str, str] | None = None):
        super().__init__("No eligible model is currently available")
        self.retry_at = retry_at
        self.failures = failures or {}


@dataclass(frozen=True)
class _Route:
    alias: str
    model: str
    protocol: str
    url: str
    headers: dict[str, str]
    max_tokens: int
    free_verified: bool
    tools: bool
    allow_keyless: bool
    quality_priority: int | None
    attempt_timeout: float
    quota_key: tuple


def _call(identifier: str, name: str, arguments: Any) -> dict:
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if (
        not isinstance(arguments, dict)
        or not isinstance(identifier, str)
        or not identifier
    ):
        raise ValueError("Invalid tool call")
    if not isinstance(name, str) or not name:
        raise ValueError("Invalid tool name")
    return {
        "id": identifier,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, allow_nan=False)},
    }


def _gemini_payload(
    route: _Route, messages: Sequence[Mapping], tools: Sequence[Mapping]
) -> dict:
    system = []
    history = []
    calls = {}
    for msg in messages:
        role, content = msg["role"], msg.get("content")
        if role == "system":
            system.append({"text": content})
            continue
        if role == "tool":
            name, native_id = calls[msg["tool_call_id"]]
            part = {"functionResponse": {"name": name, "response": {"output": content}}}
            if native_id:
                part["functionResponse"]["id"] = native_id
            if (
                history
                and history[-1]["role"] == "user"
                and "functionResponse" in history[-1]["parts"][0]
            ):
                history[-1]["parts"].append(part)
            else:
                history.append({"role": "user", "parts": [part]})
            continue

        context = msg.get("provider_context") or {}
        native = (
            context.get("route") == route.alias and context.get("model") == route.model
        )
        parts = (
            context.get("parts", [])
            if native
            else ([{"text": content}] if content else [])
        )
        native_calls = [
            part["functionCall"] for part in parts if "functionCall" in part
        ]
        for index, call in enumerate(msg.get("tool_calls", [])):
            fn = call["function"]
            if native:
                calls[call["id"]] = (fn["name"], native_calls[index].get("id"))
            else:
                calls[call["id"]] = (fn["name"], call["id"])
                parts.append(
                    {
                        "functionCall": {
                            "name": fn["name"],
                            "id": call["id"],
                            "args": json.loads(fn["arguments"]),
                        },
                        "thoughtSignature": "skip_thought_signature_validator",
                    }
                )
        history.append(
            {"role": "model" if role == "assistant" else "user", "parts": parts}
        )

    body: dict[str, Any] = {
        "contents": history,
        "generationConfig": {"maxOutputTokens": route.max_tokens},
    }
    if system:
        body["systemInstruction"] = {"parts": system}
    if tools:
        declarations = [
            {
                key: value
                for key, value in tool["function"].items()
                if key in {"name", "description", "parameters"}
            }
            for tool in tools
        ]
        body["tools"] = [{"functionDeclarations": declarations}]
    return body


def _chat_payload(
    route: _Route, messages: Sequence[Mapping], tools: Sequence[Mapping]
) -> dict:
    cleaned = []
    for msg in messages:
        item = {
            k: v
            for k, v in msg.items()
            if k in {"role", "content", "tool_calls", "tool_call_id", "name"}
        }
        if route.alias == "mistral":
            if item.get("tool_calls"):
                item["tool_calls"] = [
                    {**call, "id": hashlib.sha256(call["id"].encode()).hexdigest()[:9]}
                    for call in item["tool_calls"]
                ]
            if item.get("tool_call_id"):
                item["tool_call_id"] = hashlib.sha256(
                    item["tool_call_id"].encode()
                ).hexdigest()[:9]
        cleaned.append(item)

    body: dict[str, Any] = {
        "model": route.model,
        "messages": cleaned,
        "max_tokens": route.max_tokens,
    }
    if route.alias.startswith("openrouter"):
        body["provider"] = {"max_price": {"prompt": 0, "completion": 0}}
    if tools:
        body["tools"] = list(tools)
    return body


def _decode(
    route: _Route, role: str, body: dict, tools: Sequence[Mapping]
) -> Completion:
    context = None
    if route.protocol == "gemini":
        candidate = body["candidates"][0]
        if candidate.get("finishReason") not in {None, "STOP"}:
            raise ValueError("Gemini returned an incomplete completion")
        parts = candidate["content"]["parts"]
        content = (
            "".join(part.get("text", "") for part in parts if not part.get("thought"))
            or None
        )
        calls = tuple(
            _call(fn.get("id") or f"gemini-{index}", fn["name"], fn.get("args", {}))
            for index, part in enumerate(parts)
            if (fn := part.get("functionCall"))
        )
        usage = body.get("usageMetadata") or {}
        input_tokens = usage.get("promptTokenCount")
        output_tokens = None
        if "candidatesTokenCount" in usage or "thoughtsTokenCount" in usage:
            output_tokens = usage.get("candidatesTokenCount", 0) + usage.get(
                "thoughtsTokenCount", 0
            )
        model = body.get("modelVersion") or route.model
        context = {"route": route.alias, "model": route.model, "parts": parts}
    else:
        choice = body["choices"][0]
        if choice.get("finish_reason") in {"length", "content_filter", "error"}:
            raise ValueError("Provider returned an incomplete completion")
        message = choice["message"]
        content = message.get("content")
        calls = tuple(
            _call(call["id"], call["function"]["name"], call["function"]["arguments"])
            for call in (message.get("tool_calls") or [])
        )
        if not calls and (legacy := message.get("function_call")):
            calls = (_call("legacy-function", legacy["name"], legacy["arguments"]),)
        usage = body.get("usage") or {}
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        model = body.get("model") or route.model

    allowed_tools = {
        tool["function"]["name"]: tool["function"].get("parameters", {})
        for tool in tools
    }
    for call in calls:
        function = call["function"]
        name = function["name"]
        if name not in allowed_tools:
            raise ValueError("Provider called an undeclared tool")
        parameters = allowed_tools[name]
        arguments = json.loads(function["arguments"])
        properties = parameters.get("properties", {})
        if not set(parameters.get("required", ())) <= arguments.keys() or (
            parameters.get("additionalProperties") is False
            and not arguments.keys() <= properties.keys()
        ):
            raise ValueError("Provider returned invalid tool arguments")
        if name == "run" and "argv" in properties:
            argv = arguments.get("argv")
            if (
                not isinstance(argv, list)
                or not argv
                or not all(isinstance(arg, str) for arg in argv)
            ):
                raise ValueError("Provider returned an invalid command argument vector")
    if len({call["id"] for call in calls}) != len(calls):
        raise ValueError("Provider returned duplicate call IDs")
    if content is not None and not isinstance(content, str):
        raise ValueError("Invalid message content")
    if not (content and content.strip()) and not calls:
        raise ValueError("Empty completion")
    counts = {
        name: value
        for name, value in {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }.items()
        if value is not None
    }
    if not isinstance(model, str) or any(
        type(n) is not int or n < 0 for n in counts.values()
    ):
        raise ValueError("Invalid completion metadata")
    return Completion(
        content, calls, route.alias, model, role, counts, context, route.model
    )


def _clarification(content: str | None) -> bool:
    try:
        decoded = json.loads(content or "")
    except ValueError:
        return False
    if not isinstance(decoded, dict):
        return False
    question = decoded.get("clarification")
    return isinstance(question, str) and bool(question.strip())


def _retry_time(value: str | None) -> float:
    now = time.time()
    if value:
        try:
            return now + max(1, int(value))
        except (OverflowError, ValueError):
            try:
                return max(now + 1, parsedate_to_datetime(value).timestamp())
            except (TypeError, ValueError, OverflowError):
                pass
    return now + 60


def _model_specs(manifest: dict, *, active_only: bool = False) -> dict:
    active = manifest.get("superagent", {}).get("providers", {})
    groups = [active] if active_only else [manifest["catalog"], manifest["web"], active]
    return {
        (alias, model["id"]): (provider["protocol"], model["max_tokens"])
        for group in groups
        for alias, provider in group.items()
        for model in provider["models"]
    }


def _paid_router(alias: str, model: str) -> bool:
    return alias.startswith("openrouter") and model in NON_FREE_ROUTERS


class ProviderRouter:
    def __init__(
        self,
        manifest: dict,
        deployments: Sequence[dict],
        *,
        client: httpx.Client | None = None,
    ):
        self.client = client or httpx.Client(follow_redirects=False, trust_env=False)
        self._quota_cooldown: dict[tuple, float] = {}
        self._model_cooldown: dict[tuple[str, str], float] = {}
        self._in_flight: set[tuple] = set()
        self._lock = threading.Lock()
        if manifest["version"] != 1:
            raise ValueError("Unsupported provider manifest version")
        allowed = _model_specs(manifest)
        configured = {}
        for item in deployments:
            alias, model = item["model"].split("/", 1)
            if (alias, model) not in allowed or _paid_router(alias, model):
                raise ValueError(
                    "The deployment contains a model outside the free allowlist"
                )
            if (alias, model) in configured:
                raise ValueError("Duplicate model deployment")
            url = item.get("url", "")
            endpoint = httpx.URL(url)
            if url and (endpoint.scheme not in {"http", "https"} or not endpoint.host):
                raise ValueError("Invalid model endpoint")
            headers = item.get("headers", {})
            fingerprint = hashlib.sha256(
                json.dumps(
                    {name.lower(): value for name, value in headers.items()},
                    sort_keys=True,
                ).encode()
            ).digest()
            priority = item.get("quality_priority")
            if priority is not None and type(priority) is not int:
                raise ValueError("quality_priority must be an integer")
            attempt_timeout = item.get("attempt_timeout", 15)
            if (
                type(attempt_timeout) not in {int, float}
                or not 0 < attempt_timeout <= 60
            ):
                raise ValueError(
                    "attempt_timeout must be above zero and at most 60 seconds"
                )
            keyless = item.get("allow_keyless") is True
            if keyless and alias != "kilo-free":
                raise ValueError("allow_keyless applies only to the Kilo free route")
            protocol, max_tokens = allowed[alias, model]
            configured[alias, model] = _Route(
                alias,
                model,
                protocol,
                url.rstrip("/"),
                headers,
                max_tokens,
                item.get("free_verified") is True,
                item.get("tools") is True,
                keyless,
                priority,
                attempt_timeout,
                (endpoint.scheme, endpoint.host, endpoint.port, fingerprint),
            )
        choices = [
            (r["provider"], r["model"])
            for r in [manifest["default"], *manifest["fallback"]]
        ]
        candidates = [
            configured[key]
            for key in dict.fromkeys([*choices, *configured])
            if key in configured
        ]
        self.routes = tuple(
            sorted(
                candidates,
                key=lambda route: (
                    route.quality_priority is not None,
                    route.quality_priority or 0,
                ),
                reverse=True,
            )
        )

    def complete(
        self,
        *,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
        timeout: float | None = None,
        exclude: Sequence[tuple[str, str]] = (),
    ) -> Completion:
        exclusions = set(exclude)
        routes = (
            route
            for route in self.routes
            if (route.alias, route.model) not in exclusions
        )
        return self._complete(routes, role, messages, tools, timeout)

    def _complete(self, routes, role, messages, tools, timeout):
        deadline = time.monotonic() + min(timeout if timeout is not None else 60, 60)
        failures = {}
        retry_times = []
        needs_initial_tool = (
            tools
            and role in {"coding", "probe"}
            and not any(message.get("role") == "tool" for message in messages)
        )
        for route in routes:
            key = (route.alias, route.model)
            attempt_timeout = min(route.attempt_timeout, deadline - time.monotonic())
            if attempt_timeout <= 0:
                break
            if not route.free_verified:
                failures[route.alias] = "free eligibility unverified"
                continue
            if not route.url or not (
                route.headers or route.alias == "ollama-local" or route.allow_keyless
            ):
                failures[route.alias] = "missing endpoint or credentials"
                continue
            if tools and not route.tools:
                failures[route.alias] = "tool capability unverified"
                continue
            with self._lock:
                now = time.time()
                quota_cooldown = self._quota_cooldown.get(route.quota_key, 0)
                model_cooldown = self._model_cooldown.get(key, 0)
                cooldown = max(quota_cooldown, model_cooldown)
                if cooldown > now:
                    failures[route.alias] = (
                        "rate limited"
                        if quota_cooldown > now
                        else "temporarily unavailable"
                    )
                    retry_times.append(cooldown)
                    continue
                if route.quota_key in self._in_flight:
                    failures[route.alias] = "request in progress"
                    retry_times.append(now + 1)
                    continue
                self._in_flight.add(route.quota_key)

            try:
                if route.protocol == "gemini":
                    url = f"{route.url}/models/{quote(route.model, safe='')}:generateContent"
                    payload = _gemini_payload(route, messages, tools)
                else:
                    url = f"{route.url}/chat/completions"
                    payload = _chat_payload(route, messages, tools)
                response = self.client.post(
                    url,
                    json=payload,
                    headers=route.headers,
                    timeout=attempt_timeout,
                    follow_redirects=False,
                )
                retry_after = response.headers.get("Retry-After")
                if (
                    response.status_code == 429
                    or response.status_code == 503
                    and retry_after
                ):
                    cooldown = _retry_time(retry_after)
                    with self._lock:
                        self._quota_cooldown[route.quota_key] = cooldown
                    retry_times.append(cooldown)
                    failures[route.alias] = "rate limited"
                    continue
                response.raise_for_status()
                result = _decode(route, role, response.json(), tools)
                if (
                    needs_initial_tool
                    and not result.tool_calls
                    and not (role == "coding" and _clarification(result.content))
                ):
                    failures[route.alias] = "initial coding response did not use a tool"
                    continue
                return result
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                failures[route.alias] = {
                    401: "invalid or missing credentials",
                    403: "account or model ineligible",
                    404: "model unavailable",
                }.get(status, f"upstream HTTP {status}")
                if status is not None and status >= 500:
                    retry_times.append(self._defer_model(key))
            except httpx.TimeoutException:
                failures[route.alias] = "upstream timeout"
                retry_times.append(self._defer_model(key))
            except httpx.RequestError:
                failures[route.alias] = "upstream connection failed"
                retry_times.append(self._defer_model(key))
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                failures[route.alias] = "malformed request or upstream response"
            finally:
                with self._lock:
                    self._in_flight.remove(route.quota_key)

        raise ProviderExhausted(min(retry_times, default=None), failures)

    def _defer_model(self, key: tuple[str, str]) -> float:
        retry_at = time.time() + 15
        with self._lock:
            self._model_cooldown[key] = retry_at
        return retry_at


class _Request(BaseModel):
    role: str = Field(min_length=1, max_length=32)
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    timeout: float | None = Field(default=None, gt=0, le=60)
    exclude: list[tuple[str, str]] = Field(default_factory=list, max_length=1000)


def create_model_gateway(router: ProviderRouter, token: str) -> FastAPI:
    if not token:
        raise ValueError("Empty model gateway token")
    app = FastAPI(openapi_url=None)
    expected = f"Bearer {token}".encode()

    @app.post("/complete")
    def complete(payload: _Request, authorization: str = Header(default="")) -> dict:
        if not hmac.compare_digest(authorization.encode(), expected):
            raise HTTPException(401, "Gateway authentication required")
        try:
            return asdict(router.complete(**payload.model_dump()))
        except ProviderExhausted as exc:
            raise HTTPException(
                503, {"retry_at": exc.retry_at, "failures": exc.failures}
            ) from None

    return app


class ModelGatewayClient:
    def __init__(
        self, base_url: str, token: str, *, client: httpx.Client | None = None
    ):
        if not base_url or not token:
            raise ValueError("Model gateway URL and token are required")
        self.url = f"{base_url.rstrip('/')}/complete"
        self.headers = {"Authorization": f"Bearer {token}"}
        self.client = client or httpx.Client(follow_redirects=False, trust_env=False)

    @classmethod
    def from_env(cls) -> ModelGatewayClient:
        return cls(os.environ["MODEL_GATEWAY_URL"], os.environ["MODEL_GATEWAY_TOKEN"])

    def complete(
        self,
        *,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
        timeout: float | None = None,
        exclude: Sequence[tuple[str, str]] = (),
    ) -> Completion:
        limit = min(timeout if timeout is not None else 60, 60)
        if not math.isfinite(limit) or limit <= 0:
            raise ProviderExhausted(None, {"gateway": "deadline exceeded"})
        try:
            response = self.client.post(
                self.url,
                headers=self.headers,
                timeout=limit,
                json={
                    "role": role,
                    "messages": list(messages),
                    "tools": list(tools),
                    "timeout": limit,
                    "exclude": list(exclude),
                },
            )
            if response.status_code == 503:
                details = response.json()["detail"]
                raise ProviderExhausted(details["retry_at"], details["failures"])
            response.raise_for_status()
            result = response.json()
            result["tool_calls"] = tuple(result["tool_calls"])
            return Completion(**result)
        except (ValueError, TypeError, KeyError, httpx.HTTPError):
            raise ProviderExhausted(
                None, {"gateway": "authenticated gateway request failed"}
            ) from None


def _smoke(router: ProviderRouter) -> int:
    """Probe each explicitly enabled model without using fallback."""
    failures = 0
    if not router.routes:
        print("No model deployments configured")
        return 1
    tool = {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Return a fixed text string.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }
    for route in router.routes:
        stage = "basic chat"
        try:
            router._complete(
                (route,),
                "smoke",
                [{"role": "user", "content": "Reply with OK."}],
                [],
                route.attempt_timeout,
            )
            if route.tools:
                stage = "tool round trip"
                messages = [
                    {
                        "role": "user",
                        "content": "Call the echo tool exactly once with text set to OK.",
                    }
                ]
                result = router._complete(
                    (route,), "smoke", messages, [tool], route.attempt_timeout
                )
                if len(result.tool_calls) != 1 or json.loads(
                    result.tool_calls[0]["function"]["arguments"]
                ) != {"text": "OK"}:
                    raise ValueError("tool probe did not return the expected call")
                messages.append(
                    {
                        "role": "assistant",
                        "content": result.content,
                        "tool_calls": result.tool_calls,
                        "provider_context": result.provider_context,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.tool_calls[0]["id"],
                        "content": "OK",
                    }
                )
                final = router._complete(
                    (route,), "smoke", messages, [tool], route.attempt_timeout
                )
                if not final.content or final.tool_calls:
                    raise ValueError("tool probe did not return final text")
            print(f"{route.alias}/{route.model}: {stage} passed")
        except ProviderExhausted as exc:
            failures += 1
            print(
                f"{route.alias}/{route.model}: {stage} failed ({exc.failures.get(route.alias, 'deadline exceeded')})"
            )
        except ValueError as exc:
            failures += 1
            print(f"{route.alias}/{route.model}: {stage} failed ({exc})")
    return int(bool(failures))


def _deployment_template(manifest: dict, *, active_only: bool) -> list[dict]:
    default = manifest.get("superagent", {}).get("default", manifest["default"])
    primary = (default["provider"], default["model"])
    specs = _model_specs(manifest, active_only=active_only)
    return [
        {
            "model": f"{alias}/{model}",
            "url": "",
            "headers": {},
            "free_verified": False,
            "tools": False,
            **({"quality_priority": 1} if (alias, model) == primary else {}),
        }
        for alias, model in dict.fromkeys([primary, *specs])
        if (alias, model) in specs and not _paid_router(alias, model)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("import")
    ingest.add_argument("web", type=Path)
    ingest.add_argument("catalog", type=Path)
    ingest.add_argument("superagent", type=Path, nargs="?")
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8765, type=int)
    commands.add_parser("smoke")
    template = commands.add_parser("template")
    template.add_argument("--profile", choices=("all", "superagent"), default="all")
    args = parser.parse_args()
    if args.command == "import":
        profile = yaml.load(args.web.read_text(), Loader=yaml.BaseLoader)
        catalog = yaml.load(args.catalog.read_text(), Loader=yaml.BaseLoader)
        superagent = (
            yaml.load(args.superagent.read_text(), Loader=yaml.BaseLoader)
            if args.superagent
            else None
        )
        print(json.dumps(import_profiles(profile, catalog, superagent), indent=2))
        return 0
    if args.command == "template":
        template = _deployment_template(
            load_manifest(), active_only=args.profile == "superagent"
        )
        print(json.dumps(template, indent=2))
        return 0
    deployments = json.loads(Path(os.environ["MODEL_GATEWAY_DEPLOYMENTS"]).read_text())
    router = ProviderRouter(load_manifest(), deployments)
    if args.command == "smoke":
        return _smoke(router)
    import uvicorn

    uvicorn.run(
        create_model_gateway(router, os.environ["MODEL_GATEWAY_TOKEN"]),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

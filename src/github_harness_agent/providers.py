"""Import public model settings and route bounded requests through a credentialed gateway."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from email.utils import parsedate_to_datetime
from importlib.resources import files
from itertools import zip_longest
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import yaml
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .task_context import MESSAGE_BYTES, compact_messages, wire_size

logger = logging.getLogger(__name__)
DEFAULT_REQUEST_TIMEOUT = 60
MAX_REQUEST_TIMEOUT = 900
_OLLAMA_ORIGIN = "http://127.0.0.1:11434"
_OLLAMA_API = f"{_OLLAMA_ORIGIN}/v1"
_LOCAL_MODEL_NAME = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}:[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
)
_LOCAL_ATTEMPT_MAX = 480

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
    local_model_digest: str | None = None
    local_input_bytes: int | None = None


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
    route: _Route,
    messages: Sequence[Mapping],
    tools: Sequence[Mapping],
    max_tokens: int,
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
        "generationConfig": {"maxOutputTokens": max_tokens},
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
    route: _Route,
    messages: Sequence[Mapping],
    tools: Sequence[Mapping],
    max_tokens: int,
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
        "max_tokens": max_tokens,
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


def _local_spec(item: dict, alias: str, model: str) -> tuple[int, str] | None:
    if "local_max_tokens" not in item and "local_model_digest" not in item:
        return None
    if (
        alias != "ollama-local"
        or item.get("url") != _OLLAMA_API
        or item.get("headers", {}) != {}
        or item.get("free_verified") is not True
        or not _LOCAL_MODEL_NAME.fullmatch(model)
        or "cloud" in re.split(r"[.:_-]", model.casefold())
    ):
        raise ValueError(
            "Local model overrides require a verified loopback Ollama model name and empty headers"
        )
    max_tokens = item.get("local_max_tokens")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
        raise ValueError("local_max_tokens must be an integer between 1 and 8192")
    digest = item.get("local_model_digest")
    digest = digest.lower().removeprefix("sha256:") if isinstance(digest, str) else ""
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("local_model_digest must be a full SHA-256 digest")
    return max_tokens, digest


async def _post_pinned_ollama(
    route: _Route, payload: dict, timeout: float
) -> httpx.Response | None:
    if timeout <= 0:
        raise TimeoutError
    # An HTTP read timeout only bounds individual reads. Keep one deadline for
    # the inventory check and the complete local response.
    async with asyncio.timeout(timeout):
        async with httpx.AsyncClient(follow_redirects=False, trust_env=False) as client:
            inventory = await client.get(
                f"{_OLLAMA_ORIGIN}/api/tags", timeout=min(5, timeout)
            )
            inventory.raise_for_status()
            rows = inventory.json().get("models")
            if not isinstance(rows, list) or not any(
                isinstance(row, dict)
                and (row.get("name") or row.get("model")) == route.model
                and not (row.get("remote_model") or row.get("remote_host"))
                and type(row.get("size")) is int
                and row["size"] > 0
                and isinstance(row.get("digest"), str)
                and row["digest"].lower().removeprefix("sha256:")
                == route.local_model_digest
                for row in rows
            ):
                return None
            return await client.post(
                f"{_OLLAMA_API}/chat/completions", json=payload, timeout=timeout
            )


def _completion_timeout(timeout: float | None) -> float:
    if timeout is None:
        return DEFAULT_REQUEST_TIMEOUT
    if type(timeout) not in {int, float}:
        raise ValueError("timeout must be a number")
    if (isinstance(timeout, float) and not math.isfinite(timeout)) or timeout <= 0:
        raise ProviderExhausted(None, {"gateway": "deadline exceeded"})
    return min(timeout, MAX_REQUEST_TIMEOUT)


def _validate_offset(provider_offset: int) -> None:
    if type(provider_offset) is not int or provider_offset < 0:
        raise ValueError("provider_offset must be a nonnegative integer")


def _validate_output_limit(max_output_tokens: int | None) -> None:
    if max_output_tokens is not None and (
        type(max_output_tokens) is not int or max_output_tokens <= 0
    ):
        raise ValueError("max_output_tokens must be a positive integer")


def _tool_route_affinity(
    role: str,
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, str] | None:
    if (
        role not in {"coding", "probe"}
        or not messages
        or messages[-1].get("role") != "tool"
    ):
        return None
    assistant = next(
        (msg for msg in reversed(messages) if msg.get("role") != "tool"), {}
    )
    route = assistant.get("gateway_route")
    if (
        assistant.get("role") == "assistant"
        and assistant.get("tool_calls")
        and isinstance(route, (list, tuple))
        and len(route) == 2
        and all(isinstance(value, str) and value for value in route)
    ):
        return route[0], route[1]
    return None


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
            local = _local_spec(item, alias, model)
            if ((alias, model) not in allowed and local is None) or _paid_router(
                alias, model
            ):
                raise ValueError(
                    "The deployment contains a model outside the free allowlist"
                )
            if (alias, model) in configured:
                raise ValueError("Duplicate model deployment")
            local_input_bytes = item.get("local_input_bytes")
            if "local_input_bytes" in item and (
                local is None
                or type(local_input_bytes) is not int
                or not 8192 <= local_input_bytes <= MESSAGE_BYTES
            ):
                raise ValueError(
                    f"local_input_bytes requires a digest-pinned loopback route and an integer between 8192 and {MESSAGE_BYTES}"
                )
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
            limit = _LOCAL_ATTEMPT_MAX if local is not None else 60
            if (
                type(attempt_timeout) not in {int, float}
                or not 0 < attempt_timeout <= limit
            ):
                raise ValueError(
                    f"attempt_timeout must be above zero and at most {limit} seconds"
                )
            keyless = item.get("allow_keyless") is True
            if keyless and alias != "kilo-free":
                raise ValueError("allow_keyless applies only to the Kilo free route")
            protocol, max_tokens = (
                allowed[alias, model] if local is None else ("chat", local[0])
            )
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
                local[1] if local is not None else None,
                local_input_bytes,
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
        provider_offset: int = 0,
        after_provider: str | None = None,
        max_output_tokens: int | None = None,
    ) -> Completion:
        _validate_offset(provider_offset)
        _validate_output_limit(max_output_tokens)
        return self._complete(
            self.routes,
            role,
            messages,
            tools,
            timeout,
            exclusions=set(exclude),
            provider_offset=provider_offset,
            after_provider=after_provider,
            max_output_tokens=max_output_tokens,
        )

    def _quota_block(self, route: _Route, now: float) -> tuple[str, float] | None:
        """Read quota state while holding the router lock."""
        quota_cooldown = self._quota_cooldown.get(route.quota_key, 0)
        model_cooldown = self._model_cooldown.get((route.alias, route.model), 0)
        cooldown = max(quota_cooldown, model_cooldown)
        if cooldown > now:
            reason = (
                "rate limited" if quota_cooldown > now else "temporarily unavailable"
            )
            return reason, cooldown
        if route.quota_key in self._in_flight:
            return "request in progress", now + 1
        return None

    def _ordered_routes(self, routes, tools, exclusions, provider_offset, affinity):
        failures = {}
        groups: dict[tuple, list[_Route]] = {}
        degraded = set()
        deferred = []
        preferred = None
        with self._lock:
            now = time.time()
            for route in routes:
                if not route.free_verified:
                    failures[route.alias] = "free eligibility unverified"
                elif not route.url or not (
                    route.headers
                    or route.alias == "ollama-local"
                    or route.allow_keyless
                ):
                    failures[route.alias] = "missing endpoint or credentials"
                elif tools and not route.tools:
                    failures[route.alias] = "tool capability unverified"
                elif (route.alias, route.model) in exclusions:
                    # Leave a slot when rotating so retry offset and an exclusion
                    # don't both advance past the same provider.
                    degraded.add(route.quota_key)
                    groups.setdefault(route.quota_key, [])
                elif self._quota_block(route, now):
                    deferred.append(route)
                else:
                    group = groups.setdefault(route.quota_key, [])
                    if (route.alias, route.model) == affinity:
                        preferred = route
                    else:
                        group.append(route)

        if degraded and preferred is None and not any(groups.values()) and not deferred:
            raise ProviderExhausted(None, {"gateway": "all eligible models excluded"})
        accounts = list(groups)
        if preferred is not None:
            accounts.remove(preferred.quota_key)
        elif accounts:
            offset = provider_offset % len(accounts)
            accounts = accounts[offset:] + accounts[:offset]
        accounts.sort(key=degraded.__contains__)
        if preferred is not None:
            accounts.append(preferred.quota_key)
        ordered = [
            route
            for turn in zip_longest(*(groups[key] for key in accounts))
            for route in turn
            if route is not None
        ]
        if preferred is not None:
            ordered.insert(0, preferred)
        # A cooldown may expire while trying other providers. Check again if
        # those attempts did not produce a usable completion.
        return ordered + deferred, failures

    def _complete(self, *args, **kwargs):
        for event in self._events(*args, **kwargs):
            if isinstance(event, Completion):
                return event
        raise ProviderExhausted(None)

    def _events(
        self,
        routes,
        role,
        messages,
        tools,
        timeout,
        *,
        exclusions=frozenset(),
        provider_offset=0,
        after_provider=None,
        max_output_tokens=None,
    ):
        deadline = time.monotonic() + _completion_timeout(timeout)
        routes, failures = self._ordered_routes(
            routes,
            tools,
            exclusions,
            provider_offset,
            _tool_route_affinity(role, messages),
        )
        if after_provider:
            # Several public aliases can share a credential and its quota.
            previous = {r.quota_key for r in self.routes if r.alias == after_provider}
            routes.sort(key=lambda r: r.quota_key in previous)
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
                failures["gateway"] = "deadline exceeded"
                break
            with self._lock:
                blocked = self._quota_block(route, time.time())
                if blocked is None:
                    self._in_flight.add(route.quota_key)
            if blocked is not None:
                reason, retry_at = blocked
                failures[route.alias] = reason
                retry_times.append(retry_at)
                continue

            try:
                max_tokens = min(
                    route.max_tokens, max_output_tokens or route.max_tokens
                )
                if route.protocol == "gemini":
                    url = f"{route.url}/models/{quote(route.model, safe='')}:generateContent"
                    payload = _gemini_payload(route, messages, tools, max_tokens)
                else:
                    url = f"{route.url}/chat/completions"
                    route_messages = messages
                    if route.local_input_bytes is not None:
                        route_messages = compact_messages(
                            messages,
                            max_bytes=route.local_input_bytes,
                            role=role,
                            review_windows=True,
                        )
                    payload = _chat_payload(route, route_messages, tools, max_tokens)
                    if (
                        route.local_input_bytes is not None
                        and wire_size(payload["messages"]) > route.local_input_bytes
                    ):
                        failures[route.alias] = "local input limit exceeded"
                        continue
                logger.info(
                    "Provider attempt provider=%s configured_model=%s",
                    route.alias,
                    route.model,
                )
                yield {"provider": route.alias, "model": route.model}
                if route.local_model_digest is not None:
                    response = asyncio.run(
                        _post_pinned_ollama(
                            route,
                            payload,
                            min(attempt_timeout, deadline - time.monotonic()),
                        )
                    )
                    if response is None:
                        failures[route.alias] = "model unavailable"
                        continue
                else:
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
                    logger.info("Provider deferred provider=%s configured_model=%s status=%d",
                                route.alias, route.model, response.status_code)
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
                logger.info(
                    "Provider accepted provider=%s configured_model=%s",
                    route.alias,
                    route.model,
                )
                yield result
                return
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                logger.info("Provider rejected provider=%s configured_model=%s status=%d",
                            route.alias, route.model, status)
                failures[route.alias] = {
                    401: "invalid or missing credentials",
                    403: "account or model ineligible",
                    404: "model unavailable",
                }.get(status, f"upstream HTTP {status}")
                if status is not None and status >= 500:
                    retry_times.append(self._defer_model(key))
            except (httpx.TimeoutException, TimeoutError):
                logger.info("Provider timed out provider=%s configured_model=%s", route.alias, route.model)
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
    timeout: float | None = Field(
        default=None, gt=0, le=MAX_REQUEST_TIMEOUT, strict=True
    )
    exclude: list[tuple[str, str]] = Field(default_factory=list, max_length=1000)
    provider_offset: int = Field(default=0, ge=0, strict=True)
    after_provider: str | None = Field(default=None, max_length=128)
    max_output_tokens: int | None = Field(default=None, gt=0, strict=True)


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

    @app.post("/complete/observed")
    def observed(payload: _Request, authorization: str = Header(default="")):
        if not hmac.compare_digest(authorization.encode(), expected):
            raise HTTPException(401, "Gateway authentication required")

        def stream():
            events = router._events(
                router.routes,
                payload.role,
                payload.messages,
                payload.tools,
                payload.timeout,
                exclusions=set(payload.exclude),
                provider_offset=payload.provider_offset,
                after_provider=payload.after_provider,
                max_output_tokens=payload.max_output_tokens,
            )
            try:
                for event in events:
                    if isinstance(event, Completion):
                        yield json.dumps({"completion": asdict(event)}) + "\n"
                    else:
                        yield json.dumps({"attempt": event}) + "\n"
            except ProviderExhausted as error:
                yield (
                    json.dumps(
                        {
                            "exhausted": {
                                "retry_at": error.retry_at,
                                "failures": error.failures,
                            }
                        }
                    )
                    + "\n"
                )
            finally:
                events.close()

        return StreamingResponse(stream(), media_type="application/x-ndjson")

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
        provider_offset: int = 0,
        after_provider: str | None = None,
        max_output_tokens: int | None = None,
        on_attempt=None,
    ) -> Completion:
        _validate_offset(provider_offset)
        _validate_output_limit(max_output_tokens)
        limit = _completion_timeout(timeout)
        # Leave time for the gateway to serialize and return an exhausted chain
        # without extending an explicit task deadline.
        headroom = min(1, limit / 10)
        gateway_timeout = limit - headroom if timeout is not None else None
        receive_timeout = limit if timeout is not None else limit + headroom
        try:
            request = {
                "headers": self.headers,
                "timeout": httpx.Timeout(
                    receive_timeout,
                    connect=min(5, receive_timeout),
                    pool=min(5, receive_timeout),
                    write=min(10, receive_timeout),
                ),
                "json": {
                    "role": role,
                    "messages": list(messages),
                    "tools": list(tools),
                    "timeout": gateway_timeout,
                    "exclude": list(exclude),
                    "provider_offset": provider_offset,
                    **({"after_provider": after_provider} if after_provider else {}),
                    **(
                        {"max_output_tokens": max_output_tokens}
                        if max_output_tokens is not None
                        else {}
                    ),
                },
            }
            if on_attempt is None:
                response = self.client.post(self.url, **request)
                if response.status_code == 503:
                    details = response.json()["detail"]
                    raise ProviderExhausted(details["retry_at"], details["failures"])
                response.raise_for_status()
                result = response.json()
            else:
                with self.client.stream(
                    "POST", self.url + "/observed", **request
                ) as response:
                    if response.status_code in {404, 405}:
                        # Rollouts can briefly pair a newer worker with the previous gateway.
                        return self.complete(
                            role=role,
                            messages=messages,
                            tools=tools,
                            timeout=timeout,
                            exclude=exclude,
                            provider_offset=provider_offset,
                            after_provider=after_provider,
                            max_output_tokens=max_output_tokens,
                        )
                    response.raise_for_status()
                    for line in response.iter_lines():
                        event = json.loads(line)
                        if "attempt" in event:
                            on_attempt(
                                event["attempt"]["provider"], event["attempt"]["model"]
                            )
                        elif "exhausted" in event:
                            raise ProviderExhausted(**event["exhausted"])
                        else:
                            result = event["completion"]
                            break
                    else:
                        raise ValueError("Incomplete gateway response")
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

    log_config = deepcopy(uvicorn.config.LOGGING_CONFIG)
    log_config["loggers"][logger.name] = {
        "handlers": ["default"],
        "level": "INFO",
        "propagate": False,
    }
    uvicorn.run(
        create_model_gateway(router, os.environ["MODEL_GATEWAY_TOKEN"]),
        host=args.host,
        port=args.port,
        log_config=log_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Direct LLM callers for discussion phases and task claiming (no container)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import TYPE_CHECKING, Any

import anthropic

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

from ..tokens import LLMResponse, TokenUsage

if TYPE_CHECKING:
    from ..protocols import LLMCaller

log = logging.getLogger(__name__)

# Provider strings that map to the Anthropic caller. Empty/unset defaults to
# Anthropic (historical behaviour); anything else that is not "openai" is
# rejected so a typo'd provider fails loudly here instead of silently producing
# an Anthropic caller that 401s at call time.
_ANTHROPIC_PROVIDER_ALIASES = {"", "anthropic", "claude"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except (TypeError, ValueError):
        return default


def _build_client_kwargs(base_kwargs: dict[str, Any], *, timeout_env: str, timeout_default: int) -> dict[str, Any]:
    kwargs = dict(base_kwargs)
    timeout_sec = _env_int(timeout_env, timeout_default)
    if timeout_sec:
        kwargs["timeout"] = timeout_sec
    kwargs["max_retries"] = max(0, _env_int("KSI_DIRECT_LLM_MAX_RETRIES", 2))
    return kwargs


def _usage_value(obj: Any, key: str, default: int = 0) -> int:
    if obj is None:
        return default
    if isinstance(obj, dict):
        value = obj.get(key, default)
    else:
        value = getattr(obj, key, default)
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _usage_child(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _anthropic_tool_input(response: Any, tool_name: str | None) -> dict[str, Any] | None:
    """Extract the forced tool_use block's parsed ``input`` from a Messages
    response. Returns the dict (already parsed by the SDK) or ``None`` when no
    matching tool_use block is present (e.g. the model declined)."""
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) != "tool_use":
            continue
        if tool_name is not None and getattr(block, "name", None) != tool_name:
            continue
        payload = getattr(block, "input", None)
        if isinstance(payload, dict):
            return payload
    return None


def _is_openai_reasoning_model(model: str) -> bool:
    """Reasoning-family models (gpt-5*, o-series) reject temperature/seed in the
    Responses API."""
    m = str(model or "").lower()
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")


class AnthropicLLMCaller:
    """Thin Anthropic SDK wrapper for direct LLM calls."""

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        api_key: str | None = None,
        temperature: float | None = None,
    ):
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        # Explicitly pass api_key so the SDK doesn't try to use an OAuth
        # token from the environment (Messages API requires an API key).
        api_key = (api_key or os.environ.get("ANTHROPIC_API_KEY") or "").strip() or None
        kwargs = _build_client_kwargs({}, timeout_env="KSI_ANTHROPIC_TIMEOUT_SEC", timeout_default=120)
        self._client = anthropic.Anthropic(api_key=api_key, **kwargs) if api_key else anthropic.Anthropic(**kwargs)

    # Anthropic Messages API supports JSON-schema-constrained output via
    # tool-forcing on every SDK version this repo has shipped: define a single
    # tool whose ``input_schema`` is the requested schema, then force it with
    # ``tool_choice={"type": "tool", ...}``. The installed SDK is
    # anthropic==0.97.0 (see uv.lock); that release does NOT expose a stable
    # top-level ``response_format`` / structured-outputs parameter for the
    # Messages API, so tool-forcing is the portable mechanism here. The model's
    # ``tool_use`` block ``input`` is already a parsed dict matching the schema.
    supports_json_schema = True

    def call(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict[str, Any] | None = None,
        cache_prefix: str | None = None,
        cache_blocks: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        # Per-call model override (e.g. distillation phase) falls back to the
        # caller's default model when not set.
        model_override = kwargs.get("model") or self._model
        # Per-call max_tokens override. Distillation prompts ask for nested
        # JSON with 7 insight lists; the default 4096 truncates the closing
        # braces and makes the parser raise "unterminated JSON object" every
        # gen (observed 7/7 cross-task distills failed in the 50t/10g audit).
        max_tokens_override = kwargs.get("max_tokens") or self._max_tokens
        temperature_override = kwargs.get("temperature", self._temperature)

        # Block-form `system` with `cache_control: ephemeral` so the system
        # prompt is cache-written on the first call and cache-read on every
        # subsequent call with the same system text. Distillation, per-task
        # insight extraction, lesson extraction, and task-claiming all share
        # a stable system prompt across many calls per generation; caching
        # it is the same shape of win as the direct-adapter caching fix
        # (e.g. `anthropic_direct_forum.ts`). The user
        # message content varies per call (per task / per bucket) and has
        # no cross-call prefix stability, so it stays as a plain string.
        cached_system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        # A ``cache_prefix`` is a large, cross-call-stable block that some callers
        # (cross-task distill: the windowed forum history re-sent to every target)
        # want cache-READ rather than re-paid as plain user tokens. Send it as a
        # leading user block carrying its own ``cache_control`` breakpoint; the
        # varying ``user`` text follows as a plain block after the breakpoint so
        # only the stable prefix is cached. Haiku's minimum
        # cacheable prefix is 4096 tokens — the shared history (~120K) clears it.
        # ``cache_blocks`` is an APPEND-ONLY list of stable
        # blocks (TB2's per-step shell history): each turn appends one more,
        # earlier blocks are byte-identical. Marking the LAST block with
        # cache_control places a moving breakpoint whose cached prefix is a
        # superset of the previous turn's — so every turn after the first
        # cache-READS the accumulated history (the only way to clear Haiku's
        # 4096-token floor on the TB2 path, whose stable content is ~1670).
        blocks = [b for b in (cache_blocks or []) if b]
        if blocks:
            user_content: Any = [{"type": "text", "text": b} for b in blocks]
            user_content[-1]["cache_control"] = {"type": "ephemeral"}
            user_content.append({"type": "text", "text": user})
        elif cache_prefix:
            user_content = [
                {"type": "text", "text": cache_prefix, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": user},
            ]
        else:
            user_content = user
        request_kwargs: dict[str, Any] = {
            "model": model_override,
            "max_tokens": max_tokens_override,
            "system": cached_system,
            "messages": [{"role": "user", "content": user_content}],
        }
        if temperature_override is not None:
            request_kwargs["temperature"] = float(temperature_override)

        tool_name: str | None = None
        if json_schema is not None:
            tool_name = str(json_schema.get("name") or "structured_output")
            request_kwargs["tools"] = [
                {
                    "name": tool_name,
                    "description": "Emit the result as a single structured object matching the schema.",
                    "input_schema": json_schema["schema"],
                }
            ]
            request_kwargs["tool_choice"] = {"type": "tool", "name": tool_name}

        response = self._client.messages.create(**request_kwargs)
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_input_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        )

        if json_schema is not None:
            parsed = _anthropic_tool_input(response, tool_name)
            # Serialize the tool input back to text so the legacy
            # (system, user) -> str callable contract still holds; the text is
            # guaranteed-valid JSON, and callers that want the dict take it
            # from ``LLMResponse.parsed``.
            text = json.dumps(parsed) if parsed is not None else ""
            return LLMResponse(text=text, usage=usage, parsed=parsed)

        text = response.content[0].text if response.content else ""
        return LLMResponse(text=text, usage=usage)


class OpenAILLMCaller:
    """Thin OpenAI Responses API wrapper for direct LLM calls."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.4-mini",
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ):
        if OpenAI is None:
            raise ImportError("openai package is required for OpenAILLMCaller. Install it with: pip install openai")
        self._model = model
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort or os.environ.get("REASONING_EFFORT", "").strip() or None
        self._temperature = temperature
        self._seed = seed
        api_key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
        kwargs = _build_client_kwargs({}, timeout_env="KSI_OPENAI_TIMEOUT_SEC", timeout_default=120)
        self._client = OpenAI(api_key=api_key, **kwargs) if api_key else OpenAI(**kwargs)

    # OpenAI's Responses API supports JSON-schema-constrained output via
    # ``text={"format": {"type": "json_schema", ...}}``. The installed SDK is
    # openai==2.32.0 (see uv.lock), which accepts that parameter and returns
    # the constrained JSON in ``output_text`` (and ``output_parsed`` when the
    # SDK can parse it).
    supports_json_schema = True

    def call(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict[str, Any] | None = None,
        cache_prefix: str | None = None,
        cache_blocks: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        # Per-call model override (e.g. distillation phase) falls back to the
        # caller's default model when not set.
        model_override = kwargs.get("model") or self._model
        # Per-call max_output_tokens override (matches AnthropicLLMCaller).
        max_tokens_override = kwargs.get("max_tokens") or self._max_tokens
        # A ``cache_prefix`` (see AnthropicLLMCaller.call) is a large, stable
        # block re-sent across calls. OpenAI auto-caches the common input-token
        # prefix, so placing the shared history ahead of the varying user text
        # (and pinning ``prompt_cache_key`` to system+prefix below) makes every
        # target that shares the history read one cache shard.
        # ``cache_blocks`` is an append-only stable list
        # (TB2 per-step history). OpenAI auto-caches the common input-token
        # prefix, so placing the growing blocks ahead of the varying tail lets
        # each turn cache-read the accumulated history. The routing key stays
        # system-only (the blocks grow every turn — folding them in would send
        # each turn to a fresh shard and miss).
        blocks = [b for b in (cache_blocks or []) if b]
        if blocks:
            user_content: list[dict[str, Any]] = [{"type": "input_text", "text": b} for b in blocks]
            user_content.append({"type": "input_text", "text": user})
        elif cache_prefix:
            user_content = [
                {"type": "input_text", "text": cache_prefix},
                {"type": "input_text", "text": user},
            ]
        else:
            user_content = [{"type": "input_text", "text": user}]
        request: dict[str, Any] = {
            "model": model_override,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system}],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "max_output_tokens": max_tokens_override,
            # OpenAI's prompt cache fires automatically at >=1024 input tokens,
            # but routing is keyed off `prompt_cache_key` (or a hash of the
            # caller's user/org id when absent). Without a stable key, calls
            # from the same orchestrator can land on different cache shards
            # and miss even when the system prefix is identical. Hashing the
            # system prompt gives every distinct stable prefix its own pinned
            # routing key — same shape of fix as the Anthropic block-form
            # `cache_control` change in `AnthropicLLMCaller.call`.
            # User content is excluded from the hash because it varies per
            # call (per task / per bucket) and would defeat the routing pin.
            # A cache_prefix IS stable across the calls that share it, so it is
            # folded into the key (NUL-separated to avoid boundary collisions)
            # so those calls pin to one shard.
            "prompt_cache_key": hashlib.sha256(
                (system + "\x00" + cache_prefix if cache_prefix else system).encode("utf-8")
            ).hexdigest()[:32],
        }
        if self._reasoning_effort and "gpt-5" in model_override:
            request["reasoning"] = {"effort": self._reasoning_effort}
        # Reasoning-family models (gpt-5*, o-series) reject `temperature`;
        # only set it for chat-completion-style models. `seed` is never sent:
        # the Responses API has no such parameter (openai==2.32.0
        # ``Responses.create`` rejects the kwarg with a client-side TypeError,
        # which upstream callers swallow into silent knowledge-feature loss).
        temperature_override = kwargs.get("temperature", self._temperature)
        if not _is_openai_reasoning_model(model_override):
            if temperature_override is not None:
                request["temperature"] = float(temperature_override)

        if json_schema is not None:
            # Responses API json_schema format requires an explicit, non-null
            # ``schema`` and a ``name``. ``strict`` must be an explicit False:
            # an omitted ``strict`` is treated as strict mode, which 400-rejects
            # any schema whose objects lack ``additionalProperties: false`` —
            # the distill schemas deliberately allow extra keys
            # and optional fields, so strict mode can never accept them.
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": str(json_schema.get("name") or "structured_output"),
                    "schema": json_schema["schema"],
                    "strict": False,
                }
            }

        response = self._client.responses.create(**request)
        text = getattr(response, "output_text", "") or ""
        usage_obj = getattr(response, "usage", None)
        input_tokens = _usage_value(usage_obj, "input_tokens")
        output_tokens = _usage_value(usage_obj, "output_tokens")
        input_details = _usage_child(usage_obj, "input_tokens_details") or _usage_child(
            usage_obj, "prompt_tokens_details"
        )
        cached_input_tokens = (
            _usage_value(input_details, "cached_tokens")
            or _usage_value(input_details, "cache_read_input_tokens")
            or _usage_value(usage_obj, "cache_read_input_tokens")
        )
        usage = TokenUsage(
            input_tokens=max(0, input_tokens - cached_input_tokens),
            output_tokens=output_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_input_tokens,
        )

        if json_schema is not None:
            parsed = getattr(response, "output_parsed", None)
            if not isinstance(parsed, dict):
                try:
                    loaded = json.loads(text) if text else None
                except (ValueError, TypeError):
                    loaded = None
                parsed = loaded if isinstance(loaded, dict) else None
            return LLMResponse(text=text, usage=usage, parsed=parsed)

        return LLMResponse(text=text, usage=usage)


def build_llm_caller(
    *,
    provider: str,
    model: str,
    max_tokens: int = 4096,
    reasoning_effort: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> "LLMCaller":
    provider_norm = (provider or "").strip().lower()
    if provider_norm == "openai":
        # The Responses API has no `seed` parameter; the caller accepts it for
        # signature symmetry but never sends it. Warn (mirroring the Anthropic
        # branch) when a non-zero seed is passed so a caller expecting
        # reproducibility isn't misled.
        if seed:
            log.warning(
                "build_llm_caller: seed=%r is not supported by the OpenAI Responses API "
                "and will be ignored — host-side distill/reflection calls are not reproducible.",
                seed,
            )
        return OpenAILLMCaller(
            model=model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            api_key=api_key,
            temperature=temperature,
            seed=seed,
        )
    if provider_norm not in _ANTHROPIC_PROVIDER_ALIASES:
        raise ValueError(
            f"Unknown LLM provider {provider!r}; supported: 'anthropic' "
            f"(or '', 'claude') and 'openai'. To use a custom provider, "
            f"construct your own caller and pass it directly rather than via "
            f"build_llm_caller."
        )
    # Anthropic Messages API supports temperature but not seed; seed is accepted
    # for caller-side symmetry but is NOT forwarded (silently ignored on the
    # wire). Warn (not debug) when a caller passes a non-zero seed so anyone
    # expecting reproducibility on the default Anthropic stack isn't misled.
    if seed:
        log.warning(
            "build_llm_caller: seed=%r is not supported by the Anthropic Messages API "
            "and will be ignored — host-side distill/reflection calls are not "
            "reproducible. (No provider currently receives a usable seed.)",
            seed,
        )
    return AnthropicLLMCaller(
        model=model,
        max_tokens=max_tokens,
        api_key=api_key,
        temperature=temperature,
    )

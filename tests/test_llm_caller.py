"""Tests for direct LLM caller (forum + task claiming)."""

import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from ksi.runtime.llm import AnthropicLLMCaller, OpenAILLMCaller
from ksi.tokens import LLMResponse


def test_call_returns_text_and_usage():
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello world")]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5

    with patch("ksi.runtime.llm.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        caller = AnthropicLLMCaller(model="claude-sonnet-4-20250514")
        resp = caller.call(system="You are helpful", user="Hi")

        assert isinstance(resp, LLMResponse)
        assert resp.text == "Hello world"
        assert resp.parsed is None
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5


def test_call_json_mode():
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"tasks": ["t1", "t2"]}')]
    mock_response.usage.input_tokens = 20
    mock_response.usage.output_tokens = 10

    with patch("ksi.runtime.llm.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        caller = AnthropicLLMCaller(model="claude-sonnet-4-20250514")
        resp = caller.call(system="Return JSON", user="Pick tasks")
        parsed = json.loads(resp.text)
        assert parsed["tasks"] == ["t1", "t2"]


def test_anthropic_client_uses_sdk_timeout_and_retries(monkeypatch):
    monkeypatch.setenv("KSI_ANTHROPIC_TIMEOUT_SEC", "17")
    monkeypatch.setenv("KSI_DIRECT_LLM_MAX_RETRIES", "4")

    with patch("ksi.runtime.llm.anthropic") as mock_anthropic:
        AnthropicLLMCaller(model="claude-sonnet-4-20250514")

    assert mock_anthropic.Anthropic.call_args.kwargs["timeout"] == 17
    assert mock_anthropic.Anthropic.call_args.kwargs["max_retries"] == 4


def test_openai_client_uses_sdk_timeout_and_retries(monkeypatch):
    monkeypatch.setenv("KSI_OPENAI_TIMEOUT_SEC", "23")
    monkeypatch.setenv("KSI_DIRECT_LLM_MAX_RETRIES", "3")

    with patch("ksi.runtime.llm.OpenAI") as mock_openai:
        OpenAILLMCaller(model="gpt-5.4-mini")

    assert mock_openai.call_args.kwargs["timeout"] == 23
    assert mock_openai.call_args.kwargs["max_retries"] == 3


def test_anthropic_system_is_block_form_with_cache_control():
    """System must be sent as a block array with `cache_control: ephemeral`.

    Distillation, per-task insight extraction, lesson extraction, and
    task-claiming all share a stable system prompt across many calls per
    generation. Sending the system as a plain string opts out of prompt
    caching entirely (Anthropic only honors cache_control on block-form
    content) — the same pathology that produced cache_read=0 on the forum
    direct adapter pre-fix.
    """
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="ok")]
    mock_response.usage.input_tokens = 1
    mock_response.usage.output_tokens = 1

    with patch("ksi.runtime.llm.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        caller = AnthropicLLMCaller(model="claude-sonnet-4-20250514")
        caller.call(system="STABLE SYSTEM PROMPT", user="varying user")

    call_kwargs = mock_client.messages.create.call_args.kwargs
    system_arg = call_kwargs["system"]
    assert isinstance(system_arg, list), (
        f"system must be block-form to opt into prompt caching; got {type(system_arg).__name__}"
    )
    assert len(system_arg) == 1
    assert system_arg[0]["type"] == "text"
    assert system_arg[0]["text"] == "STABLE SYSTEM PROMPT"
    assert system_arg[0]["cache_control"] == {"type": "ephemeral"}


def test_openai_passes_prompt_cache_key_pinned_to_system():
    """`responses.create` must receive `prompt_cache_key` derived from the
    system prompt so identical prefixes route to the same cache shard.

    OpenAI's prompt cache fires automatically at >=1024 input tokens, but
    without a stable `prompt_cache_key`, calls land on different shards and
    miss even when the system prefix is identical — same gap shape as the
    Anthropic block-form `cache_control` fix in PR #546. Verified pre-fix
    against `arc2_openai_audit_pr508_repro_3runner_runtime.sqlite`:
    distill_per_task / distill_cross_task / lesson_extraction /
    task_reflection all show cache_read_tokens = 0.
    """
    mock_response = MagicMock()
    mock_response.output_text = "ok"
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 1
    mock_response.usage.output_tokens = 1
    mock_response.usage.input_tokens_details = None

    with patch("ksi.runtime.llm.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_response
        mock_openai.return_value = mock_client

        caller = OpenAILLMCaller(model="gpt-5.4-mini")
        caller.call(system="STABLE SYSTEM PROMPT", user="varying user A")
        caller.call(system="STABLE SYSTEM PROMPT", user="varying user B")
        caller.call(system="DIFFERENT SYSTEM", user="varying user C")

    calls = mock_client.responses.create.call_args_list
    assert len(calls) == 3

    expected_stable = hashlib.sha256(b"STABLE SYSTEM PROMPT").hexdigest()[:32]
    expected_diff = hashlib.sha256(b"DIFFERENT SYSTEM").hexdigest()[:32]

    keys = [c.kwargs.get("prompt_cache_key") for c in calls]
    assert keys[0] is not None, "prompt_cache_key must be set to pin cache routing"
    assert keys[0] == expected_stable
    assert keys[1] == expected_stable, "identical system → identical cache key"
    assert keys[2] == expected_diff, "different system → different cache key"
    assert keys[0] != keys[2]


def test_openai_prompt_cache_key_excludes_user_content():
    """`prompt_cache_key` must NOT incorporate the user message. User content
    varies per call (per task, per bucket) — including it in the key would
    defeat the routing pin and re-introduce the cache_read=0 pathology.
    """
    mock_response = MagicMock()
    mock_response.output_text = "ok"
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 1
    mock_response.usage.output_tokens = 1
    mock_response.usage.input_tokens_details = None

    with patch("ksi.runtime.llm.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_response
        mock_openai.return_value = mock_client

        caller = OpenAILLMCaller(model="gpt-5.4-mini")
        caller.call(system="SAME SYSTEM", user="user one")
        caller.call(system="SAME SYSTEM", user="user two — much longer payload here")

    keys = [c.kwargs.get("prompt_cache_key") for c in mock_client.responses.create.call_args_list]
    assert keys[0] == keys[1], (
        "prompt_cache_key must be a function of the system prompt only; "
        "varying user content must not change the routing key"
    )


def test_anthropic_cache_prefix_adds_cached_user_block():
    """A ``cache_prefix`` is sent as a leading user content block with
    ``cache_control: ephemeral``, and the varying ``user`` text follows it as a
    plain block. This caches system + shared prefix (issue #1252 item 3): the
    cross-task distill re-sends the SAME windowed forum history to every target,
    so it must live in a cache-read prefix, not the per-target user string."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="ok")]
    mock_response.usage.input_tokens = 1
    mock_response.usage.output_tokens = 1

    with patch("ksi.runtime.llm.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        caller = AnthropicLLMCaller(model="claude-haiku-4-5")
        caller.call(system="SYS", user="TARGET SUFFIX", cache_prefix="SHARED HISTORY PREFIX")

    content = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert isinstance(content, list) and len(content) == 2
    assert content[0]["text"] == "SHARED HISTORY PREFIX"
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    # The varying suffix must NOT carry cache_control (it changes per target).
    assert content[1]["text"] == "TARGET SUFFIX"
    assert "cache_control" not in content[1]


def test_anthropic_no_cache_prefix_keeps_plain_user_string():
    """Without cache_prefix, the user message stays a plain string (unchanged
    behavior for every existing caller)."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="ok")]
    mock_response.usage.input_tokens = 1
    mock_response.usage.output_tokens = 1

    with patch("ksi.runtime.llm.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        AnthropicLLMCaller(model="claude-haiku-4-5").call(system="SYS", user="U")

    assert mock_client.messages.create.call_args.kwargs["messages"] == [{"role": "user", "content": "U"}]


def test_openai_cache_prefix_splits_input_and_keys_cache():
    """A ``cache_prefix`` splits the OpenAI user content into a stable-prefix
    block + varying-suffix block, and pins ``prompt_cache_key`` to
    system+prefix so every target sharing the history routes to one shard
    (issue #1252 item 3)."""
    mock_response = MagicMock()
    mock_response.output_text = "ok"
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 1
    mock_response.usage.output_tokens = 1
    mock_response.usage.input_tokens_details = None

    with patch("ksi.runtime.llm.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_response
        mock_openai.return_value = mock_client

        caller = OpenAILLMCaller(model="gpt-5.4-mini")
        caller.call(system="SYS", user="TARGET A", cache_prefix="SHARED HISTORY")
        caller.call(system="SYS", user="TARGET B", cache_prefix="SHARED HISTORY")

    calls = mock_client.responses.create.call_args_list
    # User message content is split into stable-prefix + varying-suffix blocks.
    user_msg = calls[0].kwargs["input"][1]
    assert user_msg["role"] == "user"
    assert [b["text"] for b in user_msg["content"]] == ["SHARED HISTORY", "TARGET A"]
    # Same system+prefix → same routing key across targets; excludes the suffix.
    keys = [c.kwargs.get("prompt_cache_key") for c in calls]
    expected = hashlib.sha256(b"SYS\x00SHARED HISTORY").hexdigest()[:32]
    assert keys[0] == keys[1] == expected


def test_anthropic_cache_blocks_moving_breakpoint():
    """``cache_blocks`` become separate user text blocks with cache_control on
    the LAST block, and the varying ``user`` follows as a plain block. Across
    turns the earlier blocks are byte-identical, so the previous turn's cached
    prefix is a subset of this turn's — the moving-breakpoint pattern that lets
    TB2 cache-read accumulated history on Haiku (issue #1252 item 1)."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="ok")]
    mock_response.usage.input_tokens = 1
    mock_response.usage.output_tokens = 1

    with patch("ksi.runtime.llm.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        caller = AnthropicLLMCaller(model="claude-haiku-4-5")
        # Turn 1: meta + one history step. Turn 2 appends a second step.
        caller.call("SYS", "TAIL-1", cache_blocks=["META", "STEP-1"])
        caller.call("SYS", "TAIL-2", cache_blocks=["META", "STEP-1", "STEP-2"])

    calls = mock_client.messages.create.call_args_list
    c1 = calls[0].kwargs["messages"][0]["content"]
    assert [b["text"] for b in c1] == ["META", "STEP-1", "TAIL-1"]
    # cache_control only on the last STABLE block (STEP-1), not the tail.
    assert c1[1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in c1[0]
    assert "cache_control" not in c1[2]

    c2 = calls[1].kwargs["messages"][0]["content"]
    assert [b["text"] for b in c2] == ["META", "STEP-1", "STEP-2", "TAIL-2"]
    # Breakpoint moved to the new last stable block (STEP-2); STEP-1 no longer
    # carries it, keeping the turn-1 cached prefix (META, STEP-1) byte-stable.
    assert "cache_control" not in c2[1]
    assert c2[2]["cache_control"] == {"type": "ephemeral"}


def test_openai_cache_blocks_appended_before_tail_key_system_only():
    """OpenAI: cache_blocks render as leading input blocks before the varying
    tail, and the routing key stays system-only so growing history across turns
    still lands on one shard (issue #1252 item 1)."""
    mock_response = MagicMock()
    mock_response.output_text = "ok"
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 1
    mock_response.usage.output_tokens = 1
    mock_response.usage.input_tokens_details = None

    with patch("ksi.runtime.llm.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_response
        mock_openai.return_value = mock_client

        caller = OpenAILLMCaller(model="gpt-5.4-mini")
        caller.call("SYS", "TAIL-1", cache_blocks=["META", "STEP-1"])
        caller.call("SYS", "TAIL-2", cache_blocks=["META", "STEP-1", "STEP-2"])

    calls = mock_client.responses.create.call_args_list
    assert [b["text"] for b in calls[0].kwargs["input"][1]["content"]] == ["META", "STEP-1", "TAIL-1"]
    keys = [c.kwargs.get("prompt_cache_key") for c in calls]
    assert keys[0] == keys[1] == hashlib.sha256(b"SYS").hexdigest()[:32]


_DEMO_SCHEMA = {
    "name": "demo",
    "schema": {
        "type": "object",
        "properties": {"a": {"type": "array", "items": {"type": "string"}}},
        "additionalProperties": True,
    },
}


def test_anthropic_json_schema_uses_tool_forcing_and_returns_parsed():
    """With json_schema, Anthropic path forces a single tool and surfaces the
    parsed tool_use input on ``LLMResponse.parsed``."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "demo"
    tool_block.input = {"a": ["x", "y"]}

    mock_response = MagicMock()
    mock_response.content = [tool_block]
    mock_response.usage.input_tokens = 12
    mock_response.usage.output_tokens = 8

    with patch("ksi.runtime.llm.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        caller = AnthropicLLMCaller(model="claude-sonnet-4-20250514")
        result = caller.call(system="sys", user="usr", json_schema=_DEMO_SCHEMA)

    assert isinstance(result, LLMResponse)
    assert result.parsed == {"a": ["x", "y"]}
    assert json.loads(result.text) == {"a": ["x", "y"]}
    assert result.usage.input_tokens == 12

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "demo"}
    assert len(call_kwargs["tools"]) == 1
    assert call_kwargs["tools"][0]["name"] == "demo"
    assert call_kwargs["tools"][0]["input_schema"] == _DEMO_SCHEMA["schema"]


def test_anthropic_without_json_schema_returns_unparsed_response():
    """Non-structured path: omitting json_schema yields an LLMResponse whose
    ``parsed`` is None and sends no tools."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="plain text")]
    mock_response.usage.input_tokens = 3
    mock_response.usage.output_tokens = 2

    with patch("ksi.runtime.llm.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        caller = AnthropicLLMCaller(model="claude-sonnet-4-20250514")
        result = caller.call(system="sys", user="usr")

    assert isinstance(result, LLMResponse)
    assert result.text == "plain text"
    assert result.parsed is None
    assert "tools" not in mock_client.messages.create.call_args.kwargs


def test_openai_json_schema_sets_text_format_and_returns_parsed():
    """With json_schema, OpenAI path sets text.format=json_schema and returns
    the parsed dict (from output_parsed when available)."""
    mock_response = MagicMock()
    mock_response.output_text = '{"a": ["p", "q"]}'
    mock_response.output_parsed = {"a": ["p", "q"]}
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 5
    mock_response.usage.output_tokens = 4
    mock_response.usage.input_tokens_details = None

    with patch("ksi.runtime.llm.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_response
        mock_openai.return_value = mock_client

        caller = OpenAILLMCaller(model="gpt-5.4-mini")
        result = caller.call(system="sys", user="usr", json_schema=_DEMO_SCHEMA)

    assert isinstance(result, LLMResponse)
    assert result.parsed == {"a": ["p", "q"]}

    fmt = mock_client.responses.create.call_args.kwargs["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["name"] == "demo"
    assert fmt["schema"] == _DEMO_SCHEMA["schema"]
    # `strict` must be an explicit False: the Responses API treats an omitted
    # `strict` as strict mode and 400-rejects any schema whose objects lack
    # `additionalProperties: false` — which the distill schemas deliberately
    # use. Verified live against gpt-5.4-mini (2026-06-11).
    assert fmt["strict"] is False


def test_openai_json_schema_falls_back_to_output_text_parse():
    """When the SDK doesn't surface output_parsed, the wrapper parses
    output_text itself."""
    mock_response = MagicMock()
    mock_response.output_text = '{"a": ["only", "text"]}'
    mock_response.output_parsed = None
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 5
    mock_response.usage.output_tokens = 4
    mock_response.usage.input_tokens_details = None

    with patch("ksi.runtime.llm.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_response
        mock_openai.return_value = mock_client

        caller = OpenAILLMCaller(model="gpt-5.4-mini")
        result = caller.call(system="s", user="u", json_schema=_DEMO_SCHEMA)

    assert isinstance(result, LLMResponse)
    assert result.parsed == {"a": ["only", "text"]}


def test_callers_advertise_json_schema_support():
    assert AnthropicLLMCaller.supports_json_schema is True
    assert OpenAILLMCaller.supports_json_schema is True


def test_call_returns_llmresponse_on_both_paths():
    """Contract: ``call`` returns a single ``LLMResponse`` type on BOTH the
    structured (``json_schema`` requested -> ``parsed`` populated) and the
    non-structured (``parsed is None``) paths — never a variable-arity tuple.
    Exercised against the OpenAI caller; the Anthropic caller is pinned by
    ``test_anthropic_json_schema_uses_tool_forcing_and_returns_parsed`` and
    ``test_anthropic_without_json_schema_returns_unparsed_response``.
    """
    mock_response = MagicMock()
    mock_response.output_text = '{"a": ["s"]}'
    mock_response.output_parsed = {"a": ["s"]}
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 9
    mock_response.usage.output_tokens = 4
    mock_response.usage.input_tokens_details = None

    with patch("ksi.runtime.llm.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_response
        mock_openai.return_value = mock_client

        caller = OpenAILLMCaller(model="gpt-5.4-mini")

        # Structured path: parsed dict populated.
        structured = caller.call(system="s", user="u", json_schema=_DEMO_SCHEMA)
        assert isinstance(structured, LLMResponse)
        assert structured.parsed == {"a": ["s"]}
        assert structured.usage.output_tokens == 4

        # Non-structured path: same type, parsed is None.
        plain = caller.call(system="s", user="u")
        assert isinstance(plain, LLMResponse)
        assert plain.text == '{"a": ["s"]}'
        assert plain.parsed is None


def test_provider_error_is_not_retried_by_outer_wrapper(monkeypatch):
    with patch("ksi.runtime.llm.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = TimeoutError("request timed out")
        mock_anthropic.Anthropic.return_value = mock_client
        monkeypatch.setenv("KSI_DIRECT_LLM_MAX_RETRIES", "2")

        caller = AnthropicLLMCaller(model="claude-sonnet-4-20250514")

        with pytest.raises(TimeoutError):
            caller.call(system="You are helpful", user="Hi")

        assert mock_client.messages.create.call_count == 1


def test_openai_request_never_includes_seed():
    """The Responses API has no `seed` parameter — a request carrying one
    raises a client-side TypeError on the real SDK (masked here by the mock,
    which is exactly how the original bug slipped past these tests). Assert
    the request never contains `seed`, for both reasoning and chat models."""
    mock_response = MagicMock()
    mock_response.output_text = "ok"
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 1
    mock_response.usage.output_tokens = 1
    mock_response.usage.input_tokens_details = None

    with patch("ksi.runtime.llm.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_response
        mock_openai.return_value = mock_client

        OpenAILLMCaller(model="gpt-4o-mini", temperature=0.0, seed=42).call(system="s", user="u")
        OpenAILLMCaller(model="gpt-5.4-mini", seed=42).call(system="s", user="u")

    chat_call, reasoning_call = mock_client.responses.create.call_args_list
    # The chat-model branch was exercised (temperature is set there)...
    assert "temperature" in chat_call.kwargs
    # ...and neither branch sends seed.
    assert "seed" not in chat_call.kwargs
    assert "seed" not in reasoning_call.kwargs


def test_openai_responses_create_still_has_no_seed_param():
    """Pin the premise: the installed SDK's `Responses.create` has no `seed`
    parameter (and no **kwargs), so forwarding one would TypeError client-side.
    If a future SDK bump adds seed support, this fails and seed forwarding can
    be reinstated deliberately."""
    import inspect

    from openai.resources.responses import Responses

    params = inspect.signature(Responses.create).parameters
    assert "seed" not in params
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def test_build_llm_caller_openai_warns_on_nonzero_seed(caplog):
    """Mirrors the Anthropic branch: a non-zero seed on the OpenAI provider
    warns that it will be ignored; the default seed=0 stays quiet."""
    from ksi.runtime.llm import build_llm_caller

    with caplog.at_level("WARNING", logger="ksi.runtime.llm"):
        build_llm_caller(provider="openai", model="gpt-4o-mini", api_key="x", seed=42)
    assert any("seed=42" in r.message and "OpenAI Responses API" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level("WARNING", logger="ksi.runtime.llm"):
        build_llm_caller(provider="openai", model="gpt-4o-mini", api_key="x", seed=0)
    assert not caplog.records


def test_build_llm_caller_rejects_unknown_provider():
    """A typo'd / unsupported provider fails loudly instead of silently
    returning an Anthropic caller that would 401 at call time."""
    import pytest

    from ksi.runtime.llm import build_llm_caller

    with pytest.raises(ValueError):
        build_llm_caller(provider="gemini", model="x", api_key="x")


def test_build_llm_caller_provider_aliases():
    """'anthropic', 'claude', and '' map to Anthropic; 'openai' to OpenAI."""
    from ksi.runtime.llm import AnthropicLLMCaller, OpenAILLMCaller, build_llm_caller

    for p in ("anthropic", "claude", "", "ANTHROPIC"):
        assert isinstance(build_llm_caller(provider=p, model="m", api_key="x"), AnthropicLLMCaller)
    assert isinstance(build_llm_caller(provider="openai", model="m", api_key="x"), OpenAILLMCaller)

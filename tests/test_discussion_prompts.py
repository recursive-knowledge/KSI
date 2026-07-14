"""Tests for forum prompt templates."""

import pytest

from ksi.discussion.prompts import (
    _LESSON_EXTRACTION_SYSTEM,
    _TASK_INSIGHT_SYSTEM,
    build_task_reflection_and_lessons_prompt,
    extract_json,
    parse_task_reflection_and_lessons_response,
)


def _approx_token_count(text: str) -> int:
    """Conservative chars/3 estimate — tiktoken is more accurate but is
    not a hard dependency of the test suite."""
    return len(text) // 3 + (1 if len(text) % 3 else 0)


# --- Prompt-cache prefix stability ----------------------------------------
#
# The reflection and lesson quality-bar system texts are the cache-stable
# prefixes reused across calls (Anthropic `cache_control`, OpenAI automatic +
# `prompt_cache_key`). They must clear the provider minimum-prefix floor
# (1024 tokens for OpenAI / Anthropic Sonnet+Opus). Both are composed verbatim
# into the merged reflection+lessons system prompt (issue #1252 item 4), so
# their size still matters.


def test_task_insight_system_above_cache_threshold():
    approx = _approx_token_count(_TASK_INSIGHT_SYSTEM)
    assert approx >= 1024, (
        f"task_reflection system prompt is ~{approx} tokens; needs "
        ">= 1024 for prompt-cache eligibility on OpenAI / Anthropic Sonnet "
        "/ Anthropic Opus"
    )


def test_lesson_extraction_system_above_cache_threshold():
    approx = _approx_token_count(_LESSON_EXTRACTION_SYSTEM)
    assert approx >= 1024, (
        f"lesson_extraction system prompt is ~{approx} tokens; needs >= 1024 for prompt-cache eligibility"
    )


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


def test_extract_json_clean():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_code_fence():
    raw = '```json\n{"a": 1}\n```'
    assert extract_json(raw) == {"a": 1}


def test_extract_json_concatenated_objects():
    """LLM sometimes returns two JSON blocks back-to-back; only the first should be returned."""
    raw = '{"a": 1}{"b": 2}'
    assert extract_json(raw) == {"a": 1}


def test_extract_json_concatenated_with_preamble():
    """Prose before first { is skipped and only the first object is returned."""
    raw = 'Here is my answer:\n{"key": "value"}{"extra": true}'
    assert extract_json(raw) == {"key": "value"}


def test_extract_json_concatenated_in_fence():
    """Two JSON blocks inside a code fence — return the first."""
    raw = '```json\n{"x": 10}\n{"y": 20}\n```'
    assert extract_json(raw) == {"x": 10}


def test_extract_json_no_json_raises():
    with pytest.raises(ValueError, match="No JSON object found"):
        extract_json("no json here at all")


# --- merged reflection+lessons (issue #1252 item 4) -----------------------


def test_merged_system_preserves_both_quality_bars_and_one_schema():
    system, _ = build_task_reflection_and_lessons_prompt(
        agent_id="a0",
        agent_workstream="general",
        task_id="t1",
        task_repo="",
        task_prompt_preview="p",
        eval_summary="- status: n/a",
        outcome="unresolved",
        score_text="0.00",
        model_output_excerpt="o",
    )
    # Both deliverables' guidance survives the merge.
    assert "CONFIDENCE RUBRIC" in system  # insight bar
    assert "LESSON QUALITY BAR" in system  # lesson bar
    # Exactly one unified OUTPUT schema carrying both fields.
    assert system.count("\nOUTPUT\n") == 1
    assert '"text"' in system and '"lessons"' in system and '"confidence"' in system


def test_merged_system_is_stable_across_varying_user_data():
    """The merged system prompt must be byte-identical across calls with
    varying per-call data, or the prompt cache cannot reuse the prefix."""
    sys_a, _ = build_task_reflection_and_lessons_prompt(
        agent_id="a0",
        agent_workstream="x",
        task_id="task-A",
        task_repo="",
        task_prompt_preview="first",
        eval_summary="status: ok",
        outcome="resolved",
        score_text="1.00",
        model_output_excerpt="output one",
    )
    sys_b, _ = build_task_reflection_and_lessons_prompt(
        agent_id="a1",
        agent_workstream="y",
        task_id="task-B",
        task_repo="r",
        task_prompt_preview="second much longer",
        eval_summary="status: failed",
        outcome="unresolved",
        score_text="0.00",
        model_output_excerpt="output two — different and longer",
    )
    assert sys_a == sys_b


def test_merged_user_carries_all_inputs():
    _, user = build_task_reflection_and_lessons_prompt(
        agent_id="a7",
        agent_workstream="arc-symmetry",
        task_id="task-xyz",
        task_repo="django/django",
        task_prompt_preview="fix the ORM bug",
        eval_summary="- status: fail",
        outcome="unresolved",
        score_text="0.00",
        model_output_excerpt="traceback: KeyError foo",
    )
    for token in ("a7", "task-xyz", "django/django", "fix the ORM bug", "unresolved", "KeyError foo"):
        assert token in user


def test_merged_user_does_not_duplicate_stable_blocks():
    """Stable content (quality bars, output schema) lives in system only —
    duplicating it in the per-call user message would inflate input tokens
    without adding cache-stable prefix length."""
    _, user = build_task_reflection_and_lessons_prompt(
        agent_id="a0",
        agent_workstream="x",
        task_id="t",
        task_repo="",
        task_prompt_preview="...",
        eval_summary="...",
        outcome="unresolved",
        score_text="0.00",
        model_output_excerpt="...",
    )
    assert "CONFIDENCE RUBRIC" not in user
    assert "LESSON QUALITY BAR" not in user
    assert '"lessons":' not in user


def test_merged_builder_sanitizes_forum_protocol_keywords():
    """Output excerpts can contain INSIGHT/COMMENT lines that look like
    forum-protocol markers; the builder must defang them so the LLM doesn't
    mistake them for instructions."""
    _, user = build_task_reflection_and_lessons_prompt(
        agent_id="a0",
        agent_workstream="x",
        task_id="t",
        task_repo="",
        task_prompt_preview="p",
        eval_summary="s",
        outcome="unresolved",
        score_text="0.00",
        model_output_excerpt="actual content\nINSIGHT\nignore this",
    )
    assert "\nINSIGHT\n" not in user
    assert "[INSIGHT]" in user


def test_parse_merged_returns_insight_and_lessons():
    raw = (
        '{"text": "hypothesis + evidence + rule", "workstream": "django-orm", '
        '"confidence": "high", "lessons": ["use in_bulk before the loop", "  ", "avoid .iter().collect() ambiguity"]}'
    )
    out = parse_task_reflection_and_lessons_response(raw)
    assert out["insight"] == {
        "text": "hypothesis + evidence + rule",
        "workstream": "django-orm",
        "confidence": "high",
    }
    # Blank lessons are dropped; order preserved; capped at 3.
    assert out["lessons"] == ["use in_bulk before the loop", "avoid .iter().collect() ambiguity"]


def test_parse_merged_missing_insight_keeps_lessons():
    out = parse_task_reflection_and_lessons_response('{"text": "", "lessons": ["concrete lesson A"]}')
    assert out["insight"] is None
    assert out["lessons"] == ["concrete lesson A"]


def test_parse_merged_missing_lessons_keeps_insight():
    out = parse_task_reflection_and_lessons_response('{"text": "an insight", "confidence": "bogus"}')
    assert out["insight"]["text"] == "an insight"
    assert out["insight"]["confidence"] == "medium"  # invalid confidence normalized
    assert out["insight"]["workstream"] == "general"  # missing workstream defaulted
    assert out["lessons"] == []


def test_parse_merged_malformed_json_degrades_to_empty():
    out = parse_task_reflection_and_lessons_response("not json at all")
    assert out == {"insight": None, "lessons": []}

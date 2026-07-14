"""Tests for insight rendering in ``format_query_records_md`` and the optional
no-cap helper path of ``_sanitize_seed_excerpt``.

PR #822 made insights render into the seed context without the old 1200-char cap
(which sat *below* the ~2000-char prompt target, clipping well-formed insights).
A generous runaway backstop (``_INSIGHT_SEED_MAX_CHARS`` = 8000, 4x the target)
bounds the prompt-injection surface without touching a well-formed insight.
These tests pin that behavior: a rich insight longer than the old cap reaches
the seed in full; a pathological insight beyond the backstop is bounded; and the
int-cap path used by other seed fields still truncates.
"""

from __future__ import annotations

from ksi.runtime.seeding import (
    _INSIGHT_SEED_MAX_CHARS,
    _sanitize_seed_excerpt,
    format_query_records_md,
)


def _record_with_insight(text: str) -> dict:
    return {
        "gen": 1,
        "eval_results": {"native_score": 0.0, "resolved": False},
        "task_specific_insights": [{"text": text}],
    }


def _filler_records(lengths: list[int], start_gen: int = 1) -> list[dict]:
    """Build records whose ``full_memory_trace_condensed`` bodies are exactly
    ``lengths[i]`` chars each, so the caller can drive the shared cross-
    generation ``used_attempt_chars`` budget to a precise cumulative total.
    """
    return [
        {
            "gen": start_gen + i,
            "eval_results": {"native_score": 0.0, "resolved": False},
            "full_memory_trace_condensed": "x" * length,
            "task_specific_insights": [],
        }
        for i, length in enumerate(lengths)
    ]


def test_long_insight_rendered_verbatim_no_truncation():
    # ~3300 chars — well above the old 1200-char seed-render cap, within the
    # realistic range of a rich hypothesis+evidence+decision-rule insight.
    body = "Check the boundary condition before indexing. " * 70
    insight = body + "DECISION_RULE_AT_THE_TAIL"
    out = format_query_records_md([_record_with_insight(insight)], task_id="t")
    expected = _sanitize_seed_excerpt(insight, max_chars=_INSIGHT_SEED_MAX_CHARS)
    # The full body (collapsed whitespace) survives, including the trailing
    # decision rule the old cap would have clipped.
    assert f"1. {expected}" in out
    assert "DECISION_RULE_AT_THE_TAIL" in out
    assert "...(truncated)" not in out
    assert len(out) > 1200


def test_runaway_insight_bounded_by_backstop():
    # A pathological insight far beyond the ~2000-char target must not flood the
    # seed: the generous backstop truncates it (boundary-aware) at render time.
    insight = "z" * (_INSIGHT_SEED_MAX_CHARS + 5000)
    out = format_query_records_md([_record_with_insight(insight)], task_id="t")
    assert "...(truncated)" in out
    # The rendered insight body is bounded near the backstop, not the full 13k.
    assert out.count("z") <= _INSIGHT_SEED_MAX_CHARS


def test_sanitize_seed_excerpt_none_returns_full_scrubbed_text():
    raw = "line one\n\n   line two\t\tline three " + ("x" * 5000)
    result = _sanitize_seed_excerpt(raw, max_chars=None)
    # No cap, no truncation suffix; whitespace still collapsed.
    assert "...(truncated)" not in result
    assert len(result) >= 5000
    assert "\n" not in result and "\t" not in result


def test_sanitize_seed_excerpt_int_cap_still_truncates():
    # Regression: the max_chars=None addition must not break the int-cap path
    # used by every other seed field.
    result = _sanitize_seed_excerpt("y" * 500, max_chars=40)
    assert result.endswith("...(truncated)")
    assert len(result) < 100


def test_insight_only_record_renders_no_empty_attempt_header():
    # A synthesized insight-only record (standalone R0 insight with no matching
    # attempt) carries insight text but is not a real attempt: it must surface
    # its insight without an empty "### Gen N (score=None)" header, and must not
    # be counted toward the prior-attempt total.
    real_attempt = {
        "gen": 1,
        "eval_results": {"native_score": 0.0, "resolved": False},
        "full_memory_trace_condensed": "real attempt trace",
        "task_specific_insights": [],
    }
    insight_only = {
        "gen": 2,
        "eval_results": {},
        "insight_only": True,
        "task_specific_insights": ["STANDALONE_INSIGHT_BODY"],
    }
    out = format_query_records_md([insight_only, real_attempt], task_id="t")
    assert "STANDALONE_INSIGHT_BODY" in out
    assert "### Gen 2" not in out
    assert "score=None" not in out
    assert "(1 prior attempt(s)" in out


def test_shared_attempt_budget_truncation_marker_on_crossing_generation():
    # Each generation's excerpt is individually boundary-truncated (via
    # _sanitize_seed_excerpt, max_chars=280), but all excerpts are then
    # re-sliced against a SHARED 1_800-char budget (max_attempt_chars) across
    # the whole render. Before this fix, the generation whose excerpt crosses
    # that shared budget got a raw `condensed[:remaining]` slice with no
    # marker -- silently cut off mid-word with no signal to the agent reading
    # its seed context.
    #
    # Use a per-generation body short enough (249 chars) that the individual
    # 280-char cap never fires on its own, so any "...(truncated)" marker in
    # the output can only have come from the shared-budget re-slice.
    body = ("word " * 50).strip()
    assert _sanitize_seed_excerpt(body, max_chars=280) == body

    records = [
        {
            "gen": gen,
            "eval_results": {"native_score": 0.0, "resolved": False},
            "full_memory_trace_condensed": body,
            "task_specific_insights": [],
        }
        for gen in range(1, 9)  # 8 gens * 249 chars > 1_800 shared budget
    ]
    out = format_query_records_md(records, task_id="t")

    # Exactly one generation's excerpt crosses the shared budget (7 * 249 =
    # 1_743 fits; the 8th does not) and must be explicitly marked -- not
    # silently dropped mid-word.
    assert out.count("...(truncated)") == 1
    assert out.count(body) == 7
    last_section = out.split("### Gen 8")[1]
    assert body not in last_section
    assert "...(truncated)" in last_section


def test_shared_attempt_budget_below_marker_width_never_silently_truncates():
    # Residual bug found in review of PR #1056: when the shared cross-
    # generation budget's remaining headroom (`remaining`) is <=
    # len("...(truncated)") == 14, the pre-fix code fell back to an
    # unmarked `condensed[:remaining]` slice -- reproducing the exact
    # silent-truncation bug PR #1056 exists to fix, just narrowed to a
    # <=14-char window. The fix must either mark the excerpt or omit it
    # entirely -- it must never emit an unmarked fragment.
    suffix = "...(truncated)"
    assert len(suffix) == 14

    probe = {
        "gen": 100,
        "eval_results": {"native_score": 0.0, "resolved": False},
        "full_memory_trace_condensed": "z" * 50,
        "task_specific_insights": [],
    }

    # --- remaining == 14: exactly at the marker-width boundary. There is no
    # room to fit even one content char plus the marker meaningfully, so the
    # excerpt must be OMITTED -- not a silent unmarked fragment. ---
    fillers_14 = _filler_records([200] * 8 + [186])  # sums to 1_786; remaining=14
    out_14 = format_query_records_md(fillers_14 + [probe], task_id="t")
    last_section_14 = out_14.split("### Gen 100")[1]
    assert "z" not in last_section_14
    assert "...(truncated)" not in last_section_14

    # --- remaining == 15: just past the boundary. There is exactly enough
    # room for 1 content char plus the marker, so the excerpt must be
    # MARKED. ---
    fillers_15 = _filler_records([200] * 8 + [185])  # sums to 1_785; remaining=15
    out_15 = format_query_records_md(fillers_15 + [probe], task_id="t")
    last_section_15 = out_15.split("### Gen 100")[1]
    assert "z...(truncated)" in last_section_15


def test_condensed_trace_redacts_hidden_text_marker():
    # Older DB rows can bake hidden-output fragments (verifier_clues=/...) into
    # full_memory_trace_condensed. The seeding path must redact them like the
    # distillation path does, so they never reach the next-gen agent on resume.
    record = {
        "gen": 1,
        "eval_results": {"native_score": 0.0, "resolved": False},
        "full_memory_trace_condensed": ("verifier_clues=SECRET_HIDDEN_CLUE_TOKEN; reward=0"),
        "task_specific_insights": [],
    }
    out = format_query_records_md([record], task_id="t")
    assert "SECRET_HIDDEN_CLUE_TOKEN" not in out
    assert "verifier_clues=" not in out
    # Declared experience signal (the safe summary scalar) is retained.
    assert "reward=0" in out

"""Prompt builders for distillation LLM calls.

Each builder returns (system_prompt, user_prompt). The LLM is expected to emit
a JSON object with keys: transferable_insights, confirmed_constraints,
rejected_hypotheses, pitfalls, checks, next_steps, and evidence_post_ids (see
``_OUTPUT_SCHEMA`` / ``DISTILL_BUNDLE_JSON_SCHEMA`` below for the full shape).

Design note: the cross-task prompt is rewritten to force
concrete primitives and reject generic process meta-advice. The
cross-task bundle tended to be ~95% process
meta ("validate first", "decompose", "check boundary") and only ~5%
concrete operations. Because the Knowledge-Transfer sweep extracts
*only* the cross-task bundle, that noise channel was being used as the
transfer payload. The rules below (explicit banned phrases, mandatory
concrete grounding, mandatory evidence citations, tighter 5-item caps)
steer the distiller toward the signal in the posts instead.
"""

from __future__ import annotations

import re
from typing import Any

from ..memory.parity import redact_solver_hidden_text
from ..tasks.registry import resolve_source
from .types import truncate_at_boundary

_SYSTEM_PROMPT = (
    "You are a distillation assistant. Your job is to compress a set of "
    "agent attempts and discussion posts into a compact, high-signal "
    "bundle of actionable guidance for the next generation of agents.\n"
    "Output strict JSON only, no prose outside the JSON object.\n"
    "\n"
    "DOWNSTREAM CONSUMERS\n"
    "Your output is loaded verbatim into the next generation's prompt as "
    "prior knowledge. Agents act on bullets without being able to ask "
    "questions, so a vague bullet produces a vague action. Bullets must "
    "be specific enough to either succeed or fail observably when an "
    "agent applies them. Bullets that read like advice from a senior "
    'engineer skimming the problem ("think carefully", "validate '
    'first", "watch for edge cases") add nothing to a future agent\'s '
    "ability to act, and their cost is real: they push out the limited "
    "bullets-per-bundle budget and dilute the signal-to-noise that "
    "downstream retrieval ranks against.\n"
    "\n"
    "QUALITY CRITERIA\n"
    "A high-signal bullet either (a) names a specific operation, API, "
    "library, file path, or pattern that an agent should try first, "
    "(b) names a specific failure mode that an agent should avoid, or "
    "(c) records a verified invariant the next agent can rely on without "
    "re-deriving it. Anything else is noise. When in doubt, drop the "
    "bullet -- empty lists are acceptable and preferable to filler."
)

# --- Anti-meta / concreteness rules ---------------------------------------
#
# Single source of truth — see src/kcsi/discussion/concreteness.py. The block
# is rendered into both the cross-task forum prompt (write-time) and the
# distill system prompt so agents and the distiller see byte-identical rules.
# Tests assert its presence so regressions trip CI.
from kcsi.discussion.concreteness import ANTI_META_BLOCK as _ANTI_META_BLOCK

_GENERIC_DOMAIN_HINT = (
    "DOMAIN HINT: the input spans multiple benchmarks. Prefer "
    "primitives concrete enough to be named (operations, APIs, "
    "libraries, idioms) while keeping wording task-agnostic."
)


def _domain_hint(task_source: str | None) -> str:
    # Resolve aliases (arc1/arc2/arc_agi_*/swebench) to their canonical source
    # via the central registry, then read the source's spec-attached domain
    # hint; the built-in hints live on their specs in
    # ``src/kcsi/tasks/registry.py``.
    #
    # The domain hint is opt-in and defaults to off: a resolved source that
    # leaves ``distill_domain_hint`` unset injects no hint at all ("") so
    # adding a benchmark never requires supplying one. The ``_GENERIC_DOMAIN_HINT``
    # is reserved for the unresolvable/cross-task case, where there is no single
    # benchmark to key on (``task_source`` is None or an unknown string).
    spec = resolve_source(task_source)
    if spec is None:
        return _GENERIC_DOMAIN_HINT
    hint = spec.distill_domain_hint
    if not hint:
        return ""
    return hint() if callable(hint) else hint


_OUTPUT_SCHEMA = (
    "Output schema (strict JSON). Items are STRUCTURED dicts (not bare strings):\n"
    "{\n"
    '  "transferable_insights": [<Insight>, ...],\n'
    '  "confirmed_constraints": [<Insight>, ...],\n'
    '  "rejected_hypotheses": [<Insight>, ...],\n'
    '  "pitfalls": [<Insight>, ...],\n'
    '  "checks": [<Insight>, ...],\n'
    '  "next_steps": [<Insight>, ...],\n'
    '  "evidence_post_ids": [<integer>, ...]\n'
    "}\n"
    "where each <Insight> is:\n"
    "{\n"
    '  "text": "<the rule itself, of the form \'when X, do Y\' — concrete>",\n'
    '  "applies_when": "<concrete condition under which the rule holds>",\n'
    '  "does_not_apply_when": "<concrete counterexample / boundary>",\n'
    '  "evidence": [{"task_id": "<id>", "post_id": <int>, "quote": "<verbatim 1-2 sentence quote>"}, ...],\n'
    '  "confidence": "high" | "medium" | "low"\n'
    "}\n"
    "\n"
    "Field semantics:\n"
    "- transferable_insights: concrete actionable rules. Future agent reads "
    "the rule + boundary and knows when to apply.\n"
    "- confirmed_constraints: invariants that were VERIFIED by an attempt or "
    "by forum evidence. The boundary names where the invariant might break.\n"
    "- rejected_hypotheses: approaches evidence showed are wrong — stated as "
    "PARAMETERIZED rejections, never wholesale family bans. Each item's `text` "
    "must follow: 'FALSIFIED: <hypothesis family> with <exact parameterization "
    "tried> (<evidence>) — UNTRIED: <nearby variants not yet ruled out>'. Only "
    "reject a family outright after >= 2 distinct parameterizations falsified. "
    "(In practice, ~4/10 eventual solves came from families a bundle had "
    "wholesale-rejected.)\n"
    "- pitfalls: failure modes to avoid. The boundary names tasks/conditions "
    "where the pitfall does NOT apply (so a future agent doesn't over-generalize).\n"
    "- checks: quick verifications. Each must name a concrete check.\n"
    "- next_steps: specific experiments or branches a future agent should try.\n"
    "- evidence_post_ids: top-level list of post IDs supporting the bundle "
    "as a whole; per-Insight evidence is preferred. Do not invent post IDs.\n"
    "\n"
    "STRUCTURED-INPUT RULES (these are non-negotiable):\n"
    "- Every Insight MUST have non-empty `text`, `applies_when`, "
    "`does_not_apply_when`, and at least one `evidence` entry. DROP any "
    "Insight that cannot satisfy all four — empty lists are correct and "
    "preferable to filler.\n"
    "- `does_not_apply_when` cannot be 'n/a', 'always', 'never', or empty. "
    "An unbounded rule is rejected.\n"
    "- VERBATIM QUOTE REQUIRED: each `evidence[].quote` MUST be a verbatim "
    "1-sentence excerpt copied from a cited forum post. Paraphrases are "
    "rejected — if you cannot find a literal sentence in the input that "
    "supports the Insight, the Insight is unsupported and DROPPED. The "
    "quote field is what proves the Insight is grounded, not invented.\n"
    "- An Insight contradicted by another Insight's `does_not_apply_when` is "
    "either dropped or both are tagged 'low' confidence.\n"
    "- Hard caps: at most 5 items per field; each `text` <= 480 chars; each "
    "`applies_when` and `does_not_apply_when` <= 200 chars; each `quote` <= "
    "200 chars. Prefer fewer, higher-signal Insights over filler.\n"
    "- For Terminal-Bench 2, checks and next_steps should usually name the "
    "exact shell command, file path, service name, port, module, or artifact "
    "to inspect or change; prefer verifier-aligned checks over generic advice.\n"
)


_GENERIC_ADVICE_BAN = (
    "EXCLUDE GENERIC PROCESS ADVICE.\n"
    "Do NOT emit Insights that hold for any task regardless of its content — "
    "e.g. 'validate against all training pairs before submitting', 'write "
    "executable cell-by-cell checks', 'construct the full output grid', "
    "'serialize JSON correctly', 'do not infer a rule from one example'. "
    "Agents already receive these as standing instructions; restating them "
    "wastes the bundle. Every Insight's `applies_when` MUST name a "
    "task-discriminating structural condition (a grid/object/marker/pattern "
    "property that some tasks have and others do not). If you cannot name "
    "one, drop the Insight. Never place the same insight in more than one "
    "field.\n"
)


# Machine-readable counterpart of ``_OUTPUT_SCHEMA`` above. Passed to providers
# that support JSON-schema-constrained output (Anthropic tool-forcing / OpenAI
# Responses json_schema) so the distill response is guaranteed parseable JSON
# without the brace-matching + regex-repair + repair-LLM fallback path. The
# prose ``_OUTPUT_SCHEMA`` still ships in the system prompt because it carries
# the field semantics and the non-negotiable grounding rules that a bare JSON
# schema cannot express. The shape here intentionally mirrors what
# ``per_task._as_insight_list`` / ``_as_int_list`` already coerce — items are
# permissive (str OR Insight dict) because the prompt's hard rules, not the
# schema, enforce Insight quality.
_INSIGHT_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "applies_when": {"type": "string"},
        "does_not_apply_when": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "post_id": {"type": "integer", "minimum": 1},
                    "quote": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
    },
    "additionalProperties": True,
}

_INSIGHT_LIST_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": _INSIGHT_ITEM_SCHEMA,
}

DISTILL_BUNDLE_JSON_SCHEMA: dict[str, Any] = {
    "name": "distill_bundle",
    "schema": {
        "type": "object",
        "properties": {
            "transferable_insights": _INSIGHT_LIST_SCHEMA,
            "confirmed_constraints": _INSIGHT_LIST_SCHEMA,
            "rejected_hypotheses": _INSIGHT_LIST_SCHEMA,
            "pitfalls": _INSIGHT_LIST_SCHEMA,
            "checks": _INSIGHT_LIST_SCHEMA,
            "next_steps": _INSIGHT_LIST_SCHEMA,
            "evidence_post_ids": {"type": "array", "items": {"type": "integer", "minimum": 1}},
        },
        "additionalProperties": True,
    },
}

# Excerpt caps on distiller inputs. These are tuned for the KT-sweep signal
# chain: 800/400/1500 were clipping distiller inputs
# mid-sentence, so distilled bundles lost fidelity before they ever reached
# the anti-meta filter. Raised to sizes that cover typical per-turn tool-use
# traces (2000 chars ≈ 400-600 tokens) and full-bundle JSON. These are the
# binding constraint on how much signal reaches the distiller. This is separate
# from verbatim insight storage, seed-render backstops, and the smaller caps
# applied to final distillation bundle items.
_ATTEMPT_OUTPUT_EXCERPT_CHARS = 2000
# Bumped 1200 → 2000 for V2: per-task posts are structured JSON post-mortems
# averaging 1.5-2KB. Distill needs the full proposed_change + predicted_outcome
# fields, not the truncated head. Cache-safe (per-item).
_POST_TEXT_EXCERPT_CHARS = 2000
_EVAL_SUMMARY_CHARS = 1600
_EVAL_LIST_ITEMS = 4


def _sanitize_prompt_excerpt(value: Any, *, max_chars: int) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"(?m)^(\s*)(INSIGHT|COMMENT)(\s*)$", r"\1[\2]\3", text)
    text = " ".join(text.splitlines())
    if len(text) > max_chars:
        return truncate_at_boundary(text, max_chars) + "..."
    return text


def _sanitize_target_prompt(value: Any, *, max_chars: int) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"(?m)^(\s*)(INSIGHT|COMMENT)(\s*)$", r"\1[\2]\3", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) > max_chars:
        return truncate_at_boundary(text, max_chars) + "..."
    return text


def _sanitize_prompt_field(value: Any, *, max_chars: int = 120) -> str:
    text = _sanitize_prompt_excerpt(value, max_chars=max_chars).strip()
    return text or "?"


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    return [value]


def _fmt_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return _sanitize_prompt_field(value, max_chars=40)


def _fmt_test_bucket(bucket: Any) -> str:
    """Render test bucket as anonymized counts (no test names).

    In upstream-strict SWE-bench Pro runs, F2P/P2P identifiers are outside the
    declared solver-attempt feedback channel. Emitting them into the distillation
    prompt would seed subsequent generations with hidden test identity; only
    count information is safe to surface.
    """
    if not isinstance(bucket, dict):
        return ""
    parts: list[str] = []
    for key in ("success", "failure", "skipped", "unknown"):
        if key not in bucket:
            continue
        values = _as_list(bucket.get(key))
        # Count only — no test-name listing.
        parts.append(f"{key}={len(values)}")
    return " ".join(parts)


def _fmt_tests_status(tests_status: Any) -> str:
    if not isinstance(tests_status, dict) or not tests_status:
        return ""

    parts: list[str] = []
    observed = tests_status.get("observed_count")
    if observed is not None:
        parts.append(f"observed_count={_sanitize_prompt_field(observed, max_chars=40)}")

    for suite in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        rendered = _fmt_test_bucket(tests_status.get(suite))
        if rendered:
            parts.append(f"{suite} {rendered}")

    return "tests: " + "; ".join(parts) if parts else ""


def _fmt_arc_eval(eval_results: dict[str, Any]) -> str:
    has_arc = any(
        key in eval_results for key in ("arc_correct_count", "arc_total_count", "arc_pass_ratio", "arc_per_test")
    )
    if not has_arc:
        return ""

    parts: list[str] = []
    total = eval_results.get("arc_total_count")
    correct = eval_results.get("arc_correct_count")
    if total is not None:
        correct_text = "?" if correct is None else _sanitize_prompt_field(correct, max_chars=40)
        total_text = _sanitize_prompt_field(total, max_chars=40)
        parts.append(f"ARC {correct_text}/{total_text} correct")

    ratio = eval_results.get("arc_pass_ratio")
    if ratio is not None:
        parts.append(f"pass_ratio={_sanitize_prompt_field(ratio, max_chars=40)}")

    per_test = eval_results.get("arc_per_test")
    if isinstance(per_test, list) and per_test:
        counts = {"correct": 0, "wrong": 0, "unknown": 0}
        observed = 0
        for item in per_test:
            if not isinstance(item, dict):
                continue
            observed += 1
            correct_value = item.get("correct")
            if correct_value is True:
                counts["correct"] += 1
            elif correct_value is False:
                counts["wrong"] += 1
            else:
                counts["unknown"] += 1
        if observed:
            parts.append(
                "ARC per_test_summary: "
                f"observed={observed} correct={counts['correct']} "
                f"wrong={counts['wrong']} unknown={counts['unknown']}"
            )

    return "; ".join(parts)


def _fmt_eval_results(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return ""

    parts: list[str] = []
    status = value.get("status") or value.get("swebench_status")
    if status:
        parts.append(f"status={_sanitize_prompt_field(status, max_chars=80)}")

    resolved = value.get("resolved")
    if resolved is not None:
        parts.append(f"resolved={_fmt_bool(resolved)}")

    native_score = value.get("native_score")
    if native_score is not None:
        parts.append(f"native_score={_sanitize_prompt_field(native_score, max_chars=40)}")

    instance_report = value.get("instance_report")
    if isinstance(instance_report, dict):
        instance_status = instance_report.get("status")
        if instance_status and instance_status != status:
            parts.append(f"instance_status={_sanitize_prompt_field(instance_status, max_chars=80)}")
        instance_resolved = instance_report.get("resolved")
        if instance_resolved is not None and instance_resolved != resolved:
            parts.append(f"instance_resolved={_fmt_bool(instance_resolved)}")
        tests = _fmt_tests_status(instance_report.get("tests_status"))
        if tests:
            parts.append(tests)

    arc = _fmt_arc_eval(value)
    if arc:
        parts.append(arc)

    error = value.get("error") or value.get("message")
    if error:
        parts.append(f"error={_sanitize_prompt_excerpt(error, max_chars=240)}")

    if not parts:
        return ""
    return _sanitize_prompt_excerpt("; ".join(parts), max_chars=_EVAL_SUMMARY_CHARS)


def _fmt_attempts(attempts: list[dict[str, Any]]) -> str:
    if not attempts:
        return "(no attempts)"
    out = []
    for a in attempts:
        score = a.get("native_score")
        aid = _sanitize_prompt_field(a.get("agent_id", "?"))
        gen = a.get("generation")
        gen_str = f" gen={gen}" if gen is not None else ""
        eval_summary = _fmt_eval_results(a.get("eval_results"))
        trace_summary = _sanitize_prompt_excerpt(
            redact_solver_hidden_text(a.get("trace_condensed")),
            max_chars=_ATTEMPT_OUTPUT_EXCERPT_CHARS,
        )
        model_output = _sanitize_prompt_excerpt(
            a.get("model_output"),
            max_chars=_ATTEMPT_OUTPUT_EXCERPT_CHARS,
        )
        attempt_meta = a.get("attempt_meta") if isinstance(a.get("attempt_meta"), dict) else {}
        # V2: reflection is the agent's own structured 3-5 sentence summary
        # written in the same container after eval (Phase 1 reflection step).
        # It replaces native_session_memory as the rich-signal field.
        reflection = _sanitize_prompt_excerpt(
            redact_solver_hidden_text(a.get("reflection") or ""),
            max_chars=_ATTEMPT_OUTPUT_EXCERPT_CHARS,
        )
        parts = [f"- agent={aid}{gen_str} score={score}"]
        if eval_summary:
            parts.append(f"  eval={eval_summary}")
        if reflection:
            parts.append(f"  reflection={reflection}")
        if trace_summary:
            parts.append(f"  trace={trace_summary}")
        if attempt_meta:
            meta_bits: list[str] = []
            for key in ("reward", "agent_exit_code", "verifier_exit_code", "tool_count"):
                if key in attempt_meta and attempt_meta.get(key) is not None:
                    meta_bits.append(f"{key}={_sanitize_prompt_field(attempt_meta.get(key), max_chars=80)}")
            # NOTE: terminal_bench_2 ``failure_signature`` / ``verifier_clues`` /
            # ``verifier_*_tail`` are deliberately NOT rendered. The TB2 verifier
            # runs hidden pytest on held-out tests (the benchmark treats reading
            # them as cheating), so that content echoes assertions outside the
            # upstream-strict feedback channel. Declared experience signals
            # (outcome/reward, exit codes, the commands the agent ran) flow
            # forward.
            for key in ("verified_outcome", "last_state_change"):
                value = attempt_meta.get(key)
                if value:
                    meta_bits.append(f"{key}={_sanitize_prompt_excerpt(value, max_chars=180)}")
            recent_commands = attempt_meta.get("recent_commands")
            if isinstance(recent_commands, list) and recent_commands:
                command_preview = " | ".join(
                    _sanitize_prompt_excerpt(cmd, max_chars=120) for cmd in recent_commands[:3] if cmd
                )
                if command_preview:
                    meta_bits.append(f"recent_commands={command_preview}")
            if meta_bits:
                parts.append(f"  attempt_meta={'; '.join(meta_bits)}")
        parts.append(f"  output={model_output}")
        out.append("\n".join(parts))
    return "\n".join(out)


def _fmt_posts(posts: list[dict[str, Any]]) -> str:
    if not posts:
        return "(no posts)"
    out = []
    for p in posts:
        pid = _sanitize_prompt_field(p.get("id", "?"))
        aid = _sanitize_prompt_field(p.get("agent_id", "?"))
        gen = p.get("generation")
        round_num = p.get("round_num")
        native_score = p.get("native_score")
        # V2: surface round_num (distinguishes round-0 opinion from round-1
        # response in Phase 3 multi-round) and native_score (post author's
        # own task score — distinguishes "the agent who solved it said this"
        # from "the agent who failed it said this").
        meta_parts = []
        if gen is not None:
            meta_parts.append(f"gen={gen}")
        if round_num is not None:
            meta_parts.append(f"round={round_num}")
        if native_score is not None:
            meta_parts.append(f"author_score={native_score}")
        meta_str = (" " + " ".join(meta_parts)) if meta_parts else ""
        text = _sanitize_prompt_excerpt(
            p.get("text") or p.get("content"),
            max_chars=_POST_TEXT_EXCERPT_CHARS,
        )
        reply_to = p.get("reply_to")
        suffix = f" reply_to={_sanitize_prompt_field(reply_to)}" if reply_to else ""
        out.append(f"- id={pid} agent={aid}{meta_str}{suffix}: {text}")
    return "\n".join(out)


def _build_distill_system(*, role_directive: str, task_source: str | None) -> str:
    """Assemble the cache-stable system prompt for a distill call.

    Prompt-cache eligibility (Anthropic ``cache_control: ephemeral`` and
    OpenAI's automatic prompt cache) requires the same prefix to appear at
    the start of input across calls. The bits below are stable across all
    calls of the same builder + task_source: the role directive, the
    domain hint, the anti-meta rules, and the output schema. The varying
    per-call data (task id, attempts, posts) lives in the user message
    instead, so the system prefix matches byte-for-byte
    across every distill call within a generation and across generations.

    Without this split, the same stable content was being concatenated
    into the user message after the per-call data (task id was at the
    start), so no two calls shared a stable prefix and the cache never
    fired. This prefix-stability change pairs with the OpenAI
    ``prompt_cache_key`` routing pin.
    """
    hint = _domain_hint(task_source)
    hint_block = f"{hint}\n\n" if hint else ""
    return (
        f"{_SYSTEM_PROMPT}\n\n{role_directive}\n\n{hint_block}"
        f"{_ANTI_META_BLOCK}\n{_GENERIC_ADVICE_BAN}\n{_OUTPUT_SCHEMA}"
    )


_PER_TASK_ROLE_DIRECTIVE = (
    "ROLE: distill agent attempts and per-task forum posts for ONE task "
    "into a compact bundle of guidance for the next generation attempting "
    "the same task.\n"
    "\n"
    "INPUT SHAPE (V2):\n"
    "- Attempts: chronological across all gens. Each has agent_id, gen, "
    "score, eval, reflection (the agent's own 3-5 sentence post-eval "
    "summary), output. The reflection field is your richest signal — it's "
    "what the agent concluded with full session memory + eval result in "
    "working context.\n"
    "- Per-task forum posts: chronological across all gens. Each has gen, "
    "round_num, author_score (post author's own task score), text. Posts "
    "are typically free-form prose (agents do not reliably emit JSON).\n"
    "\n"
    "EXTRACTION HEURISTICS:\n"
    "- A reflection's `proposed change` or a post's `next attempt should` "
    "becomes a `next_steps` Insight.\n"
    "- An assumption that proved wrong (low-score reflection naming a "
    "specific cause) becomes `rejected_hypotheses` or `pitfalls`.\n"
    "- A verified invariant (high-score reflection or evidence in posts) "
    "becomes `confirmed_constraints`.\n"
    "- Weight high-score authors over low-score authors when claims conflict.\n"
    "- Each Insight's `evidence[]` MUST cite at least one post_id from the "
    "posts above (or no Insight)."
)

_CROSS_TASK_ROLE_DIRECTIVE = (
    "ROLE: distill the cross-task forum history into rules that apply across "
    "tasks. The forum is a multi-generation, multi-round conversation among "
    "agents who each just attempted a different task and discussed what "
    "transfers.\n"
    "\n"
    "INPUT SHAPE (V2 + concreteness):\n"
    "- Cross-task forum posts across ALL generations, chronological. Each "
    "has gen, round_num (round 0 = opinion grounded in own task; round 1+ "
    "= response to peers), agent_id, text. Posts SHOULD be JSON objects "
    "with fields {concrete_primitive, task_grounding{task_id, "
    "where_it_appeared, evidence_post_id}, transfer_claim, "
    "anti_meta_self_check}; legacy free-form prose may also appear from "
    "older generations. When `concrete_primitive` is present, treat it as "
    "the load-bearing token: the Insight you build from the post should "
    "name that primitive in `text` and quote `where_it_appeared` "
    "verbatim in `evidence[].quote`. Posts whose `concrete_primitive` is "
    'abstract (e.g. "separation of concerns", "two-phase pipeline") '
    "must be DROPPED — the JSON shape does not exempt vague content from "
    "the anti-meta rules.\n"
    "\n"
    "ANTI-VAGUENESS DISCIPLINE (HARD RULES):\n"
    "- Every Insight's `evidence[]` MUST include at least one verbatim "
    "1-sentence quote drawn from a cited forum post. Prefer the "
    "`where_it_appeared` field of the post when present. The quote field "
    "is what proves the Insight isn't a hallucinated paraphrase. Insights "
    "without a verbatim quote are DROPPED.\n"
    "- Round-1 posts (responses to peers) carry stronger signal than "
    "round-0 posts (initial opinions). When a claim appears in BOTH a "
    "round-0 opinion AND a round-1 response (especially across different "
    "agents), promote it to confidence='high'. A claim only appearing in "
    "round-0 opinions with no round-1 engagement is medium or low.\n"
    "- Cross-generation persistence is signal: a claim that recurs in "
    "discussion across multiple gens (multiple gen= values in evidence) "
    "is high confidence even at round 0.\n"
    "- Drop bullets that read like advice from a tutorial — see ANTI-META "
    "RULES."
)

_CROSS_TASK_TARGET_DIRECTIVE = (
    "\n\nTARGET TASK FOCUS: a specific downstream task is provided at the END "
    "of the input, after the forum history. Distill ONLY the insights, "
    "constraints, pitfalls, and next steps that plausibly TRANSFER TO or are "
    "RELEVANT FOR that target task. Drop cross-task lessons that do not bear "
    "on it. Keep the same anti-vagueness and verbatim-evidence rules — a "
    "relevant-but-vague insight is still dropped. Do NOT quote or copy the "
    "target task's own statement into the bundle; use it only to decide "
    "relevance."
)

_WIN_MODE_DIRECTIVE = (
    "This task was SOLVED this generation. Your primary output is "
    "`transferable_insights`: state the winning technique so an agent on a "
    "DIFFERENT task could apply it — name the structural trigger condition "
    "in `applies_when` and the procedure in `text`; no task-specific "
    "coordinates, file names, or literal values. Emit 1-3 "
    "transferable_insights items. Keep all other fields minimal (at most 2 "
    "items each)."
)

_TRANSFERABLES_SECTION_TITLE = "## Per-task transferable candidates (distilled from per-task bundles)"

_TRANSFERABLES_DIRECTIVE = (
    "These are transferable-insight candidates distilled from individual "
    "per-task bundles (both solved and still-unsolved tasks); treat them as "
    "candidate transferable insights — generalize and merge them with forum "
    "evidence rather than restating them verbatim."
)


def _fmt_transferables_section(per_task_transferables: list[dict[str, Any]] | None) -> str:
    """Render the per-task transferable-candidates section (KCSI_TRANSFER_BRIDGE).

    Returns "" when there are no transferables so every builder stays
    byte-identical with the bridge off."""
    if not per_task_transferables:
        return ""
    lines: list[str] = []
    for entry in per_task_transferables:
        tid = _sanitize_prompt_field(entry.get("task_id"))
        text = _sanitize_prompt_excerpt(entry.get("text"), max_chars=480)
        applies_when = _sanitize_prompt_excerpt(entry.get("applies_when"), max_chars=200).strip()
        suffix = f" (applies when: {applies_when})" if applies_when else ""
        lines.append(f"- [{tid}] {text}{suffix}")
    return f"{_TRANSFERABLES_SECTION_TITLE}\n" + "\n".join(lines) + f"\n{_TRANSFERABLES_DIRECTIVE}"


_TARGET_TASK_SECTION_TITLE = "## Target task (distill only what transfers to THIS task)"
# Effectively-uncapped: the spec requires the FULL target-task prompt as the
# conditioning signal, so we sanitize (newline-collapse) but do not truncate.
# The cross-task budget machinery counts this section and trims forum posts to
# compensate; it is never trimmed itself.
_TARGET_TASK_PROMPT_CHARS = 10_000_000


def _fmt_target_task_section(target_task: dict[str, Any] | None) -> str:
    """Render the downstream target-task section. Returns "" when no target so
    the non-conditioned prompt stays byte-identical."""
    if not target_task:
        return ""
    tid = _sanitize_prompt_field(target_task.get("id"))
    prompt = _sanitize_target_prompt(target_task.get("prompt"), max_chars=_TARGET_TASK_PROMPT_CHARS)
    return f"{_TARGET_TASK_SECTION_TITLE}\nTask ID: {tid}\n{prompt}"


def build_per_task_distill_prompt(
    *,
    task_id: str,
    attempts: list[dict[str, Any]],
    posts: list[dict[str, Any]],
    task_source: str | None = None,
    win_mode: bool = False,
) -> tuple[str, str]:
    """Per-task distill prompt (window mode).

    ``win_mode`` (KCSI_TRANSFER_BRIDGE): the task was solved this generation;
    append the win directive so the distiller extracts the winning technique
    into ``transferable_insights``. The system prefix stays cache-stable per
    (win_mode, task_source).
    """
    base_role = f"{_PER_TASK_ROLE_DIRECTIVE}\n\n{_WIN_MODE_DIRECTIVE}" if win_mode else _PER_TASK_ROLE_DIRECTIVE
    system = _build_distill_system(
        role_directive=base_role,
        task_source=task_source,
    )
    user = (
        f"Task ID: {_sanitize_prompt_field(task_id)}\n\n"
        f"## Attempts on this task (ALL generations, chronological)\n"
        f"{_fmt_attempts(attempts)}\n\n"
        f"## Per-task forum posts on this task (ALL generations, chronological)\n"
        f"{_fmt_posts(posts)}\n\n"
        f"Distill into the structured Insight schema in the system prompt. "
        f"Empty fields are acceptable when the input doesn't support them."
    )
    return system, user


def build_cross_task_distill_prompt(
    *,
    cross_posts: list[dict[str, Any]],
    task_source: str | None = None,
    per_task_transferables: list[dict[str, Any]] | None = None,
    target_task: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Cross-task distill prompt (window mode).

    ``per_task_transferables`` (KCSI_TRANSFER_BRIDGE): success-derived
    ``{task_id, text, applies_when}`` entries rendered as an extra section;
    omitted entirely when empty so the bridge-off prompt is byte-identical.

    ``target_task`` (target-conditioning): ``{"id", "prompt"}`` for the
    downstream task. When set, the system prompt gains a relevance directive
    and the target task is appended to the user message AFTER the forum-posts
    (and transferables) block, so the system + forum-posts prefix stays stable
    across the N per-task calls within a generation (cache-friendly). When
    None, the prompt is byte-identical to the non-conditioned path.
    """
    system, cache_prefix, suffix = build_cross_task_distill_prompt_parts(
        cross_posts=cross_posts,
        task_source=task_source,
        per_task_transferables=per_task_transferables,
        target_task=target_task,
    )
    return system, cache_prefix + suffix


def build_cross_task_distill_prompt_parts(
    *,
    cross_posts: list[dict[str, Any]],
    task_source: str | None = None,
    per_task_transferables: list[dict[str, Any]] | None = None,
    target_task: dict[str, Any] | None = None,
) -> tuple[str, str, str]:
    """Same prompt as :func:`build_cross_task_distill_prompt`, but split into
    ``(system, cache_prefix, suffix)`` where ``cache_prefix + suffix`` is the
    user message byte-for-byte.

    ``cache_prefix`` is the cross-call-STABLE portion (the windowed forum
    history + transferables) that is re-sent identically to every target in a
    target-conditioned generation; ``suffix`` is the per-target-VARYING tail
    (the target-task section + the distill instruction). Callers hand the
    prefix to the LLM caller's ``cache_prefix`` so it is cache-read on every
    target after the first, instead of re-paid as plain input tokens.
    The target directive stays in the SYSTEM prompt (as
    before), so the system is also identical across targets — both providers
    then share one cache/route key for the whole prefix."""
    transferables_section = _fmt_transferables_section(per_task_transferables)
    target_section = _fmt_target_task_section(target_task)
    role_directive = _CROSS_TASK_ROLE_DIRECTIVE + (_CROSS_TASK_TARGET_DIRECTIVE if target_task else "")
    system = _build_distill_system(
        role_directive=role_directive,
        task_source=task_source,
    )
    cache_prefix = f"## Cross-task forum history (ALL generations, chronological)\n{_fmt_posts(cross_posts)}\n\n" + (
        f"{transferables_section}\n\n" if transferables_section else ""
    )
    suffix = (f"{target_section}\n\n" if target_section else "") + (
        "Distill into the structured Insight schema in the system prompt. "
        "Empty fields are acceptable when the input doesn't support them."
    )
    return system, cache_prefix, suffix

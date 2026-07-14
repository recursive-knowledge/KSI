"""Post-attempt knowledge extraction helpers.

Turns a finished :class:`~kcsi.models.TaskTrace` / ``eval_result`` into the
knowledge artifacts the orchestrator persists and seeds forward: structured
attempt events, failure-diagnosis summaries, the terminal_bench_2 condensed
trace, and the dataset-aware score extractor.

Extracted verbatim from ``kcsi.orchestrator.engine`` to shrink the
engine hot path. ``engine`` re-imports these names, so existing call sites and
``from kcsi.orchestrator.engine import _build_attempt_event`` continue to work.

This module imports only from ``memory.parity``, ``models``, ``.scoring`` and
``..tasks.registry`` -- never from ``engine`` -- so there is no import cycle.
"""

from __future__ import annotations

import re
from typing import Any

from ..memory.parity import ARC_PER_TEST_SAFE_KEYS
from ..models import TaskSpec, TaskTrace
from ..tasks.registry import resolve_source
from .scoring import score_from_eval_results


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:
            return None
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"null", "none", "nil", "undefined"}:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if parsed != parsed else parsed


def _trim_text(value: Any, max_chars: int) -> str:
    """Trim text to *max_chars*, appending a truncation marker when shortened."""
    txt = str(value or "").strip()
    if len(txt) <= max_chars:
        return txt
    return txt[: max_chars - 14] + "...(truncated)"


_TB2_CLUE_PRIORITY_MARKERS: tuple[str, ...] = (
    "AssertionError",
    "ModuleNotFoundError",
    "ConnectionError",
    "Connection refused",
    "Permission denied",
    "No such file",
    "not found",
    "timed out",
    "FAILED",
    "Expected ",
    "Traceback",
    "E       ",
)


_TB2_CLUE_SKIP_PREFIXES: tuple[str, ...] = (
    "platform ",
    "rootdir:",
    "plugins:",
    "cachedir:",
    "collecting ...",
    "try:",
    "except ",
    "raise ",
    "return ",
    "if ",
    "with ",
    "def ",
    "self.",
    "stdout =",
    "stderr =",
)


def _tb2_extract_verifier_clues(*texts: Any, max_items: int = 4) -> list[str]:
    ranked: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    ordinal = 0
    for raw in texts:
        for line in str(raw or "").splitlines():
            ordinal += 1
            cleaned = " ".join(line.strip().split())
            if not cleaned:
                continue
            if len(cleaned) > 240:
                cleaned = cleaned[:237] + "..."
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            if cleaned.startswith(("==", "--", "PASSED ")):
                continue
            if lowered.startswith(_TB2_CLUE_SKIP_PREFIXES):
                continue
            if any(
                marker in lowered
                for marker in (
                    "/site-packages/",
                    "/usr/lib/python",
                    "urllib3",
                    "requests/adapters.py",
                    "subprocess.py:",
                )
            ):
                continue
            score = 0
            for idx, marker in enumerate(_TB2_CLUE_PRIORITY_MARKERS):
                if marker.lower() in lowered:
                    score = max(score, 100 - idx)
            if score <= 0 and (
                "error" in lowered or "failed" in lowered or "warning" in lowered or "missing" in lowered
            ):
                score = 30
            if score <= 0:
                continue
            seen.add(lowered)
            ranked.append((score, ordinal, cleaned))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [line for _, _, line in ranked[:max_items]]


def _tb2_attempt_meta(trace: "TaskTrace") -> dict[str, Any]:
    runtime_meta = trace.runtime_meta or {}
    tool_trace = trace.tool_trace or []
    commands: list[str] = []
    for step in tool_trace[-3:]:
        if not isinstance(step, dict):
            continue
        tool_input = step.get("tool_input") or {}
        if not isinstance(tool_input, dict):
            continue
        command = str(tool_input.get("command") or "").strip()
        if command:
            commands.append(command[:240])
    verifier_stdout_tail = _trim_text(runtime_meta.get("verifier_stdout_tail"), 800)
    verifier_stderr_tail = _trim_text(runtime_meta.get("verifier_stderr_tail"), 800)
    verifier_clues = _tb2_extract_verifier_clues(verifier_stdout_tail, verifier_stderr_tail)
    reward = runtime_meta.get("reward")
    try:
        reward_value = float(reward) if reward is not None else None
    except (TypeError, ValueError):
        reward_value = None
    if reward_value is not None and reward_value >= 1.0:
        verified_outcome = "Verifier passed with reward 1.0."
    elif reward_value is not None:
        verified_outcome = f"Verifier unresolved with reward {reward_value:g}."
    else:
        verified_outcome = "Verifier reward missing."
    return {
        "task_source": "terminal_bench_2",
        "reward": reward,
        "agent_exit_code": runtime_meta.get("agent_exit_code"),
        "verifier_exit_code": runtime_meta.get("verifier_exit_code"),
        "tool_count": len(tool_trace),
        "recent_commands": commands,
        "last_state_change": commands[-1] if commands else "",
        "verified_outcome": verified_outcome,
        "failure_signature": verifier_clues[0] if verifier_clues else "",
        "verifier_clues": verifier_clues,
        "verifier_stdout_tail": verifier_stdout_tail,
        "verifier_stderr_tail": verifier_stderr_tail,
    }


_POLYGLOT_TEST_FEEDBACK_SANITIZED_KEYS: tuple[str, ...] = ("native_score", "resolved", "status", "test_exit_code")


def _polyglot_test_feedback_attempt_meta(trace: "TaskTrace") -> dict[str, Any] | None:
    """Sanitized ``attempt_1_eval_result`` for the polyglot test-feedback retry loop.

    ``runtime_meta.polyglot_test_feedback_meta.attempt_1_eval_summary`` is the
    TS-side summary of the first-attempt eval (``summarizeEvalResult`` in
    ``polyglot_test_feedback.ts``) — already scalar-only, no raw
    ``test_stdout_tail``/``test_stderr_tail``. Re-filtering to the known
    scalar keys here is defense-in-depth against an upstream shape change;
    it's purely additive research bookkeeping (Aider's ``pass_rate_1``
    analog) and must never influence the final score, which still comes from
    the unchanged post-hoc ``evaluator.evaluate()`` call. Returns ``None``
    when the retry loop didn't run or the summary is absent/malformed, so
    callers merge it via ``_merge_optional_meta`` alongside the tb2 leg.
    """
    runtime_meta = trace.runtime_meta or {}
    if not isinstance(runtime_meta, dict):
        return None
    tf_meta = runtime_meta.get("polyglot_test_feedback_meta")
    if not isinstance(tf_meta, dict):
        return None
    summary = tf_meta.get("attempt_1_eval_summary")
    if not isinstance(summary, dict):
        return None
    sanitized = {k: summary[k] for k in _POLYGLOT_TEST_FEEDBACK_SANITIZED_KEYS if k in summary}
    if not sanitized:
        return None
    return {"attempt_1_eval_result": sanitized}


def _trace_condensed_tb2(trace: "TaskTrace", *, insight_text: str = "(pending reflection)") -> str:
    """Condensed-trace formatter for terminal_bench_2.

    Moved verbatim from the ``_knowledge_trace_condensed`` tb2 leg; lives here
    (alongside the engine attempt-event helpers) because it depends on the
    ``_tb2_attempt_meta`` helper rather than on the orchestrator class.
    """
    tb2_meta = _tb2_attempt_meta(trace)
    verifier_bits: list[str] = []
    if tb2_meta.get("reward") is not None:
        verifier_bits.append(f"reward={tb2_meta['reward']}")
    if tb2_meta.get("verified_outcome"):
        verifier_bits.append(str(tb2_meta["verified_outcome"]))
    verifier_bits.append(f"agent_exit={tb2_meta.get('agent_exit_code')}")
    verifier_bits.append(f"verifier_exit={tb2_meta.get('verifier_exit_code')}")
    # ``failure_signature`` / ``verifier_clues`` are intentionally omitted:
    # they are extracted from the hidden TB2 pytest verifier output (the
    # benchmark holds the tests out and treats an agent reading them as
    # cheating). This condensed trace flows into the NEXT-GEN SOLVER's
    # MEMORY.md seed (runtime/seeding.py) as well as the forum/distill
    # channels, so only the agent's OWN observations (outcome/reward,
    # exit codes, the commands it ran) may carry forward.
    return (
        f"TB2 attempt summary: {' '.join(verifier_bits)}; "
        f"tool_count={tb2_meta.get('tool_count')}; "
        f"recent_commands={tb2_meta.get('recent_commands') or []}; "
        f"Insight: {insight_text or '(pending reflection)'}"
    )


def _attach_engine_source_formatters() -> None:
    """Attach engine-local per-source memory formatters to their specs.

    Mirrors ``kcsi.benchmarks.loaders.attach_benchmark_loaders`` / scoring / approach_diagnosis
    wiring, but lives here because the tb2 ``trace_condensed`` formatter depends
    on the tb2 helpers above. Idempotent.
    """
    from dataclasses import replace as dataclass_replace

    from ..tasks.registry import REGISTRY, register_task_source

    spec = REGISTRY.get("terminal_bench_2")
    if spec is not None and spec.trace_condensed is None:
        register_task_source(dataclass_replace(spec, trace_condensed=_trace_condensed_tb2), replace=True)
    spec = REGISTRY.get("terminal_bench_2")
    if spec is not None and spec.attempt_meta_builder is None:
        register_task_source(dataclass_replace(spec, attempt_meta_builder=_tb2_attempt_meta), replace=True)
    spec = REGISTRY.get("polyglot")
    if spec is not None and spec.attempt_meta_builder is None:
        register_task_source(
            dataclass_replace(spec, attempt_meta_builder=_polyglot_test_feedback_attempt_meta), replace=True
        )


_attach_engine_source_formatters()


def _build_approach_diagnosis(
    *,
    trace: "TaskTrace",
    eval_result: dict[str, Any],
    outcome: str,
    task_source: str = "",
    seed_test_files: bool = False,
) -> str:
    """Build a failure-diagnosis summary instead of storing the raw model output.

    For resolved tasks, stores a short success note.  For failures, stores
    what went wrong: which tests failed, which passed, and the files changed.
    This prevents memory anchoring — agents learn what to AVOID rather than
    being tempted to copy a subtly-wrong patch.
    """
    parts: list[str] = []
    parts.append(f"Outcome: {outcome} (score: {trace.native_score})")

    # Benchmark-specific lines are produced by the task source's registered
    # ``approach_diagnosis`` hook (kcsi.orchestrator.approach_diagnosis); sources
    # without one (polyglot/unknown) fall back to the generic eval-status
    # line. Adding a benchmark no longer requires editing this dispatch.
    spec = resolve_source(task_source)
    formatter = spec.approach_diagnosis if spec is not None else None
    if formatter is not None:
        parts.extend(
            formatter(
                trace=trace,
                eval_result=eval_result,
                outcome=outcome,
                seed_test_files=seed_test_files,
            )
        )
    elif outcome != "resolved":
        status = eval_result.get("status") or eval_result.get("swebench_status") or ""
        if status:
            parts.append(f"Eval status: {status}")

    if outcome != "resolved":
        parts.append(
            "Next attempt: use the concrete evidence above to choose a different fix path or tighten the current one."
        )

    return "\n".join(parts)


def _extract_approach_excerpt(text: str, max_chars: int = 300) -> str:
    """Extract a substantive excerpt from model output, skipping preamble."""
    if not text:
        return ""
    # Skip common preamble lines (reasoning, meta-commentary)
    lines = text.split("\n")
    skip_patterns = re.compile(
        r"^(I'll|I'm |Let me |I need to |I should |I want to |I have |"
        r"Now |OK|Alright|First|Here's|Looking at|Container exited|"
        r"\s*$)",
        re.IGNORECASE,
    )
    start = 0
    for i, line in enumerate(lines):
        if not skip_patterns.match(line.strip()):
            start = i
            break
    # Rejoin from first substantive line (if none found, start stays 0 = full text)
    excerpt = " ".join(l.strip() for l in lines[start:] if l.strip())
    if not excerpt:
        # Fallback: use original text
        excerpt = " ".join(l.strip() for l in lines if l.strip())
    return excerpt[:max_chars].strip()


def _build_carry_forward_attempt_event(runtime_meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(runtime_meta, dict) or not runtime_meta.get("carry_forward"):
        return None
    payload: dict[str, Any] = {
        "carry_forward": True,
        "carry_forward_reason": str(runtime_meta.get("carry_forward_reason") or "best_score_preserved"),
    }
    source_generation = _coerce_int(runtime_meta.get("carry_forward_source_generation"))
    if source_generation is not None:
        payload["carry_forward_source_generation"] = source_generation
    source_agent_id = str(runtime_meta.get("carry_forward_source_agent_id") or "").strip()
    if source_agent_id:
        payload["carry_forward_source_agent_id"] = source_agent_id
    source_score = _coerce_float(runtime_meta.get("carry_forward_source_score"))
    if source_score is not None:
        payload["carry_forward_source_score"] = source_score
    threshold = _coerce_float(runtime_meta.get("carry_forward_threshold"))
    if threshold is not None:
        payload["carry_forward_threshold"] = threshold
    return payload


def _build_attempt_event(
    *,
    native_score: float | None,
    error: str,
    eval_results: dict[str, Any],
    model_output: str = "",
    runtime_meta: dict[str, Any] | None = None,
    seed_test_files: bool = False,
) -> dict[str, Any]:
    """Build structured attempt event with test failure details.

    When ``seed_test_files`` is False (upstream-strict mode), the test-name
    lists are replaced with counts (``tests_still_failing_count`` etc.).
    The raw lists are eval signal — they're persisted into
    ``attempt_history_json`` and surfaced to the next-generation agent
    via the MCP ``query`` tool. In upstream-strict the agent must not
    see those names; counts preserve the swarm-level diagnostic signal.
    Mirrors the ``_best_attempt_summary`` anonymization.
    """
    status = eval_results.get("status") or eval_results.get("swebench_status") or ""
    instance_report = eval_results.get("instance_report", {})
    tests_status = instance_report.get("tests_status", {})
    ftp = tests_status.get("FAIL_TO_PASS", {})
    ptp = tests_status.get("PASS_TO_PASS", {})
    approach_excerpt = _extract_approach_excerpt(model_output or "", max_chars=1000)
    event: dict[str, Any] = {
        "native_score": native_score,
        "resolved": bool(native_score is not None and native_score >= 1.0),
        "status": status,
        "error": error,
        "approach_excerpt": approach_excerpt,
    }
    if seed_test_files:
        event["tests_still_failing"] = ftp.get("failure", [])
        event["tests_now_passing"] = ftp.get("success", [])
        event["tests_regressed"] = ptp.get("failure", [])
        event["tests_skipped"] = [*ftp.get("skipped", []), *ptp.get("skipped", [])]
        event["tests_unobserved"] = [*ftp.get("unknown", []), *ptp.get("unknown", [])]
    else:
        event["tests_still_failing_count"] = len(ftp.get("failure", []))
        event["tests_now_passing_count"] = len(ftp.get("success", []))
        event["tests_regressed_count"] = len(ptp.get("failure", []))
        event["tests_skipped_count"] = len(ftp.get("skipped", [])) + len(ptp.get("skipped", []))
        event["tests_unobserved_count"] = len(ftp.get("unknown", [])) + len(ptp.get("unknown", []))
    if "arc_per_test" in eval_results:
        event["arc_pass_ratio"] = eval_results.get("arc_pass_ratio")
        per_test = eval_results.get("arc_per_test")
        if isinstance(per_test, list):
            event["arc_per_test"] = [
                {k: item[k] for k in ARC_PER_TEST_SAFE_KEYS if k in item} for item in per_test if isinstance(item, dict)
            ]
        else:
            event["arc_per_test"] = per_test
    carry_forward_payload = _build_carry_forward_attempt_event(runtime_meta)
    if carry_forward_payload is not None:
        event.update(carry_forward_payload)
    return event


def _knowledge_attempt_external_id(*, task_id: str, agent_id: str, generation: int) -> str:
    """Stable per-execution-attempt id for ``KnowledgeStore.record_attempt``.

    ``_eval_one_attempt`` runs exactly once per ``(agent, task_id)`` pair per
    generation, so this triple uniquely identifies one evaluated
    :class:`~kcsi.models.TaskTrace`. Both the execution phase's early
    resume-safety attempt write (``execution_phase._persist_knowledge_attempt_early``)
    and the engine's later, richer write for the same execution attempt
    (``engine._persist_task_memory_record``) pass this same id so the second
    write can supersede the first in place (``external_id=..., supersede=True``)
    instead of being silently skipped.
    """
    return f"attempt:{task_id}:{agent_id}:{generation}"


def _score_from_eval(eval_result: dict[str, Any], *, task: TaskSpec | None = None) -> float | None:
    """Dataset-aware score extractor (higher is better).

    Scoring policy:
    - swebench:
      - Binary: resolved iff ALL f2p pass AND ALL p2p pass.
      - Fallback to instance_report.resolved, then run_summary membership.
    - generic fallback:
      - native_score, resolved, instance_report.resolved, pass.
    """
    task_source = str(((task.metadata or {}).get("task_source") if task else "") or "").strip().lower()

    # A task source may register a dataset-aware scorer on its spec
    # (kcsi.orchestrator.scoring); sources without one use the generic
    # native_score/resolved/pass precedence. No per-source dispatch here.
    spec = resolve_source(task_source)
    scorer = spec.score_from_eval if spec is not None else None
    if scorer is not None:
        return scorer(eval_result, task=task)
    return score_from_eval_results(eval_result)

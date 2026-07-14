"""Deterministic helpers for the knowledge-transfer (KT) adapter.

Pure functions that shape a shared distilled cross-task bundle and the current
task into the payloads, normalized memo, and deterministic asset fallback the
orchestrator uses when adapting transferred knowledge per task.

Extracted verbatim from ``ksi.orchestrator.engine`` to shrink the
engine module. These functions take only plain data / ``TaskSpec`` and never
reference orchestrator instance state, so there is no import cycle (the engine
imports this module, not vice versa). The LLM-bound steps (the adapter call,
its repair retry, and the memo cache) remain on the orchestrator.
"""

from __future__ import annotations

from typing import Any

from ..distillation.types import CROSS_TASK_INSIGHT_FIELDS
from ..models import TaskSpec


def adapter_item_text(value: Any, *, max_chars: int = 220) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        text = str(value.get("text") or "").strip()
        if text:
            parts.append(text)
        applies = str(value.get("applies_when") or "").strip()
        if applies:
            parts.append(f"Applies when: {applies}")
        blocked = str(value.get("does_not_apply_when") or "").strip()
        if blocked:
            parts.append(f"Not when: {blocked}")
        confidence = str(value.get("confidence") or "").strip().lower()
        if confidence in {"high", "medium", "low"}:
            parts.append(f"Confidence: {confidence}")
        joined = " | ".join(parts)
    else:
        joined = str(value or "").strip()
    if len(joined) > max_chars:
        return joined[: max_chars - 3].rstrip() + "..."
    return joined


def adapter_bundle_payload(cross_task: dict[str, Any]) -> dict[str, list[str]]:
    fields = CROSS_TASK_INSIGHT_FIELDS
    payload: dict[str, list[str]] = {}
    for field in fields:
        raw_values = cross_task.get(field)
        if not isinstance(raw_values, list):
            continue
        items = [adapter_item_text(item) for item in raw_values]
        items = [item for item in items if item]
        if items:
            payload[field] = items
    return payload


def adapter_task_payload(task: TaskSpec) -> dict[str, Any]:
    metadata = task.metadata or {}
    task_source = str(metadata.get("task_source") or "").strip().lower()
    payload: dict[str, Any] = {
        "task_id": task.id,
        "task_source": task_source,
        "task_prompt": str(task.prompt or "").strip(),
    }
    # Information-parity invariant: the adapter memo is rendered into the
    # recipient's MEMORY.md, so its payload MUST be a subset of what the solver
    # itself sees. ARC solvers see train pairs (input+output) and test INPUTS
    # only — the test outputs are the hidden answer (stripped from the
    # mount), so they must never reach the memo-builder. Pass test inputs only.
    train_pairs = metadata.get("arc_train_pairs")
    test_inputs = metadata.get("arc_test_inputs")
    if not isinstance(test_inputs, list):
        raw_pairs = metadata.get("arc_eval_test_pairs")
        if not isinstance(raw_pairs, list):
            raw_pairs = metadata.get("arc_test_pairs")
        if isinstance(raw_pairs, list):
            test_inputs = [{"input": p.get("input")} for p in raw_pairs if isinstance(p, dict)]
    if isinstance(train_pairs, list):
        payload["train_pairs"] = train_pairs
    if isinstance(test_inputs, list):
        payload["test_inputs"] = test_inputs
    if task_source == "polyglot":
        # Polyglot solvers see the problem statement and starter code but NOT
        # the hidden grader tests/build files (no test feedback regime). Those
        # must not enter the memo either.
        payload["language"] = str(metadata.get("language") or "").strip()
        payload["exercise_name"] = str(metadata.get("exercise_name") or "").strip()
        payload["problem_statement"] = str(metadata.get("problem_statement") or "").strip()
        payload["starter_code"] = metadata.get("starter_code") if isinstance(metadata.get("starter_code"), dict) else {}
    return payload


def adapter_clean_list(raw_values: Any, limit: int) -> list[str]:
    if not isinstance(raw_values, list):
        return []
    out: list[str] = []
    for value in raw_values:
        text = str(value or "").strip()
        if not text:
            continue
        if len(text) > 240:
            text = text[:237].rstrip() + "..."
        out.append(text)
        if len(out) >= limit:
            break
    return out


def normalize_memo(parsed: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None
    candidate_plan = str(parsed.get("candidate_plan") or "").strip()
    rationale = str(parsed.get("knowledge_use_rationale") or "").strip()
    if len(candidate_plan) > 500:
        candidate_plan = candidate_plan[:497].rstrip() + "..."
    if len(rationale) > 500:
        rationale = rationale[:497].rstrip() + "..."
    memo = {
        "relevant_constraints": adapter_clean_list(parsed.get("relevant_constraints"), 3),
        "relevant_heuristics": adapter_clean_list(parsed.get("relevant_heuristics"), 4),
        "pitfalls_to_avoid": adapter_clean_list(parsed.get("pitfalls_to_avoid"), 3),
        "checks_before_submit": adapter_clean_list(parsed.get("checks_before_submit"), 3),
        "candidate_plan": candidate_plan,
        "knowledge_use_rationale": rationale,
    }
    if (
        not any(
            memo[key]
            for key in (
                "relevant_constraints",
                "relevant_heuristics",
                "pitfalls_to_avoid",
                "checks_before_submit",
            )
        )
        and not memo["candidate_plan"]
    ):
        return None
    return memo


def rank_bundle_items(values: list[str], limit: int) -> list[str]:
    def score(text: str) -> tuple[int, int]:
        lower = text.lower()
        confidence = 0
        if "confidence: high" in lower:
            confidence = 3
        elif "confidence: medium" in lower:
            confidence = 2
        elif "confidence: low" in lower:
            confidence = 1
        return (confidence, -len(text))

    ranked = sorted((v for v in values if str(v).strip()), key=score, reverse=True)
    return ranked[:limit]


def fallback_candidate_plan(task: TaskSpec) -> str:
    task_source = str((task.metadata or {}).get("task_source") or "").strip().lower()
    if task_source == "polyglot":
        return (
            "Read the task specification and starter code first, keep only the candidate heuristics that fit the "
            "language and API contract, rule out bad implementation choices with the listed pitfalls, and run the "
            "listed checks against the test expectations before finalizing the solution."
        )
    return (
        "Inspect the train pairs first, keep only the candidate heuristics that fit all training examples, "
        "rule out bad transformations with the listed pitfalls, and run the listed checks before submitting."
    )


def fallback_rationale(task: TaskSpec) -> str:
    task_source = str((task.metadata or {}).get("task_source") or "").strip().lower()
    if task_source == "polyglot":
        return (
            "This fallback memo preserves the strongest reusable coding-task knowledge directly from the shared "
            "distilled asset when the adapter model fails to emit valid JSON."
        )
    return (
        "This fallback memo preserves the strongest reusable items directly from the shared distilled asset "
        "when the adapter model fails to emit valid JSON."
    )


def build_fallback_memo(*, task: TaskSpec, cross_task: dict[str, Any]) -> dict[str, Any] | None:
    asset_payload = adapter_bundle_payload(cross_task)
    if not asset_payload:
        return None
    constraints = rank_bundle_items(asset_payload.get("confirmed_constraints", []), 3)
    heuristics = rank_bundle_items(asset_payload.get("transferable_insights", []), 4)
    pitfalls = rank_bundle_items(
        list(asset_payload.get("rejected_hypotheses", [])) + list(asset_payload.get("pitfalls", [])),
        3,
    )
    checks = rank_bundle_items(asset_payload.get("checks", []), 3)
    if not any((constraints, heuristics, pitfalls, checks)):
        return None
    return {
        "relevant_constraints": constraints,
        "relevant_heuristics": heuristics,
        "pitfalls_to_avoid": pitfalls,
        "checks_before_submit": checks,
        "candidate_plan": fallback_candidate_plan(task),
        "knowledge_use_rationale": fallback_rationale(task),
        "_memo_source": "asset_fallback",
    }

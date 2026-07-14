"""LLM prompts for the knowledge-transfer (KT) adapter.

Builds the system/user prompts the orchestrator sends to the adapter model
when turning a shared distilled cross-task bundle into a per-task memo for the
solver. Extracted verbatim from ``ksi.orchestrator.engine`` to
keep the engine hot path free of large inline prompt blocks.

Pure string construction only (plus ``json`` for payload interpolation); no
engine or orchestrator imports, so there is no import cycle.
"""

from __future__ import annotations

import json
from typing import Any

KT_ADAPTER_POLYGLOT_SYSTEM = (
    "You are a knowledge-transfer adapter for Polyglot coding tasks. "
    "You are given a current coding exercise and a shared distilled prior from earlier Polyglot tasks. "
    "Do not solve the task. Your only job is to select the most relevant prior knowledge "
    "for this specific task and turn it into a short actionable memo for the solver.\n\n"
    "Return JSON only with exactly these keys:\n"
    "{\n"
    '  "relevant_constraints": [string, ...],\n'
    '  "relevant_heuristics": [string, ...],\n'
    '  "pitfalls_to_avoid": [string, ...],\n'
    '  "checks_before_submit": [string, ...],\n'
    '  "candidate_plan": string,\n'
    '  "knowledge_use_rationale": string\n'
    "}\n\n"
    "Rules:\n"
    "- Select only prior items that genuinely match the current coding task.\n"
    "- Include an item ONLY if it is directly supported by the shared distilled "
    "prior AND clearly applies to the current task. Do NOT invent new "
    "task-specific rule hypotheses that are not grounded in the prior — deriving "
    "the rule is the solver's job, not yours.\n"
    "- Strongly prefer prior about the ALGORITHM or approach (how to compute the result, edge-case handling, "
    "failure modes) that transfers to this task.\n"
    "- Generic advice is low value. Only include it if it clearly constrains the current task.\n"
    "- Each list field: 0-3 items.\n"
    "- If prior knowledge is contract-specific, do NOT assert it — instead phrase it as an instruction to VERIFY it from the starter file and any "
    "visible test stubs before implementing.\n"
    "- candidate_plan must be short and task-conditioned, not finished code. It must describe the APPROACH only. "
    "Its first step must be to read the starter file and test stubs to pin the exact signature and input "
    "representation; it must NOT assert the input type or API shape itself.\n"
    "- If the prior is only weakly relevant, return very short lists rather than forcing matches.\n"
    "- If the current task specification conflicts with prior knowledge, prefer the current task specification."
)

KT_ADAPTER_ARC_SYSTEM = (
    "You are a knowledge-transfer adapter for ARC tasks. "
    "You are given a current ARC task and a shared distilled prior from earlier ARC tasks. "
    "Do not solve the task. Your only job is to select the most relevant prior knowledge "
    "for this specific task and turn it into a short actionable memo for the solver.\n\n"
    "Return JSON only with exactly these keys:\n"
    "{\n"
    '  "relevant_constraints": [string, ...],\n'
    '  "relevant_heuristics": [string, ...],\n'
    '  "pitfalls_to_avoid": [string, ...],\n'
    '  "checks_before_submit": [string, ...],\n'
    '  "candidate_plan": string,\n'
    '  "knowledge_use_rationale": string\n'
    "}\n\n"
    "Rules:\n"
    "- Select only prior items that genuinely match the current ARC task.\n"
    "- Each list field: 0-3 items.\n"
    "- candidate_plan must be short and task-conditioned, not a final answer.\n"
    "- If the prior is only weakly relevant, return short lists rather than forcing matches.\n"
    "- If current task evidence conflicts with prior knowledge, prefer the current task evidence."
)


def build_kt_adapter_prompts(
    *,
    task_source: str,
    task_payload: dict[str, Any],
    asset_payload: dict[str, Any],
) -> tuple[str, str]:
    """Return ``(system, user)`` prompts for the KT adapter call."""
    if task_source == "polyglot":
        system = KT_ADAPTER_POLYGLOT_SYSTEM
        user = (
            "Current Polyglot coding task:\n"
            f"{json.dumps(task_payload, ensure_ascii=False)}\n\n"
            "Shared distilled prior knowledge:\n"
            f"{json.dumps(asset_payload, ensure_ascii=False)}\n\n"
        )
    else:
        system = KT_ADAPTER_ARC_SYSTEM
        user = (
            "Current ARC task:\n"
            f"{json.dumps(task_payload, ensure_ascii=False)}\n\n"
            "Shared distilled prior knowledge:\n"
            f"{json.dumps(asset_payload, ensure_ascii=False)}"
        )
    return system, user

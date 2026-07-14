"""Typed inputs and outputs for the distillation module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, overload


class Evidence(TypedDict, total=False):
    task_id: str
    post_id: int
    quote: str


class Insight(TypedDict, total=False):
    """Structured bundle item.

    Both bundle types (PerTaskBundle, CrossTaskBundle) carry items as either
    legacy strings (treated as bare text) or structured Insight dicts. The
    render path in src/ksi/runtime/seeding.py handles both shapes.

    A well-formed Insight has:
    - ``text``: the actionable claim ("when X, do Y")
    - ``applies_when``: concrete condition under which the claim holds
    - ``does_not_apply_when``: concrete counterexample / boundary
    - ``evidence``: list of Evidence dicts citing the post(s) that support it
    - ``confidence``: "high" | "medium" | "low"
    """

    text: str
    applies_when: str
    does_not_apply_when: str
    evidence: list[Evidence]
    confidence: str


# Legacy items (str) and structured items (Insight dict) coexist; the render
# layer auto-detects shape via isinstance(item, str). The union (not ``Any``)
# preserves that contract for type-checkers while still accepting both JSON
# shapes from the distill LLM and old DB rows.
BundleItem = str | Insight


def truncate_at_boundary(text: str, cap: int) -> str:
    """Truncate to <= cap chars at a sentence end (past 60% of cap), else the
    last word boundary (past 60% of cap), else a hard cut at cap. The old
    hard slice cut operative clauses mid-word. An earlier version fell back to the
    first sentence/word boundary found ANYWHERE in the cut text when neither
    was past the floor, which could collapse the cap down to a handful of
    characters for text with one early boundary followed by a long unbroken
    tail (e.g. a stack trace or URL) -- worse than the hard cut it replaced.

    Lives here (rather than in ``per_task`` or ``prompts``) so both modules can
    import it without a circular import: ``per_task`` imports prompt builders
    from ``prompts``, so ``prompts`` cannot import back from ``per_task``.
    """
    if len(text) <= cap:
        return text
    cut = text[:cap]
    floor = int(cap * 0.6)
    sentence_end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "), cut.rfind(".\n"))
    if sentence_end >= floor:
        return cut[: sentence_end + 1]
    space = cut.rfind(" ")
    if space >= floor:
        return cut[:space]
    # No boundary within the floor..cap window (e.g. a long unbroken tail
    # after an early, isolated boundary): hard-cut at cap rather than
    # collapsing back to that early boundary and discarding most of the
    # budget (falling back to the first boundary found ANYWHERE in ``cut``
    # could shrink a 2000-char budget to single digits).
    return cut


def coerce_positive_int(value: Any) -> int | None:
    """Coerce a value to a positive int, or None. Rejects bools (``True``/
    ``False`` are ints in Python but never valid post ids). Single source of
    truth for the lenient parser (``per_task``) so an edge-case fix can't
    drift between copies."""
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError covers non-finite floats: ``int(float("inf"))`` raises
        # it (NaN raises ValueError, string "Infinity" raises ValueError), so a
        # malformed/non-finite evidence id is dropped rather than crashing the
        # lenient parser.
        return None
    return result if result > 0 else None


@dataclass
class PerTaskBundle:
    task_id: str
    transferable_insights: list[BundleItem]
    pitfalls: list[BundleItem]
    checks: list[BundleItem]
    evidence_post_ids: list[int]
    confirmed_constraints: list[BundleItem] = field(default_factory=list)
    rejected_hypotheses: list[BundleItem] = field(default_factory=list)
    next_steps: list[BundleItem] = field(default_factory=list)
    # Optional raw payload carried from the stored bundle. ``to_dict()``
    # intentionally excludes it — persistence decides which representation
    # to store.
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "transferable_insights": list(self.transferable_insights),
            "confirmed_constraints": list(self.confirmed_constraints),
            "rejected_hypotheses": list(self.rejected_hypotheses),
            "pitfalls": list(self.pitfalls),
            "checks": list(self.checks),
            "next_steps": list(self.next_steps),
            "evidence_post_ids": list(self.evidence_post_ids),
        }


@dataclass
class CrossTaskBundle:
    transferable_insights: list[BundleItem]
    pitfalls: list[BundleItem]
    checks: list[BundleItem]
    evidence_post_ids: list[int]
    confirmed_constraints: list[BundleItem] = field(default_factory=list)
    rejected_hypotheses: list[BundleItem] = field(default_factory=list)
    next_steps: list[BundleItem] = field(default_factory=list)
    # See PerTaskBundle.raw — kept symmetric for a sibling increment; no
    # cross-task alternate format exists yet.
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "transferable_insights": list(self.transferable_insights),
            "confirmed_constraints": list(self.confirmed_constraints),
            "rejected_hypotheses": list(self.rejected_hypotheses),
            "pitfalls": list(self.pitfalls),
            "checks": list(self.checks),
            "next_steps": list(self.next_steps),
            "evidence_post_ids": list(self.evidence_post_ids),
        }


CROSS_TASK_INSIGHT_FIELDS: tuple[str, ...] = (
    "transferable_insights",
    "confirmed_constraints",
    "rejected_hypotheses",
    "pitfalls",
    "checks",
    "next_steps",
)
"""Canonical names of the insight-bearing list fields in a cross-task
distillation bundle. Used by the extractor, engine seed-injection path,
and prompt-rendering layer so the schema has a single source of truth.

The companion field ``evidence_post_ids`` (list[int]) is not included;
it's bundle-level evidence, not a list of Insight items."""


@dataclass
class DistillLLMResult:
    """Shared carrier for a distillation LLM response that may include a
    provider-validated structured dict.

    ``text`` is the raw (when structured, guaranteed-JSON) response; ``parsed``
    is the provider-validated dict, or ``None`` when the provider declined or
    returned no schema-shaped payload (the distill path then falls back to the
    lenient free-text parser). This is the single canonical "structured result"
    type: the engine adapter returns it, and ``per_task._call_llm`` detects it
    by ``isinstance`` — no duck-typed ``hasattr(result, "text")`` matching.
    """

    text: str
    parsed: dict[str, Any] | None


# ---------------------------------------------------------------------------
# LLMCallable contract  (read this before adding a new provider)
# ---------------------------------------------------------------------------
# A distillation LLM callable is invoked as ``llm(system, user)`` and MAY also
# be passed ``json_schema=<{"name", "schema"}>`` to request provider structured
# output. To participate in the structured-output path, a provider caller must:
#
#   1. accept a keyword-only ``json_schema`` parameter on its ``call(...)``;
#   2. when a schema is passed, return ``(text, usage, parsed_dict | None)``
#      (a 3-tuple) — ``parsed_dict`` is the provider-validated object;
#   3. advertise the capability with a class attribute ``supports_json_schema
#      = True`` (the engine gates schema use on this — without it the schema is
#      never forwarded and the distiller silently uses the free-text parser).
#
# See ``ksi.runtime.llm.AnthropicLLMCaller`` / ``OpenAILLMCaller`` for the
# two reference implementations, ``ksi.protocols.LLMCaller`` for the typed
# contract, and ``tests/test_llm_caller.py`` for the conformance tests to copy.
#
# Callables that do NOT accept ``json_schema`` (legacy ``(system, user) -> str``
# stubs, unknown providers) still work: ``per_task._call_llm`` falls back to a
# schema-less call. The engine adapter additionally returns a
# ``DistillLLMResult``. The annotation stays loose to admit all of these.
class LLMCallable(Protocol):
    """Structural type for a distillation LLM callable (see the contract notes
    above). Loose by design — the return is ``Any`` because three runtime shapes
    are accepted (a ``str``, a ``(text, usage[, parsed])`` tuple, or a
    ``DistillLLMResult``). Legacy ``(system, user) -> str`` stubs that don't
    accept ``json_schema`` still work at runtime via ``per_task._call_llm``'s
    fallback even though they don't match this Protocol structurally."""

    @overload
    def __call__(self, system: str, user: str) -> Any: ...

    @overload
    def __call__(self, system: str, user: str, *, json_schema: dict[str, Any]) -> Any: ...

    def __call__(self, system: str, user: str, *, json_schema: dict[str, Any] | None = ..., **kwargs: Any) -> Any: ...


@dataclass
class DistillInput:
    generation: int
    task_ids: list[str]
    # KnowledgeStore-like: must support query_task(task_id, generation, entry_types)
    #                     and query_generation(generation, source_phase, entry_types)
    knowledge_store: Any
    llm: LLMCallable
    # Optional per-phase LLM overrides.  When non-None, these are used in
    # place of ``llm`` for per-task and cross-task distillation respectively,
    # allowing callers (e.g. the engine) to wire separate model overrides
    # for each phase.  ``llm`` remains the fallback for any phase whose
    # override is left at ``None``.
    llm_per_task: LLMCallable | None = None
    llm_cross_task: LLMCallable | None = None
    # Optional domain hint used to bias the distill prompts toward
    # benchmark-specific concrete primitives. Expected values mirror the
    # CLI ``--task-source`` flag ("arc", "swebench_pro", "polyglot") but
    # arbitrary strings are accepted (unknown values fall back to a
    # generic hint).
    task_source: str | None = None
    # Optional override of the JSON schema requested from structured-output
    # providers. Defaults to ``DISTILL_BUNDLE_JSON_SCHEMA`` when None; set it to
    # experiment with a stricter/reduced bundle schema without editing core
    # modules. Forwarded to ``distill_one_task`` / ``distill_cross_task``.
    bundle_schema: dict[str, Any] | None = None
    # Experiment scope for store queries. When set, the transfer-bridge
    # ``_latest_stored_transferables`` lookup filters to this experiment's
    # runs; None uses the store default. Distillation otherwise does not need
    # it (loaders pass through the store's default_experiment).
    experiment: str | None = None
    # Cross-task target-conditioning: when True, distill one cross-task bundle
    # PER attempted task, each conditioned on that task's prompt (see
    # ``target_task_prompts``), instead of one broadcast bundle. Default False
    # keeps the legacy single-bundle behavior for any programmatic caller.
    cross_task_target_conditioning: bool = False
    # task_id -> full task prompt, used only when cross_task_target_conditioning
    # is True. A task absent from this map is treated as a degraded/missing
    # target and skipped rather than silently producing an unconditioned bundle.
    target_task_prompts: dict[str, str] | None = None
    # Optional explicit target ids for target-conditioned cross-task distill.
    # When None, the distiller falls back to ``unsolved_task_ids``/``task_ids``.
    # This intentionally separates "tasks that should get per-task distill"
    # from "labels that will receive cross-task guidance at seed time" (hold-out
    # probes and --no-drop-solved retained tasks are seed targets, but not
    # necessarily per-task learning targets).
    cross_task_target_ids: list[str] | None = None
    # Per-target relevance selection (opt-in; only meaningful under
    # cross_task_target_conditioning). When False (default), the shared forum
    # history is trimmed ONCE against the largest target so every target
    # distills from a byte-identical post set → shared cache_prefix (the default
    # published behavior). When True, EACH target trims its own
    # relevance-ranked post set from the full history, so the post set (and
    # thus the prompt bytes) differs per target — defeating the cross-target
    # prompt cache in exchange for per-target relevance fidelity. See
    # distiller.distill() for the tradeoff.
    cross_task_per_target_selection: bool = False


@dataclass
class DistillOutput:
    per_task: dict[str, PerTaskBundle]
    cross_task: CrossTaskBundle | None
    # Count of distillation sub-failures this generation (per-task distill
    # raised + caught, or cross-task produced no bundle despite posts). Surfaced
    # by the engine into ``knowledge_phase_health`` so a degraded generation is
    # distinguishable from a healthy one in the results JSON.
    failures: int = 0
    # Populated ONLY when cross_task_target_conditioning was on: one cross-task
    # bundle per attempted task, keyed by task id. When set, ``cross_task`` is
    # None; when None, ``cross_task`` holds the single broadcast bundle.
    cross_task_by_task: dict[str, CrossTaskBundle] | None = None

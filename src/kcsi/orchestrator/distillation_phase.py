"""Distillation phase service for the generational orchestrator."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, cast

from ..distillation.types import CROSS_TASK_INSIGHT_FIELDS
from ..errors import AuthenticationFailure

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .engine import GenerationalOrchestrator

log = logging.getLogger(__name__)


def _bundle_persist_dict(bundle: Any) -> dict[str, Any]:
    """Resolve the dict to persist for a distilled per-task bundle."""
    return cast(dict[str, Any], bundle.to_dict())


def _bundle_embed_text(bundle: dict[str, Any]) -> str:
    """Flatten a distill bundle into a single text blob for embedding."""
    parts: list[str] = []
    for key in CROSS_TASK_INSIGHT_FIELDS:
        for item in bundle.get(key) or []:
            text = str(item).strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


@dataclass(frozen=True)
class DistillationCollaborators:
    """Explicit dependencies for the distillation phase service body."""

    config: Any
    knowledge: Any | None
    tasks_by_id: dict[str, Any]
    record_phase_failure: Callable[..., None]  # engine._record_knowledge_phase_failure
    record_distill_result: Callable[..., None]  # engine._record_distill_generation_result
    maybe_embed: Callable[[str], list[float] | None]  # engine._maybe_embed
    maybe_embed_batch: Callable[[list[str]], list[list[float] | None]]  # engine._maybe_embed_batch
    make_distill_llm: Callable[..., Any]  # engine._make_distill_llm (keyword-only args)
    is_holdout: Callable[[str], bool]


@dataclass(frozen=True)
class DistillationPhaseInput:
    """Inputs required to distill one generation's fresh task evidence."""

    generation: int
    task_ids: list[str]


@dataclass(frozen=True)
class DistillationPhaseResult:
    """Observable result of a distillation phase invocation."""

    attempted_task_ids: tuple[str, ...] = ()
    persisted_per_task: int = 0
    persisted_cross_task: bool = False


@dataclass
class EngineDistillationPhaseService:
    """Engine-backed implementation of the distillation phase."""

    engine: "GenerationalOrchestrator"

    def _collaborators(self) -> DistillationCollaborators:
        engine = self.engine
        return DistillationCollaborators(
            config=engine.config,
            knowledge=engine._knowledge,
            tasks_by_id=getattr(engine, "_tasks_by_id", {}) or {},
            record_phase_failure=engine._record_knowledge_phase_failure,
            record_distill_result=engine._record_distill_generation_result,
            maybe_embed=engine._maybe_embed,
            maybe_embed_batch=engine._maybe_embed_batch,
            make_distill_llm=engine._make_distill_llm,
            is_holdout=engine._is_holdout,
        )

    def run(self, phase_input: DistillationPhaseInput) -> DistillationPhaseResult:
        """Phase 4: run the distillation module and persist bundles.

        Builds ``DistillInput`` (current generation, task ids that had attempts
        this generation, the KnowledgeStore, and LLM adapters), calls
        :func:`kcsi.distillation.distill`, and persists per-task and cross-task
        bundles via :meth:`KnowledgeStore.record_distillation` with the
        appropriate scope. Failures inside the distillation module are logged
        and swallowed — the pipeline proceeds best-effort.
        """
        collab = self._collaborators()
        generation = int(phase_input.generation)
        all_task_ids = list(phase_input.task_ids)
        task_ids = list(all_task_ids)

        if collab.config.no_memory:
            log.info("[ENGINE] Skipping distill phase (--no-memory disables distillation)")
            return DistillationPhaseResult()
        if collab.knowledge is None:
            log.info("[ENGINE] Skipping distill phase (knowledge DB not configured)")
            return DistillationPhaseResult()
        # Hold-out probe tasks are excluded from learning: strip their ids
        # before the solved-set query and DistillInput.
        task_ids = [tid for tid in task_ids if not collab.is_holdout(tid)]
        if not task_ids:
            log.info("[ENGINE] Skipping distill phase (no task ids)")
            return DistillationPhaseResult()
        if not collab.config.distill_enabled:
            log.info("[ENGINE] distill disabled via config; skipping")
            return DistillationPhaseResult()

        # Import inside the method so tests can monkeypatch
        # ``kcsi.distillation.distill`` before call time.
        from .. import distillation as _distill_pkg
        from ..distillation.types import DistillInput
        from ..memory.knowledge_store import CROSS_TASK_SENTINEL

        experiment = collab.config.experiment_name

        # Always build labeled per-phase callables so distill-time tokens
        # get attributed correctly in the lifecycle accumulator.
        per_task_model = collab.config.distill_per_task_model or None
        cross_task_model = collab.config.distill_cross_task_model or None
        llm_per_task = collab.make_distill_llm(
            generation=generation,
            phase="distill_per_task",
            model_override=per_task_model,
        )
        llm_cross_task = collab.make_distill_llm(
            generation=generation,
            phase="distill_cross_task",
            model_override=cross_task_model,
        )

        # Resolve a domain hint from the first task whose metadata carries a
        # non-empty ``task_source``. This is best-effort and optional — the
        # distill prompts fall back to a generic hint when unset.
        task_source_hint: str | None = None
        tasks_by_id = collab.tasks_by_id
        for tid in task_ids:
            task = tasks_by_id.get(tid)
            if task is None:
                continue
            task_source = str((task.metadata or {}).get("task_source") or "").strip()
            if task_source:
                task_source_hint = task_source
                break

        # V2 item 4: skip per-task distill for solved tasks. Next-gen agents
        # won't see those tasks (--drop-solved), so the distilled bundle would
        # be wasted spend. Compute solved-set with one bulk knowledge-store
        # query over all task_ids across all gens up to current — a task is
        # solved if ANY attempt resolved=true or score >= solved_threshold.
        unsolved_task_ids: list[str] | None = None
        newly_solved_task_ids: list[str] | None = None
        try:
            solved_set: set[str] = set()
            threshold = float(getattr(collab.config, "solved_threshold", 1.0) or 1.0)
            if collab.knowledge is not None:
                solved_set = collab.knowledge.solved_task_ids(
                    list(task_ids),
                    threshold=threshold,
                    experiment=experiment,
                )
            unsolved_task_ids = [tid for tid in task_ids if tid not in solved_set]
            # Transfer bridge (KCSI_TRANSFER_BRIDGE): under --drop-solved a
            # previously-solved task is never dispatched, so solved ∩
            # this-gen-dispatched ≡ newly-solved-this-gen. task_ids is already
            # hold-out-filtered above. With --no-drop-solved that equivalence
            # breaks (a solved task stays dispatched forever and would get one
            # win distill per generation, unbounded) — leave it None.
            if getattr(collab.config, "drop_solved", True):
                newly_solved_task_ids = [tid for tid in task_ids if tid in solved_set]
            else:
                from ..distillation.distiller import _transfer_bridge_enabled

                if _transfer_bridge_enabled():
                    log.info(
                        "[ENGINE] transfer bridge win extraction requires --drop-solved; skipping (gen=%d)",
                        generation,
                    )
            if len(unsolved_task_ids) < len(task_ids):
                log.info(
                    "[ENGINE] distill: skipping per-task bundles for %d/%d solved tasks (gen=%d)",
                    len(task_ids) - len(unsolved_task_ids),
                    len(task_ids),
                    generation,
                )
        except Exception as exc:
            log.warning(
                "[ENGINE] could not compute solved-set for distill skip (gen=%d): %s — distilling all tasks",
                generation,
                exc,
            )
            unsolved_task_ids = None
            newly_solved_task_ids = None

        # Target-conditioning: distill one cross-task bundle per seed target,
        # each conditioned on that task's full prompt. The per-task learning set
        # stays hold-out/solved-filtered, but hold-out probes and --no-drop-solved
        # retained solved tasks still need cross-task guidance at seed time.
        conditioning = bool(getattr(collab.config, "cross_task_distill_target_conditioning", True))
        # Opt-in per-target relevance selection. Default False
        # preserves the shared-set, byte-identical-prefix behavior; True
        # trades the cross-target cache for per-target relevance (see distiller).
        per_target_selection = bool(getattr(collab.config, "cross_task_distill_per_target_selection", False))
        target_task_prompts: dict[str, str] | None = None
        cross_task_target_ids: list[str] | None = None
        if conditioning:
            if not getattr(collab.config, "drop_solved", True):
                source_ids = all_task_ids
            else:
                source_ids = list(unsolved_task_ids) if unsolved_task_ids is not None else list(task_ids)
                source_ids.extend(tid for tid in all_task_ids if collab.is_holdout(tid))
            cross_task_target_ids = list(dict.fromkeys(source_ids))
            target_task_prompts = {}
            for tid in cross_task_target_ids:
                task = tasks_by_id.get(tid)
                if task is None:
                    continue
                target_task_prompts[tid] = str(getattr(task, "prompt", "") or "")

        try:
            out = _distill_pkg.distill(
                DistillInput(
                    generation=generation,
                    task_ids=list(task_ids),
                    knowledge_store=collab.knowledge,
                    llm=llm_per_task,
                    llm_per_task=llm_per_task,
                    llm_cross_task=llm_cross_task,
                    task_source=task_source_hint,
                    experiment=experiment,
                    cross_task_target_conditioning=conditioning,
                    cross_task_per_target_selection=per_target_selection,
                    target_task_prompts=target_task_prompts,
                    cross_task_target_ids=cross_task_target_ids,
                ),
                unsolved_task_ids=unsolved_task_ids,
                newly_solved_task_ids=newly_solved_task_ids,
            )
        except AuthenticationFailure:
            # Auth failures are fatal: swallowing them would silently disable
            # knowledge improvement for the rest of the campaign at full token
            # cost. Propagate so the run aborts loudly.
            raise
        except Exception as exc:
            log.warning(
                "[ENGINE] distill phase failed (gen=%d): %s — skipping persistence",
                generation,
                exc,
            )
            collab.record_phase_failure(generation, "distill_failures")
            # The whole distill() call failed: attempted work, persisted nothing
            # — a fully-zeroed generation.
            collab.record_distill_result(generation, fully_zeroed=True, failures=len(task_ids))
            return DistillationPhaseResult(attempted_task_ids=tuple(task_ids))

        # Surface the distiller's internal per-task / cross-task sub-failures
        # (caught + continued inside distill()) so they aren't invisible.
        collab.record_phase_failure(generation, "distill_failures", getattr(out, "failures", 0))

        persisted_per_task = 0
        persistence_failures = 0
        # Persist per-task bundles (see _bundle_persist_dict). Pre-compute all
        # bundle embeddings in ONE batched model.encode instead of a per-task
        # embed() before each write — ~3-8× faster on CPU for many bundles
        # (performance). Order is preserved so each embedding aligns with its
        # bundle.
        per_task_dicts = [(tid, _bundle_persist_dict(bundle)) for tid, bundle in (out.per_task or {}).items()]
        per_task_embeddings = collab.maybe_embed_batch([_bundle_embed_text(bd) for _, bd in per_task_dicts])
        for (tid, bundle_dict), embedding in zip(per_task_dicts, per_task_embeddings):
            try:
                collab.knowledge.record_distillation(
                    task_id=tid,
                    generation=generation,
                    bundle=bundle_dict,
                    scope="per_task",
                    experiment=experiment,
                    embedding=embedding,
                )
                persisted_per_task += 1
            except Exception as exc:
                log.warning(
                    "[ENGINE] record_distillation(per_task, task=%s) failed: %s",
                    tid,
                    exc,
                )
                persistence_failures += 1
                collab.record_phase_failure(generation, "distill_failures")
        # Persist cross-task bundle(s). Under target-conditioning, one bundle
        # per downstream seed target keyed by its own task_id; otherwise the
        # single broadcast bundle under CROSS_TASK_SENTINEL (legacy).
        persisted_cross_task = False
        cross_by_task = getattr(out, "cross_task_by_task", None)
        if cross_by_task:
            cross_items = [(tid, bundle.to_dict()) for tid, bundle in cross_by_task.items()]
            cross_embeddings = collab.maybe_embed_batch([_bundle_embed_text(bd) for _, bd in cross_items])
            for (tid, cross_dict), embedding in zip(cross_items, cross_embeddings):
                try:
                    collab.knowledge.record_distillation(
                        task_id=tid,
                        generation=generation,
                        bundle=cross_dict,
                        scope="cross_task",
                        experiment=experiment,
                        embedding=embedding,
                    )
                    persisted_cross_task = True
                except Exception as exc:
                    log.warning(
                        "[ENGINE] record_distillation(cross_task, task=%s) failed: %s",
                        tid,
                        exc,
                    )
                    persistence_failures += 1
                    collab.record_phase_failure(generation, "distill_failures")
        elif out.cross_task is not None:
            cross_dict = out.cross_task.to_dict()
            try:
                collab.knowledge.record_distillation(
                    task_id=CROSS_TASK_SENTINEL,
                    generation=generation,
                    bundle=cross_dict,
                    scope="cross_task",
                    experiment=experiment,
                    embedding=collab.maybe_embed(_bundle_embed_text(cross_dict)),
                )
                persisted_cross_task = True
            except Exception as exc:
                log.warning(
                    "[ENGINE] record_distillation(cross_task) failed: %s",
                    exc,
                )
                persistence_failures += 1
                collab.record_phase_failure(generation, "distill_failures")
        # Report cross-task presence across both shapes: the single broadcast
        # bundle (out.cross_task) and the per-task target-conditioned bundles
        # (out.cross_task_by_task). Reporting only the former logged
        # cross_task=False under conditioning even when N bundles were produced.
        cross_task_count = len(cross_by_task) if cross_by_task else (1 if out.cross_task is not None else 0)
        log.info(
            "[ENGINE] distill gen=%d: %d per-task bundle(s), cross_task=%d",
            generation,
            len(out.per_task or {}),
            cross_task_count,
        )
        # A generation that attempted work but persisted nothing AND
        # recorded sub-failures is the signature of a host->provider outage
        # (not a healthy empty result). Escalate loudly / track the streak.
        distill_failures = int(getattr(out, "failures", 0) or 0)
        total_failures = distill_failures + persistence_failures
        fully_zeroed = bool(task_ids) and persisted_per_task == 0 and not persisted_cross_task and total_failures > 0
        collab.record_distill_result(generation, fully_zeroed=fully_zeroed, failures=total_failures)
        return DistillationPhaseResult(
            attempted_task_ids=tuple(task_ids),
            persisted_per_task=persisted_per_task,
            persisted_cross_task=persisted_cross_task,
        )

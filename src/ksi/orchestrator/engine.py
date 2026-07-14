from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from ..distillation._removed_env import assert_no_removed_channel_env
from ..distillation.types import DistillLLMResult
from ..errors import AuthenticationFailure, DistillationStalledError
from ..models import AgentState, GenerationConfig, Insight, TaskSpec, TaskTrace
from ..orchestrator.population import make_strategy
from ..protocols import Evaluator, ForumMessageContent, LLMCaller, PersistenceObserver, RuntimeExecutor
from ..seeding.seeder import PopulationSeeder
from ..tasks.registry import resolve_source
from ..tokens import LLMResponse, TokenAccumulator, TokenUsage, TokenUsageDict
from . import approach_diagnosis as _approach_diagnosis  # noqa: F401  # wires TaskSourceSpec.approach_diagnosis hooks
from .attempt_events import (  # post-attempt knowledge helpers; import wires the tb2 trace_condensed hook
    _build_approach_diagnosis,
    _build_attempt_event,
    _extract_approach_excerpt,
    _knowledge_attempt_external_id,
    _score_from_eval,  # noqa: F401  # re-exported for tests importing from engine
    _tb2_attempt_meta,  # noqa: F401  # re-exported for tests importing from engine
)
from .claim_phase import EngineClaimPhaseService
from .distillation_phase import EngineDistillationPhaseService
from .enrichment_phase import EngineEnrichmentPhaseService
from .execution_phase import EngineExecutionPhaseService, ExecutionPhaseInput
from .forum_phase import EngineForumPhaseService, ForumValidationError  # noqa: F401

# Forum machinery moved to ``forum_runtime`` and the shared retry/token helpers
# to ``task_retry``. Engine uses several of these internally (forum
# phase methods, task-execution retry); the rest are re-exported here purely so
# tests that import them ``from ksi.orchestrator.engine`` keep working. The
# blanket ``noqa: F401`` covers the re-export-only names.
from .forum_runtime import (  # noqa: F401
    _all_expected_signalled,
    _coerce_post_ref,
    _coerce_round_usage,
    _CrossTaskR1Coordinator,
    _drain_forum_bus,
    _forum_container_prefix,
    _ForumEarlyExitWatcher,
    _read_done_agent_ids,
    _read_done_signals,
    _run_retryable_forum_task,
    _stop_forum_containers,
)
from .kt_adapter_service import KtAdapterService
from .phase_services import EngineImprovementPhaseServices
from .resume_phase import (
    EngineResumePhaseService,
    carry_forward_payload,
    is_carried_forward_trace,
    trace_meets_preserve_threshold,
    trace_preserve_rank,
    trace_preserve_score,
)
from .seeding_phase import EngineSeedingPhaseService
from .strategy import DefaultKnowledgeStrategy, GenerationContext, ImprovementStrategy, plan_seed_next_generation
from .task_retry import (  # noqa: F401
    _NON_RETRYABLE_EXIT_CODES,
    _NON_RETRYABLE_TASK_ERROR_MARKERS,
    _TRANSIENT_TASK_ERROR_MARKERS,
    _UPSTREAM_PROVIDER_TRANSIENT_MARKERS,
    _accumulate_failed_attempt_tokens,
    _cap_native_memory_fields,
    _is_retryable_task_error,
    _runtime_retry_meta,
    run_with_distill_retry,
)

log = logging.getLogger(__name__)


def _assert_single_task_per_agent(assigned_map: dict[str, list[str]], *, generation: int) -> None:
    """Fail loud if the claim phase ever assigns more than one task to an agent
    in a single generation.

    Today this can't happen: ``EngineClaimPhaseService.claim`` tracks a
    ``claimed_agents`` set and skips any agent already claimed. But that is a
    behavioral guarantee of one implementation, not a type-level one --
    ``assigned_map`` is typed ``dict[str, list[str]]`` and
    ``forum_phase.py``'s cross-task ``phase1_by_agent`` context silently keeps
    only the LAST trace per agent (a plain dict overwrite, no ordering guard)
    if this invariant is ever violated. This assertion catches a future
    ``ClaimPhaseService`` implementation that breaks it, at the source, before
    that silent data loss can occur three phases later.
    """
    violations = {agent_id: task_ids for agent_id, task_ids in assigned_map.items() if len(task_ids) > 1}
    if violations:
        raise RuntimeError(
            f"[gen {generation}] claim phase assigned multiple tasks to a single agent in one "
            f"generation, violating the 1-task-per-agent invariant that forum_phase.py's "
            f"per-agent cross-task context (phase1_by_agent) relies on: {violations}"
        )


@dataclass
class NoopPersistence:
    def on_generation_start(self, *, generation: int, agents: list[AgentState]) -> None:
        return None

    def on_assignment(self, *, generation: int, assigned: dict[str, list[str]], total_tasks: int = 0) -> None:
        return None

    def on_task_status(self, *, generation: int, agent_id: str, task_id: str, status: str) -> None:
        return None

    def on_task_trace(self, trace: TaskTrace) -> None:
        return None

    def on_forum_message(
        self,
        *,
        generation: int,
        round_num: int,
        agent_id: str,
        message_type: str,
        content_json: ForumMessageContent,
        token_usage: TokenUsageDict,
    ) -> None:
        return None

    def on_insight(self, *, generation: int, agent_id: str, insight: "Insight") -> None:
        return None

    def on_native_memory(self, *, generation: int, agent_id: str, content: str) -> None:
        return None

    def on_generation_end(self, *, generation: int, agents: list[AgentState]) -> None:
        return None

    def on_run_end(self, *, token_summary: TokenUsage) -> None:
        return None


class _GenerationResult(NamedTuple):
    """What ``run()`` needs from one ``_run_generation`` call: accumulate + stop."""

    remaining_tasks: list[TaskSpec]
    traces: list[TaskTrace]
    should_stop: bool


class GenerationalOrchestrator:
    """Generational orchestrator with round-robin task assignment.

    Generation loop: claim -> enrich -> execute -> per-task forum ->
    cross-task forum -> distill -> seed. Tasks are distributed evenly
    across agents (round-robin), agents execute them, discuss results in
    per-task and cross-task forums to share knowledge, the knowledge is
    distilled into bundles, and the next generation is seeded from those
    distilled knowledge bundles.
    """

    def __init__(
        self,
        *,
        config: GenerationConfig,
        runtime: RuntimeExecutor,
        evaluator: Evaluator,
        llm: LLMCaller,
        persistence: PersistenceObserver | None = None,
        working_dir: str = ".",
    ) -> None:
        assert_no_removed_channel_env()
        self.config = config
        self.runtime = runtime
        self.evaluator = evaluator
        self.llm = llm
        self.persistence = persistence or NoopPersistence()
        self._working_dir = Path(working_dir).resolve()

        # Phase-1 self-reflection (Path a): wire the evaluator into the
        # container executor so its in-flight BarrierWatcher can call
        # evaluator.evaluate() between the agent's task completion and the
        # in-session follow-up reflection turn. The runtime checks
        # ``self.runtime.phase1_reflection_enabled`` independently — this
        # assignment only ensures the evaluator is reachable when the
        # feature flag is on; it's a no-op for OpenAI / non-container
        # runtimes.
        if hasattr(self.runtime, "evaluator"):
            try:
                self.runtime.evaluator = evaluator
            except Exception:
                # Don't fail engine construction over a non-settable attr.
                pass

        # Internal components
        self._population = make_strategy(config)
        self._seeder = PopulationSeeder()
        # Task execution/evaluation phase boundary. The generation loop depends
        # only on explicit ExecutionPhaseInput/ExecutionPhaseResult values.
        self._claim_phase = EngineClaimPhaseService(self)
        self._execution_phase = EngineExecutionPhaseService(self)
        self._enrichment_phase = EngineEnrichmentPhaseService(self)
        self._forum_phase_service = EngineForumPhaseService(self)
        self._distillation_phase = EngineDistillationPhaseService(self)
        # Count consecutive generations whose distillation was fully
        # zeroed by failures, so a sustained host->provider outage (which retry
        # cannot fix) escalates loudly instead of silently burning attempt
        # compute for zero learning. Reset by any generation that persists a bundle.
        self._consecutive_zeroed_distill_generations = 0
        self._seeding_phase = EngineSeedingPhaseService(self)
        self._resume_phase = EngineResumePhaseService(self)
        # Improvement-strategy seam: the
        # self-improvement mechanism (per-task forum -> cross-task forum ->
        # distill -> seed) is invoked through this strategy object and an
        # explicit phase-service adapter.  The default delegates to the phase
        # services, which preserve all existing phase gating.  Selectable via the
        # ``--improvement-strategy`` CLI flag or :meth:`set_improvement_strategy`.
        self._improvement_phases = EngineImprovementPhaseServices(
            self,
            forum_phase=self._forum_phase_service,
            distillation_phase=self._distillation_phase,
            seeding_phase=self._seeding_phase,
        )
        self._improvement_strategy: ImprovementStrategy = DefaultKnowledgeStrategy()
        self._memory_store: Any | None = None

        # Token accumulator — reset at the start of each run()
        self.accumulator = TokenAccumulator()

        # Tracks best score per task_id across all generations.
        # Seeded from runtime DB when --resume is passed.
        self._best_scores: dict[str, float] = {}
        self._best_preserved_traces: dict[str, TaskTrace] = {}
        # Hold-out transfer probe: these task ids are attempted every
        # generation with current knowledge injected but excluded from
        # learning, --drop-solved, early-stop, and headline metrics.
        self._holdout_ids: frozenset[str] = frozenset(
            str(task_id).strip()
            for task_id in (getattr(config, "holdout_task_ids", None) or [])
            if str(task_id).strip()
        )
        # Per-generation NON-cumulative hold-out scores: {gen: {task_id: best score that gen}}.
        self._holdout_gen_results: dict[int, dict[str, float]] = {}
        self._vector_required = bool(config.require_vector)
        self._vector_disabled = str(os.getenv("KSI_DISABLE_VECTOR", "")).strip().lower() in {"1", "true", "yes"}
        # Lexical FTS5 retrieval is the default. Semantic vector search
        # (sqlite-vec index + embedding model) is opt-in via --require-vector;
        # KSI_DISABLE_VECTOR remains a hard override that also forbids the
        # opt-in (raises in _build_stores). When vector is off the engine writes
        # no embeddings, so the whole system — including the in-container forum
        # ``query`` tool — degrades to FTS5.
        self._vector_enabled = self._vector_required and not self._vector_disabled
        # Host is authoritative: when vector is off, tell the in-container
        # ``query`` tool to stay on FTS too (via MEMORY_ENABLE_SEMANTIC_SEARCH,
        # passed through by providers.py). Else a resumed DB still carrying a
        # ``knowledge_vec`` table from a prior --require-vector run would
        # silently query stale embeddings that miss every new FTS-only row.
        os.environ["MEMORY_ENABLE_SEMANTIC_SEARCH"] = "1" if self._vector_enabled else "0"
        self._vector_embedding_count = 0
        self._vector_skipped_count = 0
        # Guards both counters above: _maybe_embed/_maybe_embed_batch run from
        # concurrent eval-worker threads (execution_phase.py's eval
        # ThreadPoolExecutor), and unguarded `+= 1` loses increments under
        # concurrency, which can false-trip the "vector required but none
        # written" guard below.
        self._vector_counts_lock = threading.Lock()
        self._resume_latest_generation = 0
        self._start_generation = max(1, int(getattr(config, "start_generation", 1) or 1))

        knowledge_db_path = config.knowledge_db_path
        runtime_db_path = config.runtime_db_path
        exp_name = config.experiment_name or "default"

        # Knowledge store is authoritative for run identity, resume state,
        # attempts, retrieval, forum, distillation, and seed state. Runtime DB
        # is an optional audit/log sidecar and is opened only after the
        # authoritative experiment name is resolved here.
        self._knowledge: Any | None = None
        exp_name = self._resolve_experiment_name(config, knowledge_db_path, exp_name)
        try:
            self._initialize_stores(config, knowledge_db_path, runtime_db_path, exp_name)
        except Exception:
            close_knowledge = getattr(self._knowledge, "close", None)
            if callable(close_knowledge):
                close_knowledge()
            self._knowledge = None
            if knowledge_db_path and not config.resume:
                self._release_empty_experiment_claim(knowledge_db_path, exp_name)
            raise

        # Legacy fallback only: when no authoritative KnowledgeStore is
        # configured, resume from the runtime DB. If KnowledgeStore exists,
        # stale audit rows must not override its cursor or best scores.
        if config.resume and self._knowledge is None and self._memory_store is not None:
            try:
                prior_scores = self._memory_store.get_best_scores(
                    experiment=config.experiment_name or None,
                )
                prior_scores = {
                    task_id: score for task_id, score in (prior_scores or {}).items() if not self._is_holdout(task_id)
                }
                if prior_scores:
                    self._best_scores.update(prior_scores)
                    log.info("[ENGINE] Seeded %d best scores from legacy runtime DB", len(prior_scores))
            except Exception as exc:
                log.warning("[ENGINE] Failed to seed best scores from legacy runtime DB: %s", exc)
            try:
                latest_generation = self._memory_store.get_latest_task_generation(
                    experiment=config.experiment_name or None,
                )
                if latest_generation > 0:
                    self._resume_latest_generation = latest_generation
                    self._start_generation = latest_generation + 1
                    config.start_generation = self._start_generation
                    log.info(
                        "[ENGINE] Resume cursor: latest task generation=%d, next generation=%d, target generation=%d",
                        latest_generation,
                        self._start_generation,
                        self.config.num_generations,
                    )
            except Exception as exc:
                log.warning("[ENGINE] Failed to determine resume generation cursor: %s", exc)

        config.start_generation = self._start_generation

        # Propagate any experiment name rename to persistence observer.
        # Composite observers implement `set_experiment_name` so all wrapped
        # sinks follow the final collision-resolved run identity.
        if hasattr(self.persistence, "set_experiment_name"):
            self.persistence.set_experiment_name(config.experiment_name)
        elif hasattr(self.persistence, "experiment_name"):
            self.persistence.experiment_name = config.experiment_name

        # Initial agents are blank (no workstream in gen 1)
        self.agents = [AgentState(id=f"agent-{i}") for i in range(config.num_agents)]
        # Debug artifacts for pretask/claim phase inspection.
        self._claim_debug_history: list[dict[str, Any]] = []
        self._persisted_insight_ids: set[str] = set()
        # Per-generation knowledge-phase degradation counters:
        # {generation: {drain_failures, forum_agent_failures, distill_failures}}.
        # Surfaced into the results JSON so a generation whose knowledge phases
        # partially failed is distinguishable from a healthy one post-hoc.
        # Guarded by a lock: most increments run on the main loop, but the
        # cross-task R1 coordinator's drain (``_drain_r0``) records from its
        # own daemon thread concurrently with the main as-completed loop.
        self._knowledge_phase_health: dict[int, dict[str, int]] = {}
        self._knowledge_phase_dropped_event_ids: dict[int, set[str]] = {}
        self._knowledge_phase_health_lock = threading.Lock()
        self._external_per_task_bundles: dict[str, dict[str, Any]] = {}
        # KT adapter-transfer memo logic (build/repair/memoize) lives in an
        # injected service so the engine stays a sequencer. It owns
        # its own (generation, task_id) memo cache + lock.
        # llm_call is injected as a late-binding thunk (not the bound method
        # captured at construction) so a later reassignment of ``self._llm_call``
        # — as tests do to stub the LLM — is honored, matching the pre-extraction
        # behavior where the memo builder called ``self._llm_call`` dynamically.
        self._kt_adapter_service = KtAdapterService(
            llm_call=lambda **kw: self._llm_call(**kw),
            # Late-bound: run() reassigns self.accumulator each run, so pass a
            # getter (like llm_call) rather than the ctor-time instance, which
            # would orphan kt_adapter/kt_adapter_repair token recordings.
            accumulator_getter=lambda: self.accumulator,
        )

        # Inject external knowledge bundle into gen-1 agents if provided.
        if config.seed_bundle_path:
            self._inject_seed_bundle(config.seed_bundle_path)
        if config.seed_per_task_bundles_path:
            self._load_external_per_task_bundles(config.seed_per_task_bundles_path)
        self._pending_next_task_labels: list[str] = []

    def _resolve_experiment_name(
        self,
        config: GenerationConfig,
        knowledge_db_path: str | None,
        exp_name: str,
    ) -> str:
        """Resolve the authoritative experiment name via a closed probe store.

        Opens a throwaway ``KnowledgeStore`` solely to detect resume vs.
        collision, then closes it (the ``finally`` block) BEFORE the real store
        is opened — preserving the invariant that the probe never holds the
        DB open across the real-store construction. On a non-resume name
        collision the resolved ``_2``-style name is written back to
        ``config.experiment_name`` and returned. When ``knowledge_db_path`` is
        falsy, ``exp_name`` is returned unchanged (no probe).
        """
        if not knowledge_db_path:
            return exp_name
        # The probe construction shares the original block's failure contract:
        # any error opening/closing the probe surfaces as the wrapped
        # ``KnowledgeStore initialization failed`` RuntimeError, exactly as when
        # this lived inline inside __init__'s try/except.
        try:
            from ..memory.knowledge_store import KnowledgeStore

            probe = KnowledgeStore(knowledge_db_path, default_experiment=exp_name)
            try:
                has_experiment = getattr(probe, "has_experiment", lambda _experiment: False)
                claim_experiment = getattr(probe, "claim_experiment", None)
                if config.resume:
                    if not has_experiment(exp_name):
                        log.warning(
                            "[ENGINE] --resume passed but experiment %r not found in knowledge DB; starting fresh.",
                            exp_name,
                        )
                    else:
                        log.info("[ENGINE] Resuming experiment %r from knowledge DB", exp_name)
                    if callable(claim_experiment):
                        claim_experiment(exp_name, resume=True)
                else:
                    # Atomically claim the name so two concurrent same-name
                    # launches deterministically get distinct names (closes the
                    # read-then-rename race). The later _ensure_run INSERT OR
                    # IGNORE resolves the pre-claimed row.
                    if callable(claim_experiment):
                        new_name = claim_experiment(exp_name, resume=False)
                    else:
                        next_experiment_name = getattr(
                            probe,
                            "next_experiment_name",
                            lambda experiment: f"{experiment}_2",
                        )
                        new_name = next_experiment_name(exp_name) if has_experiment(exp_name) else exp_name
                    if new_name != exp_name:
                        log.warning(
                            "[ENGINE] Experiment %r already exists in knowledge DB — starting "
                            "fresh as %r. Use --resume to continue the prior run.",
                            exp_name,
                            new_name,
                        )
                        exp_name = new_name
                        config.experiment_name = new_name
            finally:
                close_probe = getattr(probe, "close", None)
                if callable(close_probe):
                    close_probe()
        except Exception as exc:
            raise RuntimeError(f"KnowledgeStore initialization failed: {exc}") from exc
        return exp_name

    def _release_empty_experiment_claim(self, knowledge_db_path: str, exp_name: str) -> None:
        """Best-effort cleanup for a failed non-resume initialization claim."""
        try:
            from ..memory.knowledge_store import KnowledgeStore

            store = KnowledgeStore(knowledge_db_path, default_experiment=exp_name)
            try:
                release = getattr(store, "release_empty_experiment_claim", None)
                if callable(release) and release(exp_name):
                    log.warning(
                        "[ENGINE] Released empty experiment-name claim %r after initialization failure",
                        exp_name,
                    )
            finally:
                store.close()
        except Exception as exc:
            log.warning(
                "[ENGINE] Could not release empty experiment-name claim %r after initialization failure: %s",
                exp_name,
                exc,
            )

    def _initialize_stores(
        self,
        config: GenerationConfig,
        knowledge_db_path: str | None,
        runtime_db_path: str | None,
        exp_name: str,
    ) -> None:
        """Open the authoritative knowledge store, runtime sidecar, and embedder.

        Ordering is load-bearing: the (already-resolved) experiment name has
        been fixed by :meth:`_resolve_experiment_name` with its probe closed, so
        the real ``KnowledgeStore`` opens cleanly here, followed by the optional
        runtime ``MemoryStore`` and the optional vector ``Embedder``.
        """
        if knowledge_db_path:
            try:
                from ..memory.knowledge_store import KnowledgeStore

                self._knowledge = KnowledgeStore(
                    knowledge_db_path,
                    default_experiment=exp_name,
                    enable_vec=self._vector_enabled,
                )
                config.knowledge_db_path = knowledge_db_path
                if config.resume:
                    # Hold-out tasks never enter best-score bookkeeping — drop
                    # their ids from resumed scores so they are re-attempted.
                    resumed_scores = self._knowledge.get_best_scores(experiment=exp_name)
                    self._best_scores.update(
                        {task_id: score for task_id, score in resumed_scores.items() if not self._is_holdout(task_id)}
                    )
                    latest_generation = self._knowledge.get_latest_task_generation(experiment=exp_name)
                    if latest_generation > 0:
                        self._resume_latest_generation = latest_generation
                        self._start_generation = max(self._start_generation, latest_generation + 1)
                _vec_ready = getattr(self._knowledge, "_vec_enabled", False)
                if _vec_ready:
                    _vec_status, _vec_detail = "enabled", "knowledge_vec ready"
                elif not self._vector_required:
                    # Off by design: FTS5 is the default, not a degradation.
                    _vec_status, _vec_detail = (
                        "off",
                        "FTS5 lexical retrieval is the default (opt in with --require-vector)",
                    )
                else:
                    _vec_status, _vec_detail = "degraded", "sqlite-vec unavailable"
                self._knowledge.record_vector_status(
                    phase="init",
                    status=_vec_status,
                    detail=_vec_detail,
                    experiment=exp_name,
                )
                if self._vector_required and not getattr(self._knowledge, "_vec_enabled", False):
                    raise RuntimeError("Vector memory is required but sqlite-vec unavailable")
            except Exception as exc:
                raise RuntimeError(f"KnowledgeStore initialization failed: {exc}") from exc

        # Runtime store for non-authoritative transcripts, artifacts, token
        # accounting, and audit metadata.
        if runtime_db_path:
            from ..memory.store import MemoryStore

            self._memory_store = MemoryStore(
                runtime_db_path,
                default_experiment=exp_name,
            )
            config.runtime_db_path = runtime_db_path
            # Share the single runtime store with the persistence observer so
            # there is exactly ONE MemoryStore on the runtime DB (root cause of
            # the AB-BA deadlock). hasattr-guarded like
            # set_experiment_name above — deliberately NOT a PersistenceObserver
            # Protocol member, so observers without a store are simply skipped.
            if hasattr(self.persistence, "share_runtime_store"):
                self.persistence.share_runtime_store(self._memory_store)

        # Optional embedder for vector search (opt-in via --require-vector; the
        # default retrieval path is lexical FTS5). Can also be hard-disabled
        # with KSI_DISABLE_VECTOR=1.
        self._embedder = None
        if self._vector_required and self._vector_disabled:
            raise RuntimeError("Vector memory is required but KSI_DISABLE_VECTOR is set")
        if self._knowledge is not None and self._vector_enabled and not self.config.no_memory:
            try:
                from ..layout import RUNTIME_STATE_DIR
                from ..memory.embeddings import Embedder, point_embeddings_cache_at

                # Point the host embedder at the shared model cache that
                # solver containers mount, so the host populates it via direct
                # filesystem access and the container mount can be read-only.
                point_embeddings_cache_at(RUNTIME_STATE_DIR / "model_cache")

                embedder = Embedder(background=not self._vector_required)
                if self._vector_required and not getattr(embedder, "is_ready", False):
                    raise RuntimeError("embedding model did not become ready")
                self._embedder = embedder  # Model loads in background thread
                if self._knowledge is not None:
                    self._knowledge.record_vector_status(
                        phase="embedder",
                        status="enabled",
                        detail="embedding model ready"
                        if self._vector_required
                        else "embedding model loading in background",
                        experiment=self.config.experiment_name,
                    )
                log.info(
                    "[ENGINE] Embedder ready"
                    if self._vector_required
                    else "[ENGINE] Embedder loading in background, vector search will activate when ready"
                )
            except Exception as exc:
                if self._vector_required:
                    raise RuntimeError(f"Vector memory is required but embedder failed: {exc}") from exc
                self._embedder = None
                if self._knowledge is not None:
                    self._knowledge.record_vector_status(
                        phase="embedder",
                        status="degraded",
                        detail=str(exc),
                        experiment=self.config.experiment_name,
                    )
                log.warning("[ENGINE] Embedder/vec unavailable, vector search disabled: %s", exc)
        elif self._vector_disabled:
            log.info("[ENGINE] Vector memory disabled via KSI_DISABLE_VECTOR")
        elif not self._vector_required:
            log.info(
                "[ENGINE] Vector memory off (FTS5 lexical retrieval is the default; pass --require-vector to enable)"
            )

    def set_improvement_strategy(self, strategy: ImprovementStrategy) -> None:
        """Swap the self-improvement mechanism (refactor move 3 seam).

        Selectable via the ``--improvement-strategy`` CLI flag or
        programmatically here (see ``docs/improvement_strategies.md``).  The default
        :class:`DefaultKnowledgeStrategy` reproduces the inline behaviour; pass
        a different :class:`ImprovementStrategy` to alter the
        forum/distill/seed phases without touching the engine loop.
        """
        self._improvement_strategy = strategy

    def _inject_seed_bundle(self, bundle_path: str) -> None:
        """Load an external knowledge bundle and inject into gen-1 agents.

        Supports two bundle formats:

        - **Cross-task distillation bundle** (current):
          ``{"cross_task": {"transferable_insights": [...], "confirmed_constraints":
          [...], "rejected_hypotheses": [...], "pitfalls": [...], "checks": [...],
          "next_steps": [...], "evidence_post_ids": [...]}, "meta": {...}}``.
          An exported cross-task knowledge bundle from a donor ``knowledge.sqlite``.
          List items are structured Insight dicts
          (``{text, applies_when, does_not_apply_when, confidence, evidence}``)
          mirroring what ``KnowledgeStore.load_distillation`` returns to the
          main-campaign gen-2 path. Rendered via
          ``seed_package["cross_task_bundle"]`` so it flows through the same
          prompt path as internal cross-gen seeds.

        - **Legacy asset bundle** (pre-three-phase): ``{"assets": [{text, ...}]}``.
          Kept for back-compat with older bundles extracted before the schema
          migration.
        """
        if self.config.no_memory:
            return
        path = Path(bundle_path)
        # Fail-loud on bad --seed-bundle-path: the user explicitly asked to
        # inject this bundle, so silently skipping (the prior behavior)
        # produces a baseline run with no KT and no failure signal — exactly
        # the ablation-noise mode we want to avoid.
        if not path.exists():
            raise FileNotFoundError(
                f"--seed-bundle-path {bundle_path!r} does not exist. "
                "If you intended to run without KT, omit the flag entirely."
            )
        try:
            bundle = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"--seed-bundle-path {bundle_path!r} is not valid JSON: {exc}. "
                "Ensure the bundle is a valid exported cross-task knowledge bundle."
            ) from exc

        if not isinstance(bundle, dict):
            raise ValueError(
                f"--seed-bundle-path {bundle_path!r} must contain a JSON object "
                f"at the top level; found {type(bundle).__name__}."
            )

        meta = bundle.get("meta", {}) or {}

        from ..distillation.types import CROSS_TASK_INSIGHT_FIELDS

        cross_task = bundle.get("cross_task")
        if isinstance(cross_task, dict) and any(
            isinstance(cross_task.get(k), list) and cross_task.get(k) for k in CROSS_TASK_INSIGHT_FIELDS
        ):
            # Mark the bundle as externally injected so the seed renderer
            # applies the generous external caps (mirrors the per-task
            # loader's marker; the key passes _inject_cross_task_seed's
            # normalization verbatim and is never rendered as content).
            cross_task["_external_seed_source"] = str(path)
            self._inject_cross_task_seed(cross_task, meta=meta)
            return

        raw_assets = bundle.get("assets", [])
        if not raw_assets:
            raise ValueError(
                f"--seed-bundle-path {bundle_path!r} contains no usable content: "
                "neither a populated `cross_task` object (with any of "
                f"{sorted(CROSS_TASK_INSIGHT_FIELDS)}) nor a non-empty legacy "
                "`assets` list. Ensure the bundle is a valid exported cross-task knowledge bundle."
            )

        # Legacy path: map asset_id → id so _normalize_bundle_item picks it up.
        assets = []
        for a in raw_assets:
            item = dict(a)
            if "asset_id" in item and "id" not in item:
                item["id"] = item.pop("asset_id")
            assets.append(item)

        from ..seeding.seeder import _build_shared_seed_package

        seed_package = _build_shared_seed_package(
            assets,
            generation=1,
            workstream_description=str(
                bundle.get("bundle_summary")
                or bundle.get("bundle_title")
                or "Shared condensed bundle for task-mode execution"
            ).strip()
            or "Shared condensed bundle for task-mode execution",
        )

        for agent in self.agents:
            # deepcopy so nested mutations (e.g. downstream filters on
            # cross_task_bundle) don't propagate across agents that share
            # the same source seed_package.
            agent.seed_package = copy.deepcopy(seed_package)
            agent.workstream = seed_package.get("workstream_name", "")
            agent.workstream_description = seed_package.get("workstream_description", "")

        log.info(
            "[ENGINE] Injected legacy asset bundle: %d assets from %s (gen %s, mode=%s)",
            len(assets),
            meta.get("source_experiment", "unknown"),
            meta.get("generation_extracted", "?"),
            meta.get("extraction_mode", "?"),
        )

    def _inject_cross_task_seed(self, cross_task: dict[str, Any], *, meta: dict[str, Any]) -> None:
        """Inject a cross-task distillation bundle as the gen-1 shared seed.

        Mirrors the ``cross_task_bundle`` path used internally between
        generations: ``KnowledgeStore.load_distillation`` returns the raw
        stored dict (7 insight-bearing fields plus ``evidence_post_ids``)
        and the seeder passes it straight through to
        ``seed_package["cross_task_bundle"]`` for ``runtime/seeding.py``
        to render. We do the same here, dropping only the ``scope``
        bookkeeping key. ``_render_bundle_item`` handles structured
        Insight dicts and falls back to strings, so no normalization is
        needed.

        Note: ``load_distillation`` also stamps a ``_knowledge_id``
        provenance key on its return value (see ``knowledge_store.py``);
        the KT path here has no equivalent provenance handle, so the
        rendered bundle is otherwise identical but lacks that internal
        id. This affects nothing downstream — ``_render_bundle_item``
        and the seed-package consumers don't read the field — but it's
        worth knowing if you compare a KT recipient's seed dict with a
        gen-2 recipient's and notice the missing key.
        """

        from ..distillation.types import CROSS_TASK_INSIGHT_FIELDS

        # Defensively coerce the known insight-bearing list fields and
        # ``evidence_post_ids`` to ``list`` (a hand-built bundle whose
        # ``transferable_insights`` was a single string would otherwise
        # iterate chars in ``_render_bundle_item``). Other keys pass
        # through verbatim — the extractor is the canonical type
        # boundary, this is just a defense-in-depth guard for hand-built
        # or mis-migrated bundles. See ``ksi.distillation.types`` for
        # the canonical schema.
        _LIST_FIELDS = CROSS_TASK_INSIGHT_FIELDS + ("evidence_post_ids",)
        normalized: dict[str, Any] = {}
        for k, v in cross_task.items():
            if k == "scope":
                continue
            if k in _LIST_FIELDS:
                normalized[k] = v if isinstance(v, list) else []
            else:
                normalized[k] = v

        # `source_experiment` is preserved in `meta` for server-side provenance
        # (logs, audit trails, reproducibility) but MUST NOT be interpolated
        # into `workstream_description` — that string is rendered into the
        # recipient agent's MEMORY.md via `seeding.py::seed_package_to_memory_md`
        # and would leak donor experiment identity (model/config encoded in the
        # name) into the recipient's prompt. The donor anonymization in
        # an exported cross-task bundle hashes `donor_task_ids` but intentionally
        # leaves `source_experiment` untouched for the audit-trail use; closing
        # the prompt-side leak here.
        source_experiment = meta.get("source_experiment") or "donor experiment"
        workstream_description = (
            str(meta.get("bundle_summary") or meta.get("bundle_title") or "").strip()
            or "Cross-task knowledge transferred from an anonymized donor experiment"
        )

        seed_package = {
            "generation": 1,
            "workstream_name": "kt-cross-task-bundle",
            "workstream_description": workstream_description,
            "insight_bundle": [],
            "shared_insight_bundle": [],
            "evidence_refs": [],
            "cross_task_bundle": normalized,
            "_kt_mode": "adapter_transfer",
            "_kt_source_experiment": source_experiment,
            "_kt_source_generation": meta.get("generation_extracted"),
        }

        for agent in self.agents:
            # deepcopy so the per-agent KT seed_package is isolated from
            # nested mutations elsewhere (cross_task_bundle is a nested dict
            # — shallow dict() would share its sub-objects across agents).
            agent.seed_package = copy.deepcopy(seed_package)
            agent.workstream = seed_package["workstream_name"]
            agent.workstream_description = seed_package["workstream_description"]

        log.info(
            "[ENGINE] Injected KT cross-task bundle from %s (gen=%s): "
            "insights=%d, constraints=%d, rejected=%d, pitfalls=%d, "
            "checks=%d, next_steps=%d, evidence=%d",
            source_experiment,
            meta.get("generation_extracted", "?"),
            len(normalized.get("transferable_insights") or []),
            len(normalized.get("confirmed_constraints") or []),
            len(normalized.get("rejected_hypotheses") or []),
            len(normalized.get("pitfalls") or []),
            len(normalized.get("checks") or []),
            len(normalized.get("next_steps") or []),
            len(normalized.get("evidence_post_ids") or []),
        )

    def _load_external_per_task_bundles(self, bundle_path: str) -> None:
        """Load external per-task distilled bundles for gen-1 task execution.

        Expected format is a JSON object with ``bundles`` where each item
        contains ``task_id`` and either ``distilled_knowledge`` or a direct
        per-task bundle payload. These are attached later in
        ``EngineEnrichmentPhaseService.enrich`` once tasks have been assigned to agents.

        Bad paths raise loudly — matches `_inject_seed_bundle`'s fail-loud
        contract. The user explicitly asked for per-task
        KT via `--seed-per-task-bundles-path`; silently continuing with
        no KT produces a baseline run with no failure signal.
        """
        if self.config.no_memory:
            return
        path = Path(bundle_path)
        if not path.exists():
            raise FileNotFoundError(
                f"--seed-per-task-bundles-path {bundle_path!r} does not exist. "
                "If you intended a baseline-with-no-KT run, omit the flag; "
                "otherwise regenerate the per-task bundle JSON."
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"--seed-per-task-bundles-path {bundle_path!r} is not valid JSON: {exc}") from exc

        rows = payload.get("bundles") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError(
                f"--seed-per-task-bundles-path {bundle_path!r} contains no usable "
                "`bundles` list. Expected a JSON object with a non-empty "
                "`bundles` array of {task_id, distilled_knowledge|per_task_bundle} rows."
            )

        loaded: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            task_id = str(row.get("task_id") or "").strip()
            if not task_id:
                continue
            bundle = row.get("distilled_knowledge")
            if not isinstance(bundle, dict):
                bundle = row.get("per_task_bundle")
            if not isinstance(bundle, dict) or not bundle:
                continue
            normalized = dict(bundle)
            knowledge_id = row.get("knowledge_id")
            if isinstance(knowledge_id, int):
                normalized["_knowledge_id"] = knowledge_id
            normalized["_external_seed_source"] = str(path)
            loaded[task_id] = normalized

        if not loaded:
            raise ValueError(
                f"--seed-per-task-bundles-path {bundle_path!r} yielded zero usable "
                "per-task bundles: no row carried a non-empty `distilled_knowledge` "
                "or `per_task_bundle` dict with a task_id. This silently degrades to "
                "a baseline-with-no-KT run — likely donor schema drift. Regenerate "
                "the per-task bundle JSON, or omit the flag for an intentional "
                "baseline run."
            )

        self._external_per_task_bundles = loaded
        log.info(
            "[ENGINE] Loaded %d external per-task seed bundle(s) from %s",
            len(loaded),
            path,
        )

    def _llm_call(self, *, system: str, user: str, context: dict[str, Any] | None = None) -> LLMResponse:
        try:
            return self.llm.call(system=system, user=user, context=context)
        except TypeError as exc:
            if "unexpected keyword argument 'context'" not in str(exc):
                raise
            return self.llm.call(system=system, user=user)

    def _maybe_embed(self, text: str) -> list[float] | None:
        """Return an embedding for ``text`` if the embedder is loaded, else ``None``.

        Guarded against empty/whitespace input and embedder load failures so
        callers never have to repeat the boilerplate.
        """
        if not text or not text.strip():
            return None
        embedder = self._embedder
        if embedder is None or not getattr(embedder, "is_ready", False):
            with self._vector_counts_lock:
                self._vector_skipped_count += 1
            return None
        try:
            embedding = embedder.embed(text)
            with self._vector_counts_lock:
                self._vector_embedding_count += 1
            return embedding
        except Exception as exc:
            with self._vector_counts_lock:
                self._vector_skipped_count += 1
            log.debug("[ENGINE] embed failed (%d chars): %s", len(text), exc)
            return None

    def _maybe_embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Batch variant of :py:meth:`_maybe_embed` for the forum drain path.

        Returns one embedding (or ``None``) per input text. Empty/whitespace
        texts and embedder-not-ready cases yield ``None`` without invoking
        the model. The model call itself uses ``embedder.embed_batch`` so
        all texts pay one fixed-cost ``model.encode`` invocation rather
        than N synchronous encode calls — this is the hot drain perf win.

        On batch failure, all texts get ``None`` (drain proceeds without
        embeddings — the row still lands in the knowledge table). The
        caller can fall back to per-event ``_maybe_embed`` if needed,
        though this should rarely fire. NOTE: chunking the batch to bound
        memory at very large drain sizes (>10k events) is a known
        future-work item; revisit if a real campaign hits that scale.
        """
        result: list[list[float] | None] = [None] * len(texts)
        if not texts:
            return result
        embedder = self._embedder
        if embedder is None or not getattr(embedder, "is_ready", False):
            with self._vector_counts_lock:
                self._vector_skipped_count += len(texts)
            return result
        # Filter to non-empty texts so the encode call doesn't waste cycles
        # on whitespace; preserve the index so we can splice back into the
        # parallel result list.
        non_empty: list[tuple[int, str]] = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        empty_count = len(texts) - len(non_empty)
        if empty_count:
            with self._vector_counts_lock:
                self._vector_skipped_count += empty_count
        if not non_empty:
            return result
        try:
            vectors = embedder.embed_batch([t for _, t in non_empty])
            if not isinstance(vectors, list) or len(vectors) != len(non_empty):
                raise RuntimeError(
                    f"embed_batch returned {len(vectors) if isinstance(vectors, list) else type(vectors).__name__} "
                    f"for {len(non_empty)} texts"
                )
            for (i, _), vec in zip(non_empty, vectors):
                result[i] = vec
            with self._vector_counts_lock:
                self._vector_embedding_count += len(non_empty)
        except Exception as exc:
            with self._vector_counts_lock:
                self._vector_skipped_count += len(non_empty)
            log.debug("[ENGINE] embed_batch failed (%d texts): %s", len(non_empty), exc)
        return result

    @staticmethod
    def _retrieved_distillation_ids(agent: AgentState | None) -> dict[str, int] | None:
        """Return distillation knowledge.id refs that the agent received via seeding.

        ``load_distillation`` stamps ``_knowledge_id`` on the bundle dicts it
        returns; the seeder copies those dicts verbatim into
        ``agent.seed_package["per_task_bundle"]`` and
        ``["cross_task_bundle"]``.  Reading them back here lets us write
        ``retrieved_distillation_ids`` onto the attempt's ``attempt_meta``,
        which is the missing edge in the knowledge → solve provenance graph
        (attempt rows otherwise don't link back to the distillation entries
        that fed the agent).  Returns ``None`` for cold-start gen-1 attempts
        or carried-forward placeholders, where no agent knowledge was
        retrieved.
        """
        if agent is None:
            return None
        seed = getattr(agent, "seed_package", None)
        if not isinstance(seed, dict):
            return None
        ids: dict[str, int] = {}
        cross = seed.get("cross_task_bundle")
        if isinstance(cross, dict):
            cid = cross.get("_knowledge_id")
            if isinstance(cid, int):
                ids["cross_task"] = cid
        per_task = seed.get("per_task_bundle")
        if isinstance(per_task, dict):
            pid = per_task.get("_knowledge_id")
            if isinstance(pid, int):
                ids["per_task"] = pid
        return ids or None

    @staticmethod
    def _merge_attempt_meta(
        base: dict[str, Any] | None,
        retrieved_ids: dict[str, int] | None,
    ) -> dict[str, Any] | None:
        """Merge ``retrieved_distillation_ids`` into the attempt_meta payload.

        ``base`` is the carry-forward payload (or ``None``) — keep it intact
        and add a sibling key so analysis code can read both independently.
        """
        if not retrieved_ids:
            return base
        merged: dict[str, Any] = dict(base or {})
        merged["retrieved_distillation_ids"] = retrieved_ids
        return merged

    @staticmethod
    def _merge_optional_meta(
        left: dict[str, Any] | None,
        right: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        merged: dict[str, Any] = {}
        if isinstance(left, dict):
            merged.update(left)
        if isinstance(right, dict):
            merged.update(right)
        return merged or None

    # ── Hold-out transfer probe helpers ────────────────────────────────────
    def _is_holdout(self, task_id: str) -> bool:
        """True when ``task_id`` is a designated hold-out probe task."""
        return str(task_id) in self._holdout_ids

    def _non_holdout(self, traces: list[TaskTrace]) -> list[TaskTrace]:
        """Filter out hold-out traces (learning phases must never see them)."""
        if not self._holdout_ids:
            return traces
        return [t for t in traces if not self._is_holdout(t.task_id)]

    def _tag_holdout_meta(self, meta: dict[str, Any] | None, task_id: str) -> dict[str, Any] | None:
        """Merge ``{"holdout": True}`` into attempt_meta for hold-out attempts."""
        if not self._is_holdout(task_id):
            return meta
        merged: dict[str, Any] = dict(meta or {})
        merged["holdout"] = True
        return merged

    def _record_task_tokens(self, trace: TaskTrace) -> None:
        """Record task-execution token usage; hold-out attempts land under the
        ``task_execution_holdout`` phase so headline token analysis can split
        them out."""
        if self._is_holdout(trace.task_id):
            self.accumulator.record_lifecycle(
                trace.generation,
                trace.agent_id,
                "task_execution_holdout",
                trace.token_usage,
            )
        else:
            self.accumulator.record_task(trace.generation, trace.agent_id, trace.task_id, trace.token_usage)

    def holdout_solve_rate_by_generation(self) -> dict[int, dict[str, Any]]:
        """Per-generation NON-cumulative hold-out solve rate.

        ``{gen: {"solved": n, "total": N, "rate": n/N}}`` where ``solved``
        counts hold-out tasks whose best attempt score THAT generation is
        >= ``config.solved_threshold`` and ``total`` is the number of
        designated hold-out tasks. Empty dict when the feature is unused.
        """
        if not self._holdout_ids:
            return {}
        total = len(self._holdout_ids)
        threshold = float(self.config.solved_threshold)
        out: dict[int, dict[str, Any]] = {}
        for gen in sorted(self._holdout_gen_results):
            scores = self._holdout_gen_results[gen]
            solved = sum(1 for score in scores.values() if score >= threshold)
            out[gen] = {"solved": solved, "total": total, "rate": solved / total}
        return out

    # ``knowledge_phase_health`` counter kinds. Kept as a tuple so
    # ``knowledge_phase_health_by_generation``
    # always emits a complete, stable schema (zeros for clean generations) rather
    # than sparse, kind-dependent keys.
    _KNOWLEDGE_PHASE_HEALTH_KINDS = (
        "drain_failures",
        "forum_agent_failures",
        "distill_failures",
        "seed_failures",
    )

    def _record_knowledge_phase_failure(self, generation: int, kind: str, n: int = 1) -> None:
        """Increment a per-generation knowledge-phase degradation counter.

        Called at the silent-degradation sites (forum-bus drain failures,
        terminal forum-agent failures, distill sub-failures). Most call sites
        run on the main loop after worker futures are collected, but the
        cross-task R1 coordinator's ``_drain_r0`` records from its own daemon
        thread while the main as-completed loop may also be recording — so the
        read-modify-write is guarded by ``_knowledge_phase_health_lock``.
        """
        if n <= 0:
            return
        with self._knowledge_phase_health_lock:
            bucket = self._knowledge_phase_health.setdefault(generation, {})
            bucket[kind] = bucket.get(kind, 0) + n

    def _record_distill_generation_result(self, generation: int, *, fully_zeroed: bool, failures: int) -> None:
        """Escalate loudly when a generation's distillation is fully zeroed.

        ``fully_zeroed`` means: tasks were attempted this generation but no
        per-task or cross-task bundle was persisted AND there were sub-failures
        — the signature of a host->provider outage, not a healthy "nothing new
        to distill" result. Retry (``run_with_distill_retry``) rides out short
        blips; this backstops a *sustained* outage retry cannot fix by logging
        at ERROR (not the per-task WARNING that buries the signal)
        and counting consecutive zeroed generations so an operator notices the
        run is spending attempt compute for zero learning. Any generation that
        persists a bundle resets the counter.
        """
        if not fully_zeroed:
            self._consecutive_zeroed_distill_generations = 0
            return
        self._consecutive_zeroed_distill_generations += 1
        streak = self._consecutive_zeroed_distill_generations
        log.error(
            "[ENGINE] distillation produced ZERO knowledge at gen=%d (%d sub-failure(s)); "
            "%d consecutive generation(s) fully zeroed. Attempt compute is being spent for no "
            "learning — check host->provider connectivity (DNS/network to the LLM API).",
            generation,
            failures,
            streak,
        )
        # Opt-in hard abort: stop the run once the streak reaches the
        # operator-set threshold rather than burning the rest of the run.
        threshold = int(getattr(self.config, "abort_on_distill_stall", 0) or 0)
        if threshold > 0 and streak >= threshold:
            raise DistillationStalledError(
                f"distillation fully zeroed for {streak} consecutive generation(s) "
                f"(--abort-on-distill-stall={threshold}); aborting run at gen={generation}. "
                "Check host->provider connectivity (DNS/network to the LLM API)."
            )

    def _should_count_knowledge_drain_drop(self, generation: int, event_id: str) -> bool:
        """Return True once per failed forum event id for health accounting."""
        event_id = str(event_id or "").strip()
        if not event_id:
            return True
        with self._knowledge_phase_health_lock:
            seen = self._knowledge_phase_dropped_event_ids.setdefault(generation, set())
            if event_id in seen:
                return False
            seen.add(event_id)
            return True

    def knowledge_phase_health_by_generation(self) -> dict[int, dict[str, int]]:
        """Per-generation knowledge-phase degradation counts for the results JSON.

        ``{gen: {"drain_failures": d, "forum_agent_failures": f,
        "distill_failures": x, "seed_failures": s}}`` for every generation that
        recorded at least one failure. Each entry carries the full kind schema
        (zeros included) so downstream consumers can rely on stable keys. Empty
        dict for a fully healthy run.

        An empty dict is ambiguous on its own — a clean run, an old pre-feature
        run, and a ``--no-memory`` run all report ``{}``. Pair it with
        :meth:`knowledge_phase_health_measured` to tell "measured and clean" from
        "not measured".
        """
        out: dict[int, dict[str, int]] = {}
        with self._knowledge_phase_health_lock:
            for gen in sorted(self._knowledge_phase_health):
                bucket = self._knowledge_phase_health[gen]
                out[gen] = {kind: int(bucket.get(kind, 0)) for kind in self._KNOWLEDGE_PHASE_HEALTH_KINDS}
        return out

    def knowledge_phase_health_measured(self) -> bool:
        """Whether the knowledge phases this metric instruments actually ran.

        ``knowledge_phase_health_by_generation()`` returns ``{}`` for a healthy
        run AND for a ``--no-memory`` run (which skips the forum, distill, and
        seed phases entirely). This boolean disambiguates the two: ``True`` means
        the instrumented phases ran, so an empty health block is genuinely clean;
        ``False`` means measurement did not happen and ``{}`` carries no signal
        Surfaced into the results JSON as
        ``knowledge_phase_health_measured``.
        """
        return not self.config.no_memory

    def run(self, tasks: list[TaskSpec]) -> list[TaskTrace]:
        # Fail loudly on a removed distillation channel/strategy env var BEFORE
        # any work begins. The in-distill() call to this same guard is swallowed
        # by the distill phase's `except Exception` (it would silently skip
        # distillation), and it never fires at all on --no-memory / raw_attempts
        # runs that don't reach distill(); validating here covers every run.
        assert_no_removed_channel_env()
        # Provenance stamp: stamp code commit / resolved model / scoring
        # mode onto the runs row as early as possible, so the stamp lands even if
        # the run fails before any generation completes (the per-generation
        # re-stamps below, for both self._knowledge and self._memory_store,
        # retry this if it fails transiently here).
        provenance_store = self._knowledge if self._knowledge is not None else self._memory_store
        if provenance_store is not None:
            try:
                provenance_store.ensure_run(
                    self.config.experiment_name,
                    code_commit=self.config.code_commit,
                    resolved_model=f"{self.config.model_provider}/{self.config.model}",
                    scoring_mode=self.config.scoring_mode,
                    config_json=self.config.config_json,
                )
            except Exception as exc:
                log.warning("[ENGINE] Failed to stamp run provenance metadata: %s", exc)
        self.accumulator = TokenAccumulator()
        # Ensure the shared embedding cache is fully populated by the
        # host before any container mounts it read-only. Non-fatal: a load
        # failure leaves in-container semantic retrieval to degrade to FTS, same as today.
        if self._embedder is not None:
            if not self._embedder.wait_ready(timeout=600):
                log.warning(
                    "[ENGINE] embedding model not ready before launch; "
                    "in-container semantic retrieval may fall back to FTS"
                )
        # On --resume the accumulator above starts empty, so token_usage_total
        # would undercount every generation completed before the resume cursor.
        # Rehydrate the persisted token_phases rows for prior
        # generations from the runtime DB — the only store that holds them.
        if self._start_generation > 1:
            if self._memory_store is not None:
                try:
                    replayed = self.accumulator.load_from_store(
                        self._memory_store,
                        experiment=self.config.experiment_name,
                        before_generation=self._start_generation,
                    )
                    if replayed:
                        log.info(
                            "[ENGINE] --resume: rehydrated %d token_phases row(s) from generations < %d",
                            replayed,
                            self._start_generation,
                        )
                    else:
                        log.warning(
                            "[ENGINE] --resume: no token_phases rows found for experiment=%r before "
                            "generation %d; token_usage_total will undercount prior generations "
                            "(check --runtime-db-path / --experiment-name match the original run).",
                            self.config.experiment_name,
                            self._start_generation,
                        )
                except Exception as exc:
                    log.warning("[ENGINE] Failed to rehydrate token accumulator on resume: %s", exc)
            else:
                log.warning(
                    "[ENGINE] --resume without a runtime DB: token_usage_total will "
                    "undercount generations before the resume cursor."
                )
        all_traces: list[TaskTrace] = []
        # Hold-out transfer probe: split the designated hold-out tasks out of
        # the training pool. ``remaining_tasks`` (drop-solved bookkeeping,
        # early-stop, seeding pool size) tracks TRAINING tasks only; hold-out
        # tasks are appended to every generation's dispatch set below.
        holdout_tasks = [t for t in tasks if self._is_holdout(t.id)]
        remaining_tasks = [t for t in tasks if not self._is_holdout(t.id)]
        if holdout_tasks:
            log.info(
                "[ENGINE] hold-out probe enabled: %d task(s) attempted every generation, "
                "excluded from learning and headline metrics: %s",
                len(holdout_tasks),
                ", ".join(t.id for t in holdout_tasks),
            )
        # Stash a task-id -> task lookup so phases that only receive
        # ``task_ids`` can resolve back to the original ``TaskSpec.metadata`` -
        # notably ``task_source``, which is
        # forwarded to the distillation prompts as a domain hint.
        self._tasks_by_id = {t.id: t for t in tasks}
        start_generation = self._start_generation

        if start_generation > 1:
            if self.config.drop_solved:
                remaining_tasks = self._next_remaining_tasks(remaining_tasks)
            if start_generation <= self.config.num_generations and remaining_tasks:
                self._seeding_phase.prepare_resume_population(
                    source_generation=start_generation - 1,
                    next_tasks=remaining_tasks + holdout_tasks,
                )
            elif start_generation > self.config.num_generations:
                log.info(
                    "[ENGINE] Resume target already reached: latest generation=%d target=%d",
                    start_generation - 1,
                    self.config.num_generations,
                )

        try:
            for gen in range(start_generation, self.config.num_generations + 1):
                result = self._run_generation(gen, remaining_tasks, holdout_tasks)
                remaining_tasks = result.remaining_tasks
                all_traces.extend(result.traces)
                if result.should_stop:
                    break

            # Final summary: count unique solved for this invocation's task
            # set. Hold-out probe tasks are excluded from the headline counts.
            current_task_ids = {t.id for t in tasks if not self._is_holdout(t.id)}
            total_solved = sum(
                1
                for task_id in current_task_ids
                if (
                    self._best_scores.get(task_id) is not None
                    and self._best_scores[task_id] >= self.config.solved_threshold
                )
            )
            total_tasks = len(tasks) - len(holdout_tasks)
            holdout_stats = self.holdout_solve_rate_by_generation()
            holdout_summary = ""
            if holdout_stats:
                last_gen = max(holdout_stats)
                hs = holdout_stats[last_gen]
                holdout_summary = " holdout=%d/%d (%.1f%%)" % (hs["solved"], hs["total"], 100 * hs["rate"])
            log.info(
                "completed traces=%d tasks=%d solved=%d/%d (%.1f%%)%s",
                len(all_traces),
                total_tasks,
                total_solved,
                total_tasks,
                100 * total_solved / max(total_tasks, 1),
                holdout_summary,
            )
            if self._knowledge is not None:
                status = (
                    "enabled"
                    if self._vector_embedding_count > 0
                    else "disabled"
                    if self._vector_disabled
                    else "off"
                    if not self._vector_required
                    else "degraded"
                )
                self._knowledge.record_vector_status(
                    phase="run_summary",
                    status=status,
                    detail=(f"embeddings={self._vector_embedding_count}, skipped={self._vector_skipped_count}"),
                    embedding_count=self._vector_embedding_count,
                    skipped_count=self._vector_skipped_count,
                    experiment=self.config.experiment_name,
                )
                if self._vector_required and self._vector_embedding_count <= 0:
                    raise RuntimeError("Vector memory was required but no embeddings were written")
            self.persistence.on_run_end(token_summary=self.accumulator.total())
            return all_traces
        finally:
            if self._memory_store is not None:
                try:
                    self._memory_store.close()
                except Exception:
                    log.warning("[ENGINE] Failed to close _memory_store", exc_info=True)
            if self._knowledge is not None:
                try:
                    self._knowledge.close()
                except Exception:
                    log.warning("[ENGINE] Failed to close _knowledge store", exc_info=True)
            if hasattr(self.runtime, "close"):
                try:
                    self.runtime.close()
                except Exception:
                    log.warning("[ENGINE] Failed to close runtime", exc_info=True)

    def _run_generation(
        self,
        gen: int,
        remaining_tasks: list[TaskSpec],
        holdout_tasks: list[TaskSpec],
    ) -> _GenerationResult:
        """Run one generation: CLAIM -> ENRICH -> EXECUTE -> improve -> SEED.

        Pure extraction of the former ``run()`` loop body — every side effect,
        phase order, log line, and ``raise`` path is unchanged; ``should_stop``
        carries the two early-stop breaks.
        """
        # Drop solved tasks (None means never scored — keep it)
        if self.config.drop_solved:
            before = len(remaining_tasks)
            remaining_tasks = [
                t
                for t in remaining_tasks
                if self._best_scores.get(t.id) is None or self._best_scores[t.id] < self.config.solved_threshold
            ]
            dropped = before - len(remaining_tasks)
            if dropped:
                log.info(
                    "[gen %s] dropped %d solved task(s), %d remaining",
                    gen,
                    dropped,
                    len(remaining_tasks),
                )

        if not remaining_tasks:
            log.info("[gen %s] no remaining tasks — stopping early", gen)
            return _GenerationResult(remaining_tasks=remaining_tasks, traces=[], should_stop=True)

        # Dispatch set = surviving training tasks + ALL hold-out tasks.
        # Hold-outs are re-attempted every generation (fresh, even
        # after being solved) while staying out of ``remaining_tasks``
        # bookkeeping above.
        gen_tasks = remaining_tasks + holdout_tasks
        if holdout_tasks:
            log.info(
                "[gen %s] appending %d hold-out probe task(s) to the dispatch set",
                gen,
                len(holdout_tasks),
            )

        self._align_task_spawn_agents(len(gen_tasks))

        self.persistence.on_generation_start(generation=gen, agents=list(self.agents))

        # Phase 1: CLAIM
        assignments = self._claim_phase.claim(gen, gen_tasks)
        assigned_map: dict[str, list[str]] = defaultdict(list)
        for a in assignments:
            assigned_map[a.agent_id].append(a.task_id)
        _assert_single_task_per_agent(assigned_map, generation=gen)
        self.persistence.on_assignment(
            generation=gen,
            assigned=dict(assigned_map),
            total_tasks=len(gen_tasks),
        )
        task_by_id = {t.id: t for t in gen_tasks}

        execute_map, carried_traces = self._resume_phase.split_assignments(
            generation=gen,
            assigned_map=assigned_map,
            task_by_id=task_by_id,
        )

        # Phase 1.5: ENRICH seed packages — only for tasks that will execute
        # (carried-forward tasks are already solved and skip the container).
        if execute_map:
            self._enrichment_phase.enrich(gen, execute_map, gen_tasks)
        if carried_traces:
            agent_by_id = {agent.id: agent for agent in self.agents}
            log.info(
                "[gen %s] carried forward %d task(s) at or above solved_threshold=%.3f",
                gen,
                len(carried_traces),
                float(self.config.solved_threshold),
            )
            carried_persist_error: BaseException | None = None
            for trace in carried_traces:
                try:
                    self._resume_phase.persist_carried(trace, task_by_id)
                except Exception as exc:
                    if carried_persist_error is None:
                        carried_persist_error = exc
                    log.error(
                        "[ENGINE] authoritative persist failed for carried-forward "
                        "task %s; aborting after the carried loop: %s",
                        trace.task_id,
                        exc,
                    )
                agent = agent_by_id.get(trace.agent_id)
                if agent is not None and trace.error is None:
                    agent.tasks_completed += 1
            if carried_persist_error is not None:
                raise RuntimeError(
                    f"authoritative KnowledgeStore persist failed for a carried-forward task in generation {gen}"
                ) from carried_persist_error

        # Phase 2: EXECUTE
        gen_traces = list(carried_traces)
        if execute_map:
            execution = self._execution_phase.run(
                ExecutionPhaseInput(
                    generation=gen,
                    tasks=gen_tasks,
                    assigned_map=execute_map,
                )
            )
            gen_traces.extend(execution.traces)

        # Update best scores (hold-out traces feed the per-gen probe
        # metric instead — never the headline best-score bookkeeping).
        self._update_score_tracking(gen, gen_traces)

        fresh_gen_traces = [trace for trace in gen_traces if not is_carried_forward_trace(trace)]

        # Improvement-mechanism phases (Phase 2/3/4) run through the
        # ImprovementStrategy seam.  ``ctx`` exposes the phase
        # capabilities and per-generation state the inline calls
        # consumed, while the engine keeps the try/except +
        # AuthenticationFailure re-raise policy that wraps each phase.
        ctx = GenerationContext(
            generation=gen,
            fresh_traces=fresh_gen_traces,
            phases=self._improvement_phases,
            next_task_pool_size=self._next_remaining_task_count(remaining_tasks),
            config=self.config,
            knowledge=self._knowledge,
            next_remaining_tasks=self._next_remaining_tasks,
        )

        # Phase 2: PER-TASK FORUM
        try:
            self._improvement_strategy.per_task_forum(ctx)
        except AuthenticationFailure:
            raise
        except Exception as exc:
            log.error("[ENGINE] Forum phase failed (gen=%d): %s — continuing without forum", gen, exc)

        # Phase 3: CROSS-TASK FORUM (shared room, one agent per container)
        try:
            self._improvement_strategy.cross_task_forum(ctx)
        except AuthenticationFailure:
            raise
        except Exception as exc:
            log.error(
                "[ENGINE] Cross-task forum phase failed (gen=%d): %s — continuing",
                gen,
                exc,
            )

        # Phase 4: DISTILL per-task + cross-task bundles
        try:
            self._improvement_strategy.distill(ctx)
        except (AuthenticationFailure, DistillationStalledError):
            # DistillationStalledError is the opt-in hard abort:
            # propagate to stop the run, don't swallow-and-continue.
            raise
        except Exception as exc:
            log.error(
                "[ENGINE] Distill phase failed (gen=%d): %s — continuing",
                gen,
                exc,
            )

        # Phase 5: SEED (prepare next gen)
        self.persistence.on_generation_end(generation=gen, agents=list(self.agents))

        # Flush token usage for this generation to the runtime DB.
        if self._memory_store is not None:
            try:
                run_id = self._memory_store.ensure_run(
                    self.config.experiment_name,
                    code_commit=self.config.code_commit,
                    resolved_model=f"{self.config.model_provider}/{self.config.model}",
                    scoring_mode=self.config.scoring_mode,
                    config_json=self.config.config_json,
                )
                self.accumulator.flush_to_store(
                    self._memory_store,
                    run_id=run_id,
                    generation=gen,
                    model=self.config.model,
                )
            except Exception as exc:
                log.warning("[ENGINE] Failed to flush token usage for gen %d: %s", gen, exc)

        # Re-stamp provenance metadata on the knowledge store too,
        # each generation — retries the run-start stamp above if it failed
        # transiently before any generation completed.
        if self._knowledge is not None:
            try:
                self._knowledge.ensure_run(
                    self.config.experiment_name,
                    code_commit=self.config.code_commit,
                    resolved_model=f"{self.config.model_provider}/{self.config.model}",
                    scoring_mode=self.config.scoring_mode,
                    config_json=self.config.config_json,
                )
            except Exception as exc:
                log.warning("[ENGINE] Failed to stamp run provenance metadata for gen %d: %s", gen, exc)

        seed_plan = plan_seed_next_generation(self._improvement_strategy, ctx, remaining_tasks=remaining_tasks)
        if seed_plan.action == "stop":
            log.info(
                "[gen %s] %s — stopping before generation %s", gen, seed_plan.reason or "strategy stopped", gen + 1
            )
            return _GenerationResult(remaining_tasks=remaining_tasks, traces=gen_traces, should_stop=True)
        if seed_plan.action == "seed":
            next_tasks = list(seed_plan.next_tasks)
            # Append hold-out labels so the seeder sizes for them too; hold-out
            # tasks have no per-task bundle, which the seeder already tolerates.
            # The strategy stop decision above intentionally runs on training only.
            self._pending_next_task_labels = [task.id for task in next_tasks] + [task.id for task in holdout_tasks]
            # Phase 5: SEED via the improvement-strategy seam.  Reuse the
            # per-generation ctx, refreshing next_task_pool_size to the
            # seed-time value (the inline call passed this exact count).
            ctx.next_task_pool_size = len(next_tasks) + len(holdout_tasks)
            self._improvement_strategy.seed_next_generation(ctx)
            return _GenerationResult(remaining_tasks=next_tasks, traces=gen_traces, should_stop=False)
        elif seed_plan.action != "skip":
            raise RuntimeError(f"unsupported seed schedule action: {seed_plan.action!r}")
        return _GenerationResult(remaining_tasks=remaining_tasks, traces=gen_traces, should_stop=False)

    def _update_score_tracking(self, generation: int, gen_traces: list[TaskTrace]) -> None:
        """Update best-score / preserved-trace bookkeeping for ``gen_traces``.

        Hold-out traces are diverted into ``_holdout_gen_results`` (per-gen,
        NON-cumulative) and never touch ``_best_scores`` or
        ``_best_preserved_traces`` — so ``--drop-solved``, early-stop, and
        carried-forward replay see training tasks only.
        """
        for trace in gen_traces:
            task_obj = self._tasks_by_id.get(trace.task_id)
            score = trace_preserve_score(trace, task=task_obj)
            if self._is_holdout(trace.task_id):
                if score is not None:
                    gen_scores = self._holdout_gen_results.setdefault(generation, {})
                    prev = gen_scores.get(trace.task_id)
                    if prev is None or score > prev:
                        gen_scores[trace.task_id] = score
                continue
            if score is not None:
                current = self._best_scores.get(trace.task_id, float("-inf"))
                if score > current:
                    self._best_scores[trace.task_id] = score
            if not is_carried_forward_trace(trace) and trace_meets_preserve_threshold(
                trace, task=task_obj, solved_threshold=self.config.solved_threshold
            ):
                cached = self._best_preserved_traces.get(trace.task_id)
                if cached is None or trace_preserve_rank(trace, task=task_obj) > trace_preserve_rank(
                    cached, task=task_obj
                ):
                    self._best_preserved_traces[trace.task_id] = copy.deepcopy(trace)

    def _next_remaining_task_count(self, remaining_tasks: list[TaskSpec]) -> int:
        return len(self._next_remaining_tasks(remaining_tasks))

    def _next_remaining_tasks(self, remaining_tasks: list[TaskSpec]) -> list[TaskSpec]:
        if not self.config.drop_solved:
            return list(remaining_tasks)
        return [
            t
            for t in remaining_tasks
            if self._best_scores.get(t.id) is None or self._best_scores[t.id] < self.config.solved_threshold
        ]

    def _align_task_spawn_agents(self, task_pool_size: int) -> None:
        target = max(0, int(task_pool_size))
        current = len(self.agents)
        if target == current:
            return
        if target <= 0:
            self.agents = []
            return
        if target < current:
            self.agents = self.agents[:target]
            return
        # For gen 1 this is normal (initial population); for gen 2+ the seeder
        # handles sizing before _align_task_spawn_agents runs, so blank agents
        # here indicate an unexpected scale-up.
        next_generation = max((a.generation for a in self.agents), default=1)
        if next_generation > 1:
            log.warning("Appending %d blank agents in gen %d (no seed package)", target - current, next_generation)
        for i in range(current, target):
            self.agents.append(AgentState(id=f"agent-{i}", generation=next_generation))

    def get_claim_debug_history(self) -> list[dict[str, Any]]:
        """Return captured claim-phase debug artifacts per generation."""
        return list(self._claim_phase.debug_history())

    @staticmethod
    def _knowledge_trace_condensed(
        trace: TaskTrace,
        *,
        insight_text: str = "(pending reflection)",
    ) -> str:
        task_source = str(((trace.runtime_meta or {}).get("task_source") or "")).strip().lower()
        # A task source may register a ``trace_condensed`` formatter on its spec
        # (wired below by _attach_engine_source_formatters); sources without one
        # use the generic approach-excerpt default. No per-source dispatch here.
        spec = resolve_source(task_source)
        formatter = spec.trace_condensed if spec is not None else None
        if formatter is not None:
            return formatter(trace, insight_text=insight_text)
        final_output = trace.model_output or ""
        if trace.error and not final_output:
            return f"task failed: {trace.error}"
        approach_excerpt = (
            _extract_approach_excerpt(
                final_output or "",
                max_chars=1000,
            )
            or "(no output)"
        )
        return (
            f"Approach: {approach_excerpt}. "
            f"Score: {trace.native_score}. "
            f"Insight: {insight_text or '(pending reflection)'}"
        )

    @staticmethod
    def _extract_key_files(model_output: str) -> list[str]:
        """Extract changed file paths from diff headers in model output."""
        paths = re.findall(r"diff --git a/(.+?) b/", model_output)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def _persist_task_summary(
        self,
        trace: TaskTrace,
        task_by_id: dict[str, Any],
        *,
        lessons: list[str] | None = None,
    ) -> None:
        """Persist a structured task summary with optional vector embedding.

        The ``approach`` field stores a *failure diagnosis* (what went wrong and
        which tests failed) rather than the raw model output.  This prevents
        memory anchoring — agents reading the summary learn what to AVOID
        rather than being tempted to copy a subtly-wrong patch verbatim.
        """
        if self._memory_store is None or trace.error is not None:
            return
        try:
            import uuid

            summary_id = str(uuid.uuid4())
            key_files = self._extract_key_files(trace.model_output or "")
            eval_res = trace.eval_result or {}
            outcome = "resolved" if eval_res.get("resolved") or trace.native_score == 1.0 else "unresolved"
            score = trace.native_score
            task_obj = task_by_id.get(trace.task_id)
            repo = getattr(task_obj, "repo", "") or "" if task_obj else ""
            task_metadata = (task_obj.metadata or {}) if task_obj else {}
            task_source = str(task_metadata.get("task_source") or "").strip().lower()
            # Upstream-strict by default for SWE-bench Pro: anonymize test
            # names in the persisted approach diagnosis. The metadata flag is
            # stamped by ``prepare_swebench_repo_snapshots`` and
            # defaults to False in every other reader (``runtime/seeding.py``,
            # ``prompts/__init__.py``, ``tasks/repo_cache.py``, CLI).
            # The leakage path (test-name LISTS in ``attempt_history_json``,
            # surfaced to next-gen agents via the MCP ``query`` tool) is
            # opt-in via an explicit ``swebench_pro_seed_tests=True``
            # (DGM-equivalent mode). For non-Pro tasks the metadata key is
            # absent and ``False`` is the correct default — non-Pro tasks
            # don't have grader-test-name-leak concerns.
            seed_test_files = bool(task_metadata.get("swebench_pro_seed_tests", False))

            approach = _build_approach_diagnosis(
                trace=trace,
                eval_result=eval_res,
                outcome=outcome,
                task_source=task_source,
                seed_test_files=seed_test_files,
            )

            # Audit-only write: the runtime DB no longer keeps a vector index,
            # so no embedding is computed here. Semantic retrieval lives in the
            # knowledge DB (KnowledgeStore vec/FTS paths).
            self._memory_store.insert_task_summary(
                id=summary_id,
                experiment=self.config.experiment_name,
                agent_id=trace.agent_id,
                generation=trace.generation,
                task_id=trace.task_id,
                repo=repo,
                approach=approach,
                key_files=key_files,
                outcome=outcome,
                score=score,
                lessons=lessons or [],
            )
        except Exception as exc:
            log.warning("[ENGINE] Failed to persist task summary for %s: %s", trace.task_id, exc)

    def _safe_on_task_trace(self, trace: TaskTrace) -> None:
        """Fire the best-effort audit-sidecar ``on_task_trace`` callback.

        The runtime DB sidecar is best-effort by design (a failed write is
        logged at WARNING and dropped, never fatal). This must NOT be called
        bare inside a collection loop: a ``WriteIndeterminateError`` (or any
        sidecar fault) would break the loop and silently drop every
        not-yet-collected task. Swallow-and-warn keeps the main path alive.
        """
        try:
            self.persistence.on_task_trace(trace)
        except AuthenticationFailure:
            raise
        except Exception as exc:  # best-effort sidecar — must never abort the run
            log.warning(
                "[ENGINE] on_task_trace sidecar write failed for %s: %s",
                trace.task_id,
                exc,
            )

    def _persist_task_memory_record(
        self,
        *,
        trace: TaskTrace,
        insight: Insight | None,
        lessons: list[str] | None,
        agent: AgentState | None = None,
    ) -> None:
        """Persist task-centric canonical memory record keyed by task_id.

        The legacy ``MemoryStore`` and the unified ``KnowledgeStore`` are
        independent SQLite files. We write to both here — a failure in one
        must NOT skip the other. Previously the KnowledgeStore write was
        nested inside the outer ``try:`` for the legacy write, so any
        ``sqlite3.OperationalError: database is locked`` from the legacy
        store silently dropped the attempt from the knowledge store too.
        """
        if self._memory_store is None and self._knowledge is None:
            return
        # Build the persistable payload once. Pure data prep — if this raises
        # (e.g., a malformed trace), we log once and skip both writes since
        # neither would get meaningful content.
        try:
            eval_results = trace.eval_result or {}
            final_output = trace.model_output or ""
            full_trace = str((trace.runtime_meta or {}).get("native_session_memory") or "").strip()
            if not full_trace:
                full_trace = final_output
            if not full_trace and trace.error:
                full_trace = f"runtime_error: {trace.error}"

            insight_text = insight.text.strip() if insight and insight.text else ""
            compact_lines: list[str] = []
            if insight_text:
                compact_lines.append(insight_text)
            for lesson in lessons or []:
                if isinstance(lesson, str) and lesson.strip():
                    compact_lines.append(lesson.strip())

            approach_excerpt = _extract_approach_excerpt(final_output or "", max_chars=1000) or "(no output)"
            score = trace.native_score
            if insight_text:
                _insight_for_condensed = insight_text
            elif compact_lines:
                _insight_for_condensed = compact_lines[0]
            else:
                _insight_for_condensed = "(no insight)"

            if trace.error and not final_output:
                condensed = f"task failed: {trace.error}"
            else:
                condensed = f"Approach: {approach_excerpt}. Score: {score}. Insight: {_insight_for_condensed}"

            task_specific_insights: list[str] = []
            seen: set[str] = set()
            for item in compact_lines:
                if item not in seen:
                    seen.add(item)
                    task_specific_insights.append(item)

            # Upstream-strict by default for SWE-bench Pro: anonymize test
            # names in the attempt history (see corresponding comment at the
            # _build_approach_diagnosis call above). Default flipped from
            # True to False in the swebench_pro_seed_tests fix; the leakage
            # path is now opt-in via explicit metadata stamping.
            task_for_meta = self._tasks_by_id.get(trace.task_id) if hasattr(self, "_tasks_by_id") else None
            task_meta_for_event = (task_for_meta.metadata or {}) if task_for_meta else {}
            attempt_seed_test_files = bool(task_meta_for_event.get("swebench_pro_seed_tests", False))
            attempt_event = _build_attempt_event(
                native_score=trace.native_score,
                error=trace.error or "",
                eval_results=eval_results or {},
                model_output=final_output,
                runtime_meta=trace.runtime_meta or {},
                seed_test_files=attempt_seed_test_files,
            )
            injected_memory_md = str((trace.runtime_meta or {}).get("injected_memory_md") or "")
            cf_payload = carry_forward_payload(trace.runtime_meta)
            attempt_meta_for_knowledge = self._merge_attempt_meta(
                cf_payload,
                self._retrieved_distillation_ids(agent),
            )
            # A task source may register an ``attempt_meta_builder`` on its spec
            # (wired in attempt_events._attach_engine_source_formatters); sources
            # without one contribute no extra attempt_meta. No per-source dispatch here.
            attempt_task_source = str(((trace.runtime_meta or {}).get("task_source") or "")).strip().lower()
            attempt_spec = resolve_source(attempt_task_source)
            attempt_meta_builder = attempt_spec.attempt_meta_builder if attempt_spec is not None else None
            attempt_meta_for_knowledge = self._merge_optional_meta(
                attempt_meta_for_knowledge,
                attempt_meta_builder(trace) if attempt_meta_builder is not None else None,
            )
            attempt_meta_for_knowledge = self._tag_holdout_meta(attempt_meta_for_knowledge, trace.task_id)
        except Exception as exc:
            # The payload build has no side effects, so a silent ``return`` here
            # would drop the attempt from BOTH the sidecar and the authoritative
            # KnowledgeStore (the writes below never run). Re-raise so the failure
            # surfaces through the caller's deferred-error mechanism (see
            # execution_phase.py / _resume_phase.persist_carried), exactly like an
            # authoritative-write failure below.
            log.error(
                "[ENGINE] Failed to build task memory payload for %s: %s",
                trace.task_id,
                exc,
                exc_info=True,
            )
            raise

        # Independent write path 1: legacy MemoryStore (task_memory_records).
        if self._memory_store is not None:
            try:
                self._memory_store.upsert_task_memory_record(
                    experiment=self.config.experiment_name,
                    generation=trace.generation,
                    agent_id=trace.agent_id,
                    task_id=trace.task_id,
                    eval_results=eval_results,
                    final_model_output=final_output,
                    full_memory_trace=full_trace,
                    full_memory_trace_condensed=condensed,
                    task_specific_insights=task_specific_insights,
                    attempt_event=attempt_event,
                    injected_memory_md=injected_memory_md,
                )
            except Exception as exc:
                log.warning(
                    "[ENGINE] Failed to persist task memory record for %s: %s",
                    trace.task_id,
                    exc,
                    exc_info=True,
                )

        # Independent write path 2: unified KnowledgeStore (knowledge table).
        # Runs even if the legacy write above raised — the two stores are
        # separate SQLite files and must not share a failure domain.
        #
        # ``supersede=True`` + a stable ``external_id`` (shared with
        # execution_phase._persist_knowledge_attempt_early) means this
        # write UPDATES the early resume-safety placeholder in place when
        # it already ran (~99% of the time), instead of being skipped
        # outright — that skip previously stranded every attempt row with
        # ``insights=[]`` and no reflection. When the early
        # write did NOT run, ``record_attempt`` falls through to a normal
        # insert, so this is safe unconditionally.
        if self._knowledge is not None:
            try:
                # Embed the condensed trace so attempt rows participate in
                # semantic search.  ``_maybe_embed`` returns None if the
                # embedder is still loading in the background or disabled,
                # in which case the attempt is written without an embedding
                # (FTS-only) — consistent with how forum-drain posts handle
                # a not-ready embedder.
                attempt_embed_text = condensed or final_output or ""
                attempt_embedding = self._maybe_embed(attempt_embed_text)
                reflection_text = str((trace.runtime_meta or {}).get("phase1_reflection") or "").strip()
                self._knowledge.record_attempt(
                    task_id=trace.task_id,
                    agent_id=trace.agent_id,
                    generation=trace.generation,
                    eval_results=eval_results,
                    model_output=final_output,
                    trace_condensed=self._knowledge_trace_condensed(
                        trace,
                        insight_text=_insight_for_condensed,
                    ),
                    insights=task_specific_insights,
                    native_score=trace.native_score,
                    experiment=self.config.experiment_name,
                    embedding=attempt_embedding,
                    attempt_meta=attempt_meta_for_knowledge,
                    reflection=reflection_text,
                    repo=trace.repo,
                    external_id=_knowledge_attempt_external_id(
                        task_id=trace.task_id,
                        agent_id=trace.agent_id,
                        generation=trace.generation,
                    ),
                    supersede=True,
                )
            except Exception as exc:
                log.warning(
                    "[ENGINE] KnowledgeStore record_attempt failed for %s: %s",
                    trace.task_id,
                    exc,
                    exc_info=True,
                )
                raise RuntimeError(f"authoritative KnowledgeStore record_attempt failed for {trace.task_id}") from exc

    def _make_distill_llm(
        self,
        *,
        generation: int,
        phase: str,
        model_override: str | None = None,
    ):
        """Build an ``LLMCallable`` adapter around the orchestrator LLM.

        The returned adapter is called as ``adapter(system, user, *,
        json_schema=None)``. When a ``json_schema`` is requested AND
        ``self.llm`` advertises ``supports_json_schema``, the schema is
        forwarded and the adapter returns a ``DistillLLMResult(text, parsed)``
        carrier (``parsed`` is the provider-validated dict, else ``None`` and
        the distiller falls back to its lenient free-text parser); otherwise it
        returns plain text. Either way this wrapper records token usage into the
        lifecycle accumulator under ``agent_id="__distill__"`` and the given
        ``phase`` label (``"distill_per_task"`` or ``"distill_cross_task"``)
        so totals reflect actual cost.

        Distillation token instrumentation is centralized here. If a completed
        run has no ``token_phases`` rows for these labels, treat it as evidence
        the distill phase did not run or the LLM call raised before returning
        usage, not as a separate persistence path to patch.

        When ``model_override`` is provided, it is passed through to the
        underlying LLM caller via ``model=`` kwarg so that per-phase
        model overrides (``distill_per_task_model`` /
        ``distill_cross_task_model``) actually take effect.
        """

        # Distillation prompts produce nested JSON with up to 7 insight
        # lists, each with text/applies_when/does_not_apply_when/evidence/
        # confidence fields. The LLM caller's default 4096 max_tokens
        # truncates the closing braces every time on cross_task distill
        # (observed 10/10 failed parses on cross_task distill). Default 32768
        # — 8× the truncation point, well within Haiku/Sonnet/Opus 4.x's 64K
        # output limit. Models stop emitting when they're done; oversize cap
        # never costs anything when output is shorter. Tunable via
        # KSI_DISTILL_MAX_OUTPUT_TOKENS.
        distill_max_tokens = max(
            1024,
            int(os.environ.get("KSI_DISTILL_MAX_OUTPUT_TOKENS", "32768") or "32768"),
        )

        def call(
            system: str,
            user: str,
            *,
            json_schema: dict[str, Any] | None = None,
            cache_prefix: str | None = None,
        ) -> Any:
            llm_kwargs: dict[str, Any] = {"max_tokens": distill_max_tokens}
            if model_override:
                llm_kwargs["model"] = model_override
            # Forward the cross-task distill's shared-history cache_prefix so the
            # provider caller cache-reads it across targets.
            if cache_prefix:
                llm_kwargs["cache_prefix"] = cache_prefix
            # Request provider structured output when the distill caller asked
            # for it AND the underlying LLM caller advertises support. Unknown
            # providers (no ``supports_json_schema``) skip the schema and the
            # distill path falls back to its lenient free-text parser.
            if json_schema is not None and getattr(self.llm, "supports_json_schema", False):
                llm_kwargs["json_schema"] = json_schema

            def _attempt() -> Any:
                # Call self.llm.call directly to preserve the model and
                # max_tokens kwargs (which self._llm_call doesn't forward).
                # json_schema acceptance is decided statically above via
                # supports_json_schema, so we never probe for it at runtime —
                # a TypeError-string-match on json_schema would misread a real
                # TypeError raised INSIDE a schema-capable llm.call as "schema
                # rejected" and silently drop to a worse parse path.
                try:
                    return self.llm.call(system=system, user=user, **llm_kwargs)
                except TypeError as exc:
                    # A legacy/mock adapter (bare ``(system, user) -> str``) may
                    # reject the optional max_tokens/model/json_schema kwargs.
                    # ONLY the standard "unexpected keyword argument" rejection
                    # is a fall-back signal; a real TypeError from inside a
                    # provider surfaces (mirrors the distiller's _call_llm
                    # contract, so an internal bug isn't masked or double-called).
                    if "unexpected keyword argument" not in str(exc):
                        raise
                    return self._llm_call(system=system, user=user)

            try:
                # Ride out a transient host->provider blip (e.g. a DNS
                # ``getaddrinfo EAI_AGAIN``) rather than zeroing the whole
                # generation's distillation on the first connection error.
                # This single chokepoint covers per-task AND cross-task
                # distill; auth/deterministic failures are not retried.
                result = run_with_distill_retry(_attempt, generation=generation, phase=phase)
            except Exception as exc:
                log.warning("[ENGINE] distill LLM call raised: %s", exc)
                raise
            # LLMCaller contract: returns an LLMResponse (``parsed`` populated
            # only when a json_schema was requested and the provider returned a
            # parseable dict). Tolerate a bare return for legacy/mock adapters
            # that don't return an LLMResponse.
            if isinstance(result, LLMResponse):
                text = str(result.text or "")
                usage = result.usage
                parsed = result.parsed if isinstance(result.parsed, dict) else None
                if usage is not None:
                    try:
                        self.accumulator.record_lifecycle(
                            generation,
                            "__distill__",
                            phase,
                            usage,
                        )
                    except Exception as exc:
                        log.debug(
                            "[ENGINE] distill token accounting failed: %s",
                            exc,
                        )
                if json_schema is not None:
                    return DistillLLMResult(text=text, parsed=parsed)
                return text
            # Bare result (e.g. a legacy ``(system, user) -> str`` adapter or a
            # test mock). Preserve the structured-result contract: when a schema
            # was requested, always hand back a DistillLLMResult so the distill
            # consumer's isinstance detection holds even on this path
            # (parsed is None -> lenient parser).
            if json_schema is not None:
                return DistillLLMResult(text=str(result or ""), parsed=None)
            return str(result or "")

        return call

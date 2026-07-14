"""Seeding phase service for the generational orchestrator."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..models import AgentState, TaskSpec
    from .engine import GenerationalOrchestrator

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeedingPhaseInput:
    """Inputs required to seed the next generation."""

    generation: int
    next_task_pool_size: int | None = None


@dataclass(frozen=True)
class SeedingPhaseResult:
    """Observable result of a seed phase invocation."""

    agent_count: int = 0
    task_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class SeedingCollaborators:
    """Explicit dependencies for the seeding phase service body."""

    config: Any
    knowledge: Any | None
    seeder: Any  # engine._seeder
    population: Any  # engine._population
    holdout_ids: Any  # engine._holdout_ids
    record_phase_failure: Callable[..., None]
    agents: list[Any]  # snapshot of engine.agents at build time (read)
    set_agents: Callable[[list[Any]], None]
    pending_next_task_labels: list[str]  # snapshot (read)
    set_pending_next_task_labels: Callable[[list[str]], None]


@dataclass
class EngineSeedingPhaseService:
    """Engine-backed implementation of resume and next-generation seeding."""

    engine: "GenerationalOrchestrator"

    def _collaborators(self) -> SeedingCollaborators:
        engine = self.engine
        return SeedingCollaborators(
            config=engine.config,
            knowledge=engine._knowledge,
            seeder=engine._seeder,
            population=engine._population,
            holdout_ids=engine._holdout_ids,
            record_phase_failure=engine._record_knowledge_phase_failure,
            agents=engine.agents,
            set_agents=lambda value: setattr(engine, "agents", value),
            pending_next_task_labels=engine._pending_next_task_labels,
            set_pending_next_task_labels=lambda value: setattr(engine, "_pending_next_task_labels", value),
        )

    def prepare_resume_population(
        self,
        *,
        source_generation: int,
        next_tasks: list["TaskSpec"],
    ) -> None:
        """Seed the first resumed generation from the latest completed bundles.

        A normal uninterrupted run calls ``run`` between
        generations. A resumed run stopped before that next-generation seed
        exists in memory, so we synthesize the same seed packages from the
        latest completed generation without re-running prior task/forum phases.
        """
        collab = self._collaborators()
        source_generation = int(source_generation)
        next_tasks = list(next_tasks or [])
        if source_generation < 1 or not next_tasks:
            return
        if collab.config.no_memory:
            return

        next_task_pool_size = len(next_tasks)

        next_num_agents = collab.population.next_agent_count(
            generation=source_generation + 1,
            remaining_tasks=next_task_pool_size,
        )
        if next_num_agents <= 0:
            collab.set_agents([])
            return

        task_labels = [task.id for task in next_tasks[:next_num_agents]]
        # Under target-conditioning the seeder loads a per-task cross-task
        # bundle for each agent's label; no single broadcast bundle is loaded.
        conditioning = bool(getattr(collab.config, "cross_task_distill_target_conditioning", True))
        cross_task_bundle = None if conditioning else self.load_cross_task_seed_bundle(generation=source_generation)
        new_agents = collab.seeder.seed(
            num_agents=next_num_agents,
            task_labels=task_labels,
            cross_task_bundle=cross_task_bundle,
            knowledge_store=collab.knowledge,
            generation=source_generation,
            experiment=collab.config.experiment_name,
            skip_per_task_labels=collab.holdout_ids,
            cross_task_target_conditioning=conditioning,
        )

        collab.set_agents(new_agents)
        self.persist_seed_snapshots_once(
            generation=source_generation,
            agents=new_agents,
        )
        log.info(
            "[ENGINE] Prepared resume population from generation %d for generation %d (%d agent(s), %d task(s))",
            source_generation,
            source_generation + 1,
            len(new_agents),
            len(next_tasks),
        )

    def persist_seed_snapshots_once(
        self,
        *,
        generation: int,
        agents: list["AgentState"],
    ) -> None:
        """Persist source-generation seed snapshots once, avoiding duplicates."""
        collab = self._collaborators()
        if collab.knowledge is None:
            return
        try:
            existing = collab.knowledge.count_seed_snapshots(
                generation=generation,
                experiment=collab.config.experiment_name,
            )
        except Exception as exc:
            log.warning(
                "[ENGINE] count_seed_snapshots failed for gen=%s: %s",
                generation,
                exc,
            )
            existing = 0
        if existing > 0:
            log.info(
                "[ENGINE] Seed snapshots already exist for gen=%s; skipping resume snapshot persistence",
                generation,
            )
            return
        for agent in agents:
            if isinstance(agent.seed_package, dict) and agent.seed_package:
                try:
                    collab.knowledge.record_seed_snapshot(
                        generation=generation,
                        payload=agent.seed_package,
                        agent_id=agent.id,
                        experiment=collab.config.experiment_name,
                    )
                except Exception as exc:
                    log.warning("Failed to persist seed snapshot for agent %s: %s", agent.id, exc)
                    raise

    def load_cross_task_seed_bundle(
        self,
        *,
        generation: int,
    ) -> dict[str, Any] | None:
        """Load the cross-task broadcast bundle for one seed phase."""
        collab = self._collaborators()
        if collab.knowledge is None:
            return None

        try:
            from ..memory.knowledge_store import CROSS_TASK_SENTINEL
            from ..runtime.seeding import is_canonical_distillation_bundle

            cross_task_bundle = collab.knowledge.load_distillation(
                generation=generation,
                task_id=CROSS_TASK_SENTINEL,
                scope="cross_task",
                experiment=collab.config.experiment_name,
            )
            if not is_canonical_distillation_bundle(cross_task_bundle):
                cross_task_bundle = None
        except Exception as exc:
            log.warning("[ENGINE] load_distillation(cross_task) failed for gen=%s: %r", generation, exc)
            # A load failure here silently degrades the NEXT generation's seed
            # (it starts with no cross-task bundle). Record it as a seed-phase
            # degradation event so the campaign-health gate sees it.
            collab.record_phase_failure(generation, "seed_failures")
            cross_task_bundle = None
        return cast(dict[str, Any] | None, cross_task_bundle)

    def run(self, phase_input: SeedingPhaseInput) -> SeedingPhaseResult:
        """Phase 5: seed next generation from distilled knowledge bundles."""
        collab = self._collaborators()
        generation = int(phase_input.generation)
        if collab.config.no_memory:
            log.info(
                "[ENGINE] no_memory is enabled; skipping seed phase for generation %s",
                generation,
            )
            collab.set_pending_next_task_labels([])
            return SeedingPhaseResult()

        # Preserve cumulative stats before replacing agents.
        old_usage = {a.id: a.token_usage for a in collab.agents}
        old_completed = {a.id: a.tasks_completed for a in collab.agents}

        next_task_pool_size = (
            len(collab.pending_next_task_labels)
            if phase_input.next_task_pool_size is None
            else int(phase_input.next_task_pool_size)
        )
        next_num_agents = collab.population.next_agent_count(
            generation=generation + 1,
            remaining_tasks=next_task_pool_size,
        )
        if next_num_agents <= 0:
            collab.set_agents([])
            return SeedingPhaseResult()

        pending = collab.pending_next_task_labels
        if next_num_agents < len(pending):
            log.warning(
                "[ENGINE] task_labels truncation: agents=%d < pending_labels=%d; "
                "some tasks will not receive agents (they may be silently dropped).",
                next_num_agents,
                len(pending),
            )
        task_labels = list(pending[:next_num_agents])
        # Under target-conditioning the seeder loads a per-task cross-task
        # bundle for each agent's label; otherwise load the single broadcast
        # bundle once for this seed phase. Per-task same-task bundles are always
        # loaded by the seeder per agent.
        conditioning = bool(getattr(collab.config, "cross_task_distill_target_conditioning", True))
        cross_task_bundle = None if conditioning else self.load_cross_task_seed_bundle(generation=generation)
        new_agents = collab.seeder.seed(
            num_agents=next_num_agents,
            task_labels=task_labels,
            cross_task_bundle=cross_task_bundle,
            knowledge_store=collab.knowledge,
            generation=generation,
            experiment=collab.config.experiment_name,
            skip_per_task_labels=collab.holdout_ids,
            cross_task_target_conditioning=conditioning,
        )

        # Carry over cumulative stats. Workstream/seed package comes from seeder.
        for agent in new_agents:
            agent.token_usage = old_usage.get(agent.id, 0)
            agent.tasks_completed = old_completed.get(agent.id, 0)

        # Persist seed snapshots for each agent.
        if collab.knowledge is not None:
            for agent in new_agents:
                if isinstance(agent.seed_package, dict) and agent.seed_package:
                    try:
                        collab.knowledge.record_seed_snapshot(
                            generation=generation,
                            payload=agent.seed_package,
                            agent_id=agent.id,
                            experiment=collab.config.experiment_name,
                        )
                    except Exception as exc:
                        log.warning("Failed to persist seed snapshot for agent %s: %s", agent.id, exc)
                        raise

        collab.set_agents(new_agents)
        collab.set_pending_next_task_labels([])
        return SeedingPhaseResult(agent_count=len(new_agents), task_labels=tuple(task_labels))

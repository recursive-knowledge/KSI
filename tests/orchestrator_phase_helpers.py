from __future__ import annotations

from typing import Any

from kcsi.models import AgentState, TaskSpec, TaskTrace
from kcsi.orchestrator.distillation_phase import DistillationPhaseInput
from kcsi.orchestrator.execution_phase import ExecutionPhaseInput
from kcsi.orchestrator.forum_phase import CrossTaskForumPhaseInput, PerTaskForumPhaseInput
from kcsi.orchestrator.seeding_phase import SeedingPhaseInput


def execute_generation(
    orch: Any,
    generation: int,
    tasks: list[TaskSpec],
    assigned_map: dict[str, list[str]],
) -> list[TaskTrace]:
    return orch._execution_phase.run(
        ExecutionPhaseInput(generation=generation, tasks=tasks, assigned_map=assigned_map)
    ).traces


def per_task_forum(
    orch: Any,
    generation: int,
    traces: list[TaskTrace],
    *,
    next_task_pool_size: int | None = None,
) -> None:
    orch._forum_phase_service.per_task_forum(
        PerTaskForumPhaseInput(
            generation=generation,
            traces=traces,
            next_task_pool_size=next_task_pool_size,
        )
    )


def cross_task_forum(orch: Any, *, generation: int, traces: list[TaskTrace]) -> None:
    orch._forum_phase_service.cross_task_forum(CrossTaskForumPhaseInput(generation=generation, traces=traces))


def run_distill(orch: Any, *, generation: int, task_ids: list[str]) -> None:
    orch._distillation_phase.run(DistillationPhaseInput(generation=generation, task_ids=task_ids))


def seed_next_generation(orch: Any, generation: int, *, next_task_pool_size: int | None = None) -> None:
    orch._seeding_phase.run(SeedingPhaseInput(generation=generation, next_task_pool_size=next_task_pool_size))


def prepare_resume_population(orch: Any, *, source_generation: int, next_tasks: list[TaskSpec]) -> None:
    orch._seeding_phase.prepare_resume_population(source_generation=source_generation, next_tasks=next_tasks)


def load_cross_task_seed_bundle(
    orch: Any,
    *,
    generation: int,
) -> dict[str, Any] | None:
    return orch._seeding_phase.load_cross_task_seed_bundle(generation=generation)


def persist_seed_snapshots_once(orch: Any, *, generation: int, agents: list[AgentState]) -> None:
    orch._seeding_phase.persist_seed_snapshots_once(generation=generation, agents=agents)

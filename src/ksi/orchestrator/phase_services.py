"""Engine-backed phase-service adapter for improvement strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .distillation_phase import DistillationPhaseInput, EngineDistillationPhaseService
from .forum_phase import (
    CrossTaskForumPhaseInput,
    EngineForumPhaseService,
    PerTaskForumPhaseInput,
)
from .seeding_phase import EngineSeedingPhaseService, SeedingPhaseInput

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..models import TaskTrace
    from .engine import GenerationalOrchestrator


@dataclass
class EngineImprovementPhaseServices:
    """Expose orchestrator improvement phases as explicit capabilities.

    This adapter coordinates the phase-specific services used by improvement
    strategies. Strategies do not reach through ``ctx.engine`` or name private
    engine methods.
    """

    engine: "GenerationalOrchestrator"
    forum_phase: EngineForumPhaseService | None = None
    distillation_phase: EngineDistillationPhaseService | None = None
    seeding_phase: EngineSeedingPhaseService | None = None

    def __post_init__(self) -> None:
        if self.forum_phase is None:
            self.forum_phase = EngineForumPhaseService(self.engine)
        if self.distillation_phase is None:
            self.distillation_phase = EngineDistillationPhaseService(self.engine)
        if self.seeding_phase is None:
            self.seeding_phase = EngineSeedingPhaseService(self.engine)

    def per_task_forum(
        self,
        generation: int,
        traces: "list[TaskTrace]",
        *,
        next_task_pool_size: int | None = None,
    ) -> None:
        assert self.forum_phase is not None
        self.forum_phase.per_task_forum(
            PerTaskForumPhaseInput(
                generation=generation,
                traces=traces,
                next_task_pool_size=next_task_pool_size,
            )
        )

    def cross_task_forum(self, *, generation: int, traces: "list[TaskTrace]") -> None:
        assert self.forum_phase is not None
        self.forum_phase.cross_task_forum(CrossTaskForumPhaseInput(generation=generation, traces=traces))

    def distill(self, *, generation: int, task_ids: list[str]) -> None:
        assert self.distillation_phase is not None
        self.distillation_phase.run(DistillationPhaseInput(generation=generation, task_ids=task_ids))

    def seed_next_generation(self, generation: int, *, next_task_pool_size: int | None = None) -> None:
        assert self.seeding_phase is not None
        self.seeding_phase.run(SeedingPhaseInput(generation=generation, next_task_pool_size=next_task_pool_size))

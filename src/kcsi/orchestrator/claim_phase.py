"""Claim phase-service boundary for the orchestrator."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..models import Assignment, TaskSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .engine import GenerationalOrchestrator

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClaimCollaborators:
    """Explicit dependencies for the claim phase service body."""

    agents: list[Any]  # live list ref (read for round-robin assignment)
    debug_sink: list[dict[str, Any]]  # live list ref (appended in place by claim)


@runtime_checkable
class ClaimPhaseService(Protocol):
    """Capability for running the deterministic claim phase."""

    def claim(self, generation: int, tasks: list[TaskSpec]) -> list[Assignment]: ...


@dataclass
class EngineClaimPhaseService:
    """Engine-backed claim phase adapter.

    The deterministic round-robin claim body lives here behind an explicit
    service boundary used by the generation loop and tests.
    """

    engine: "GenerationalOrchestrator"

    def _collaborators(self) -> ClaimCollaborators:
        engine = self.engine
        return ClaimCollaborators(
            agents=engine.agents,
            debug_sink=engine._claim_debug_history,
        )

    def claim(self, generation: int, tasks: list[TaskSpec]) -> list[Assignment]:
        """Phase 1: Distribute tasks evenly across agents deterministically."""
        collab = self._collaborators()

        if not tasks:
            return []

        assignments: list[Assignment] = []
        agent_ids = [a.id for a in collab.agents]
        if not agent_ids:
            return []

        claimed_agents: set[str] = set()
        next_agent = 0
        for task in tasks:
            # One task per agent: walk agent_ids in a ring until we find a free one.
            assigned = False
            for _ in range(len(agent_ids)):
                agent_id = agent_ids[next_agent % len(agent_ids)]
                next_agent += 1
                if agent_id in claimed_agents:
                    continue
                assignments.append(Assignment(generation=generation, agent_id=agent_id, task_id=task.id))
                claimed_agents.add(agent_id)
                # Task-mode: label agent with task ID (overwrites any seed-phase label).
                for agent in collab.agents:
                    if agent.id == agent_id:
                        agent.workstream = task.id
                        agent.workstream_description = task.id
                        break
                assigned = True
                break
            if not assigned:
                log.warning(
                    "[ENGINE] task %s unassignable: all %d agents already claimed a task", task.id, len(agent_ids)
                )

        claim_debug: dict[str, Any] = {
            "generation": generation,
            "mode": "deterministic",
            "num_agents": len(collab.agents),
            "num_tasks": len(tasks),
            "assignments": [{"agent_id": a.agent_id, "task_id": a.task_id} for a in assignments],
        }
        collab.debug_sink.append(claim_debug)
        return assignments

    def debug_history(self) -> list[dict[str, Any]]:
        """Return the live per-generation claim-phase debug sink."""
        return self._collaborators().debug_sink

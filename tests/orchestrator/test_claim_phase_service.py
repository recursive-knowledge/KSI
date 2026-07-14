from unittest.mock import MagicMock

import pytest

from ksi.models import GenerationConfig, TaskSpec
from ksi.orchestrator.claim_phase import (
    ClaimCollaborators,
    ClaimPhaseService,
    EngineClaimPhaseService,
)
from ksi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from tests.orchestrator_phase_decoupling_guard import functions_referencing_engine


def _make_orch(num_agents: int) -> GenerationalOrchestrator:
    return GenerationalOrchestrator(
        config=GenerationConfig(num_generations=1, num_agents=num_agents, no_memory=True),
        runtime=MagicMock(),
        evaluator=MagicMock(),
        llm=MagicMock(),
        persistence=NoopPersistence(),
    )


def test_engine_claim_phase_service_satisfies_protocol():
    service = EngineClaimPhaseService(_make_orch(2))
    assert isinstance(service, ClaimPhaseService)


def test_claim_body_has_no_engine_access():
    from ksi.orchestrator import claim_phase

    offenders = functions_referencing_engine(claim_phase.__file__)
    assert offenders <= {"_collaborators"}, offenders


def test_claim_collaborators_is_frozen():
    from dataclasses import FrozenInstanceError

    c = ClaimCollaborators(agents=[], debug_sink=[])
    with pytest.raises(FrozenInstanceError):
        c.agents = []  # type: ignore[misc]


def test_round_robin_one_task_per_agent_with_more_tasks_than_agents():
    """N agents, M>N tasks: each agent claims exactly one task round-robin;
    first-in-order tasks win, surplus tasks go unassigned."""
    service = EngineClaimPhaseService(_make_orch(3))
    tasks = [TaskSpec(id=f"t{i}") for i in range(5)]

    assignments = service.claim(1, tasks)

    # One task per agent (structural one-task-per-agent invariant).
    assert len(assignments) == 3
    by_agent: dict[str, list[str]] = {}
    for a in assignments:
        by_agent.setdefault(a.agent_id, []).append(a.task_id)
    assert all(len(tids) == 1 for tids in by_agent.values())
    assert set(by_agent.keys()) == {"agent-0", "agent-1", "agent-2"}
    # First three tasks in order win; surplus dropped.
    assert {a.task_id for a in assignments} == {"t0", "t1", "t2"}
    # Round-robin pairing: agent-i gets t-i.
    assert all(a.task_id == f"t{a.agent_id.split('-')[1]}" for a in assignments)


def test_round_robin_assigns_all_tasks_when_balanced():
    service = EngineClaimPhaseService(_make_orch(3))
    tasks = [TaskSpec(id="t1"), TaskSpec(id="t2"), TaskSpec(id="t3")]

    assignments = service.claim(1, tasks)

    assert len(assignments) == 3
    assert {a.task_id for a in assignments} == {"t1", "t2", "t3"}
    assert len({a.agent_id for a in assignments}) == 3


def test_debug_history_populated_and_is_engine_live_sink():
    orch = _make_orch(2)
    service = EngineClaimPhaseService(orch)

    service.claim(1, [TaskSpec(id="t0"), TaskSpec(id="t1")])

    hist = service.debug_history()
    assert hist is orch._claim_debug_history  # live sink, not a copy
    assert len(hist) == 1
    assert hist[0]["mode"] == "deterministic"
    assert hist[0]["generation"] == 1
    assert hist[0]["num_agents"] == 2
    assert hist[0]["num_tasks"] == 2


def test_claim_empty_tasks_returns_empty_without_debug_record():
    service = EngineClaimPhaseService(_make_orch(2))
    assert service.claim(1, []) == []
    assert service.debug_history() == []

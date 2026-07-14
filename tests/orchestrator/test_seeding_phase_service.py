from unittest.mock import MagicMock

from kcsi.models import AgentState, GenerationConfig
from kcsi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from kcsi.orchestrator.seeding_phase import (
    EngineSeedingPhaseService,
    SeedingPhaseInput,
    SeedingPhaseResult,
)
from tests.orchestrator_phase_decoupling_guard import functions_referencing_engine
from tests.orchestrator_phase_helpers import seed_next_generation


def _make_orch(config: GenerationConfig | None = None) -> GenerationalOrchestrator:
    return GenerationalOrchestrator(
        config=config or GenerationConfig(num_generations=2, num_agents=2),
        runtime=MagicMock(),
        evaluator=MagicMock(),
        llm=MagicMock(),
        persistence=NoopPersistence(),
    )


def test_engine_seeding_phase_service_exposes_run():
    orch = _make_orch()
    assert callable(EngineSeedingPhaseService(orch).run)


def test_seed_phase_test_helper_delegates_to_service():
    orch = _make_orch()
    fake_service = MagicMock()
    fake_service.run.return_value = SeedingPhaseResult()
    orch._seeding_phase = fake_service

    seed_next_generation(orch, 1, next_task_pool_size=3)

    fake_service.run.assert_called_once_with(SeedingPhaseInput(generation=1, next_task_pool_size=3))


def test_seed_next_generation_service_preserves_stats_and_returns_result(monkeypatch):
    orch = _make_orch(GenerationConfig(num_generations=2, num_agents=2))
    orch.agents = [
        AgentState(id="agent-0", token_usage=11, tasks_completed=2),
        AgentState(id="agent-1", token_usage=13, tasks_completed=3),
    ]
    orch._pending_next_task_labels = ["t1", "t2"]
    monkeypatch.setattr(orch._seeding_phase, "load_cross_task_seed_bundle", lambda **kwargs: None)

    result = orch._seeding_phase.run(SeedingPhaseInput(generation=1, next_task_pool_size=2))

    assert result == SeedingPhaseResult(agent_count=2, task_labels=("t1", "t2"))
    assert [agent.id for agent in orch.agents] == ["agent-0", "agent-1"]
    assert [agent.token_usage for agent in orch.agents] == [11, 13]
    assert [agent.tasks_completed for agent in orch.agents] == [2, 3]
    assert [agent.workstream for agent in orch.agents] == ["t1", "t2"]
    assert orch._pending_next_task_labels == []


def test_seeding_body_has_no_engine_access():
    from kcsi.orchestrator import seeding_phase

    offenders = functions_referencing_engine(seeding_phase.__file__)
    assert offenders <= {"_collaborators"}, offenders


def test_seeding_collaborators_is_frozen():
    from dataclasses import FrozenInstanceError

    from kcsi.orchestrator.seeding_phase import SeedingCollaborators

    c = SeedingCollaborators(
        config=object(),
        knowledge=None,
        seeder=object(),
        population=object(),
        holdout_ids=set(),
        record_phase_failure=lambda *a, **k: None,
        agents=[],
        set_agents=lambda v: None,
        pending_next_task_labels=[],
        set_pending_next_task_labels=lambda v: None,
    )
    try:
        c.agents = [1]  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("SeedingCollaborators must be frozen")


def _seed_spy_orch(*, conditioning: bool):
    orch = _make_orch(
        GenerationConfig(
            num_generations=2,
            num_agents=2,
            cross_task_distill_target_conditioning=conditioning,
        )
    )
    orch.agents = [AgentState(id="agent-0"), AgentState(id="agent-1")]
    orch._pending_next_task_labels = ["t1", "t2"]
    calls: dict = {}

    def spy_seed(**kwargs):
        calls.update(kwargs)
        return [
            AgentState(id="agent-0", workstream="t1"),
            AgentState(id="agent-1", workstream="t2"),
        ]

    orch._seeder.seed = spy_seed
    return orch, calls


def test_seeding_phase_conditioning_on_skips_broadcast_and_flags_seeder(monkeypatch):
    orch, calls = _seed_spy_orch(conditioning=True)
    # Broadcast loader must NOT be consulted when conditioning is on.
    orch._seeding_phase.load_cross_task_seed_bundle = lambda **k: (_ for _ in ()).throw(
        AssertionError("broadcast bundle must not be loaded under conditioning")
    )
    orch._seeding_phase.run(SeedingPhaseInput(generation=1, next_task_pool_size=2))
    assert calls["cross_task_target_conditioning"] is True
    assert calls["cross_task_bundle"] is None


def test_seeding_phase_conditioning_off_loads_broadcast(monkeypatch):
    orch, calls = _seed_spy_orch(conditioning=False)
    sentinel_bundle = {"transferable_insights": ["B"], "evidence_post_ids": []}
    orch._seeding_phase.load_cross_task_seed_bundle = lambda **k: sentinel_bundle
    orch._seeding_phase.run(SeedingPhaseInput(generation=1, next_task_pool_size=2))
    assert calls["cross_task_target_conditioning"] is False
    assert calls["cross_task_bundle"] == sentinel_bundle

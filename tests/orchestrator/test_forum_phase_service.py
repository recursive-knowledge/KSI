from unittest.mock import MagicMock, call

from kcsi.models import GenerationConfig, TaskTrace
from kcsi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from kcsi.orchestrator.forum_phase import (
    CrossTaskForumPhaseInput,
    EngineForumPhaseService,
    PerTaskForumPhaseInput,
    _cross_task_coordinator_timeout_sec,
)
from tests.orchestrator_phase_decoupling_guard import functions_referencing_engine
from tests.orchestrator_phase_helpers import cross_task_forum, per_task_forum


def _make_orch() -> GenerationalOrchestrator:
    return GenerationalOrchestrator(
        config=GenerationConfig(num_generations=1, num_agents=1),
        runtime=MagicMock(),
        evaluator=MagicMock(),
        llm=MagicMock(),
        persistence=NoopPersistence(),
    )


def test_engine_forum_phase_service_exposes_forum_methods():
    orch = _make_orch()
    service = EngineForumPhaseService(orch)

    assert callable(service.per_task_forum)
    assert callable(service.cross_task_forum)


def test_engine_forum_phase_service_returns_none_when_skipped():
    orch = _make_orch()
    service = EngineForumPhaseService(orch)
    traces = [TaskTrace(agent_id="agent-0", task_id="t1", generation=1)]

    per_task_result = service.per_task_forum(PerTaskForumPhaseInput(generation=1, traces=traces, next_task_pool_size=4))
    cross_task_result = service.cross_task_forum(CrossTaskForumPhaseInput(generation=1, traces=traces))

    assert per_task_result is None
    assert cross_task_result is None


def test_forum_phase_test_helpers_route_through_phase_service():
    orch = _make_orch()
    phase_service = MagicMock()
    orch._forum_phase_service = phase_service
    traces = [TaskTrace(agent_id="agent-0", task_id="t1", generation=1)]

    per_task_forum(orch, 1, traces, next_task_pool_size=4)
    per_task_forum(orch, 2, traces, next_task_pool_size=5)
    cross_task_forum(orch, generation=3, traces=traces)

    assert phase_service.per_task_forum.call_args_list == [
        call(PerTaskForumPhaseInput(generation=1, traces=traces, next_task_pool_size=4)),
        call(PerTaskForumPhaseInput(generation=2, traces=traces, next_task_pool_size=5)),
    ]
    phase_service.cross_task_forum.assert_called_once_with(CrossTaskForumPhaseInput(generation=3, traces=traces))


def test_shared_container_service_accepts_phase_arguments():
    orch = _make_orch()
    service = EngineForumPhaseService(orch)

    service.run_cross_task_shared_container(
        generation=1,
        debate_agents=[],
        forum_bus=object(),
        cross_task_history=[],
        phase1_by_agent={},
        cross_task_evidence_ids=[],
        expected_agents_set=set(),
        workers=1,
    )

    orch.runtime.run_task.assert_not_called()


def test_forum_body_has_no_engine_access():
    from kcsi.orchestrator import forum_phase

    offenders = functions_referencing_engine(forum_phase.__file__)
    assert offenders <= {"_collaborators"}, offenders


def test_forum_collaborators_is_frozen():
    from dataclasses import FrozenInstanceError

    from kcsi.orchestrator.forum_phase import ForumCollaborators

    c = ForumCollaborators(
        config=object(),
        persistence=object(),
        runtime=object(),
        accumulator=object(),
        knowledge=None,
        memory_store=None,
        agents=[],
        record_phase_failure=lambda *a, **k: None,
        maybe_embed=lambda t: None,
        maybe_embed_batch=lambda ts: [],
        should_count_knowledge_drain_drop=lambda g, e: False,
        non_holdout=lambda tr: tr,
        last_task_by_id={},
    )
    try:
        c.knowledge = object()  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("ForumCollaborators must be frozen")


def _container_timeout_sec(forum_timeout_sec: float) -> float:
    # Mirror of container_host._build_runner_env's CONTAINER_TIMEOUT
    # (the container's hard external kill deadline, pre-milliseconds).
    return max(forum_timeout_sec - 15, 300)


def _in_container_poll_timeout_sec(forum_timeout_sec: float) -> float:
    # Mirror of container_host._maybe_setup_cross_task_r1's poll_timeout_sec.
    return max(30, _container_timeout_sec(forum_timeout_sec) - 5)


def test_coordinator_timeout_below_container_timeout_default():
    # Default cross_task_forum_timeout_sec is 900s. The coordinator must stop
    # waiting strictly before the container's hard external kill deadline so it
    # never blocks past the point where every agent's container is already dead.
    coord = _cross_task_coordinator_timeout_sec(900.0)
    assert coord == 855.0
    assert coord < _container_timeout_sec(900.0)  # 855 < 885


def test_cross_task_timeouts_nest_across_configs():
    # The three timeouts must nest coord < poll <= container for the default,
    # the 300s floor, sub-floor values, and a large custom value, so the
    # in-container graceful R0-only fallback always fires before the external
    # hard-kill and the coordinator never over-waits.
    for forum_timeout in (900.0, 300.0, 100.0, 1800.0):
        coord = _cross_task_coordinator_timeout_sec(forum_timeout)
        poll = _in_container_poll_timeout_sec(forum_timeout)
        container = _container_timeout_sec(forum_timeout)
        assert coord >= 60, (forum_timeout, coord)
        assert coord < poll, (forum_timeout, coord, poll)
        assert poll <= container, (forum_timeout, poll, container)

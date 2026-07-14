from unittest.mock import MagicMock

import pytest

from ksi.errors import AuthenticationFailure
from ksi.models import AgentState, GenerationConfig, TaskSpec, TaskTrace
from ksi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from ksi.orchestrator.execution_phase import (
    EngineExecutionPhaseService,
    ExecutionPhaseInput,
    ExecutionPhaseResult,
)
from ksi.runtime import RuntimeResult
from ksi.tokens import TokenUsage
from tests.orchestrator_phase_decoupling_guard import functions_referencing_engine


def _make_orch(config: GenerationConfig | None = None) -> GenerationalOrchestrator:
    return GenerationalOrchestrator(
        config=config or GenerationConfig(num_generations=1, num_agents=1, no_memory=True),
        runtime=MagicMock(),
        evaluator=MagicMock(),
        llm=MagicMock(),
        persistence=NoopPersistence(),
    )


def test_engine_execution_phase_service_exposes_run():
    orch = _make_orch()
    service = EngineExecutionPhaseService(orch)

    assert callable(service.run)


def test_execution_body_has_no_engine_access():
    from ksi.orchestrator import execution_phase

    offenders = functions_referencing_engine(execution_phase.__file__)
    assert offenders <= {"_collaborators"}, offenders


def test_execution_collaborators_is_frozen():
    from dataclasses import FrozenInstanceError

    from ksi.orchestrator.execution_phase import ExecutionCollaborators

    c = ExecutionCollaborators(
        config=object(),
        runtime=object(),
        persistence=object(),
        evaluator=object(),
        accumulator=object(),
        knowledge=None,
        memory_store=None,
        agents=[],
        best_scores={},
        set_last_task_by_id=lambda v: None,
        safe_on_task_trace=lambda t: None,
        record_task_tokens=lambda t: None,
        persist_task_summary=lambda *a, **k: None,
        persist_task_memory_record=lambda **k: None,
        get_or_build_kt_adapter_memo=lambda **k: None,
        llm_call=lambda **k: None,
        maybe_embed=lambda t: None,
        is_holdout=lambda t: False,
        tag_holdout_meta=lambda *a, **k: None,
        merge_optional_meta=lambda *a, **k: None,
        merge_attempt_meta=lambda *a, **k: None,
        retrieved_distillation_ids=lambda *a, **k: None,
        knowledge_trace_condensed=lambda *a, **k: "",
    )
    try:
        c.config = object()  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("ExecutionCollaborators must be frozen")


def test_run_routes_task_attempts_through_execution_phase_service():
    orch = _make_orch(GenerationConfig(num_generations=1, num_agents=1, no_memory=True))
    task = TaskSpec(id="t1", repo="", prompt="task")
    trace = TaskTrace(
        agent_id="agent-0",
        task_id="t1",
        generation=1,
        native_score=1.0,
    )
    fake_service = MagicMock()
    fake_service.run.return_value = ExecutionPhaseResult(traces=[trace])
    orch._execution_phase = fake_service

    traces = orch.run([task])

    assert traces == [trace]
    fake_service.run.assert_called_once()
    phase_input = fake_service.run.call_args.args[0]
    assert phase_input == ExecutionPhaseInput(
        generation=1,
        tasks=[task],
        assigned_map={"agent-0": ["t1"]},
    )


def test_eval_one_attempt_lives_on_execution_phase_service():
    """The attempt-eval pipeline is now a service method, not an engine method."""
    assert callable(getattr(EngineExecutionPhaseService, "_eval_one_attempt", None))
    assert not hasattr(GenerationalOrchestrator, "_eval_one_attempt")


def test_eval_one_attempt_produces_scored_trace_via_service():
    """Characterization: a successful container run flows through the service's
    ``_eval_one_attempt`` and yields a scored, non-errored TaskTrace."""
    orch = _make_orch(GenerationConfig(num_generations=1, num_agents=1, no_memory=True))
    orch.evaluator.evaluate = MagicMock(return_value={"resolved": True, "native_score": 1.0})
    service = EngineExecutionPhaseService(orch)

    agent = AgentState(id="agent-0", workstream="w")
    task = TaskSpec(id="t1", repo="r", prompt="solve")
    run_result = RuntimeResult(output="done", token_usage=TokenUsage(input_tokens=3, output_tokens=2))

    trace, insight, lessons, extra_tokens = service._eval_one_attempt(agent, task, run_result, None, 1, {"t1": task})

    assert isinstance(trace, TaskTrace)
    assert trace.task_id == "t1"
    assert trace.agent_id == "agent-0"
    assert trace.error is None
    assert trace.native_score == 1.0
    assert trace.model_output == "done"
    # no_memory short-circuits reflection.
    assert insight is None
    assert lessons == []
    assert extra_tokens == 0


def test_eval_one_attempt_stamps_both_persistence_flags(tmp_path):
    """Characterization of the eager-persist side-effect contract: a successful
    attempt sets BOTH ``runtime_meta`` stamp flags that ``_execute_default``'s
    skip-check reads later, and writes a raw transcript to the runtime DB.

    Pins the invariant across the god-method decomposition — the flags must be
    stamped on the SAME trace object the caller receives, at the same points.
    """
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        # no_memory isolates the three persistence side-effects from reflection;
        # the knowledge + runtime DBs stay enabled so all three branches fire.
        no_memory=True,
        knowledge_db_path=str(tmp_path / "k.sqlite"),
        runtime_db_path=str(tmp_path / "r.sqlite"),
        experiment_name="stamp_flags",
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=MagicMock(),
        evaluator=MagicMock(),
        llm=MagicMock(),
        persistence=NoopPersistence(),
    )
    orch.evaluator.evaluate = MagicMock(return_value={"resolved": True, "native_score": 1.0})
    service = EngineExecutionPhaseService(orch)
    try:
        assert orch._knowledge is not None
        assert orch._memory_store is not None
        # Spy on the raw-transcript write without disturbing its behaviour.
        transcript_task_ids: list[str] = []
        _orig_insert = orch._memory_store.insert_raw_transcript

        def _spy(**kwargs):
            transcript_task_ids.append(kwargs["task_id"])
            return _orig_insert(**kwargs)

        orch._memory_store.insert_raw_transcript = _spy  # type: ignore[method-assign]

        agent = AgentState(id="agent-0", workstream="w")
        task = TaskSpec(id="t1", repo="r", prompt="solve")
        run_result = RuntimeResult(
            output="done",
            runtime_meta={"native_session_memory": "full transcript"},
            token_usage=TokenUsage(input_tokens=3, output_tokens=2),
        )

        trace, *_ = service._eval_one_attempt(agent, task, run_result, None, 1, {"t1": task})

        meta = trace.runtime_meta or {}
        assert meta.get("_knowledge_attempt_persisted_early") is True
        assert meta.get("_task_trace_persisted_early") is True
        assert transcript_task_ids == ["t1"]
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_execute_default_skips_task_trace_persist_when_stamped():
    """The ``_execute_default`` skip-check (execution_phase.py:~467) must NOT
    re-fire the audit-sidecar ``_safe_on_task_trace`` when ``_eval_one_attempt``
    already stamped ``_task_trace_persisted_early``. This proves the stamp-flag
    contract that de-duplicates the eager trace write is intact end-to-end."""
    orch = _make_orch(GenerationConfig(num_generations=1, num_agents=1, no_memory=True))
    orch.runtime.run_task.return_value = RuntimeResult(
        output="done",
        token_usage=TokenUsage(input_tokens=3, output_tokens=2),
    )
    orch.evaluator.evaluate = MagicMock(return_value={"resolved": True, "native_score": 1.0})
    # Spy on the sidecar callback captured by ``_collaborators()``.
    orch._safe_on_task_trace = MagicMock()  # type: ignore[method-assign]

    orch.run([TaskSpec(id="t1", repo="r", prompt="solve")])

    # NoopPersistence.on_task_trace succeeds inside _eval_one_attempt, so the
    # stamp is set and the redundant sidecar write is skipped.
    orch._safe_on_task_trace.assert_not_called()


def test_execute_default_persists_task_trace_when_not_stamped():
    """Non-vacuity companion to the skip-check test: when the eager trace
    persist raises (flag NOT stamped), ``_execute_default`` falls back to the
    audit-sidecar ``_safe_on_task_trace``. Confirms the guard keys on the flag,
    not on some unconditional path."""
    orch = _make_orch(GenerationConfig(num_generations=1, num_agents=1, no_memory=True))
    orch.runtime.run_task.return_value = RuntimeResult(
        output="done",
        token_usage=TokenUsage(input_tokens=3, output_tokens=2),
    )
    orch.evaluator.evaluate = MagicMock(return_value={"resolved": True, "native_score": 1.0})
    # Force the eager trace persist inside _eval_one_attempt to fail so the
    # stamp is never set; on_task_status stays functional.
    orch.persistence.on_task_trace = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
    orch._safe_on_task_trace = MagicMock()  # type: ignore[method-assign]

    orch.run([TaskSpec(id="t1", repo="r", prompt="solve")])

    orch._safe_on_task_trace.assert_called_once()


def test_eval_one_attempt_task_trace_auth_failure_is_fatal():
    """The eager task-trace write is best-effort for sidecar outages, but
    AuthenticationFailure must abort loudly."""
    orch = _make_orch(GenerationConfig(num_generations=1, num_agents=1, no_memory=True))
    orch.evaluator.evaluate = MagicMock(return_value={"resolved": True, "native_score": 1.0})
    orch.persistence.on_task_trace = MagicMock(side_effect=AuthenticationFailure("invalid token"))  # type: ignore[method-assign]
    service = EngineExecutionPhaseService(orch)

    agent = AgentState(id="agent-0", workstream="w")
    task = TaskSpec(id="t1", repo="r", prompt="solve")
    run_result = RuntimeResult(output="done", token_usage=TokenUsage(input_tokens=3, output_tokens=2))

    with pytest.raises(AuthenticationFailure, match="invalid token"):
        service._eval_one_attempt(agent, task, run_result, None, 1, {"t1": task})

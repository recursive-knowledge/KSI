"""Characterization tests for the extracted carry-forward / resume phase.

Pins the behavior that the ``EngineResumePhaseService`` must preserve after the
carry-forward cluster is lifted out of ``engine.py``:

* ``split_assignments`` partitions a previously-solved task (recorded in the
  KnowledgeStore above ``solved_threshold``) into ``carried_traces`` and keeps
  it OUT of ``execute_map``.
* ``_make_carried_forward_trace`` stamps the ``carry_forward`` provenance chain
  on the replayed trace identically to the pre-refactor engine body.
"""

from unittest.mock import MagicMock

from conftest import _build_mock_llm

from ksi.models import GenerationConfig, TaskSpec, TaskTrace
from ksi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from ksi.tokens import TokenUsage


def _resume_orch(tmp_path):
    from ksi.memory.knowledge_store import KnowledgeStore

    knowledge_db_path = str(tmp_path / "resume_phase_knowledge.sqlite")
    knowledge = KnowledgeStore(knowledge_db_path, default_experiment="resume-phase-exp")
    try:
        knowledge.record_attempt(
            task_id="arc-task-1",
            agent_id="agent-0",
            generation=1,
            eval_results={"resolved": True, "native_score": 1.0, "task_type": "arc"},
            model_output="solved-grid",
            native_score=1.0,
            experiment="resume-phase-exp",
        )
    finally:
        knowledge.close()

    config = GenerationConfig(
        num_generations=2,
        num_agents=1,
        per_task_forum_rounds=0,
        drop_solved=False,
        solved_threshold=1.0,
        knowledge_db_path=knowledge_db_path,
        experiment_name="resume-phase-exp",
        resume=True,
    )
    config.cross_task_forum_rounds = 0
    config.distill_enabled = False

    orch = GenerationalOrchestrator(
        config=config,
        runtime=MagicMock(),
        evaluator=MagicMock(),
        llm=_build_mock_llm(),
        persistence=NoopPersistence(),
    )
    return orch


def test_split_assignments_partitions_solved_task_into_carried(tmp_path):
    orch = _resume_orch(tmp_path)
    try:
        task = TaskSpec(id="arc-task-1", prompt="Solve ARC task", metadata={"task_source": "arc"})
        execute_map, carried = orch._resume_phase.split_assignments(
            generation=2,
            assigned_map={"agent-0": ["arc-task-1"]},
            task_by_id={"arc-task-1": task},
        )
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()

    assert execute_map == {}
    assert len(carried) == 1
    trace = carried[0]
    assert trace.task_id == "arc-task-1"
    assert trace.generation == 2
    assert trace.native_score == 1.0
    assert trace.model_output == "solved-grid"
    assert trace.runtime_meta["carry_forward"] is True
    assert trace.runtime_meta["carry_forward_source_generation"] == 1


def test_make_carried_forward_trace_stamps_provenance_chain(tmp_path):
    orch = _resume_orch(tmp_path)
    try:
        task = TaskSpec(id="arc-task-1", prompt="Solve ARC task", metadata={"task_source": "arc"})
        source = TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="arc-task-1",
            model_output="solved-grid",
            eval_result={"resolved": True, "native_score": 1.0},
            native_score=1.0,
            tool_trace=[],
            runtime_meta={"history_source": "knowledge_store"},
            token_usage=TokenUsage(),
        )
        carried = orch._resume_phase._make_carried_forward_trace(
            generation=3,
            agent_id="agent-9",
            task=task,
            source_trace=source,
        )
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()

    assert carried.generation == 3
    assert carried.agent_id == "agent-9"
    assert carried.native_score == 1.0
    assert carried.model_output == "solved-grid"
    meta = carried.runtime_meta
    assert meta["status"] == "carry_forward"
    assert meta["carry_forward"] is True
    assert meta["carry_forward_reason"] == "best_score_preserved"
    assert meta["carry_forward_source_generation"] == 1
    assert meta["carry_forward_source_agent_id"] == "agent-0"
    assert meta["carry_forward_source_score"] == 1.0
    assert meta["carry_forward_threshold"] == 1.0

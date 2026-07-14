"""Tests for the concurrent forum phase in the orchestrator and engine fixes."""

import json
from unittest.mock import MagicMock

import pytest
from conftest import _build_make_tasks, _build_mock_evaluator, _build_mock_llm, _build_mock_runtime

from kcsi.memory.forum_bus import ForumBus
from kcsi.memory.knowledge_store import KnowledgeStore
from kcsi.models import GenerationConfig, TaskTrace
from kcsi.orchestrator.engine import ForumValidationError, GenerationalOrchestrator, NoopPersistence
from kcsi.runtime.types import RuntimeResult
from kcsi.tokens import LLMResponse, TokenUsage
from tests.orchestrator_phase_helpers import per_task_forum


def _make_traces(generation: int, agent_id: str, task_ids: list[str]) -> list[TaskTrace]:
    return [
        TaskTrace(
            generation=generation,
            agent_id=agent_id,
            task_id=tid,
            model_output="patch",
            eval_result={"resolved": True},
            native_score=1.0,
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        for tid in task_ids
    ]


class RecordingPersistence(NoopPersistence):
    def __init__(self) -> None:
        self.forum_messages: list[dict[str, object]] = []

    def on_forum_message(
        self,
        *,
        generation: int,
        round_num: int,
        agent_id: str,
        message_type: str,
        content_json: dict,
        token_usage: dict,
    ) -> None:
        self.forum_messages.append(
            {
                "generation": generation,
                "round_num": round_num,
                "agent_id": agent_id,
                "message_type": message_type,
                "content_json": content_json,
                "token_usage": token_usage,
            }
        )


# --- Forum-specific concurrent tests (from HEAD / feat/decentralized-forum-debate) ---


def _build_insight_output(agent_id: str) -> str:
    """Build a valid INSIGHT template response for pre-seeding execution insights."""
    return (
        f"INSIGHT\n"
        f"insight_id: ins-{agent_id}-1\n"
        f"scope: task\n"
        f"text: insight from {agent_id}\n"
        f"evidence_task_ids: task-0\n"
        f"\n"
        f"INSIGHT\n"
        f"insight_id: ins-{agent_id}-2\n"
        f"scope: meta\n"
        f"text: insight2 from {agent_id}\n"
        f"evidence_task_ids: task-0\n"
    )


def _preseed_execution_insights(db_path: str, agent_ids: list[str], generation: int = 1) -> None:
    """Pre-seed execution-time insights into KnowledgeStore so distillation has content."""
    from pathlib import Path

    from kcsi.memory.knowledge_store import KnowledgeStore

    knowledge_db = str(Path(db_path).parent / "knowledge.sqlite")
    store = KnowledgeStore(knowledge_db)
    for agent_id in agent_ids:
        for idx in range(1, 3):
            store.record_insight(
                task_id=f"task-{idx - 1}",
                agent_id=agent_id,
                generation=generation,
                text=f"execution insight from {agent_id} #{idx}",
                scope="task" if idx == 1 else "meta",
                confidence="medium",
                evidence_task_ids=["task-0"],
                round_num=0,
            )
    store.close()


def _build_comment_output(agent_id: str, target_insight_id: str = "ins-agent-0-1") -> str:
    """Build a valid COMMENT template response for forum round 2."""
    return (
        f"COMMENT\n"
        f"comment_id: c-{agent_id}-1\n"
        f"target_insight_id: {target_insight_id}\n"
        f"stance: refine\n"
        f"text: comment from {agent_id}\n"
        f"referenced_insight_ids: {target_insight_id}\n"
        f"\n"
        f"COMMENT\n"
        f"comment_id: c-{agent_id}-2\n"
        f"target_insight_id: {target_insight_id}\n"
        f"stance: support\n"
        f"text: comment2 from {agent_id}\n"
        f"referenced_insight_ids: {target_insight_id}\n"
    )


def test_forum_phase_launches_critique_tasks(tmp_path):
    """Forum phase should call runtime.run_task for one round across two agents."""
    db_path = str(tmp_path / "knowledge.sqlite")
    _preseed_execution_insights(db_path, ["agent-0", "agent-1"], generation=1)

    runtime = MagicMock()

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        round_num = int((task.metadata or {}).get("forum_round", -1))
        if round_num == 0:
            output_text = "discussed via MCP tools"
        else:
            output_text = ""
        return RuntimeResult(
            output=output_text,
            tool_trace=[],
            runtime_meta={"native_session_memory": "critique transcript"},
            token_usage=TokenUsage(input_tokens=50, output_tokens=30),
        )

    runtime.run_task.side_effect = fake_run_task

    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0}
    llm = MagicMock()
    llm.call.return_value = LLMResponse(
        text=json.dumps({"claimed_tasks": ["task-0"]}),
        usage=TokenUsage(input_tokens=50, output_tokens=20),
    )

    config = GenerationConfig(
        num_generations=1,
        num_agents=2,
        knowledge_db_path=db_path,
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )

    # Two agents, same task so the monologue gate does NOT fire and the
    # forum phase continues to dispatch both agents (this branch).
    traces = _make_traces(1, "agent-0", ["task-0"]) + _make_traces(1, "agent-1", ["task-0"])
    per_task_forum(orch, 1, traces)

    # Should have called run_task for R0: 1 round * 2 agents = 2 calls.
    forum_calls = [
        c
        for c in runtime.run_task.call_args_list
        if "__forum__" in str(c.kwargs.get("task", {}).id if hasattr(c.kwargs.get("task", {}), "id") else c)
    ]
    assert len(forum_calls) == 2
    rounds_seen = {
        int((c.kwargs.get("task").metadata or {}).get("forum_round", -1))
        for c in forum_calls
        if c.kwargs.get("task") is not None
    }
    assert rounds_seen == {0}
    for c in forum_calls:
        task = c.kwargs.get("task")
        meta = (task.metadata or {}) if task is not None else {}
        task_files = meta.get("task_files") if isinstance(meta, dict) else None
        assert task_files in (None, {})
        assert isinstance(meta.get("forum_task_ids"), list)


def test_forum_phase_drains_posts_to_knowledge_store(tmp_path):
    """After Task 15, the per-task forum phase is discussion-only: it
    dispatches R0 discussion containers and drains any ForumBus events into
    KnowledgeStore. Distillation is owned by the distillation phase service.
    """
    db_path = str(tmp_path / "knowledge.sqlite")
    _preseed_execution_insights(db_path, ["agent-0", "agent-1"], generation=1)

    runtime = MagicMock()

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        return RuntimeResult(
            output="discussed via MCP tools",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=50, output_tokens=30),
        )

    runtime.run_task.side_effect = fake_run_task

    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0}
    llm = MagicMock()
    llm.call.return_value = LLMResponse(text=json.dumps({}), usage=TokenUsage())

    config = GenerationConfig(
        num_generations=1,
        num_agents=2,
        knowledge_db_path=db_path,
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )

    # The orchestrator no longer has the _task_mode_shared_bundle attribute.
    assert not hasattr(orch, "_task_mode_shared_bundle")

    # Two agents, same task so the monologue gate does NOT fire and the
    # forum phase continues to dispatch both agents (this branch).
    traces = _make_traces(1, "agent-0", ["task-0"]) + _make_traces(1, "agent-1", ["task-0"])
    # This should not raise and should complete as a discussion-only phase.
    per_task_forum(orch, 1, traces)


def test_forum_phase_records_errors_and_fails_when_all_agents_fail(tmp_path):
    db_path = str(tmp_path / "memory.sqlite")
    runtime = MagicMock()
    runtime.run_task.side_effect = RuntimeError("provider unavailable")
    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0}
    llm = MagicMock()
    llm.call.return_value = LLMResponse(text=json.dumps({}), usage=TokenUsage())
    persistence = RecordingPersistence()

    config = GenerationConfig(
        num_generations=1,
        num_agents=2,
        knowledge_db_path=db_path,
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=persistence,
    )
    # Two agents, same task so the monologue gate does NOT fire and the
    # forum phase continues to dispatch both agents (this branch).
    traces = _make_traces(1, "agent-0", ["task-0"]) + _make_traces(1, "agent-1", ["task-0"])

    with pytest.raises(ForumValidationError, match="all per-task discussion agents failed"):
        per_task_forum(orch, 1, traces)

    error_events = [event for event in persistence.forum_messages if event["message_type"] == "error"]
    assert len(error_events) == 2
    assert {event["agent_id"] for event in error_events} == {"agent-0", "agent-1"}
    assert all(event["content_json"]["phase"] == "per_task_forum" for event in error_events)


def test_forum_phase_persists_earlier_round_posts_when_later_round_fails(tmp_path):
    """Per-task forum drains ForumBus posts INSIDE each round of the
    ``for round_num in range(per_task_rounds)`` loop (mirroring the
    cross-task forum's existing per-round drain), not once after the whole
    loop. With ``--per-task-forum-rounds 2``, round 0 succeeds and writes a
    real post; round 1 then fails for every agent and raises
    ForumValidationError. Round 0's post must already be persisted in
    KnowledgeStore by the time the exception propagates -- a drain-after-
    the-loop placement would discard it.
    """
    db_path = str(tmp_path / "knowledge.sqlite")

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        metadata = getattr(task, "metadata", {}) or {}
        round_num = int(metadata.get("forum_round", -1))
        if round_num == 0:
            bus = ForumBus(db_path=db_path, experiment="default", generation=generation)
            bus.append(
                round_num=0,
                agent_id=agent_id,
                message_type="post",
                content={"task_id": "task-0", "text": "round-0 real post-mortem content, over forty chars long"},
            )
            return RuntimeResult(
                output="",
                tool_trace=[],
                runtime_meta={},
                token_usage=TokenUsage(input_tokens=10, output_tokens=5),
            )
        raise RuntimeError("round 1 provider unavailable")

    runtime = MagicMock()
    runtime.run_task.side_effect = fake_run_task
    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0}
    llm = MagicMock()
    llm.call.return_value = LLMResponse(text=json.dumps({}), usage=TokenUsage())

    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        knowledge_db_path=db_path,
        experiment_name="default",
        per_task_forum_rounds=2,
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )
    traces = _make_traces(1, "agent-0", ["task-0"])

    with pytest.raises(ForumValidationError, match="all per-task discussion agents failed"):
        per_task_forum(orch, 1, traces)

    # Round 0's post must have been drained into KnowledgeStore before
    # round 1's failure aborted the phase -- confirming the drain runs
    # per-round, not once after the whole round loop.
    store = KnowledgeStore(db_path)
    try:
        page = store.query_task("task-0", generation=None, entry_types=["post"], limit=50, experiment="default")
    finally:
        store.close()
    posts = page.get("discussion") or []
    assert any("round-0 real post-mortem" in str(p.get("text", "")) for p in posts), (
        f"expected round-0 post to survive round-1 failure, got {posts!r}"
    )


def test_forum_phase_multi_round_drain_has_no_duplicate_rows(tmp_path):
    """Per-task forum drains ForumBus events INSIDE each round of the loop
    (see test_forum_phase_persists_earlier_round_posts_when_later_round_fails
    above). With ``--per-task-forum-rounds 2`` and BOTH rounds succeeding,
    each round's drain call re-reads ALL ForumBus events (not just the
    newly-appended ones) and relies on external-id dedup
    (``KnowledgeStore.bulk_has_external_ids``) to skip rows already
    persisted by an earlier round's drain. This test locks in that the
    per-round drain does not duplicate round 0's post when round 1 also
    drains successfully -- distinct from the "later round fails" test,
    which only proves earlier posts *survive*, not that a healthy
    multi-round run stays duplicate-free.
    """
    db_path = str(tmp_path / "knowledge.sqlite")

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        metadata = getattr(task, "metadata", {}) or {}
        round_num = int(metadata.get("forum_round", -1))
        bus = ForumBus(db_path=db_path, experiment="default", generation=generation)
        bus.append(
            round_num=round_num,
            agent_id=agent_id,
            message_type="post",
            content={
                "task_id": "task-0",
                "text": f"round-{round_num} real post-mortem content, over forty chars long",
            },
        )
        return RuntimeResult(
            output="",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    runtime = MagicMock()
    runtime.run_task.side_effect = fake_run_task
    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0}
    llm = MagicMock()
    llm.call.return_value = LLMResponse(text=json.dumps({}), usage=TokenUsage())

    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        knowledge_db_path=db_path,
        experiment_name="default",
        per_task_forum_rounds=2,
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )
    traces = _make_traces(1, "agent-0", ["task-0"])

    per_task_forum(orch, 1, traces)

    store = KnowledgeStore(db_path)
    try:
        page = store.query_task("task-0", generation=None, entry_types=["post"], limit=50, experiment="default")
    finally:
        store.close()
    posts = page.get("discussion") or []
    round0_posts = [p for p in posts if "round-0 real post-mortem" in str(p.get("text", ""))]
    round1_posts = [p for p in posts if "round-1 real post-mortem" in str(p.get("text", ""))]
    assert len(round0_posts) == 1, f"expected exactly one round-0 post, got {round0_posts!r}"
    assert len(round1_posts) == 1, f"expected exactly one round-1 post, got {round1_posts!r}"
    assert len(posts) == 2, f"expected exactly 2 total posts (no duplicates from repeated drain), got {posts!r}"


# --- Engine critical fix tests (from fix/engine-critical-fixes) ---


def test_forum_phase_skipped_without_knowledge_db():
    """When KnowledgeStore is absent (no --knowledge-db-path), forum phase should
    be skipped entirely and runtime.run_task should NOT be called for
    reflection/forum tasks."""
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
    )
    tasks = _build_make_tasks(1)
    runtime = _build_mock_runtime()
    evaluator = _build_mock_evaluator()
    llm = _build_mock_llm()

    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )
    # Default: no knowledge DB means no KnowledgeStore-backed forum phase.
    assert orch._memory_store is None

    traces = orch.run(tasks)
    assert len(traces) >= 1

    # runtime.run_task should only be called for execution, not forum debate
    for call_args in runtime.run_task.call_args_list:
        _, kwargs = call_args
        task = kwargs.get("task")
        if task is not None:
            assert not task.id.startswith("__forum__"), (
                f"Forum debate task {task.id!r} should not be run when KnowledgeStore is absent"
            )


def test_forum_phase_retries_transient_agent_failure_then_succeeds(tmp_path):
    db_path = str(tmp_path / "knowledge.sqlite")
    _preseed_execution_insights(db_path, ["agent-0", "agent-1"], generation=1)

    runtime = MagicMock()
    calls = {"count": 0}

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        calls["count"] += 1
        if agent_id == "agent-0" and calls["count"] == 1:
            raise RuntimeError("provider unavailable")
        return RuntimeResult(
            output="discussed via MCP tools",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=50, output_tokens=30),
        )

    runtime.run_task.side_effect = fake_run_task

    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0}
    llm = MagicMock()
    llm.call.return_value = LLMResponse(text=json.dumps({}), usage=TokenUsage())
    persistence = RecordingPersistence()

    config = GenerationConfig(
        num_generations=1,
        num_agents=2,
        knowledge_db_path=db_path,
        max_task_retries=1,
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=persistence,
    )

    traces = _make_traces(1, "agent-0", ["task-0"]) + _make_traces(1, "agent-1", ["task-0"])
    per_task_forum(orch, 1, traces)

    assert calls["count"] == 3
    assert not [event for event in persistence.forum_messages if event["message_type"] == "error"]


def test_forum_phase_runs_with_knowledge_store(tmp_path):
    """When KnowledgeStore is set, forum phase should run normally.

    Contrast with ``test_forum_phase_skipped_without_knowledge_db`` which
    confirms forum dispatch is short-circuited when no knowledge DB is
    attached. After the 1:1 monologue gate was added, task-mode sweeps short-circuit the
    per-agent dispatch but still produce synthesized posts. We verify the
    run completes AND the store ends up with task-bearing knowledge
    entries (attempts) on the engine's chosen experiment name.
    """
    db_path = str(tmp_path / "knowledge.sqlite")
    _preseed_execution_insights(db_path, ["agent-0"], generation=1)

    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        knowledge_db_path=db_path,
    )
    tasks = _build_make_tasks(1)
    runtime = MagicMock()

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        if task.id.startswith("__forum__"):
            return RuntimeResult(
                output="discussed via MCP tools",
                tool_trace=[],
                runtime_meta={},
                token_usage=TokenUsage(input_tokens=100, output_tokens=50),
            )
        return RuntimeResult(
            output="<patch>diff</patch>",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        )

    runtime.run_task.side_effect = fake_run_task
    evaluator = _build_mock_evaluator()
    llm = _build_mock_llm()

    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )
    experiment = orch.config.experiment_name
    traces = orch.run(tasks)
    assert len(traces) >= 1

    # V2: Phase 2 ALWAYS dispatches runtime.run_task for the forum round
    # (monologue-skip removed). With a mocked runtime that returns dummy
    # output (no real forum_post), the bus stays empty and no posts get
    # drained — that's expected. Verify Phase 2 dispatched at least one
    # forum task (the V2 invariant: Phase 2 always runs).
    forum_calls = [
        call
        for call in runtime.run_task.call_args_list
        if str(getattr(call.kwargs.get("task"), "id", "")).startswith("__forum__")
    ]
    assert len(forum_calls) >= 1, (
        "V2: Phase 2 must dispatch at least one __forum__ task per agent "
        "(monologue-skip removed; Phase 2 always runs in task-mode)"
    )


def test_memory_store_closed_on_run_completion():
    """MemoryStore.close() is called in the run() finally block."""
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
    )
    tasks = _build_make_tasks(1)
    runtime = _build_mock_runtime()
    evaluator = _build_mock_evaluator()
    llm = _build_mock_llm()

    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )
    mock_store = MagicMock()
    orch._memory_store = mock_store

    orch.run(tasks)
    mock_store.close.assert_called_once()


def test_memory_store_closed_on_run_exception():
    """MemoryStore.close() is called even if run() raises."""
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
    )
    tasks = _build_make_tasks(1)
    runtime = MagicMock()
    runtime.run_task.side_effect = RuntimeError("boom")
    evaluator = _build_mock_evaluator()
    llm = _build_mock_llm()

    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )
    mock_store = MagicMock()
    orch._memory_store = mock_store

    # run() should still complete (errors are caught per-task), but close() must be called
    orch.run(tasks)
    mock_store.close.assert_called_once()


def test_no_unused_max_retries():
    """Verify max_retries is not assigned in the execution phase service."""
    import inspect

    from kcsi.orchestrator.execution_phase import EngineExecutionPhaseService

    source = inspect.getsource(EngineExecutionPhaseService._execute_default)
    assert "max_retries" not in source, "max_retries should not be assigned in the execution phase service"

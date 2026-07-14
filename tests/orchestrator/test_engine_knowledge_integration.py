"""Tests for KnowledgeStore dual-write integration in the orchestrator engine."""

import sqlite3
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kcsi.models import GenerationConfig, Insight, TaskTrace
from kcsi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence

# ---------------------------------------------------------------------------
# 1. _knowledge is None when knowledge DB is not set
# ---------------------------------------------------------------------------


def test_knowledge_store_none_without_knowledge_db(mock_runtime, mock_evaluator, mock_llm):
    """When knowledge_db_path is empty, _knowledge should be None."""
    config = GenerationConfig(num_generations=1, num_agents=1, per_task_forum_rounds=0)
    assert config.knowledge_db_path == ""

    orch = GenerationalOrchestrator(
        config=config,
        runtime=mock_runtime(),
        evaluator=mock_evaluator(),
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )
    assert orch._knowledge is None


def test_no_memory_keeps_knowledge_store_but_disables_agent_memory(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """--no-memory keeps authoritative KnowledgeStore; it only disables agent-facing memory phases."""
    db_path = str(tmp_path / "test_exp_knowledge.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="test_exp",
        no_memory=True,
    )

    orch = GenerationalOrchestrator(
        config=config,
        runtime=mock_runtime(),
        evaluator=mock_evaluator(),
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )
    try:
        assert orch._memory_store is None
        assert orch._knowledge is not None
        assert Path(db_path).exists()
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()


def test_resume_uses_knowledge_store_not_stale_runtime_sidecar(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """When both DBs exist, KnowledgeStore is authoritative for resume state."""
    from kcsi.memory.knowledge_store import KnowledgeStore
    from kcsi.memory.store import MemoryStore

    knowledge_db_path = str(tmp_path / "resume_knowledge.sqlite")
    runtime_db_path = str(tmp_path / "resume_runtime.sqlite")

    knowledge = KnowledgeStore(knowledge_db_path, default_experiment="resume_exp")
    runtime = MemoryStore(runtime_db_path, default_experiment="resume_exp")
    try:
        knowledge.record_attempt(
            task_id="task-from-knowledge",
            agent_id="agent-knowledge",
            generation=2,
            eval_results={"native_score": 1.0, "resolved": True},
            native_score=1.0,
            experiment="resume_exp",
        )
        runtime.upsert_task_memory_record(
            experiment="resume_exp",
            generation=1,
            agent_id="agent-runtime",
            task_id="task-from-runtime",
            eval_results={"native_score": 0.25, "resolved": False},
            final_model_output="stale sidecar output",
            full_memory_trace="stale sidecar trace",
            full_memory_trace_condensed="stale sidecar condensed",
            task_specific_insights=[],
        )
    finally:
        knowledge.close()
        runtime.close()

    config = GenerationConfig(
        num_generations=3,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=knowledge_db_path,
        runtime_db_path=runtime_db_path,
        experiment_name="resume_exp",
        resume=True,
    )

    orch = GenerationalOrchestrator(
        config=config,
        runtime=mock_runtime(),
        evaluator=mock_evaluator(),
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )
    try:
        assert orch._best_scores == {"task-from-knowledge": 1.0}
        assert orch._start_generation == 3
        assert orch._resume_latest_generation == 2
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_no_memory_failed_attempt_persists_to_knowledge_store(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """Failed traces still advance authoritative attempt state under --no-memory."""
    db_path = str(tmp_path / "failed_no_memory_knowledge.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="failed_no_memory",
        no_memory=True,
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=mock_runtime(),
        evaluator=mock_evaluator(),
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )
    try:
        trace = TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="failed-task",
            model_output=None,
            eval_result={},
            native_score=None,
            error="container crashed",
        )

        assert orch._execution_phase._persist_knowledge_attempt_early(trace) is True
        page = orch._knowledge.query_task("failed-task", experiment="failed_no_memory")
        assert len(page["attempts"]) == 1
        content = page["attempts"][0]["content"]
        assert content["eval_results"]["status"] == "error"
        assert content["eval_results"]["error"] == "container crashed"
        assert orch._knowledge.get_latest_task_generation(experiment="failed_no_memory") == 1
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()


# ---------------------------------------------------------------------------
# 2. _knowledge is initialized when knowledge_db_path is set
# ---------------------------------------------------------------------------


def test_knowledge_store_initialized_with_knowledge_db(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """When knowledge_db_path is set, _knowledge should be a KnowledgeStore instance."""
    # Use the canonical "<stem>_knowledge.sqlite" name for the authoritative
    # per-experiment knowledge DB.
    db_path = str(tmp_path / "test_exp_knowledge.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="test_exp",
    )

    orch = GenerationalOrchestrator(
        config=config,
        runtime=mock_runtime(),
        evaluator=mock_evaluator(),
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )
    try:
        assert orch._knowledge is not None
        # Per-experiment authoritative knowledge DB exists at the configured path.
        knowledge_db = tmp_path / "test_exp_knowledge.sqlite"
        assert knowledge_db.exists()
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_require_vector_fails_when_knowledge_vec_degrades(
    tmp_path,
    mock_runtime,
    mock_evaluator,
    mock_llm,
):
    """--require-vector must fail if sqlite-vec init degraded inside KnowledgeStore."""

    class FakeKnowledgeStore:
        _vec_enabled = False

        def __init__(self, *args, **kwargs):
            pass

        def record_vector_status(self, **kwargs):
            return 1

    db_path = str(tmp_path / "require_vec_knowledge.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="require_vec_test",
    )
    config.require_vector = True

    with patch("kcsi.memory.knowledge_store.KnowledgeStore", FakeKnowledgeStore):
        with pytest.raises(RuntimeError, match="Vector memory is required"):
            GenerationalOrchestrator(
                config=config,
                runtime=mock_runtime(),
                evaluator=mock_evaluator(),
                llm=mock_llm(),
                persistence=NoopPersistence(),
            )


# ---------------------------------------------------------------------------
# 3. _persist_task_memory_record writes to both stores
# ---------------------------------------------------------------------------


def test_persist_task_memory_record_dual_write(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """_persist_task_memory_record should write to both _memory_store and _knowledge."""
    knowledge_db_path = str(tmp_path / "mem_knowledge.sqlite")
    runtime_db_path = str(tmp_path / "mem_runtime.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=knowledge_db_path,
        runtime_db_path=runtime_db_path,
        experiment_name="dual_write_test",
    )

    orch = GenerationalOrchestrator(
        config=config,
        runtime=mock_runtime(),
        evaluator=mock_evaluator(),
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )

    try:
        assert orch._knowledge is not None
        assert orch._memory_store is not None

        # Build a trace
        trace = TaskTrace(
            generation=1,
            agent_id="agent-1",
            task_id="task-42",
            model_output="Fixed the bug by changing line 42",
            eval_result={"resolved": True, "native_score": 1.0},
            native_score=1.0,
            runtime_meta={
                "native_session_memory": "full session trace",
                "injected_memory_md": "# MEMORY Seed\n\nRemember cache invalidation.",
            },
        )

        insight = Insight(
            id="ins-1",
            text="Always check cache invalidation",
            author_agent_id="agent-1",
            generation=1,
            workstream="django-orm",
            source_task_id="task-42",
            confidence="high",
        )

        orch._persist_task_memory_record(
            trace=trace,
            insight=insight,
            lessons=["lesson one"],
        )

        # Verify the knowledge store has the attempt record
        rows = orch._knowledge.query_task("task-42", experiment="dual_write_test")
        assert len(rows["attempts"]) == 1
        attempt = rows["attempts"][0]
        assert attempt["agent_id"] == "agent-1"
        assert attempt["score"] == 1.0

        mem_rows = orch._memory_store._execute(
            """
            SELECT injected_memory_md
            FROM task_memory_records
            WHERE task_id = ? AND generation = ? AND agent_id = ?
            """,
            ("task-42", 1, "agent-1"),
            fetchall=True,
        )
        assert len(mem_rows) == 1
        assert "Remember cache invalidation" in mem_rows[0]["injected_memory_md"]
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_persist_task_memory_record_supersedes_early_write_placeholder(
    tmp_path,
    mock_runtime,
    mock_evaluator,
    mock_llm,
):
    """Late memory persistence must not duplicate an early KnowledgeStore
    attempt row, AND must overwrite its placeholder content with the real
    insight/reflection/lessons computed after the early write.

    Regression test for issue #1039 (trace-mining.md #1): before the fix,
    the late write was skipped entirely whenever the early resume-safety
    write had already run (~99% of the time in production), permanently
    stranding every attempt row with ``insights=[]``, empty ``reflection``,
    and a ``trace_condensed`` ending in the literal placeholder
    ``"(pending reflection)"``.
    """
    db_path = str(tmp_path / "mem.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="early_write_test",
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=mock_runtime(),
        evaluator=mock_evaluator(),
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )

    try:
        assert orch._knowledge is not None
        trace = TaskTrace(
            generation=1,
            agent_id="agent-1",
            task_id="task-42",
            model_output="Fixed the bug by changing line 42",
            eval_result={"resolved": True, "native_score": 1.0},
            native_score=1.0,
            runtime_meta={
                "native_session_memory": "full session trace",
                "_knowledge_attempt_persisted_early": True,
            },
        )

        assert orch._execution_phase._persist_knowledge_attempt_early(trace) is True

        early_rows = orch._knowledge.query_task("task-42", experiment="early_write_test")
        assert len(early_rows["attempts"]) == 1
        assert early_rows["attempts"][0]["content"]["insights"] == []
        assert "(pending reflection)" in early_rows["attempts"][0]["content"]["trace_condensed"]

        insight = Insight(
            id="ins-1",
            text="Use QuerySet.filter instead of get",
            author_agent_id="agent-1",
            generation=1,
            workstream="django-orm",
            source_task_id="task-42",
            confidence="high",
        )
        orch._persist_task_memory_record(
            trace=trace,
            insight=insight,
            lessons=["Never trust a cold cache"],
        )

        rows = orch._knowledge.query_task("task-42", experiment="early_write_test")
        # Superseded in place — still exactly one attempt row, not two.
        assert len(rows["attempts"]) == 1
        attempt = rows["attempts"][0]
        assert attempt["agent_id"] == "agent-1"
        assert attempt["score"] == 1.0
        assert "(pending reflection)" not in attempt["content"]["trace_condensed"]
        assert "QuerySet" in attempt["content"]["trace_condensed"]
        assert "Use QuerySet.filter instead of get" in attempt["content"]["insights"]
        assert "Never trust a cold cache" in attempt["content"]["insights"]
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_persist_knowledge_attempt_early_writes_embedding_when_embedder_ready(
    tmp_path,
    mock_runtime,
    mock_evaluator,
    mock_llm,
):
    db_path = str(tmp_path / "mem.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="embedding_attempt_test",
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=mock_runtime(),
        evaluator=mock_evaluator(),
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )

    class FakeEmbedder:
        is_ready = True

        def embed(self, text):
            assert "Approach:" in text
            return [0.25] * 768

    try:
        assert orch._knowledge is not None
        orch._embedder = FakeEmbedder()
        trace = TaskTrace(
            generation=1,
            agent_id="agent-1",
            task_id="task-embed",
            model_output="final answer",
            eval_result={"native_score": 0.5},
            native_score=0.5,
        )
        wrapped = MagicMock(wraps=orch._knowledge)
        orch._knowledge = wrapped

        assert orch._execution_phase._persist_knowledge_attempt_early(trace) is True

        kwargs = wrapped.record_attempt.call_args.kwargs
        assert kwargs["embedding"] == [0.25] * 768
        assert orch._vector_embedding_count == 1
    finally:
        real_knowledge = getattr(orch._knowledge, "_mock_wraps", orch._knowledge)
        if real_knowledge is not None:
            real_knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


# ---------------------------------------------------------------------------
# 4. KnowledgeStore failure is fatal
# ---------------------------------------------------------------------------


def test_knowledge_store_failure_is_fatal(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """If authoritative KnowledgeStore raises, the engine should fail the write."""
    db_path = str(tmp_path / "mem.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="crash_test",
    )

    orch = GenerationalOrchestrator(
        config=config,
        runtime=mock_runtime(),
        evaluator=mock_evaluator(),
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )

    try:
        assert orch._knowledge is not None

        # Replace _knowledge with a mock that always raises
        broken_knowledge = MagicMock()
        broken_knowledge.record_attempt.side_effect = RuntimeError("DB write boom")
        orch._knowledge = broken_knowledge

        trace = TaskTrace(
            generation=1,
            agent_id="agent-1",
            task_id="task-99",
            model_output="output",
            eval_result={"resolved": False, "native_score": 0.0},
            native_score=0.0,
            runtime_meta={},
        )

        with pytest.raises(RuntimeError, match="authoritative KnowledgeStore record_attempt failed"):
            orch._persist_task_memory_record(
                trace=trace,
                insight=None,
                lessons=None,
            )

        # Verify the broken knowledge store was called
        broken_knowledge.record_attempt.assert_called_once()
    finally:
        if orch._memory_store is not None:
            orch._memory_store.close()


# ---------------------------------------------------------------------------
# 5. _knowledge store is closed during cleanup
# ---------------------------------------------------------------------------


def test_knowledge_store_closed_on_run(tmp_path, make_tasks, mock_runtime, mock_evaluator, mock_llm):
    """_knowledge.close() should be called after run() completes."""
    db_path = str(tmp_path / "mem.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="close_test",
    )
    tasks = make_tasks(1)
    runtime = mock_runtime()
    evaluator = mock_evaluator()
    llm = mock_llm()

    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )

    assert orch._knowledge is not None
    # Replace with mock to track close()
    real_knowledge = orch._knowledge
    mock_knowledge = MagicMock(wraps=real_knowledge)
    orch._knowledge = mock_knowledge

    orch.run(tasks)

    # close() should have been called
    mock_knowledge.close.assert_called_once()


# ---------------------------------------------------------------------------
# 6. KnowledgeStore init failure is fatal
# ---------------------------------------------------------------------------


def test_knowledge_store_init_failure_is_fatal(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """If KnowledgeStore.__init__ raises, engine startup fails because knowledge is authoritative."""
    db_path = str(tmp_path / "mem.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="init_fail",
    )

    # Patch at the source module so the lazy import inside __init__ picks it up
    with patch(
        "kcsi.memory.knowledge_store.KnowledgeStore.__init__",
        side_effect=RuntimeError("init boom"),
    ):
        with pytest.raises(RuntimeError, match="KnowledgeStore initialization failed"):
            GenerationalOrchestrator(
                config=config,
                runtime=mock_runtime(),
                evaluator=mock_evaluator(),
                llm=mock_llm(),
                persistence=NoopPersistence(),
            )


# ---------------------------------------------------------------------------
# 7. Legacy MemoryStore write failure must NOT skip the KnowledgeStore write.
#    Regression test for the ARC-AGI-2 g10 run where every generation lost
#    one attempt to both stores because a `database is locked` warning on the
#    legacy store short-circuited the dual-write (the KnowledgeStore call was
#    nested inside the legacy-write try/except).
# ---------------------------------------------------------------------------


def test_legacy_store_failure_does_not_skip_knowledge_write(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """KnowledgeStore.record_attempt must run even when MemoryStore raises."""
    knowledge_db_path = str(tmp_path / "mem_knowledge.sqlite")
    runtime_db_path = str(tmp_path / "mem_runtime.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=knowledge_db_path,
        runtime_db_path=runtime_db_path,
        experiment_name="isolation_test",
    )

    orch = GenerationalOrchestrator(
        config=config,
        runtime=mock_runtime(),
        evaluator=mock_evaluator(),
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )

    real_memory_store = orch._memory_store
    try:
        assert real_memory_store is not None
        assert orch._knowledge is not None

        # Simulate the `sqlite3.OperationalError: database is locked` that was
        # observed in arc2_haiku_swarm_g10.log — the legacy store's write
        # blows up, and previously that prevented the KnowledgeStore write.
        broken_memory_store = MagicMock(wraps=real_memory_store)
        broken_memory_store.upsert_task_memory_record.side_effect = sqlite3.OperationalError("database is locked")
        orch._memory_store = broken_memory_store

        trace = TaskTrace(
            generation=1,
            agent_id="agent-7",
            task_id="task-locked",
            model_output="attempted fix",
            eval_result={"resolved": False, "native_score": 0.0},
            native_score=0.0,
            runtime_meta={"native_session_memory": "session trace"},
        )

        # Must not raise — both failure paths are logged, not propagated.
        orch._persist_task_memory_record(
            trace=trace,
            insight=None,
            lessons=["keep trying"],
        )

        # Legacy store was called and raised.
        broken_memory_store.upsert_task_memory_record.assert_called_once()

        # KnowledgeStore must still have recorded the attempt.
        rows = orch._knowledge.query_task("task-locked", experiment="isolation_test")
        assert len(rows["attempts"]) == 1, (
            "KnowledgeStore dual-write was skipped when MemoryStore raised — "
            "the two stores must have independent failure domains."
        )
        assert rows["attempts"][0]["agent_id"] == "agent-7"
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if real_memory_store is not None:
            real_memory_store.close()


# ---------------------------------------------------------------------------
# 8. Early KnowledgeStore write failure must NOT corrupt the eval trace.
#    Regression test for the early-write site in `_eval_stage`: previously
#    this raised RuntimeError, which the eval-future `as_completed` handler
#    converted into a `eval_stage_exception` failed trace — silently turning
#    a successfully-solved task into a 0/1 in solve-rate accounting. The
#    late-path `_persist_task_memory_record` will retry persistence; losing
#    one resume-cursor write is preferable to phantom failures.
# ---------------------------------------------------------------------------


def test_early_knowledge_failure_preserves_solve_rate(tmp_path, make_tasks, mock_runtime, mock_evaluator, mock_llm):
    """A transient early-write KnowledgeStore failure must not produce a phantom failed trace."""
    db_path = str(tmp_path / "early_fail_knowledge.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="early_fail_test",
        no_memory=True,  # disable late-path memory persistence to isolate the early-write code path
    )
    tasks = make_tasks(1)

    orch = GenerationalOrchestrator(
        config=config,
        runtime=mock_runtime(),
        evaluator=mock_evaluator(),
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )

    try:
        assert orch._knowledge is not None

        # Simulate `sqlite3.OperationalError: database is locked` on the
        # early-write call. Critically, the late-path `record_attempt`
        # (called from `_persist_task_memory_record`) must still succeed —
        # otherwise resume cursors break for real.
        original_early = orch._execution_phase._persist_knowledge_attempt_early
        call_log: list[str] = []

        def boom_then_record(trace, agent=None):
            call_log.append(trace.task_id)
            raise sqlite3.OperationalError("database is locked")

        orch._execution_phase._persist_knowledge_attempt_early = boom_then_record  # type: ignore[assignment]

        # Run the generation. Previously the inner raise propagated out of
        # _eval_stage, was caught by the as_completed handler at the call
        # site, and produced an `eval_stage_exception` trace with
        # native_score=None. After the fix, the failure is logged and the
        # trace retains its real evaluation result.
        orch.run(tasks)

        # The early-write site was reached.
        assert call_log == ["task-0"], f"early write was not invoked exactly once; call_log={call_log}"

        # Solve-rate accounting must reflect the successful container run,
        # not a phantom failure: the mock evaluator returns resolved=True,
        # native_score=1.0. With the bug present, the trace would carry
        # error="eval_stage_exception: ..." and native_score=None.
        assert orch._best_scores.get("task-0") == 1.0, (
            f"_best_scores corrupted by early-write failure; got {orch._best_scores!r}"
        )
    finally:
        # Restore so the engine's run() finally-block close() targets a real
        # method (not strictly necessary, but tidy).
        orch._execution_phase._persist_knowledge_attempt_early = original_early  # type: ignore[assignment]


def test_early_knowledge_failure_late_path_still_records_attempt(
    tmp_path, make_tasks, mock_runtime, mock_evaluator, mock_llm
):
    """Sibling to the soft-fail test: with memory enabled, the late
    `_persist_task_memory_record` path must successfully record the attempt
    even when the early-write path failed transiently.

    The first test (`..._preserves_solve_rate`) uses `no_memory=True` to
    isolate the early-write code path — it confirms the trace isn't
    corrupted but doesn't exercise the late retry. This test confirms the
    late path picks up the slack: solve-rate stays correct AND the
    KnowledgeStore actually has the attempt row at the end of the run.
    """
    db_path = str(tmp_path / "late_path_knowledge.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="late_path_test",
    )
    tasks = make_tasks(1)

    orch = GenerationalOrchestrator(
        config=config,
        runtime=mock_runtime(),
        evaluator=mock_evaluator(),
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )

    try:
        assert orch._knowledge is not None
        original_early = orch._execution_phase._persist_knowledge_attempt_early

        def early_fails(trace, agent=None):
            raise sqlite3.OperationalError("database is locked")

        orch._execution_phase._persist_knowledge_attempt_early = early_fails  # type: ignore[assignment]

        orch.run(tasks)

        # Solve rate preserved (early-write soft-fail).
        assert orch._best_scores.get("task-0") == 1.0
    finally:
        orch._execution_phase._persist_knowledge_attempt_early = original_early  # type: ignore[assignment]

    # Re-open the DB in a fresh handle (engine.run()'s finally-block closed
    # the original) and verify the late path actually recorded the attempt.
    from kcsi.memory.knowledge_store import KnowledgeStore

    verify_store = KnowledgeStore(db_path, default_experiment="late_path_test")
    try:
        rows = verify_store.query_task("task-0", experiment="late_path_test")
        assert len(rows["attempts"]) == 1, (
            f"late path failed to record attempt despite early-path soft-fail; rows={rows!r}"
        )
        assert rows["attempts"][0]["agent_id"] == "agent-0"
    finally:
        verify_store.close()


# ---------------------------------------------------------------------------
# _maybe_embed_batch — robust under embedder failures
# ---------------------------------------------------------------------------


class TestMaybeEmbedBatch:
    """Unit tests for the orchestrator's batch-embed helper used by the
    forum-drain perf path. Per-batch failure must not abort the drain.
    """

    def _make_orch(self, mock_runtime, mock_evaluator, mock_llm):
        config = GenerationConfig(
            num_generations=1,
            num_agents=1,
            per_task_forum_rounds=0,
        )
        return GenerationalOrchestrator(
            config=config,
            runtime=mock_runtime(),
            evaluator=mock_evaluator(),
            llm=mock_llm(),
            persistence=NoopPersistence(),
        )

    def test_single_call_for_all_texts(self, mock_runtime, mock_evaluator, mock_llm):
        """All texts go through one embed_batch invocation (no chunking yet —
        chunking is a future-work item once a real campaign exposes the
        memory/latency concern at very large drain sizes).
        """
        orch = self._make_orch(mock_runtime, mock_evaluator, mock_llm)

        call_count = {"n": 0}
        seen_lengths: list[int] = []

        class FakeEmbedder:
            def is_ready(self):
                return True

            def embed_batch(self, texts):
                call_count["n"] += 1
                seen_lengths.append(len(texts))
                return [[0.1] * 8 for _ in texts]

        orch._embedder = FakeEmbedder()
        texts = [f"text-{i}" for i in range(50)]
        result = orch._maybe_embed_batch(texts)

        assert len(result) == 50
        assert all(v is not None for v in result)
        assert call_count["n"] == 1
        assert seen_lengths == [50]

    def test_embed_failure_falls_through_with_none(self, mock_runtime, mock_evaluator, mock_llm):
        """If embed_batch raises, all texts get None — drain proceeds."""
        orch = self._make_orch(mock_runtime, mock_evaluator, mock_llm)

        class FlakyEmbedder:
            def is_ready(self):
                return True

            def embed_batch(self, texts):
                raise RuntimeError("embedder offline")

        orch._embedder = FlakyEmbedder()
        result = orch._maybe_embed_batch([f"t-{i}" for i in range(5)])
        assert result == [None] * 5

    def test_empty_input_returns_empty(self, mock_runtime, mock_evaluator, mock_llm):
        orch = self._make_orch(mock_runtime, mock_evaluator, mock_llm)
        assert orch._maybe_embed_batch([]) == []

    def test_embedder_not_ready_returns_all_none(self, mock_runtime, mock_evaluator, mock_llm):
        orch = self._make_orch(mock_runtime, mock_evaluator, mock_llm)

        class NotReadyEmbedder:
            def is_ready(self):
                return False

            def embed_batch(self, texts):
                raise AssertionError("must not be called when not ready")

        orch._embedder = NotReadyEmbedder()
        result = orch._maybe_embed_batch(["a", "b", "c"])
        assert result == [None, None, None]

    def test_whitespace_texts_skipped_without_embedder_call(self, mock_runtime, mock_evaluator, mock_llm):
        orch = self._make_orch(mock_runtime, mock_evaluator, mock_llm)

        seen_texts: list[str] = []

        class FakeEmbedder:
            def is_ready(self):
                return True

            def embed_batch(self, texts):
                seen_texts.extend(texts)
                return [[0.1] * 8 for _ in texts]

        orch._embedder = FakeEmbedder()
        result = orch._maybe_embed_batch(["real", "", "  ", "also real"])

        assert result[0] is not None
        assert result[1] is None
        assert result[2] is None
        assert result[3] is not None
        assert seen_texts == ["real", "also real"], (
            f"whitespace texts should be filtered before embedder call; got {seen_texts}"
        )


# ---------------------------------------------------------------------------
# _vector_embedding_count / _vector_skipped_count — lock-guarded (issue #981)
# ---------------------------------------------------------------------------


class _CountingLock:
    """Context-manager spy that wraps a real ``threading.Lock`` and counts

    every ``with lock:`` acquisition. A timing-based lost-update race test
    is not a reliable regression signal for plain ``int += 1`` on an
    instance attribute under CPython's GIL (verified empirically: 16
    threads x 200k unguarded increments each still landed on the exact
    expected total in this environment) — call-site instrumentation is.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.acquire_count = 0

    def __enter__(self) -> "_CountingLock":
        self._lock.acquire()
        self.acquire_count += 1  # only ever mutated while self._lock is held
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._lock.release()


class TestVectorCountsLockGuarded:
    """``_maybe_embed``/``_maybe_embed_batch`` run from concurrent eval-worker

    threads (execution_phase.py's eval ThreadPoolExecutor); an unguarded
    ``self._vector_*_count += 1`` can lose increments under real thread
    interleaving. Substitute a spy lock for ``_vector_counts_lock`` and
    assert it's acquired exactly once per increment across every call
    shape (embed success/failure/not-ready, batch success/failure/
    not-ready/partial-whitespace) — this fails deterministically if a
    future edit removes a ``with self._vector_counts_lock:`` guard.
    """

    def _make_orch(self, mock_runtime, mock_evaluator, mock_llm):
        config = GenerationConfig(num_generations=1, num_agents=1, per_task_forum_rounds=0)
        return GenerationalOrchestrator(
            config=config,
            runtime=mock_runtime(),
            evaluator=mock_evaluator(),
            llm=mock_llm(),
            persistence=NoopPersistence(),
        )

    def test_maybe_embed_every_branch_is_lock_guarded(self, mock_runtime, mock_evaluator, mock_llm):
        orch = self._make_orch(mock_runtime, mock_evaluator, mock_llm)
        spy_lock = _CountingLock()
        orch._vector_counts_lock = spy_lock

        class FlakyEmbedder:
            is_ready = True

            def embed(self, text):
                if text == "boom":
                    raise RuntimeError("embedder offline")
                return [0.1] * 8

        orch._embedder = FlakyEmbedder()
        assert orch._maybe_embed("ok") is not None  # embed success branch
        assert orch._maybe_embed("boom") is None  # embed exception branch
        orch._embedder = None
        assert orch._maybe_embed("ok") is None  # embedder-not-ready branch

        assert orch._vector_embedding_count == 1
        assert orch._vector_skipped_count == 2
        assert spy_lock.acquire_count == 3

    def test_maybe_embed_batch_every_branch_is_lock_guarded(self, mock_runtime, mock_evaluator, mock_llm):
        orch = self._make_orch(mock_runtime, mock_evaluator, mock_llm)
        spy_lock = _CountingLock()
        orch._vector_counts_lock = spy_lock

        class FakeEmbedder:
            is_ready = True

            def embed_batch(self, texts):
                return [[0.1] * 8 for _ in texts]

        orch._embedder = FakeEmbedder()
        orch._maybe_embed_batch(["a", "", "b"])  # 2 embedded + 1 whitespace-skipped

        class FlakyEmbedder:
            is_ready = True

            def embed_batch(self, texts):
                raise RuntimeError("embedder offline")

        orch._embedder = FlakyEmbedder()
        orch._maybe_embed_batch(["c", "d"])  # both skipped on batch failure

        orch._embedder = None
        orch._maybe_embed_batch(["e"])  # embedder-not-ready branch

        assert orch._vector_embedding_count == 2
        assert orch._vector_skipped_count == 4
        # One acquisition per increment call, regardless of how many texts
        # that call's `+= len(...)` covers: whitespace-skip(1) + batch-embed(1)
        # + batch-failure-skip(1) + not-ready-skip(1) = 4.
        assert spy_lock.acquire_count == 4

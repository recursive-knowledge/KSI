"""Integration tests: orchestrator memory — transcripts, summaries, and embeddings."""

import json
from unittest.mock import MagicMock

from ksi.memory.store import MemoryStore
from ksi.models import GenerationConfig, TaskSpec
from ksi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from ksi.runtime.types import RuntimeResult
from ksi.tokens import LLMResponse, TokenUsage

# ---------------------------------------------------------------------------
# Transcript persistence tests (from PR #67 / #70)
# ---------------------------------------------------------------------------


def test_transcripts_persisted_to_runtime_db(tmp_path):
    """After a run, raw transcripts should exist in the runtime audit DB."""
    db_path = str(tmp_path / "runtime.sqlite")

    runtime = MagicMock()
    runtime.run_task.return_value = RuntimeResult(
        output="<patch>diff</patch>",
        tool_trace=[],
        runtime_meta={"native_session_memory": "session transcript content here", "session_scope": "task"},
        token_usage=TokenUsage(input_tokens=100, output_tokens=50),
    )

    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0}

    llm = MagicMock()
    llm.call.return_value = LLMResponse(
        text=json.dumps({"claimed_tasks": ["task-0"]}),
        usage=TokenUsage(input_tokens=50, output_tokens=20),
    )

    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        runtime_db_path=db_path,
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )
    tasks = [TaskSpec(id="task-0", repo="r", prompt="Fix bug")]
    orch.run(tasks)

    store = MemoryStore(db_path)
    result = store.get_raw_transcript(task_id="task-0")
    assert result is not None
    assert "session transcript" in result["content"]
    store.close()


def test_transcript_available_before_next_task(tmp_path):
    """Transcripts are persisted to the docs DB after execution.

    With task-mode (one agent per task), tasks run in parallel so we
    verify post-run that all transcripts land in the DB rather than
    checking sequential visibility.
    """
    db_path = str(tmp_path / "knowledge.sqlite")

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        return RuntimeResult(
            output="patch",
            tool_trace=[],
            runtime_meta={"native_session_memory": f"transcript for {task.id}"},
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    runtime = MagicMock()
    runtime.run_task.return_value = RuntimeResult(
        output="patch",
        tool_trace=[],
        runtime_meta={"native_session_memory": "transcript content"},
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
    )

    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0}

    llm = MagicMock()
    llm.call.return_value = LLMResponse(
        text=json.dumps({"buckets": []}),
        usage=TokenUsage(input_tokens=50, output_tokens=20),
    )

    config = GenerationConfig(
        num_generations=1,
        num_agents=3,
        per_task_forum_rounds=0,
        runtime_db_path=db_path,
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )
    tasks = [
        TaskSpec(id="task-0", repo="r", prompt="Fix bug 0"),
        TaskSpec(id="task-1", repo="r", prompt="Fix bug 1"),
        TaskSpec(id="task-2", repo="r", prompt="Fix bug 2"),
    ]
    orch.run(tasks)

    # Verify all 3 transcripts were persisted

    store = MemoryStore(db_path)
    rows = store._execute(
        "SELECT task_id FROM raw_transcripts ORDER BY id",
        fetchall=True,
    )
    store.close()
    db_task_ids = {r["task_id"] for r in rows}
    assert db_task_ids == {"task-0", "task-1", "task-2"}


# ---------------------------------------------------------------------------
# Summary persistence tests (from PR #74)
# ---------------------------------------------------------------------------


def test_task_summaries_persisted_after_execution(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """Task summaries should land in the runtime audit DB after execution."""
    db_path = str(tmp_path / "knowledge.sqlite")

    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        runtime_db_path=db_path,
        experiment_name="test_embed",
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=mock_runtime(),
        evaluator=mock_evaluator(),
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )

    tasks = [TaskSpec(id="task-0", repo="r", prompt="Fix bug")]
    orch.run(tasks)

    # Verify summary was stored in the runtime DB (via the compat view)

    store = MemoryStore(db_path)
    row = store._execute("SELECT * FROM task_summaries WHERE task_id = ?", ("task-0",), fetchone=True)
    assert row is not None
    assert row["outcome"] == "resolved"
    assert row["experiment"] == "test_embed"
    store.close()


def test_transcript_persisted_after_execution(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """Raw transcripts should be persisted in knowledge DB after task execution."""
    db_path = str(tmp_path / "knowledge.sqlite")

    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        runtime_db_path=db_path,
        experiment_name="test_transcript",
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=mock_runtime(),
        evaluator=mock_evaluator(),
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )

    # Disable embedder to isolate transcript test
    orch._embedder = None

    tasks = [TaskSpec(id="task-0", repo="r", prompt="Fix bug")]
    orch.run(tasks)

    # Verify transcript was stored

    store = MemoryStore(db_path)
    row = store.get_raw_transcript(task_id="task-0")
    assert row is not None
    assert row["content"] == "transcript content here"
    assert row["experiment"] == "test_transcript"
    store.close()


def test_no_memory_when_db_path_empty(mock_runtime, mock_evaluator, mock_llm):
    """When knowledge_db_path is empty, no memory store or embedder should be created."""
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=mock_runtime(),
        evaluator=mock_evaluator(),
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )
    assert orch._memory_store is None
    assert orch._embedder is None


def test_no_memory_with_runtime_db_keeps_runtime_store(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """no_memory disables semantic memory, not the runtime audit sidecar."""
    db_path = str(tmp_path / "runtime.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        runtime_db_path=db_path,
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
        assert orch._memory_store is not None
        assert orch._knowledge is None
        assert orch._embedder is None
    finally:
        if orch._memory_store is not None:
            orch._memory_store.close()


# ---------------------------------------------------------------------------
# Data quality tests: _extract_key_files and transcript fallback
# ---------------------------------------------------------------------------


def test_extract_key_files_from_diff_output():
    """_extract_key_files should parse file paths from diff --git headers."""
    model_output = (
        "Here is my patch:\n"
        "diff --git a/src/utils.py b/src/utils.py\n"
        "--- a/src/utils.py\n"
        "+++ b/src/utils.py\n"
        "@@ -1,3 +1,4 @@\n"
        "+import os\n"
        "diff --git a/tests/test_utils.py b/tests/test_utils.py\n"
        "--- a/tests/test_utils.py\n"
        "+++ b/tests/test_utils.py\n"
    )
    files = GenerationalOrchestrator._extract_key_files(model_output)
    assert files == ["src/utils.py", "tests/test_utils.py"]


def test_extract_key_files_deduplicates():
    """Duplicate diff headers for the same file should be deduplicated."""
    model_output = "diff --git a/foo.py b/foo.py\nsome content\ndiff --git a/foo.py b/foo.py\nmore content\n"
    files = GenerationalOrchestrator._extract_key_files(model_output)
    assert files == ["foo.py"]


def test_extract_key_files_no_diffs():
    """No diff headers should return an empty list."""
    assert GenerationalOrchestrator._extract_key_files("Just a plain answer") == []
    assert GenerationalOrchestrator._extract_key_files("") == []


def test_transcript_fallback_to_model_output(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """When native_session_memory is empty, model_output should be stored as transcript."""
    db_path = str(tmp_path / "knowledge.sqlite")

    # Runtime returns empty native_session_memory (the common case for containers)
    runtime = mock_runtime(native_session_memory="")

    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        runtime_db_path=db_path,
        experiment_name="test_fallback",
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=mock_evaluator(),
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )
    orch._embedder = None

    tasks = [TaskSpec(id="task-0", repo="r", prompt="Fix bug")]
    orch.run(tasks)

    store = MemoryStore(db_path)
    row = store.get_raw_transcript(task_id="task-0")
    assert row is not None
    # Should contain the model_output text, not the empty native_session_memory
    assert "Fixed the bug" in row["content"]
    assert row["experiment"] == "test_fallback"
    store.close()


def test_key_files_stored_in_summary(tmp_path, mock_evaluator, mock_llm):
    """Task summaries should contain key_files extracted from diff output."""
    db_path = str(tmp_path / "knowledge.sqlite")

    # Runtime returns output with diff headers
    rt = MagicMock()
    rt.run_task.return_value = RuntimeResult(
        output="diff --git a/main.py b/main.py\n--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-old\n+new",
        tool_trace=[],
        runtime_meta={"native_session_memory": ""},
        token_usage=TokenUsage(input_tokens=100, output_tokens=50),
    )

    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        runtime_db_path=db_path,
        experiment_name="test_keyfiles",
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=rt,
        evaluator=mock_evaluator(),
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.1] * 768
    orch._embedder = mock_embedder

    tasks = [TaskSpec(id="task-0", repo="r", prompt="Fix bug")]
    orch.run(tasks)

    store = MemoryStore(db_path)
    row = store._execute(
        "SELECT key_files FROM task_summaries WHERE task_id = ?",
        ("task-0",),
        fetchone=True,
    )
    assert row is not None
    key_files = json.loads(row["key_files"])
    assert "main.py" in key_files
    store.close()


def test_lessons_extracted_and_stored(tmp_path, mock_runtime, mock_evaluator):
    """Lessons extracted via LLM should be stored in task summaries."""
    db_path = str(tmp_path / "knowledge.sqlite")

    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        runtime_db_path=db_path,
        experiment_name="test_lessons",
    )

    # LLM mock that returns lessons for lesson_extraction calls
    llm = MagicMock()

    def llm_side_effect(system, user, **kwargs):
        if "lessons" in system.lower() or "lessons" in user.lower():
            return json.dumps(
                {
                    "lessons": [
                        "Root cause was missing null check in queryset filter",
                        "Always test with empty querysets",
                    ]
                }
            ), TokenUsage(input_tokens=50, output_tokens=30)
        elif "claimed_tasks" in user.lower() or "workstream" in user.lower():
            return json.dumps({"claimed_tasks": ["task-0"]}), TokenUsage(input_tokens=50, output_tokens=20)
        else:
            return json.dumps(
                {
                    "insights": [{"text": "Found bug", "workstream": "debugging", "confidence": "high"}],
                    "workstream_claim": "debugging",
                    "proposed_workstreams": ["debugging"],
                }
            ), TokenUsage(input_tokens=100, output_tokens=50)

    def _llm_response_side_effect(system, user, **kwargs):
        text, usage = llm_side_effect(system, user, **kwargs)
        return LLMResponse(text=text, usage=usage)

    llm.call.side_effect = _llm_response_side_effect

    orch = GenerationalOrchestrator(
        config=config,
        runtime=mock_runtime(),
        evaluator=mock_evaluator(),
        llm=llm,
        persistence=NoopPersistence(),
    )
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.1] * 768
    orch._embedder = mock_embedder

    tasks = [TaskSpec(id="task-0", repo="r", prompt="Fix bug")]
    orch.run(tasks)

    store = MemoryStore(db_path)
    row = store._execute(
        "SELECT lessons FROM task_summaries WHERE task_id = ?",
        ("task-0",),
        fetchone=True,
    )
    assert row is not None
    lessons = json.loads(row["lessons"])
    assert len(lessons) >= 1
    assert any("null check" in l or "queryset" in l for l in lessons)
    store.close()


def test_lessons_fallback_on_llm_failure(tmp_path, mock_runtime, mock_evaluator):
    """If lesson extraction LLM call fails, lessons should be empty list (not crash)."""
    db_path = str(tmp_path / "knowledge.sqlite")

    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        runtime_db_path=db_path,
        experiment_name="test_lessons_fail",
    )

    # LLM that raises on lesson extraction
    llm = MagicMock()

    def llm_side_effect(system, user, **kwargs):
        if "lessons" in system.lower():
            raise RuntimeError("LLM crashed")
        elif "claimed_tasks" in user.lower() or "workstream" in user.lower():
            return json.dumps({"claimed_tasks": ["task-0"]}), TokenUsage(input_tokens=50, output_tokens=20)
        else:
            return json.dumps(
                {
                    "insights": [{"text": "Found bug", "workstream": "debugging", "confidence": "high"}],
                    "workstream_claim": "debugging",
                }
            ), TokenUsage(input_tokens=100, output_tokens=50)

    def _llm_response_side_effect(system, user, **kwargs):
        text, usage = llm_side_effect(system, user, **kwargs)
        return LLMResponse(text=text, usage=usage)

    llm.call.side_effect = _llm_response_side_effect

    orch = GenerationalOrchestrator(
        config=config,
        runtime=mock_runtime(),
        evaluator=mock_evaluator(),
        llm=llm,
        persistence=NoopPersistence(),
    )
    orch._embedder = None

    tasks = [TaskSpec(id="task-0", repo="r", prompt="Fix bug")]
    traces = orch.run(tasks)

    # Should complete without crashing
    assert len(traces) >= 1

    store = MemoryStore(db_path)
    row = store._execute(
        "SELECT lessons FROM task_summaries WHERE task_id = ?",
        ("task-0",),
        fetchone=True,
    )
    assert row is not None
    lessons = json.loads(row["lessons"])
    assert lessons == []
    store.close()


def test_tool_call_counts_logged(tmp_path, caplog, mock_evaluator, mock_llm):
    """Tool call counts from runtime_meta should be logged for observability."""
    import logging

    db_path = str(tmp_path / "knowledge.sqlite")

    rt = MagicMock()
    rt.run_task.return_value = RuntimeResult(
        output="Fixed the bug",
        tool_trace=[
            {"idx": 1, "type": "tool_call", "tool_name": "Read"},
            {"idx": 2, "type": "tool_call", "tool_name": "Edit"},
            {"idx": 3, "type": "tool_call", "tool_name": "mcp__memory__memory_search"},
            {"idx": 4, "type": "tool_call", "tool_name": "mcp__memory__memory_acknowledge"},
        ],
        runtime_meta={
            "native_session_memory": "",
            "tool_call_counts": {
                "Read": 5,
                "Edit": 2,
                "Bash": 3,
                "mcp__memory__memory_search": 1,
                "mcp__memory__memory_acknowledge": 1,
            },
        },
        token_usage=TokenUsage(input_tokens=100, output_tokens=50),
    )

    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        runtime_db_path=db_path,
        experiment_name="test_tool_counts",
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=rt,
        evaluator=mock_evaluator(),
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )
    orch._embedder = None

    tasks = [TaskSpec(id="task-0", repo="r", prompt="Fix bug")]
    with caplog.at_level(logging.INFO, logger="ksi.orchestrator.engine"):
        orch.run(tasks)

    # Should log memory tool counts
    assert any("memory_tools=" in rec.message and "memory_search" in rec.message for rec in caplog.records), (
        f"Expected memory tool log entry, got: {[r.message for r in caplog.records]}"
    )


def test_embedding_failure_does_not_crash_execution(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """If embedding fails for a task, execution should continue gracefully."""
    db_path = str(tmp_path / "knowledge.sqlite")

    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        runtime_db_path=db_path,
        experiment_name="test_fail",
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=mock_runtime(),
        evaluator=mock_evaluator(),
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )

    # Mock embedder that raises an exception
    mock_embedder = MagicMock()
    mock_embedder.embed.side_effect = RuntimeError("Embedding model crashed")
    orch._embedder = mock_embedder

    tasks = [TaskSpec(id="task-0", repo="r", prompt="Fix bug")]
    traces = orch.run(tasks)

    # Execution should still complete despite embedding failure
    assert len(traces) >= 1
    assert traces[0].task_id == "task-0"

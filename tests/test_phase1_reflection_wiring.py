"""Unit tests for the Phase-1 reflection wiring across engine and host.

These tests do not spin up a container — they verify the in-process
plumbing: engine -> KcsiContainerExecutor.evaluator handshake,
``record_attempt`` accepts and persists ``reflection``, and
``_persist_knowledge_attempt_early`` extracts the reflection text from
``runtime_meta.phase1_reflection``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kcsi.memory.knowledge_store import KnowledgeStore
from kcsi.models import GenerationConfig, TaskTrace
from kcsi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from kcsi.runtime.container_host import KcsiContainerExecutor


def test_executor_phase1_reflection_defaults_off():
    ex = KcsiContainerExecutor(command=["fake"], working_dir=".")
    assert ex.phase1_reflection_enabled is False
    assert ex.evaluator is None


def test_executor_accepts_phase1_flag_and_evaluator():
    ev = MagicMock()
    ex = KcsiContainerExecutor(
        command=["fake"],
        working_dir=".",
        phase1_reflection_enabled=True,
        evaluator=ev,
    )
    assert ex.phase1_reflection_enabled is True
    assert ex.evaluator is ev


def test_engine_sets_executor_evaluator(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """When the engine is constructed, it forwards its evaluator to the runtime
    so the runtime's BarrierWatcher (when phase1_reflection is on) can call
    evaluator.evaluate() between task completion and the reflection turn."""

    # Use a real KcsiContainerExecutor so we can assert evaluator was set.

    executor = KcsiContainerExecutor(
        command=["fake-runner"],
        working_dir=str(tmp_path),
        phase1_reflection_enabled=True,
    )
    assert executor.evaluator is None

    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=str(tmp_path / "k.sqlite"),
        experiment_name="phase1_wire_test",
    )
    ev = mock_evaluator()
    orch = GenerationalOrchestrator(
        config=config,
        runtime=executor,
        evaluator=ev,
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )
    try:
        assert executor.evaluator is ev
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_record_attempt_accepts_reflection_kwarg(tmp_path):
    db_path = str(tmp_path / "k.sqlite")
    store = KnowledgeStore(db_path, default_experiment="t")
    try:
        entry_id = store.record_attempt(
            task_id="t1",
            agent_id="a1",
            generation=1,
            eval_results={"native_score": 0.5},
            model_output="output",
            trace_condensed="trace",
            insights=[],
            native_score=0.5,
            experiment="t",
            reflection="My load-bearing assumption was X. I would change Y. Predicted outcome: Z.",
        )
        assert entry_id > 0

        page = store.query_task("t1", generation=None, entry_types=["attempt"], limit=10)
        attempts = page.get("attempts") or []
        assert len(attempts) == 1
        content = attempts[0]["content"]
        assert "reflection" in content
        assert content["reflection"].startswith("My load-bearing assumption")
    finally:
        store.close()


def test_record_attempt_reflection_default_empty(tmp_path):
    """When reflection is not passed, the field is still present (empty string)
    so distill consumers don't have to handle missing keys."""
    db_path = str(tmp_path / "k.sqlite")
    store = KnowledgeStore(db_path, default_experiment="t")
    try:
        store.record_attempt(
            task_id="t1",
            agent_id="a1",
            generation=1,
            eval_results={},
            model_output="o",
            insights=[],
            experiment="t",
        )
        page = store.query_task("t1", generation=None, entry_types=["attempt"], limit=10)
        attempts = page.get("attempts") or []
        assert attempts[0]["content"]["reflection"] == ""
    finally:
        store.close()


def test_persist_knowledge_attempt_early_extracts_phase1_reflection(
    tmp_path,
    mock_runtime,
    mock_evaluator,
    mock_llm,
):
    """The engine should pull `runtime_meta.phase1_reflection` and pass it as
    `reflection` to `record_attempt`, so the persisted attempt's content carries
    the reflection text."""
    db_path = str(tmp_path / "mem.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="phase1_persist_test",
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
            task_id="task-r",
            model_output="final answer",
            eval_result={"native_score": 0.5},
            native_score=0.5,
            runtime_meta={
                "phase1_reflection": "Assumption: input is sorted. Change: drop the sort. Outcome: 2x speedup.",
            },
        )
        wrapped = MagicMock(wraps=orch._knowledge)
        orch._knowledge = wrapped

        assert orch._execution_phase._persist_knowledge_attempt_early(trace) is True
        kwargs = wrapped.record_attempt.call_args.kwargs
        assert kwargs["reflection"].startswith("Assumption: input is sorted.")

        # And the persisted row carries the reflection in content.
        # Use the wrapped instance's underlying store to read back.
        real = wrapped._extract_mock_name and wrapped
        # MagicMock(wraps=...) forwards .query_task to the real store
        page = wrapped.query_task("task-r", generation=None, entry_types=["attempt"], limit=10)
        attempts = page.get("attempts") or []
        assert len(attempts) == 1
        assert "Assumption: input is sorted" in attempts[0]["content"]["reflection"]
    finally:
        real_knowledge = getattr(orch._knowledge, "_mock_wraps", orch._knowledge)
        if real_knowledge is not None:
            real_knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_persist_knowledge_attempt_early_no_phase1_reflection_writes_empty(
    tmp_path,
    mock_runtime,
    mock_evaluator,
    mock_llm,
):
    """No phase1_reflection in runtime_meta => empty reflection string (not None)."""
    db_path = str(tmp_path / "mem.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="phase1_persist_noop",
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
            agent_id="a",
            task_id="t",
            model_output="x",
            eval_result={},
            native_score=0.0,
            runtime_meta={},
        )
        wrapped = MagicMock(wraps=orch._knowledge)
        orch._knowledge = wrapped
        orch._execution_phase._persist_knowledge_attempt_early(trace)
        kwargs = wrapped.record_attempt.call_args.kwargs
        assert kwargs["reflection"] == ""
    finally:
        real_knowledge = getattr(orch._knowledge, "_mock_wraps", orch._knowledge)
        if real_knowledge is not None:
            real_knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_persist_knowledge_attempt_early_persists_sanitized_attempt_1_eval_result(
    tmp_path,
    mock_runtime,
    mock_evaluator,
    mock_llm,
):
    """The polyglot test-feedback retry loop (Aider --tries protocol) surfaces
    `runtime_meta.polyglot_test_feedback_meta.attempt_1_eval_summary`; the
    engine must persist a sanitized subset as
    `attempt_meta["attempt_1_eval_result"]` for research visibility (Aider's
    `pass_rate_1` analog), stripping any raw stdout/stderr the summary might
    carry (defense-in-depth even though the TS-side summarizer never emits
    those keys)."""
    db_path = str(tmp_path / "mem.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="polyglot_test_feedback_persist_test",
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
            task_id="task-tf",
            model_output="final answer",
            eval_result={"native_score": 1.0, "resolved": True},
            native_score=1.0,
            runtime_meta={
                "task_source": "polyglot",
                "polyglot_test_feedback_meta": {
                    "enabled": True,
                    "rounds_used": 1,
                    "attempt_1_eval_summary": {
                        "native_score": 0.0,
                        "resolved": False,
                        "status": "ok",
                        "test_exit_code": 1,
                        "test_stdout_tail": "should never be persisted",
                    },
                    "captured": True,
                },
            },
        )
        wrapped = MagicMock(wraps=orch._knowledge)
        orch._knowledge = wrapped

        assert orch._execution_phase._persist_knowledge_attempt_early(trace) is True
        kwargs = wrapped.record_attempt.call_args.kwargs
        attempt_1_eval_result = kwargs["attempt_meta"]["attempt_1_eval_result"]
        assert attempt_1_eval_result == {
            "native_score": 0.0,
            "resolved": False,
            "status": "ok",
            "test_exit_code": 1,
        }
        assert "test_stdout_tail" not in attempt_1_eval_result
        assert "test_stderr_tail" not in attempt_1_eval_result
    finally:
        real_knowledge = getattr(orch._knowledge, "_mock_wraps", orch._knowledge)
        if real_knowledge is not None:
            real_knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_persist_knowledge_attempt_early_no_test_feedback_meta_omits_attempt_1_eval_result(
    tmp_path,
    mock_runtime,
    mock_evaluator,
    mock_llm,
):
    """No `polyglot_test_feedback_meta` in runtime_meta => no
    `attempt_1_eval_result` key at all (not None, not present)."""
    db_path = str(tmp_path / "mem.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="polyglot_test_feedback_absent_test",
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
            agent_id="a",
            task_id="t",
            model_output="x",
            eval_result={},
            native_score=0.0,
            runtime_meta={},
        )
        wrapped = MagicMock(wraps=orch._knowledge)
        orch._knowledge = wrapped
        orch._execution_phase._persist_knowledge_attempt_early(trace)
        kwargs = wrapped.record_attempt.call_args.kwargs
        attempt_meta = kwargs["attempt_meta"] or {}
        assert "attempt_1_eval_result" not in attempt_meta
    finally:
        real_knowledge = getattr(orch._knowledge, "_mock_wraps", orch._knowledge)
        if real_knowledge is not None:
            real_knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_persist_task_memory_record_fallback_persists_sanitized_attempt_1_eval_result(
    tmp_path,
    mock_runtime,
    mock_evaluator,
    mock_llm,
):
    """The engine's ``_persist_task_memory_record`` fallback (used only when
    ``_persist_knowledge_attempt_early`` did not already write the attempt,
    e.g. resume/carry-forward paths) must mirror the same
    ``attempt_1_eval_result`` merge as the early-persist path — same
    sanitized keys, same raw-tail stripping."""
    db_path = str(tmp_path / "mem.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="polyglot_test_feedback_fallback_persist_test",
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
            task_id="task-tf-fallback",
            model_output="final answer",
            eval_result={"native_score": 1.0, "resolved": True},
            native_score=1.0,
            runtime_meta={
                "task_source": "polyglot",
                "polyglot_test_feedback_meta": {
                    "enabled": True,
                    "rounds_used": 1,
                    "attempt_1_eval_summary": {
                        "native_score": 0.0,
                        "resolved": False,
                        "status": "ok",
                        "test_exit_code": 1,
                        "test_stdout_tail": "should never be persisted",
                    },
                    "captured": True,
                },
            },
        )
        wrapped = MagicMock(wraps=orch._knowledge)
        orch._knowledge = wrapped

        orch._persist_task_memory_record(trace=trace, insight=None, lessons=None)

        kwargs = wrapped.record_attempt.call_args.kwargs
        attempt_1_eval_result = kwargs["attempt_meta"]["attempt_1_eval_result"]
        assert attempt_1_eval_result == {
            "native_score": 0.0,
            "resolved": False,
            "status": "ok",
            "test_exit_code": 1,
        }
        assert "test_stdout_tail" not in attempt_1_eval_result
        assert "test_stderr_tail" not in attempt_1_eval_result
    finally:
        real_knowledge = getattr(orch._knowledge, "_mock_wraps", orch._knowledge)
        if real_knowledge is not None:
            real_knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_eval_stage_reuses_phase1_eval_result_to_avoid_double_evaluate(
    tmp_path,
    mock_llm,
):
    """Critical-2 regression guard: when the runtime emits
    ``runtime_meta.phase1_reflection_enabled=True`` AND
    ``runtime_meta.phase1_eval_result=<dict>``, the engine's _eval_stage
    must reuse the cached eval_result instead of calling
    ``evaluator.evaluate()`` again. Doubles the docker subprocess cost
    on polyglot/swebench_pro otherwise.
    """
    from kcsi.models import TaskSpec
    from kcsi.runtime.types import RuntimeResult
    from kcsi.tokens import TokenUsage

    db_path = str(tmp_path / "mem.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="phase1_no_double_eval",
    )

    # Runtime emits a RuntimeResult whose runtime_meta carries the
    # watcher's already-computed eval result.
    runtime = MagicMock()
    runtime.evaluator = None  # the engine will set it; assignment must succeed
    runtime.run_task.return_value = RuntimeResult(
        output="agent's final output",
        tool_trace=[],
        runtime_meta={
            "native_session_memory": "x",
            "session_scope": "task",
            "phase1_reflection_enabled": True,
            "phase1_eval_result": {
                "resolved": True,
                "native_score": 0.75,
                "task_type": "polyglot",
            },
            "phase1_reflection": "Assumption: ... Change: ... Predicted: ...",
        },
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
    )

    evaluator = MagicMock()
    # If the engine wrongly calls evaluate() again, this side-effect
    # would dominate the persisted score.
    evaluator.evaluate.return_value = {
        "resolved": False,
        "native_score": 0.0,
        "task_type": "polyglot",
    }

    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )
    try:
        tasks = [TaskSpec(id="t-no-double", repo="r", prompt="solve")]
        orch.run(tasks)
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()

    # The engine must NOT have called evaluate(): the watcher already did.
    assert evaluator.evaluate.call_count == 0, (
        f"Engine evaluator.evaluate was called {evaluator.evaluate.call_count} "
        "times despite runtime_meta carrying phase1_eval_result — this is the "
        "double-evaluate bug from PR #573 review."
    )


def test_eval_stage_recomputes_phase1_empty_patch_with_final_tool_trace(tmp_path, mock_llm):
    """Phase-1 has no final tool trace, so an empty-patch cache is stale."""
    from kcsi.models import TaskSpec
    from kcsi.runtime.types import RuntimeResult
    from kcsi.tokens import TokenUsage

    runtime = MagicMock()
    runtime.evaluator = None
    runtime.run_task.return_value = RuntimeResult(
        output="final",
        tool_trace=[{"tool_name": "apply_patch"}],
        runtime_meta={"phase1_reflection_enabled": True, "phase1_eval_result": {"swebench_status": "no_patch"}},
        token_usage=TokenUsage(),
    )
    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"swebench_status": "capture_failed"}
    orch = GenerationalOrchestrator(
        config=GenerationConfig(
            num_generations=1,
            num_agents=1,
            per_task_forum_rounds=0,
            knowledge_db_path=str(tmp_path / "mem.sqlite"),
            experiment_name="phase1_recompute",
        ),
        runtime=runtime,
        evaluator=evaluator,
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )
    try:
        orch.run([TaskSpec(id="t-recompute", repo="r", prompt="solve")])
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()
    evaluator.evaluate.assert_called_once()


def test_eval_stage_falls_through_when_phase1_disabled(
    tmp_path,
    mock_runtime,
    mock_llm,
):
    """When the flag is OFF, runtime_meta has no phase1_reflection_enabled
    marker, and the engine must call evaluate() exactly once (the
    pre-Phase-1 baseline behavior)."""
    from kcsi.models import TaskSpec

    db_path = str(tmp_path / "mem.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="phase1_disabled_path",
    )

    evaluator = MagicMock()
    evaluator.evaluate.return_value = {
        "resolved": True,
        "native_score": 1.0,
        "task_type": "polyglot",
    }

    orch = GenerationalOrchestrator(
        config=config,
        runtime=mock_runtime(),
        evaluator=evaluator,
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )
    try:
        tasks = [TaskSpec(id="t-baseline", repo="r", prompt="solve")]
        orch.run(tasks)
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()

    assert evaluator.evaluate.call_count == 1, (
        f"Engine evaluator.evaluate should be called exactly once when the "
        f"flag is off, got {evaluator.evaluate.call_count}."
    )


def test_eval_stage_reuses_polyglot_test_feedback_eval_result_to_avoid_double_evaluate(
    tmp_path,
    mock_llm,
):
    """PR #1032 deep-review regression guard (tests.md): mirrors
    test_eval_stage_reuses_phase1_eval_result_to_avoid_double_evaluate for
    the polyglot test-feedback retry loop's cache-reuse branch
    (execution_phase.py's `elif polyglot_test_feedback_reuse_eligible`
    gate). Without this test, a regression there (e.g. a typo in the key
    name) would silently double the Docker-run cost on every polyglot task
    using the retry loop, with nothing catching it."""
    from kcsi.models import TaskSpec
    from kcsi.runtime.types import RuntimeResult
    from kcsi.tokens import TokenUsage

    db_path = str(tmp_path / "mem.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="polyglot_tf_no_double_eval",
    )

    runtime = MagicMock()
    runtime.evaluator = None
    runtime.run_task.return_value = RuntimeResult(
        output="agent's final output",
        tool_trace=[],
        runtime_meta={
            "native_session_memory": "x",
            "session_scope": "task",
            "polyglot_test_feedback_reuse_eligible": True,
            "polyglot_test_feedback_eval_result": {
                "resolved": True,
                "native_score": 1.0,
                "task_type": "polyglot",
            },
        },
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
    )

    evaluator = MagicMock()
    # If the engine wrongly calls evaluate() again, this side-effect would
    # dominate the persisted score.
    evaluator.evaluate.return_value = {
        "resolved": False,
        "native_score": 0.0,
        "task_type": "polyglot",
    }

    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )
    try:
        tasks = [TaskSpec(id="t-polyglot-tf-no-double", repo="r", prompt="solve")]
        orch.run(tasks)
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()

    assert evaluator.evaluate.call_count == 0, (
        f"Engine evaluator.evaluate was called {evaluator.evaluate.call_count} "
        "times despite runtime_meta carrying polyglot_test_feedback_eval_result "
        "— the cache-reuse branch in execution_phase.py is broken."
    )


def test_eval_stage_falls_through_when_polyglot_test_feedback_not_reuse_eligible(
    tmp_path,
    mock_llm,
):
    """When runtime_meta lacks polyglot_test_feedback_reuse_eligible (e.g.
    the retry loop exhausted its tries after an edit turn -- the cached
    eval scored the pre-turn state), the engine must fall through to its
    own evaluate() call exactly once rather than silently reusing a stale
    result."""
    from kcsi.models import TaskSpec
    from kcsi.runtime.types import RuntimeResult
    from kcsi.tokens import TokenUsage

    db_path = str(tmp_path / "mem.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="polyglot_tf_fallthrough",
    )

    runtime = MagicMock()
    runtime.evaluator = None
    runtime.run_task.return_value = RuntimeResult(
        output="agent's final output",
        tool_trace=[],
        runtime_meta={
            "native_session_memory": "x",
            "session_scope": "task",
            "polyglot_test_feedback_meta": {"final_eval_matches_output": False},
        },
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
    )

    evaluator = MagicMock()
    evaluator.evaluate.return_value = {
        "resolved": True,
        "native_score": 1.0,
        "task_type": "polyglot",
    }

    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )
    try:
        tasks = [TaskSpec(id="t-polyglot-tf-fallthrough", repo="r", prompt="solve")]
        orch.run(tasks)
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()

    assert evaluator.evaluate.call_count == 1, (
        f"Engine evaluator.evaluate should be called exactly once when the "
        f"cached eval is not reuse-eligible, got {evaluator.evaluate.call_count}."
    )


def test_eval_stage_records_phase1_reflection_token_phase(
    tmp_path,
    mock_llm,
    caplog,
):
    """Important-3 regression guard: when the runtime emits
    ``runtime_meta.phase1_reflection_token_usage``, the engine must
    record a ``phase1_reflection`` lifecycle entry in
    ``self.accumulator`` so the eventual ``token_phases`` flush has a
    dedicated row. Without this the reflection-turn tokens silently
    vanish from cost reports.

    Issue #704 regression guard (stronger than the lifecycle-only check
    below): the engine ALSO bumps the per-agent ``AgentState.token_usage``
    int counter (engine.py:3602) by ``p1_usage.total``. Before the fix
    that line read ``agent.token_usage += p1_usage`` — an ``int +=
    TokenUsage`` ``TypeError`` swallowed by the inner ``except`` — so the
    per-agent counter silently EXCLUDED phase-1 tokens (inconsistent with
    the phase-row aggregate this same test asserts on) and the run emitted
    a misleading "non-numeric fields" warning for perfectly numeric input.
    The lifecycle-accumulator assertions below passed even with the bug,
    because the accumulator is written one line *earlier* (engine.py:3594).
    """
    import logging

    from kcsi.models import TaskSpec
    from kcsi.runtime.types import RuntimeResult
    from kcsi.tokens import TokenUsage

    db_path = str(tmp_path / "mem.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="phase1_token_phase",
    )

    task_result = RuntimeResult(
        output="agent output",
        tool_trace=[],
        runtime_meta={
            "native_session_memory": "x",
            "session_scope": "task",
            "phase1_reflection_enabled": True,
            "phase1_eval_result": {"resolved": True, "native_score": 1.0, "task_type": "polyglot"},
            "phase1_reflection": "Assumption A. Change B. Outcome C.",
            "phase1_reflection_token_usage": {
                "input_tokens": 312,
                "output_tokens": 64,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 128,
            },
        },
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
    )
    # Any later run_task calls (cross-task forum / distillation phases that
    # still fire with per_task_forum_rounds=0) must contribute ZERO tokens so the
    # per-agent counter is exactly trace_total + p1_total and the #704
    # assertion below stays deterministic.
    zero_result = RuntimeResult(
        output="",
        tool_trace=[],
        runtime_meta={"session_scope": "task"},
        token_usage=TokenUsage(),
    )

    def _run_task_side_effect(*args, **kwargs):
        # First call = the task execution (carries phase-1); rest = 0 tokens.
        if not getattr(_run_task_side_effect, "_fired", False):
            _run_task_side_effect._fired = True
            return task_result
        return zero_result

    runtime = MagicMock()
    runtime.evaluator = None
    runtime.run_task.side_effect = _run_task_side_effect
    p1_total = 312 + 64 + 0 + 128  # == 504
    trace_total = 10 + 5  # the main-turn token_usage.total
    evaluator = MagicMock()
    evaluator.evaluate.return_value = {
        "resolved": True,
        "native_score": 1.0,
        "task_type": "polyglot",
    }

    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )
    # Pin the per-task insight/lesson token costs to 0 so the per-agent
    # counter is exactly trace_total + p1_total (no double counting, no
    # noise from the reflection/lesson LLM calls). Returning (None, 0) /
    # ([], 0) keeps the no_memory=False insight path active but free.
    orch._execution_phase._generate_reflection_and_lessons = MagicMock(return_value=(None, [], 0))  # type: ignore[method-assign]
    try:
        tasks = [TaskSpec(id="t-tok", repo="r", prompt="solve")]
        with caplog.at_level(logging.WARNING, logger="kcsi.orchestrator.engine"):
            orch.run(tasks)

        # Inspect the accumulator's lifecycle entries: there must be a
        # ``__lc:phase1_reflection`` key for this generation/agent.
        entries = orch.accumulator._entries  # noqa: SLF001 - test-only introspection
        keys = list(entries.keys())
        matches = [(gen, aid, src) for (gen, aid, src) in keys if src == "__lc:phase1_reflection"]
        assert matches, f"expected a ``__lc:phase1_reflection`` accumulator entry; got {keys}"
        # Numbers should match what we shipped through runtime_meta.
        usage = entries[matches[0]]
        assert usage.input_tokens == 312
        assert usage.output_tokens == 64
        assert usage.cache_read_input_tokens == 128

        # --- Issue #704: the per-agent int counter must INCLUDE phase-1 ---
        assert len(orch.agents) == 1
        agent = orch.agents[0]
        assert agent.token_usage == trace_total + p1_total, (
            "AgentState.token_usage must include the phase-1 reflection "
            f"tokens exactly once; expected {trace_total + p1_total} "
            f"(trace {trace_total} + phase1 {p1_total}), got "
            f"{agent.token_usage}. A value of {trace_total} means the "
            "`int += TokenUsage` TypeError silently dropped phase-1 "
            "tokens from the per-agent counter (the #704 bug)."
        )

        # --- Issue #704: numeric input must NOT trigger the misleading
        # "non-numeric fields" warning emitted by the swallowing except. ---
        offending = [r.getMessage() for r in caplog.records if "non-numeric" in r.getMessage()]
        assert not offending, (
            f"engine emitted a 'non-numeric fields' warning for numeric phase1_reflection_token_usage: {offending}"
        )
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_eval_stage_records_polyglot_test_feedback_token_phase(
    tmp_path,
    mock_llm,
):
    """Finding #6 (fixed): mirrors
    ``test_eval_stage_records_phase1_reflection_token_phase`` above, but for
    ``runtime_meta.polyglot_test_feedback_token_usage`` -- without this, the
    polyglot test-feedback retry loop's extra SDK-turn tokens silently
    vanish from ``token_phases`` cost reports."""
    from kcsi.models import TaskSpec
    from kcsi.runtime.types import RuntimeResult
    from kcsi.tokens import TokenUsage

    db_path = str(tmp_path / "mem.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="polyglot_tf_token_phase",
    )

    task_result = RuntimeResult(
        output="agent output",
        tool_trace=[],
        runtime_meta={
            "native_session_memory": "x",
            "session_scope": "task",
            "polyglot_test_feedback_meta": {
                "enabled": True,
                "rounds_used": 1,
                "attempt_1_eval_summary": {"resolved": False},
                "captured": True,
            },
            "polyglot_test_feedback_token_usage": {
                "input_tokens": 900,
                "output_tokens": 200,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 50,
            },
        },
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
    )
    zero_result = RuntimeResult(
        output="",
        tool_trace=[],
        runtime_meta={"session_scope": "task"},
        token_usage=TokenUsage(),
    )

    def _run_task_side_effect(*args, **kwargs):
        if not getattr(_run_task_side_effect, "_fired", False):
            _run_task_side_effect._fired = True
            return task_result
        return zero_result

    runtime = MagicMock()
    runtime.evaluator = None
    runtime.run_task.side_effect = _run_task_side_effect
    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0, "task_type": "polyglot"}

    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )
    orch._execution_phase._generate_reflection_and_lessons = MagicMock(return_value=(None, [], 0))  # type: ignore[method-assign]
    try:
        tasks = [TaskSpec(id="t-tok-tf", repo="r", prompt="solve")]
        orch.run(tasks)

        entries = orch.accumulator._entries  # noqa: SLF001 - test-only introspection
        keys = list(entries.keys())
        matches = [(gen, aid, src) for (gen, aid, src) in keys if src == "__lc:polyglot_test_feedback"]
        assert matches, f"expected a ``__lc:polyglot_test_feedback`` accumulator entry; got {keys}"
        usage = entries[matches[0]]
        assert usage.input_tokens == 900
        assert usage.output_tokens == 200
        assert usage.cache_read_input_tokens == 50

        assert len(orch.agents) == 1
        agent = orch.agents[0]
        trace_total = 10 + 5
        tf_total = 900 + 200 + 0 + 50
        assert agent.token_usage == trace_total + tf_total, (
            "AgentState.token_usage must include the polyglot test-feedback "
            f"tokens exactly once; expected {trace_total + tf_total}, got {agent.token_usage}."
        )
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_eval_stage_reevaluates_when_watcher_errored(
    tmp_path,
    mock_llm,
):
    """When phase1_reflection_enabled is True but the watcher's eval
    raised (so phase1_eval_result is absent), the engine must fall back
    to running evaluate() itself rather than silently shipping a 0."""
    from kcsi.models import TaskSpec
    from kcsi.runtime.types import RuntimeResult
    from kcsi.tokens import TokenUsage

    db_path = str(tmp_path / "mem.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="phase1_watcher_error_fallback",
    )

    runtime = MagicMock()
    runtime.evaluator = None
    runtime.run_task.return_value = RuntimeResult(
        output="agent output",
        tool_trace=[],
        runtime_meta={
            "native_session_memory": "x",
            "session_scope": "task",
            # Flag was on, but no cached value (watcher errored or
            # didn't fire because the agent never wrote a sentinel).
            "phase1_reflection_enabled": True,
            "phase1_eval_error": "FooError: nope",
        },
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
    )
    evaluator = MagicMock()
    evaluator.evaluate.return_value = {
        "resolved": True,
        "native_score": 1.0,
        "task_type": "polyglot",
    }

    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )
    try:
        tasks = [TaskSpec(id="t-fallback", repo="r", prompt="solve")]
        orch.run(tasks)
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()

    assert evaluator.evaluate.call_count == 1, (
        "Engine must fall through to evaluate() when watcher errored (phase1_eval_result absent)."
    )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

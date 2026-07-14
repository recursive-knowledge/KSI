# tests/orchestrator/test_resume.py
"""Tests for --resume / experiment-collision logic."""

import logging
from unittest.mock import MagicMock

from conftest import _build_mock_llm

from kcsi.models import GenerationConfig, TaskSpec
from kcsi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from kcsi.runtime.types import RuntimeResult
from kcsi.tokens import TokenAccumulator, TokenUsage


def _make_store(tmp_path, experiment="exp1"):
    from kcsi.memory.store import MemoryStore

    db_path = str(tmp_path / "test.sqlite")
    return MemoryStore(db_path, default_experiment=experiment)


class TestHasExperiment:
    def test_false_when_empty(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.has_experiment("exp1") is False

    def test_true_after_ensure_run(self, tmp_path):
        store = _make_store(tmp_path)
        store._ensure_run("exp1")
        assert store.has_experiment("exp1") is True

    def test_false_for_other_name(self, tmp_path):
        store = _make_store(tmp_path)
        store._ensure_run("exp1")
        assert store.has_experiment("exp2") is False


class TestNextExperimentName:
    def test_no_collision(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.next_experiment_name("myexp") == "myexp"

    def test_single_collision(self, tmp_path):
        store = _make_store(tmp_path)
        store._ensure_run("myexp")
        assert store.next_experiment_name("myexp") == "myexp_2"

    def test_multiple_collisions(self, tmp_path):
        store = _make_store(tmp_path)
        store._ensure_run("myexp")
        store._ensure_run("myexp_2")
        store._ensure_run("myexp_3")
        assert store.next_experiment_name("myexp") == "myexp_4"


class TestResumeSeeding:
    """Integration: --resume seeds best_scores; without it, scores are NOT seeded."""

    def _populate_scores(self, store, experiment, task_scores):
        """Insert task_memory_records so get_best_scores returns them."""
        run_id = store._ensure_run(experiment)
        store._ensure_generation(run_id, 1)
        for i, (task_id, score) in enumerate(task_scores.items()):
            store._execute(
                """INSERT INTO task_memory_records
                   (run_id, generation, agent_id, task_id, eval_results_json)
                   VALUES (?, 1, ?, ?, ?)""",
                (run_id, f"agent-{i}", task_id, f'{{"native_score": {score}}}'),
            )
        store._conn.commit()

    def test_resume_seeds_scores(self, tmp_path):
        """When resuming, get_best_scores returns prior data."""
        store = _make_store(tmp_path, experiment="myexp")
        self._populate_scores(store, "myexp", {"t1": 1.0, "t2": 0.5})
        scores = store.get_best_scores(experiment="myexp")
        assert scores == {"t1": 1.0, "t2": 0.5}

    def test_fresh_run_no_scores(self, tmp_path):
        """A suffixed experiment name has no prior scores."""
        store = _make_store(tmp_path, experiment="myexp")
        self._populate_scores(store, "myexp", {"t1": 1.0})
        # Simulate collision detection -> new name
        new_name = store.next_experiment_name("myexp")
        assert new_name == "myexp_2"
        scores = store.get_best_scores(experiment=new_name)
        assert scores == {}


class TestGetBestScoresFallbacks:
    """Verify get_best_scores handles all eval result formats."""

    def _insert_eval(self, store, experiment, task_id, eval_json_str):
        run_id = store._ensure_run(experiment)
        store._execute(
            """INSERT INTO task_memory_records
               (run_id, generation, agent_id, task_id, eval_results_json)
               VALUES (?, 1, ?, ?, ?)""",
            (run_id, "agent-0", task_id, eval_json_str),
        )
        store._conn.commit()

    def test_resolved_true(self, tmp_path):
        """SWE-bench format: {resolved: true} -> 1.0."""
        store = _make_store(tmp_path)
        self._insert_eval(store, "exp1", "swe-t1", '{"resolved": true}')
        scores = store.get_best_scores(experiment="exp1")
        assert scores == {"swe-t1": 1.0}

    def test_resolved_false(self, tmp_path):
        """SWE-bench format: {resolved: false} -> 0.0."""
        store = _make_store(tmp_path)
        self._insert_eval(store, "exp1", "swe-t2", '{"resolved": false}')
        scores = store.get_best_scores(experiment="exp1")
        assert scores == {"swe-t2": 0.0}

    def test_instance_report_resolved(self, tmp_path):
        """Nested instance_report.resolved -> 1.0."""
        store = _make_store(tmp_path)
        self._insert_eval(
            store,
            "exp1",
            "t3",
            '{"instance_report": {"resolved": true}}',
        )
        scores = store.get_best_scores(experiment="exp1")
        assert scores == {"t3": 1.0}

    def test_pass_flag(self, tmp_path):
        """Generic pass flag -> 1.0."""
        store = _make_store(tmp_path)
        self._insert_eval(store, "exp1", "t4", '{"pass": true}')
        scores = store.get_best_scores(experiment="exp1")
        assert scores == {"t4": 1.0}

    def test_native_score_takes_precedence(self, tmp_path):
        """native_score wins over resolved."""
        store = _make_store(tmp_path)
        self._insert_eval(
            store,
            "exp1",
            "t5",
            '{"native_score": 0.75, "resolved": true}',
        )
        scores = store.get_best_scores(experiment="exp1")
        assert scores == {"t5": 0.75}

    def test_empty_eval_results(self, tmp_path):
        """Empty eval results -> no score."""
        store = _make_store(tmp_path)
        self._insert_eval(store, "exp1", "t6", "{}")
        scores = store.get_best_scores(experiment="exp1")
        assert scores == {}

    def test_mixed_formats_best_wins(self, tmp_path):
        """Multiple records for same task: best score wins."""
        store = _make_store(tmp_path)
        run_id = store._ensure_run("exp1")
        # First attempt: resolved=false -> 0.0
        store._execute(
            """INSERT INTO task_memory_records
               (run_id, generation, agent_id, task_id, eval_results_json)
               VALUES (?, 1, ?, ?, ?)""",
            (run_id, "agent-0", "swe-t7", '{"resolved": false}'),
        )
        # Second attempt: resolved=true -> 1.0
        store._execute(
            """INSERT INTO task_memory_records
               (run_id, generation, agent_id, task_id, eval_results_json)
               VALUES (?, 2, ?, ?, ?)""",
            (run_id, "agent-1", "swe-t7", '{"resolved": true}'),
        )
        store._conn.commit()
        scores = store.get_best_scores(experiment="exp1")
        assert scores == {"swe-t7": 1.0}


def test_resume_carries_forward_preserved_task_without_rerunning(tmp_path):
    store = _make_store(tmp_path, experiment="resume-exp")
    store.upsert_task_memory_record(
        experiment="resume-exp",
        generation=1,
        agent_id="agent-0",
        task_id="arc-task-1",
        eval_results={"resolved": True, "native_score": 1.0, "task_type": "arc"},
        final_model_output="solved-grid",
        full_memory_trace="trace-1",
        full_memory_trace_condensed="condensed-1",
        task_specific_insights=[],
        attempt_event={"status": "ok", "resolved": True},
    )
    runtime_db_path = store._db_path
    store.close()

    config = GenerationConfig(
        num_generations=2,
        num_agents=1,
        per_task_forum_rounds=0,
        drop_solved=False,
        solved_threshold=1.0,
        runtime_db_path=runtime_db_path,
        experiment_name="resume-exp",
        resume=True,
    )
    config.cross_task_forum_rounds = 0
    config.distill_enabled = False

    runtime = MagicMock()
    runtime.run_task.side_effect = AssertionError("runtime.run_task should not execute for carried trace")
    evaluator = MagicMock()
    evaluator.evaluate.side_effect = AssertionError("evaluator.evaluate should not execute for carried trace")

    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=_build_mock_llm(),
        persistence=NoopPersistence(),
    )
    traces = orch.run([TaskSpec(id="arc-task-1", prompt="Solve ARC task", metadata={"task_source": "arc"})])

    assert len(traces) == 1
    trace = traces[0]
    assert trace.generation == 2
    assert trace.native_score == 1.0
    assert trace.model_output == "solved-grid"
    assert trace.runtime_meta["carry_forward"] is True
    assert trace.runtime_meta["carry_forward_source_generation"] == 1


def test_resume_knowledge_store_carries_forward_preserved_task_without_rerunning(tmp_path):
    from kcsi.memory.knowledge_store import KnowledgeStore

    knowledge_db_path = str(tmp_path / "resume_knowledge.sqlite")
    knowledge = KnowledgeStore(knowledge_db_path, default_experiment="resume-exp")
    try:
        knowledge.record_attempt(
            task_id="arc-task-1",
            agent_id="agent-0",
            generation=1,
            eval_results={"resolved": True, "native_score": 1.0, "task_type": "arc"},
            model_output="solved-grid",
            native_score=1.0,
            experiment="resume-exp",
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
        experiment_name="resume-exp",
        resume=True,
    )
    config.cross_task_forum_rounds = 0
    config.distill_enabled = False

    runtime = MagicMock()
    runtime.run_task.side_effect = AssertionError("runtime.run_task should not execute for carried trace")
    evaluator = MagicMock()
    evaluator.evaluate.side_effect = AssertionError("evaluator.evaluate should not execute for carried trace")

    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=_build_mock_llm(),
        persistence=NoopPersistence(),
    )
    try:
        traces = orch.run([TaskSpec(id="arc-task-1", prompt="Solve ARC task", metadata={"task_source": "arc"})])
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()

    assert len(traces) == 1
    trace = traces[0]
    assert trace.generation == 2
    assert trace.native_score == 1.0
    assert trace.model_output == "solved-grid"
    assert trace.runtime_meta["carry_forward"] is True
    assert trace.runtime_meta["carry_forward_source_generation"] == 1


def test_resume_preserves_original_carry_forward_provenance_from_knowledge_history(tmp_path):
    knowledge_db_path = str(tmp_path / "preserve_knowledge.sqlite")
    task = TaskSpec(id="arc-task-1", prompt="Solve ARC task", metadata={"task_source": "arc"})

    config = GenerationConfig(
        num_generations=2,
        num_agents=1,
        per_task_forum_rounds=0,
        drop_solved=False,
        solved_threshold=1.0,
        knowledge_db_path=knowledge_db_path,
        experiment_name="resume-exp",
    )
    config.cross_task_forum_rounds = 0
    config.distill_enabled = False

    runtime = MagicMock()
    runtime.run_task.return_value = RuntimeResult(
        output="solved-grid",
        tool_trace=[],
        runtime_meta={"native_session_memory": "gen1", "session_scope": "task"},
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
    )
    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0, "task_type": "arc"}

    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=_build_mock_llm(),
        persistence=NoopPersistence(),
    )
    try:
        first_run_traces = orch.run([task])
        assert len(first_run_traces) == 2
        assert runtime.run_task.call_count == 1
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()

    resume_config = GenerationConfig(
        num_generations=3,
        num_agents=1,
        per_task_forum_rounds=0,
        drop_solved=False,
        solved_threshold=1.0,
        knowledge_db_path=knowledge_db_path,
        experiment_name="resume-exp",
        resume=True,
    )
    resume_config.cross_task_forum_rounds = 0
    resume_config.distill_enabled = False

    resumed_runtime = MagicMock()
    resumed_runtime.run_task.side_effect = AssertionError(
        "runtime.run_task should not execute for resumed carried trace"
    )
    resumed_evaluator = MagicMock()
    resumed_evaluator.evaluate.side_effect = AssertionError(
        "evaluator.evaluate should not execute for resumed carried trace"
    )

    resumed = GenerationalOrchestrator(
        config=resume_config,
        runtime=resumed_runtime,
        evaluator=resumed_evaluator,
        llm=_build_mock_llm(),
        persistence=NoopPersistence(),
    )
    try:
        resumed_traces = resumed.run([task])
    finally:
        if resumed._knowledge is not None:
            resumed._knowledge.close()

    assert len(resumed_traces) == 1
    trace = resumed_traces[0]
    assert trace.generation == 3
    assert trace.runtime_meta["carry_forward"] is True
    assert trace.runtime_meta["carry_forward_source_generation"] == 1


class TestResumeTokenRehydration:
    """Issue #668: on --resume the accumulator must rehydrate prior generations'
    token_phases so token_usage_total does not undercount the pre-resume run."""

    def _flush_gen(self, store, run_id, accumulator, *, generation, task_usage, lifecycle=None):
        accumulator.record_task(generation, f"agent-g{generation}", f"t{generation}", task_usage)
        if lifecycle is not None:
            accumulator.record_lifecycle(generation, f"agent-g{generation}", "kt_adapter", lifecycle)
        accumulator.flush_to_store(store, run_id=run_id, generation=generation, model="haiku")

    def test_get_token_phases_filters_by_generation(self, tmp_path):
        store = _make_store(tmp_path)
        run_id = store._ensure_run("exp1")
        acc = TokenAccumulator()
        self._flush_gen(store, run_id, acc, generation=1, task_usage=TokenUsage(input_tokens=100, output_tokens=50))
        self._flush_gen(store, run_id, acc, generation=2, task_usage=TokenUsage(input_tokens=200, output_tokens=80))

        only_g1 = store.get_token_phases(experiment="exp1", before_generation=2)
        assert {r["generation"] for r in only_g1} == {1}
        both = store.get_token_phases(experiment="exp1", before_generation=3)
        assert {r["generation"] for r in both} == {1, 2}
        # Unknown experiment -> no run -> empty.
        assert store.get_token_phases(experiment="nope", before_generation=99) == []

    def test_load_from_store_restores_prior_total(self, tmp_path):
        store = _make_store(tmp_path)
        run_id = store._ensure_run("exp1")
        writer = TokenAccumulator()
        self._flush_gen(
            store,
            run_id,
            writer,
            generation=1,
            task_usage=TokenUsage(input_tokens=100, output_tokens=50, cache_read_input_tokens=10),
            lifecycle=TokenUsage(input_tokens=5, output_tokens=3),
        )
        self._flush_gen(store, run_id, writer, generation=2, task_usage=TokenUsage(input_tokens=200, output_tokens=80))

        # Simulate resume at gen 2: a fresh accumulator rehydrates gens < 2.
        resumed = TokenAccumulator()
        replayed = resumed.load_from_store(store, experiment="exp1", before_generation=2)
        assert replayed == 2  # one task row + one lifecycle row from gen 1
        assert resumed.total() == TokenUsage(input_tokens=105, output_tokens=53, cache_read_input_tokens=10)
        assert set(resumed.by_generation().keys()) == {1}

    def test_resume_total_includes_prior_and_new_generations(self, tmp_path):
        store = _make_store(tmp_path)
        run_id = store._ensure_run("exp1")
        writer = TokenAccumulator()
        self._flush_gen(store, run_id, writer, generation=1, task_usage=TokenUsage(input_tokens=100, output_tokens=50))

        # Resume: rehydrate gen 1, then record gen 2 work in the new run.
        resumed = TokenAccumulator()
        resumed.load_from_store(store, experiment="exp1", before_generation=2)
        resumed.record_task(2, "agent-g2", "t2", TokenUsage(input_tokens=200, output_tokens=80))

        assert resumed.total() == TokenUsage(input_tokens=300, output_tokens=130)
        assert set(resumed.by_generation().keys()) == {1, 2}

    def test_load_from_store_no_run_returns_zero(self, tmp_path):
        store = _make_store(tmp_path)
        acc = TokenAccumulator()
        assert acc.load_from_store(store, experiment="exp1", before_generation=5) == 0
        assert acc.total() == TokenUsage()

    def test_rehydrated_rows_are_not_reflushed(self, tmp_path):
        """Rehydrated prior-gen keys are marked flushed so a later flush of the
        resumed generation never double-writes them."""
        store = _make_store(tmp_path)
        run_id = store._ensure_run("exp1")
        writer = TokenAccumulator()
        self._flush_gen(store, run_id, writer, generation=1, task_usage=TokenUsage(input_tokens=100, output_tokens=50))

        resumed = TokenAccumulator()
        resumed.load_from_store(store, experiment="exp1", before_generation=2)
        resumed.record_task(2, "agent-g2", "t2", TokenUsage(input_tokens=200, output_tokens=80))
        resumed.flush_to_store(store, run_id=run_id, generation=2, model="haiku")

        # Gen 1 still has exactly its original rows (no duplicates from rehydration).
        g1_rows = store.get_token_phases(experiment="exp1", before_generation=2)
        assert len(g1_rows) == 1

    def test_load_from_store_collapses_multiple_task_rows(self, tmp_path):
        """A generation with several tasks writes several task_execution rows;
        rehydration must collapse-and-sum them into a single entry without
        losing or double-counting tokens (the riskiest accounting path)."""
        store = _make_store(tmp_path)
        run_id = store._ensure_run("exp1")
        writer = TokenAccumulator()
        # Three tasks, same gen + same agent -> three task_execution rows.
        writer.record_task(1, "agent-0", "t1", TokenUsage(input_tokens=100, output_tokens=10))
        writer.record_task(1, "agent-0", "t2", TokenUsage(input_tokens=200, output_tokens=20))
        writer.record_task(1, "agent-0", "t3", TokenUsage(input_tokens=300, output_tokens=30))
        writer.flush_to_store(store, run_id=run_id, generation=1, model="haiku")
        assert len(store.get_token_phases(experiment="exp1", before_generation=2)) == 3

        resumed = TokenAccumulator()
        replayed = resumed.load_from_store(store, experiment="exp1", before_generation=2)
        assert replayed == 3
        assert resumed.total() == TokenUsage(input_tokens=600, output_tokens=60)
        # All three rows collapse into ONE (gen, agent, "task_execution") key.
        task_keys = [k for k in resumed._entries if k[2] == "task_execution"]
        assert len(task_keys) == 1

    def test_load_from_store_preserves_by_agent_breakdown(self, tmp_path):
        """by_agent() must survive the round-trip keyed on the original agent_ref,
        not collapse to an empty-string key."""
        store = _make_store(tmp_path)
        run_id = store._ensure_run("exp1")
        writer = TokenAccumulator()
        writer.record_task(1, "agent-A", "t1", TokenUsage(input_tokens=100, output_tokens=10))
        writer.record_task(1, "agent-B", "t2", TokenUsage(input_tokens=200, output_tokens=20))
        writer.flush_to_store(store, run_id=run_id, generation=1, model="haiku")

        resumed = TokenAccumulator()
        resumed.load_from_store(store, experiment="exp1", before_generation=2)
        by_agent = resumed.by_agent()
        assert set(by_agent.keys()) == {"agent-A", "agent-B"}
        assert by_agent["agent-A"] == TokenUsage(input_tokens=100, output_tokens=10)
        assert by_agent["agent-B"] == TokenUsage(input_tokens=200, output_tokens=20)


def _seed_carried_runtime_store(tmp_path, experiment="resume-exp", token_rows=None):
    """A runtime MemoryStore with a gen-1 solved task (so resume advances to gen 2
    and the task carries forward without re-running). Optionally pre-write gen-1
    token_phases rows. Returns the db path (store is closed)."""
    store = _make_store(tmp_path, experiment=experiment)
    run_id = store._ensure_run(experiment)
    for phase, agent_ref, usage in token_rows or []:
        store.insert_token_phase(
            run_id=run_id, generation=1, phase=phase, agent_ref=agent_ref, token_usage=usage, cost_usd=0.0
        )
    store.upsert_task_memory_record(
        experiment=experiment,
        generation=1,
        agent_id="agent-0",
        task_id="arc-task-1",
        eval_results={"resolved": True, "native_score": 1.0, "task_type": "arc"},
        final_model_output="solved-grid",
        full_memory_trace="trace-1",
        full_memory_trace_condensed="condensed-1",
        task_specific_insights=[],
        attempt_event={"status": "ok", "resolved": True},
    )
    db_path = store._db_path
    store.close()
    return db_path


def _run_resumed_orchestrator(runtime_db_path, experiment="resume-exp"):
    config = GenerationConfig(
        num_generations=2,
        num_agents=1,
        per_task_forum_rounds=0,
        drop_solved=False,
        solved_threshold=1.0,
        runtime_db_path=runtime_db_path,
        experiment_name=experiment,
        resume=True,
    )
    config.cross_task_forum_rounds = 0
    config.distill_enabled = False
    runtime = MagicMock()
    runtime.run_task.side_effect = AssertionError("runtime.run_task should not execute for carried trace")
    evaluator = MagicMock()
    evaluator.evaluate.side_effect = AssertionError("evaluator.evaluate should not execute for carried trace")
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=_build_mock_llm(),
        persistence=NoopPersistence(),
    )
    orch.run([TaskSpec(id="arc-task-1", prompt="Solve ARC task", metadata={"task_source": "arc"})])
    return orch


def test_resume_engine_rehydrates_prior_generation_tokens(tmp_path, caplog):
    """Issue #668 / H1: engine.run() on --resume rehydrates prior-gen token_phases
    into the accumulator (end-to-end wiring, not just the unit method)."""
    runtime_db_path = _seed_carried_runtime_store(
        tmp_path,
        token_rows=[
            ("task_execution", "agent-0", TokenUsage(input_tokens=100, output_tokens=50, cache_read_input_tokens=10)),
            ("kt_adapter", "agent-0", TokenUsage(input_tokens=5, output_tokens=3)),
        ],
    )
    with caplog.at_level(logging.INFO):
        orch = _run_resumed_orchestrator(runtime_db_path)

    assert orch._start_generation == 2
    assert orch.accumulator.total() == TokenUsage(input_tokens=105, output_tokens=53, cache_read_input_tokens=10)
    assert "rehydrated" in caplog.text


def test_resume_engine_warns_when_no_token_phases_rows(tmp_path, caplog):
    """M1: resuming against a runtime DB that has the run but NO token_phases rows
    must warn about the undercount instead of silently rehydrating zero."""
    runtime_db_path = _seed_carried_runtime_store(tmp_path, token_rows=None)
    with caplog.at_level(logging.WARNING):
        orch = _run_resumed_orchestrator(runtime_db_path)

    assert orch._start_generation == 2
    assert orch.accumulator.total() == TokenUsage()
    assert "no token_phases rows found" in caplog.text


def test_resume_engine_warns_without_runtime_db(tmp_path, caplog):
    """M3: resuming with only a knowledge DB (no runtime DB) must warn that
    token_usage_total will undercount, since token_phases lives only in the
    runtime DB."""
    from kcsi.memory.knowledge_store import KnowledgeStore

    knowledge_db_path = str(tmp_path / "resume_knowledge.sqlite")
    knowledge = KnowledgeStore(knowledge_db_path, default_experiment="resume-exp")
    try:
        knowledge.record_attempt(
            task_id="arc-task-1",
            agent_id="agent-0",
            generation=1,
            eval_results={"resolved": True, "native_score": 1.0, "task_type": "arc"},
            model_output="solved-grid",
            native_score=1.0,
            experiment="resume-exp",
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
        experiment_name="resume-exp",
        resume=True,
    )
    config.cross_task_forum_rounds = 0
    config.distill_enabled = False
    runtime = MagicMock()
    runtime.run_task.side_effect = AssertionError("runtime.run_task should not execute for carried trace")
    evaluator = MagicMock()
    evaluator.evaluate.side_effect = AssertionError("evaluator.evaluate should not execute for carried trace")
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=_build_mock_llm(),
        persistence=NoopPersistence(),
    )
    try:
        with caplog.at_level(logging.WARNING):
            orch.run([TaskSpec(id="arc-task-1", prompt="Solve ARC task", metadata={"task_source": "arc"})])
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()

    assert orch._start_generation == 2
    assert "without a runtime DB" in caplog.text

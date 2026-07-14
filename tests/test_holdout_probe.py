"""Tests for the in-run hold-out transfer probe (engine side).

Hold-out tasks (``GenerationConfig.holdout_task_ids``) are attempted EVERY
generation with the current cross-task knowledge injected, but are completely
excluded from learning (forums, distillation, seeding inputs), from
``--drop-solved`` bookkeeping, from early-stop, and from headline metrics.

Uses the same fake-engine pattern as ``tests/test_distill_phase.py``: a real
``GenerationalOrchestrator`` with a real KnowledgeStore on ``tmp_path`` and
MagicMock runtime/evaluator/llm.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from ksi.models import GenerationConfig, TaskSpec, TaskTrace
from ksi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from ksi.orchestrator.execution_phase import ExecutionPhaseResult
from ksi.tokens import LLMResponse, TokenUsage
from tests.orchestrator_phase_helpers import cross_task_forum, per_task_forum, run_distill, seed_next_generation


def _make_orch(
    tmp_path,
    *,
    holdout_task_ids: list[str] | None = None,
    num_generations: int = 1,
    num_agents: int = 1,
    no_memory: bool = False,
    resume: bool = False,
    db_name: str = "knowledge.sqlite",
) -> GenerationalOrchestrator:
    db_path = str(tmp_path / db_name)
    runtime = MagicMock()
    evaluator = MagicMock()
    llm = MagicMock()
    llm.call.return_value = LLMResponse(
        text=json.dumps({"transferable_insights": [], "pitfalls": [], "checks": [], "evidence_post_ids": []}),
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )
    config = GenerationConfig(
        num_generations=num_generations,
        num_agents=num_agents,
        per_task_forum_rounds=1,
        knowledge_db_path=db_path,
        holdout_task_ids=list(holdout_task_ids or []),
        no_memory=no_memory,
        resume=resume,
    )
    return GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )


def _trace(task_id: str, *, generation: int = 1, agent_id: str = "agent-0", score: float = 1.0) -> TaskTrace:
    return TaskTrace(
        generation=generation,
        agent_id=agent_id,
        task_id=task_id,
        model_output="output",
        eval_result={"score": score},
        native_score=score,
    )


class TestHoldoutAttemptTagging:
    def test_holdout_attempt_meta_tagged(self, tmp_path):
        orch = _make_orch(tmp_path, holdout_task_ids=["h1"])
        orch._tasks_by_id = {}
        assert orch._execution_phase._persist_knowledge_attempt_early(_trace("h1"))
        page = orch._knowledge.query_task("h1", entry_types=["attempt"])
        attempts = page["attempts"]
        assert attempts, f"no attempt rows found for h1: {page!r}"
        content = attempts[0]["content"]
        assert content["attempt_meta"]["holdout"] is True

    def test_training_attempt_meta_not_tagged(self, tmp_path):
        orch = _make_orch(tmp_path, holdout_task_ids=["h1"])
        orch._tasks_by_id = {}
        assert orch._execution_phase._persist_knowledge_attempt_early(_trace("t1"))
        page = orch._knowledge.query_task("t1", entry_types=["attempt"])
        attempts = page["attempts"]
        assert attempts
        content = attempts[0]["content"]
        assert "holdout" not in (content.get("attempt_meta") or {})


class TestHoldoutScoreIsolation:
    def test_best_scores_untouched_by_holdout(self, tmp_path):
        orch = _make_orch(tmp_path, holdout_task_ids=["h1"])
        orch._tasks_by_id = {}
        orch._update_score_tracking(1, [_trace("h1", score=1.0), _trace("t1", score=0.5, agent_id="agent-1")])
        assert "h1" not in orch._best_scores
        assert "h1" not in orch._best_preserved_traces
        assert orch._best_scores.get("t1") == 0.5

    def test_resume_drops_holdout_best_scores(self, tmp_path):
        seed = _make_orch(tmp_path, db_name="resume.sqlite")
        seed._knowledge.record_attempt(
            task_id="t1",
            agent_id="agent-0",
            generation=1,
            eval_results={"score": 1.0},
            model_output="o",
            trace_condensed="c",
            insights=[],
            native_score=1.0,
            experiment=seed.config.experiment_name,
        )
        seed._knowledge.record_attempt(
            task_id="h1",
            agent_id="agent-0",
            generation=1,
            eval_results={"score": 1.0},
            model_output="o",
            trace_condensed="c",
            insights=[],
            native_score=1.0,
            experiment=seed.config.experiment_name,
        )
        seed._knowledge.close()

        resumed = _make_orch(tmp_path, holdout_task_ids=["h1"], resume=True, db_name="resume.sqlite")
        assert resumed._best_scores.get("t1") == 1.0
        assert "h1" not in resumed._best_scores


class TestHoldoutLearningExclusion:
    def test_distill_task_ids_exclude_holdouts(self, tmp_path, monkeypatch):
        from ksi.distillation import DistillOutput

        orch = _make_orch(tmp_path, holdout_task_ids=["h1"])
        captured: dict = {}

        def fake_distill(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
            captured["task_ids"] = list(inp.task_ids)
            captured["cross_task_target_ids"] = list(inp.cross_task_target_ids or [])
            return DistillOutput(per_task={}, cross_task=None)

        import ksi.distillation as dist_pkg

        monkeypatch.setattr(dist_pkg, "distill", fake_distill)
        run_distill(orch, generation=1, task_ids=["h1", "t1"])
        assert captured["task_ids"] == ["t1"]
        assert captured["cross_task_target_ids"] == ["t1", "h1"]

    def test_distill_skipped_when_only_holdouts(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path, holdout_task_ids=["h1"])
        called = {"n": 0}

        def fake_distill(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
            called["n"] += 1
            raise AssertionError("distill must not run for holdout-only task ids")

        import ksi.distillation as dist_pkg

        monkeypatch.setattr(dist_pkg, "distill", fake_distill)
        run_distill(orch, generation=1, task_ids=["h1"])
        assert called["n"] == 0

    def test_non_holdout_filters_traces(self, tmp_path):
        orch = _make_orch(tmp_path, holdout_task_ids=["h1"])
        traces = [_trace("t1"), _trace("h1"), _trace("t2")]
        assert [t.task_id for t in orch._non_holdout(traces)] == ["t1", "t2"]

    def test_non_holdout_noop_when_feature_unused(self, tmp_path):
        orch = _make_orch(tmp_path)
        traces = [_trace("t1"), _trace("h1")]
        assert orch._non_holdout(traces) == traces

    def test_per_task_forum_skips_holdout_only_traces(self, tmp_path, monkeypatch):
        import ksi.memory.forum_bus as forum_bus_mod

        orch = _make_orch(tmp_path, holdout_task_ids=["h1"])

        def bomb(*args, **kwargs):
            raise AssertionError("ForumBus must not be constructed for holdout-only traces")

        monkeypatch.setattr(forum_bus_mod, "ForumBus", bomb)
        # Filtering all-holdout traces leaves nothing — the phase must return
        # before any forum machinery spins up.
        per_task_forum(orch, 1, [_trace("h1")])

    def test_cross_task_forum_skips_holdout_only_traces(self, tmp_path, monkeypatch):
        import ksi.memory.forum_bus as forum_bus_mod

        orch = _make_orch(tmp_path, holdout_task_ids=["h1"])

        def bomb(*args, **kwargs):
            raise AssertionError("ForumBus must not be constructed for holdout-only traces")

        monkeypatch.setattr(forum_bus_mod, "ForumBus", bomb)
        cross_task_forum(orch, generation=1, traces=[_trace("h1")])


class TestHoldoutTokenPhase:
    def test_holdout_tokens_recorded_under_holdout_phase(self, tmp_path):
        orch = _make_orch(tmp_path, holdout_task_ids=["h1"])
        holdout_trace = _trace("h1")
        holdout_trace.token_usage = TokenUsage(input_tokens=10, output_tokens=5)
        training_trace = _trace("t1", agent_id="agent-1")
        training_trace.token_usage = TokenUsage(input_tokens=3, output_tokens=2)
        orch._record_task_tokens(holdout_trace)
        orch._record_task_tokens(training_trace)
        store = MagicMock()
        orch.accumulator.flush_to_store(store, run_id=1, generation=1, model="m")
        phases = [call.kwargs["phase"] for call in store.insert_token_phase.call_args_list]
        assert "task_execution_holdout" in phases
        assert "task_execution" in phases

    def test_training_tokens_keep_task_execution_phase(self, tmp_path):
        orch = _make_orch(tmp_path)
        trace = _trace("t1")
        trace.token_usage = TokenUsage(input_tokens=3, output_tokens=2)
        orch._record_task_tokens(trace)
        store = MagicMock()
        orch.accumulator.flush_to_store(store, run_id=1, generation=1, model="m")
        phases = [call.kwargs["phase"] for call in store.insert_token_phase.call_args_list]
        assert phases == ["task_execution"]


class TestHoldoutSeedPackageIsolation:
    """Hold-out agents must receive exactly what a brand-new task would:
    the cross-task channel only — no own-task prior attempts, no related-task
    summaries — and hold-out attempts must never enrich training agents."""

    @staticmethod
    def _record_attempt(orch, task_id: str, *, generation: int = 1, score: float = 1.0) -> None:
        orch._knowledge.record_attempt(
            task_id=task_id,
            agent_id="agent-0",
            generation=generation,
            eval_results={"score": score},
            model_output="o",
            trace_condensed="c",
            insights=[],
            native_score=score,
            experiment=orch.config.experiment_name,
        )

    @staticmethod
    def _enrich_single_agent(orch, task_id: str) -> dict:
        from ksi.models import AgentState

        agent = AgentState(id="agent-0", generation=2, seed_package={})
        orch.agents = [agent]
        orch._enrichment_phase.enrich(2, {"agent-0": [task_id]}, [TaskSpec(id=task_id)])
        return agent.seed_package

    def test_holdout_seed_package_has_no_prior_attempts(self, tmp_path):
        orch = _make_orch(tmp_path, holdout_task_ids=["h1"])
        self._record_attempt(orch, "h1")  # prior-gen hold-out attempt exists
        pkg = self._enrich_single_agent(orch, "h1")
        assert pkg["prior_attempts"] == []
        assert pkg["related_summaries"] == []
        assert pkg["memory_snapshot"]["query_records_by_task"] == {}
        assert pkg["memory_snapshot"]["search_rows"] == []

    def test_training_seed_package_still_gets_prior_attempts(self, tmp_path):
        """Contrast: without the hold-out designation the same history IS
        injected — proves the gate (not an empty store) explains the test
        above."""
        orch = _make_orch(tmp_path)
        self._record_attempt(orch, "h1")
        pkg = self._enrich_single_agent(orch, "h1")
        assert pkg["prior_attempts"] != []

    def test_training_related_summaries_exclude_holdout(self, tmp_path):
        """Reverse leak: a hold-out attempt summary that prefix-matches a
        training task must not appear in the training agent's
        related-summaries or search rows."""
        orch = _make_orch(tmp_path, holdout_task_ids=["repo__h1"])
        self._record_attempt(orch, "repo__h1")  # hold-out, same prefix
        self._record_attempt(orch, "repo__t2")  # training sibling
        pkg = self._enrich_single_agent(orch, "repo__t1")
        related_ids = {row["task_id"] for row in pkg["related_summaries"]}
        assert "repo__t2" in related_ids
        assert "repo__h1" not in related_ids
        search_ids = {row["task_id"] for row in pkg["memory_snapshot"]["search_rows"]}
        assert "repo__h1" not in search_ids

    def test_related_summaries_unchanged_when_feature_unused(self, tmp_path):
        orch = _make_orch(tmp_path)
        self._record_attempt(orch, "repo__h1")
        self._record_attempt(orch, "repo__t2")
        pkg = self._enrich_single_agent(orch, "repo__t1")
        related_ids = {row["task_id"] for row in pkg["related_summaries"]}
        assert related_ids == {"repo__h1", "repo__t2"}


def _fake_execute(orch, score_by_gen_task: dict[tuple[int, str], float], dispatch_log: list):
    """Replace the execution phase service with a deterministic fake."""

    class FakeExecutionPhase:
        def run(self, phase_input):
            generation = phase_input.generation
            assigned_map = phase_input.assigned_map
            dispatched = sorted(tid for tids in assigned_map.values() for tid in tids)
            dispatch_log.append((generation, dispatched))
            traces = []
            for agent_id, tids in assigned_map.items():
                for tid in tids:
                    score = score_by_gen_task.get((generation, tid), 0.0)
                    traces.append(
                        TaskTrace(
                            generation=generation,
                            agent_id=agent_id,
                            task_id=tid,
                            model_output="o",
                            eval_result={"score": score},
                            native_score=score,
                        )
                    )
            return ExecutionPhaseResult(traces=traces)

    orch._execution_phase = FakeExecutionPhase()


class TestHoldoutRunLoop:
    def test_holdout_redispatched_after_solving(self, tmp_path):
        """A solved hold-out task is attempted fresh in the next generation."""
        orch = _make_orch(tmp_path, holdout_task_ids=["h1"], num_generations=3, num_agents=2, no_memory=True)
        dispatch_log: list = []
        _fake_execute(
            orch,
            {(1, "t1"): 0.0, (1, "h1"): 1.0, (2, "t1"): 1.0, (2, "h1"): 1.0},
            dispatch_log,
        )
        orch.run(tasks=[TaskSpec(id="t1"), TaskSpec(id="h1")])
        # h1 solved in gen 1 but is dispatched again in gen 2.
        assert dispatch_log[0] == (1, ["h1", "t1"])
        assert dispatch_log[1] == (2, ["h1", "t1"])

    def test_early_stop_ignores_holdouts(self, tmp_path):
        """Once all training tasks are solved the run stops, even though
        hold-out tasks would still be re-attempted forever."""
        orch = _make_orch(tmp_path, holdout_task_ids=["h1"], num_generations=4, num_agents=2, no_memory=True)
        dispatch_log: list = []
        _fake_execute(
            orch,
            {(1, "t1"): 0.0, (1, "h1"): 1.0, (2, "t1"): 1.0, (2, "h1"): 1.0},
            dispatch_log,
        )
        orch.run(tasks=[TaskSpec(id="t1"), TaskSpec(id="h1")])
        # t1 solves in gen 2 -> gen 3 has no training tasks left -> stop.
        assert [gen for gen, _ in dispatch_log] == [1, 2]

    def test_holdout_rate_is_per_gen_non_cumulative(self, tmp_path):
        orch = _make_orch(tmp_path, holdout_task_ids=["h1"], num_generations=3, num_agents=2, no_memory=True)
        dispatch_log: list = []
        _fake_execute(
            orch,
            {
                (1, "t1"): 0.0,
                (1, "h1"): 1.0,  # solved in gen 1...
                (2, "t1"): 0.0,
                (2, "h1"): 0.0,  # ...but NOT in gen 2 (rate must drop back to 0)
                (3, "t1"): 1.0,
                (3, "h1"): 1.0,
            },
            dispatch_log,
        )
        orch.run(tasks=[TaskSpec(id="t1"), TaskSpec(id="h1")])
        rates = orch.holdout_solve_rate_by_generation()
        assert rates[1] == {"solved": 1, "total": 1, "rate": 1.0}
        assert rates[2] == {"solved": 0, "total": 1, "rate": 0.0}
        assert rates[3] == {"solved": 1, "total": 1, "rate": 1.0}
        # Hold-out never leaks into headline best scores.
        assert "h1" not in orch._best_scores

    def test_holdout_rate_empty_when_feature_unused(self, tmp_path):
        orch = _make_orch(tmp_path, num_generations=1, no_memory=True)
        dispatch_log: list = []
        _fake_execute(orch, {(1, "t1"): 1.0}, dispatch_log)
        orch.run(tasks=[TaskSpec(id="t1")])
        assert orch.holdout_solve_rate_by_generation() == {}


class TestHoldoutR0InsightExclusion:
    """Hold-out attempts must produce NO R0 reflection insight rows in the
    knowledge store — insight rows are untagged and would otherwise leak
    hold-out content into forum retrieval."""

    @staticmethod
    def _insight(task_id: str):
        from ksi.models import Insight

        return Insight(
            id="ins-1",
            text="holdout reflection text",
            author_agent_id="agent-0",
            generation=1,
            workstream=task_id,
            source_task_id=task_id,
        )

    @staticmethod
    def _agent():
        from ksi.models import AgentState

        return AgentState(id="agent-0", generation=1)

    def _insight_rows(self, orch, task_id: str) -> list:
        page = orch._knowledge.query_task(task_id, entry_types=["insight"])
        return list(page["insights"])

    def test_no_insight_row_for_holdout_trace(self, tmp_path):
        orch = _make_orch(tmp_path, holdout_task_ids=["h1"])
        orch._execution_phase._record_r0_insight(
            generation=1, agent=self._agent(), trace=_trace("h1"), insight=self._insight("h1")
        )
        assert self._insight_rows(orch, "h1") == []

    def test_insight_row_recorded_for_training_trace(self, tmp_path):
        orch = _make_orch(tmp_path, holdout_task_ids=["h1"])
        orch._execution_phase._record_r0_insight(
            generation=1, agent=self._agent(), trace=_trace("t1"), insight=self._insight("t1")
        )
        assert len(self._insight_rows(orch, "t1")) == 1

    def test_insight_recording_noop_when_feature_unused(self, tmp_path):
        orch = _make_orch(tmp_path)
        orch._execution_phase._record_r0_insight(
            generation=1, agent=self._agent(), trace=_trace("h1"), insight=self._insight("h1")
        )
        assert len(self._insight_rows(orch, "h1")) == 1


class TestHoldoutRunEndLog:
    """#986: the run-end summary log line must surface the holdout solve rate
    (in addition to the pre-existing cumulative-solved line) when a holdout
    probe is configured, and must omit it entirely when the feature is unused."""

    def test_run_end_log_includes_holdout_summary_when_holdout_configured(self, tmp_path, caplog):
        caplog.set_level("INFO")
        orch = _make_orch(tmp_path, holdout_task_ids=["h1"], num_generations=1, num_agents=2, no_memory=True)
        dispatch_log: list = []
        _fake_execute(orch, {(1, "t1"): 1.0, (1, "h1"): 1.0}, dispatch_log)
        orch.run(tasks=[TaskSpec(id="t1"), TaskSpec(id="h1")])
        assert "holdout=" in caplog.text

    def test_run_end_log_omits_holdout_summary_when_unused(self, tmp_path, caplog):
        caplog.set_level("INFO")
        orch = _make_orch(tmp_path, num_generations=1, no_memory=True)
        dispatch_log: list = []
        _fake_execute(orch, {(1, "t1"): 1.0}, dispatch_log)
        orch.run(tasks=[TaskSpec(id="t1")])
        assert "holdout=" not in caplog.text


class TestHoldoutSeederBundleExclusion:
    """A stale per-task distillation bundle keyed by a hold-out id (e.g. a
    repurposed/resumed DB where the task used to be a training task) must not
    be injected into the hold-out agent at seed time."""

    @staticmethod
    def _record_bundle(orch, task_id: str) -> None:
        orch._knowledge.record_distillation(
            task_id=task_id,
            generation=1,
            bundle={"transferable_insights": ["stale"], "evidence_post_ids": []},
            scope="per_task",
            experiment=orch.config.experiment_name,
        )

    def test_seeder_skips_per_task_bundle_for_holdout_labels(self, tmp_path):
        from ksi.seeding.seeder import PopulationSeeder

        orch = _make_orch(tmp_path)
        self._record_bundle(orch, "h1")
        self._record_bundle(orch, "t1")
        agents = PopulationSeeder().seed(
            num_agents=2,
            task_labels=["t1", "h1"],
            knowledge_store=orch._knowledge,
            generation=1,
            experiment=orch.config.experiment_name,
            skip_per_task_labels={"h1"},
        )
        assert "per_task_bundle" in agents[0].seed_package  # t1 keeps its bundle
        assert "per_task_bundle" not in agents[1].seed_package  # h1 stays clean

    def test_seed_phase_passes_holdout_ids_to_seeder(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path, holdout_task_ids=["h1"], num_agents=2)
        orch._pending_next_task_labels = ["t1", "h1"]
        captured: dict = {}

        def fake_seed(**kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr(orch._seeder, "seed", fake_seed)
        seed_next_generation(orch, 1, next_task_pool_size=2)
        assert "h1" in (captured.get("skip_per_task_labels") or set())

    def test_external_per_task_bundle_not_attached_to_holdout(self, tmp_path):
        orch = _make_orch(tmp_path, holdout_task_ids=["h1"])
        orch._external_per_task_bundles = {
            "h1": {"transferable_insights": ["external"]},
            "t1": {"transferable_insights": ["external"]},
        }
        pkg_holdout = TestHoldoutSeedPackageIsolation._enrich_single_agent(orch, "h1")
        assert "per_task_bundle" not in pkg_holdout
        pkg_training = TestHoldoutSeedPackageIsolation._enrich_single_agent(orch, "t1")
        assert "per_task_bundle" in pkg_training

"""Stress test for task-level parallel execution and two-pool pipeline.

Verifies:
1. Single agent with many tasks runs them concurrently (not sequentially)
2. Two-pool pipeline overlaps agent work with evaluation
3. All traces are collected correctly
4. --max-concurrent-tasks controls actual parallelism
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from kcsi.errors import ContainerRegistryError
from kcsi.models import AgentState, GenerationConfig, TaskSpec, TaskTrace
from kcsi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from kcsi.runtime import RuntimeResult
from kcsi.tokens import TokenUsage
from tests.orchestrator_phase_helpers import execute_generation


# ---------------------------------------------------------------------------
# Mock runtime: sleeps to simulate agent work, tracks concurrency
# ---------------------------------------------------------------------------
@dataclass
class ConcurrencyTracker:
    """Thread-safe tracker for peak concurrent executions."""

    current: int = 0
    peak: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    completed: int = 0

    def enter(self) -> None:
        with self.lock:
            self.current += 1
            if self.current > self.peak:
                self.peak = self.current

    def exit(self) -> None:
        with self.lock:
            self.current -= 1
            self.completed += 1


class SlowMockRuntime:
    """Simulates agent work with a configurable sleep."""

    def __init__(self, sleep_sec: float, tracker: ConcurrencyTracker):
        self.sleep_sec = sleep_sec
        self.tracker = tracker

    def run_task(self, *, generation: int, agent_id: str, task: TaskSpec, **kwargs) -> RuntimeResult:
        self.tracker.enter()
        try:
            time.sleep(self.sleep_sec)
            return RuntimeResult(
                output=f"patch for {task.id}",
                tool_trace=[],
                runtime_meta={},
                token_usage=TokenUsage(input_tokens=100, output_tokens=50),
            )
        finally:
            self.tracker.exit()


class FlakyMockRuntime:
    """Fails a configurable number of times before succeeding."""

    def __init__(self, failures_before_success: int, message: str):
        self.failures_before_success = failures_before_success
        self.message = message
        self.calls = 0

    def run_task(self, *, generation: int, agent_id: str, task: TaskSpec, **kwargs) -> RuntimeResult:
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise RuntimeError(self.message)
        return RuntimeResult(
            output=f"patch for {task.id}",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        )


class RegistryMockRuntime:
    """Task-aware registry failure runtime for retry and systemic-failure tests."""

    def __init__(
        self,
        *,
        retryable: bool,
        failures_before_success: int | None = None,
        healthy_task_ids: set[str] | None = None,
    ) -> None:
        self.retryable = retryable
        self.failures_before_success = failures_before_success
        self.healthy_task_ids = set(healthy_task_ids or set())
        self.calls: dict[str, int] = {}

    def run_task(self, *, generation: int, agent_id: str, task: TaskSpec, **kwargs) -> RuntimeResult:
        calls = self.calls.get(task.id, 0) + 1
        self.calls[task.id] = calls
        if task.id in self.healthy_task_ids or (
            self.failures_before_success is not None and calls > self.failures_before_success
        ):
            return RuntimeResult(
                output=f"patch for {task.id}",
                tool_trace=[],
                runtime_meta={},
                token_usage=TokenUsage(input_tokens=100, output_tokens=50),
            )
        message = "unauthorized: authentication required" if self.retryable else "manifest unknown"
        raise ContainerRegistryError(
            message,
            retryable=self.retryable,
            reason="transient" if self.retryable else "non_transient",
            image=f"registry.example/{task.id}:latest",
        )


class RecordingPersistence(NoopPersistence):
    def __init__(self) -> None:
        self.traces: list[TaskTrace] = []

    def on_task_trace(self, trace: TaskTrace) -> None:
        self.traces.append(trace)


class SlowMockEvaluator:
    """Simulates evaluation with a configurable sleep."""

    def __init__(self, sleep_sec: float, tracker: ConcurrencyTracker):
        self.sleep_sec = sleep_sec
        self.tracker = tracker

    def evaluate(self, *, task: TaskSpec, model_output: str, **kwargs: Any) -> dict[str, Any]:
        self.tracker.enter()
        try:
            time.sleep(self.sleep_sec)
            return {"status": "resolved", "resolved": True}
        finally:
            self.tracker.exit()


class StubLLM:
    """Returns empty JSON for any LLM call (claim phase, insights, etc.)."""

    def query(self, prompt: str, **kwargs) -> str:
        return '{"claimed_tasks": [], "proposed_workstreams": [], "insights": []}'


def _make_tasks(n: int) -> list[TaskSpec]:
    return [TaskSpec(id=f"task-{i}", repo=f"repo-{i % 3}", prompt=f"Fix bug {i}", metadata={}) for i in range(n)]


def _build_orchestrator(
    *,
    num_agents: int,
    num_tasks: int,
    max_concurrent_tasks: int,
    agent_sleep: float = 0.05,
    eval_sleep: float = 0.05,
    knowledge_db_path: str | None = None,
) -> tuple[GenerationalOrchestrator, ConcurrencyTracker, ConcurrencyTracker, list[TaskSpec]]:
    agent_tracker = ConcurrencyTracker()
    eval_tracker = ConcurrencyTracker()
    tasks = _make_tasks(num_tasks)

    config_kwargs: dict[str, Any] = dict(
        num_generations=1,
        num_agents=num_agents,
        per_task_forum_rounds=0,
        drop_solved=False,
        max_concurrent_tasks=max_concurrent_tasks,
    )
    if knowledge_db_path:
        # Issue #979 MEDIUM: the rest of this file's fixtures use
        # NoopPersistence, so execute_generation -> record_attempt is never
        # exercised under real concurrency against an on-disk KnowledgeStore.
        # Opting in here (only when the caller asks) turns on the real
        # dual-write path (execution_phase._persist_knowledge_attempt_early,
        # called from the eval ThreadPoolExecutor) without disturbing every
        # other test in this file.
        config_kwargs["knowledge_db_path"] = knowledge_db_path
        config_kwargs["experiment_name"] = "parallel_execution_test"
    config = GenerationConfig(**config_kwargs)

    orch = GenerationalOrchestrator(
        config=config,
        runtime=SlowMockRuntime(agent_sleep, agent_tracker),
        evaluator=SlowMockEvaluator(eval_sleep, eval_tracker),
        llm=StubLLM(),
        persistence=NoopPersistence(),
    )
    # Override auto-created agents to match our test agent IDs
    orch.agents = [AgentState(id=f"agent-{i}", generation=1) for i in range(num_agents)]
    return orch, agent_tracker, eval_tracker, tasks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParallelExecution:
    """Verify task-level parallelism works."""

    def test_single_agent_sequential_with_cap_1(self):
        """With --max-concurrent-tasks 1, tasks run sequentially."""
        orch, agent_tracker, eval_tracker, tasks = _build_orchestrator(
            num_agents=1,
            num_tasks=8,
            max_concurrent_tasks=1,
            agent_sleep=0.05,
            eval_sleep=0.02,
        )
        assigned = {"agent-0": [t.id for t in tasks]}
        traces = execute_generation(orch, 1, tasks, assigned)

        assert len(traces) == 8
        assert agent_tracker.peak == 1
        assert agent_tracker.completed == 8

    def test_single_agent_parallel_with_explicit_cap(self):
        """With --agents 1 --max-concurrent-tasks 4, tasks run in parallel."""
        orch, agent_tracker, eval_tracker, tasks = _build_orchestrator(
            num_agents=1,
            num_tasks=8,
            max_concurrent_tasks=4,
            agent_sleep=0.1,
            eval_sleep=0.02,
        )
        assigned = {"agent-0": [t.id for t in tasks]}
        traces = execute_generation(orch, 1, tasks, assigned)

        assert len(traces) == 8
        # With cap=4 and 0.1s sleep, multiple tasks should overlap
        assert agent_tracker.peak > 1, f"Expected parallel execution, got peak={agent_tracker.peak}"
        assert agent_tracker.peak <= 4
        assert agent_tracker.completed == 8

    def test_multi_agent_parallel(self):
        """With 3 agents and cap=6, all agents' tasks run concurrently."""
        orch, agent_tracker, eval_tracker, tasks = _build_orchestrator(
            num_agents=3,
            num_tasks=9,
            max_concurrent_tasks=6,
            agent_sleep=0.1,
            eval_sleep=0.02,
        )
        # 3 tasks per agent
        assigned = {
            "agent-0": [tasks[i].id for i in range(0, 3)],
            "agent-1": [tasks[i].id for i in range(3, 6)],
            "agent-2": [tasks[i].id for i in range(6, 9)],
        }
        traces = execute_generation(orch, 1, tasks, assigned)

        assert len(traces) == 9
        # With cap=6, at least 3 should run concurrently (one per agent minimum)
        assert agent_tracker.peak >= 3, f"Expected >=3 concurrent, got peak={agent_tracker.peak}"
        assert agent_tracker.completed == 9

    def test_eval_pipeline_overlaps_with_agent(self):
        """Eval pool runs concurrently with agent pool (two-pool pipeline)."""
        orch, agent_tracker, eval_tracker, tasks = _build_orchestrator(
            num_agents=1,
            num_tasks=4,
            max_concurrent_tasks=4,
            agent_sleep=0.1,
            eval_sleep=0.1,
        )
        assigned = {"agent-0": [t.id for t in tasks]}

        start = time.monotonic()
        traces = execute_generation(orch, 1, tasks, assigned)
        elapsed = time.monotonic() - start

        assert len(traces) == 4
        assert eval_tracker.completed == 4
        # If sequential: 4 × (0.1 agent + 0.1 eval) = 0.8s
        # If pipelined: ~0.1 agent + 0.1 eval overlap ≈ 0.2-0.3s
        # Allow generous margin but should be well under sequential time
        assert elapsed < 0.6, f"Pipeline too slow ({elapsed:.2f}s), expected <0.6s"

    def test_all_traces_have_correct_fields(self):
        """Every trace has valid generation, agent_id, task_id, and score."""
        orch, agent_tracker, eval_tracker, tasks = _build_orchestrator(
            num_agents=2,
            num_tasks=6,
            max_concurrent_tasks=4,
            agent_sleep=0.02,
            eval_sleep=0.02,
        )
        assigned = {
            "agent-0": [tasks[i].id for i in range(0, 3)],
            "agent-1": [tasks[i].id for i in range(3, 6)],
        }
        traces = execute_generation(orch, 1, tasks, assigned)
        task_ids = {t.id for t in tasks}

        assert len(traces) == 6
        for trace in traces:
            assert trace.generation == 1
            assert trace.agent_id in {"agent-0", "agent-1"}
            assert trace.task_id in task_ids
            assert trace.native_score is not None
            assert trace.error is None
            assert trace.model_output is not None

    def test_per_task_progress_logged_at_info(self, caplog):
        """Each task emits one start and one done INFO line with id/agent/elapsed/score."""
        import logging

        orch, _agent_tracker, _eval_tracker, tasks = _build_orchestrator(
            num_agents=1,
            num_tasks=2,
            max_concurrent_tasks=2,
            agent_sleep=0.0,
            eval_sleep=0.0,
        )
        assigned = {"agent-0": [t.id for t in tasks]}
        with caplog.at_level(logging.INFO, logger="kcsi"):
            execute_generation(orch, 1, tasks, assigned)

        msgs = [r.getMessage() for r in caplog.records]
        for t in tasks:
            assert any(f"task={t.id} agent=agent-0 start" in m for m in msgs), f"no start line for {t.id}"
            done = [m for m in msgs if f"task={t.id} agent=agent-0 done" in m]
            assert done, f"no done line for {t.id}"
            assert "elapsed=" in done[0] and "score=" in done[0]

    def test_transient_task_failure_retries_within_generation(self, monkeypatch):
        monkeypatch.setattr("kcsi.orchestrator.execution_phase.time.sleep", lambda _delay: None)
        tasks = _make_tasks(1)
        runtime = FlakyMockRuntime(
            failures_before_success=2,
            message="Shared container runner timed out after 1800s for task task-0",
        )
        config = GenerationConfig(
            num_generations=1,
            num_agents=1,
            per_task_forum_rounds=0,
            drop_solved=False,
            max_concurrent_tasks=1,
            max_task_retries=2,
        )
        orch = GenerationalOrchestrator(
            config=config,
            runtime=runtime,
            evaluator=SlowMockEvaluator(0.0, ConcurrencyTracker()),
            llm=StubLLM(),
            persistence=NoopPersistence(),
        )
        orch.agents = [AgentState(id="agent-0", generation=1)]
        traces = execute_generation(orch, 1, tasks, {"agent-0": ["task-0"]})

        assert runtime.calls == 3
        assert len(traces) == 1
        assert traces[0].error is None
        assert traces[0].model_output == "patch for task-0"
        assert traces[0].runtime_meta["retry_attempts"] == 2
        assert len(traces[0].runtime_meta["runtime_attempt_errors"]) == 2
        assert traces[0].runtime_meta["runtime_attempt_errors"][0]["attempt"] == 1

    def test_retryable_registry_failure_retries_with_structured_metadata(self, monkeypatch):
        monkeypatch.setattr("kcsi.orchestrator.execution_phase.time.sleep", lambda _delay: None)
        tasks = _make_tasks(1)
        runtime = RegistryMockRuntime(retryable=True, failures_before_success=2)
        config = GenerationConfig(
            num_generations=1,
            num_agents=1,
            per_task_forum_rounds=0,
            drop_solved=False,
            max_concurrent_tasks=1,
            max_task_retries=2,
        )
        orch = GenerationalOrchestrator(
            config=config,
            runtime=runtime,
            evaluator=SlowMockEvaluator(0.0, ConcurrencyTracker()),
            llm=StubLLM(),
            persistence=NoopPersistence(),
        )
        orch.agents = [AgentState(id="agent-0", generation=1)]

        traces = execute_generation(orch, 1, tasks, {"agent-0": ["task-0"]})

        assert runtime.calls == {"task-0": 3}
        assert traces[0].error is None
        assert traces[0].runtime_meta["retry_attempts"] == 2
        assert [entry["error_origin"] for entry in traces[0].runtime_meta["runtime_attempt_errors"]] == [
            "container_registry",
            "container_registry",
        ]

    def test_all_registry_failures_escalate_after_traces_are_persisted(self):
        tasks = _make_tasks(2)
        runtime = RegistryMockRuntime(retryable=False)
        persistence = RecordingPersistence()
        config = GenerationConfig(
            num_generations=1,
            num_agents=2,
            per_task_forum_rounds=0,
            drop_solved=False,
            max_concurrent_tasks=2,
            max_task_retries=3,
        )
        orch = GenerationalOrchestrator(
            config=config,
            runtime=runtime,
            evaluator=SlowMockEvaluator(0.0, ConcurrencyTracker()),
            llm=StubLLM(),
            persistence=persistence,
        )
        orch.agents = [AgentState(id=f"agent-{idx}", generation=1) for idx in range(2)]
        assigned = {f"agent-{idx}": [tasks[idx].id] for idx in range(2)}

        with pytest.raises(ContainerRegistryError, match="all 2 dispatched tasks") as caught:
            execute_generation(orch, 1, tasks, assigned)

        assert caught.value.reason == "generation_registry_failure"
        assert runtime.calls == {"task-0": 1, "task-1": 1}
        assert {trace.task_id for trace in persistence.traces} == {"task-0", "task-1"}
        assert all(trace.runtime_meta["error_origin"] == "container_registry" for trace in persistence.traces)
        assert all(trace.runtime_meta["retry_attempts"] == 0 for trace in persistence.traces)

    def test_isolated_registry_failure_preserves_healthy_sibling(self):
        tasks = _make_tasks(2)
        runtime = RegistryMockRuntime(retryable=False, healthy_task_ids={"task-1"})
        config = GenerationConfig(
            num_generations=1,
            num_agents=2,
            per_task_forum_rounds=0,
            drop_solved=False,
            max_concurrent_tasks=2,
            max_task_retries=3,
        )
        orch = GenerationalOrchestrator(
            config=config,
            runtime=runtime,
            evaluator=SlowMockEvaluator(0.0, ConcurrencyTracker()),
            llm=StubLLM(),
            persistence=NoopPersistence(),
        )
        orch.agents = [AgentState(id=f"agent-{idx}", generation=1) for idx in range(2)]
        assigned = {f"agent-{idx}": [tasks[idx].id] for idx in range(2)}

        traces = execute_generation(orch, 1, tasks, assigned)
        by_task = {trace.task_id: trace for trace in traces}

        assert runtime.calls == {"task-0": 1, "task-1": 1}
        assert by_task["task-0"].error == "manifest unknown"
        assert by_task["task-0"].runtime_meta["error_origin"] == "container_registry"
        assert by_task["task-1"].error is None
        assert by_task["task-1"].native_score is not None

    def test_retry_exhaustion_preserves_attempt_errors(self, monkeypatch):
        monkeypatch.setattr("kcsi.orchestrator.execution_phase.time.sleep", lambda _delay: None)
        tasks = _make_tasks(1)
        runtime = FlakyMockRuntime(
            failures_before_success=99,
            message="Shared container runner timed out after 1800s for task task-0",
        )
        config = GenerationConfig(
            num_generations=1,
            num_agents=1,
            per_task_forum_rounds=0,
            drop_solved=False,
            max_concurrent_tasks=1,
            max_task_retries=2,
        )
        orch = GenerationalOrchestrator(
            config=config,
            runtime=runtime,
            evaluator=SlowMockEvaluator(0.0, ConcurrencyTracker()),
            llm=StubLLM(),
            persistence=NoopPersistence(),
        )
        orch.agents = [AgentState(id="agent-0", generation=1)]
        traces = execute_generation(orch, 1, tasks, {"agent-0": ["task-0"]})

        assert runtime.calls == 3
        assert len(traces) == 1
        assert traces[0].error is not None
        assert traces[0].runtime_meta["retry_attempts"] == 2
        assert len(traces[0].runtime_meta["runtime_attempt_errors"]) == 3
        assert traces[0].runtime_meta["runtime_attempt_errors"][-1]["attempt"] == 3

    def test_non_retryable_task_failure_does_not_retry(self):
        tasks = _make_tasks(1)
        runtime = FlakyMockRuntime(
            failures_before_success=99,
            message="400 Invalid prompt: your prompt was flagged as potentially violating our usage policy",
        )
        config = GenerationConfig(
            num_generations=1,
            num_agents=1,
            per_task_forum_rounds=0,
            drop_solved=False,
            max_concurrent_tasks=1,
            max_task_retries=3,
        )
        orch = GenerationalOrchestrator(
            config=config,
            runtime=runtime,
            evaluator=SlowMockEvaluator(0.0, ConcurrencyTracker()),
            llm=StubLLM(),
            persistence=NoopPersistence(),
        )
        orch.agents = [AgentState(id="agent-0", generation=1)]
        traces = execute_generation(orch, 1, tasks, {"agent-0": ["task-0"]})

        assert runtime.calls == 1
        assert len(traces) == 1
        assert traces[0].error is not None

    def test_cap_respected(self):
        """Peak concurrency never exceeds --max-concurrent-tasks."""
        orch, agent_tracker, eval_tracker, tasks = _build_orchestrator(
            num_agents=1,
            num_tasks=20,
            max_concurrent_tasks=3,
            agent_sleep=0.08,
            eval_sleep=0.02,
        )
        assigned = {"agent-0": [t.id for t in tasks]}
        traces = execute_generation(orch, 1, tasks, assigned)

        assert len(traces) == 20
        assert agent_tracker.peak <= 3, f"Exceeded cap: peak={agent_tracker.peak}"
        assert agent_tracker.completed == 20

    def test_agent_stage_exception_produces_failed_trace(self):
        """When agent_fut.result() raises, a failed TaskTrace must be emitted."""

        class ExplodingRuntime:
            def run_task(self, **kwargs):
                raise RuntimeError("agent exploded")

        tasks = [TaskSpec(id="t1", prompt="fix bug")]
        config = GenerationConfig(
            num_generations=1,
            num_agents=1,
            per_task_forum_rounds=0,
            max_task_retries=0,
        )
        orch = GenerationalOrchestrator(
            config=config,
            runtime=ExplodingRuntime(),
            evaluator=SlowMockEvaluator(0.0, ConcurrencyTracker()),
            llm=StubLLM(),
            persistence=NoopPersistence(),
        )
        orch.agents = [AgentState(id="agent-0", generation=1)]
        traces = execute_generation(orch, 1, tasks, {"agent-0": ["t1"]})

        assert len(traces) >= 1, "Task was silently dropped — no trace emitted"
        assert traces[0].error is not None, "Trace should carry the error"
        assert "agent exploded" in traces[0].error


class TestParallelExecutionWithRealKnowledgeStore:
    """Issue #979 MEDIUM: every other test in this file uses NoopPersistence,

    so execute_generation -> record_attempt is never exercised at
    max_concurrent_tasks>1 against a real on-disk KnowledgeStore — the full
    dual-write path (execution_phase._persist_knowledge_attempt_early,
    called concurrently from the eval ThreadPoolExecutor) stays untested
    under contention. Drive real concurrent execution against a real sqlite
    KnowledgeStore and verify every attempt lands exactly once — no drops,
    no duplicate rows from a racing writer.
    """

    def test_concurrent_execution_writes_exactly_one_attempt_row_per_task(self, tmp_path):
        db_path = str(tmp_path / "parallel_exec_knowledge.sqlite")
        orch, agent_tracker, eval_tracker, tasks = _build_orchestrator(
            num_agents=4,
            num_tasks=24,
            max_concurrent_tasks=8,
            agent_sleep=0.03,
            eval_sleep=0.02,
            knowledge_db_path=db_path,
        )
        assert orch._knowledge is not None, "knowledge_db_path was set but no KnowledgeStore was created"
        try:
            assigned: dict[str, list[str]] = {f"agent-{i}": [] for i in range(4)}
            for i, task in enumerate(tasks):
                assigned[f"agent-{i % 4}"].append(task.id)

            traces = execute_generation(orch, 1, tasks, assigned)

            assert len(traces) == 24
            # Real concurrency actually happened — otherwise this test would
            # pass trivially even with a broken/removed thread pool.
            assert agent_tracker.peak > 1, f"expected real concurrency, got peak={agent_tracker.peak}"
            assert eval_tracker.peak > 1, f"expected real eval concurrency, got peak={eval_tracker.peak}"

            # This tmp_path DB holds exactly one experiment/run, so a bare
            # generation filter (no join through runs.experiment) is enough.
            row = orch._knowledge._execute(
                "SELECT COUNT(*) AS n FROM attempts WHERE generation = 1",
                fetchone=True,
            )
            assert row is not None and row["n"] == 24, (
                f"expected exactly 24 attempt rows (one per task, no drops/duplicates "
                f"from the concurrent early+late KnowledgeStore writes racing each "
                f"other), got {row['n'] if row else 'none'}"
            )
            # Every task_id must be represented exactly once — a lost-write
            # bug and a duplicate-write bug can otherwise cancel out in the
            # bare COUNT(*) above.
            id_rows = orch._knowledge._execute(
                "SELECT task_id, COUNT(*) AS n FROM attempts WHERE generation = 1 GROUP BY task_id",
                fetchall=True,
            )
            counts_by_task = {r["task_id"]: r["n"] for r in id_rows}
            assert set(counts_by_task) == {t.id for t in tasks}
            assert all(n == 1 for n in counts_by_task.values()), (
                f"expected exactly one attempt row per task, got {counts_by_task}"
            )
        finally:
            orch._knowledge.close()

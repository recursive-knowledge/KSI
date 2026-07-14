"""Engine-level error-path handling for authoritative vs. best-effort writes.

Covers the deferred-abort behavior added in PR #856. The Stage-2 collection
loop distinguishes two write paths:

  * The *authoritative* KnowledgeStore write (``_persist_task_memory_record``).
    A failure here is fatal, but it must NOT break the collection loop
    mid-flight — that would discard already-collected in-flight eval results
    that already cost API tokens + container time. The first failure is
    captured and re-raised exactly once, AFTER every in-flight result has been
    collected and every other trace has attempted its own persist.

  * The *best-effort* runtime-DB sidecar (``on_task_trace`` via
    ``_safe_on_task_trace``). A failure here must never abort the generation;
    it is swallowed and logged at WARNING per the sidecar's documented
    best-effort contract.

These exercise the execution phase service end-to-end with stub
runtime/evaluator/LLM so the control flow (not the SQLite layer) is under test.
The persistence-layer retry/swallow behavior is covered separately by
``test_persistence_trace_guard.py``.
"""

from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock

import pytest

from kcsi.errors import AuthenticationFailure
from kcsi.models import AgentState, GenerationConfig, TaskSpec, TaskTrace
from kcsi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from kcsi.runtime import RuntimeResult
from kcsi.tokens import TokenUsage
from tests.orchestrator_phase_helpers import execute_generation


class _OkRuntime:
    """Runtime that always succeeds; counts completed runs (thread-safe)."""

    def __init__(self) -> None:
        self.completed = 0
        self._lock = threading.Lock()

    def run_task(self, *, generation: int, agent_id: str, task: TaskSpec, **kwargs) -> RuntimeResult:
        with self._lock:
            self.completed += 1
        return RuntimeResult(
            output=f"patch for {task.id}",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )


class _OkEvaluator:
    def evaluate(self, *, task: TaskSpec, model_output: str, **kwargs) -> dict:
        return {"status": "resolved", "resolved": True}


class _StubLLM:
    def query(self, prompt: str, **kwargs) -> str:
        return '{"claimed_tasks": [], "proposed_workstreams": [], "insights": []}'


class _SidecarFailPersistence(NoopPersistence):
    """Best-effort sidecar whose ``on_task_trace`` always fails."""

    def on_task_trace(self, trace) -> None:
        raise RuntimeError("sidecar DB unavailable")


def _make_tasks(n: int) -> list[TaskSpec]:
    return [TaskSpec(id=f"task-{i}", repo="repo", prompt=f"Fix bug {i}", metadata={}) for i in range(n)]


def _build_orchestrator(
    *,
    persistence=None,
    num_tasks: int = 6,
    max_concurrent_tasks: int = 3,
) -> tuple[GenerationalOrchestrator, _OkRuntime, list[TaskSpec], dict[str, list[str]]]:
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        per_task_forum_rounds=0,
        drop_solved=False,
        max_concurrent_tasks=max_concurrent_tasks,
    )
    runtime = _OkRuntime()
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=_OkEvaluator(),
        llm=_StubLLM(),
        persistence=persistence or NoopPersistence(),
    )
    orch.agents = [AgentState(id="agent-0", generation=1)]
    tasks = _make_tasks(num_tasks)
    assigned = {"agent-0": [t.id for t in tasks]}
    return orch, runtime, tasks, assigned


def test_authoritative_persist_failure_defers_abort_until_all_collected():
    """An authoritative-store failure aborts the generation — but only after
    every in-flight result is collected and every trace attempts its persist."""
    orch, runtime, tasks, assigned = _build_orchestrator(num_tasks=6, max_concurrent_tasks=3)

    attempted: list[str] = []
    lock = threading.Lock()

    def _boom(*, trace, insight, lessons, agent=None):
        with lock:
            attempted.append(trace.task_id)
        raise RuntimeError("knowledge store down")

    orch._persist_task_memory_record = _boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError) as ei:
        execute_generation(orch, 1, tasks, assigned)

    # A single deferred error is raised, chained from the first injected failure.
    assert "authoritative KnowledgeStore persist failed during generation" in str(ei.value)
    assert isinstance(ei.value.__cause__, RuntimeError)
    assert "knowledge store down" in str(ei.value.__cause__)

    # No mid-loop abort: every container ran to completion AND every collected
    # trace attempted its authoritative persist. The pre-#856 bare-raise broke
    # the loop at the first failure, so neither count would reach 6.
    assert runtime.completed == 6
    assert sorted(attempted) == [t.id for t in tasks]


def test_sidecar_on_task_trace_failure_does_not_abort_generation(caplog):
    """A best-effort sidecar (``on_task_trace``) failure is swallowed-and-warned
    and never aborts the collection loop."""
    orch, runtime, tasks, assigned = _build_orchestrator(
        persistence=_SidecarFailPersistence(), num_tasks=4, max_concurrent_tasks=2
    )

    with caplog.at_level(logging.WARNING):
        traces = execute_generation(orch, 1, tasks, assigned)

    # Generation completes normally; every task is collected despite the
    # sidecar raising on every write.
    assert len(traces) == 4
    assert runtime.completed == 4

    # The failure surfaced as a WARNING from _safe_on_task_trace, not a raise.
    sidecar_warnings = [r for r in caplog.records if "on_task_trace sidecar write failed" in r.message]
    assert sidecar_warnings, "expected a swallow-and-warn from _safe_on_task_trace"


def test_sidecar_on_task_trace_auth_failure_is_fatal():
    """Authentication failures are credential/configuration errors, not
    best-effort runtime-DB outages."""
    orch, _runtime, _tasks, _assigned = _build_orchestrator(persistence=NoopPersistence(), num_tasks=1)
    orch.persistence.on_task_trace = MagicMock(side_effect=AuthenticationFailure("invalid token"))  # type: ignore[method-assign]

    trace = TaskTrace(agent_id="agent-0", generation=1, task_id="task-0", repo="repo")

    with pytest.raises(AuthenticationFailure, match="invalid token"):
        orch._safe_on_task_trace(trace)

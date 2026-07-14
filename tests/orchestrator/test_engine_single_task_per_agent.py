"""Regression guard: a generation must never assign >1 task to one agent.

``forum_phase.py``'s cross-task ``phase1_by_agent`` context is a dict keyed by
agent_id, built with a plain per-trace overwrite (no list/append) -- it would
silently keep only the last task's data if this invariant were ever violated.
``_assert_single_task_per_agent`` is the fail-loud guard, called right after
``assigned_map`` is built in ``GenerationalOrchestrator.run()``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from conftest import _build_make_tasks, _build_mock_evaluator, _build_mock_llm, _build_mock_runtime

from kcsi.models import Assignment, GenerationConfig
from kcsi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence, _assert_single_task_per_agent


def test_single_task_per_agent_is_allowed():
    # One task per agent (the normal case) must not raise.
    _assert_single_task_per_agent({"agent-1": ["task-a"], "agent-2": ["task-b"]}, generation=0)


def test_agent_with_no_tasks_is_allowed():
    _assert_single_task_per_agent({"agent-1": []}, generation=0)


def test_agent_with_multiple_tasks_raises():
    with pytest.raises(RuntimeError, match="agent-1"):
        _assert_single_task_per_agent({"agent-1": ["task-a", "task-b"], "agent-2": ["task-c"]}, generation=3)


def test_error_message_includes_generation_number():
    with pytest.raises(RuntimeError, match=r"gen 3"):
        _assert_single_task_per_agent({"agent-1": ["task-a", "task-b"]}, generation=3)


def test_run_wires_the_assertion_with_the_real_assigned_map(monkeypatch):
    # The unit tests above call the guard directly; this proves the call site
    # at engine.py's `run()` actually invokes it with `assigned_map` built
    # from the real claim-phase output, not just that the function itself
    # is correct in isolation.
    spy = MagicMock(wraps=_assert_single_task_per_agent)
    monkeypatch.setattr("kcsi.orchestrator.engine._assert_single_task_per_agent", spy)

    config = GenerationConfig(num_generations=1, num_agents=1)
    tasks = _build_make_tasks(1)
    orch = GenerationalOrchestrator(
        config=config,
        runtime=_build_mock_runtime(),
        evaluator=_build_mock_evaluator(),
        llm=_build_mock_llm(),
        persistence=NoopPersistence(),
    )
    orch.run(tasks)

    spy.assert_called_once()
    assigned_map, kwargs = spy.call_args
    assert kwargs == {"generation": 1}
    assert dict(assigned_map[0]) == {"agent-0": ["task-0"]}


def test_run_raises_when_claim_phase_violates_the_invariant(monkeypatch):
    # If the claim phase ever regresses to handing one agent two tasks, `run()`
    # must fail loud instead of silently corrupting the forum's per-agent
    # cross-task context.
    config = GenerationConfig(num_generations=1, num_agents=1)
    tasks = _build_make_tasks(2)
    orch = GenerationalOrchestrator(
        config=config,
        runtime=_build_mock_runtime(),
        evaluator=_build_mock_evaluator(),
        llm=_build_mock_llm(),
        persistence=NoopPersistence(),
    )
    monkeypatch.setattr(
        orch._claim_phase,
        "claim",
        lambda generation, gen_tasks: [
            Assignment(generation=generation, agent_id="agent-0", task_id=t.id) for t in gen_tasks
        ],
    )

    with pytest.raises(RuntimeError, match="agent-0"):
        orch.run(tasks)

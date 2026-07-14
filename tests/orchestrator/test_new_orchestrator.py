"""Tests for the orchestrator engine (simplified — no shrink/chaos modes)."""

import json
from unittest.mock import MagicMock

from conftest import _build_mock_evaluator, _build_mock_llm, _build_mock_runtime

from kcsi.models import AgentState, GenerationConfig, Insight, TaskSpec, TaskTrace
from kcsi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from kcsi.runtime.types import RuntimeResult
from kcsi.tokens import LLMResponse, TokenUsage


def test_single_generation_runs(make_tasks, mock_runtime, mock_evaluator, mock_llm):
    config = GenerationConfig(num_generations=1, num_agents=1)
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
    traces = orch.run(tasks)
    assert len(traces) >= 1
    assert runtime.run_task.called
    assert evaluator.evaluate.called


def test_drop_solved_tasks(make_tasks, mock_runtime, mock_evaluator):
    config = GenerationConfig(
        num_generations=2,
        num_agents=1,
        drop_solved=True,
        solved_threshold=1.0,
    )
    tasks = make_tasks(2)
    runtime = mock_runtime()
    evaluator = mock_evaluator()  # all tasks score 1.0
    llm = MagicMock()

    def _llm_drop_side_effect(system, user, **kwargs):
        u = user
        if "transferable insight" in u.lower():
            return json.dumps({"text": "lesson", "workstream": "fixing", "confidence": "medium"}), TokenUsage(50, 10)
        else:
            return json.dumps(
                {
                    "proposals": [],
                    "workstream_claim": "fixing",
                }
            ), TokenUsage(80, 40)

    def _llm_drop_response_side_effect(system, user, **kwargs):
        text, usage = _llm_drop_side_effect(system, user, **kwargs)
        return LLMResponse(text=text, usage=usage)

    llm.call.side_effect = _llm_drop_response_side_effect

    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )
    traces = orch.run(tasks)
    # Gen 1 should execute tasks, gen 2 should have no tasks (all solved)
    assert len(traces) == len(tasks)
    assert runtime.run_task.call_count == len(tasks)
    assert all(trace.generation == 1 for trace in traces)


def test_preserves_best_trace_when_drop_solved_is_disabled(mock_llm):
    config = GenerationConfig(
        num_generations=2,
        num_agents=1,
        per_task_forum_rounds=0,
        drop_solved=False,
        solved_threshold=1.0,
        max_concurrent_tasks=1,
    )
    config.cross_task_forum_rounds = 0
    config.distill_enabled = False
    task = TaskSpec(
        id="arc-task-1",
        repo="",
        prompt="Solve ARC task",
        metadata={"task_source": "arc"},
    )
    runtime = MagicMock()
    runtime.run_task.side_effect = [
        RuntimeResult(
            output="solved-grid",
            tool_trace=[],
            runtime_meta={"native_session_memory": "gen1", "session_scope": "task"},
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        ),
        RuntimeResult(
            output="worse-grid",
            tool_trace=[],
            runtime_meta={"native_session_memory": "gen2", "session_scope": "task"},
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        ),
    ]
    evaluator = MagicMock()

    def _eval_side_effect(*, task, model_output, runtime_meta, tool_trace):
        if model_output == "solved-grid":
            return {"resolved": True, "native_score": 1.0, "task_type": "arc"}
        return {"resolved": False, "native_score": 0.0, "task_type": "arc"}

    evaluator.evaluate.side_effect = _eval_side_effect

    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )
    traces = orch.run([task])

    assert runtime.run_task.call_count == 1
    assert evaluator.evaluate.call_count == 1
    assert len(traces) == 2
    gen1 = next(trace for trace in traces if trace.generation == 1)
    gen2 = next(trace for trace in traces if trace.generation == 2)
    assert gen1.native_score == 1.0
    assert gen2.native_score == 1.0
    assert gen2.model_output == "solved-grid"
    assert gen2.runtime_meta["carry_forward"] is True
    assert gen2.runtime_meta["carry_forward_source_generation"] == 1
    assert gen2.runtime_meta["carry_forward_threshold"] == 1.0
    assert orch.agents[0].tasks_completed == 2


def test_no_tasks_produces_empty_traces(mock_runtime, mock_evaluator, mock_llm):
    config = GenerationConfig(num_generations=1, num_agents=2)
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
    traces = orch.run([])
    assert traces == []


def test_token_accumulator_populated(make_tasks, mock_runtime, mock_evaluator, mock_llm):
    config = GenerationConfig(num_generations=1, num_agents=1)
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
    orch.run(tasks)
    total = orch.accumulator.total()
    assert total.total > 0


def test_persistence_callbacks_called(make_tasks, mock_runtime, mock_evaluator, mock_llm):
    config = GenerationConfig(num_generations=1, num_agents=1)
    tasks = make_tasks(1)
    runtime = mock_runtime()
    evaluator = mock_evaluator()
    llm = mock_llm()
    persistence = MagicMock()

    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=persistence,
    )
    orch.run(tasks)
    assert persistence.on_generation_start.called
    assert persistence.on_assignment.called
    assert persistence.on_task_trace.called
    assert persistence.on_generation_end.called
    assert persistence.on_run_end.called


def _make_orch(llm=None):
    config = GenerationConfig(num_generations=1, num_agents=1)
    return GenerationalOrchestrator(
        config=config,
        runtime=_build_mock_runtime(),
        evaluator=_build_mock_evaluator(),
        llm=llm or _build_mock_llm(),
        persistence=NoopPersistence(),
    )


def test_generate_task_insight_returns_insight():
    orch = _make_orch()
    agent = AgentState(id="a0", workstream="django-orm")
    trace = TaskTrace(
        generation=1,
        agent_id="a0",
        task_id="t1",
        model_output="Applied patch",
        eval_result={"resolved": True},
        native_score=1.0,
        tool_trace=[],
        runtime_meta={},
        token_usage=TokenUsage(),
    )
    task = TaskSpec(id="t1", repo="django/django", prompt="Fix ORM caching bug")
    insight, _lessons, tokens = orch._execution_phase._generate_reflection_and_lessons(
        generation=1, agent=agent, trace=trace, task=task
    )
    assert isinstance(insight, Insight)
    assert insight.text == "Always verify cache invalidation after ORM changes"
    assert insight.workstream == "django-orm"
    assert insight.confidence == "high"
    assert insight.source_task_id == "t1"
    assert insight.author_agent_id == "a0"
    assert isinstance(tokens, int)


def test_generate_task_insight_skips_errored_traces():
    orch = _make_orch()
    agent = AgentState(id="a0")
    trace = TaskTrace(
        generation=1,
        agent_id="a0",
        task_id="t1",
        model_output=None,
        eval_result={},
        native_score=None,
        tool_trace=[],
        runtime_meta={},
        token_usage=TokenUsage(),
        error="container timeout",
    )
    insight, _lessons, tokens = orch._execution_phase._generate_reflection_and_lessons(
        generation=1, agent=agent, trace=trace, task=None
    )
    assert insight is None
    assert tokens == 0


def test_generate_reflection_skips_empty_output_without_llm_call():
    """A non-errored trace with empty/whitespace model_output has nothing to
    reflect on: the merged reflection+lessons call must be skipped entirely
    (no LLM call, no hallucinated lessons from an empty excerpt), matching the
    former two-call path's lesson-extraction guard (issue #1252 item 4)."""
    llm = MagicMock()
    orch = _make_orch(llm=llm)
    agent = AgentState(id="a0")
    trace = TaskTrace(
        generation=1,
        agent_id="a0",
        task_id="t1",
        model_output="   ",
        eval_result={"resolved": False},
        native_score=0.0,
        tool_trace=[],
        runtime_meta={},
        token_usage=TokenUsage(),
    )
    insight, lessons, tokens = orch._execution_phase._generate_reflection_and_lessons(
        generation=1, agent=agent, trace=trace, task=None
    )
    assert insight is None
    assert lessons == []
    assert tokens == 0
    llm.call.assert_not_called()


def test_generate_task_insight_handles_llm_failure():
    llm = MagicMock()
    llm.call.side_effect = RuntimeError("API down")
    orch = _make_orch(llm=llm)
    agent = AgentState(id="a0")
    trace = TaskTrace(
        generation=1,
        agent_id="a0",
        task_id="t1",
        model_output="output",
        eval_result={"resolved": False},
        native_score=0.0,
        tool_trace=[],
        runtime_meta={},
        token_usage=TokenUsage(),
    )
    insight, _lessons, tokens = orch._execution_phase._generate_reflection_and_lessons(
        generation=1, agent=agent, trace=trace, task=None
    )
    assert insight is None
    assert tokens == 0


def test_generate_task_insight_handles_empty_parse():
    llm = MagicMock()
    llm.call.return_value = LLMResponse(text='{"text": ""}', usage=TokenUsage(input_tokens=10, output_tokens=5))
    orch = _make_orch(llm=llm)
    agent = AgentState(id="a0")
    trace = TaskTrace(
        generation=1,
        agent_id="a0",
        task_id="t1",
        model_output="output",
        eval_result={},
        native_score=0.5,
        tool_trace=[],
        runtime_meta={},
        token_usage=TokenUsage(),
    )
    insight, _lessons, tokens = orch._execution_phase._generate_reflection_and_lessons(
        generation=1, agent=agent, trace=trace, task=None
    )
    assert insight is None

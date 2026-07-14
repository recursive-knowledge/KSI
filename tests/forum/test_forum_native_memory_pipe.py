"""Tests that Phase-1 native memory flows into the per-task forum prompt."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from ksi.forum import build_per_task_discussion_parts
from ksi.models import GenerationConfig, TaskTrace
from ksi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from ksi.runtime.types import RuntimeResult
from ksi.tokens import LLMResponse, TokenUsage
from tests.orchestrator_phase_helpers import per_task_forum

# ---------------------------------------------------------------------------
# Prompt-level tests
# ---------------------------------------------------------------------------


def test_native_memory_included_in_prompt():
    p = build_per_task_discussion_parts(
        agent_id="a1",
        generation=1,
        traces=[],
        task_ids=["t1"],
        task_descriptions={"t1": "desc"},
        native_memory="NOTES: I noticed X failed because of Y",
    ).as_text()
    assert "NOTES: I noticed X failed" in p
    assert "native memory" in p.lower() or "notes from" in p.lower()


def test_native_memory_none_omits_section():
    p = build_per_task_discussion_parts(
        agent_id="a1",
        generation=1,
        traces=[],
        task_ids=["t1"],
        task_descriptions={"t1": "desc"},
    ).as_text()
    assert "native memory" not in p.lower()


def test_native_memory_empty_string_omits_section():
    """Empty / whitespace-only native_memory must not emit the section."""
    p = build_per_task_discussion_parts(
        agent_id="a1",
        generation=1,
        traces=[],
        task_ids=["t1"],
        native_memory="   \n  ",
    ).as_text()
    assert "native memory" not in p.lower()


def test_native_memory_truncated_at_inject_cap():
    """Cap native memory render at the module-level inject constant.

    Audit 2026-04-20 raised the cap from 8000 → 32000 so that users who
    set --native-memory-max-chars beyond 8k actually see the effect in
    forum prompts. This test pins to the module constant so future
    changes to the cap have to update the constant explicitly.
    """
    from ksi.forum.prompt import _NATIVE_MEMORY_FORUM_INJECT_CHARS

    cap = _NATIVE_MEMORY_FORUM_INJECT_CHARS
    big = "x" * (cap + 5000)
    p = build_per_task_discussion_parts(
        agent_id="a1",
        generation=1,
        traces=[],
        task_ids=["t1"],
        native_memory=big,
    ).as_text()
    # The truncated x-run should be exactly `cap` characters.
    assert "x" * (cap + 1) not in p
    assert "x" * cap in p


# ---------------------------------------------------------------------------
# Engine-level test: verify runtime_meta["native_session_memory"] from
# Phase-1 traces reaches the prompt passed to runtime.run_task() for the
# per-task forum phase.
# ---------------------------------------------------------------------------


def test_per_task_forum_phase_pipes_native_memory_into_prompt(tmp_path):
    """Phase 1's trace.runtime_meta['native_session_memory'] must end up in
    the Phase 2 forum prompt."""
    db_path = str(tmp_path / "knowledge.sqlite")

    runtime = MagicMock()

    captured: dict[str, str] = {}

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        prompt = getattr(task, "prompt", "") or ""
        metadata = getattr(task, "metadata", {}) or {}
        if metadata.get("task_source") == "per_task_forum":
            captured[agent_id] = prompt
        return RuntimeResult(
            output="discussion",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=5, output_tokens=5),
        )

    runtime.run_task.side_effect = fake_run_task

    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0}
    llm = MagicMock()
    llm.call.return_value = LLMResponse(
        text=json.dumps({"claimed_tasks": ["task-alpha"]}),
        usage=TokenUsage(input_tokens=10, output_tokens=5),
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

    # Phase-1 trace with native session memory populated in runtime_meta.
    # Include a second agent trace on the same task so the per-task forum
    # sees >1 agent per task and actually runs (the 1:1 task→agent monologue
    # is now short-circuited upstream).
    trace = TaskTrace(
        generation=1,
        agent_id="agent-0",
        task_id="task-alpha",
        model_output="patch",
        eval_result={"resolved": True},
        native_score=1.0,
        tool_trace=[],
        runtime_meta={"native_session_memory": "MARKER_P1_MEMORY_STRING"},
        token_usage=TokenUsage(input_tokens=5, output_tokens=5),
    )
    trace_peer = TaskTrace(
        generation=1,
        agent_id="agent-1",
        task_id="task-alpha",
        model_output="patch",
        eval_result={"resolved": True},
        native_score=1.0,
        tool_trace=[],
        runtime_meta={},
        token_usage=TokenUsage(input_tokens=5, output_tokens=5),
    )

    per_task_forum(orch, 1, [trace, trace_peer])

    # Expect the Phase-1 memory marker in the Phase-2 prompt.
    assert "agent-0" in captured, f"no forum task captured for agent-0: {captured!r}"
    prompt = captured["agent-0"]
    assert "MARKER_P1_MEMORY_STRING" in prompt
    assert "native memory" in prompt.lower() or "notes from" in prompt.lower()


# Note: tests `test_per_task_forum_phase_skips_single_agent_monologue` and
# `test_per_task_forum_phase_synthesizes_posts_when_skipping` were removed in
# the V2 design. The monologue-skip optimization (PR #483) was reverted: V2
# Phase 2 always runs even with 1:1 task→agent assignment, because the
# structured single-agent post-mortem is now load-bearing for the V2 contrast
# pipeline. Synthesis is retained only as the error-recovery fallback.
# See `test_per_task_forum_always_runs_in_v2` below.


def test_per_task_forum_gate_recomputed_per_generation(tmp_path):
    """Gate predicate is computed per-generation from that gen's traces.

    PR #483 review (must-fix #2): with gen-1 assigned 1:1 (1 agent/task) the
    gate fires and the phase is skipped; gen-2 sees 2 agents on the same task
    (retries/seeding have produced a peer trace) and the gate must NOT fire —
    the phase must run as a genuine multi-agent discussion.

    This test runs both generations against the same orchestrator and
    verifies that (a) runtime.run_task was never invoked in gen 1 and (b)
    runtime.run_task WAS invoked in gen 2 despite sharing orchestrator
    state. Catches regressions where the gate caches across generations.
    """
    db_path = str(tmp_path / "knowledge.sqlite")

    runtime = MagicMock()
    run_task_calls: list[dict] = []

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        run_task_calls.append({"generation": generation, "agent_id": agent_id, "task": task})
        return RuntimeResult(
            output="discussion-turn",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    runtime.run_task.side_effect = fake_run_task

    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0}
    llm = MagicMock()
    llm.call.return_value = LLMResponse(
        text=json.dumps({"claimed_tasks": ["task-alpha"]}),
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )

    config = GenerationConfig(
        num_generations=2,
        num_agents=3,
        knowledge_db_path=db_path,
        experiment_name="crossgen_gate",
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )

    # Gen 1: 3 agents, 3 tasks, 1:1 assignment — gate must fire, no run_task.
    gen1_traces = [
        TaskTrace(
            generation=1,
            agent_id=f"agent-{i}",
            task_id=f"task-{i}",
            model_output="patch",
            eval_result={"resolved": True},
            native_score=1.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )
        for i in range(3)
    ]

    per_task_forum(orch, 1, gen1_traces)
    gen1_call_count = runtime.run_task.call_count
    # V2: Phase 2 always runs, even with 1:1 task→agent assignment. The
    # monologue-skip gate is removed; agents always emit a structured
    # post-mortem.
    assert gen1_call_count == 3, (
        f"V2 gen-1 must dispatch one Phase-2 task per agent (3 expected); got {gen1_call_count}"
    )

    # Gen 2: 2 agents on the SAME task (retries/seeding collapsed two agents
    # onto one task). Gate predicate must not fire — max_agents_per_task=2.
    gen2_traces = [
        TaskTrace(
            generation=2,
            agent_id="agent-0",
            task_id="task-retry",
            model_output="patch",
            eval_result={"resolved": False},
            native_score=0.0,
            tool_trace=[],
            runtime_meta={"native_session_memory": "GEN2_MEMORY_AGENT0"},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
        TaskTrace(
            generation=2,
            agent_id="agent-1",
            task_id="task-retry",
            model_output="patch",
            eval_result={"resolved": True},
            native_score=1.0,
            tool_trace=[],
            runtime_meta={"native_session_memory": "GEN2_MEMORY_AGENT1"},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
    ]

    per_task_forum(orch, 2, gen2_traces)
    # V2: each generation dispatches one Phase-2 task per agent that produced
    # a trace. Gen-2 dispatched agents-0 and agent-1.
    assert runtime.run_task.call_count > gen1_call_count, (
        "gen-2 forum phase must run runtime.run_task in addition to gen-1"
    )
    # Both gen-1 (3 calls) and gen-2 (2 calls) traces show up across the
    # combined call log; the V2 invariant is that gen-2 calls are present
    # and tagged with generation=2.
    assert 2 in {call["generation"] for call in run_task_calls}, (
        f"gen-2 run_task calls must appear; saw {sorted({call['generation'] for call in run_task_calls})}"
    )
    gen2_agents = {call["agent_id"] for call in run_task_calls if call["generation"] == 2}
    assert gen2_agents == {"agent-0", "agent-1"}, (
        f"gen-2 run_task must dispatch both agents on task-retry, got {gen2_agents}"
    )


def test_per_task_forum_always_runs_in_v2(tmp_path):
    """V2 design: per-task forum phase ALWAYS runs (no monologue-skip gate).

    Previously the phase was auto-skipped when ``max_agents_per_task <= 1``
    (the 1:1 task→agent monologue case) with an escape-hatch flag
    ``per_task_forum_skip_when_monologue=False``. The V2 design removes the
    skip entirely: Phase 2 always runs because it produces structured
    single-agent post-mortems that cross-reference prior-generation posts,
    a channel that the old monologue-skip optimization silently killed.
    """
    db_path = str(tmp_path / "escape_hatch.sqlite")
    runtime = MagicMock()
    run_task_calls: list[dict] = []

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        run_task_calls.append({"agent_id": agent_id, "generation": generation})
        return RuntimeResult(
            output="self-reflect",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    runtime.run_task.side_effect = fake_run_task
    evaluator = MagicMock()
    llm = MagicMock()
    llm.call.return_value = LLMResponse(
        text=json.dumps({"claimed_tasks": ["task-0"]}),
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )

    config = GenerationConfig(
        num_generations=1,
        num_agents=2,
        knowledge_db_path=db_path,
        experiment_name="escape_hatch_test",
    )

    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )

    traces = [
        TaskTrace(
            generation=1,
            agent_id=f"agent-{i}",
            task_id=f"task-{i}",
            model_output="patch",
            eval_result={"resolved": True},
            native_score=1.0,
            tool_trace=[],
            runtime_meta={"native_session_memory": f"notes-{i}"},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )
        for i in range(2)
    ]

    per_task_forum(orch, 1, traces)
    assert runtime.run_task.call_count > 0, (
        "V2 Phase 2 must always run runtime.run_task, even with 1:1 traces (monologue-skip removed)"
    )

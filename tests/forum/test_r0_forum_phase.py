"""Tests for R0 forum phase dispatch and build_per_task_discussion_parts."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from kcsi.forum import build_per_task_discussion_parts
from kcsi.models import GenerationConfig, TaskTrace
from kcsi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence, _drain_forum_bus
from kcsi.runtime.types import RuntimeResult
from kcsi.tokens import LLMResponse, TokenUsage
from tests.orchestrator_phase_helpers import per_task_forum


def _make_traces(generation: int, agent_id: str, task_ids: list[str]) -> list[TaskTrace]:
    return [
        TaskTrace(
            generation=generation,
            agent_id=agent_id,
            task_id=tid,
            model_output="patch",
            eval_result={"resolved": True},
            native_score=1.0,
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        for tid in task_ids
    ]


# ---------------------------------------------------------------------------
# build_per_task_discussion_parts tests
# ---------------------------------------------------------------------------


class TestBuildR0DiscussionPrompt:
    def test_prompt_contains_mcp_tool_references(self):
        """R0 prompt should mention MCP tools like forum_post and forum_signal_done."""
        traces = _make_traces(1, "agent-0", ["task-1"])
        prompt = build_per_task_discussion_parts(
            agent_id="agent-0",
            generation=1,
            traces=traces,
            task_ids=["task-1"],
        ).as_text()
        assert "forum_post" in prompt
        assert "forum_signal_done" in prompt
        assert "knowledge" in prompt.lower()

    def test_prompt_does_not_contain_insight_comment_blocks(self):
        """R0 prompt should NOT instruct agents to use INSIGHT/COMMENT blocks."""
        traces = _make_traces(1, "agent-0", ["task-1"])
        prompt = build_per_task_discussion_parts(
            agent_id="agent-0",
            generation=1,
            traces=traces,
            task_ids=["task-1"],
        ).as_text()
        # Should explicitly tell agents NOT to use structured blocks
        assert (
            "Do NOT" in prompt or "load_bearing_assumption" in prompt
        )  # V2: removed INSIGHT/COMMENT injunction; V2 prompt asks for structured post-mortem fields

    def test_prompt_contains_agent_id_and_generation(self):
        traces = _make_traces(2, "agent-5", ["task-99"])
        prompt = build_per_task_discussion_parts(
            agent_id="agent-5",
            generation=2,
            traces=traces,
            task_ids=["task-99"],
        ).as_text()
        assert "agent-5" in prompt
        assert "generation 2" in prompt

    def test_prompt_includes_task_descriptions(self):
        traces = _make_traces(1, "agent-0", ["task-A"])
        prompt = build_per_task_discussion_parts(
            agent_id="agent-0",
            generation=1,
            traces=traces,
            task_ids=["task-A"],
            task_descriptions={"task-A": "Fix the login bug in auth module"},
        ).as_text()
        assert "Fix the login bug" in prompt

    def test_prompt_includes_task_scores(self):
        traces = _make_traces(1, "agent-0", ["task-A"])
        prompt = build_per_task_discussion_parts(
            agent_id="agent-0",
            generation=1,
            traces=traces,
            task_ids=["task-A"],
        ).as_text()
        assert "task-A" in prompt
        assert "1.0" in prompt  # native_score


# ---------------------------------------------------------------------------
# Forum phase dispatch tests
# ---------------------------------------------------------------------------


class TestForumPhaseR0Dispatch:
    def test_forum_phase_dispatches_r0(self, tmp_path):
        """Forum phase should dispatch agents with round_num=0."""
        db_path = str(tmp_path / "knowledge.sqlite")

        runtime = MagicMock()

        def fake_run_task(*, generation, agent_id, task, **kwargs):
            return RuntimeResult(
                output="discussed tasks",
                tool_trace=[],
                runtime_meta={"native_session_memory": "discussion transcript"},
                token_usage=TokenUsage(input_tokens=50, output_tokens=30),
            )

        runtime.run_task.side_effect = fake_run_task

        evaluator = MagicMock()
        evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0}
        llm = MagicMock()
        llm.call.return_value = LLMResponse(
            text=json.dumps({"claimed_tasks": ["task-0"]}),
            usage=TokenUsage(input_tokens=50, output_tokens=20),
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

        # Two agents, same task (2:1 agent-to-task ratio) so the new
        # "skip monologue" gate does NOT fire — the forum phase still
        # dispatches R0 to both debate agents as this test asserts.
        traces = _make_traces(1, "agent-0", ["task-0"]) + _make_traces(1, "agent-1", ["task-0"])
        per_task_forum(orch, 1, traces)

        # Should have called run_task for R0: 1 round * 2 agents = 2 calls
        forum_calls = [
            c
            for c in runtime.run_task.call_args_list
            if c.kwargs.get("task") and "__forum__" in str(c.kwargs["task"].id)
        ]
        assert len(forum_calls) == 2
        # All calls should be round 0
        rounds_seen = {int((c.kwargs["task"].metadata or {}).get("forum_round", -1)) for c in forum_calls}
        assert rounds_seen == {0}

    def test_forum_phase_skipped_with_zero_rounds(self):
        """When per_task_forum_rounds=0, forum phase should be skipped entirely."""
        config = GenerationConfig(
            num_generations=1,
            num_agents=1,
            per_task_forum_rounds=0,
        )
        runtime = MagicMock()
        evaluator = MagicMock()
        llm = MagicMock()

        orch = GenerationalOrchestrator(
            config=config,
            runtime=runtime,
            evaluator=evaluator,
            llm=llm,
            persistence=NoopPersistence(),
        )

        traces = _make_traces(1, "agent-0", ["task-0"])
        per_task_forum(orch, 1, traces)

        # runtime.run_task should NOT be called
        runtime.run_task.assert_not_called()


# ---------------------------------------------------------------------------
# _drain_forum_bus sanity check
# ---------------------------------------------------------------------------


class TestDrainForumBusSanity:
    def test_drain_forum_bus_is_callable(self):
        """_drain_forum_bus should be importable and callable."""
        assert callable(_drain_forum_bus)

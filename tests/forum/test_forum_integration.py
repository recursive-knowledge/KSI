"""Integration test: forum phase with discussion and distillation."""

import json
from unittest.mock import MagicMock

from kcsi.models import GenerationConfig, TaskSpec
from kcsi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from kcsi.runtime.types import RuntimeResult
from kcsi.tokens import LLMResponse, TokenUsage
from tests.orchestrator_phase_helpers import per_task_forum


def test_full_generation_with_per_task_forum(tmp_path):
    """Full generation: execution insights, MCP discussion, and distillation."""
    db_path = str(tmp_path / "knowledge.sqlite")

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        # Forum round tasks: discussion via MCP tools.
        if task.id.startswith("__forum__"):
            output = "discussed via MCP tools"
            return RuntimeResult(
                output=output,
                tool_trace=[],
                runtime_meta={},
                token_usage=TokenUsage(input_tokens=50, output_tokens=30),
            )
        # Regular tasks
        return RuntimeResult(
            output="<patch>diff</patch>",
            tool_trace=[],
            runtime_meta={"native_session_memory": f"transcript for {task.id}"},
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        )

    runtime = MagicMock()
    runtime.run_task.side_effect = fake_run_task

    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0}

    llm = MagicMock()

    def fake_llm_call(*, system, user, **kwargs):
        # Task-mode distillation bundle.
        return json.dumps(
            {
                "bundle_summary": "Shared task-mode strategy bundle",
                "shared_insight_bundle": [
                    {
                        "insight_id": "asset-1",
                        "text": "Check the failing assertions first.",
                        "source_insight_ids": ["ins-agent-0-1"],
                    }
                ],
            }
        ), TokenUsage(input_tokens=50, output_tokens=20)

    def _llm_response_side_effect(*args, **kwargs):
        text, usage = fake_llm_call(*args, **kwargs)
        return LLMResponse(text=text, usage=usage)

    llm.call.side_effect = _llm_response_side_effect

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

    # Pre-seed insights into KnowledgeStore for distillation.
    if orch._knowledge is not None:
        for agent_id in ["agent-0", "agent-1"]:
            for idx in range(1, 3):
                orch._knowledge.record_insight(
                    task_id=f"task-{idx - 1}",
                    agent_id=agent_id,
                    generation=1,
                    text=f"R0 lesson from {agent_id} #{idx}",
                    scope="task" if idx == 1 else "meta",
                    confidence="medium",
                    evidence_task_ids=[f"task-{idx - 1}"],
                    round_num=0,
                )

    tasks = [
        TaskSpec(id="task-0", repo="r", prompt="Fix bug 0"),
        TaskSpec(id="task-1", repo="r", prompt="Fix bug 1"),
    ]
    orch.run(tasks)

    # In task mode, workstreams are direct task labels.
    workstreams = {a.id: a.workstream for a in orch.agents}
    assert workstreams["agent-0"] == "task-0"
    assert workstreams["agent-1"] == "task-1"


def test_forum_phase_skipped_without_knowledge_db():
    """Without --knowledge-db-path, forum stage should be skipped cleanly."""
    runtime = MagicMock()
    runtime.run_task.return_value = RuntimeResult(
        output=(
            '{"insights": [], "workstream_claim": "general", "proposed_workstreams": ["general"], "task_summaries": []}'
        ),
        tool_trace=[],
        runtime_meta={"native_session_memory": "transcript"},
        token_usage=TokenUsage(input_tokens=50, output_tokens=30),
    )

    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0}

    llm = MagicMock()
    llm.call.return_value = LLMResponse(
        text=json.dumps({"claimed_tasks": ["task-0"]}), usage=TokenUsage(input_tokens=50, output_tokens=20)
    )

    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        knowledge_db_path="",  # No knowledge DB
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )
    tasks = [TaskSpec(id="task-0", repo="r", prompt="Fix bug")]
    # Should not raise.
    orch.run(tasks)


def test_forum_phase_skipped_when_knowledge_none():
    """When _knowledge is None, forum phase should be skipped."""
    runtime = MagicMock()
    runtime.run_task.return_value = RuntimeResult(
        output="output",
        tool_trace=[],
        runtime_meta={},
        token_usage=TokenUsage(input_tokens=50, output_tokens=30),
    )
    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0}
    llm = MagicMock()
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        knowledge_db_path="",
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )
    assert orch._knowledge is None
    # Forum phase should be a no-op
    per_task_forum(orch, 1, [])

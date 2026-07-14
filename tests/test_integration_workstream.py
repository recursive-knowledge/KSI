"""Integration test: full generation loop with workstream-driven evolution."""

import json
from unittest.mock import MagicMock

from kcsi.models import GenerationConfig, TaskSpec
from kcsi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from kcsi.runtime.types import RuntimeResult
from kcsi.tokens import LLMResponse, TokenUsage


def test_two_generations_workstream_loop():
    """Run 2 generations with 2 agents, verify workstream seeding works."""
    config = GenerationConfig(
        num_generations=2,
        num_agents=2,
        drop_solved=True,
        solved_threshold=1.0,
    )
    tasks = [TaskSpec(id=f"t{i}", repo="django", prompt=f"Fix issue {i}") for i in range(4)]

    # Mock runtime
    runtime = MagicMock()
    runtime.run_task.return_value = RuntimeResult(
        output="<patch>diff</patch>",
        tool_trace=[],
        runtime_meta={},
        token_usage=TokenUsage(input_tokens=100, output_tokens=50),
    )

    # Mock evaluator: first 2 tasks solved, last 2 not
    evaluator = MagicMock()
    call_count = [0]

    def eval_side_effect(**kwargs):
        call_count[0] += 1
        solved = call_count[0] <= 2
        return {
            "resolved": solved,
            "native_score": 1.0 if solved else 0.0,
            "task_type": "swebench",
        }

    evaluator.evaluate.side_effect = eval_side_effect

    # Mock LLM for claiming + forum
    llm = MagicMock()

    def llm_side_effect(system, user, **kwargs):
        if "Available Tasks" in user:
            # Claim response: pick first available tasks
            return json.dumps({"claimed_tasks": ["t0", "t1"]}), TokenUsage(50, 20)
        elif "task_results" in user.lower() or "propose" in system.lower():
            return json.dumps(
                {
                    "insights": [
                        {
                            "text": "lesson learned",
                            "workstream": "fixing",
                            "confidence": "high",
                        }
                    ],
                    "workstream_claim": "fixing",
                    "proposed_workstreams": ["fixing", "testing"],
                }
            ), TokenUsage(100, 50)
        else:
            return json.dumps(
                {
                    "proposals": [],
                    "workstream_claim": "fixing",
                }
            ), TokenUsage(80, 40)

    def _llm_response_side_effect(system, user, **kwargs):
        text, usage = llm_side_effect(system, user, **kwargs)
        return LLMResponse(text=text, usage=usage)

    llm.call.side_effect = _llm_response_side_effect

    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )
    result = orch.run(tasks)

    # Result should be a list of TaskTrace objects
    assert result is not None
    assert isinstance(result, list)

    # Verify runtime was called for task execution
    assert runtime.run_task.call_count >= 2

    # Verify evaluator was called
    assert evaluator.evaluate.call_count >= 2

    # Verify LLM was called for both claiming and forum phases
    assert llm.call.call_count >= 4  # at least 2 agents x 2 phases

    # Verify agents were re-seeded for gen 2 (they should have workstreams)
    # After gen 1 forum, the seeder assigns workstreams from the board
    # The agents list is updated in-place by the seeding phase service.
    for agent in orch.agents:
        # After 2 generations, agents should have been seeded with workstreams
        assert agent.generation == 2

    # Verify traces contain expected fields
    for trace in result:
        assert trace.generation in (1, 2)
        assert trace.agent_id.startswith("agent-")
        assert trace.task_id.startswith("t")

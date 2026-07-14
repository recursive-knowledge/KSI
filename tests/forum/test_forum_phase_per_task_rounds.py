"""Engine-level wiring test for #1043: per-task forum round_num + peer posts.

`_per_task_forum_default` must thread `round_num` and same-generation peer
posts from earlier rounds into `build_per_task_discussion_parts`, so
round 1+ of `--per-task-forum-rounds` actually differs from round 0 instead
of reproducing the exact same prompt.

The round-0 peer post is pre-seeded directly into KnowledgeStore (rather
than relying on the real ForumBus dispatch + drain cycle) to isolate this
test from PR #1060's separate fix (drain now happens per-round instead of
once after the whole round loop) — that fix governs whether a *real*
round-0 dispatch's post is visible in time for round 1; this test only
pins that *once visible*, the per-task builder wiring surfaces it.
"""

from pathlib import Path
from unittest.mock import MagicMock

from kcsi.memory.knowledge_store import KnowledgeStore
from kcsi.models import GenerationConfig, TaskTrace
from kcsi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
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


def test_round1_prompt_includes_same_gen_round0_peer_post(tmp_path):
    db_path = str(tmp_path / "knowledge.sqlite")
    knowledge_db = str(Path(db_path).parent / "knowledge.sqlite")

    # Simulate "round 0 already ran and its post was drained into
    # KnowledgeStore" — exactly the state PR #1060's per-round drain fix
    # produces before round 1 dispatches.
    store = KnowledgeStore(knowledge_db)
    store.record_post(
        task_id="task-0",
        agent_id="agent-1",
        generation=1,
        text="ROUND0_PEER_POST_MARKER",
        round_num=0,
        source_phase="per_task_forum",
        # Matches GenerationConfig.experiment_name's default ("kcsi") since
        # the test below doesn't override it.
        experiment="kcsi",
    )
    store.close()

    runtime = MagicMock()
    captured_prompts: dict[int, str] = {}

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        round_num = int((task.metadata or {}).get("forum_round", -1))
        captured_prompts[round_num] = task.metadata.get("task_md_override", "")
        return RuntimeResult(
            output="discussed via MCP tools",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=50, output_tokens=30),
        )

    runtime.run_task.side_effect = fake_run_task

    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0}
    llm = MagicMock()
    llm.call.return_value = LLMResponse(text="{}", usage=TokenUsage())

    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        knowledge_db_path=db_path,
        per_task_forum_rounds=2,
        # Without --resume the engine auto-suffixes the experiment name on
        # collision with the pre-seeded "kcsi" experiment above (starting
        # fresh as "kcsi_2"), which would silently orphan the pre-seeded
        # peer post from this run's queries.
        resume=True,
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )

    traces = _make_traces(1, "agent-0", ["task-0"])
    per_task_forum(orch, 1, traces)

    assert set(captured_prompts) == {0, 1}
    assert "ROUND0_PEER_POST_MARKER" not in captured_prompts[0]
    assert "ROUND0_PEER_POST_MARKER" in captured_prompts[1]
    assert captured_prompts[0] != captured_prompts[1]

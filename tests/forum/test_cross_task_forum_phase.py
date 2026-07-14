"""Tests for the cross-task forum phase service (Plan Task 13).

The cross-task forum phase dispatches N parallel containers (one per agent
with attempts this generation) into a shared room keyed by the sentinel
``__cross_task__`` task_id.  Each agent's prompt contains the current-gen
per-task posts across all tasks.  Posts written to the cross-task room are
drained from ForumBus into KnowledgeStore with ``source_phase="cross_task_forum"``
and ``task_id=CROSS_TASK_SENTINEL``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from ksi.memory.forum_bus import ForumBus
from ksi.memory.knowledge_store import CROSS_TASK_SENTINEL
from ksi.models import GenerationConfig, TaskTrace
from ksi.orchestrator.engine import ForumValidationError, GenerationalOrchestrator, NoopPersistence
from ksi.orchestrator.forum_phase import EngineForumPhaseService
from ksi.runtime.types import RuntimeResult
from ksi.tokens import LLMResponse, TokenUsage
from tests.orchestrator_phase_helpers import cross_task_forum


def _make_orch(tmp_path, runtime) -> GenerationalOrchestrator:
    db_path = str(tmp_path / "knowledge.sqlite")
    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0}
    llm = MagicMock()
    llm.call.return_value = LLMResponse(
        text=json.dumps({"claimed_tasks": []}),
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )
    config = GenerationConfig(
        num_generations=1,
        num_agents=2,
        knowledge_db_path=db_path,
        # Pin explicitly: the ForumBus writes below use experiment="default",
        # and the dataclass default is now "ksi" (CLI parity, #732).
        experiment_name="default",
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )
    return orch


def test_cross_task_forum_service_exists(tmp_path):
    runtime = MagicMock()
    orch = _make_orch(tmp_path, runtime)
    service = EngineForumPhaseService(orch)

    assert callable(service.cross_task_forum)


def test_dispatches_one_container_per_agent(tmp_path):
    """N agents with attempts this generation => N container tasks."""
    runtime = MagicMock()
    captured: dict[str, str] = {}

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        metadata = getattr(task, "metadata", {}) or {}
        if metadata.get("task_source") == "cross_task_forum":
            captured[agent_id] = getattr(task, "prompt", "") or ""
        return RuntimeResult(
            output="",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    runtime.run_task.side_effect = fake_run_task

    orch = _make_orch(tmp_path, runtime)

    # V2: Phase 3 requires both failures and successes (contrast premise).
    # Two agents, one fail + one success.
    traces = [
        TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="t1",
            model_output="",
            eval_result={"resolved": False},
            native_score=0.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
        TaskTrace(
            generation=1,
            agent_id="agent-1",
            task_id="t2",
            model_output="",
            eval_result={"resolved": True},
            native_score=1.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
    ]

    cross_task_forum(orch, generation=1, traces=traces)

    assert set(captured) == {"agent-0", "agent-1"}, (
        f"expected one container per agent with traces, got {list(captured)}"
    )


def test_default_config_runs_two_cross_task_rounds(tmp_path):
    """A default-constructed GenerationConfig drives 2 cross-task rounds
    (the declared cross_task_forum_rounds default, #702): run_task is
    dispatched rounds x n_agents times."""
    runtime = MagicMock()
    calls = {"count": 0}

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        metadata = getattr(task, "metadata", {}) or {}
        if metadata.get("task_source") == "cross_task_forum":
            calls["count"] += 1
        return RuntimeResult(
            output="",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    runtime.run_task.side_effect = fake_run_task

    # _make_orch does not set cross_task_forum_rounds, so the declared
    # default (2) must flow through the engine's round loop.
    orch = _make_orch(tmp_path, runtime)

    # V2: contrast premise — one fail, one success.
    traces = [
        TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="t1",
            model_output="",
            eval_result={"resolved": False},
            native_score=0.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
        TaskTrace(
            generation=1,
            agent_id="agent-1",
            task_id="t2",
            model_output="",
            eval_result={"resolved": True},
            native_score=1.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
    ]

    cross_task_forum(orch, generation=1, traces=traces)

    assert calls["count"] == 2 * 2, (
        f"expected 2 default rounds x 2 agents = 4 cross-task dispatches, got {calls['count']}"
    )


def test_cross_task_forum_phase_fails_when_all_agents_fail(tmp_path):
    runtime = MagicMock()
    runtime.run_task.side_effect = RuntimeError("sandbox unavailable")
    orch = _make_orch(tmp_path, runtime)
    # V2: needs at least one failure and one success for contrast pairing.
    traces = [
        TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="t1",
            model_output="",
            eval_result={"resolved": False},
            native_score=0.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
        TaskTrace(
            generation=1,
            agent_id="agent-1",
            task_id="t2",
            model_output="",
            eval_result={"resolved": True},
            native_score=1.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
    ]

    with pytest.raises(ForumValidationError, match="all cross-task forum agents failed"):
        cross_task_forum(orch, generation=1, traces=traces)


def test_cross_task_forum_phase_retries_transient_agent_failure_then_succeeds(tmp_path):
    runtime = MagicMock()
    calls = {"count": 0}

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("provider unavailable")
        return RuntimeResult(
            output="",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    runtime.run_task.side_effect = fake_run_task

    db_path = str(tmp_path / "knowledge.sqlite")
    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0}
    llm = MagicMock()
    llm.call.return_value = LLMResponse(
        text=json.dumps({"claimed_tasks": []}), usage=TokenUsage(input_tokens=1, output_tokens=1)
    )
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        # Pin a single cross-task round: this test exercises the per-round
        # transient-failure retry path (1 fail + 1 retry = 2 calls), not the
        # default round count. cross_task_forum_rounds now defaults to 2 (#702),
        # so it must be set explicitly here to keep the call count deterministic.
        cross_task_forum_rounds=1,
        knowledge_db_path=db_path,
        max_task_retries=1,
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )
    # V2: needs both failure and success for contrast pairing.
    traces = [
        TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="t1",
            model_output="",
            eval_result={"resolved": False},
            native_score=0.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
        TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="t2",
            model_output="",
            eval_result={"resolved": True},
            native_score=1.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
    ]

    cross_task_forum(orch, generation=1, traces=traces)

    assert calls["count"] == 2


def test_peer_posts_query_hoisted_once_per_round_not_per_agent(tmp_path):
    """Perf: the this-gen peer-posts read is agent-independent, so it must be
    issued ONCE per round regardless of agent count — not once per agent.

    ``_fetch_peer_posts_this_gen`` queries the fixed ``CROSS_TASK_SENTINEL``
    page filtered only by generation/round_num (no agent predicate), and every
    concurrent agent's page is byte-identical. Pre-fix each of the N agents in a
    round re-issued the identical ``query_task(CROSS_TASK_SENTINEL, ...)`` under
    the KnowledgeStore process RLock, serializing N redundant reads. This pins
    the hoist: with the default 2 rounds (round 0 does no query, round 1 does
    one) and 2 agents, exactly ONE sentinel query fires — and both agents in
    round 1 still receive the same peer-post content.
    """
    runtime = MagicMock()
    # Capture the cross-task prompt per (agent_id, round_num).
    captured: dict[tuple[str, int], str] = {}

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        metadata = getattr(task, "metadata", {}) or {}
        if metadata.get("task_source") == "cross_task_forum":
            round_num = int(metadata.get("forum_round", -1))
            captured[(agent_id, round_num)] = getattr(task, "prompt", "") or ""
            # In round 0, drop a distinctive cross-task post so round 1's peer
            # page (drained after round 0) has content to deliver to everyone.
            if round_num == 0:
                bus = ForumBus(
                    db_path=str(tmp_path / "knowledge.sqlite"),
                    experiment="default",
                    generation=generation,
                )
                bus.append(
                    round_num=0,
                    agent_id=agent_id,
                    message_type="post",
                    content={
                        "task_id": CROSS_TASK_SENTINEL,
                        "text": f"MARKER_PEER_POST_FROM_{agent_id}",
                    },
                )
        return RuntimeResult(
            output="",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    runtime.run_task.side_effect = fake_run_task

    orch = _make_orch(tmp_path, runtime)

    # Spy on the store's query_task, counting only the sentinel-page reads that
    # _fetch_peer_posts_this_gen issues (round >= 1). Delegates to the real
    # implementation so behavior is unchanged.
    assert orch._knowledge is not None
    real_query_task = orch._knowledge.query_task
    sentinel_query_calls = {"count": 0}

    def spy_query_task(task_id, *args, **kwargs):
        if task_id == CROSS_TASK_SENTINEL:
            sentinel_query_calls["count"] += 1
        return real_query_task(task_id, *args, **kwargs)

    orch._knowledge.query_task = spy_query_task

    traces = [
        TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="t1",
            model_output="",
            eval_result={"resolved": False},
            native_score=0.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
        TaskTrace(
            generation=1,
            agent_id="agent-1",
            task_id="t2",
            model_output="",
            eval_result={"resolved": True},
            native_score=1.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
    ]

    cross_task_forum(orch, generation=1, traces=traces)

    # Default 2 rounds x 2 agents = 4 dispatches, but only round 1 queries the
    # sentinel page, and it must do so ONCE (hoisted), not once per agent.
    assert sentinel_query_calls["count"] == 1, (
        "expected exactly one hoisted sentinel peer-posts query for round 1 "
        f"(2 agents), got {sentinel_query_calls['count']} — the read was not "
        "hoisted out of the per-agent worker"
    )

    # Behavior preserved: both agents in round 1 received the SAME peer posts.
    r1_a0 = captured.get(("agent-0", 1))
    r1_a1 = captured.get(("agent-1", 1))
    assert r1_a0 is not None and r1_a1 is not None, (
        f"expected both agents dispatched in round 1, captured keys={list(captured)}"
    )
    for marker in ("MARKER_PEER_POST_FROM_agent-0", "MARKER_PEER_POST_FROM_agent-1"):
        assert marker in r1_a0, f"{marker} missing from agent-0's round-1 prompt"
        assert marker in r1_a1, f"{marker} missing from agent-1's round-1 prompt"


def test_v2_prompt_includes_phase1_reflection_not_per_task_posts(tmp_path):
    """V2: cross-task prompt does NOT include per-task posts (Phase 3 is
    structurally independent of Phase 2). Instead it injects each agent's
    Phase 1 reflection so they have their just-attempted-task context.
    """
    runtime = MagicMock()
    captured: dict[str, str] = {}

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        metadata = getattr(task, "metadata", {}) or {}
        if metadata.get("task_source") == "cross_task_forum":
            captured[agent_id] = getattr(task, "prompt", "") or ""
        return RuntimeResult(
            output="",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    runtime.run_task.side_effect = fake_run_task

    orch = _make_orch(tmp_path, runtime)

    # Seed an unrelated per-task post — V2 should NOT surface it in cross-task prompt.
    assert orch._knowledge is not None
    orch._knowledge.record_post(
        task_id="t1",
        agent_id="agent-0",
        generation=1,
        text="MARKER_PER_TASK_POST_UNIQUE_NOT_IN_PHASE3",
        source_phase="per_task_forum",
    )

    traces = [
        TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="t1",
            model_output="",
            eval_result={"resolved": False},
            native_score=0.0,
            tool_trace=[],
            runtime_meta={"phase1_reflection": "MARKER_PHASE1_REFLECTION_FROM_AGENT0"},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
    ]

    cross_task_forum(orch, generation=1, traces=traces)

    assert "agent-0" in captured
    prompt = captured["agent-0"]
    # V2 invariant: Phase 1 reflection IS in the prompt.
    assert "MARKER_PHASE1_REFLECTION_FROM_AGENT0" in prompt
    # V2 invariant: per-task posts are NOT in the prompt.
    assert "MARKER_PER_TASK_POST_UNIQUE_NOT_IN_PHASE3" not in prompt


def test_drains_cross_task_posts_with_sentinel_task_id(tmp_path):
    """Posts written to the cross-task room are drained into KnowledgeStore
    with source_phase='cross_task_forum' and task_id=CROSS_TASK_SENTINEL.
    """
    runtime = MagicMock()

    # The fake runtime appends a cross-task post to the ForumBus via the
    # same path the MCP handler would use, so the drain has something to
    # read once all containers finish.
    def fake_run_task(*, generation, agent_id, task, **kwargs):
        metadata = getattr(task, "metadata", {}) or {}
        if metadata.get("task_source") == "cross_task_forum":
            # Post as this agent into the cross-task room.
            bus = ForumBus(
                db_path=str(tmp_path / "knowledge.sqlite"),
                experiment="default",
                generation=generation,
            )
            bus.append(
                round_num=0,
                agent_id=agent_id,
                message_type="post",
                content={
                    "task_id": CROSS_TASK_SENTINEL,
                    "text": f"cross insight from {agent_id}",
                },
            )
        return RuntimeResult(
            output="",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    runtime.run_task.side_effect = fake_run_task

    orch = _make_orch(tmp_path, runtime)

    # V2: contrast premise — one fail, one success.
    traces = [
        TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="t1",
            model_output="",
            eval_result={"resolved": False},
            native_score=0.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
        TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="t2",
            model_output="",
            eval_result={"resolved": True},
            native_score=1.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
    ]

    cross_task_forum(orch, generation=1, traces=traces)

    # Posts should land in KnowledgeStore under the sentinel task_id with
    # source_phase=cross_task_forum.
    rows = orch._knowledge.query_generation(
        generation=1,
        source_phase="cross_task_forum",
        entry_types=["post"],
    )
    assert len(rows) >= 1, f"expected >=1 cross_task_forum post, got {len(rows)}: {rows!r}"
    assert all(r["task_id"] == CROSS_TASK_SENTINEL for r in rows), (
        f"all cross-task posts must use the sentinel task_id, got task_ids={[r['task_id'] for r in rows]}"
    )
    assert any("cross insight from agent-0" in (r["content"] or {}).get("text", "") for r in rows), (
        "expected the agent's cross-task post text to round-trip"
    )


def test_no_agents_with_traces_is_noop(tmp_path):
    """When no agents have traces, the method returns without dispatching."""
    runtime = MagicMock()
    orch = _make_orch(tmp_path, runtime)

    cross_task_forum(orch, generation=1, traces=[])

    # Nothing dispatched — runtime.run_task should not be called.
    runtime.run_task.assert_not_called()


def test_cross_task_phase_does_not_re_persist_per_task_posts(tmp_path):
    """Regression: the cross-task forum service must clear the shared
    ForumBus JSONL before reading, so per-task events left over from the
    per-task forum service are not re-drained and re-persisted under
    ``source_phase="cross_task_forum"``.
    """
    runtime = MagicMock()

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        # Do NOT append any cross-task events — we only want to test
        # whether the phase leaks pre-existing per-task events.
        return RuntimeResult(
            output="",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    runtime.run_task.side_effect = fake_run_task

    orch = _make_orch(tmp_path, runtime)

    # Simulate the side-effect of the per-task forum service: write a post
    # event into the shared ForumBus JSONL for this generation.  The
    # per-task phase drains (but does not truncate) the JSONL, so the
    # event is still on disk when the cross-task phase starts.
    seed_bus = ForumBus(
        db_path=str(tmp_path / "knowledge.sqlite"),
        experiment="default",
        generation=1,
    )
    leaked_text = "LEAKED_PER_TASK_POST_SHOULD_NOT_APPEAR"
    seed_bus.append(
        round_num=0,
        agent_id="agent-0",
        message_type="post",
        content={"task_id": "t1", "text": leaked_text},
    )

    traces = [
        TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="t1",
            model_output="",
            eval_result={},
            native_score=0.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
    ]

    cross_task_forum(orch, generation=1, traces=traces)

    # The per-task event must NOT have been re-persisted with
    # source_phase="cross_task_forum".
    rows = orch._knowledge.query_generation(
        generation=1,
        source_phase="cross_task_forum",
        entry_types=["post"],
    )
    for row in rows:
        text = (row.get("content") or {}).get("text", "")
        assert leaked_text not in text, (
            "cross-task drain re-persisted a per-task post; ForumBus was not cleared before the cross-task phase ran"
        )


def test_cross_task_post_read_back_has_nonempty_text(tmp_path):
    """Regression for the blank-cross-task-post bug (content-vs-text key mismatch).

    ``_append_task_page_row`` (query_task/query_tasks) flattens a post row into a
    dict keyed ``"text"`` — there is NO ``"content"`` key. The three cross-task
    forum readers must therefore read ``post.get("text")``; reading
    ``post.get("content")`` collapses every history/peer post to ``""``. This
    pins the writer↔reader contract so a regression on either side fails here.
    """
    from ksi.memory.knowledge_store import KnowledgeStore

    store = KnowledgeStore(str(tmp_path / "knowledge.sqlite"), default_experiment="default")
    try:
        body = "concrete cross-task insight that must survive round-trip"
        store.record_post(
            task_id=CROSS_TASK_SENTINEL,
            agent_id="agent-A",
            generation=1,
            text=body,
            round_num=0,
            experiment="default",
            source_phase="cross_task_forum",
        )

        page = store.query_task(
            CROSS_TASK_SENTINEL,
            generation=1,
            entry_types=["post"],
            experiment="default",
        )
        discussion = page.get("discussion") or []
        assert len(discussion) == 1
        post = discussion[0]

        # The flattened post carries "text", not "content" — the exact contract
        # the three forum_phase readers now depend on.
        assert "content" not in post
        assert post.get("text", "") == body

        # Exercise the reader expression used at all three fixed sites.
        text = post.get("text", "")
        assert text == body
    finally:
        store.close()

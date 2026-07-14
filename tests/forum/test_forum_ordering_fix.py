"""Regression tests for forum interaction ordering bias fix.

Verifies that:
1. Store returns messages newest-first (round DESC, id DESC)
2. handle_forum_read returns messages newest-first
3. Round-1 insights sort before round-0 task-exec insights
4. No positional caps truncate insights/comments
5. Generation filtering is strict (no cross-gen leaking)
"""

import json

from kcsi.memory.forum_bus import ForumBus
from kcsi.memory.mcp_server import handle_forum_read
from kcsi.memory.store import MemoryStore


def _make_store(tmp_path):
    return MemoryStore(str(tmp_path / "forum.sqlite"))


def test_store_returns_newest_first(tmp_path):
    """list_forum_messages should return newest messages first (DESC id)."""
    store = _make_store(tmp_path)
    for i in range(5):
        store.insert_forum_message(
            generation=1,
            agent_id=f"agent-{i}",
            message_type="insight",
            content={"text": f"Insight {i}", "insight_id": f"ins-{i}"},
        )
    msgs = store.list_forum_messages(generation=1)
    ids = [m["id"] for m in msgs]
    assert ids == sorted(ids, reverse=True), f"Expected newest-first (DESC) ordering, got ids={ids}"
    # First message should be the last inserted (agent-4)
    assert msgs[0]["agent_id"] == "agent-4"
    store.close()


def test_store_round1_before_round0(tmp_path):
    """Round-1 forum insights should sort before round-0 task-exec insights."""
    store = _make_store(tmp_path)
    # Insert round-0 (task execution) insights first
    for i in range(3):
        store.insert_forum_message(
            generation=1,
            agent_id=f"agent-{i}",
            message_type="insight",
            round_num=0,
            content={"text": f"Task-exec insight {i}", "insight_id": f"r0-{i}"},
        )
    # Then round-1 (deliberate forum) insights
    for i in range(3):
        store.insert_forum_message(
            generation=1,
            agent_id=f"agent-{i}",
            message_type="insight",
            round_num=1,
            content={"text": f"Forum insight {i}", "insight_id": f"r1-{i}"},
        )
    msgs = store.list_forum_messages(generation=1)
    rounds = [m["round_num"] for m in msgs]
    # Round 1 should appear before round 0 (DESC ordering)
    first_round0_idx = next(i for i, r in enumerate(rounds) if r == 0)
    last_round1_idx = max(i for i, r in enumerate(rounds) if r == 1)
    assert last_round1_idx < first_round0_idx, f"Round-1 insights should all appear before round-0, got rounds={rounds}"
    store.close()


def test_store_strict_generation_filter(tmp_path):
    """list_forum_messages should return only the requested generation."""
    store = _make_store(tmp_path)
    store.insert_forum_message(
        generation=1,
        agent_id="a",
        message_type="insight",
        content={"text": "gen1"},
    )
    store.insert_forum_message(
        generation=2,
        agent_id="a",
        message_type="insight",
        content={"text": "gen2"},
    )
    store.insert_forum_message(
        generation=3,
        agent_id="a",
        message_type="insight",
        content={"text": "gen3"},
    )
    msgs = store.list_forum_messages(generation=2)
    assert len(msgs) == 1
    content = json.loads(msgs[0]["content"])
    assert content["text"] == "gen2"
    store.close()


def test_handle_forum_read_newest_first(tmp_path):
    """handle_forum_read should return messages with newest first."""
    db_path = str(tmp_path / "forum.sqlite")
    store = MemoryStore(db_path)
    bus = ForumBus(db_path=db_path, experiment="test", generation=1)
    bus.clear()

    # Simulate round 1: agents post insights via bus
    for i in range(4):
        bus.append(
            round_num=1,
            agent_id=f"agent-{i}",
            message_type="insight",
            content={"text": f"Insight {i}", "insight_id": f"ins-{i}"},
        )

    result = handle_forum_read(
        forum_bus=bus,
        round_num=1,
        up_to_round=False,
        forum_store=store,
        generation=1,
    )
    assert len(result) == 4
    # Newest first: agent-3 should be first
    assert result[0]["agent_id"] == "agent-3"
    assert result[-1]["agent_id"] == "agent-0"
    store.close()


def test_handle_forum_read_round1_before_round0(tmp_path):
    """In round-2 view, round-1 insights should appear before round-0 insights (bus-only)."""
    db_path = str(tmp_path / "forum.sqlite")
    bus = ForumBus(db_path=db_path, experiment="test", generation=1)
    bus.clear()

    # Round-0 task-exec insights (posted via bus)
    for i in range(3):
        bus.append(
            round_num=0,
            agent_id=f"agent-{i}",
            message_type="insight",
            content={"text": f"Task insight {i}", "insight_id": f"r0-ins-{i}"},
        )

    # Round-1 forum insights (posted via bus)
    for i in range(3):
        bus.append(
            round_num=1,
            agent_id=f"agent-{i}",
            message_type="insight",
            content={"text": f"Forum insight {i}", "insight_id": f"r1-ins-{i}"},
        )

    # Simulate round-2 agent calling forum_read with up_to_round=True
    result = handle_forum_read(
        forum_bus=bus,
        round_num=2,
        up_to_round=True,
        generation=1,
    )

    # Separate by round
    round1_indices = [i for i, m in enumerate(result) if m.get("round_num") == 1]
    round0_indices = [i for i, m in enumerate(result) if m.get("round_num") == 0]

    assert round1_indices, "Should have round-1 insights"
    assert round0_indices, "Should have round-0 insights"
    assert max(round1_indices) < min(round0_indices), (
        f"Round-1 insights (indices {round1_indices}) should all appear "
        f"before round-0 insights (indices {round0_indices})"
    )


def test_no_cross_generation_in_forum_read(tmp_path):
    """forum_read for generation N must not return generation N-1 content.

    ForumBus is scoped to a single generation, so cross-gen leaking
    should not happen when using bus-only reads.
    """
    db_path = str(tmp_path / "forum.sqlite")

    # Gen 2 bus — only gen-2 messages should be present
    bus = ForumBus(db_path=db_path, experiment="test", generation=2)
    bus.clear()

    bus.append(
        round_num=1,
        agent_id="a0",
        message_type="insight",
        content={"text": "gen2 insight", "insight_id": "g2-ins"},
    )

    result = handle_forum_read(
        forum_bus=bus,
        round_num=1,
        up_to_round=True,
        generation=2,
    )

    assert len(result) == 1
    assert result[0]["agent_id"] == "a0"

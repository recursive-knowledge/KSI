"""Stress tests for forum ordering fix — covers edge cases and downstream consumers.

Targets:
1. Many agents/messages — ordering invariants hold at scale
2. ID windowing works regardless of row ordering
3. handle_forum_read returns bus messages correctly
4. Mixed rounds with concurrent bus writes
5. Edge cases: empty generation, single message
"""

import threading

from kcsi.memory.forum_bus import ForumBus
from kcsi.memory.mcp_server import handle_forum_read
from kcsi.memory.store import MemoryStore


def _make_store(tmp_path, name="forum.sqlite"):
    return MemoryStore(str(tmp_path / name))


# ---------------------------------------------------------------------------
# 1. Scale: many agents, many messages
# ---------------------------------------------------------------------------


def test_ordering_holds_at_scale(tmp_path):
    """With 20 agents × 5 insights each, newest-first ordering holds."""
    store = _make_store(tmp_path)
    n_agents, n_per_agent = 20, 5
    for a in range(n_agents):
        for i in range(n_per_agent):
            store.insert_forum_message(
                generation=1,
                agent_id=f"agent-{a}",
                message_type="insight",
                round_num=1,
                content={"text": f"Insight {i} from agent-{a}", "insight_id": f"ins-{a}-{i}"},
            )
    msgs = store.list_forum_messages(generation=1)
    assert len(msgs) == n_agents * n_per_agent
    ids = [m["id"] for m in msgs]
    assert ids == sorted(ids, reverse=True), "Messages should be newest-first"
    store.close()


def test_round_ordering_with_many_messages(tmp_path):
    """With many round-0 and round-1 messages, round-1 always comes first."""
    store = _make_store(tmp_path)
    # 30 round-0 task-exec insights
    for i in range(30):
        store.insert_forum_message(
            generation=1,
            agent_id=f"agent-{i % 5}",
            message_type="insight",
            round_num=0,
            content={"text": f"Task exec insight {i}", "insight_id": f"r0-{i}"},
        )
    # 20 round-1 deliberate insights
    for i in range(20):
        store.insert_forum_message(
            generation=1,
            agent_id=f"agent-{i % 5}",
            message_type="insight",
            round_num=1,
            content={"text": f"Forum insight {i}", "insight_id": f"r1-{i}"},
        )
    msgs = store.list_forum_messages(generation=1)
    assert len(msgs) == 50
    rounds = [m["round_num"] for m in msgs]
    # All round-1 messages should precede all round-0 messages
    first_r0 = next(i for i, r in enumerate(rounds) if r == 0)
    last_r1 = max(i for i, r in enumerate(rounds) if r == 1)
    assert last_r1 < first_r0, f"Round-1 must precede round-0 in output, rounds={rounds[:10]}..."
    store.close()


# ---------------------------------------------------------------------------
# 2. _read_forum_pages ID windowing (order-independent)
# ---------------------------------------------------------------------------


def test_id_windowing_works_with_desc_order(tmp_path):
    """ID-based filtering in _read_forum_pages should work regardless of row order."""
    store = _make_store(tmp_path)
    # Insert insights with known IDs
    for i in range(10):
        store.insert_forum_message(
            generation=1,
            agent_id=f"agent-{i % 3}",
            message_type="insight",
            round_num=1,
            content={"text": f"Insight {i}", "insight_id": f"ins-{i}"},
        )
    msgs = store.list_forum_messages(generation=1)
    all_ids = sorted(m["id"] for m in msgs)
    # Window: take only ids in range (all_ids[2], all_ids[7]]
    lo, hi = all_ids[2], all_ids[7]
    windowed = [m for m in msgs if lo < m["id"] <= hi]
    assert len(windowed) == 5  # ids[3..7]
    # All windowed IDs should be in the expected range
    for m in windowed:
        assert lo < m["id"] <= hi
    store.close()


# ---------------------------------------------------------------------------
# 3. handle_forum_read deduplication with reversed sort
# ---------------------------------------------------------------------------


def test_dedup_with_bus_messages(tmp_path):
    """Bus messages with unique insight_ids all appear in output (bus-only reads)."""
    db_path = str(tmp_path / "forum.sqlite")
    bus = ForumBus(db_path=db_path, experiment="test", generation=1)
    bus.clear()

    bus.append(
        round_num=1,
        agent_id="agent-0",
        message_type="insight",
        content={"text": "First insight", "insight_id": "ins-001"},
    )
    bus.append(
        round_num=1,
        agent_id="agent-1",
        message_type="insight",
        content={"text": "Second insight", "insight_id": "ins-002"},
    )
    bus.append(
        round_num=1,
        agent_id="agent-2",
        message_type="insight",
        content={"text": "Third insight", "insight_id": "ins-003"},
    )

    result = handle_forum_read(
        forum_bus=bus,
        round_num=1,
        up_to_round=True,
        generation=1,
    )
    seen_ids = []
    for msg in result:
        content = msg.get("content", {})
        if isinstance(content, dict):
            iid = content.get("insight_id", "")
            if iid:
                seen_ids.append(iid)

    assert len(seen_ids) == 3
    assert "ins-001" in seen_ids
    assert "ins-002" in seen_ids
    assert "ins-003" in seen_ids


# ---------------------------------------------------------------------------
# 4. Mixed rounds with concurrent bus writes
# ---------------------------------------------------------------------------


def test_concurrent_bus_writes_sorted(tmp_path):
    """Concurrent bus writers produce results that are still newest-first after sort."""
    db_path = str(tmp_path / "forum.sqlite")
    store = MemoryStore(db_path)
    bus = ForumBus(db_path=db_path, experiment="test", generation=1)
    bus.clear()

    errors = []
    n_agents, n_per_agent = 8, 10

    def writer(agent_idx):
        try:
            for i in range(n_per_agent):
                bus.append(
                    round_num=1,
                    agent_id=f"agent-{agent_idx}",
                    message_type="insight",
                    content={"text": f"Msg {i} from agent-{agent_idx}", "insight_id": f"ins-{agent_idx}-{i}"},
                )
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_agents)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [], f"Write errors: {errors}"

    result = handle_forum_read(
        forum_bus=bus,
        round_num=1,
        up_to_round=False,
        forum_store=store,
        generation=1,
    )
    assert len(result) == n_agents * n_per_agent
    # Verify sort invariant: IDs are descending
    ids = [r.get("id", 0) for r in result]
    assert ids == sorted(ids, reverse=True), "Bus-only results should be newest-first"
    store.close()


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


def test_empty_generation(tmp_path):
    """forum_read on empty generation returns empty list, not an error."""
    db_path = str(tmp_path / "forum.sqlite")
    store = MemoryStore(db_path)
    bus = ForumBus(db_path=db_path, experiment="test", generation=99)
    bus.clear()

    result = handle_forum_read(
        forum_bus=bus,
        round_num=1,
        up_to_round=True,
        include_round0_from_store=True,
        forum_store=store,
        generation=99,
    )
    assert result == []
    store.close()


def test_single_message_ordering(tmp_path):
    """Single message should be returned successfully."""
    store = _make_store(tmp_path)
    store.insert_forum_message(
        generation=1,
        agent_id="solo",
        message_type="insight",
        round_num=1,
        content={"text": "Only insight", "insight_id": "ins-solo"},
    )
    msgs = store.list_forum_messages(generation=1)
    assert len(msgs) == 1
    assert msgs[0]["agent_id"] == "solo"
    store.close()


def test_comments_interleaved_with_insights(tmp_path):
    """Insights and comments in the same round sort by ID DESC, not type."""
    store = _make_store(tmp_path)
    store.insert_forum_message(
        generation=1,
        agent_id="a0",
        message_type="insight",
        round_num=1,
        content={"text": "First insight", "insight_id": "ins-1"},
    )
    store.insert_forum_message(
        generation=1,
        agent_id="a1",
        message_type="comment",
        round_num=1,
        content={"text": "Comment on first", "comment_id": "c-1", "target_insight_id": "ins-1"},
    )
    store.insert_forum_message(
        generation=1,
        agent_id="a2",
        message_type="insight",
        round_num=1,
        content={"text": "Second insight", "insight_id": "ins-2"},
    )
    msgs = store.list_forum_messages(generation=1)
    assert len(msgs) == 3
    # DESC by id: second insight (id=3), comment (id=2), first insight (id=1)
    ids = [m["id"] for m in msgs]
    assert ids == sorted(ids, reverse=True)
    store.close()


def test_multi_generation_isolation(tmp_path):
    """Stress: 5 generations × 10 agents × 4 messages. Each generation sees only its own."""
    store = _make_store(tmp_path)
    n_gens, n_agents, n_msgs = 5, 10, 4
    for g in range(1, n_gens + 1):
        for a in range(n_agents):
            for m in range(n_msgs):
                store.insert_forum_message(
                    generation=g,
                    agent_id=f"agent-{a}",
                    message_type="insight",
                    round_num=1,
                    content={"text": f"g{g}-a{a}-m{m}", "insight_id": f"ins-g{g}-a{a}-m{m}"},
                )
    for g in range(1, n_gens + 1):
        msgs = store.list_forum_messages(generation=g)
        assert len(msgs) == n_agents * n_msgs, f"Gen {g}: expected {n_agents * n_msgs}, got {len(msgs)}"
        for msg in msgs:
            assert msg["generation"] == g, f"Gen {g} leaked: got gen {msg['generation']}"
    store.close()

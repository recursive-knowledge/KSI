"""Tests that ForumBus append and drain preserve the reply_to field."""

import json
import tempfile
from pathlib import Path


def _make_bus(tmp: str):
    from ksi.memory.forum_bus import ForumBus

    # ForumBus resolves db_parent.parent / "forum_bus" for its event dir; we
    # create a throwaway sqlite path under tmp so the bus has a stable anchor.
    db_path = Path(tmp) / "k.sqlite"
    db_path.touch()
    return ForumBus(db_path=str(db_path), experiment="test-exp", generation=1)


def test_append_with_reply_to_persists_field():
    with tempfile.TemporaryDirectory() as tmp:
        bus = _make_bus(tmp)
        bus.append(
            round_num=0,
            agent_id="a1",
            message_type="post",
            content={"task_id": "t1", "text": "reply", "reply_to": 42},
        )
        events = list(Path(tmp).glob("**/*.events.jsonl"))
        assert events, "forum bus did not create the events.jsonl file"
        line = events[0].read_text().strip().splitlines()[0]
        payload = json.loads(line)
        assert payload["content"]["reply_to"] == 42


def test_drain_forum_bus_propagates_reply_to_to_record_post():
    """Orchestrator drain should carry reply_to through to knowledge_store.record_post."""
    from ksi.memory.knowledge_store import KnowledgeStore
    from ksi.orchestrator.engine import _drain_forum_bus

    with tempfile.TemporaryDirectory() as tmp:
        bus = _make_bus(tmp)
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="test-exp")
        try:
            # First post: parent
            bus.append(
                round_num=0,
                agent_id="a1",
                message_type="post",
                content={"task_id": "t1", "text": "parent"},
            )
            # Second post: reply references the first post by a fake prior id
            bus.append(
                round_num=0,
                agent_id="a2",
                message_type="post",
                content={"task_id": "t1", "text": "reply", "reply_to": 999},
            )
            n = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test-exp",
            )
            assert n == 2
            # Confirm reply_to was persisted on the second row.
            rows = (
                ks._connection()
                .execute(
                    "SELECT text_json, reply_to FROM ("
                    " SELECT content AS text_json, reply_to FROM knowledge "
                    " WHERE entry_type='post' ORDER BY id ASC)"
                )
                .fetchall()
            )
            assert len(rows) == 2
            assert rows[0][1] is None
            assert rows[1][1] == 999
        finally:
            ks.close()


def test_drain_forum_bus_is_idempotent_by_event_id():
    """Re-draining the same ForumBus file must not duplicate knowledge rows."""
    from ksi.memory.knowledge_store import KnowledgeStore
    from ksi.orchestrator.engine import _drain_forum_bus

    with tempfile.TemporaryDirectory() as tmp:
        bus = _make_bus(tmp)
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="test-exp")
        try:
            event = bus.append(
                round_num=0,
                agent_id="a1",
                message_type="post",
                content={"task_id": "t1", "text": "stable event"},
            )

            first = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test-exp",
            )
            second = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test-exp",
            )

            assert first == 1
            assert second == 0
            rows = ks._connection().execute("SELECT external_id FROM knowledge WHERE entry_type='post'").fetchall()
            assert [row["external_id"] for row in rows] == [event["event_id"]]
            page = ks.query_task("t1", experiment="test-exp")
            assert [post["text"] for post in page["discussion"]] == ["stable event"]
        finally:
            ks.close()

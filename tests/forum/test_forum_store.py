"""Tests for forum DB schema and MemoryStore forum methods."""

import json
import threading

import pytest

import kcsi.memory.store as store_module
from kcsi.memory.store import MemoryStore


def test_forum_tables_created(tmp_path):
    """Forum tables should exist after MemoryStore init."""
    store = MemoryStore(str(tmp_path / "test.sqlite"))
    tables = store._execute(
        "SELECT name FROM sqlite_master WHERE type='table'",
        fetchall=True,
    )
    table_names = {r["name"] for r in tables}
    assert "forum_events" in table_names
    store.close()


def test_insert_and_list_forum_messages(tmp_path):
    """Should insert and retrieve forum messages for a generation."""
    store = MemoryStore(str(tmp_path / "test.sqlite"))
    store.insert_forum_message(
        generation=1,
        agent_id="agent-0",
        message_type="insight",
        content={"text": "Django ORM caching is tricky", "workstream": "django"},
    )
    store.insert_forum_message(
        generation=1,
        agent_id="agent-1",
        message_type="insight",
        content={"text": "Testing patterns are important", "workstream": "testing"},
    )
    store.insert_forum_message(
        generation=2,
        agent_id="agent-0",
        message_type="insight",
        content={"text": "unrelated"},
    )
    msgs = store.list_forum_messages(generation=1)
    assert len(msgs) == 2
    # Newest-first ordering (DESC by id) — agent-1 was inserted second
    assert msgs[0]["agent_id"] == "agent-1"
    assert msgs[0]["message_type"] == "insight"
    parsed = json.loads(msgs[0]["content"])
    assert parsed["text"] == "Testing patterns are important"
    assert msgs[1]["agent_id"] == "agent-0"
    assert all(m["generation"] == 1 for m in msgs)
    store.close()


def test_forum_insights_and_comments(tmp_path):
    """Should store and retrieve insight and comment messages."""
    store = MemoryStore(str(tmp_path / "test.sqlite"))
    store.insert_forum_message(
        generation=1,
        agent_id="agent-0",
        message_type="insight",
        content={"text": "Django ORM needs caching", "scope": "task"},
    )
    store.insert_forum_message(
        generation=1,
        agent_id="agent-1",
        message_type="comment",
        content={"text": "Agree", "target_insight_id": "ins-1"},
    )
    msgs = store.list_forum_messages(generation=1)
    assert len(msgs) == 2
    types = {m["message_type"] for m in msgs}
    assert types == {"insight", "comment"}
    store.close()


def test_wal_mode_enabled(tmp_path):
    """DB should use WAL journal mode for concurrent access."""
    store = MemoryStore(str(tmp_path / "test.sqlite"))
    result = store._execute("PRAGMA journal_mode", fetchone=True)
    assert result["journal_mode"] == "wal"
    store.close()


def test_concurrent_forum_writes(tmp_path):
    """Multiple threads can write forum messages simultaneously via WAL mode."""
    db_path = str(tmp_path / "concurrent.sqlite")
    num_threads = 5
    messages_per_thread = 20
    errors = []

    def writer(agent_id):
        try:
            store = MemoryStore(db_path)
            for i in range(messages_per_thread):
                store.insert_forum_message(
                    generation=1,
                    agent_id=agent_id,
                    message_type="insight",
                    content={"text": f"Message {i} from {agent_id}"},
                )
            store.close()
        except Exception as e:
            errors.append((agent_id, str(e)))

    threads = [threading.Thread(target=writer, args=(f"agent-{i}",)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Concurrent write errors: {errors}"

    # Verify all messages were written
    store = MemoryStore(db_path)
    messages = store.list_forum_messages(generation=1)
    assert len(messages) == num_threads * messages_per_thread
    store.close()


def test_forum_comments_reference_insights(tmp_path):
    """Comments should store target_insight_id linking back to insights."""
    store = MemoryStore(str(tmp_path / "test.sqlite"))
    store.insert_forum_message(
        generation=1,
        agent_id="agent-0",
        message_type="insight",
        content={"text": "Django ORM caching", "insight_id": "ins-a0-1"},
    )
    msgs = store.list_forum_messages(generation=1)
    store.insert_forum_message(
        generation=1,
        agent_id="agent-1",
        message_type="comment",
        content={
            "text": "Agree with caching insight",
            "target_insight_id": "ins-a0-1",
            "comment_id": "c-a1-1",
        },
    )
    msgs = store.list_forum_messages(generation=1)
    assert len(msgs) == 2
    comment = [m for m in msgs if m["message_type"] == "comment"][0]
    content = json.loads(comment["content"]) if isinstance(comment["content"], str) else comment["content"]
    assert content["target_insight_id"] == "ins-a0-1"
    store.close()


def test_invalid_message_type_skipped(tmp_path):
    """Unknown message types should be silently skipped."""
    store = MemoryStore(str(tmp_path / "test.sqlite"))
    store.insert_forum_message(
        generation=1,
        agent_id="agent-0",
        message_type="totally_bogus",
        content={"workstreams": ["django"]},
    )
    msgs = store.list_forum_messages(generation=1)
    assert len(msgs) == 0
    store.close()


def test_insert_forum_message_failure_rolls_back_dimension_rows(tmp_path, monkeypatch):
    """An exception mid-insert_forum_message must not leave partial rows (#736).

    The _ensure_run/_ensure_generation/_ensure_agent upserts and the final
    forum_events INSERT share one _batched() transaction; if the final INSERT
    fails, the already-applied dimension rows must be rolled back too.
    """
    store = MemoryStore(str(tmp_path / "test.sqlite"))
    try:
        original_json_dumps = store_module._json_dumps

        def boom(value):
            raise RuntimeError("simulated serialization failure")

        # _json_dumps runs after the _ensure_* upserts and before the
        # forum_events INSERT lands — i.e. mid-transaction.
        monkeypatch.setattr(store_module, "_json_dumps", boom)
        with pytest.raises(RuntimeError, match="simulated serialization failure"):
            store.insert_forum_message(
                generation=1,
                agent_id="agent-0",
                message_type="insight",
                content={"text": "will-fail"},
                experiment="rb-forum",
            )

        runs = store._execute(
            "SELECT COUNT(*) AS cnt FROM runs WHERE experiment = ?",
            ("rb-forum",),
            fetchone=True,
        )
        assert runs["cnt"] == 0, "rollback failed: dimension run row survived the failed insert"
        events = store._execute("SELECT COUNT(*) AS cnt FROM forum_events", fetchone=True)
        assert events["cnt"] == 0

        # Restoring + retrying must succeed cleanly (connection not wedged).
        monkeypatch.setattr(store_module, "_json_dumps", original_json_dumps)
        store.insert_forum_message(
            generation=1,
            agent_id="agent-0",
            message_type="insight",
            content={"text": "works-now"},
            experiment="rb-forum",
        )
        msgs = store.list_forum_messages(generation=1, experiment="rb-forum")
        assert len(msgs) == 1
        assert json.loads(msgs[0]["content"])["text"] == "works-now"
    finally:
        store.close()

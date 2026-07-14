"""Tests for the ``reply_to`` column on the ``knowledge`` table."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from kcsi.memory.knowledge_store import KnowledgeStore


def test_reply_to_column_exists_on_fresh_db():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            cols = ks._connection().execute("PRAGMA table_info(knowledge)").fetchall()
            names = {row[1] for row in cols}
            assert "reply_to" in names
        finally:
            ks.close()


def test_reply_to_migration_on_existing_db():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "k.sqlite"
        ks1 = KnowledgeStore(str(db), default_experiment="exp")
        # Simulate an older DB that never had reply_to by dropping the column
        # (drop the dependent index first so ALTER TABLE succeeds).
        conn = ks1._connection()
        conn.execute("DROP INDEX IF EXISTS idx_knowledge_reply_to")
        conn.execute("ALTER TABLE knowledge DROP COLUMN reply_to")
        conn.commit()
        ks1.close()

        ks2 = KnowledgeStore(str(db), default_experiment="exp")
        try:
            cols = ks2._connection().execute("PRAGMA table_info(knowledge)").fetchall()
            names = {row[1] for row in cols}
            assert "reply_to" in names
        finally:
            ks2.close()


def test_record_post_with_reply_to_persists():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            parent_id = ks.record_post(
                task_id="t1",
                agent_id="a1",
                generation=0,
                text="parent",
            )
            child_id = ks.record_post(
                task_id="t1",
                agent_id="a2",
                generation=0,
                text="reply",
                reply_to=parent_id,
            )
            row = ks._connection().execute("SELECT reply_to FROM knowledge WHERE id=?", (child_id,)).fetchone()
            assert row[0] == parent_id
        finally:
            ks.close()


def test_query_task_returns_reply_to_for_posts():
    """query_task must surface reply_to so readers see threading."""
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(
            str(Path(tmp) / "k.sqlite"),
            default_experiment="exp",
        )
        try:
            parent_id = ks.record_post(
                task_id="t1",
                agent_id="a1",
                generation=0,
                text="parent",
            )
            ks.record_post(
                task_id="t1",
                agent_id="a2",
                generation=0,
                text="reply",
                reply_to=parent_id,
            )
            page = ks.query_task("t1", generation=0, entry_types=["post"])
            posts = page["discussion"]
            assert len(posts) == 2
            reply_post = [p for p in posts if p["text"] == "reply"][0]
            assert reply_post["reply_to"] == parent_id
        finally:
            ks.close()


def test_query_generation_returns_reply_to_for_posts():
    """query_generation must surface reply_to on all returned rows."""
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(
            str(Path(tmp) / "k.sqlite"),
            default_experiment="exp",
        )
        try:
            parent_id = ks.record_post(
                task_id="t1",
                agent_id="a1",
                generation=0,
                text="parent",
            )
            ks.record_post(
                task_id="t1",
                agent_id="a2",
                generation=0,
                text="reply",
                reply_to=parent_id,
            )
            rows = ks.query_generation(0, entry_types=["post"])
            assert len(rows) == 2
            reply_row = [r for r in rows if (r["content"] or {}).get("text") == "reply"][0]
            assert reply_row["reply_to"] == parent_id
        finally:
            ks.close()


# ---------------------------------------------------------------------------
# #1122: opt-in SQLite FK enforcement + app-level reply_to validation
# ---------------------------------------------------------------------------


def test_dangling_reply_to_tolerated_by_default(monkeypatch):
    """Default (FK enforcement OFF): a dangling reply_to still inserts.

    The forum drain deliberately tolerates agent-supplied ``reply_to`` ids that
    don't resolve to a real row (see forum_runtime._coerce_post_ref). This must
    remain the default so posts are never dropped on normal runs.
    """
    monkeypatch.delenv("KCSI_SQLITE_FOREIGN_KEYS", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            child_id = ks.record_post(
                task_id="t1",
                agent_id="a1",
                generation=0,
                text="orphan reply",
                reply_to=999999,  # no such knowledge row
            )
            row = ks._connection().execute("SELECT reply_to FROM knowledge WHERE id=?", (child_id,)).fetchone()
            assert row[0] == 999999
        finally:
            ks.close()


def test_reply_to_validation_raises_when_fk_enforced(monkeypatch):
    """FK enforcement ON: an app-level check raises on a dangling reply_to."""
    monkeypatch.setenv("KCSI_SQLITE_FOREIGN_KEYS", "1")
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            with pytest.raises(ValueError, match="reply_to=999999"):
                ks.record_post(
                    task_id="t1",
                    agent_id="a1",
                    generation=0,
                    text="orphan reply",
                    reply_to=999999,
                )
        finally:
            ks.close()


def test_valid_reply_to_still_persists_when_fk_enforced(monkeypatch):
    """FK enforcement ON: a reply_to pointing at a real row still inserts."""
    monkeypatch.setenv("KCSI_SQLITE_FOREIGN_KEYS", "1")
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            parent_id = ks.record_post(task_id="t1", agent_id="a1", generation=0, text="parent")
            child_id = ks.record_post(
                task_id="t1",
                agent_id="a2",
                generation=0,
                text="reply",
                reply_to=parent_id,
            )
            row = ks._connection().execute("SELECT reply_to FROM knowledge WHERE id=?", (child_id,)).fetchone()
            assert row[0] == parent_id
        finally:
            ks.close()


def test_fk_pragma_off_by_default(monkeypatch):
    """The PRAGMA is opt-in: default connections keep foreign_keys=OFF."""
    monkeypatch.delenv("KCSI_SQLITE_FOREIGN_KEYS", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            assert ks._connection().execute("PRAGMA foreign_keys").fetchone()[0] == 0
        finally:
            ks.close()


def test_fk_pragma_on_when_flag_set_and_bad_fk_raises(monkeypatch):
    """FK enforcement ON: the PRAGMA is enabled and a bad FK ref raises.

    Insert a knowledge row with a bogus ``run_id`` via raw SQL (bypassing the
    ``_ensure_run_locked`` resolution) — SQLite must reject it because
    ``run_id REFERENCES runs(id)`` is now enforced.
    """
    monkeypatch.setenv("KCSI_SQLITE_FOREIGN_KEYS", "1")
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            conn = ks._connection()
            assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO knowledge
                        (run_id, generation, task_id, agent_id, entry_type,
                         source_phase, content)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (999999, 0, "t1", "a1", "insight", "execution", "{}"),
                )
        finally:
            ks.close()

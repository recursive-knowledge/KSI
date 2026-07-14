"""Tests for the ``idx_knowledge_task_type`` index (issue #949, E2).

The per-bucket ``knowledge_page`` reads join ``runs`` by experiment (so no
fixed ``run_id``) and filter on ``(task_id, entry_type)``. The run_id-leading
indexes can't seek ``entry_type`` there; ``idx_knowledge_task_type`` does, and
also orders by ``id`` so the ``ORDER BY k.id DESC`` needs no sort.
"""

import tempfile
from pathlib import Path

from kcsi.memory.knowledge_store import KnowledgeStore


def _index_names(ks: KnowledgeStore) -> set[str]:
    rows = ks._connection().execute("PRAGMA index_list(knowledge)").fetchall()
    return {row[1] for row in rows}


def test_task_type_index_exists_on_fresh_db():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            assert "idx_knowledge_task_type" in _index_names(ks)
        finally:
            ks.close()


def test_task_type_index_added_by_migration():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "k.sqlite"
        ks1 = KnowledgeStore(str(db), default_experiment="exp")
        # Simulate an older DB that predates the index.
        conn = ks1._connection()
        conn.execute("DROP INDEX IF EXISTS idx_knowledge_task_type")
        conn.commit()
        assert "idx_knowledge_task_type" not in _index_names(ks1)
        ks1.close()

        ks2 = KnowledgeStore(str(db), default_experiment="exp")
        try:
            assert "idx_knowledge_task_type" in _index_names(ks2)
        finally:
            ks2.close()


def test_task_type_index_serves_knowledge_page_query():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            plan = (
                ks._connection()
                .execute(
                    """
                EXPLAIN QUERY PLAN
                SELECT k.id FROM knowledge k JOIN runs r ON r.id = k.run_id
                WHERE r.experiment = ? AND k.task_id = ? AND k.entry_type = ?
                ORDER BY k.id DESC LIMIT ?
                """,
                    ("exp", "t1", "attempt", 10),
                )
                .fetchall()
            )
            detail = " | ".join(str(row[3]) for row in plan)
            # #1014 added idx_knowledge_run_entry_task (run_id, entry_type,
            # task_id, id), a more selective 3-column seek for this query
            # (via the runs join binding run_id) than idx_knowledge_task_type's
            # 2-column one — SQLite now prefers it. Either index satisfies the
            # original intent here: an index seek instead of a scan.
            assert "idx_knowledge_task_type" in detail or "idx_knowledge_run_entry_task" in detail
            # The index supplies the id ordering, so no sort is needed.
            assert "TEMP B-TREE FOR ORDER BY" not in detail
        finally:
            ks.close()

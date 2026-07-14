"""Tests for the ``idx_knowledge_run_entry_task`` index (issue #1014).

``list_task_summaries``' inner subquery filters ``(run_id, entry_type)`` and
groups by ``task_id``. The closest existing index, ``idx_knowledge_task``
(``run_id, task_id, generation, id``), doesn't lead with ``entry_type`` there,
so the subquery can't seek on it. ``idx_knowledge_run_entry_task`` does.
Mirrors ``test_knowledge_store_task_type_index.py``'s structure.
"""

import tempfile
from pathlib import Path

from ksi.memory.knowledge_store import KnowledgeStore


def _index_names(ks: KnowledgeStore) -> set[str]:
    rows = ks._connection().execute("PRAGMA index_list(knowledge)").fetchall()
    return {row[1] for row in rows}


def test_run_entry_task_index_exists_on_fresh_db():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            assert "idx_knowledge_run_entry_task" in _index_names(ks)
        finally:
            ks.close()


def test_run_entry_task_index_added_by_migration():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "k.sqlite"
        ks1 = KnowledgeStore(str(db), default_experiment="exp")
        # Simulate an older DB that predates the index.
        conn = ks1._connection()
        conn.execute("DROP INDEX IF EXISTS idx_knowledge_run_entry_task")
        conn.commit()
        assert "idx_knowledge_run_entry_task" not in _index_names(ks1)
        ks1.close()

        ks2 = KnowledgeStore(str(db), default_experiment="exp")
        try:
            assert "idx_knowledge_run_entry_task" in _index_names(ks2)
        finally:
            ks2.close()


def test_run_entry_task_index_serves_list_task_summaries_subquery():
    """EXPLAIN QUERY PLAN against ``list_task_summaries``'s inner subquery
    must seek ``idx_knowledge_run_entry_task`` on ``(run_id, entry_type)``,
    not fall back to a less-selective index/scan."""
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            plan = (
                ks._connection()
                .execute(
                    """
                EXPLAIN QUERY PLAN
                SELECT task_id, MAX(id) AS latest_id
                FROM knowledge
                WHERE run_id = ? AND entry_type = ?
                GROUP BY task_id
                """,
                    (1, "attempt"),
                )
                .fetchall()
            )
            detail = " | ".join(str(row[3]) for row in plan)
            assert "idx_knowledge_run_entry_task" in detail
            assert "(run_id=? AND entry_type=?)" in detail
        finally:
            ks.close()


def test_run_entry_task_index_improves_plan_over_dropped_baseline():
    """Empirical before/after: dropping the index makes SQLite fall back to a
    less-selective index (or scan) for ``list_task_summaries``'s full query;
    the new index gives a covering equality seek on ``(run_id, entry_type)``.

    Uses fresh connections per side (not the shared ``KnowledgeStore``
    connection) because Python's ``sqlite3`` module caches prepared
    statements by SQL text, which would otherwise mask the DROP INDEX by
    replaying the earlier query's cached plan.
    """
    import sqlite3

    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "k.sqlite")
        ks = KnowledgeStore(db, default_experiment="exp")
        for gen in range(20):
            for t in range(30):
                ks.record_attempt(
                    task_id=f"t{t}",
                    agent_id="a1",
                    generation=gen,
                    model_output="x",
                    native_score=0.5,
                )
        run_id = ks._find_run("exp")
        ks.close()

        query = """
            SELECT k.task_id, k.generation, k.agent_id, k.native_score, k.content, k.created_at
            FROM knowledge k
            JOIN (
                SELECT task_id, MAX(id) AS latest_id
                FROM knowledge
                WHERE run_id = ? AND entry_type = 'attempt'
                GROUP BY task_id
            ) latest ON latest.latest_id = k.id
            WHERE k.run_id = ?
            ORDER BY k.id DESC
            LIMIT ?
        """

        with_index_conn = sqlite3.connect(db)
        try:
            with_index_plan = with_index_conn.execute("EXPLAIN QUERY PLAN " + query, (run_id, run_id, 200)).fetchall()
        finally:
            with_index_conn.close()
        with_index_detail = " | ".join(str(row[3]) for row in with_index_plan)
        assert "idx_knowledge_run_entry_task (run_id=? AND entry_type=?)" in with_index_detail

        drop_conn = sqlite3.connect(db)
        drop_conn.execute("DROP INDEX IF EXISTS idx_knowledge_run_entry_task")
        drop_conn.commit()
        drop_conn.close()

        without_index_conn = sqlite3.connect(db)
        try:
            without_index_plan = without_index_conn.execute(
                "EXPLAIN QUERY PLAN " + query, (run_id, run_id, 200)
            ).fetchall()
        finally:
            without_index_conn.close()
        without_index_detail = " | ".join(str(row[3]) for row in without_index_plan)
        assert "idx_knowledge_run_entry_task" not in without_index_detail

"""Tests for #990a: runs.code_commit/resolved_model/scoring_mode provenance stamp.

Covers ``MemoryStore`` (src/kcsi/memory/store.py). See
tests/memory/test_run_metadata_stamp.py for the equivalent KnowledgeStore coverage.
"""

import sqlite3

from kcsi.memory.store import MemoryStore


class TestRunMetadataColumns:
    def test_runs_table_has_metadata_columns(self, tmp_path):
        store = MemoryStore(str(tmp_path / "m.sqlite"))
        try:
            cols = {row[1] for row in store._conn.execute("PRAGMA table_info(runs)").fetchall()}
            assert {"code_commit", "resolved_model", "scoring_mode", "config_json"} <= cols
        finally:
            store.close()

    def test_ensure_run_stamps_config_json(self, tmp_path):
        store = MemoryStore(str(tmp_path / "m.sqlite"))
        try:
            run_id = store.ensure_run("exp1", config_json='{"seed": 0, "task_source": "arc"}')
            row = store._execute("SELECT config_json FROM runs WHERE id=?", (run_id,), fetchone=True)
            assert row["config_json"] == '{"seed": 0, "task_source": "arc"}'
        finally:
            store.close()

    def test_ensure_run_stamps_metadata(self, tmp_path):
        store = MemoryStore(str(tmp_path / "m.sqlite"))
        try:
            run_id = store.ensure_run(
                "exp1",
                code_commit="abc1234",
                resolved_model="anthropic/claude-haiku-4-5-20251001",
                scoring_mode="arc_session",
            )
            row = store._execute("SELECT * FROM runs WHERE id=?", (run_id,), fetchone=True)
            assert row["code_commit"] == "abc1234"
            assert row["resolved_model"] == "anthropic/claude-haiku-4-5-20251001"
            assert row["scoring_mode"] == "arc_session"
        finally:
            store.close()

    def test_ensure_run_does_not_clobber_on_repeat_call(self, tmp_path):
        store = MemoryStore(str(tmp_path / "m.sqlite"))
        try:
            run_id = store.ensure_run("exp1", code_commit="abc1234", resolved_model="m", scoring_mode="s")
            run_id_2 = store.ensure_run("exp1")
            assert run_id_2 == run_id
            row = store._execute("SELECT * FROM runs WHERE id=?", (run_id,), fetchone=True)
            assert row["code_commit"] == "abc1234"
            assert row["resolved_model"] == "m"
            assert row["scoring_mode"] == "s"
        finally:
            store.close()

    def test_ensure_run_write_once_keeps_first_stamp_on_drift(self, tmp_path):
        # Sidecar mirrors KnowledgeStore's write-once contract: a --resume under a
        # changed HEAD/model keeps the original stamp (warning is emitted by the
        # authoritative KnowledgeStore, not duplicated here).
        store = MemoryStore(str(tmp_path / "m.sqlite"))
        try:
            run_id = store.ensure_run(
                "exp1",
                code_commit="abc1234",
                resolved_model="m1",
                scoring_mode="s",
                config_json='{"seed": 1}',
            )
            store.ensure_run(
                "exp1",
                code_commit="def5678",
                resolved_model="m2",
                scoring_mode="s",
                config_json='{"seed": 2}',
            )
            row = store._execute("SELECT * FROM runs WHERE id=?", (run_id,), fetchone=True)
            assert row["code_commit"] == "abc1234"
            assert row["resolved_model"] == "m1"
            assert row["config_json"] == '{"seed": 1}'
        finally:
            store.close()

    def test_existing_db_migrates_missing_columns(self, tmp_path):
        db_path = tmp_path / "legacy.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "experiment TEXT NOT NULL UNIQUE, created_at TEXT DEFAULT (datetime('now')))"
        )
        conn.execute("INSERT INTO runs (experiment) VALUES ('pre_existing')")
        conn.commit()
        conn.close()

        store = MemoryStore(str(db_path))
        try:
            cols = {row[1] for row in store._conn.execute("PRAGMA table_info(runs)").fetchall()}
            assert {"code_commit", "resolved_model", "scoring_mode"} <= cols
            row = store._execute("SELECT code_commit FROM runs WHERE experiment='pre_existing'", fetchone=True)
            assert row["code_commit"] is None
        finally:
            store.close()

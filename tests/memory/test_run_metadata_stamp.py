"""Tests for #990a: runs.code_commit/resolved_model/scoring_mode provenance stamp.

Covers ``KnowledgeStore`` (src/ksi/memory/knowledge_store.py). See
tests/memory/test_store_run_metadata.py for the equivalent MemoryStore coverage.
"""

import sqlite3

from ksi.memory.knowledge_store import KnowledgeStore


class TestRunMetadataColumns:
    def test_runs_table_has_metadata_columns(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "k.sqlite"))
        try:
            cols = {row[1] for row in store._connection().execute("PRAGMA table_info(runs)").fetchall()}
            assert {"code_commit", "resolved_model", "scoring_mode"} <= cols
        finally:
            store.close()

    def test_ensure_run_stamps_metadata(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "k.sqlite"))
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

    def test_ensure_run_stamps_config_json(self, tmp_path):
        # The full effective launch config is persisted into the runs row so the
        # knowledge DB is self-describing (recoverable without the --output-json).
        store = KnowledgeStore(str(tmp_path / "k.sqlite"))
        try:
            cols = {row[1] for row in store._connection().execute("PRAGMA table_info(runs)").fetchall()}
            assert "config_json" in cols
            run_id = store.ensure_run("exp1", config_json='{"seed": 0, "task_source": "arc"}')
            row = store._execute("SELECT config_json FROM runs WHERE id=?", (run_id,), fetchone=True)
            assert row["config_json"] == '{"seed": 0, "task_source": "arc"}'
        finally:
            store.close()

    def test_ensure_run_does_not_clobber_on_repeat_call(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "k.sqlite"))
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

    def test_ensure_run_write_once_keeps_first_stamp_on_drift(self, tmp_path, caplog):
        # A --resume under a changed HEAD/model must NOT overwrite the original
        # provenance stamp (else the row misreports one version for generations
        # that ran under several). Keep the first value and warn on drift.
        import logging

        store = KnowledgeStore(str(tmp_path / "k.sqlite"))
        try:
            run_id = store.ensure_run(
                "exp1",
                code_commit="abc1234",
                resolved_model="m1",
                scoring_mode="s",
                config_json='{"seed": 1}',
            )
            with caplog.at_level(logging.WARNING):
                run_id_2 = store.ensure_run(
                    "exp1",
                    code_commit="def5678",
                    resolved_model="m2",
                    scoring_mode="s",
                    config_json='{"seed": 2}',
                )
            assert run_id_2 == run_id
            row = store._execute("SELECT * FROM runs WHERE id=?", (run_id,), fetchone=True)
            assert row["code_commit"] == "abc1234"  # original preserved, not clobbered
            assert row["resolved_model"] == "m1"
            assert row["scoring_mode"] == "s"  # unchanged value = no warning
            assert row["config_json"] == '{"seed": 1}'
            text = caplog.text
            assert "code_commit drift" in text
            assert "resolved_model drift" in text
            assert "config_json drift" in text
            assert "scoring_mode drift" not in text
        finally:
            store.close()

    def test_ensure_run_fills_null_field_left_unstamped(self, tmp_path):
        # Write-once fills a still-NULL field on a later call (only same-valued
        # or NULL->value transitions are allowed; differing values are ignored).
        store = KnowledgeStore(str(tmp_path / "k.sqlite"))
        try:
            run_id = store.ensure_run("exp1", code_commit="abc1234")
            store.ensure_run("exp1", resolved_model="m1")
            row = store._execute("SELECT * FROM runs WHERE id=?", (run_id,), fetchone=True)
            assert row["code_commit"] == "abc1234"
            assert row["resolved_model"] == "m1"
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

        store = KnowledgeStore(str(db_path))
        try:
            cols = {row[1] for row in store._connection().execute("PRAGMA table_info(runs)").fetchall()}
            assert {"code_commit", "resolved_model", "scoring_mode", "config_json"} <= cols
            row = store._execute("SELECT code_commit FROM runs WHERE experiment='pre_existing'", fetchone=True)
            assert row["code_commit"] is None
        finally:
            store.close()

"""Tests for KnowledgeStore.migrate_from_legacy() and --migrate-memory CLI flag."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ksi.memory.knowledge_store import KnowledgeStore

# ---------------------------------------------------------------------------
# Helpers: create legacy DB files with the old schema
# ---------------------------------------------------------------------------

_LEGACY_SHARED_SCHEMA = """\
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment  TEXT NOT NULL UNIQUE,
    created_at  TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS generations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    generation  INTEGER NOT NULL,
    created_at  TEXT DEFAULT (datetime('now')),
    UNIQUE(run_id, generation),
    FOREIGN KEY(run_id) REFERENCES runs(id)
);
CREATE TABLE IF NOT EXISTS agents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    agent_id    TEXT NOT NULL,
    created_at  TEXT DEFAULT (datetime('now')),
    UNIQUE(run_id, agent_id),
    FOREIGN KEY(run_id) REFERENCES runs(id)
);
"""

_LEGACY_TASK_MEMORY_SCHEMA = """\
CREATE TABLE IF NOT EXISTS task_memory_records (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                      INTEGER NOT NULL,
    generation                  INTEGER NOT NULL,
    agent_id                    TEXT NOT NULL,
    task_id                     TEXT NOT NULL,
    eval_results_json           TEXT DEFAULT '{}',
    final_model_output          TEXT DEFAULT '',
    full_memory_trace           TEXT DEFAULT '',
    full_memory_trace_condensed TEXT DEFAULT '',
    task_specific_insights_json TEXT DEFAULT '[]',
    attempt_history_json        TEXT DEFAULT '[]',
    injected_memory_md          TEXT DEFAULT '',
    forum_summary               TEXT DEFAULT '',
    created_at                  TEXT DEFAULT (datetime('now')),
    updated_at                  TEXT DEFAULT (datetime('now')),
    UNIQUE(run_id, generation, agent_id, task_id),
    FOREIGN KEY(run_id) REFERENCES runs(id)
);
"""

_LEGACY_FORUM_SCHEMA = """\
CREATE TABLE IF NOT EXISTS forum_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL,
    generation_id   INTEGER NOT NULL,
    round_num       INTEGER,
    agent_ref       INTEGER,
    message_type    TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(run_id) REFERENCES runs(id),
    FOREIGN KEY(generation_id) REFERENCES generations(id),
    FOREIGN KEY(agent_ref) REFERENCES agents(id)
);
"""

_LEGACY_DOCS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS memory_docs (
    id              TEXT PRIMARY KEY,
    run_id          INTEGER NOT NULL,
    generation_id   INTEGER,
    agent_ref       INTEGER,
    task_ref        INTEGER,
    attempt_ref     INTEGER,
    scope           TEXT NOT NULL,
    title           TEXT,
    body            TEXT NOT NULL,
    metadata_json   TEXT DEFAULT '{}',
    metadata_text   TEXT DEFAULT '',
    created_at      TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(run_id) REFERENCES runs(id)
);
"""


def _create_legacy_memory_db(path: Path) -> None:
    """Create a legacy task_memory DB with sample data."""
    conn = sqlite3.connect(str(path))
    conn.executescript(_LEGACY_SHARED_SCHEMA + _LEGACY_TASK_MEMORY_SCHEMA)
    # Insert shared run/generation/agent
    conn.execute("INSERT INTO runs (id, experiment) VALUES (1, 'test_exp')")
    conn.execute("INSERT INTO generations (id, run_id, generation) VALUES (1, 1, 1)")
    conn.execute("INSERT INTO agents (id, run_id, agent_id) VALUES (1, 1, 'agent_alpha')")
    # Insert task memory records
    conn.execute(
        "INSERT INTO task_memory_records "
        "(run_id, generation, agent_id, task_id, eval_results_json, "
        "final_model_output, full_memory_trace_condensed, task_specific_insights_json) "
        "VALUES (1, 1, 'agent_alpha', 'task_001', ?, 'output text', 'condensed trace', ?)",
        (
            json.dumps({"native_score": 0.85}),
            json.dumps(["insight A", "insight B"]),
        ),
    )
    conn.execute(
        "INSERT INTO task_memory_records "
        "(run_id, generation, agent_id, task_id, eval_results_json, "
        "final_model_output, full_memory_trace_condensed, task_specific_insights_json) "
        "VALUES (1, 1, 'agent_alpha', 'task_002', ?, 'output 2', 'trace 2', ?)",
        (
            json.dumps({"resolved": True}),
            json.dumps([{"text": "dict insight"}]),
        ),
    )
    conn.commit()
    conn.close()


def _create_legacy_forum_db(path: Path) -> None:
    """Create a legacy forum DB with insight, comment, and cluster events."""
    conn = sqlite3.connect(str(path))
    conn.executescript(_LEGACY_SHARED_SCHEMA + _LEGACY_FORUM_SCHEMA)
    conn.execute("INSERT INTO runs (id, experiment) VALUES (1, 'test_exp')")
    conn.execute("INSERT INTO generations (id, run_id, generation) VALUES (1, 1, 1)")
    conn.execute("INSERT INTO agents (id, run_id, agent_id) VALUES (1, 1, 'agent_beta')")
    # Insight event
    conn.execute(
        "INSERT INTO forum_events (run_id, generation_id, round_num, agent_ref, message_type, content) "
        "VALUES (1, 1, 1, 1, 'insight', ?)",
        (
            json.dumps(
                {
                    "text": "The task requires grid rotation",
                    "scope": "global",
                    "confidence": "high",
                    "evidence_task_ids": ["task_001"],
                    "task_id": "task_001",
                }
            ),
        ),
    )
    # Comment event
    conn.execute(
        "INSERT INTO forum_events (run_id, generation_id, round_num, agent_ref, message_type, content) "
        "VALUES (1, 1, 2, 1, 'comment', ?)",
        (json.dumps({"text": "I agree with that analysis", "task_id": "task_001"}),),
    )
    # Cluster event (distillation)
    conn.execute(
        "INSERT INTO forum_events (run_id, generation_id, round_num, agent_ref, message_type, content) "
        "VALUES (1, 1, 3, NULL, 'cluster', ?)",
        (
            json.dumps(
                {
                    "assets": [{"title": "Pattern", "body": "Grid rotation is common"}],
                    "task_id": "task_001",
                }
            ),
        ),
    )
    # Vote event (should be skipped during migration)
    conn.execute(
        "INSERT INTO forum_events (run_id, generation_id, round_num, agent_ref, message_type, content) "
        "VALUES (1, 1, 3, 1, 'vote', ?)",
        (json.dumps({"action": "approve"}),),
    )
    conn.commit()
    conn.close()


def _create_legacy_docs_db(path: Path) -> None:
    """Create a legacy docs DB with task summary docs."""
    conn = sqlite3.connect(str(path))
    conn.executescript(_LEGACY_SHARED_SCHEMA + _LEGACY_DOCS_SCHEMA)
    conn.execute("INSERT INTO runs (id, experiment) VALUES (1, 'test_exp')")
    # Task summary doc
    conn.execute(
        "INSERT INTO memory_docs (id, run_id, scope, title, body, metadata_json) "
        "VALUES ('doc_1', 1, 'task_summary', 'Task 001 Summary', "
        "'This task involves color mapping', ?)",
        (json.dumps({"task_id": "task_001", "generation": 1}),),
    )
    # Another task summary
    conn.execute(
        "INSERT INTO memory_docs (id, run_id, scope, title, body, metadata_json) "
        "VALUES ('doc_2', 1, 'task_summary', 'Task 002 Summary', "
        "'This task involves pattern matching', ?)",
        (json.dumps({"task_id": "task_002", "generation": 2}),),
    )
    # Non-task-summary doc (should be skipped)
    conn.execute(
        "INSERT INTO memory_docs (id, run_id, scope, title, body, metadata_json) "
        "VALUES ('doc_3', 1, 'raw_transcript', 'Transcript', "
        "'full agent transcript text', '{}')",
    )
    conn.commit()
    conn.close()


def _read_knowledge_rows(output_path: str) -> list[dict]:
    """Read all knowledge rows from the output KnowledgeStore DB."""
    conn = sqlite3.connect(output_path)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM knowledge ORDER BY id").fetchall()]
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Test: task_memory_records migration
# ---------------------------------------------------------------------------


class TestMigrateTaskMemory:
    def test_task_memory_records_produce_attempt_entries(self, tmp_path):
        memory_db = tmp_path / "task_memory.sqlite"
        _create_legacy_memory_db(memory_db)

        output = str(tmp_path / "knowledge.sqlite")
        count = KnowledgeStore.migrate_from_legacy(
            memory_db_path=str(memory_db),
            forum_db_path=str(tmp_path / "nonexistent_forum.sqlite"),
            docs_db_path=str(tmp_path / "nonexistent_docs.sqlite"),
            output_path=output,
            experiment="migrated",
        )
        assert count == 2

        rows = _read_knowledge_rows(output)
        assert len(rows) == 2
        assert all(r["entry_type"] == "attempt" for r in rows)
        assert all(r["source_phase"] == "execution" for r in rows)
        assert {r["external_id"] for r in rows} == {"legacy:task_memory:1", "legacy:task_memory:2"}

        # Verify content was parsed correctly
        content_0 = json.loads(rows[0]["content"])
        assert content_0["eval_results"]["native_score"] == 0.85
        assert content_0["model_output"] == "output text"
        assert content_0["trace_condensed"] == "condensed trace"
        assert content_0["insights"] == ["insight A", "insight B"]
        assert rows[0]["native_score"] == 0.85

        content_1 = json.loads(rows[1]["content"])
        assert content_1["eval_results"]["resolved"] is True
        assert rows[1]["native_score"] == 1.0


# ---------------------------------------------------------------------------
# Test: forum_events migration (insights)
# ---------------------------------------------------------------------------


class TestMigrateForumInsights:
    def test_insight_events_produce_insight_entries(self, tmp_path):
        forum_db = tmp_path / "forum.sqlite"
        _create_legacy_forum_db(forum_db)

        output = str(tmp_path / "knowledge.sqlite")
        count = KnowledgeStore.migrate_from_legacy(
            memory_db_path=str(tmp_path / "nonexistent_memory.sqlite"),
            forum_db_path=str(forum_db),
            docs_db_path=str(tmp_path / "nonexistent_docs.sqlite"),
            output_path=output,
            experiment="migrated",
        )
        # 1 insight + 1 comment + 1 cluster = 3 (vote is skipped)
        assert count == 3

        rows = _read_knowledge_rows(output)
        insight_rows = [r for r in rows if r["entry_type"] == "insight"]
        assert len(insight_rows) == 1

        content = json.loads(insight_rows[0]["content"])
        assert content["text"] == "The task requires grid rotation"
        assert content["scope"] == "global"
        assert content["confidence"] == "high"
        assert content["evidence_task_ids"] == ["task_001"]
        assert insight_rows[0]["task_id"] == "task_001"
        assert insight_rows[0]["round_num"] == 1


# ---------------------------------------------------------------------------
# Test: forum_events migration (comments)
# ---------------------------------------------------------------------------


class TestMigrateForumComments:
    def test_comment_events_produce_post_entries(self, tmp_path):
        forum_db = tmp_path / "forum.sqlite"
        _create_legacy_forum_db(forum_db)

        output = str(tmp_path / "knowledge.sqlite")
        KnowledgeStore.migrate_from_legacy(
            memory_db_path=str(tmp_path / "nonexistent_memory.sqlite"),
            forum_db_path=str(forum_db),
            docs_db_path=str(tmp_path / "nonexistent_docs.sqlite"),
            output_path=output,
            experiment="migrated",
        )

        rows = _read_knowledge_rows(output)
        post_rows = [r for r in rows if r["entry_type"] == "post"]
        assert len(post_rows) == 1

        content = json.loads(post_rows[0]["content"])
        assert content["text"] == "I agree with that analysis"
        assert post_rows[0]["task_id"] == "task_001"
        assert post_rows[0]["round_num"] == 2


# ---------------------------------------------------------------------------
# Test: memory_docs migration (task summaries)
# ---------------------------------------------------------------------------


class TestMigrateDocs:
    def test_task_summaries_produce_distillation_entries(self, tmp_path):
        docs_db = tmp_path / "task_docs.sqlite"
        _create_legacy_docs_db(docs_db)

        output = str(tmp_path / "knowledge.sqlite")
        count = KnowledgeStore.migrate_from_legacy(
            memory_db_path=str(tmp_path / "nonexistent_memory.sqlite"),
            forum_db_path=str(tmp_path / "nonexistent_forum.sqlite"),
            docs_db_path=str(docs_db),
            output_path=output,
            experiment="migrated",
        )
        # Only 2 task_summary docs (raw_transcript is skipped)
        assert count == 2

        rows = _read_knowledge_rows(output)
        assert len(rows) == 2
        assert all(r["entry_type"] == "distillation" for r in rows)
        assert all(r["source_phase"] == "condensation" for r in rows)

        content_0 = json.loads(rows[0]["content"])
        assert content_0["assets"][0]["title"] == "Task 001 Summary"
        assert content_0["assets"][0]["body"] == "This task involves color mapping"
        assert rows[0]["task_id"] == "task_001"
        assert rows[0]["generation"] == 1

        content_1 = json.loads(rows[1]["content"])
        assert rows[1]["task_id"] == "task_002"
        assert rows[1]["generation"] == 2


# ---------------------------------------------------------------------------
# Test: missing source file is skipped gracefully
# ---------------------------------------------------------------------------


class TestMissingSourceFiles:
    def test_missing_files_skipped_gracefully(self, tmp_path):
        output = str(tmp_path / "knowledge.sqlite")
        count = KnowledgeStore.migrate_from_legacy(
            memory_db_path=str(tmp_path / "does_not_exist_memory.sqlite"),
            forum_db_path=str(tmp_path / "does_not_exist_forum.sqlite"),
            docs_db_path=str(tmp_path / "does_not_exist_docs.sqlite"),
            output_path=output,
            experiment="migrated",
        )
        assert count == 0
        # Output DB is still created (empty knowledge table)
        assert Path(output).exists()
        rows = _read_knowledge_rows(output)
        assert len(rows) == 0

    def test_partial_sources(self, tmp_path):
        """Only the legacy task_memory DB exists; forum and docs are missing."""
        memory_db = tmp_path / "task_memory.sqlite"
        _create_legacy_memory_db(memory_db)

        output = str(tmp_path / "knowledge.sqlite")
        count = KnowledgeStore.migrate_from_legacy(
            memory_db_path=str(memory_db),
            forum_db_path=str(tmp_path / "missing_forum.sqlite"),
            docs_db_path=str(tmp_path / "missing_docs.sqlite"),
            output_path=output,
            experiment="migrated",
        )
        assert count == 2  # Only the 2 task_memory records


# ---------------------------------------------------------------------------
# Test: CLI --migrate-memory flag is accepted by the parser
# ---------------------------------------------------------------------------


class TestMigrateMemoryCLI:
    def test_parser_accepts_migrate_memory_flag(self):
        """Verify --migrate-memory is a valid CLI argument."""
        from ksi.cli import build_parser

        parser = build_parser()
        # Parse with minimal required args plus --migrate-memory
        args = parser.parse_args(
            [
                "--task-source",
                "arc",
                "--tasks-path",
                "/tmp/fake_tasks.json",
                "--migrate-memory",
                "/tmp/output_knowledge.sqlite",
            ]
        )
        assert args.migrate_memory == "/tmp/output_knowledge.sqlite"

    def test_parser_default_is_none(self):
        """Without --migrate-memory, the value defaults to None."""
        from ksi.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "arc",
                "--tasks-path",
                "/tmp/fake_tasks.json",
            ]
        )
        assert args.migrate_memory is None


# ---------------------------------------------------------------------------
# Test: full round-trip migration
# ---------------------------------------------------------------------------


class TestFullMigration:
    def test_all_sources_combined(self, tmp_path):
        """Migrate from all 3 legacy files at once."""
        memory_db = tmp_path / "task_memory.sqlite"
        forum_db = tmp_path / "forum.sqlite"
        docs_db = tmp_path / "task_docs.sqlite"
        _create_legacy_memory_db(memory_db)
        _create_legacy_forum_db(forum_db)
        _create_legacy_docs_db(docs_db)

        output = str(tmp_path / "knowledge.sqlite")
        count = KnowledgeStore.migrate_from_legacy(
            memory_db_path=str(memory_db),
            forum_db_path=str(forum_db),
            docs_db_path=str(docs_db),
            output_path=output,
            experiment="full_test",
        )
        # 2 task_memory + 3 forum (1 insight + 1 comment + 1 cluster) + 2 docs = 7
        assert count == 7

        rows = _read_knowledge_rows(output)
        types = {r["entry_type"] for r in rows}
        assert types == {"attempt", "insight", "post", "distillation"}

    def test_idempotent_migration(self, tmp_path):
        """Running migration twice skips rows imported by the first pass."""
        memory_db = tmp_path / "task_memory.sqlite"
        _create_legacy_memory_db(memory_db)

        output = str(tmp_path / "knowledge.sqlite")
        count1 = KnowledgeStore.migrate_from_legacy(
            memory_db_path=str(memory_db),
            forum_db_path=str(tmp_path / "no_forum.sqlite"),
            docs_db_path=str(tmp_path / "no_docs.sqlite"),
            output_path=output,
        )
        count2 = KnowledgeStore.migrate_from_legacy(
            memory_db_path=str(memory_db),
            forum_db_path=str(tmp_path / "no_forum.sqlite"),
            docs_db_path=str(tmp_path / "no_docs.sqlite"),
            output_path=output,
        )
        assert count1 == 2
        assert count2 == 0
        rows = _read_knowledge_rows(output)
        assert len(rows) == 2
        assert {r["external_id"] for r in rows} == {"legacy:task_memory:1", "legacy:task_memory:2"}

    def test_idempotent_migration_forum(self, tmp_path):
        """Forum-event migration is also idempotent (legacy:forum_event:* ids)."""
        forum_db = tmp_path / "forum.sqlite"
        _create_legacy_forum_db(forum_db)

        output = str(tmp_path / "knowledge.sqlite")
        kwargs = dict(
            memory_db_path=str(tmp_path / "no_memory.sqlite"),
            forum_db_path=str(forum_db),
            docs_db_path=str(tmp_path / "no_docs.sqlite"),
            output_path=output,
        )
        count1 = KnowledgeStore.migrate_from_legacy(**kwargs)
        count2 = KnowledgeStore.migrate_from_legacy(**kwargs)

        assert count1 == 3  # 1 insight + 1 comment + 1 cluster (vote skipped)
        assert count2 == 0
        rows = _read_knowledge_rows(output)
        assert len(rows) == 3
        assert {r["external_id"] for r in rows} == {
            "legacy:forum_event:1",
            "legacy:forum_event:2",
            "legacy:forum_event:3",
        }

    def test_idempotent_migration_docs(self, tmp_path):
        """Memory-doc migration is also idempotent (legacy:memory_doc:* ids)."""
        docs_db = tmp_path / "task_docs.sqlite"
        _create_legacy_docs_db(docs_db)

        output = str(tmp_path / "knowledge.sqlite")
        kwargs = dict(
            memory_db_path=str(tmp_path / "no_memory.sqlite"),
            forum_db_path=str(tmp_path / "no_forum.sqlite"),
            docs_db_path=str(docs_db),
            output_path=output,
        )
        count1 = KnowledgeStore.migrate_from_legacy(**kwargs)
        count2 = KnowledgeStore.migrate_from_legacy(**kwargs)

        assert count1 == 2  # 2 task_summary docs (raw_transcript skipped)
        assert count2 == 0
        rows = _read_knowledge_rows(output)
        assert len(rows) == 2
        assert {r["external_id"] for r in rows} == {
            "legacy:memory_doc:doc_1",
            "legacy:memory_doc:doc_2",
        }

"""Legacy migration helpers for :mod:`ksi.memory.knowledge_store`.

Extracted from ``knowledge_store.py`` (behaviour-preserving). These functions
migrate data from the old 3-file layout (memory / forum / docs SQLite DBs) into
a unified knowledge DB owned by :class:`KnowledgeStore`.

``KnowledgeStore.migrate_from_legacy`` is a thin ``staticmethod`` shim over
``migrate_from_legacy`` here, so the public API is unchanged.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

try:
    from ._store_common import _json_loads
except ImportError:  # pragma: no cover - script mode fallback inside container MCP
    # See ``store.py`` for the rationale: under a direct script-mode load the
    # relative import fails, so put this module's own directory on sys.path to
    # resolve the sibling ``_store_common``. No-op in the real container.
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _store_common import _json_loads  # type: ignore[no-redef]


log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Legacy migration
# ------------------------------------------------------------------


def migrate_from_legacy(
    *,
    memory_db_path: str,
    forum_db_path: str,
    docs_db_path: str,
    output_path: str,
    experiment: str = "default",
) -> int:
    """Migrate data from the old 3-file layout to a unified knowledge DB.

    Reads from the source files (read-only). Writes to *output_path*.
    Returns the number of entries migrated.

    Missing source files are skipped with a warning. The migration is
    re-runnable: rows imported from a previous run are identified by stable
    legacy external ids (``legacy:<source>:<source_row_id>``) and skipped.

    Idempotency assumes a fixed 1:1 mapping from each source file to its
    ``output_path``/``experiment``: re-running the *same* sources is a no-op,
    but migrating *different* source DBs into the same output+experiment can
    collide on ``legacy:*:<id>`` ids (the ids key on the source row PK only).
    Run offline against an output DB no live experiment is writing to — do
    not point a concurrent run at *output_path*.
    """

    try:
        from .knowledge_store import KnowledgeStore
    except ImportError:
        from knowledge_store import KnowledgeStore

    migrated = 0
    store = KnowledgeStore(output_path, default_experiment=experiment)
    try:
        # ── A. task_memory_records from memory_db_path ────────────────
        migrated += _migrate_task_memory(
            memory_db_path,
            store,
            experiment,
        )

        # ── B. forum_events from forum_db_path ────────────────────────
        migrated += _migrate_forum_events(
            forum_db_path,
            store,
            experiment,
        )

        # ── C. memory_docs (task summaries) from docs_db_path ─────────
        migrated += _migrate_docs(
            docs_db_path,
            store,
            experiment,
        )
    finally:
        store.close()

    return migrated


# -- private migration helpers -----------------------------------------


def _open_legacy_readonly(path: str) -> "sqlite3.Connection | None":
    """Open a legacy SQLite DB read-only, returning None if missing."""
    import sqlite3 as _sqlite3

    p = Path(path)
    if not p.exists():
        log.warning("[MIGRATE] Source file does not exist, skipping: %s", path)
        return None
    try:
        uri = f"file:{p.resolve()}?mode=ro"
        conn = _sqlite3.connect(uri, uri=True)
        conn.row_factory = _sqlite3.Row
        return conn
    except Exception as exc:
        log.warning("[MIGRATE] Failed to open %s: %s", path, exc)
        return None


def _migrate_task_memory(
    memory_db_path: str,
    store: "KnowledgeStore",
    experiment: str,
) -> int:
    """Migrate task_memory_records → record_attempt()."""
    conn = _open_legacy_readonly(memory_db_path)
    if conn is None:
        return 0
    count = 0
    try:
        # Check that the table exists before querying.
        tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "task_memory_records" not in tables:
            log.warning(
                "[MIGRATE] task_memory_records table not found in %s, skipping",
                memory_db_path,
            )
            return 0

        rows = conn.execute(
            "SELECT id, generation, agent_id, task_id, eval_results_json, "
            "final_model_output, full_memory_trace_condensed, "
            "task_specific_insights_json, created_at "
            "FROM task_memory_records"
        ).fetchall()
        seen_external_ids = store.bulk_has_external_ids(
            [f"legacy:task_memory:{row['id']}" for row in rows],
            experiment=experiment,
        )
        for row in rows:
            external_id = f"legacy:task_memory:{row['id']}"
            if external_id in seen_external_ids:
                continue
            eval_results = _json_loads(row["eval_results_json"], {})
            insights_raw = _json_loads(row["task_specific_insights_json"], [])
            # insights may be a list of strings or a list of dicts
            insights: list[str] = []
            for item in insights_raw if isinstance(insights_raw, list) else []:
                if isinstance(item, str):
                    insights.append(item)
                elif isinstance(item, dict):
                    insights.append(item.get("text", str(item)))
                else:
                    insights.append(str(item))

            native_score: float | None = None
            if isinstance(eval_results, dict):
                ns = eval_results.get("native_score")
                if isinstance(ns, (int, float)):
                    native_score = float(ns)
                elif isinstance(eval_results.get("resolved"), bool):
                    # Only a real ``bool`` is an authoritative verdict; a
                    # non-bool ``resolved`` (e.g. the truthy string
                    # ``"false"``) must not score a false 1.0.
                    native_score = 1.0 if eval_results["resolved"] else 0.0

            try:
                store.record_attempt(
                    task_id=str(row["task_id"]),
                    agent_id=str(row["agent_id"]),
                    generation=int(row["generation"]),
                    eval_results=eval_results,
                    model_output=str(row["final_model_output"] or ""),
                    trace_condensed=str(row["full_memory_trace_condensed"] or ""),
                    insights=insights,
                    native_score=native_score,
                    experiment=experiment,
                    external_id=external_id,
                )
                count += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("[MIGRATE] Failed to migrate task_memory row (%s): %s", external_id, exc)
    finally:
        conn.close()
    log.info("[MIGRATE] Migrated %d task_memory_records from %s", count, memory_db_path)
    return count


def _migrate_forum_events(
    forum_db_path: str,
    store: "KnowledgeStore",
    experiment: str,
) -> int:
    """Migrate forum_events → record_insight / record_post / record_distillation."""
    conn = _open_legacy_readonly(forum_db_path)
    if conn is None:
        return 0
    count = 0
    try:
        tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "forum_events" not in tables:
            log.warning(
                "[MIGRATE] forum_events table not found in %s, skipping",
                forum_db_path,
            )
            return 0

        # The forum DB has the same shared schema (runs, generations, agents)
        # alongside forum_events.  We join to resolve generation and agent_id.
        query = (
            "SELECT fe.id, fe.round_num, "
            "COALESCE(a.agent_id, '') AS agent_id, "
            "fe.message_type, fe.content, fe.created_at, "
            "g.generation "
            "FROM forum_events fe "
            "JOIN generations g ON fe.generation_id = g.id "
            "LEFT JOIN agents a ON fe.agent_ref = a.id"
        )
        rows = conn.execute(query).fetchall()
        seen_external_ids = store.bulk_has_external_ids(
            [f"legacy:forum_event:{row['id']}" for row in rows],
            experiment=experiment,
        )
        for row in rows:
            external_id = f"legacy:forum_event:{row['id']}"
            if external_id in seen_external_ids:
                continue
            message_type = str(row["message_type"] or "").strip()
            content_raw = _json_loads(row["content"], {})
            agent_id = str(row["agent_id"] or "unknown")
            generation = int(row["generation"])
            round_num = int(row["round_num"] or 0)

            try:
                if message_type == "insight":
                    text = ""
                    scope = "task"
                    confidence = "medium"
                    evidence_task_ids: list[str] = []
                    task_id = "__migrated__"
                    if isinstance(content_raw, dict):
                        text = str(content_raw.get("text", content_raw.get("insight", "")))
                        scope = str(content_raw.get("scope", "task"))
                        confidence = str(content_raw.get("confidence", "medium"))
                        evidence_task_ids = content_raw.get("evidence_task_ids", [])
                        task_id = str(content_raw.get("task_id", "__migrated__"))
                    elif isinstance(content_raw, str):
                        text = content_raw
                    if not text:
                        text = json.dumps(content_raw) if content_raw else "(empty)"

                    store.record_insight(
                        task_id=task_id,
                        agent_id=agent_id,
                        generation=generation,
                        text=text,
                        scope=scope,
                        confidence=confidence,
                        evidence_task_ids=evidence_task_ids if isinstance(evidence_task_ids, list) else [],
                        round_num=round_num,
                        experiment=experiment,
                        external_id=external_id,
                    )
                    count += 1

                elif message_type == "comment":
                    text = ""
                    task_id = "__migrated__"
                    if isinstance(content_raw, dict):
                        text = str(content_raw.get("text", content_raw.get("comment", "")))
                        task_id = str(content_raw.get("task_id", "__migrated__"))
                    elif isinstance(content_raw, str):
                        text = content_raw
                    if not text:
                        text = json.dumps(content_raw) if content_raw else "(empty)"

                    store.record_post(
                        task_id=task_id,
                        agent_id=agent_id,
                        generation=generation,
                        text=text,
                        round_num=round_num,
                        experiment=experiment,
                        external_id=external_id,
                    )
                    count += 1

                elif message_type == "cluster":
                    assets: list[dict] = []
                    task_id = "__migrated__"
                    if isinstance(content_raw, dict):
                        raw_assets = content_raw.get("assets", [])
                        task_id = str(content_raw.get("task_id", "__migrated__"))
                        if isinstance(raw_assets, list):
                            assets = [a if isinstance(a, dict) else {"text": str(a)} for a in raw_assets]
                        else:
                            assets = [{"text": str(raw_assets)}]
                    else:
                        assets = [{"text": json.dumps(content_raw)}]

                    store.record_distillation(
                        task_id=task_id,
                        generation=generation,
                        assets=assets,
                        experiment=experiment,
                        external_id=external_id,
                    )
                    count += 1
                # Other message_types (vote, claim, etc.) are not migrated.
            except Exception as exc:  # noqa: BLE001
                log.warning("[MIGRATE] Failed to migrate forum_event row (%s): %s", external_id, exc)
    finally:
        conn.close()
    log.info("[MIGRATE] Migrated %d forum_events from %s", count, forum_db_path)
    return count


def _migrate_docs(
    docs_db_path: str,
    store: "KnowledgeStore",
    experiment: str,
) -> int:
    """Migrate memory_docs (task_summary scope) → record_distillation."""
    conn = _open_legacy_readonly(docs_db_path)
    if conn is None:
        return 0
    count = 0
    try:
        tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "memory_docs" not in tables:
            log.warning(
                "[MIGRATE] memory_docs table not found in %s, skipping",
                docs_db_path,
            )
            return 0

        rows = conn.execute(
            "SELECT id, scope, title, body, metadata_json, created_at FROM memory_docs WHERE scope = 'task_summary'"
        ).fetchall()
        seen_external_ids = store.bulk_has_external_ids(
            [f"legacy:memory_doc:{row['id']}" for row in rows],
            experiment=experiment,
        )
        for row in rows:
            external_id = f"legacy:memory_doc:{row['id']}"
            if external_id in seen_external_ids:
                continue
            title = str(row["title"] or "")
            body = str(row["body"] or "")
            combined = f"{title}\n{body}".strip() if title else body.strip()
            if not combined:
                continue

            metadata = _json_loads(row["metadata_json"], {})
            task_id = "__migrated__"
            generation = 0
            if isinstance(metadata, dict):
                task_id = str(metadata.get("task_id", "__migrated__"))
                gen_val = metadata.get("generation", 0)
                if isinstance(gen_val, (int, float)):
                    generation = int(gen_val)

            try:
                store.record_distillation(
                    task_id=task_id,
                    generation=generation,
                    assets=[{"title": title, "body": body}],
                    experiment=experiment,
                    external_id=external_id,
                )
                count += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("[MIGRATE] Failed to migrate memory_doc row (%s): %s", external_id, exc)
    finally:
        conn.close()
    log.info("[MIGRATE] Migrated %d memory_docs from %s", count, docs_db_path)
    return count

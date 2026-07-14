"""Hierarchical SQLite runtime memory store.

Schema (no legacy migrations):
- runs, generations, agents, tasks
- assignments, attempts, attempt_artifacts
- memory_docs
- forum_events

Public API is intentionally small and task/runtime-centric.
"""

from __future__ import annotations

import logging
import os
import queue
import sqlite3
import threading
import time
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    from ..orchestrator.scoring import score_from_eval_results
except Exception:  # pragma: no cover - script mode fallback inside container MCP
    # The container MCP server runs ``python3 /app/memory/mcp_server.py`` with
    # only ``src/kcsi/memory/`` mounted, so the ``..orchestrator`` package is not
    # importable. Mirror the canonical helper inline (same precedence) — the
    # package-mode import above stays the single source of truth.
    def score_from_eval_results(eval_r: dict[str, Any]) -> float | None:  # type: ignore[misc]
        ns = eval_r.get("native_score")
        if isinstance(ns, (int, float)):
            return float(ns)
        # Mirror the canonical guard: only a real ``bool`` is an authoritative
        # verdict (a non-bool ``resolved`` falls through).
        if isinstance(eval_r.get("resolved"), bool):
            return 1.0 if eval_r["resolved"] else 0.0
        instance_report = eval_r.get("instance_report")
        if isinstance(instance_report, dict):
            if isinstance(instance_report.get("resolved"), bool):
                return 1.0 if instance_report["resolved"] else 0.0
        if "pass" in eval_r and eval_r["pass"] is not None:
            return 1.0 if bool(eval_r["pass"]) else 0.0
        return None


try:
    from ._store_common import (
        WriteIndeterminateError,
        _apply_init_pragmas,
        _cleanup_stale_locks,
        _get_process_db_lock,
        _json_dumps,
        _json_loads,
        _locked_guard,
        _wal_checkpoint,
    )
except ImportError:  # pragma: no cover - script mode fallback inside container MCP
    # Container MCP runs ``python3 /app/memory/mcp_server.py`` with only
    # ``src/kcsi/memory/`` mounted (no parent package), so the relative import
    # above fails. Put this module's own directory on sys.path so the sibling
    # ``_store_common`` resolves as a top-level module. No-op in the real
    # container (its dir is already ``sys.path[0]``); only exercised under a
    # direct ``spec_from_file_location`` load.
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _store_common import (  # type: ignore[no-redef]
        WriteIndeterminateError,
        _apply_init_pragmas,
        _cleanup_stale_locks,
        _get_process_db_lock,
        _json_dumps,
        _json_loads,
        _locked_guard,
        _wal_checkpoint,
    )


try:
    from ..trace_events import append_trace_event, get_trace_dir
except Exception:  # pragma: no cover - script mode fallback inside container MCP

    def append_trace_event(*args: Any, **kwargs: Any) -> None:
        return None

    def get_trace_dir() -> str:
        return ""


log = logging.getLogger(__name__)


# Per-store process-lock registry. AB-BA invariant: this dict MUST stay
# separate from KnowledgeStore's identical registry — the shared helper in
# ``_store_common`` only shares code, never the registry. See that module's
# docstring and tests/test_sqlite_persistence_deadlock.py.
_PROCESS_DB_LOCKS: dict[str, threading.RLock] = {}
_PROCESS_DB_LOCKS_GUARD = threading.Lock()


def _extract_score(eval_r: dict[str, Any]) -> float | None:
    """Extract a numeric score from an eval results dict.

    Mirrors the generic fallback chain in ``_score_from_eval``
    (``src/kcsi/orchestrator/engine.py``) so that ``get_best_scores()``
    correctly recognises results from all evaluators -- including
    SWE-bench which stores ``{"resolved": true}`` without a
    ``native_score`` key.

    Precedence:
    1. ``native_score`` (numeric)
    2. ``resolved`` (bool -> 1.0 / 0.0)
    3. ``instance_report.resolved`` (bool -> 1.0 / 0.0)
    4. ``pass`` (bool -> 1.0 / 0.0)
    """
    return score_from_eval_results(eval_r)


def _flatten_metadata(metadata: dict[str, Any]) -> str:
    tokens: list[str] = []

    def _walk(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, dict):
            for vv in v.values():
                _walk(vv)
            return
        if isinstance(v, list):
            for vv in v:
                _walk(vv)
            return
        txt = str(v).strip()
        if txt:
            tokens.append(txt)

    _walk(metadata)
    return " ".join(tokens)


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment     TEXT NOT NULL UNIQUE,
    created_at     TEXT DEFAULT (datetime('now')),
    code_commit    TEXT,
    resolved_model TEXT,
    scoring_mode   TEXT,
    config_json    TEXT
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

CREATE INDEX IF NOT EXISTS idx_generations_run ON generations(run_id, generation);
CREATE INDEX IF NOT EXISTS idx_agents_run ON agents(run_id, agent_id);
"""

_SCHEMA_TASK_MEMORY = """\
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

CREATE INDEX IF NOT EXISTS idx_task_memory_records_run_task ON task_memory_records(run_id, task_id, generation, updated_at);
CREATE INDEX IF NOT EXISTS idx_tmr_task_id ON task_memory_records(task_id);
"""

_SCHEMA_TASK_DOCS = """\
CREATE TABLE IF NOT EXISTS arc_task_refs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL,
    task_id         TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(run_id, task_id),
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_arc_task_refs_run_task ON arc_task_refs(run_id, task_id);

CREATE TABLE IF NOT EXISTS tasks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL,
    task_id       TEXT NOT NULL,
    repo          TEXT,
    prompt_preview TEXT,
    metadata_json TEXT DEFAULT '{}',
    created_at    TEXT DEFAULT (datetime('now')),
    UNIQUE(run_id, task_id),
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS assignments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id INTEGER NOT NULL,
    agent_ref   INTEGER NOT NULL,
    task_ref    INTEGER NOT NULL,
    status      TEXT DEFAULT 'created',
    created_at  TEXT DEFAULT (datetime('now')),
    started_at  TEXT,
    ended_at    TEXT,
    UNIQUE(generation_id, agent_ref, task_ref),
    FOREIGN KEY(generation_id) REFERENCES generations(id),
    FOREIGN KEY(agent_ref) REFERENCES agents(id),
    FOREIGN KEY(task_ref) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS attempts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    assignment_id   INTEGER NOT NULL,
    attempt_no      INTEGER NOT NULL,
    model_output    TEXT,
    eval_result_json TEXT DEFAULT '{}',
    native_score    REAL,
    tool_trace_json TEXT DEFAULT '[]',
    runtime_meta_json TEXT DEFAULT '{}',
    error_text      TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(assignment_id, attempt_no),
    FOREIGN KEY(assignment_id) REFERENCES assignments(id)
);

CREATE TABLE IF NOT EXISTS attempt_artifacts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id      INTEGER NOT NULL,
    artifact_type   TEXT NOT NULL,
    content         TEXT NOT NULL,
    content_bytes   INTEGER NOT NULL,
    metadata_json   TEXT DEFAULT '{}',
    created_at      TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(attempt_id) REFERENCES attempts(id)
);

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
    FOREIGN KEY(run_id) REFERENCES runs(id),
    FOREIGN KEY(generation_id) REFERENCES generations(id),
    FOREIGN KEY(agent_ref) REFERENCES agents(id),
    FOREIGN KEY(task_ref) REFERENCES tasks(id),
    FOREIGN KEY(attempt_ref) REFERENCES attempts(id)
);

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

CREATE TABLE IF NOT EXISTS token_phases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES runs(id),
    generation INTEGER NOT NULL,
    phase TEXT NOT NULL,
    agent_ref TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    -- SQLite keeps the shorter column name, but writers map it from
    -- TokenUsage.cache_creation_input_tokens. Do not treat the naming split as
    -- two different counters.
    cache_creation_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tasks_run ON tasks(run_id, task_id);
CREATE INDEX IF NOT EXISTS idx_assignments_gen ON assignments(generation_id, agent_ref, task_ref);
CREATE INDEX IF NOT EXISTS idx_attempts_assignment ON attempts(assignment_id, attempt_no);
CREATE INDEX IF NOT EXISTS idx_artifacts_attempt ON attempt_artifacts(attempt_id, artifact_type);
CREATE INDEX IF NOT EXISTS idx_memory_docs_scope ON memory_docs(scope, created_at);
CREATE INDEX IF NOT EXISTS idx_forum_events_gen ON forum_events(generation_id, id);
CREATE INDEX IF NOT EXISTS idx_token_phases_run_gen ON token_phases(run_id, generation);
"""


VALID_FORUM_MESSAGE_TYPES = frozenset(
    {
        "error",
        "insight",
        "post",
        "comment",
        "done",
        "cluster",
        "vote",
        "rebuttal",
        "claim",
        "proposal",
        "diagnostic",
        "forum_round",
    }
)


class MemoryStore:
    @staticmethod
    def _cleanup_stale_locks(directory: str | Path, max_age_seconds: int = 3600) -> None:
        """Remove ``.sqlite.lock`` files older than *max_age_seconds* in *directory*.

        Called at the start of ``__init__`` to avoid accumulating stale lock
        files from crashed processes. Delegates to the shared implementation in
        ``_store_common`` (identical between both stores).
        """
        _cleanup_stale_locks(directory, max_age_seconds)

    def __init__(
        self,
        db_path: str,
        *,
        default_experiment: str = "default",
        read_only: bool = False,
    ) -> None:
        self._db_path = db_path
        self._default_experiment = (default_experiment or "default").strip() or "default"
        self._read_only = bool(read_only)

        if not self._read_only:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            MemoryStore._cleanup_stale_locks(Path(db_path).parent)
        self._db_key = str(Path(db_path).resolve())
        self._trace_dir = get_trace_dir() or str((os.environ.get("KCSI_TRACE_DIR", "") or "").strip())
        # Single connection per process; cross-process coordination uses an advisory lock file.
        if self._read_only:
            uri = f"file:{Path(db_path).resolve()}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=30.0)
        else:
            self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._process_lock = _get_process_db_lock(self._db_key, _PROCESS_DB_LOCKS, _PROCESS_DB_LOCKS_GUARD)
        self._lock_state = threading.local()
        self._writer_queue: queue.Queue | None = None
        self._writer_thread: threading.Thread | None = None
        self._writer_thread_id: int | None = None
        self._batch_mode: bool = False
        self._lock_file_path = f"{db_path}.lock"
        self._lock_fd = None
        if not self._read_only:
            Path(self._lock_file_path).touch(exist_ok=True)
            self._lock_fd = open(self._lock_file_path, "a+b")

            with self._locked():
                # Set WAL mode and busy timeout BEFORE schema creation so that
                # concurrent processes opening the same DB during init already
                # see WAL journal mode and don't fall back to delete-journal.
                _apply_init_pragmas(self._conn)
                self._conn.executescript(_SCHEMA + _SCHEMA_TASK_MEMORY + _SCHEMA_TASK_DOCS)
                self._ensure_compat_schema_locked()
                self._conn.commit()
            self._start_writer()

    def _locked(self):
        """Thread + process lock guard for SQLite operations.

        This prevents concurrent host/container writers from stepping on each
        other on bind-mounted DB files. Delegates to the shared
        ``_store_common._locked_guard``; ``self._process_lock`` comes from this
        module's own per-store ``_PROCESS_DB_LOCKS`` registry (the AB-BA invariant).
        """
        return _locked_guard(
            lock_state=self._lock_state,
            process_lock=self._process_lock,
            thread_lock=self._lock,
            lock_fd=self._lock_fd,
            logger=log,
        )

    @contextmanager
    def _batched(self):
        """Suppress intermediate commits from helpers; commit once at the end.

        Holds ``_locked()`` for the whole batch so the implicit SQLite write
        transaction opened by the first INSERT is committed before the shared
        process-level lock is released. Without this, two ``MemoryStore``
        instances sharing the same DB (e.g. engine._memory_store and
        SqlitePersistence._store) can AB-BA deadlock: thread A's uncommitted
        txn on conn_A blocks thread B's INSERT on conn_B while thread B holds
        the process_lock that A needs to reacquire for its next _ensure_*.

        Transactional integrity: if the wrapped block raises, the OUTERMOST
        ``_batched()`` block rolls back the connection before propagating the
        exception, so a multi-statement closure that fails partway (e.g.
        ``insert_task_summary`` whose artifact insert fails after the
        memory-doc row landed) does not leave a half-applied transaction. Inner
        ``_batched()`` blocks (``was_batch=True``) defer commit/rollback to the
        outer owner.
        """
        was_batch = self._batch_mode
        self._batch_mode = True
        if was_batch:
            try:
                yield
            finally:
                self._batch_mode = was_batch
            return
        with self._locked():
            try:
                yield
            except BaseException:
                self._batch_mode = was_batch
                try:
                    self._conn.rollback()
                except Exception:
                    log.warning(
                        "[MemoryStore] rollback after _batched failure raised; connection may be in inconsistent state",
                        exc_info=True,
                    )
                raise
            self._batch_mode = was_batch
            self._conn.commit()

    def _commit(self) -> None:
        """Commit unless we are inside a _batched() block."""
        if not self._batch_mode:
            self._conn.commit()

    @property
    def db_path(self) -> str:
        return self._db_path

    def _start_writer(self) -> None:
        if self._read_only or self._writer_thread is not None:
            return
        writer_queue: queue.Queue = queue.Queue()
        self._writer_queue = writer_queue
        self_ref = weakref.ref(self)
        db_name = Path(self._db_path).name

        def _worker() -> None:
            store = self_ref()
            if store is not None:
                store._writer_thread_id = threading.get_ident()
            store = None
            while True:
                item = writer_queue.get()
                if item is None:
                    break
                fn, done, box, claim = item
                # A submitter whose stall timeout fired cancels its write by
                # winning the claim first; execute only if the worker wins it,
                # so a timed-out write can never apply later.
                if claim.acquire(blocking=False):
                    try:
                        box["result"] = fn()
                    except Exception as exc:  # noqa: BLE001
                        box["error"] = exc
                    finally:
                        done.set()
                item = None
                fn = None
                done = None
                box = None
                claim = None
            store = self_ref()
            if store is not None:
                store._writer_thread_id = None
            store = None

        self._writer_thread = threading.Thread(
            target=_worker,
            name=f"MemoryStoreWriter[{db_name}]",
            daemon=True,
        )
        self._writer_thread.start()

    def _run_write(self, fn):
        if self._read_only:
            return fn()
        if threading.get_ident() == self._writer_thread_id:
            return fn()
        if self._writer_queue is None:
            return fn()
        # Retry SQLite "database is locked" transients: another connection
        # (e.g., read-heavy auditor or WAL checkpoint) may briefly hold the
        # write lock. Backoff 0.2s, 0.4s, 0.8s; other errors propagate.
        last_error: Exception | None = None
        for attempt in range(3):
            done = threading.Event()
            box: dict[str, Any] = {}
            claim = threading.Lock()
            self._writer_queue.put((fn, done, box, claim))
            _queue_depth = self._writer_queue.qsize() if self._writer_queue else 0
            _timeout = max(180.0, 5.0 * (_queue_depth + 1))
            done.wait(timeout=_timeout)
            if not done.is_set():
                if claim.acquire(blocking=False):
                    # Won the claim: the worker will skip this closure, so the
                    # write is guaranteed not to apply and callers may safely
                    # retry.
                    raise RuntimeError(
                        f"MemoryStore writer thread did not respond within {_timeout:.0f}s "
                        f"(queue depth was {_queue_depth}); write cancelled"
                    )
                # The worker claimed the closure and is executing it right
                # now; it cannot be cancelled, so give it one more window.
                done.wait(timeout=_timeout)
                if not done.is_set():
                    raise WriteIndeterminateError(
                        f"MemoryStore writer thread still executing a write after {2 * _timeout:.0f}s "
                        f"(queue depth was {_queue_depth}); the write may still be applied — do not retry"
                    )
            if "error" not in box:
                return box.get("result")
            err = box["error"]
            if isinstance(err, sqlite3.OperationalError) and "database is locked" in str(err):
                last_error = err
                time.sleep(0.2 * (2**attempt))
                continue
            raise err
        assert last_error is not None
        raise last_error

    # ---------- core helpers ----------

    def _execute(self, sql: str, params: tuple = (), *, fetchall: bool = False, fetchone: bool = False):
        sql_kind = str(sql or "").lstrip().split(None, 1)[0].upper() if str(sql or "").strip() else ""

        def _op():
            with self._locked():
                cur = self._conn.execute(sql, params)
                if fetchall:
                    return [dict(row) for row in cur.fetchall()]
                if fetchone:
                    row = cur.fetchone()
                    return dict(row) if row else None
                if not self._read_only:
                    self._commit()
                return None

        if not self._read_only and sql_kind in {"INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE", "DROP", "ALTER"}:
            return self._run_write(_op)
        return _op()

    def _legacy_task_summaries_available(self) -> bool:
        row = self._execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name = 'task_summaries'",
            fetchone=True,
        )
        return bool(row)

    def _trace_sqlite_event(self, event: str, payload: dict[str, Any]) -> None:
        append_trace_event(
            self._trace_dir,
            "sqlite_events.jsonl",
            {
                "event": event,
                "db_path": self._db_path,
                **payload,
            },
        )

    def _ensure_compat_schema_locked(self) -> None:
        """Create compatibility views. MUST only be called from __init__
        before _start_writer() — executes DDL directly on the connection
        without routing through the writer queue."""
        row = self._conn.execute("SELECT type FROM sqlite_master WHERE name = 'task_summaries' LIMIT 1").fetchone()
        if row is None:
            self._conn.execute(
                """
                CREATE VIEW IF NOT EXISTS task_summaries AS
                SELECT
                    md.id AS id,
                    r.experiment AS experiment,
                    COALESCE(ag.agent_id, '') AS agent_id,
                    COALESCE(g.generation, 0) AS generation,
                    COALESCE(t.task_id, '') AS task_id,
                    t.repo AS repo,
                    COALESCE(json_extract(md.metadata_json, '$.approach'), '') AS approach,
                    COALESCE(json_extract(md.metadata_json, '$.key_files'), '[]') AS key_files,
                    COALESCE(json_extract(md.metadata_json, '$.outcome'), '') AS outcome,
                    json_extract(md.metadata_json, '$.score') AS score,
                    COALESCE(json_extract(md.metadata_json, '$.lessons'), '[]') AS lessons,
                    md.created_at AS created_at
                FROM memory_docs md
                JOIN runs r ON r.id = md.run_id
                LEFT JOIN generations g ON g.id = md.generation_id
                LEFT JOIN agents ag ON ag.id = md.agent_ref
                LEFT JOIN tasks t ON t.id = md.task_ref
                WHERE md.scope = 'task_summary'
                """
            )
        row = self._conn.execute("SELECT type FROM sqlite_master WHERE name = 'raw_transcripts' LIMIT 1").fetchone()
        if row is None:
            self._conn.execute(
                """
                CREATE VIEW IF NOT EXISTS raw_transcripts AS
                SELECT
                    aa.id AS id,
                    r.experiment AS experiment,
                    COALESCE(ag.agent_id, '') AS agent_id,
                    COALESCE(g.generation, 0) AS generation,
                    COALESCE(t.task_id, '') AS task_id,
                    aa.content AS content,
                    aa.content_bytes AS bytes,
                    at.tool_trace_json AS tool_trace_json,
                    aa.created_at AS created_at
                FROM attempt_artifacts aa
                JOIN attempts at ON at.id = aa.attempt_id
                JOIN assignments asg ON asg.id = at.assignment_id
                JOIN generations g ON g.id = asg.generation_id
                JOIN runs r ON r.id = g.run_id
                LEFT JOIN agents ag ON ag.id = asg.agent_ref
                LEFT JOIN tasks t ON t.id = asg.task_ref
                WHERE aa.artifact_type = 'transcript'
                """
            )
        # ALTER TABLE fallbacks for columns added after initial schema.
        for col_name, col_def in [
            ("injected_memory_md", "TEXT DEFAULT ''"),
            ("forum_summary", "TEXT DEFAULT ''"),
        ]:
            try:
                self._conn.execute(f"ALTER TABLE task_memory_records ADD COLUMN {col_name} {col_def}")
            except sqlite3.OperationalError:
                pass  # column already exists
        # ALTER TABLE fallbacks for runs provenance columns.
        for col_name, col_def in [
            ("code_commit", "TEXT"),
            ("resolved_model", "TEXT"),
            ("scoring_mode", "TEXT"),
            ("config_json", "TEXT"),
        ]:
            try:
                self._conn.execute(f"ALTER TABLE runs ADD COLUMN {col_name} {col_def}")
            except sqlite3.OperationalError:
                pass  # column already exists

    def _experiment(self, experiment: str | None = None) -> str:
        exp = (experiment or self._default_experiment or "default").strip()
        return exp or "default"

    def set_default_experiment(self, experiment: str) -> None:
        """Public setter for the default experiment used by unscoped writes.

        Lets external observers (e.g. ``SqlitePersistence``) retarget a live
        store after a late experiment-name change without reaching into the
        private ``_default_experiment`` attribute.
        """
        self._default_experiment = experiment

    def has_experiment(self, experiment: str | None = None) -> bool:
        """Return True if an experiment with this name already exists in the DB."""
        return self._find_run(experiment) is not None

    def next_experiment_name(self, base: str) -> str:
        """Find the next available experiment name by appending _2, _3, … on collision."""
        if not self.has_experiment(base):
            return base
        n = 2
        while True:
            candidate = f"{base}_{n}"
            if not self.has_experiment(candidate):
                return candidate
            n += 1

    def ensure_run(
        self,
        experiment: str | None = None,
        *,
        code_commit: str | None = None,
        resolved_model: str | None = None,
        scoring_mode: str | None = None,
        config_json: str | None = None,
    ) -> int:
        """Public alias for :meth:`_ensure_run`.

        Resolves (creating if needed) the ``runs`` row id for ``experiment``
        so callers in other modules (e.g. the orchestrator engine flushing
        token usage) need not reach into the private method. When any of
        ``code_commit``/``resolved_model``/``scoring_mode``/``config_json`` is
        provided (not ``None``), it is stamped onto the row via ``COALESCE`` so a
        repeat call omitting the kwargs never clobbers an already-stamped value
        back to NULL.
        """
        return self._ensure_run(
            experiment,
            code_commit=code_commit,
            resolved_model=resolved_model,
            scoring_mode=scoring_mode,
            config_json=config_json,
        )

    def _ensure_run(
        self,
        experiment: str | None = None,
        *,
        code_commit: str | None = None,
        resolved_model: str | None = None,
        scoring_mode: str | None = None,
        config_json: str | None = None,
    ) -> int:
        exp = self._experiment(experiment)
        with self._locked():
            self._conn.execute("INSERT OR IGNORE INTO runs (experiment) VALUES (?)", (exp,))
            row = self._conn.execute("SELECT id FROM runs WHERE experiment = ?", (exp,)).fetchone()
            if not row:
                raise RuntimeError(f"failed to resolve run for experiment={exp}")
            run_id = int(row["id"])
            if (
                code_commit is not None
                or resolved_model is not None
                or scoring_mode is not None
                or config_json is not None
            ):
                # Write-once provenance (mirrors KnowledgeStore.ensure_run): keep
                # the first stamp so a --resume under a changed HEAD/model never
                # overwrites the audit row. The drift WARNING is emitted once, by
                # the authoritative KnowledgeStore, not duplicated here.
                self._conn.execute(
                    "UPDATE runs SET code_commit=COALESCE(code_commit, ?), "
                    "resolved_model=COALESCE(resolved_model, ?), "
                    "scoring_mode=COALESCE(scoring_mode, ?), "
                    "config_json=COALESCE(config_json, ?) WHERE id=?",
                    (code_commit, resolved_model, scoring_mode, config_json, run_id),
                )
            self._commit()
        return run_id

    def _find_run(self, experiment: str | None = None) -> int | None:
        exp = self._experiment(experiment)
        row = self._execute(
            "SELECT id FROM runs WHERE experiment = ?",
            (exp,),
            fetchone=True,
        )
        return int(row["id"]) if row else None

    def _ensure_generation(self, run_id: int, generation: int) -> int:
        with self._locked():
            self._conn.execute(
                "INSERT OR IGNORE INTO generations (run_id, generation) VALUES (?, ?)",
                (run_id, int(generation)),
            )
            row = self._conn.execute(
                "SELECT id FROM generations WHERE run_id = ? AND generation = ?",
                (run_id, int(generation)),
            ).fetchone()
            self._commit()
        if not row:
            raise RuntimeError("failed to resolve generation id")
        return int(row["id"])

    def _ensure_agent(self, run_id: int, agent_id: str) -> int:
        with self._locked():
            self._conn.execute(
                "INSERT OR IGNORE INTO agents (run_id, agent_id) VALUES (?, ?)",
                (run_id, agent_id),
            )
            row = self._conn.execute(
                "SELECT id FROM agents WHERE run_id = ? AND agent_id = ?",
                (run_id, agent_id),
            ).fetchone()
            self._commit()
        if not row:
            raise RuntimeError("failed to resolve agent id")
        return int(row["id"])

    def _ensure_task(
        self,
        run_id: int,
        task_id: str,
        *,
        repo: str | None = None,
        prompt_preview: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        metadata_json = _json_dumps(metadata or {})
        with self._locked():
            self._conn.execute(
                "INSERT OR IGNORE INTO tasks (run_id, task_id, repo, prompt_preview, metadata_json) VALUES (?, ?, ?, ?, ?)",
                (run_id, task_id, repo or "", prompt_preview or "", metadata_json),
            )
            if repo or prompt_preview or metadata:
                self._conn.execute(
                    "UPDATE tasks SET repo = COALESCE(NULLIF(?, ''), repo), prompt_preview = COALESCE(NULLIF(?, ''), prompt_preview), metadata_json = CASE WHEN ? != '{}' THEN ? ELSE metadata_json END WHERE run_id = ? AND task_id = ?",
                    (repo or "", prompt_preview or "", metadata_json, metadata_json, run_id, task_id),
                )
            row = self._conn.execute(
                "SELECT id FROM tasks WHERE run_id = ? AND task_id = ?",
                (run_id, task_id),
            ).fetchone()
            self._commit()
        if not row:
            raise RuntimeError("failed to resolve task id")
        return int(row["id"])

    def _ensure_assignment(self, generation_id: int, agent_ref: int, task_ref: int, *, status: str = "created") -> int:
        with self._locked():
            self._conn.execute(
                "INSERT OR IGNORE INTO assignments (generation_id, agent_ref, task_ref, status) VALUES (?, ?, ?, ?)",
                (generation_id, agent_ref, task_ref, status),
            )
            self._conn.execute(
                "UPDATE assignments SET status = ? WHERE generation_id = ? AND agent_ref = ? AND task_ref = ?",
                (status, generation_id, agent_ref, task_ref),
            )
            row = self._conn.execute(
                "SELECT id FROM assignments WHERE generation_id = ? AND agent_ref = ? AND task_ref = ?",
                (generation_id, agent_ref, task_ref),
            ).fetchone()
            self._commit()
        if not row:
            raise RuntimeError("failed to resolve assignment id")
        return int(row["id"])

    def _latest_attempt_id(self, assignment_id: int) -> int | None:
        row = self._execute(
            "SELECT id FROM attempts WHERE assignment_id = ? ORDER BY attempt_no DESC, id DESC LIMIT 1",
            (assignment_id,),
            fetchone=True,
        )
        return int(row["id"]) if row else None

    def _insert_attempt(
        self,
        *,
        assignment_id: int,
        model_output: str | None = None,
        eval_result: dict[str, Any] | None = None,
        native_score: float | None = None,
        tool_trace_json: str = "[]",
        runtime_meta: dict[str, Any] | None = None,
        error_text: str | None = None,
    ) -> int:
        # Atomic INSERT: compute attempt_no via subquery to avoid TOCTOU race
        # between a separate SELECT MAX and the INSERT.
        with self._locked():
            cur = self._conn.execute(
                "INSERT INTO attempts (assignment_id, attempt_no, model_output, eval_result_json, native_score, tool_trace_json, runtime_meta_json, error_text)"
                " VALUES (?, (SELECT COALESCE(MAX(attempt_no), 0) + 1 FROM attempts WHERE assignment_id = ?), ?, ?, ?, ?, ?, ?)",
                (
                    assignment_id,
                    assignment_id,
                    model_output,
                    _json_dumps(eval_result or {}),
                    native_score,
                    tool_trace_json or "[]",
                    _json_dumps(runtime_meta or {}),
                    error_text,
                ),
            )
            self._commit()
            return int(cur.lastrowid)

    def _insert_artifact(
        self,
        *,
        attempt_id: int,
        artifact_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with self._locked():
            cur = self._conn.execute(
                "INSERT INTO attempt_artifacts (attempt_id, artifact_type, content, content_bytes, metadata_json) VALUES (?, ?, ?, ?, ?)",
                (
                    attempt_id,
                    artifact_type,
                    content,
                    len(content.encode("utf-8")),
                    _json_dumps(metadata or {}),
                ),
            )
            self._commit()
            return int(cur.lastrowid)

    def _upsert_memory_doc(
        self,
        *,
        doc_id: str,
        run_id: int,
        generation_id: int | None,
        agent_ref: int | None,
        task_ref: int | None,
        attempt_ref: int | None,
        scope: str,
        title: str,
        body: str,
        metadata: dict[str, Any],
    ) -> int:
        metadata_json = _json_dumps(metadata)
        metadata_text = _flatten_metadata(metadata)
        with self._locked():
            self._conn.execute(
                """
                INSERT INTO memory_docs
                (id, run_id, generation_id, agent_ref, task_ref, attempt_ref, scope, title, body, metadata_json, metadata_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    run_id=excluded.run_id,
                    generation_id=excluded.generation_id,
                    agent_ref=excluded.agent_ref,
                    task_ref=excluded.task_ref,
                    attempt_ref=excluded.attempt_ref,
                    scope=excluded.scope,
                    title=excluded.title,
                    body=excluded.body,
                    metadata_json=excluded.metadata_json,
                    metadata_text=excluded.metadata_text
                """,
                (
                    doc_id,
                    run_id,
                    generation_id,
                    agent_ref,
                    task_ref,
                    attempt_ref,
                    scope,
                    title,
                    body,
                    metadata_json,
                    metadata_text,
                ),
            )
            row = self._conn.execute("SELECT rowid FROM memory_docs WHERE id = ?", (doc_id,)).fetchone()
            self._commit()
        if not row:
            raise RuntimeError("failed to resolve memory_doc rowid")
        return int(row["rowid"])

    # ---------- assignment timing API ----------

    def mark_assignment_started(
        self,
        *,
        experiment: str,
        generation: int,
        agent_id: str,
        task_id: str,
        repo: str | None = None,
    ) -> None:
        """Record that an assignment has begun execution.

        Creates the assignment row if missing and populates ``started_at`` with the
        current UTC timestamp (matching the schema default format). Preserves any
        existing ``started_at`` so retries do not overwrite the original start time.
        Sets status to ``'started'`` only if the row is still in its initial
        ``'created'`` state (so we never regress a later ``'completed'``/``'failed'``
        status during a retry).
        """

        def _write() -> None:
            with self._batched():
                run_id = self._ensure_run(experiment)
                generation_id = self._ensure_generation(run_id, generation)
                agent_ref = self._ensure_agent(run_id, agent_id)
                task_ref = self._ensure_task(run_id, task_id, repo=repo)
                assignment_id = self._ensure_assignment(
                    generation_id,
                    agent_ref,
                    task_ref,
                    status="started",
                )
                with self._locked():
                    self._conn.execute(
                        "UPDATE assignments SET started_at = COALESCE(started_at, datetime('now')) WHERE id = ?",
                        (assignment_id,),
                    )
                    self._commit()

        self._run_write(_write)
        self._trace_sqlite_event(
            "mark_assignment_started",
            {
                "experiment": experiment,
                "generation": generation,
                "agent_id": agent_id,
                "task_id": task_id,
            },
        )

    def mark_assignment_ended(
        self,
        *,
        experiment: str,
        generation: int,
        agent_id: str,
        task_id: str,
        status: str = "completed",
        repo: str | None = None,
    ) -> None:
        """Record that an assignment has finished.

        Ensures the assignment row exists, sets ``ended_at`` to the current UTC
        timestamp, and updates ``status`` to the supplied terminal value. If
        ``started_at`` is still NULL (for example when the orchestrator never
        emitted a 'started' event, or the row was created lazily by a persistence
        writer), back-fills it with the current timestamp so duration analytics
        have a lower bound rather than a NULL.
        """

        def _write() -> None:
            with self._batched():
                run_id = self._ensure_run(experiment)
                generation_id = self._ensure_generation(run_id, generation)
                agent_ref = self._ensure_agent(run_id, agent_id)
                task_ref = self._ensure_task(run_id, task_id, repo=repo)
                assignment_id = self._ensure_assignment(
                    generation_id,
                    agent_ref,
                    task_ref,
                    status=status,
                )
                with self._locked():
                    self._conn.execute(
                        "UPDATE assignments"
                        " SET status = ?,"
                        "     started_at = COALESCE(started_at, datetime('now')),"
                        "     ended_at = datetime('now')"
                        " WHERE id = ?",
                        (status, assignment_id),
                    )
                    self._commit()

        self._run_write(_write)
        self._trace_sqlite_event(
            "mark_assignment_ended",
            {
                "experiment": experiment,
                "generation": generation,
                "agent_id": agent_id,
                "task_id": task_id,
                "status": status,
            },
        )

    # ---------- transcript API ----------

    def insert_raw_transcript(
        self,
        *,
        experiment: str,
        agent_id: str,
        generation: int,
        task_id: str,
        content: str,
        tool_trace: str = "[]",
        model_output: str | None = None,
        eval_result_json: str = "{}",
        native_score: float | None = None,
        runtime_meta_json: str = "{}",
    ) -> None:
        def _write() -> None:
            with self._batched():
                run_id = self._ensure_run(experiment)
                generation_id = self._ensure_generation(run_id, generation)
                agent_ref = self._ensure_agent(run_id, agent_id)
                task_ref = self._ensure_task(run_id, task_id)
                assignment_id = self._ensure_assignment(generation_id, agent_ref, task_ref, status="completed")
                attempt_id = self._latest_attempt_id(assignment_id)
                if attempt_id is None:
                    attempt_id = self._insert_attempt(
                        assignment_id=assignment_id,
                        model_output=model_output,
                        eval_result=_json_loads(eval_result_json, {}),
                        native_score=native_score,
                        tool_trace_json=tool_trace or "[]",
                        runtime_meta=_json_loads(runtime_meta_json, {}),
                        error_text=None,
                    )
                self._insert_artifact(
                    attempt_id=attempt_id,
                    artifact_type="transcript",
                    content=content,
                    metadata={"tool_trace": _json_loads(tool_trace, [])},
                )
                # NOTE: We intentionally do NOT also write the transcript body to
                # memory_docs (scope='transcript'). The canonical transcript store is
                # attempt_artifacts(artifact_type='transcript') — all production readers
                # (_transcript_query, raw_transcripts view, knowledge_store) go through
                # that path, so a transcript row in memory_docs was pure dead weight.
                # Observed in a real ARC-AGI-2 run (479 tasks, 539 attempts): dropping
                # this duplicate saves ~24.7 MB in memory_docs.body.

        self._run_write(_write)
        self._trace_sqlite_event(
            "insert_raw_transcript",
            {
                "experiment": experiment,
                "generation": generation,
                "agent_id": agent_id,
                "task_id": task_id,
                "bytes": len(content.encode("utf-8")),
            },
        )

    def _transcript_query(
        self,
        *,
        task_id: str,
        generation: int | None,
        agent_id: str | None,
        experiment: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        clauses = ["t.task_id = ?", "aa.artifact_type = 'transcript'"]
        params: list[Any] = [task_id]

        if experiment is not None:
            clauses.append("r.experiment = ?")
            params.append(self._experiment(experiment))
        if generation is not None:
            clauses.append("g.generation = ?")
            params.append(int(generation))
        if agent_id is not None:
            clauses.append("ag.agent_id = ?")
            params.append(agent_id)

        sql = f"""
            SELECT
                aa.id,
                aa.content,
                aa.content_bytes AS bytes,
                aa.created_at,
                ag.agent_id,
                g.generation,
                r.experiment,
                at.tool_trace_json
            FROM attempt_artifacts aa
            JOIN attempts at ON at.id = aa.attempt_id
            JOIN assignments asg ON asg.id = at.assignment_id
            JOIN generations g ON g.id = asg.generation_id
            JOIN runs r ON r.id = g.run_id
            JOIN agents ag ON ag.id = asg.agent_ref
            JOIN tasks t ON t.id = asg.task_ref
            WHERE {" AND ".join(clauses)}
            ORDER BY aa.id DESC
            LIMIT ?
        """
        params.append(max(1, int(limit)))
        rows = self._execute(sql, tuple(params), fetchall=True)
        return rows or []

    def get_raw_transcript(
        self,
        *,
        task_id: str,
        generation: int | None = None,
        agent_id: str | None = None,
        experiment: str | None = None,
    ) -> dict | None:
        rows = self._transcript_query(
            task_id=task_id,
            generation=generation,
            agent_id=agent_id,
            experiment=experiment,
            limit=1,
        )
        if not rows:
            return None
        row = dict(rows[0])
        row["tool_trace"] = row.get("tool_trace_json") or "[]"
        return row

    # ---------- summary API ----------

    def insert_task_summary(
        self,
        *,
        id: str,
        experiment: str,
        agent_id: str,
        generation: int,
        task_id: str,
        repo: str | None,
        approach: str,
        key_files: list[str],
        outcome: str,
        score: float | None,
        lessons: list[str],
    ) -> None:
        def _write() -> None:
            with self._batched():
                run_id = self._ensure_run(experiment)
                generation_id = self._ensure_generation(run_id, generation)
                agent_ref = self._ensure_agent(run_id, agent_id)
                task_ref = self._ensure_task(run_id, task_id, repo=repo)
                assignment_id = self._ensure_assignment(generation_id, agent_ref, task_ref, status="completed")
                attempt_id = self._latest_attempt_id(assignment_id)
                if attempt_id is None:
                    attempt_id = self._insert_attempt(assignment_id=assignment_id)

                metadata = {
                    "summary_id": id,
                    "experiment": experiment,
                    "agent_id": agent_id,
                    "generation": generation,
                    "task_id": task_id,
                    "repo": repo or "",
                    "approach": approach,
                    "key_files": key_files,
                    "outcome": outcome,
                    "score": score,
                    "lessons": lessons,
                }
                body = "\n".join(
                    [
                        f"task_id: {task_id}",
                        f"repo: {repo or ''}",
                        f"outcome: {outcome}",
                        f"score: {score if score is not None else 'n/a'}",
                        "approach:",
                        approach,
                        "key_files:",
                        ", ".join(key_files),
                        "lessons:",
                        "\n".join(lessons),
                    ]
                ).strip()

                self._insert_artifact(
                    attempt_id=attempt_id,
                    artifact_type="task_summary",
                    content=body,
                    metadata=metadata,
                )

                self._upsert_memory_doc(
                    doc_id=id,
                    run_id=run_id,
                    generation_id=generation_id,
                    agent_ref=agent_ref,
                    task_ref=task_ref,
                    attempt_ref=attempt_id,
                    scope="task_summary",
                    title=f"Summary {task_id}",
                    body=body,
                    metadata=metadata,
                )

        self._run_write(_write)
        self._trace_sqlite_event(
            "insert_task_summary",
            {
                "summary_id": id,
                "experiment": experiment,
                "generation": generation,
                "agent_id": agent_id,
                "task_id": task_id,
                "repo": repo or "",
                "outcome": outcome,
                "score": score,
            },
        )

    def insert_task_trace(
        self,
        *,
        experiment: str,
        generation: int,
        agent_id: str,
        task_id: str,
        repo: str | None = None,
        model_output: str | None = None,
        eval_result: dict[str, Any] | None = None,
        native_score: float | None = None,
        tool_trace: list[dict[str, Any]] | None = None,
        runtime_meta: dict[str, Any] | None = None,
        error_text: str | None = None,
    ) -> None:
        """Persist a fully evaluated task trace into the primary runtime DB."""

        def _write() -> None:
            with self._batched():
                run_id = self._ensure_run(experiment)
                generation_id = self._ensure_generation(run_id, generation)
                agent_ref = self._ensure_agent(run_id, agent_id)
                task_ref = self._ensure_task(run_id, task_id, repo=repo)
                status = "failed" if error_text else "completed"
                assignment_id = self._ensure_assignment(generation_id, agent_ref, task_ref, status=status)
                attempt_id = self._insert_attempt(
                    assignment_id=assignment_id,
                    model_output=model_output,
                    eval_result=eval_result or {},
                    native_score=native_score,
                    tool_trace_json=_json_dumps(tool_trace or []),
                    runtime_meta=runtime_meta or {},
                    error_text=error_text,
                )
                body = model_output if isinstance(model_output, str) else ""
                self._insert_artifact(
                    attempt_id=attempt_id,
                    artifact_type="task_trace",
                    content=body,
                    metadata={
                        "experiment": experiment,
                        "generation": generation,
                        "agent_id": agent_id,
                        "task_id": task_id,
                        "native_score": native_score,
                        "error": error_text or "",
                        "tool_count": len(tool_trace or []),
                    },
                )

        self._run_write(_write)
        self._trace_sqlite_event(
            "insert_task_trace",
            {
                "experiment": experiment,
                "generation": generation,
                "agent_id": agent_id,
                "task_id": task_id,
                "native_score": native_score,
                "error": error_text or "",
            },
        )

    def _summary_rows(self, *, where_sql: str, params: tuple[Any, ...], limit: int = 0) -> list[dict[str, Any]]:
        limit_clause = f"LIMIT {int(limit)}" if limit > 0 else ""
        sql = f"""
            SELECT
                md.id,
                md.metadata_json,
                md.created_at,
                r.experiment,
                g.generation,
                ag.agent_id,
                t.task_id,
                t.repo
            FROM memory_docs md
            JOIN runs r ON r.id = md.run_id
            LEFT JOIN generations g ON g.id = md.generation_id
            LEFT JOIN agents ag ON ag.id = md.agent_ref
            LEFT JOIN tasks t ON t.id = md.task_ref
            WHERE md.scope = 'task_summary' AND {where_sql}
            ORDER BY COALESCE(g.generation, 0) DESC, md.created_at DESC
            {limit_clause}
        """
        rows = self._execute(sql, params, fetchall=True) or []
        out: list[dict[str, Any]] = []
        for row in rows:
            meta = _json_loads(row.get("metadata_json"), {})
            key_files = meta.get("key_files", [])
            lessons = meta.get("lessons", [])
            out.append(
                {
                    "id": row.get("id"),
                    "experiment": row.get("experiment", ""),
                    "agent_id": row.get("agent_id", ""),
                    "generation": row.get("generation"),
                    "task_id": row.get("task_id", ""),
                    "repo": row.get("repo", ""),
                    "approach": str(meta.get("approach", "")),
                    "key_files": _json_dumps(key_files if isinstance(key_files, list) else []),
                    "outcome": str(meta.get("outcome", "")),
                    "score": meta.get("score"),
                    "lessons": _json_dumps(lessons if isinstance(lessons, list) else []),
                    "created_at": row.get("created_at", ""),
                }
            )
        return out

    def list_task_summaries(self, experiment: str | None = None, *, limit: int = 0) -> list[dict]:
        if experiment:
            rows = self._summary_rows(where_sql="r.experiment = ?", params=(self._experiment(experiment),), limit=limit)
        else:
            rows = self._summary_rows(where_sql="1=1", params=(), limit=limit)
        if rows:
            return rows
        if not self._legacy_task_summaries_available():
            return []
        params: list[Any] = []
        where = "1=1"
        if experiment:
            where = "experiment = ?"
            params.append(self._experiment(experiment))
        return (
            self._execute(
                f"""
            SELECT id, experiment, agent_id, generation, task_id, repo, approach, key_files, outcome, score, lessons, created_at
            FROM task_summaries
            WHERE {where}
            ORDER BY generation DESC, created_at DESC
            """,
                tuple(params),
                fetchall=True,
            )
            or []
        )

    # ---------- forum API ----------

    def insert_forum_message(
        self,
        *,
        generation: int,
        agent_id: str,
        message_type: str,
        content: dict,
        round_num: int | None = None,
        experiment: str | None = None,
    ) -> None:
        if message_type not in VALID_FORUM_MESSAGE_TYPES:
            log.warning("insert_forum_message: unknown message_type %r, skipping", message_type)
            return

        def _write() -> None:
            with self._batched():
                run_id = self._ensure_run(experiment)
                generation_id = self._ensure_generation(run_id, generation)
                agent_ref = self._ensure_agent(run_id, agent_id) if agent_id else None
                with self._locked():
                    self._conn.execute(
                        "INSERT INTO forum_events (run_id, generation_id, round_num, agent_ref, message_type, content) VALUES (?, ?, ?, ?, ?, ?)",
                        (run_id, generation_id, round_num, agent_ref, message_type, _json_dumps(content or {})),
                    )
                    self._commit()

        self._run_write(_write)
        self._trace_sqlite_event(
            "insert_forum_message",
            {
                "experiment": self._experiment(experiment),
                "generation": generation,
                "agent_id": agent_id,
                "round_num": round_num,
                "message_type": message_type,
            },
        )

    def insert_token_phase(
        self,
        *,
        run_id: int,
        generation: int,
        phase: str,
        agent_ref: str | None = None,
        token_usage: Any = None,
        cost_usd: float = 0.0,
    ) -> None:
        """Insert a row into the token_phases table."""
        input_tokens = getattr(token_usage, "input_tokens", 0) if token_usage else 0
        output_tokens = getattr(token_usage, "output_tokens", 0) if token_usage else 0
        cache_read = getattr(token_usage, "cache_read_input_tokens", 0) if token_usage else 0
        # DB column ``cache_creation_tokens`` is the persisted alias for
        # TokenUsage.cache_creation_input_tokens.
        cache_creation = getattr(token_usage, "cache_creation_input_tokens", 0) if token_usage else 0

        def _write() -> None:
            with self._locked():
                self._conn.execute(
                    "INSERT INTO token_phases "
                    "(run_id, generation, phase, agent_ref, input_tokens, output_tokens, "
                    "cache_read_tokens, cache_creation_tokens, cost_usd) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        generation,
                        phase,
                        agent_ref,
                        input_tokens,
                        output_tokens,
                        cache_read,
                        cache_creation,
                        cost_usd,
                    ),
                )
                self._commit()

        self._run_write(_write)

    def get_token_phases(
        self,
        *,
        experiment: str | None = None,
        before_generation: int | None = None,
    ) -> list[dict]:
        """Return ``token_phases`` rows for the run, optionally bounded.

        Used on ``--resume`` to rehydrate the in-memory
        :class:`~kcsi.tokens.TokenAccumulator` so ``token_usage_total``
        reflects generations completed before the resume cursor.
        Passing *before_generation* restricts the result to strictly-earlier
        generations (``generation < before_generation``).  Returns ``[]`` when
        the experiment has no run.
        """
        run_id = self._find_run(experiment)
        if run_id is None:
            return []
        where = "run_id = ?"
        params: list[Any] = [run_id]
        if before_generation is not None:
            where += " AND generation < ?"
            params.append(int(before_generation))
        rows = self._execute(
            "SELECT generation, phase, agent_ref, input_tokens, output_tokens, "
            f"cache_read_tokens, cache_creation_tokens FROM token_phases WHERE {where}",
            tuple(params),
            fetchall=True,
        )
        return rows or []

    def list_forum_messages(
        self,
        generation: int,
        *,
        up_to: bool = False,
        experiment: str | None = None,
    ) -> list[dict]:
        run_id = self._find_run(experiment)
        comp = "<=" if up_to else "="
        if run_id is not None:
            sql = f"""
                SELECT
                    fe.id,
                    g.generation,
                    COALESCE(ag.agent_id, '') AS agent_id,
                    fe.message_type,
                    fe.content,
                    fe.round_num,
                    fe.created_at
                FROM forum_events fe
                JOIN generations g ON g.id = fe.generation_id
                LEFT JOIN agents ag ON ag.id = fe.agent_ref
                WHERE fe.run_id = ? AND g.generation {comp} ?
                ORDER BY fe.round_num DESC, fe.id DESC
            """
            return self._execute(sql, (run_id, int(generation)), fetchall=True) or []
        return []

    # ---------- hidden ARC references ----------

    def upsert_arc_task_reference(
        self,
        *,
        task_id: str,
        payload: dict[str, Any],
        experiment: str | None = None,
    ) -> None:
        def _write() -> None:
            run_id = self._ensure_run(experiment)
            now_expr = "datetime('now')"
            with self._locked():
                self._conn.execute(
                    f"""
                    INSERT INTO arc_task_refs (run_id, task_id, payload_json, created_at, updated_at)
                    VALUES (?, ?, ?, {now_expr}, {now_expr})
                    ON CONFLICT(run_id, task_id) DO UPDATE SET
                        payload_json = excluded.payload_json,
                        updated_at = {now_expr}
                    """,
                    (run_id, task_id, _json_dumps(payload or {})),
                )
                self._commit()

        self._run_write(_write)

    def get_arc_task_reference(
        self,
        *,
        task_id: str,
        experiment: str | None = None,
    ) -> dict[str, Any] | None:
        run_id = self._find_run(experiment)
        if run_id is None:
            return None
        row = self._execute(
            """
            SELECT payload_json
            FROM arc_task_refs
            WHERE run_id = ? AND task_id = ?
            LIMIT 1
            """,
            (run_id, task_id),
            fetchone=True,
        )
        if not row:
            return None
        return _json_loads(row.get("payload_json"), None)

    # ---------- task-centric memory record ----------

    def upsert_task_memory_record(
        self,
        *,
        experiment: str,
        generation: int,
        agent_id: str,
        task_id: str,
        eval_results: dict[str, Any] | None,
        final_model_output: str | None,
        full_memory_trace: str | None,
        full_memory_trace_condensed: str | None,
        task_specific_insights: list[str] | None,
        attempt_event: dict[str, Any] | None = None,
        injected_memory_md: str | None = None,
    ) -> None:
        meta_counts: dict[str, int] = {"insights_count": 0, "history_count": 0}

        def _write() -> None:
            # NOTE: This function assumes single-caller per (generation, agent_id, task_id).
            # The read-then-write of attempt_history is not atomic across lock acquisitions.
            # Safe today because the engine calls this sequentially per trace, never concurrently
            # for the same key. If parallelised in future, wrap read+write in a single lock.
            run_id = self._ensure_run(experiment)
            existing = self._execute(
                """
                SELECT attempt_history_json
                FROM task_memory_records
                WHERE run_id = ? AND generation = ? AND agent_id = ? AND task_id = ?
                LIMIT 1
                """,
                (run_id, int(generation), agent_id, task_id),
                fetchone=True,
            )
            history = _json_loads((existing or {}).get("attempt_history_json"), [])
            if not isinstance(history, list):
                history = []
            if isinstance(attempt_event, dict) and attempt_event:
                history.append(attempt_event)

            eval_json = _json_dumps(eval_results or {})
            insights = [str(x).strip() for x in (task_specific_insights or []) if isinstance(x, str) and str(x).strip()]
            meta_counts["insights_count"] = len(insights)
            meta_counts["history_count"] = len(history)
            now_expr = "datetime('now')"
            with self._locked():
                self._conn.execute(
                    f"""
                    INSERT INTO task_memory_records
                    (run_id, generation, agent_id, task_id,
                     eval_results_json, final_model_output, full_memory_trace,
                     full_memory_trace_condensed, task_specific_insights_json,
                     attempt_history_json, injected_memory_md, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {now_expr}, {now_expr})
                    ON CONFLICT(run_id, generation, agent_id, task_id) DO UPDATE SET
                        eval_results_json = CASE
                            WHEN excluded.eval_results_json IN ('{{}}', '[]', '')
                            THEN task_memory_records.eval_results_json
                            ELSE excluded.eval_results_json END,
                        final_model_output = CASE
                            WHEN excluded.final_model_output = ''
                            THEN task_memory_records.final_model_output
                            ELSE excluded.final_model_output END,
                        full_memory_trace = CASE
                            WHEN excluded.full_memory_trace = ''
                            THEN task_memory_records.full_memory_trace
                            ELSE excluded.full_memory_trace END,
                        full_memory_trace_condensed = CASE
                            WHEN excluded.full_memory_trace_condensed = ''
                            THEN task_memory_records.full_memory_trace_condensed
                            ELSE excluded.full_memory_trace_condensed END,
                        task_specific_insights_json = CASE
                            WHEN excluded.task_specific_insights_json IN ('[]', '')
                            THEN task_memory_records.task_specific_insights_json
                            ELSE excluded.task_specific_insights_json END,
                        attempt_history_json = CASE
                            WHEN excluded.attempt_history_json IN ('[]', '')
                            THEN task_memory_records.attempt_history_json
                            ELSE excluded.attempt_history_json END,
                        injected_memory_md = CASE
                            WHEN excluded.injected_memory_md = ''
                            THEN task_memory_records.injected_memory_md
                            ELSE excluded.injected_memory_md END,
                        updated_at = {now_expr}
                    """,
                    (
                        run_id,
                        int(generation),
                        agent_id,
                        task_id,
                        eval_json,
                        str(final_model_output or ""),
                        str(full_memory_trace or ""),
                        str(full_memory_trace_condensed or ""),
                        _json_dumps(insights),
                        _json_dumps(history),
                        str(injected_memory_md or ""),
                    ),
                )
                self._commit()

        self._run_write(_write)
        self._trace_sqlite_event(
            "upsert_task_memory_record",
            {
                "experiment": experiment,
                "generation": int(generation),
                "agent_id": agent_id,
                "task_id": task_id,
                "insights_count": meta_counts["insights_count"],
                "history_count": meta_counts["history_count"],
            },
        )

    def query_task_memory(
        self,
        *,
        task_id: str,
        experiment: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        if experiment is None:
            rows = (
                self._execute(
                    """
                SELECT generation, agent_id, task_id,
                       eval_results_json, final_model_output, full_memory_trace,
                       full_memory_trace_condensed, task_specific_insights_json,
                       attempt_history_json, created_at, updated_at
                FROM task_memory_records
                WHERE task_id = ?
                ORDER BY generation DESC, updated_at DESC
                LIMIT ?
                """,
                    (task_id, max(1, int(limit))),
                    fetchall=True,
                )
                or []
            )
        else:
            run_id = self._find_run(experiment)
            if run_id is None:
                return []
            rows = (
                self._execute(
                    """
                SELECT generation, agent_id, task_id,
                       eval_results_json, final_model_output, full_memory_trace,
                       full_memory_trace_condensed, task_specific_insights_json,
                       attempt_history_json, created_at, updated_at
                FROM task_memory_records
                WHERE run_id = ? AND task_id = ?
                ORDER BY generation DESC, updated_at DESC
                LIMIT ?
                """,
                    (run_id, task_id, max(1, int(limit))),
                    fetchall=True,
                )
                or []
            )
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "gen": int(row.get("generation") or 0),
                    "agent_id": str(row.get("agent_id") or ""),
                    "task_id": str(row.get("task_id") or ""),
                    "eval_results": _json_loads(row.get("eval_results_json"), {}),
                    "final_model_output": str(row.get("final_model_output") or ""),
                    "full_memory_trace": str(row.get("full_memory_trace") or ""),
                    "full_memory_trace_condensed": str(row.get("full_memory_trace_condensed") or ""),
                    "task_specific_insights": _json_loads(row.get("task_specific_insights_json"), []),
                    "attempt_history": _json_loads(row.get("attempt_history_json"), []),
                    "created_at": row.get("created_at", ""),
                    "updated_at": row.get("updated_at", ""),
                }
            )
        return out

    def get_best_scores(self, *, experiment: str | None = None) -> dict[str, float]:
        """Return {task_id: best_native_score} across all generations.

        Score extraction mirrors the generic fallback chain in
        ``_score_from_eval`` (engine.py) so that SWE-bench results
        (which store ``{"resolved": true}`` without a ``native_score``
        key) are correctly recognised on ``--resume``.
        """
        where = "1=1"
        params: tuple = ()
        if experiment:
            run_id = self._find_run(experiment)
            if run_id is None:
                return {}
            where = "run_id = ?"
            params = (run_id,)
        rows = (
            self._execute(
                f"""
            SELECT task_id, eval_results_json
            FROM task_memory_records
            WHERE {where} AND agent_id != '__forum__'
            """,
                params,
                fetchall=True,
            )
            or []
        )
        best: dict[str, float] = {}
        for row in rows:
            tid = str(row.get("task_id") or "")
            if not tid:
                continue
            eval_r = _json_loads(row.get("eval_results_json"), {})
            score = _extract_score(eval_r)
            if score is not None and score > best.get(tid, float("-inf")):
                best[tid] = score
        return best

    def get_latest_task_generation(self, *, experiment: str | None = None) -> int:
        """Return the latest generation with persisted task attempts.

        This intentionally reads ``task_memory_records`` rather than the
        generic ``generations`` table. The latter can contain forum, token, or
        seed-only rows from partial runs; task records are the canonical signal
        that a generation actually executed tasks and can be continued from.
        """
        run_id = self._find_run(experiment)
        if run_id is None:
            return 0
        row = self._execute(
            """
            SELECT MAX(generation) AS max_generation
            FROM task_memory_records
            WHERE run_id = ?
            """,
            (run_id,),
            fetchone=True,
        )
        value = (row or {}).get("max_generation")
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def checkpoint(self) -> None:
        """Run a WAL checkpoint to compact the write-ahead log."""
        _wal_checkpoint(
            read_only=self._read_only,
            conn=self._conn,
            locked=self._locked,
            logger=log,
            tag="STORE",
        )

    def close(self) -> None:
        if self._writer_queue is not None and self._writer_thread is not None:
            try:
                self._writer_queue.put(None)
                self._writer_thread.join(timeout=30.0)
                if self._writer_thread.is_alive():
                    log.warning(
                        "[STORE] Writer thread did not drain within 30s — ~%d items may be lost",
                        self._writer_queue.qsize(),
                    )
            except Exception:
                pass
            self._writer_queue = None
            self._writer_thread = None
            self._writer_thread_id = None
        if self._conn:
            try:
                self.checkpoint()
            except Exception:
                pass
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        lock_fd = getattr(self, "_lock_fd", None)
        if lock_fd is not None:
            try:
                lock_fd.close()
            except Exception:
                pass
            self._lock_fd = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

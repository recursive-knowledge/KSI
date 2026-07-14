"""Unified knowledge store for the kcsi memory system.

Replaces the split MemoryStore (task_memory, forum, task_docs) with a single
``knowledge`` table that captures attempt, insight, post, and distillation
entries. Seed snapshots live in the runtime DB.

Infrastructure patterns (WAL, writer thread, advisory flock, process-level
locks) are identical to ``store.py`` so the two can coexist during migration.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import sqlite3
import struct
import threading
import time
import weakref
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    from ._store_common import (
        WriteIndeterminateError,
        _apply_init_pragmas,
        _cleanup_stale_locks,
        _foreign_keys_enabled,
        _get_process_db_lock,
        _json_dumps,
        _json_loads,
        _locked_guard,
        _wal_checkpoint,
    )
except ImportError:  # pragma: no cover - script mode fallback inside container MCP
    # See ``store.py`` for the rationale: under a direct script-mode load the
    # relative import fails, so put this module's own directory on sys.path to
    # resolve the sibling ``_store_common``. No-op in the real container.
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _store_common import (  # type: ignore[no-redef]
        WriteIndeterminateError,
        _apply_init_pragmas,
        _cleanup_stale_locks,
        _foreign_keys_enabled,
        _get_process_db_lock,
        _json_dumps,
        _json_loads,
        _locked_guard,
        _wal_checkpoint,
    )

try:
    from .knowledge_store_migrations import migrate_from_legacy as _migrate_from_legacy
except ImportError:  # pragma: no cover - script mode fallback inside container MCP
    from knowledge_store_migrations import migrate_from_legacy as _migrate_from_legacy


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

# Per-store process-lock registry. AB-BA invariant: this dict MUST stay
# separate from MemoryStore's identical registry — the shared helper in
# ``_store_common`` only shares code, never the registry. See that module's
# docstring and tests/test_sqlite_persistence_deadlock.py.
_PROCESS_DB_LOCKS: dict[str, threading.RLock] = {}
_PROCESS_DB_LOCKS_GUARD = threading.Lock()


# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

VALID_ENTRY_TYPES = frozenset({"attempt", "insight", "post", "distillation"})
# Legacy phase names are retained alongside the new three-phase labels so that
# older DBs/tests keep working while new code migrates to the scoped labels.
VALID_SOURCE_PHASES = frozenset(
    {
        # Legacy (pre three-phase split)
        "execution",
        "discussion",
        "condensation",
        "seeding",
        # Three-phase knowledge generation (per-task vs cross-task)
        "per_task_forum",
        "cross_task_forum",
        "per_task_distill",
        "cross_task_distill",
    }
)

# Sentinel task_id used for cross-task (experiment-wide) distillation bundles.
# Per-task bundles are stored under the real task_id; cross-task bundles use
# this sentinel so there is exactly one bundle per (generation, scope).
CROSS_TASK_SENTINEL = "__cross_task__"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

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

CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    task_id     TEXT NOT NULL,
    repo        TEXT DEFAULT '',
    metadata    TEXT DEFAULT '{}',
    created_at  TEXT DEFAULT (datetime('now')),
    UNIQUE(run_id, task_id)
);

CREATE TABLE IF NOT EXISTS attempts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(id),
    generation      INTEGER NOT NULL,
    task_id         TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    entry_id        INTEGER,
    status          TEXT DEFAULT '',
    native_score    REAL,
    output_summary  TEXT DEFAULT '',
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS task_state (
    run_id              INTEGER NOT NULL REFERENCES runs(id),
    task_id             TEXT NOT NULL,
    best_score          REAL,
    solved              INTEGER NOT NULL DEFAULT 0,
    last_generation     INTEGER,
    last_attempt_id     INTEGER,
    updated_at          TEXT DEFAULT (datetime('now')),
    PRIMARY KEY(run_id, task_id)
);

CREATE TABLE IF NOT EXISTS knowledge (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(id),
    generation      INTEGER NOT NULL,
    task_id         TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    entry_type      TEXT NOT NULL,
    source_phase    TEXT NOT NULL,
    content         TEXT NOT NULL,
    parent_id       INTEGER,
    round_num       INTEGER,
    native_score    REAL,
    reply_to        INTEGER REFERENCES knowledge(id),
    external_id     TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS vector_status (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(id),
    generation      INTEGER,
    phase           TEXT NOT NULL,
    status          TEXT NOT NULL,
    detail          TEXT DEFAULT '',
    embedding_count INTEGER DEFAULT 0,
    skipped_count   INTEGER DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS seed_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(id),
    generation      INTEGER NOT NULL,
    agent_id        TEXT,
    payload_json    TEXT NOT NULL,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tasks_run_task ON tasks(run_id, task_id);
CREATE INDEX IF NOT EXISTS idx_attempts_run_task ON attempts(run_id, task_id, generation, id);
CREATE INDEX IF NOT EXISTS idx_attempts_run_generation ON attempts(run_id, generation, id);
CREATE INDEX IF NOT EXISTS idx_knowledge_task ON knowledge(run_id, task_id, generation, id);
CREATE INDEX IF NOT EXISTS idx_knowledge_discussion ON knowledge(run_id, task_id, generation, entry_type, id);
-- Serves the per-bucket knowledge_page reads, which join `runs` by experiment
-- (so no fixed run_id) and filter (task_id, entry_type). The run_id-leading
-- indexes above can't seek on entry_type there; this one pushes it to an
-- equality seek and orders by id, eliminating the ORDER BY sort.
CREATE INDEX IF NOT EXISTS idx_knowledge_task_type ON knowledge(task_id, entry_type);
-- Serves list_task_summaries' inner subquery, which filters on
-- (run_id, entry_type) and groups by task_id — the run_id-leading indexes
-- above don't lead with entry_type, so that lookup falls back to a scan.
CREATE INDEX IF NOT EXISTS idx_knowledge_run_entry_task ON knowledge(run_id, entry_type, task_id, id);
CREATE INDEX IF NOT EXISTS idx_knowledge_gen ON knowledge(run_id, generation, source_phase, id);
CREATE INDEX IF NOT EXISTS idx_knowledge_agent ON knowledge(run_id, agent_id, generation);
CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_external_id
    ON knowledge(run_id, external_id) WHERE external_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_vector_status_run ON vector_status(run_id, generation, phase, id);
CREATE INDEX IF NOT EXISTS idx_seed_snapshots_run_gen ON seed_snapshots(run_id, generation, id);
-- NOTE: idx_knowledge_reply_to is created in _run_migrations() because the
-- reply_to column may be missing on older DBs and a partial index referencing
-- a non-existent column would cause CREATE INDEX to fail on re-open.

CREATE TABLE IF NOT EXISTS discussion_done (
    run_id          INTEGER NOT NULL REFERENCES runs(id),
    generation      INTEGER NOT NULL,
    task_id         TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    done_at         TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (run_id, generation, task_id, agent_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    task_id, agent_id, entry_type, content,
    content='knowledge',
    content_rowid='id'
);

-- FTS5 content-sync triggers (keep knowledge_fts in sync with knowledge)
CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
    INSERT INTO knowledge_fts(rowid, task_id, agent_id, entry_type, content)
    VALUES (new.id, new.task_id, new.agent_id, new.entry_type, new.content);
END;

CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, task_id, agent_id, entry_type, content)
    VALUES ('delete', old.id, old.task_id, old.agent_id, old.entry_type, old.content);
END;

CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, task_id, agent_id, entry_type, content)
    VALUES ('delete', old.id, old.task_id, old.agent_id, old.entry_type, old.content);
    INSERT INTO knowledge_fts(rowid, task_id, agent_id, entry_type, content)
    VALUES (new.id, new.task_id, new.agent_id, new.entry_type, new.content);
END;
"""


# ---------------------------------------------------------------------------
# KnowledgeStore
# ---------------------------------------------------------------------------


class KnowledgeStore:
    """Unified SQLite store for task knowledge, discussion, and distillation.

    Follows the same infrastructure patterns as ``MemoryStore`` (WAL mode,
    writer thread, advisory flock, process-level locks) so the two can run
    side-by-side during migration.

    Concurrency contract (DIVERGES from ``MemoryStore`` — read before changing
    the locking in ``_batched()``): the two stores look near-textually
    identical but rely on *different* safety invariants for the same AB-BA
    deadlock class:

    * ``MemoryStore._batched()`` holds ``_locked()`` for the *whole* batch, so
      it is hardened for two writable ``MemoryStore`` instances sharing one DB
      (``engine._memory_store`` + ``SqlitePersistence._store``).
    * ``KnowledgeStore._batched()`` does NOT hold ``_locked()`` across the
      batch. Instead it relies on **single-writer-thread affinity**: every
      batched multi-statement write must run inside a single ``_run_write()``
      operation on this store's own writer thread (``_batched()`` raises if
      invoked off that thread). The production topology relies on one
      authoritative writable store (the engine's writable ``KnowledgeStore``;
      its init probe closes before the real store opens, the MCP server opens
      ``read_only=True``, and the forum bus is a JSONL file bus).

    A second writable ``KnowledgeStore`` on the same DB path is outside that
    production contract. Public methods are still protected by a
    defense-in-depth no-deadlock regression, but future production code must not
    add a second writer, cross-store batched writes, or direct ``_batched()``
    usage without hardening ``_batched()`` the way ``MemoryStore._batched()`` is
    hardened (or routing both through one store). See
    ``tests/memory/test_knowledge_store_two_writer_deadlock.py`` for the public
    two-writer guardrail and
    ``tests/memory/test_knowledge_store_batched_writer_affinity.py`` for the
    writer-thread invariant.
    """

    @staticmethod
    def _cleanup_stale_locks(directory: str | Path, max_age_seconds: int = 3600) -> None:
        """Remove ``.sqlite.lock`` files older than *max_age_seconds*.

        Delegates to the shared implementation in ``_store_common`` (identical
        between both stores).
        """
        _cleanup_stale_locks(directory, max_age_seconds)

    def __init__(
        self,
        db_path: str,
        *,
        read_only: bool = False,
        default_experiment: str = "default",
        enable_vec: bool = False,
        vec_dimensions: int = 768,
    ) -> None:
        self._db_path = db_path
        self._default_experiment = (default_experiment or "default").strip() or "default"
        self._read_only = bool(read_only)
        self._vec_enabled = False
        self._vec_dimensions = vec_dimensions

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        KnowledgeStore._cleanup_stale_locks(Path(db_path).parent)
        self._db_key = str(Path(db_path).resolve())

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
        # Batch-scoped run/generation resolution for run_drain_batch:
        # run_id and gen_id are constant across every op in a drain, so they are
        # resolved ONCE at the top of the drain (in the outer transaction) and
        # cached here; the per-op _ensure_run/gen_locked then short-circuit.
        # Set only for the duration of a drain and cleared in its finally, so a
        # stale entry can never leak to an unrelated write. Empty/None outside a
        # drain (never memoized across calls — see the rollback-safety note on
        # run_drain_batch).
        self._drain_ensured_run: tuple[str, int] | None = None
        self._drain_ensured_gens: dict[tuple[int, int], int] = {}
        self._lock_file_path = f"{db_path}.lock"
        self._lock_fd = None

        if not self._read_only:
            Path(self._lock_file_path).touch(exist_ok=True)
            self._lock_fd = open(self._lock_file_path, "a+b")  # noqa: SIM115

            with self._locked():
                _apply_init_pragmas(self._conn)
                self._conn.executescript(_SCHEMA)
                self._conn.commit()
            self._run_migrations()
            self._start_writer()

            if enable_vec:
                try:
                    self._init_vec(vec_dimensions)
                except Exception:
                    log.warning(
                        "[KNOWLEDGE_STORE] sqlite-vec not available — vector search disabled; "
                        "retrieval falls back to lexical FTS5"
                    )
        elif enable_vec:
            try:
                self._init_vec_read_only(vec_dimensions)
            except Exception:
                log.warning(
                    "[KNOWLEDGE_STORE] sqlite-vec not available in read-only mode — vector "
                    "search disabled; retrieval falls back to lexical FTS5"
                )

        if enable_vec and not self._vec_enabled:
            # Either the extension load failed above, or (read-only) the DB has
            # no vector index. Either way retrieval degrades to lexical FTS5;
            # surface it so the run is visibly degraded, not silently empty.
            log.info(
                "[KNOWLEDGE_STORE] vector index unavailable for %s — agent retrieval uses lexical FTS5 fallback",
                db_path,
            )

    # ------------------------------------------------------------------
    # Locking
    # ------------------------------------------------------------------

    def _locked(self):
        """Thread + process + advisory flock guard.

        Delegates to the shared ``_store_common._locked_guard``;
        ``self._process_lock`` comes from this module's own per-store
        ``_PROCESS_DB_LOCKS`` registry (the AB-BA invariant).
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
        """Suppress intermediate commits; commit once at the end.

        Invariant (DIFFERS from ``MemoryStore._batched()`` — see the class
        docstring): unlike ``MemoryStore``, this does NOT hold ``_locked()``
        for the whole batch. Its deadlock safety comes instead from
        **single-writer-thread affinity** — the guard below raises if
        ``_batched()`` runs off this store's writer thread. This protects the
        current production topology, where one writable ``KnowledgeStore`` owns
        the DB. A future production second writer or cross-store batched path
        must harden this method the way ``MemoryStore._batched()`` is hardened;
        the two-writer public-method test is only a defense-in-depth guardrail,
        not permission to add another production writer.

        KnowledgeStore writes are serialized through a writer thread. A caller
        thread cannot safely hold batch mode across multiple ``_run_write()``
        round-trips because the first uncommitted SQLite write can block a
        concurrent writer while that writer holds the shared process lock. Keep
        batches inside a single writer-thread operation.

        Transactional integrity: if the wrapped block raises, the OUTERMOST
        ``_batched()`` block rolls back the connection before propagating the
        exception. Without this, a multi-statement closure (e.g.
        ``record_post`` doing ensure_run + ensure_gen + ensure_agent + insert)
        that fails on the final insert would have committed the prerequisite
        rows on context exit — leaving a half-applied transaction. Inner
        ``_batched()`` blocks (``was_batch=True``) defer commit/rollback to
        the outer owner.
        """
        if not self._read_only and self._writer_queue is not None and threading.get_ident() != self._writer_thread_id:
            raise RuntimeError(
                "KnowledgeStore._batched() must run on the writer thread; "
                "wrap batched writes in a single _run_write() operation"
            )
        was_batch = self._batch_mode
        self._batch_mode = True
        try:
            yield
        except BaseException:
            self._batch_mode = was_batch
            if not was_batch:
                try:
                    self._conn.rollback()
                except Exception:
                    log.warning(
                        "[KnowledgeStore] rollback after _batched failure raised; "
                        "connection may be in inconsistent state",
                        exc_info=True,
                    )
            raise
        self._batch_mode = was_batch
        if not was_batch:
            self._conn.commit()

    def _commit(self) -> None:
        """Commit unless inside a ``_batched()`` block."""
        if not self._batch_mode:
            self._conn.commit()

    def _connection(self) -> sqlite3.Connection:
        """Expose the underlying sqlite3 connection for tests and migrations."""
        return self._conn

    def _run_migrations(self) -> None:
        """Apply additive schema migrations for existing databases.

        Called from ``__init__`` after ``_SCHEMA`` so that DBs created by older
        versions pick up new columns/indexes without requiring a rebuild.
        """
        if self._read_only:
            return
        with self._locked():
            cols = {row[1] for row in self._conn.execute("PRAGMA table_info(knowledge)").fetchall()}
            if "reply_to" not in cols:
                self._conn.execute("ALTER TABLE knowledge ADD COLUMN reply_to INTEGER")
            if "external_id" not in cols:
                self._conn.execute("ALTER TABLE knowledge ADD COLUMN external_id TEXT")
            run_cols = {row[1] for row in self._conn.execute("PRAGMA table_info(runs)").fetchall()}
            for col in ("code_commit", "resolved_model", "scoring_mode", "config_json"):
                if col not in run_cols:
                    self._conn.execute(f"ALTER TABLE runs ADD COLUMN {col} TEXT")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_status (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id          INTEGER NOT NULL REFERENCES runs(id),
                    generation      INTEGER,
                    phase           TEXT NOT NULL,
                    status          TEXT NOT NULL,
                    detail          TEXT DEFAULT '',
                    embedding_count INTEGER DEFAULT 0,
                    skipped_count   INTEGER DEFAULT 0,
                    created_at      TEXT DEFAULT (datetime('now'))
                )
                """
            )
            # Always ensure the partial index exists (cheap no-op if already present).
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_reply_to ON knowledge(reply_to) WHERE reply_to IS NOT NULL"
            )
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_external_id "
                "ON knowledge(run_id, external_id) WHERE external_id IS NOT NULL"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vector_status_run ON vector_status(run_id, generation, phase, id)"
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_task_type ON knowledge(task_id, entry_type)")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_run_entry_task ON knowledge(run_id, entry_type, task_id, id)"
            )
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id      INTEGER NOT NULL REFERENCES runs(id),
                    task_id     TEXT NOT NULL,
                    repo        TEXT DEFAULT '',
                    metadata    TEXT DEFAULT '{}',
                    created_at  TEXT DEFAULT (datetime('now')),
                    UNIQUE(run_id, task_id)
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id          INTEGER NOT NULL REFERENCES runs(id),
                    generation      INTEGER NOT NULL,
                    task_id         TEXT NOT NULL,
                    agent_id        TEXT NOT NULL,
                    entry_id        INTEGER,
                    status          TEXT DEFAULT '',
                    native_score    REAL,
                    output_summary  TEXT DEFAULT '',
                    created_at      TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS task_state (
                    run_id              INTEGER NOT NULL REFERENCES runs(id),
                    task_id             TEXT NOT NULL,
                    best_score          REAL,
                    solved              INTEGER NOT NULL DEFAULT 0,
                    last_generation     INTEGER,
                    last_attempt_id     INTEGER,
                    updated_at          TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY(run_id, task_id)
                );
                CREATE TABLE IF NOT EXISTS seed_snapshots (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id          INTEGER NOT NULL REFERENCES runs(id),
                    generation      INTEGER NOT NULL,
                    agent_id        TEXT,
                    payload_json    TEXT NOT NULL,
                    created_at      TEXT DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_run_task ON tasks(run_id, task_id);
                CREATE INDEX IF NOT EXISTS idx_attempts_run_task ON attempts(run_id, task_id, generation, id);
                CREATE INDEX IF NOT EXISTS idx_attempts_run_generation ON attempts(run_id, generation, id);
                CREATE INDEX IF NOT EXISTS idx_seed_snapshots_run_gen ON seed_snapshots(run_id, generation, id);
                """
            )
            knowledge_count = int(self._conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0])
            fts_count = int(self._conn.execute("SELECT COUNT(*) FROM knowledge_fts").fetchone()[0])
            rebuild_fts = knowledge_count != fts_count
            if not rebuild_fts:
                try:
                    self._conn.execute("INSERT INTO knowledge_fts(knowledge_fts, rank) VALUES ('integrity-check', 1)")
                except sqlite3.DatabaseError:
                    rebuild_fts = True
            if rebuild_fts:
                self._conn.execute("INSERT INTO knowledge_fts(knowledge_fts) VALUES ('rebuild')")
            self._conn.commit()

    # ------------------------------------------------------------------
    # Writer thread
    # ------------------------------------------------------------------

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
                # so a timed-out write can never apply later (ported
                # from store.py).
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
            name=f"KnowledgeStoreWriter[{db_name}]",
            daemon=True,
        )
        self._writer_thread.start()

    # Backoff schedule for transient SQLite lock errors. Total ~2s budget;
    # tuned to absorb millisecond-scale BUSY contention from the runtime DB
    # sidecar without blocking long enough to hide a real wedge. Six
    # attempts including the initial: 0, 50ms, 150ms, 400ms, 800ms, 1500ms.
    _BUSY_RETRY_DELAYS_SEC = (0.05, 0.15, 0.4, 0.8, 1.5)

    def _call_with_busy_retry(self, fn):
        """Invoke ``fn`` retrying on transient SQLite ``database is locked``.

        Used by ``_run_write`` to absorb millisecond-scale BUSY contention
        between this process and the optional runtime-DB sidecar. Non-retryable
        errors (schema, disk-full, programming errors) propagate immediately so
        they remain loud. Logs a warning on first retry so chronic contention
        is visible.
        """
        last_exc: BaseException | None = None
        for attempt, delay in enumerate((0.0,) + self._BUSY_RETRY_DELAYS_SEC):
            if delay:
                time.sleep(delay)
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                if not self._is_transient_sqlite_error(exc):
                    raise
                last_exc = exc
                if attempt == 0:
                    log.warning(
                        "[KnowledgeStore] transient SQLite lock on %s; retrying with backoff (%s)",
                        Path(self._db_path).name,
                        exc,
                    )
        # All retries exhausted — surface the last lock error so callers can
        # decide (engine.py:_eval_stage now logs+continues on these).
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _is_transient_sqlite_error(exc: BaseException) -> bool:
        """Return True for SQLite errors that are safe to retry.

        SQLite raises ``OperationalError("database is locked")`` /
        ``"database is busy"`` when a writer can't acquire the database lock
        within ``busy_timeout``. These are transient — the DB state is
        unchanged (the failed write never committed) and a fresh attempt
        after a short backoff usually succeeds.

        We deliberately do NOT retry generic ``OperationalError`` (e.g.
        schema mismatch, disk full, syntax error) because those won't go
        away with a backoff and retrying would mask a real bug.
        """
        if not isinstance(exc, sqlite3.OperationalError):
            return False
        msg = str(exc).lower()
        return "locked" in msg or "busy" in msg

    def _run_write(self, fn):
        if self._read_only:
            return self._call_with_busy_retry(fn)
        if threading.get_ident() == self._writer_thread_id:
            return self._call_with_busy_retry(fn)
        if self._writer_queue is None:
            return self._call_with_busy_retry(fn)
        done = threading.Event()
        box: dict[str, Any] = {}
        # Wrap fn() in the retry helper so transient SQLite lock errors
        # encountered on the writer thread don't propagate out as fatal
        # RuntimeErrors. The writer thread's `box["error"]` plumbing is
        # preserved — only success or a non-retryable exception lands in
        # the box. See `_is_transient_sqlite_error` for the retry policy.
        wrapped = lambda: self._call_with_busy_retry(fn)
        claim = threading.Lock()
        self._writer_queue.put((wrapped, done, box, claim))
        _queue_depth = self._writer_queue.qsize() if self._writer_queue else 0
        _timeout = max(120.0, 3.0 * _queue_depth)
        done.wait(timeout=_timeout)
        if not done.is_set():
            # The knowledge DB is authoritative, so both stall errors below
            # PROPAGATE to callers exactly as the old stall RuntimeError did —
            # no new swallowing. What changes: a raised stall
            # error now truthfully guarantees the write will never apply
            # (before, the closure stayed queued and silently applied on
            # writer recovery), and the rare uncancellable case is
            # distinguishable as WriteIndeterminateError.
            if claim.acquire(blocking=False):
                # Won the claim: the worker will skip this closure, so the
                # write is guaranteed not to apply.
                raise RuntimeError(
                    f"KnowledgeStore writer thread did not respond within {_timeout:.0f}s "
                    f"(queue depth was {_queue_depth}); write cancelled"
                )
            # The worker claimed the closure and is executing it right
            # now; it cannot be cancelled, so give it one more window.
            done.wait(timeout=_timeout)
            if not done.is_set():
                raise WriteIndeterminateError(
                    f"KnowledgeStore writer thread still executing a write after {2 * _timeout:.0f}s "
                    f"(queue depth was {_queue_depth}); the write may still be applied — do not retry"
                )
        if "error" in box:
            raise box["error"]
        return box.get("result")

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _execute(
        self,
        sql: str,
        params: tuple = (),
        *,
        fetchall: bool = False,
        fetchone: bool = False,
    ):
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

        if not self._read_only and sql_kind in {
            "INSERT",
            "UPDATE",
            "DELETE",
            "REPLACE",
            "CREATE",
            "DROP",
            "ALTER",
        }:
            return self._run_write(_op)
        return _op()

    def _experiment(self, experiment: str | None = None) -> str:
        exp = (experiment or self._default_experiment or "default").strip()
        return exp or "default"

    # ------------------------------------------------------------------
    # Run / generation / agent resolution
    # ------------------------------------------------------------------

    def _ensure_run_locked(
        self,
        experiment: str | None = None,
        *,
        code_commit: str | None = None,
        resolved_model: str | None = None,
        scoring_mode: str | None = None,
        config_json: str | None = None,
    ) -> int:
        """Get or create ``run_id`` using the caller's lock/transaction.

        Caller must already hold ``_locked()`` and own the writer-thread
        ``_run_write()`` closure).

        Run metadata (``code_commit``/``resolved_model``/``scoring_mode``/
        ``config_json``) is **write-once**: a field is set only while still
        NULL. A later call that supplies a DIFFERENT non-null value (e.g. a
        ``--resume`` under a changed HEAD, provider profile, or launch config)
        keeps the ORIGINAL stamp and logs a drift warning, rather than silently
        overwriting. ``config_json`` is the full effective launch config, making
        the DB self-describing without sacrificing provenance pinning.
        """
        exp = self._experiment(experiment)
        ctx = self._drain_ensured_run
        if ctx is not None and ctx[0] == exp:
            # Already resolved once at the top of the current run_drain_batch
            # (same experiment); the runs row is ensured in this transaction, so
            # skip the redundant INSERT OR IGNORE + SELECT.
            return ctx[1]
        self._conn.execute("INSERT OR IGNORE INTO runs (experiment) VALUES (?)", (exp,))
        row = self._conn.execute("SELECT id FROM runs WHERE experiment = ?", (exp,)).fetchone()
        if not row:
            raise RuntimeError(f"failed to resolve run for experiment={exp}")
        run_id = int(row["id"])
        if code_commit is not None or resolved_model is not None or scoring_mode is not None or config_json is not None:
            current = self._conn.execute(
                "SELECT code_commit, resolved_model, scoring_mode, config_json FROM runs WHERE id=?",
                (run_id,),
            ).fetchone()
            for field, incoming in (
                ("code_commit", code_commit),
                ("resolved_model", resolved_model),
                ("scoring_mode", scoring_mode),
                ("config_json", config_json),
            ):
                existing = current[field] if current is not None else None
                if incoming is not None and existing is not None and str(existing) != str(incoming):
                    log.warning(
                        "run %s: %s drift on resume — keeping original %r, ignoring incoming %r. "
                        "This run spans code/model versions; provenance stays pinned to the first "
                        "stamp. Split the run or re-stamp deliberately if the change was intended.",
                        exp,
                        field,
                        existing,
                        incoming,
                    )
            # Write-once: COALESCE(existing, incoming) keeps a non-null stamp and
            # only fills a still-NULL field — the opposite argument order from the
            # old clobbering form.
            self._conn.execute(
                "UPDATE runs SET code_commit=COALESCE(code_commit, ?), "
                "resolved_model=COALESCE(resolved_model, ?), "
                "scoring_mode=COALESCE(scoring_mode, ?), "
                "config_json=COALESCE(config_json, ?) WHERE id=?",
                (code_commit, resolved_model, scoring_mode, config_json, run_id),
            )
        return run_id

    def _ensure_generation_locked(self, run_id: int, generation: int) -> int:
        """Get or create ``generation_id`` using the caller's lock/transaction."""
        cached = self._drain_ensured_gens.get((int(run_id), int(generation)))
        if cached is not None:
            # Resolved once at the top of the current run_drain_batch; skip the
            # redundant INSERT OR IGNORE + SELECT (constant across ops).
            return cached
        self._conn.execute(
            "INSERT OR IGNORE INTO generations (run_id, generation) VALUES (?, ?)",
            (run_id, int(generation)),
        )
        row = self._conn.execute(
            "SELECT id FROM generations WHERE run_id = ? AND generation = ?",
            (run_id, int(generation)),
        ).fetchone()
        if not row:
            raise RuntimeError("failed to resolve generation id")
        return int(row["id"])

    def _ensure_agent_locked(self, run_id: int, agent_id: str) -> int:
        """Get or create agent ref using the caller's lock/transaction."""
        self._conn.execute(
            "INSERT OR IGNORE INTO agents (run_id, agent_id) VALUES (?, ?)",
            (run_id, agent_id),
        )
        row = self._conn.execute(
            "SELECT id FROM agents WHERE run_id = ? AND agent_id = ?",
            (run_id, agent_id),
        ).fetchone()
        if not row:
            raise RuntimeError("failed to resolve agent id")
        return int(row["id"])

    def _ensure_run(self, experiment: str | None = None) -> int:
        """Get or create ``run_id`` for *experiment*. Public, dispatches to writer thread."""
        exp = self._experiment(experiment)

        def _op():
            with self._locked():
                rid = self._ensure_run_locked(exp)
                self._commit()
                return rid

        return self._run_write(_op)

    def ensure_run(
        self,
        experiment: str | None = None,
        *,
        code_commit: str | None = None,
        resolved_model: str | None = None,
        scoring_mode: str | None = None,
        config_json: str | None = None,
    ) -> int:
        """Public: get or create ``run_id`` for *experiment*, dispatched to the writer thread.

        Stamps ``code_commit``/``resolved_model``/``scoring_mode``/``config_json``
        provenance metadata onto the ``runs`` row when provided. Values are
        applied via ``COALESCE`` so a repeat call omitting a kwarg never clobbers an
        already-stamped value back to NULL.
        """
        exp = self._experiment(experiment)

        def _op():
            with self._locked():
                rid = self._ensure_run_locked(
                    exp,
                    code_commit=code_commit,
                    resolved_model=resolved_model,
                    scoring_mode=scoring_mode,
                    config_json=config_json,
                )
                self._commit()
                return rid

        return self._run_write(_op)

    def _find_run(self, experiment: str | None = None) -> int | None:
        """Return the existing ``run_id`` for *experiment* without creating one."""
        row = self._execute(
            "SELECT id FROM runs WHERE experiment = ?",
            (self._experiment(experiment),),
            fetchone=True,
        )
        return int(row["id"]) if row else None

    def _ensure_generation(self, run_id: int, generation: int) -> int:
        """Get or create ``generation_id``. Public, dispatches to writer thread."""

        def _op():
            with self._locked():
                gid = self._ensure_generation_locked(run_id, generation)
                self._commit()
                return gid

        return self._run_write(_op)

    def _ensure_agent(self, run_id: int, agent_id: str) -> int:
        """Get or create agent ref. Public, dispatches to writer thread."""

        def _op():
            with self._locked():
                aid = self._ensure_agent_locked(run_id, agent_id)
                self._commit()
                return aid

        return self._run_write(_op)

    def ensure_refs(
        self,
        *,
        experiment: str | None = None,
        generation: int,
        agent_id: str,
    ) -> tuple[int, int, int]:
        """Resolve run/generation/agent refs in a single writer-thread round-trip.

        Returns ``(run_id, generation_id, agent_ref_id)``. Useful when a
        caller needs the IDs themselves (e.g. to insert into a sibling
        table) rather than just side-effecting the rows. ``record_attempt``
        / ``record_insight`` / ``record_post`` no longer route through this
        — they call the ``*_locked`` variants directly inside their own
        ``_run_write`` closure for end-to-end batch behavior.
        """

        def _op() -> tuple[int, int, int]:
            with self._locked(), self._batched():
                run_id = self._ensure_run_locked(experiment)
                gen_id = self._ensure_generation_locked(run_id, generation)
                agent_ref = self._ensure_agent_locked(run_id, agent_id)
                return run_id, gen_id, agent_ref

        return self._run_write(_op)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_entry_type(entry_type: str) -> None:
        if entry_type not in VALID_ENTRY_TYPES:
            raise ValueError(f"Invalid entry_type {entry_type!r}; must be one of {sorted(VALID_ENTRY_TYPES)}")

    @staticmethod
    def _validate_source_phase(source_phase: str) -> None:
        if source_phase not in VALID_SOURCE_PHASES:
            raise ValueError(f"Invalid source_phase {source_phase!r}; must be one of {sorted(VALID_SOURCE_PHASES)}")

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    # SQLite caps the number of bound parameters per statement at
    # SQLITE_MAX_VARIABLE_NUMBER (default 999 in older builds, 32766 in
    # 3.32+). Chunk to a conservative bound so this works on any sqlite
    # the host Python ships with. 500 leaves plenty of headroom for the
    # one extra `?` we use for the experiment column.
    _BULK_HAS_EXTERNAL_IDS_CHUNK = 500

    def bulk_has_external_ids(
        self,
        external_ids: list[str],
        *,
        experiment: str | None = None,
    ) -> set[str]:
        """Return the subset of *external_ids* already ingested.

        Chunked so that very large drains (e.g. multi-task forums with
        thousands of events) don't trip SQLite's bound-parameter cap.
        """
        cleaned = [s for s in (str(e).strip() for e in external_ids) if s]
        if not cleaned:
            return set()
        exp = self._experiment(experiment)
        chunk_size = self._BULK_HAS_EXTERNAL_IDS_CHUNK
        found: set[str] = set()
        for i in range(0, len(cleaned), chunk_size):
            batch = cleaned[i : i + chunk_size]
            placeholders = ",".join("?" for _ in batch)
            rows = (
                self._execute(
                    f"""
                SELECT k.external_id
                FROM knowledge k
                JOIN runs r ON r.id = k.run_id
                WHERE r.experiment = ?
                  AND k.external_id IN ({placeholders})
                """,
                    (exp, *batch),
                    fetchall=True,
                )
                or []
            )
            found.update(str(r["external_id"]) for r in rows)
        return found

    def _insert_knowledge_locked(
        self,
        *,
        run_id: int,
        generation: int,
        task_id: str,
        agent_id: str,
        entry_type: str,
        source_phase: str,
        content: str,
        parent_id: int | None = None,
        round_num: int | None = None,
        native_score: float | None = None,
        reply_to: int | None = None,
        embedding: list[float] | None = None,
        external_id: str | None = None,
    ) -> int:
        """Insert knowledge using the caller's lock/transaction."""
        self._validate_entry_type(entry_type)
        self._validate_source_phase(source_phase)
        external_id = str(external_id).strip() if external_id is not None else None
        if not external_id:
            external_id = None

        # reply_to integrity: only enforced when FK enforcement is
        # opt-in enabled. By default the forum drain tolerates dangling
        # ``reply_to`` (agent-supplied ids that need not resolve) — see the
        # note in ``_store_common._FOREIGN_KEYS_ENV`` — so validating
        # unconditionally would drop otherwise-valid posts on normal runs.
        if reply_to is not None and _foreign_keys_enabled():
            ref = self._conn.execute(
                "SELECT run_id FROM knowledge WHERE id = ?",
                (int(reply_to),),
            ).fetchone()
            if ref is None:
                raise ValueError(f"reply_to={reply_to} does not reference an existing knowledge row")
            if int(ref["run_id"]) != int(run_id):
                raise ValueError(
                    f"reply_to={reply_to} belongs to run_id={int(ref['run_id'])}, "
                    f"not the inserting run_id={int(run_id)}"
                )

        vec_blob: bytes | None = None
        if embedding is not None and self._vec_enabled:
            vec_blob = struct.pack(f"{len(embedding)}f", *embedding)

        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO knowledge
                (run_id, generation, task_id, agent_id, entry_type, source_phase,
                 content, parent_id, round_num, native_score, reply_to, external_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                int(generation),
                task_id,
                agent_id,
                entry_type,
                source_phase,
                content,
                parent_id,
                round_num,
                native_score,
                reply_to,
                external_id,
            ),
        )
        if cur.rowcount == 0 and external_id is not None:
            row = self._conn.execute(
                "SELECT id FROM knowledge WHERE run_id = ? AND external_id = ?",
                (run_id, external_id),
            ).fetchone()
            if not row:
                raise RuntimeError("failed to resolve existing knowledge row")
            return int(row["id"])

        rowid = int(cur.lastrowid)
        if vec_blob is not None:
            self._conn.execute(
                "INSERT INTO knowledge_vec(knowledge_rowid, embedding) VALUES (?, ?)",
                (rowid, vec_blob),
            )
        return rowid

    def _knowledge_id_for_external_id_locked(self, run_id: int, external_id: str | None) -> int | None:
        external_id = str(external_id).strip() if external_id is not None else None
        if not external_id:
            return None
        row = self._conn.execute(
            "SELECT id FROM knowledge WHERE run_id = ? AND external_id = ?",
            (run_id, external_id),
        ).fetchone()
        return int(row["id"]) if row else None

    def record_attempt(
        self,
        *,
        task_id: str,
        agent_id: str,
        generation: int,
        eval_results: dict | None = None,
        model_output: str = "",
        trace_condensed: str = "",
        insights: list[str] | None = None,
        native_score: float | None = None,
        experiment: str | None = None,
        embedding: list[float] | None = None,
        attempt_meta: dict | None = None,
        reflection: str = "",
        external_id: str | None = None,
        repo: str = "",
        supersede: bool = False,
    ) -> int:
        """Record a task execution attempt. Returns the knowledge entry id.

        When ``embedding`` is provided and vector search is enabled, the
        attempt row is also indexed in ``knowledge_vec`` for semantic
        search.  Passing ``None`` (the default) preserves the historical
        behaviour of writing only to the ``knowledge`` table.  Callers
        that want semantic coverage should pass a pre-computed embedding
        (e.g. via ``engine._maybe_embed(...)``) — the store never computes
        the embedding inline because embedder load is slow and must not
        block the write path.

        ``reflection`` is the agent's structured 3-5 sentence Phase-1
        self-reflection (load-bearing assumption + proposed change +
        predicted outcome). Stored at ``content.reflection`` so the per-
        task distillation prompt can pick it up directly (see
        ``src/kcsi/distillation/distiller.py``); empty string when the
        feature flag is off or the agent didn't produce one.

        ``repo`` (from ``TaskSpec.repo``/``TaskTrace.repo``) is persisted to
        the ``tasks`` table so ``list_task_summaries`` can surface it for
        cross-task repo-matched retrieval.

        ``external_id`` + ``supersede`` control what happens when a row with
        the same ``external_id`` already exists (within the same run):
        by default (``supersede=False``) the existing row wins and is
        returned unchanged — idempotent re-ingestion, used by the legacy
        KnowledgeStore migration. When ``supersede=True``, the existing
        row's content/native_score/embedding and its ``attempts`` row are
        UPDATED in place instead — used by the engine's late, richer
        attempt write to supersede the execution phase's early
        resume-safety placeholder (both calls share one external_id per
        execution attempt, see ``attempt_events._knowledge_attempt_external_id``).
        """
        content = _json_dumps(
            {
                "eval_results": eval_results or {},
                "model_output": model_output,
                "trace_condensed": trace_condensed,
                "insights": insights or [],
                "attempt_meta": attempt_meta or {},
                "reflection": reflection or "",
            }
        )

        # Single writer-thread closure. Previously this was 4 separate
        # _run_write dispatches (run / gen / agent / insert+state); each
        # paid its own writer-queue round-trip + commit. With N drained
        # forum events that's 4N round-trips serialized through one
        # writer thread — at high event counts the queue becomes the
        # bottleneck. _batched() suppresses intermediate commits so the
        # whole record lands in one transaction. _locked() is reentrant
        # within the writer thread so the inner helpers nest safely.
        def _op() -> int:
            with self._locked(), self._batched():
                try:
                    run_id = self._ensure_run_locked(experiment)
                    existing_id = self._knowledge_id_for_external_id_locked(run_id, external_id)
                    if existing_id is not None:
                        if not supersede:
                            return existing_id
                        self._update_attempt_content_locked(
                            run_id=run_id,
                            entry_id=existing_id,
                            task_id=task_id,
                            content=content,
                            native_score=native_score,
                            eval_results=eval_results or {},
                            output_summary=trace_condensed or model_output,
                            embedding=embedding,
                            repo=repo,
                        )
                        return existing_id
                    self._ensure_generation_locked(run_id, generation)
                    self._ensure_agent_locked(run_id, agent_id)
                    entry_id = self._insert_knowledge_locked(
                        run_id=run_id,
                        generation=generation,
                        task_id=task_id,
                        agent_id=agent_id,
                        entry_type="attempt",
                        source_phase="execution",
                        content=content,
                        native_score=native_score,
                        embedding=embedding,
                        external_id=external_id,
                    )
                    self._record_attempt_state_locked(
                        run_id=run_id,
                        generation=generation,
                        task_id=task_id,
                        agent_id=agent_id,
                        entry_id=entry_id,
                        native_score=native_score,
                        eval_results=eval_results or {},
                        output_summary=trace_condensed or model_output,
                        repo=repo,
                    )
                    return entry_id
                except Exception:
                    self._conn.rollback()
                    raise

        return int(self._run_write(_op))

    def _upsert_task_repo_locked(self, *, run_id: int, task_id: str, repo: str) -> None:
        """Insert the ``tasks`` row if absent; backfill ``repo`` once known.

        ``repo`` often isn't known yet at the very first write for a task
        (e.g. a source with no ``TaskSpec.repo``) and/or the early
        resume-safety attempt write and the later, richer write race each
        other — this upsert lets either write order populate the column
        without ever clobbering an already-known value with a blank one.
        """
        self._conn.execute(
            """
            INSERT INTO tasks(run_id, task_id, repo)
            VALUES (?, ?, ?)
            ON CONFLICT(run_id, task_id) DO UPDATE SET
                repo = excluded.repo
            WHERE excluded.repo != '' AND (tasks.repo IS NULL OR tasks.repo = '')
            """,
            (run_id, task_id, str(repo or "")),
        )

    def _update_attempt_content_locked(
        self,
        *,
        run_id: int,
        entry_id: int,
        task_id: str,
        content: str,
        native_score: float | None,
        eval_results: dict,
        output_summary: str,
        embedding: list[float] | None,
        repo: str,
    ) -> None:
        """Overwrite an existing attempt's content in place (supersede path).

        Used when the engine's late, richer write (real insight_text /
        reflection / lessons) arrives under the same ``external_id`` as the
        execution phase's early resume-safety placeholder. The embedding is
        refreshed via delete+reinsert (sqlite-vec's ``vec0`` tables don't
        support UPDATE); the FTS5 index stays in sync automatically via the
        existing ``knowledge_au`` AFTER UPDATE trigger. ``native_score``
        doesn't change between the early and late write for one execution
        attempt, so ``task_state`` (already correct from the early write)
        is intentionally left untouched here.
        """
        self._conn.execute(
            "UPDATE knowledge SET content = ?, native_score = ? WHERE id = ?",
            (content, native_score, entry_id),
        )
        if embedding is not None and self._vec_enabled:
            vec_blob = struct.pack(f"{len(embedding)}f", *embedding)
            self._conn.execute("DELETE FROM knowledge_vec WHERE knowledge_rowid = ?", (entry_id,))
            self._conn.execute(
                "INSERT INTO knowledge_vec(knowledge_rowid, embedding) VALUES (?, ?)",
                (entry_id, vec_blob),
            )
        status = str(
            eval_results.get("status") or eval_results.get("swebench_status") or eval_results.get("outcome") or ""
        )
        summary = str(output_summary or "")[:4000]
        self._conn.execute(
            """
            UPDATE attempts
            SET status = ?, native_score = ?, output_summary = ?
            WHERE run_id = ? AND entry_id = ?
            """,
            (status, native_score, summary, run_id, entry_id),
        )
        self._upsert_task_repo_locked(run_id=run_id, task_id=task_id, repo=repo)

    def _record_attempt_state_locked(
        self,
        *,
        run_id: int,
        generation: int,
        task_id: str,
        agent_id: str,
        entry_id: int,
        native_score: float | None,
        eval_results: dict,
        output_summary: str,
        repo: str = "",
    ) -> int:
        """Record attempt/task state using the caller's lock/transaction."""
        # ``swebench_status`` is the swebench_pro/polyglot evaluators' failure
        # vocabulary (``no_patch``/``capture_failed``/...); without it these
        # attempts persisted a blank status, making the diff-capture failure
        # mode invisible/unqueryable in the knowledge DB.
        status = str(
            eval_results.get("status") or eval_results.get("swebench_status") or eval_results.get("outcome") or ""
        )
        summary = str(output_summary or "")[:4000]
        self._upsert_task_repo_locked(run_id=run_id, task_id=task_id, repo=repo)
        cur = self._conn.execute(
            """
            INSERT INTO attempts
                (run_id, generation, task_id, agent_id, entry_id, status, native_score, output_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                int(generation),
                task_id,
                agent_id,
                int(entry_id),
                status,
                native_score,
                summary,
            ),
        )
        attempt_id = int(cur.lastrowid)
        self._conn.execute(
            """
            INSERT INTO task_state
                (run_id, task_id, best_score, solved, last_generation, last_attempt_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(run_id, task_id) DO UPDATE SET
                best_score = CASE
                    WHEN excluded.best_score IS NOT NULL
                     AND (task_state.best_score IS NULL OR excluded.best_score > task_state.best_score)
                    THEN excluded.best_score
                    ELSE task_state.best_score
                END,
                solved = CASE
                    WHEN excluded.solved = 1 THEN 1
                    ELSE task_state.solved
                END,
                last_generation = excluded.last_generation,
                last_attempt_id = excluded.last_attempt_id,
                updated_at = datetime('now')
            """,
            (
                run_id,
                task_id,
                native_score,
                1 if native_score is not None and float(native_score) >= 1.0 else 0,
                int(generation),
                attempt_id,
            ),
        )
        return attempt_id

    def record_vector_status(
        self,
        *,
        phase: str,
        status: str,
        detail: str = "",
        generation: int | None = None,
        embedding_count: int = 0,
        skipped_count: int = 0,
        experiment: str | None = None,
    ) -> int:
        """Record vector availability/coverage metadata for auditability."""

        def _op() -> int:
            with self._locked(), self._batched():
                run_id = self._ensure_run_locked(experiment)
                cur = self._conn.execute(
                    """
                    INSERT INTO vector_status
                        (run_id, generation, phase, status, detail, embedding_count, skipped_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        int(generation) if generation is not None else None,
                        phase,
                        status,
                        detail,
                        int(embedding_count),
                        int(skipped_count),
                    ),
                )
                return int(cur.lastrowid)

        return int(self._run_write(_op))

    def has_experiment(self, experiment: str | None = None) -> bool:
        """Return whether an experiment already exists in the authoritative DB."""
        row = self._execute(
            "SELECT 1 AS present FROM runs WHERE experiment = ? LIMIT 1",
            (self._experiment(experiment),),
            fetchone=True,
        )
        return bool(row)

    def next_experiment_name(self, base_experiment: str) -> str:
        """Return a non-conflicting experiment name using ``_<n>`` suffixes."""
        base = self._experiment(base_experiment)
        if not self.has_experiment(base):
            return base
        idx = 2
        while self.has_experiment(f"{base}_{idx}"):
            idx += 1
        return f"{base}_{idx}"

    def claim_experiment(self, base_name: str, *, resume: bool = False) -> str:
        """Atomically claim an experiment name, returning the name claimed.

        Unlike the read-only :meth:`has_experiment` +
        :meth:`next_experiment_name` probe (which two concurrent same-name
        launches can both pass, then collide), this inserts the ``runs`` row
        directly inside a single write transaction and relies on the
        ``runs.experiment`` UNIQUE constraint to detect a competing claim: on
        :class:`sqlite3.IntegrityError` it retries with the next ``_<n>``
        suffix, so concurrent launches deterministically claim distinct names.

        With ``resume=True`` the requested name is returned as-is (via
        ``INSERT OR IGNORE`` so an existing run row is preserved), matching the
        resume semantics of ``engine._resolve_experiment_name``.
        """
        base = self._experiment(base_name)

        def _op():
            with self._locked():
                if resume:
                    self._conn.execute("INSERT OR IGNORE INTO runs (experiment) VALUES (?)", (base,))
                    self._commit()
                    return base
                candidate = base
                idx = 2
                while True:
                    try:
                        self._conn.execute("INSERT INTO runs (experiment) VALUES (?)", (candidate,))
                        self._commit()
                        return candidate
                    except sqlite3.IntegrityError:
                        # Name already taken (an existing run or a concurrent
                        # claim). Back out the aborted statement and try the
                        # next suffix.
                        self._conn.rollback()
                        candidate = f"{base}_{idx}"
                        idx += 1

        return self._run_write(_op)

    def release_empty_experiment_claim(self, experiment: str | None = None) -> bool:
        """Delete an empty run row created by :meth:`claim_experiment`.

        This is used when orchestration initialization fails after a non-resume
        name claim but before real run state is written. It is deliberately
        conservative: any generation, agent, task, attempt, knowledge,
        best-score, or seed-snapshot row keeps the claim. Init-only
        vector-status rows are removed together with the otherwise empty run.
        """
        exp = self._experiment(experiment)

        def _op() -> bool:
            with self._locked():
                row = self._conn.execute("SELECT id FROM runs WHERE experiment = ?", (exp,)).fetchone()
                if row is None:
                    return False
                run_id = int(row["id"])
                content_tables = (
                    "generations",
                    "agents",
                    "tasks",
                    "attempts",
                    "knowledge",
                    "task_state",
                    "seed_snapshots",
                )
                for table in content_tables:
                    table_row = self._conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                        (table,),
                    ).fetchone()
                    if table_row is None:
                        continue
                    count_row = self._conn.execute(
                        f"SELECT COUNT(*) AS count FROM {table} WHERE run_id = ?",
                        (run_id,),
                    ).fetchone()
                    if int(count_row["count"] if count_row is not None else 0) > 0:
                        return False
                self._conn.execute("DELETE FROM vector_status WHERE run_id = ?", (run_id,))
                self._conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
                self._commit()
                return True

        return self._run_write(_op)

    def get_latest_task_generation(self, *, experiment: str | None = None) -> int:
        """Latest generation with recorded task attempts."""
        run_id = self._find_run(experiment)
        if run_id is None:
            return 0
        row = self._execute(
            "SELECT MAX(generation) AS max_generation FROM attempts WHERE run_id = ?",
            (run_id,),
            fetchone=True,
        )
        try:
            return int(row["max_generation"] or 0) if row else 0
        except (TypeError, ValueError):
            return 0

    def get_best_scores(self, *, experiment: str | None = None) -> dict[str, float]:
        """Return task_id -> best_score for resume/drop-solved logic."""
        run_id = self._find_run(experiment)
        if run_id is None:
            return {}
        rows = (
            self._execute(
                """
            SELECT task_id, best_score
            FROM task_state
            WHERE run_id = ? AND best_score IS NOT NULL
            """,
                (run_id,),
                fetchall=True,
            )
            or []
        )
        result: dict[str, float] = {}
        for row in rows:
            try:
                result[str(row["task_id"])] = float(row["best_score"])
            except (TypeError, ValueError):
                continue
        return result

    def solved_task_ids(
        self,
        task_ids: list[str],
        *,
        threshold: float = 1.0,
        experiment: str | None = None,
    ) -> set[str]:
        """Return the subset of ``task_ids`` with at least one solved attempt.

        A task counts as solved when ANY recorded attempt has
        ``eval_results.resolved is True`` OR ``native_score >= threshold``. Note this
        is deliberately NOT ``task_state.solved`` / ``get_best_scores``: those
        are derived from ``native_score`` alone and drop the ``resolved``
        branch, which matters for third-party registry evaluators (where
        resolved⟹score≥1.0 is unenforced) and for ``threshold > 1.0``.

        Runs a single bulk query per ~500 task ids instead of one
        ``query_task`` per task; unlike the historical per-task loop it
        examines ALL attempts for a task, not just the latest 100.
        """
        ids = [str(t) for t in task_ids]
        if not ids:
            return set()
        exp = self._experiment(experiment)
        thr = float(threshold)
        solved: set[str] = set()
        # Stay well under SQLite's bound-parameter limit.
        chunk_size = 500
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start : start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            rows = (
                self._execute(
                    f"""
                SELECT k.task_id AS task_id, k.native_score AS native_score,
                       CASE WHEN json_valid(k.content)
                            THEN json_extract(k.content, '$.eval_results') END AS eval_results
                FROM knowledge k
                JOIN runs r ON r.id = k.run_id
                WHERE r.experiment = ?
                  AND k.entry_type = 'attempt'
                  AND k.task_id IN ({placeholders})
                """,
                    tuple([exp, *chunk]),
                    fetchall=True,
                )
                or []
            )
            for row in rows:
                tid = str(row["task_id"])
                if tid in solved:
                    continue
                eval_results = _json_loads(row["eval_results"], {})
                resolved_flag = eval_results.get("resolved") is True if isinstance(eval_results, dict) else False
                score = row["native_score"]
                if resolved_flag or (score is not None and float(score) >= thr):
                    solved.add(tid)
        return solved

    def list_task_summaries(
        self,
        *,
        experiment: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return latest attempt summaries for cross-task seed enrichment."""
        run_id = self._find_run(experiment)
        if run_id is None:
            return []
        capped_limit = max(1, int(limit))
        rows = (
            self._execute(
                """
            SELECT k.task_id, k.generation, k.agent_id, k.native_score,
                   k.content, k.created_at, t.repo AS task_repo
            FROM knowledge k
            JOIN (
                SELECT task_id, MAX(id) AS latest_id
                FROM knowledge
                WHERE run_id = ? AND entry_type = 'attempt'
                GROUP BY task_id
            ) latest ON latest.latest_id = k.id
            LEFT JOIN tasks t ON t.run_id = k.run_id AND t.task_id = k.task_id
            WHERE k.run_id = ?
            ORDER BY k.id DESC
            LIMIT ?
            """,
                # Fetch one row past the cap so an exact truncation can be
                # detected (and logged) without a second COUNT query.
                (run_id, run_id, capped_limit + 1),
                fetchall=True,
            )
            or []
        )
        if len(rows) > capped_limit:
            log.warning(
                "[KNOWLEDGE] list_task_summaries truncated at limit=%d for experiment=%r; "
                "candidate task summaries beyond the limit were dropped",
                capped_limit,
                experiment,
            )
            rows = rows[:capped_limit]

        summaries: list[dict[str, Any]] = []
        for row in rows:
            content = _json_loads(row["content"], {})
            eval_results = content.get("eval_results") if isinstance(content, dict) else {}
            eval_results = eval_results if isinstance(eval_results, dict) else {}
            score = row.get("native_score")
            resolved = bool(eval_results.get("resolved"))
            if score is not None:
                try:
                    resolved = resolved or float(score) >= 1.0
                except (TypeError, ValueError):
                    pass
            summaries.append(
                {
                    "task_id": str(row.get("task_id") or ""),
                    "repo": str(row.get("task_repo") or ""),
                    "approach": str(content.get("trace_condensed") or ""),
                    "outcome": "resolved" if resolved else "unresolved",
                    "score": score,
                    "lessons": _json_dumps(content.get("insights") or []),
                    "generation": row.get("generation"),
                    "agent_id": row.get("agent_id"),
                    "updated_at": row.get("created_at"),
                }
            )
        return summaries

    def record_seed_snapshot(
        self,
        *,
        generation: int,
        agent_id: str | None = None,
        payload: dict | None = None,
        experiment: str | None = None,
    ) -> int:
        """Persist derived seed state in the authoritative knowledge DB."""

        def _op() -> int:
            with self._locked(), self._batched():
                run_id = self._ensure_run_locked(experiment)
                cur = self._conn.execute(
                    """
                    INSERT INTO seed_snapshots(run_id, generation, agent_id, payload_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        int(generation),
                        agent_id,
                        _json_dumps(payload or {}),
                    ),
                )
                return int(cur.lastrowid)

        return int(self._run_write(_op))

    def count_seed_snapshots(
        self,
        *,
        generation: int,
        experiment: str | None = None,
    ) -> int:
        run_id = self._find_run(experiment)
        if run_id is None:
            return 0
        row = self._execute(
            "SELECT COUNT(*) AS count FROM seed_snapshots WHERE run_id = ? AND generation = ?",
            (run_id, int(generation)),
            fetchone=True,
        )
        try:
            return int(row["count"] or 0) if row else 0
        except (TypeError, ValueError):
            return 0

    def record_insight(
        self,
        *,
        task_id: str,
        agent_id: str,
        generation: int,
        text: str,
        scope: str = "task",
        confidence: str = "medium",
        evidence_task_ids: list[str] | None = None,
        round_num: int = 0,
        experiment: str | None = None,
        embedding: list[float] | None = None,
        external_id: str | None = None,
    ) -> int:
        """Record an insight (R0 execution-time or discussion). Returns the entry id.

        ``text`` is stored verbatim — insights are the primary transfer signal
        and must never be clipped. The generation prompt is the length governor.
        """

        # Collapse the 4-op chain (run/gen/agent + insert) into one
        # writer-thread closure under _batched() — see record_attempt for
        # the rationale. Forum drain calls record_insight per event so
        # this directly multiplies the savings by event count.
        def _op() -> int:
            with self._locked(), self._batched():
                return self._record_insight_locked(
                    task_id=task_id,
                    agent_id=agent_id,
                    generation=generation,
                    text=text,
                    scope=scope,
                    confidence=confidence,
                    evidence_task_ids=evidence_task_ids,
                    round_num=round_num,
                    experiment=experiment,
                    embedding=embedding,
                    external_id=external_id,
                )

        return int(self._run_write(_op))

    def _record_insight_locked(
        self,
        *,
        task_id: str,
        agent_id: str,
        generation: int,
        text: str,
        scope: str = "task",
        confidence: str = "medium",
        evidence_task_ids: list[str] | None = None,
        round_num: int = 0,
        experiment: str | None = None,
        embedding: list[float] | None = None,
        external_id: str | None = None,
    ) -> int:
        """Insert an insight row using the caller's lock/transaction.

        Body of :py:meth:`record_insight` without the writer-thread dispatch
        so the forum drain can call it from inside a single batched
        ``_run_write`` (one transaction for many events; see
        :py:meth:`run_drain_batch`).
        """
        content = _json_dumps(
            {
                "text": text,
                "scope": scope,
                "confidence": confidence,
                "evidence_task_ids": evidence_task_ids or [],
            }
        )
        # Insights during discussion (round_num > 0) are "discussion" phase;
        # round_num == 0 are execution-time insights.
        source_phase = "discussion" if round_num > 0 else "execution"
        run_id = self._ensure_run_locked(experiment)
        self._ensure_generation_locked(run_id, generation)
        self._ensure_agent_locked(run_id, agent_id)
        return self._insert_knowledge_locked(
            run_id=run_id,
            generation=generation,
            task_id=task_id,
            agent_id=agent_id,
            entry_type="insight",
            source_phase=source_phase,
            content=content,
            round_num=round_num,
            embedding=embedding,
            external_id=external_id,
        )

    def record_post(
        self,
        *,
        task_id: str,
        agent_id: str,
        generation: int,
        text: str,
        parent_id: int | None = None,
        round_num: int = 0,
        experiment: str | None = None,
        reply_to: int | None = None,
        source_phase: str = "discussion",
        native_score: float | None = None,
        embedding: list[float] | None = None,
        external_id: str | None = None,
    ) -> int:
        """Record a discussion post on a task page. Returns the entry id.

        ``source_phase`` defaults to ``"discussion"`` for backward compat.
        The three-phase pipeline passes ``"per_task_forum"`` or
        ``"cross_task_forum"`` so the distiller can query each phase's
        posts separately via :py:meth:`query_generation`.

        ``native_score`` is the post author's own task score this generation
        (``None`` when unknown). It is surfaced back through ``query_task`` so
        the per-task distiller can weight high-score authors over low-score
        authors when their claims conflict.
        """

        # Collapse the 4-op chain (run/gen/agent + insert) into one
        # writer-thread closure under _batched() — same rationale as
        # record_attempt / record_insight. Forum drain calls record_post
        # per event so this is the biggest cumulative win.
        def _op() -> int:
            with self._locked(), self._batched():
                return self._record_post_locked(
                    task_id=task_id,
                    agent_id=agent_id,
                    generation=generation,
                    text=text,
                    parent_id=parent_id,
                    round_num=round_num,
                    experiment=experiment,
                    reply_to=reply_to,
                    source_phase=source_phase,
                    native_score=native_score,
                    embedding=embedding,
                    external_id=external_id,
                )

        return int(self._run_write(_op))

    def _record_post_locked(
        self,
        *,
        task_id: str,
        agent_id: str,
        generation: int,
        text: str,
        parent_id: int | None = None,
        round_num: int = 0,
        experiment: str | None = None,
        reply_to: int | None = None,
        source_phase: str = "discussion",
        native_score: float | None = None,
        embedding: list[float] | None = None,
        external_id: str | None = None,
    ) -> int:
        """Insert a post/comment row using the caller's lock/transaction.

        Body of :py:meth:`record_post` without the writer-thread dispatch so
        the forum drain can call it from inside a single batched
        ``_run_write`` (see :py:meth:`run_drain_batch`).
        """
        content = _json_dumps({"text": text})
        run_id = self._ensure_run_locked(experiment)
        self._ensure_generation_locked(run_id, generation)
        self._ensure_agent_locked(run_id, agent_id)
        return self._insert_knowledge_locked(
            run_id=run_id,
            generation=generation,
            task_id=task_id,
            agent_id=agent_id,
            entry_type="post",
            source_phase=source_phase,
            content=content,
            parent_id=parent_id,
            round_num=round_num,
            native_score=native_score,
            reply_to=reply_to,
            embedding=embedding,
            external_id=external_id,
        )

    def run_drain_batch(
        self,
        ops: list[Callable[[], Any]],
        *,
        experiment: str | None = None,
        generation: int | None = None,
    ) -> list[tuple[bool, Any]]:
        """Execute forum-drain write ops in ONE writer-thread transaction.

        Each op is a zero-arg callable that performs its inserts via the
        ``*_locked`` helpers (the batch already holds ``_locked()`` +
        ``_batched()`` — do NOT call the public ``record_*`` methods, which
        re-dispatch to the writer queue). Every op runs inside its own
        SQLite SAVEPOINT so a failing op rolls back to that savepoint and
        the remaining ops still commit — preserving the per-event
        partial-drain contract while collapsing N writer-queue
        round-trips and N COMMIT fsyncs into one.

        When ``experiment`` + ``generation`` are supplied (they are constant
        across every op in a drain), the run/generation refs are resolved ONCE
        here and cached so the per-op ``_ensure_run/gen_locked`` short-circuit
        instead of re-running ``INSERT OR IGNORE`` + ``SELECT`` per event.
        Rollback safety: the resolution runs in the OUTER transaction,
        BEFORE any per-op ``SAVEPOINT``, so a per-op ``ROLLBACK TO`` cannot
        orphan the run/generation rows; and the cache is per-call (cleared in
        ``finally``), never memoized across drains, so a rolled-back row can
        never leave a stale id behind. Omitting them preserves the exact prior
        behavior (each op resolves its own refs).

        Returns a list aligned with ``ops``: ``(True, result)`` on success
        or ``(False, exc)`` on failure. Never raises for an individual op's
        failure; only an infra error (lock acquisition, commit) propagates.
        """
        if not ops:
            return []

        def _batch() -> list[tuple[bool, Any]]:
            results: list[tuple[bool, Any]] = []
            with self._locked(), self._batched():
                # A SAVEPOINT issued outside an open transaction would itself
                # begin one, and RELEASE of that (outermost) savepoint would
                # COMMIT — one fsync per op, defeating the batching. Open an
                # explicit transaction first so savepoints nest inside it and
                # the single commit happens on _batched() exit.
                if not self._conn.in_transaction:
                    self._conn.execute("BEGIN")
                # Resolve run/gen ONCE for the whole drain (before any savepoint,
                # so per-op rollbacks can't orphan these rows) and cache them for
                # the per-op _ensure_*_locked fast path. Always cleared below.
                if generation is not None:
                    run_id = self._ensure_run_locked(experiment)
                    gen_id = self._ensure_generation_locked(run_id, generation)
                    self._drain_ensured_run = (self._experiment(experiment), run_id)
                    self._drain_ensured_gens = {(run_id, int(generation)): gen_id}
                try:
                    for i, op in enumerate(ops):
                        sp = f"drain_op_{i}"
                        self._conn.execute(f"SAVEPOINT {sp}")
                        try:
                            res = op()
                        except Exception as exc:  # noqa: BLE001 — per-op isolation
                            self._conn.execute(f"ROLLBACK TO {sp}")
                            self._conn.execute(f"RELEASE {sp}")
                            results.append((False, exc))
                        else:
                            self._conn.execute(f"RELEASE {sp}")
                            results.append((True, res))
                finally:
                    self._drain_ensured_run = None
                    self._drain_ensured_gens = {}
            return results

        return self._run_write(_batch)

    def record_distillation(
        self,
        *,
        task_id: str,
        generation: int,
        assets: list[dict] | None = None,
        bundle: dict | None = None,
        scope: str | None = None,
        experiment: str | None = None,
        embedding: list[float] | None = None,
        external_id: str | None = None,
    ) -> int:
        """Record a distillation entry.

        Two calling forms are supported:

        1. **Legacy asset-list form** (pre three-phase split)::

               record_distillation(task_id=..., generation=..., assets=[...])

           Stores ``{"assets": [...]}`` under ``source_phase='condensation'``.

        2. **Scoped bundle form** (three-phase split)::

               record_distillation(
                   task_id=...,
                   generation=...,
                   bundle={"transferable_insights": [...], "pitfalls": [...], ...},
                   scope="per_task" | "cross_task",
               )

           Stores the bundle dict (with ``scope`` merged in) under
           ``source_phase='per_task_distill'`` or ``'cross_task_distill'``.
           Legacy broadcast cross-task bundles use ``CROSS_TASK_SENTINEL`` as
           ``task_id``; target-conditioned cross-task bundles use the downstream
           target task id with ``scope="cross_task"``.

        Returns the knowledge row id.
        """
        if bundle is not None and assets is not None:
            raise ValueError(
                "record_distillation: pass either `assets=` (legacy) or `bundle=`+`scope=` (new), not both"
            )
        if bundle is None and scope is None and assets is None:
            raise ValueError("record_distillation: must pass either `assets=` or `bundle=`+`scope=`")
        if bundle is not None or scope is not None:
            if scope not in ("per_task", "cross_task"):
                raise ValueError(f"invalid scope: {scope!r}; expected 'per_task' or 'cross_task'")
            payload = dict(bundle or {})
            payload["scope"] = scope
            source_phase = "per_task_distill" if scope == "per_task" else "cross_task_distill"
            content = _json_dumps(payload)
            agent_id = "__distiller__"
        else:
            # Legacy `assets=` wire format. The engine writes the scoped
            # bundle form above; this branch is retained (not awaiting
            # removal) because the legacy-DB import path
            # (_migrate_forum_events / _migrate_docs) re-records old rows in
            # their original `condensation` format.
            source_phase = "condensation"
            content = _json_dumps({"assets": assets or []})
            agent_id = "orchestrator"

        # Single writer-thread closure: 3 _ensure_* / insert combined into
        # one round-trip. Distillation calls happen once per task per
        # generation, but the savings compound across the per-task forum
        # bundle write loop in distill_phase.
        def _op() -> int:
            with self._locked(), self._batched():
                run_id = self._ensure_run_locked(experiment)
                self._ensure_generation_locked(run_id, generation)
                self._ensure_agent_locked(run_id, agent_id)
                return self._insert_knowledge_locked(
                    run_id=run_id,
                    generation=generation,
                    task_id=task_id,
                    agent_id=agent_id,
                    entry_type="distillation",
                    source_phase=source_phase,
                    content=content,
                    embedding=embedding,
                    external_id=external_id,
                )

        return int(self._run_write(_op))

    def load_distillation(
        self,
        *,
        generation: int,
        task_id: str,
        scope: str,
        experiment: str | None = None,
    ) -> dict | None:
        """Load a scoped distillation bundle for a (generation, task_id, scope).

        Returns the stored bundle dict (including ``"scope"``) or ``None`` if
        no matching entry exists.  Scope must be ``"per_task"`` or
        ``"cross_task"``.
        """
        if scope not in ("per_task", "cross_task"):
            raise ValueError(f"invalid scope: {scope!r}; expected 'per_task' or 'cross_task'")
        source_phase = "per_task_distill" if scope == "per_task" else "cross_task_distill"
        exp = self._experiment(experiment)

        row = self._execute(
            """
            SELECT k.id AS knowledge_id, k.content
            FROM knowledge k
            JOIN runs r ON r.id = k.run_id
            WHERE r.experiment = ?
              AND k.generation = ?
              AND k.task_id = ?
              AND k.entry_type = 'distillation'
              AND k.source_phase = ?
            ORDER BY k.id DESC
            LIMIT 1
            """,
            (exp, int(generation), task_id, source_phase),
            fetchone=True,
        )
        if not row:
            return None
        try:
            data = json.loads(row["content"])
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        # Stamp the stable knowledge.id so seed packages (which copy this dict
        # verbatim into agent.seed_package) carry a backref for provenance joins
        # at attempt-record time.  Existing callers ignore the extra key.
        data["_knowledge_id"] = int(row["knowledge_id"])
        return data

    def load_distillations_batch(
        self,
        *,
        generation: int,
        task_ids: list[str],
        scope: str,
        experiment: str | None = None,
    ) -> dict[str, dict]:
        """Batched :meth:`load_distillation` for many task ids in one SQL pass.

        Returns ``{task_id: bundle}`` containing ONLY the task ids that have a
        stored bundle for ``(generation, scope)`` — a task with no bundle is
        absent from the map (callers treat absent as ``None``, matching the
        singular method's return). Each returned bundle carries the same
        ``_knowledge_id`` provenance backref as :meth:`load_distillation`, and
        the same "latest row wins" (``MAX(id)`` per task) semantics. Replaces
        the seeder's per-agent ``load_distillation`` loop, each of which took
        the process lock.
        """
        if scope not in ("per_task", "cross_task"):
            raise ValueError(f"invalid scope: {scope!r}; expected 'per_task' or 'cross_task'")
        source_phase = "per_task_distill" if scope == "per_task" else "cross_task_distill"

        ordered_ids: list[str] = []
        seen: set[str] = set()
        for tid in task_ids:
            tid = str(tid)
            if tid and tid not in seen:
                seen.add(tid)
                ordered_ids.append(tid)

        out: dict[str, dict] = {}
        if not ordered_ids:
            return out

        exp = self._experiment(experiment)
        # Stay well under SQLite's bound-parameter limit (cf. solved_task_ids).
        chunk_size = 500
        for start in range(0, len(ordered_ids), chunk_size):
            chunk = ordered_ids[start : start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            rows = (
                self._execute(
                    f"""
                SELECT id, task_id, content FROM (
                    SELECT k.id AS id, k.task_id AS task_id, k.content AS content,
                           ROW_NUMBER() OVER (
                               PARTITION BY k.task_id ORDER BY k.id DESC
                           ) AS _rn
                    FROM knowledge k
                    JOIN runs r ON r.id = k.run_id
                    WHERE r.experiment = ?
                      AND k.generation = ?
                      AND k.entry_type = 'distillation'
                      AND k.source_phase = ?
                      AND k.task_id IN ({placeholders})
                )
                WHERE _rn = 1
                """,
                    tuple([exp, int(generation), source_phase, *chunk]),
                    fetchall=True,
                )
                or []
            )
            for row in rows:
                try:
                    data = json.loads(row["content"])
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(data, dict):
                    continue
                data["_knowledge_id"] = int(row["id"])
                out[str(row["task_id"])] = data
        return out

    # ------------------------------------------------------------------
    # Read methods
    # ------------------------------------------------------------------

    def query_task(
        self,
        task_id: str,
        *,
        generation: int | None = None,
        entry_types: list[str] | None = None,
        limit: int = 50,
        experiment: str | None = None,
    ) -> dict:
        """Query the knowledge page for a task.

        Returns a structured dict grouped by entry type:
            attempts, discussion (posts), insights, distilled.

        The ``limit`` is applied *per bucket* (attempts, posts, insights,
        distillations) rather than as a single SELECT LIMIT, so a task
        with many attempts can't starve the discussion/distillation
        views shown to agents.
        """
        exp = self._experiment(experiment)

        base_clauses = ["r.experiment = ?", "k.task_id = ?"]
        base_params: list[Any] = [exp, task_id]

        if generation is not None:
            base_clauses.append("k.generation = ?")
            base_params.append(int(generation))

        types_to_fetch = (
            list(entry_types)
            if entry_types
            else [
                "attempt",
                "post",
                "insight",
                "distillation",
            ]
        )
        per_bucket_limit = max(1, int(limit))

        rows: list[dict[str, Any]] = []
        for et in types_to_fetch:
            clauses = base_clauses + ["k.entry_type = ?"]
            params = list(base_params) + [et, per_bucket_limit]
            where = " AND ".join(clauses)
            fetched = (
                self._execute(
                    f"""
                SELECT id, generation, task_id, agent_id, entry_type,
                       source_phase, content, parent_id, reply_to,
                       round_num, native_score, created_at
                FROM (
                    SELECT k.id AS id, k.generation AS generation,
                           k.task_id AS task_id, k.agent_id AS agent_id,
                           k.entry_type AS entry_type,
                           k.source_phase AS source_phase, k.content AS content,
                           k.parent_id AS parent_id, k.reply_to AS reply_to,
                           k.round_num AS round_num, k.native_score AS native_score,
                           k.created_at AS created_at
                    FROM knowledge k
                    JOIN runs r ON r.id = k.run_id
                    WHERE {where}
                    ORDER BY k.id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                    tuple(params),
                    fetchall=True,
                )
                or []
            )
            rows.extend(fetched)

        result = self._empty_task_page(task_id)
        for row in rows:
            self._append_task_page_row(result, row)
        return result

    @staticmethod
    def _empty_task_page(task_id: str) -> dict[str, Any]:
        """Empty structured page for a task (shared by query_task/query_tasks)."""
        return {
            "task_id": task_id,
            "attempts": [],
            "discussion": [],
            "insights": [],
            "distilled": [],
        }

    @staticmethod
    def _append_task_page_row(result: dict[str, Any], row: dict[str, Any]) -> None:
        """Append one knowledge row to a structured page.

        Shared by ``query_task`` (per-task loop) and ``query_tasks`` (batched
        ``WHERE task_id IN (...)``) so both produce byte-identical pages.
        """
        entry_type = row["entry_type"]
        content = _json_loads(row["content"], {})

        if entry_type == "attempt":
            result["attempts"].append(
                {
                    "gen": row["generation"],
                    "agent_id": row["agent_id"],
                    "score": row["native_score"],
                    "content": content,
                }
            )
        elif entry_type == "post":
            result["discussion"].append(
                {
                    "id": row["id"],
                    "agent_id": row["agent_id"],
                    "generation": row["generation"],
                    "text": content.get("text", ""),
                    "parent_id": row["parent_id"],
                    "reply_to": row["reply_to"],
                    "round_num": row["round_num"],
                    "native_score": row["native_score"],
                    "ts": row["created_at"],
                }
            )
        elif entry_type == "insight":
            result["insights"].append(
                {
                    "id": row["id"],
                    "gen": row["generation"],
                    "generation": row["generation"],
                    "agent_id": row["agent_id"],
                    "text": content.get("text", ""),
                    "scope": content.get("scope", "task"),
                    "ts": row["created_at"],
                }
            )
        elif entry_type == "distillation":
            # Two wire formats co-exist: legacy `{"assets": [...]}` and
            # the three-phase scoped bundle `{"transferable_insights":
            # [...], "pitfalls": [...], "scope": "per_task" | "cross_task"}`.
            assets = content.get("assets")
            if isinstance(assets, list) and assets:
                for asset in assets:
                    result["distilled"].append(
                        {
                            "asset_type": asset.get("asset_type", ""),
                            "text": asset.get("text", ""),
                            "gen": row["generation"],
                        }
                    )
            else:
                # Surface scoped bundles so agents reading the page
                # still see distilled knowledge.
                result["distilled"].append(
                    {
                        "asset_type": "bundle",
                        "scope": content.get("scope", ""),
                        "source_phase": row["source_phase"],
                        "bundle": content,
                        "gen": row["generation"],
                    }
                )

    def query_tasks(
        self,
        task_ids: list[str],
        *,
        generation: int | None = None,
        entry_types: list[str] | None = None,
        limit: int = 50,
        experiment: str | None = None,
    ) -> dict[str, dict]:
        """Batched ``query_task`` for many task ids in a single SQL pass.

        Returns ``{task_id: page}`` where each ``page`` is identical to the
        result of calling :meth:`query_task` with the same kwargs. Replaces the
        engine's N+1 loop (one sub-query per entry-type *per task*, each
        serialized through the store lock) with one ``WHERE task_id IN (...)``
        query per entry-type, using ``ROW_NUMBER() OVER (PARTITION BY ...)`` to
        reproduce ``query_task``'s per-bucket ``limit``.

        Every requested id is present in the returned mapping (empty page when
        the task has no rows), matching the per-call behavior.
        """
        # Preserve input order, drop blanks/dupes (a dup id would otherwise get
        # two IN-list slots and double-count its rows).
        ordered_ids: list[str] = []
        seen: set[str] = set()
        for tid in task_ids:
            tid = str(tid)
            if tid and tid not in seen:
                seen.add(tid)
                ordered_ids.append(tid)

        results: dict[str, dict] = {tid: self._empty_task_page(tid) for tid in ordered_ids}
        if not ordered_ids:
            return results

        exp = self._experiment(experiment)
        types_to_fetch = list(entry_types) if entry_types else ["attempt", "post", "insight", "distillation"]
        per_bucket_limit = max(1, int(limit))
        id_placeholders = ",".join("?" for _ in ordered_ids)

        gen_clause = ""
        gen_params: list[Any] = []
        if generation is not None:
            gen_clause = " AND k.generation = ?"
            gen_params = [int(generation)]

        for et in types_to_fetch:
            params: list[Any] = [exp, *ordered_ids, *gen_params, et, per_bucket_limit]
            fetched = (
                self._execute(
                    f"""
                SELECT id, generation, task_id, agent_id, entry_type,
                       source_phase, content, parent_id, reply_to,
                       round_num, native_score, created_at
                FROM (
                    SELECT k.id AS id, k.generation AS generation,
                           k.task_id AS task_id, k.agent_id AS agent_id,
                           k.entry_type AS entry_type,
                           k.source_phase AS source_phase, k.content AS content,
                           k.parent_id AS parent_id, k.reply_to AS reply_to,
                           k.round_num AS round_num, k.native_score AS native_score,
                           k.created_at AS created_at,
                           ROW_NUMBER() OVER (
                               PARTITION BY k.task_id ORDER BY k.id DESC
                           ) AS _rn
                    FROM knowledge k
                    JOIN runs r ON r.id = k.run_id
                    WHERE r.experiment = ? AND k.task_id IN ({id_placeholders})
                          {gen_clause} AND k.entry_type = ?
                )
                WHERE _rn <= ?
                ORDER BY task_id ASC, id ASC
                """,
                    tuple(params),
                    fetchall=True,
                )
                or []
            )
            for row in fetched:
                page = results.get(row["task_id"])
                if page is not None:
                    self._append_task_page_row(page, row)

        return results

    def query_generation(
        self,
        generation: int,
        *,
        source_phase: str | None = None,
        entry_types: list[str] | None = None,
        experiment: str | None = None,
    ) -> list[dict]:
        """Bulk read all entries for a generation. Used by distillation."""
        exp = self._experiment(experiment)

        clauses = ["r.experiment = ?", "k.generation = ?"]
        params: list[Any] = [exp, int(generation)]

        if source_phase:
            clauses.append("k.source_phase = ?")
            params.append(source_phase)

        if entry_types:
            placeholders = ",".join("?" for _ in entry_types)
            clauses.append(f"k.entry_type IN ({placeholders})")
            params.extend(entry_types)

        where = " AND ".join(clauses)

        rows = (
            self._execute(
                f"""
            SELECT k.id, k.generation, k.task_id, k.agent_id, k.entry_type,
                   k.source_phase, k.content, k.parent_id, k.reply_to,
                   k.round_num, k.native_score, k.created_at
            FROM knowledge k
            JOIN runs r ON r.id = k.run_id
            WHERE {where}
            ORDER BY k.id ASC
            """,
                tuple(params),
                fetchall=True,
            )
            or []
        )

        return [
            {
                "id": row["id"],
                "generation": row["generation"],
                "task_id": row["task_id"],
                "agent_id": row["agent_id"],
                "entry_type": row["entry_type"],
                "source_phase": row["source_phase"],
                "content": _json_loads(row["content"], {}),
                "parent_id": row["parent_id"],
                "round_num": row["round_num"],
                "native_score": row["native_score"],
                "created_at": row["created_at"],
                "reply_to": row["reply_to"],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Vector search (sqlite-vec)
    # ------------------------------------------------------------------

    def _init_vec(self, dimensions: int) -> None:
        """Load sqlite-vec extension and create the vector virtual table."""
        import sqlite_vec

        with self._locked():
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_vec USING vec0("
                f"knowledge_rowid INTEGER PRIMARY KEY, "
                f"embedding float[{dimensions}])"
            )
            self._conn.commit()
            self._vec_enabled = True
            self._vec_dimensions = dimensions

    def _init_vec_read_only(self, dimensions: int) -> None:
        """Load sqlite-vec for querying an existing vector table read-only."""
        import sqlite_vec

        with self._locked():
            exists = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='knowledge_vec'"
            ).fetchone()
            if not exists:
                return
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._vec_enabled = True
            self._vec_dimensions = dimensions

    def vec_search(
        self,
        embedding: list[float],
        *,
        max_results: int = 20,
        entry_types: list[str] | None = None,
        experiment: str | None = None,
    ) -> list[dict]:
        """Semantic nearest-neighbor search across knowledge entries."""
        if not self._vec_enabled:
            return []

        blob = struct.pack(f"{len(embedding)}f", *embedding)

        # vec0 knn queries require `k = ?` constraint (not LIMIT).
        # Filters are applied as post-filters since vec0 has limited
        # WHERE clause support.
        filtered = bool(entry_types or experiment)
        fetch_limit = max_results * 5 if filtered else max_results  # over-fetch then filter

        # The table alias must NOT be ``k`` — sqlite-vec's vec0 virtual table
        # exposes a hidden ``k`` column used for knn limit, and the predicate
        # ``AND k = ?`` needs to resolve to that column, not to the joined
        # ``knowledge`` alias. Using ``kn`` keeps the two namespaces distinct.
        #
        # ``runs`` is LEFT JOINed only to *return* the experiment column in one
        # query (avoiding a per-row SELECT); experiment stays a Python
        # post-filter below — a WHERE prefilter would change which ``k`` rows
        # the vec0 KNN returns and thus the over-fetch selection semantics.
        base_sql = """
            SELECT kn.id, kn.task_id, kn.agent_id, kn.entry_type, kn.source_phase,
                   kn.content, kn.native_score, kn.generation, kn.run_id,
                   r.experiment AS experiment, v.distance
            FROM knowledge_vec v
            JOIN knowledge kn ON kn.id = v.knowledge_rowid
            LEFT JOIN runs r ON r.id = kn.run_id
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance
        """
        exp_filter = self._experiment(experiment) if experiment else None

        def _query(k: int) -> list:
            return self._execute(base_sql, (blob, max(1, int(k))), fetchall=True) or []

        def _filter(rows: list) -> list[dict]:
            picked: list[dict] = []
            for row in rows:
                if entry_types and row["entry_type"] not in entry_types:
                    continue
                if exp_filter is not None and row["experiment"] != exp_filter:
                    continue
                picked.append(
                    {
                        "id": row["id"],
                        "task_id": row["task_id"],
                        "agent_id": row["agent_id"],
                        "entry_type": row["entry_type"],
                        "source_phase": row["source_phase"],
                        "content": _json_loads(row["content"], {}),
                        "native_score": row["native_score"],
                        "generation": row["generation"],
                        "distance": row["distance"],
                    }
                )
                if len(picked) >= max_results:
                    break
            return picked

        rows = _query(fetch_limit)
        out = _filter(rows)

        # Under-return guard: a fixed ``max_results * 5`` over-fetch can
        # still leave < max_results survivors when the filter excludes most of
        # the nearest neighbors. If the KNN returned a full page (``len(rows) >=
        # fetch_limit`` → more matches may lie further out) and we came up short,
        # escalate the fetch to the whole vec table once and re-filter, so every
        # valid row that exists is considered.
        if filtered and len(out) < max_results and len(rows) >= fetch_limit:
            total_row = self._execute("SELECT COUNT(*) AS n FROM knowledge_vec", fetchone=True)
            total = int(total_row["n"]) if total_row else 0
            if total > fetch_limit:
                out = _filter(_query(total))

        return out

    # ------------------------------------------------------------------
    # Full-text search (FTS5)
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize a query string for FTS5 MATCH.

        Removes special characters and FTS5 operators (NOT, OR, AND, NEAR)
        that could cause query parse errors.
        """
        # Remove special FTS5 characters
        cleaned = re.sub(r"[^\w\s]", " ", query)
        # Remove FTS5 operators (case-insensitive, whole word)
        cleaned = re.sub(r"\b(NOT|OR|AND|NEAR)\b", " ", cleaned, flags=re.IGNORECASE)
        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def fts_search(
        self,
        query: str,
        *,
        max_results: int = 20,
        entry_types: list[str] | None = None,
        experiment: str | None = None,
        task_id: str | None = None,
        raw_match: bool = False,
    ) -> list[dict]:
        """Full-text search across knowledge entries.

        ``task_id`` scopes results to a single task (a per-task search must not
        surface other tasks' posts; pass ``CROSS_TASK_SENTINEL`` to restrict to
        cross-task posts).

        ``raw_match`` skips ``_sanitize_fts_query``: the caller guarantees
        ``query`` is already a valid FTS5 MATCH expression built from sanitized
        tokens (e.g. ``"a OR b OR c"``). The default
        sanitizer strips FTS5 operators (``OR``/``AND``/``NOT``/``NEAR``), which
        would silently collapse an intended OR-query into an implicit-AND of
        every term — matching almost nothing. Use ``raw_match=True`` only with
        caller-sanitized tokens (no injection surface)."""
        sanitized = query.strip() if raw_match else self._sanitize_fts_query(query)
        if not sanitized:
            return []

        clauses = ["f.knowledge_fts MATCH ?"]
        params: list[Any] = [sanitized]

        if entry_types:
            placeholders = ",".join("?" for _ in entry_types)
            clauses.append(f"k.entry_type IN ({placeholders})")
            params.extend(entry_types)

        if experiment:
            clauses.append("r.experiment = ?")
            params.append(self._experiment(experiment))

        if task_id is not None:
            clauses.append("k.task_id = ?")
            params.append(task_id)

        where = " AND ".join(clauses)
        params.append(max(1, int(max_results)))

        # Join FTS results back to the knowledge table for full row data
        sql = f"""
            SELECT k.id, k.task_id, k.agent_id, k.entry_type, k.source_phase,
                   k.content, k.native_score, k.generation, k.created_at,
                   rank
            FROM knowledge_fts f
            JOIN knowledge k ON k.id = f.rowid
            JOIN runs r ON r.id = k.run_id
            WHERE {where}
            ORDER BY rank
            LIMIT ?
        """

        rows = self._execute(sql, tuple(params), fetchall=True) or []

        return [
            {
                "id": row["id"],
                "task_id": row["task_id"],
                "agent_id": row["agent_id"],
                "entry_type": row["entry_type"],
                "source_phase": row["source_phase"],
                "content": _json_loads(row["content"], {}),
                "native_score": row["native_score"],
                "generation": row["generation"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Discussion completion
    # ------------------------------------------------------------------

    def signal_done(
        self,
        *,
        task_id: str,
        agent_id: str,
        generation: int,
        experiment: str | None = None,
    ) -> None:
        """Signal that *agent_id* is done discussing *task_id*."""

        # Single writer-thread closure: ensure_run + insert in one
        # round-trip. signal_done is called per-agent at end of round, so
        # collapsing 2 ops → 1 saves a writer-queue dispatch per agent.
        def _op() -> None:
            with self._locked(), self._batched():
                self._signal_done_locked(
                    task_id=task_id,
                    agent_id=agent_id,
                    generation=generation,
                    experiment=experiment,
                )

        self._run_write(_op)

    def _signal_done_locked(
        self,
        *,
        task_id: str,
        agent_id: str,
        generation: int,
        experiment: str | None = None,
    ) -> None:
        """Insert a discussion_done row using the caller's lock/transaction.

        Body of :py:meth:`signal_done` without the writer-thread dispatch so
        the forum drain can fold ``done`` events into its single batched
        ``_run_write`` (see :py:meth:`run_drain_batch`).
        """
        run_id = self._ensure_run_locked(experiment)
        self._conn.execute(
            """
            INSERT OR IGNORE INTO discussion_done
                (run_id, generation, task_id, agent_id)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, int(generation), task_id, agent_id),
        )

    def get_done_status(
        self,
        *,
        task_id: str,
        generation: int,
        expected_agents: int,
        experiment: str | None = None,
    ) -> dict:
        """Check discussion completion status.

        Returns ``{"agents_done": int, "agents_expected": int, "all_done": bool}``.
        """
        exp = self._experiment(experiment)

        row = self._execute(
            """
            SELECT COUNT(*) AS cnt
            FROM discussion_done d
            JOIN runs r ON r.id = d.run_id
            WHERE r.experiment = ? AND d.generation = ? AND d.task_id = ?
            """,
            (exp, int(generation), task_id),
            fetchone=True,
        )

        agents_done = int(row["cnt"]) if row else 0
        return {
            "agents_done": agents_done,
            "agents_expected": expected_agents,
            "all_done": agents_done >= expected_agents,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def checkpoint(self) -> None:
        """Run a WAL checkpoint to compact the write-ahead log."""
        _wal_checkpoint(
            read_only=self._read_only,
            conn=self._conn,
            locked=self._locked,
            logger=log,
            tag="KNOWLEDGE_STORE",
        )

    def close(self) -> None:
        """Shut down writer thread and close connection."""
        if self._writer_queue is not None and self._writer_thread is not None:
            try:
                self._writer_queue.put(None)
                self._writer_thread.join(timeout=30.0)
                if self._writer_thread.is_alive():
                    log.warning(
                        "[KNOWLEDGE_STORE] Writer thread did not drain within 30s — ~%d items may be lost",
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

    @property
    def db_path(self) -> str:
        return self._db_path

    # ------------------------------------------------------------------
    # Legacy migration
    #
    # Implementation lives in ``knowledge_store_migrations`` (module-level
    # functions); this ``staticmethod`` shim keeps the public API
    # ``KnowledgeStore.migrate_from_legacy`` unchanged.
    # ------------------------------------------------------------------

    migrate_from_legacy = staticmethod(_migrate_from_legacy)

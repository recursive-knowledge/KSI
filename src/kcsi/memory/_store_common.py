"""Shared lock + JSON infrastructure for the two SQLite memory stores.

Extracted from ``store.py`` and ``knowledge_store.py``: the two
stores carried byte-identical copies of the process-level lock helper, the JSON
codecs, and the ``WriteIndeterminateError`` fallback. They now import them from
here.

CRITICAL — per-store lock-registry separation (the AB-BA deadlock
invariant): each store MUST keep its OWN module-level ``_PROCESS_DB_LOCKS``
registry. Only the helper *code* is shared here; the registry dict and its guard
are passed in by each store, so the same DB path opened by two different stores
yields two DISTINCT locks. Do NOT introduce a single shared registry in this
module — that would re-open the cross-store AB-BA deadlock window (see
tests/test_sqlite_persistence_deadlock.py).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# When the advisory flock cannot be acquired within the deadline, the store
# raises by default (surfacing the pathological cross-process hold loudly) —
# proceeding without the only cross-process lock silently degraded the
# "one writer at a time" invariant into "retry 30s then fail". Set
# ``KCSI_FLOCK_TIMEOUT_PROCEED=1`` to restore the legacy proceed-anyway
# behavior (degrade over abort) for campaigns that prefer it.
_FLOCK_TIMEOUT_PROCEED_ENV = "KCSI_FLOCK_TIMEOUT_PROCEED"

# SQLite FK enforcement is OFF by default. The schemas DECLARE foreign
# keys (e.g. knowledge.reply_to REFERENCES knowledge(id)), but the forum drain
# deliberately tolerates dangling references: agents supply arbitrary
# ``reply_to``/``parent_post_id`` integers that need not resolve to a real
# ``knowledge.id`` (see orchestrator/forum_runtime.py ``_coerce_post_ref`` and
# memory/mcp_server.py). Turning ``PRAGMA foreign_keys=ON`` unconditionally
# would make those tolerated orphans raise IntegrityError and drop posts on
# normal runs. So enforcement is opt-in via this flag — intended for auditing /
# integrity debugging, NOT production campaigns.
_FOREIGN_KEYS_ENV = "KCSI_SQLITE_FOREIGN_KEYS"


def _flock_timeout_should_proceed() -> bool:
    """Whether a flock-acquire timeout should proceed instead of raising."""
    return os.environ.get(_FLOCK_TIMEOUT_PROCEED_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _foreign_keys_enabled() -> bool:
    """Whether SQLite foreign-key enforcement is opt-in enabled.

    Defaults to False — see ``_FOREIGN_KEYS_ENV`` for why enforcement is
    opt-in rather than the SQLite-recommended default.
    """
    return os.environ.get(_FOREIGN_KEYS_ENV, "").strip().lower() in ("1", "true", "yes", "on")


try:
    import fcntl
except Exception:  # pragma: no cover - non-posix fallback
    fcntl = None  # type: ignore[assignment]

try:
    from ..errors import WriteIndeterminateError
except ImportError:  # pragma: no cover - script mode fallback inside container MCP

    class WriteIndeterminateError(RuntimeError):  # type: ignore[no-redef]
        """Script-mode fallback; the canonical class lives in src/kcsi/errors.py."""


def _get_process_db_lock(
    db_key: str,
    registry: dict[str, threading.RLock],
    guard: threading.Lock,
) -> threading.RLock:
    """Return the process-wide RLock for ``db_key`` from the caller's registry.

    ``registry`` and ``guard`` are owned per-store (see module docstring): the
    same DB path opened by two different stores yields two distinct locks, which
    is the AB-BA invariant.
    """
    with guard:
        lock = registry.get(db_key)
        if lock is None:
            lock = threading.RLock()
            registry[db_key] = lock
        return lock


def _cleanup_stale_locks(directory: str | Path, max_age_seconds: int = 3600) -> None:
    """Remove ``.sqlite.lock`` files older than *max_age_seconds* in *directory*.

    Called at the start of each store's ``__init__`` to avoid accumulating stale
    lock files from crashed processes.
    """
    try:
        cutoff = time.time() - max_age_seconds
        for lock_file in Path(directory).glob("*.sqlite.lock"):
            try:
                if lock_file.stat().st_mtime < cutoff:
                    lock_file.unlink(missing_ok=True)
            except OSError:
                pass
    except OSError:
        pass


def _apply_init_pragmas(conn: sqlite3.Connection) -> None:
    """Apply the shared WAL/concurrency PRAGMAs to a freshly opened connection.

    Both stores set WAL mode and busy timeout BEFORE schema creation so that
    concurrent processes opening the same DB during init already see WAL journal
    mode and don't fall back to delete-journal.
    """
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA wal_autocheckpoint=1000")
    # Foreign-key enforcement is per-connection in SQLite and OFF by default;
    # gate it behind an opt-in flag. See ``_FOREIGN_KEYS_ENV``.
    if _foreign_keys_enabled():
        conn.execute("PRAGMA foreign_keys=ON")


@contextmanager
def _locked_guard(
    *,
    lock_state: threading.local,
    process_lock: threading.RLock,
    thread_lock: threading.RLock,
    lock_fd: Any,
    logger: logging.Logger,
):
    """Thread + process + advisory flock guard for SQLite operations.

    This prevents concurrent host/container writers from stepping on each other
    on bind-mounted DB files. ``process_lock`` and ``thread_lock`` are owned by
    the calling store instance (see module docstring): the registry that backs
    ``process_lock`` is per-store, preserving the AB-BA invariant.

    Reentrancy is tracked on the caller-supplied ``lock_state`` (a
    ``threading.local``); nested ``with`` blocks bump a depth counter and do not
    reacquire the locks.
    """
    depth = int(getattr(lock_state, "depth", 0) or 0)
    if depth > 0:
        lock_state.depth = depth + 1
        try:
            yield
        finally:
            lock_state.depth = depth
        return

    # Lock ORDER (process -> thread -> flock) is the AB-BA deadlock fix
    # and MUST NOT change. The outer try/finally only guarantees process_lock is
    # released even if a KeyboardInterrupt (or any exception) lands between
    # acquiring process_lock and entering the inner body — otherwise process_lock
    # could leak and wedge the store forever.
    process_lock.acquire()
    try:
        thread_lock.acquire()
        lock_state.depth = 1
        flock_acquired = False
        try:
            if fcntl is not None and lock_fd is not None:
                try:
                    # Use non-blocking flock with retry loop and 60s timeout
                    # to avoid indefinite hangs if a container holds the lock.
                    _deadline = time.monotonic() + 60.0
                    while True:
                        try:
                            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            flock_acquired = True
                            break
                        except OSError:
                            if time.monotonic() >= _deadline:
                                if _flock_timeout_should_proceed():
                                    logger.warning(
                                        "Advisory flock timeout after 60s — proceeding without flock (%s opt-in)",
                                        _FLOCK_TIMEOUT_PROCEED_ENV,
                                    )
                                    break
                                # Raise loudly instead of silently proceeding
                                # without the only cross-process lock.
                                # No DB write has happened yet (before yield),
                                # and the enclosing finally blocks still release
                                # thread_lock/process_lock; flock was never
                                # acquired so its release is correctly skipped.
                                raise TimeoutError(
                                    "advisory flock not acquired within 60s; another writer holds "
                                    f"the DB lock. Set {_FLOCK_TIMEOUT_PROCEED_ENV}=1 to proceed "
                                    "without the cross-process lock (degrade over abort)."
                                )
                            time.sleep(0.05)
                except TimeoutError:
                    # Deliberate fail-loud path — must escape the broad guard
                    # below, which exists only to tolerate flock() quirks
                    # (e.g. unsupported fs) by degrading to no-flock.
                    raise
                except Exception:
                    pass
            yield
        finally:
            lock_state.depth = 0
            if flock_acquired and fcntl is not None and lock_fd is not None:
                try:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
            thread_lock.release()
    finally:
        process_lock.release()


def _wal_checkpoint(
    *,
    read_only: bool,
    conn: sqlite3.Connection | None,
    locked: Any,
    logger: logging.Logger,
    tag: str,
) -> None:
    """Run a WAL checkpoint (TRUNCATE) to compact the write-ahead log.

    ``locked`` is the calling store's ``_locked`` context-manager factory; ``tag``
    is the store-specific log prefix (e.g. ``STORE`` / ``KNOWLEDGE_STORE``).
    """
    if read_only or conn is None:
        return
    try:
        with locked():
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception as exc:
        logger.warning("[%s] WAL checkpoint failed: %s", tag, exc)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default

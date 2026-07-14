"""Regression test for the two-writer AB-BA deadlock (2026-04-21).

Symptom (before fix): when engine._memory_store and SqlitePersistence._store
both opened the same SQLite DB, a 10-concurrent dispatch stalled the writer
thread indefinitely — first INSERT on conn_A held an uncommitted write txn
while conn_B's first INSERT on another MemoryStore blocked on the SQLite
database-level lock. The two writer threads also shared the same
_process_lock RLock keyed on the DB path; between consecutive _ensure_*
helpers (each its own _locked() block) they ping-pong the process_lock,
creating the AB-BA.

Fix: _batched() now holds _locked() for the full batch so the implicit txn
is committed before the process_lock is released.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ksi.memory.store import MemoryStore


def test_two_stores_concurrent_writes_do_not_deadlock(tmp_path):
    """Two MemoryStore instances on the same DB, 10-thread concurrent writes.

    Before the _batched()/_locked() fix this deadlocked within ~5s of launch
    and never made progress past 3-ish committed rows. After the fix all 20
    writes land within a few seconds.
    """
    db_path = str(tmp_path / "shared.sqlite")
    store_a = MemoryStore(db_path, default_experiment="exp")
    store_b = MemoryStore(db_path, default_experiment="exp")

    def _worker_a(i: int) -> None:
        store_a.mark_assignment_started(
            experiment="exp",
            generation=1,
            agent_id=f"agent-a-{i}",
            task_id=f"task-{i}",
        )

    def _worker_b(i: int) -> None:
        store_b.mark_assignment_started(
            experiment="exp",
            generation=1,
            agent_id=f"agent-b-{i}",
            task_id=f"task-{i}",
        )

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = []
        for i in range(10):
            futures.append(pool.submit(_worker_a, i))
            futures.append(pool.submit(_worker_b, i))
        for fut in as_completed(futures, timeout=25):
            fut.result()
    elapsed = time.monotonic() - start
    # Generous ceiling; in practice each write is <100ms even with the
    # shared process_lock. Pre-fix this hit the 180s writer-thread timeout.
    assert elapsed < 20, f"two-store concurrent writes took {elapsed:.1f}s (deadlock?)"

    rows_a = store_a._execute("SELECT COUNT(*) AS n FROM assignments", fetchone=True)
    assert rows_a and rows_a["n"] == 20, f"expected 20 assignment rows, got {rows_a['n'] if rows_a else 'none'}"
    store_a.close()
    store_b.close()


def test_batched_holds_locked_through_commit(tmp_path):
    """Inside _batched(), _process_lock must stay held until commit.

    Spawns a watcher thread that repeatedly tries to acquire _process_lock
    while the main thread is inside _batched(). If the fix is intact, the
    watcher must never succeed until the _batched block exits.
    """
    db_path = str(tmp_path / "batched.sqlite")
    store = MemoryStore(db_path, default_experiment="exp")

    watcher_stole_lock = threading.Event()
    batched_entered = threading.Event()
    batched_exited = threading.Event()

    def _watch() -> None:
        # Wait until main has entered _batched() AND executed _ensure_run
        # (first inner _locked()) so we only test the window between
        # ensure_* calls, which is the AB-BA hotspot.
        batched_entered.wait(timeout=2.0)
        while not batched_exited.is_set():
            if store._process_lock.acquire(blocking=False):
                try:
                    watcher_stole_lock.set()
                    return
                finally:
                    store._process_lock.release()
            time.sleep(0.005)

    t = threading.Thread(target=_watch, daemon=True)
    t.start()

    with store._batched():
        store._ensure_run("exp")
        batched_entered.set()
        # Give the watcher ~40 attempts to steal the lock between
        # _ensure_run and _ensure_generation.
        time.sleep(0.2)
        store._ensure_generation(1, 1)
        time.sleep(0.2)
    batched_exited.set()
    t.join(timeout=2.0)

    assert not watcher_stole_lock.is_set(), (
        "another thread acquired _process_lock in the middle of _batched() — the AB-BA deadlock window is still open"
    )
    store.close()

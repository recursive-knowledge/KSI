"""Regression test for the *production* AB-BA deadlock (CLAUDE.md PR #368).

The documented production hang was NOT between two bare ``MemoryStore``
instances (that pairing is covered by
``tests/test_memory_store_two_writer_deadlock.py``). It was between two
*different types* sharing the same per-DB-path process lock:

    engine._memory_store        -> a ``MemoryStore``
    SqlitePersistence._store     -> a ``MemoryStore`` the persistence
                                    observer constructs internally on the
                                    SAME ``runtime_db_path``

``SqlitePersistence`` interleaves its own multi-step writes
(``on_task_status`` -> ``mark_assignment_started`` / ``mark_assignment_ended``;
``on_task_trace`` -> ``insert_task_trace``) against the engine's writes on the
shared DB file. Each of those write methods runs several ``_ensure_*`` helpers
inside one ``_batched()`` block.

How it deadlocks if the fix is reverted
---------------------------------------
``MemoryStore._process_lock`` is keyed on the DB path, so the bare
``MemoryStore`` and the ``SqlitePersistence`` store share the SAME RLock.
Before PR #368, ``_batched()`` did NOT hold ``_locked()`` for the whole batch:
each inner ``_ensure_*`` helper acquired and released ``_process_lock``
independently, but the implicit SQLite write transaction opened by the first
INSERT stayed open across those releases. So:

* Thread A (one store): first INSERT opens an uncommitted write txn on conn_A,
  then releases ``_process_lock`` between ``_ensure_run`` and
  ``_ensure_generation``.
* Thread B (the other store): grabs ``_process_lock``, issues its first INSERT
  on conn_B -> blocks on SQLite's database-level write lock held by A's
  uncommitted txn.
* Thread A: tries to reacquire ``_process_lock`` for ``_ensure_generation`` ->
  blocked behind B. Classic AB-BA; the writer threads stall until the 180s
  writer timeout.

The PR #368 fix makes ``_batched()`` hold ``_locked()`` for the entire batch,
so the implicit txn is committed before ``_process_lock`` is released. This
test drives >=10 concurrent writes through BOTH a ``MemoryStore`` and a real
``SqlitePersistence`` on the same DB path and asserts they all complete within
a bounded timeout. Revert the ``_batched()``/``_locked()`` discipline and this
test fails fast (within the 30s ``as_completed`` guard) instead of hanging CI
indefinitely.
"""

from __future__ import annotations

import threading
import time

from kcsi.cli import SqlitePersistence
from kcsi.memory.store import MemoryStore
from kcsi.models import TaskTrace
from kcsi.tokens import TokenUsage

N = 12  # >=10 concurrent writes per side


def _make_trace(i: int) -> TaskTrace:
    return TaskTrace(
        generation=1,
        agent_id=f"agent-p-{i}",
        task_id=f"task-{i}",
        model_output=None,
        eval_result={},
        native_score=None,
        tool_trace=[],
        runtime_meta={},
        token_usage=TokenUsage(),
        error=None,
        repo=f"org/repo-{i}",
    )


def test_memory_store_and_sqlite_persistence_concurrent_writes_do_not_deadlock(tmp_path):
    """One MemoryStore + one SqlitePersistence on the same DB, concurrent writes.

    This reproduces the *real* production pairing (engine store vs persistence
    store) rather than two bare MemoryStores. The two write paths share the
    per-DB-path ``_process_lock``; without the ``_batched()``-holds-``_locked()``
    fix the cross-type interleaving AB-BA deadlocks. With the fix every write
    lands within a few seconds.
    """
    db_path = str(tmp_path / "shared.sqlite")

    # engine-side store (bare MemoryStore)
    engine_store = MemoryStore(db_path, default_experiment="exp")
    # persistence-side observer — constructs its OWN MemoryStore internally on
    # the same db_path via _ensure_store(). Force it open now so both stores
    # (and their shared _process_lock) exist before the threads start.
    persist = SqlitePersistence(runtime_db_path=db_path, experiment_name="exp")
    persist._ensure_store()
    # Sanity: confirm the two stores are distinct objects on the same path,
    # i.e. the cross-type pairing the production deadlock required.
    assert persist._store is not engine_store
    assert persist._store._process_lock is engine_store._process_lock

    def _engine_writer(i: int) -> None:
        # Multi-_ensure_* batched write straight through the engine store.
        engine_store.mark_assignment_started(
            experiment="exp",
            generation=1,
            agent_id=f"agent-e-{i}",
            task_id=f"task-{i}",
        )

    def _persistence_writer(i: int) -> None:
        # Drive the REAL persistence callbacks: status (started -> completed)
        # plus a full task-trace insert. Each goes through _batched() with
        # several _ensure_* helpers — the AB-BA hotspot.
        persist.on_task_status(generation=1, agent_id=f"agent-p-{i}", task_id=f"task-{i}", status="started")
        persist.on_task_trace(_make_trace(i))
        persist.on_task_status(generation=1, agent_id=f"agent-p-{i}", task_id=f"task-{i}", status="completed")

    # Run the writers on DAEMON threads — this is what makes the guard fail
    # fast instead of hanging CI. Under a real deadlock the worker threads (and
    # the stores' own ``daemon=True`` writer threads) never finish, but daemon
    # threads are *abandoned* at interpreter exit rather than joined, so the
    # AssertionError below surfaces and the process still exits cleanly.
    # (A ``ThreadPoolExecutor`` would NOT work here: its worker threads are
    # non-daemon and ``concurrent.futures`` registers an ``atexit`` hook that
    # ``join()``s them on exit, re-introducing the unbounded hang this test is
    # meant to prevent.)
    errors: list[BaseException] = []

    def _runner(idx: int, fn) -> None:
        try:
            fn(idx)
        except BaseException as exc:  # noqa: BLE001 - re-raised after join
            errors.append(exc)

    threads = [
        threading.Thread(target=_runner, args=(i, fn), daemon=True, name=f"{fn.__name__}-{i}")
        for i in range(N)
        for fn in (_engine_writer, _persistence_writer)
    ]

    start = time.monotonic()
    for t in threads:
        t.start()

    # The join deadline is the real deadlock guard: a hung writer thread never
    # finishes, so we stop waiting after 30s instead of stalling for the 180s
    # per-write writer timeout.
    deadline = start + 30.0
    for t in threads:
        t.join(timeout=max(0.0, deadline - time.monotonic()))
    elapsed = time.monotonic() - start

    alive = [t for t in threads if t.is_alive()]
    if alive:
        # Deadlock detected. Deliberately do NOT fall through to ``close()`` —
        # ``close()`` calls ``checkpoint()`` which re-acquires the (unbounded)
        # ``_process_lock`` these threads are stuck on, which would re-hang the
        # main thread. The daemon threads are abandoned at exit, so leaking the
        # stores here is the lesser evil and keeps the failure fast and red.
        raise AssertionError(
            "MemoryStore + SqlitePersistence concurrent writes deadlocked: "
            f"{len(alive)}/{len(threads)} writer threads still running after "
            f"{elapsed:.1f}s — the _batched()/_locked() discipline likely regressed"
        )

    # No deadlock: every writer returned. Surface any real (non-deadlock)
    # writer error, verify the rows committed, then close cleanly (safe now
    # that the writer threads are idle).
    try:
        if errors:
            raise errors[0]
        # Generous ceiling; in practice each write is well under 100ms even
        # with the shared process_lock.
        assert elapsed < 25, f"MemoryStore + SqlitePersistence concurrent writes took {elapsed:.1f}s (deadlock?)"

        # Verify all writes actually committed (exercises real rows, not no-ops):
        # N engine assignments + N persistence assignments = 2N.
        rows = engine_store._execute("SELECT COUNT(*) AS n FROM assignments", fetchone=True)
        assert rows and rows["n"] == 2 * N, f"expected {2 * N} assignment rows, got {rows['n'] if rows else 'none'}"
        # And the persistence task traces landed too.
        trace_rows = engine_store._execute("SELECT COUNT(*) AS n FROM tasks", fetchone=True)
        assert trace_rows and trace_rows["n"] >= N, (
            f"expected >= {N} task rows from on_task_trace, got {trace_rows['n'] if trace_rows else 'none'}"
        )
    finally:
        persist.close()
        engine_store.close()

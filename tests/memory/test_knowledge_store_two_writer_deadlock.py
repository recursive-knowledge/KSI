"""Defense-in-depth: two writable KnowledgeStore instances on one DB (#738).

This is the KnowledgeStore analog of ``test_memory_store_two_writer_deadlock.py``
(the PR #368 MemoryStore AB-BA deadlock). No production config currently opens
two *writable* KnowledgeStore instances on the same DB — the engine holds the
only writable store (its init probe closes before the real store opens), the MCP
server opens ``read_only=True``, and the forum bus is a JSONL file bus. This test
exists so that if someone later wires a second writable KnowledgeStore into
persistence (the way SqlitePersistence did for MemoryStore), the deadlock class
is caught immediately rather than in a campaign stall.

Shape of the hazard (from the MemoryStore incident): conn_A holds an uncommitted
write txn while conn_B's first INSERT blocks on the SQLite database-level lock;
the two writer threads also share a ``_process_lock`` keyed on the DB path, and
ping-ponging it between batch helpers creates the AB-BA.

Constraint (knowledge_store.py): the batched writer path raises off the writer
thread, so this drives only the public ``record_*`` methods.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ksi.memory.knowledge_store import KnowledgeStore


def test_two_writable_stores_concurrent_writes_do_not_deadlock(tmp_path):
    """Two writable KnowledgeStore instances, ~10-worker concurrent writes.

    Interleaves ``record_attempt`` and ``record_post`` across both instances.
    A deadlock would stall the writer thread until its internal timeout; the
    ``as_completed(timeout=25)`` gate plus the wall-clock ceiling fail fast
    instead. A third read-only reader asserts zero-loss.
    """
    db_path = str(tmp_path / "shared_knowledge.sqlite")
    store_a = KnowledgeStore(db_path)
    store_b = KnowledgeStore(db_path)

    n = 10  # 4n = 40 writes total: n attempts + n posts per store

    def _attempt(store: KnowledgeStore, who: str, i: int) -> None:
        store.record_attempt(
            task_id=f"{who}-attempt-{i}",
            agent_id=f"agent-{who}-{i}",
            generation=1,
            eval_results={"resolved": False},
            model_output=f"output-{who}-{i}",
            native_score=float(i) / n,
        )

    def _post(store: KnowledgeStore, who: str, i: int) -> None:
        store.record_post(
            task_id="shared-posts",
            agent_id=f"agent-{who}-{i}",
            generation=1,
            text=f"post-{who}-{i}",
            external_id=f"{who}-{i}",
        )

    reader: KnowledgeStore | None = None
    try:
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = []
            for i in range(n):
                futures.append(pool.submit(_attempt, store_a, "a", i))
                futures.append(pool.submit(_attempt, store_b, "b", i))
                futures.append(pool.submit(_post, store_a, "a", i))
                futures.append(pool.submit(_post, store_b, "b", i))
            # as_completed(timeout) is the deadlock gate: a stalled writer never
            # resolves its future and this raises TimeoutError well before any
            # internal writer-thread timeout.
            for fut in as_completed(futures, timeout=25):
                fut.result()  # surface any writer-side exception
        elapsed = time.monotonic() - start
        assert elapsed < 20, f"two-store concurrent writes took {elapsed:.1f}s (deadlock?)"

        # Zero-loss row count via an independent read-only reader.
        reader = KnowledgeStore(db_path, read_only=True)
        attempts = 0
        for i in range(n):
            attempts += len(reader.query_task(f"a-attempt-{i}")["attempts"])
            attempts += len(reader.query_task(f"b-attempt-{i}")["attempts"])
        assert attempts == 2 * n, f"expected {2 * n} attempts, got {attempts}"

        posts = reader.query_task("shared-posts", limit=4 * n)["discussion"]
        assert len(posts) == 2 * n, f"expected {2 * n} posts, got {len(posts)}"
    finally:
        # Always close both writers (and the reader) so their daemon writer
        # threads don't outlive the tmp_path teardown when an assertion above
        # fails — store_a/store_b are opened before the timing gate.
        if reader is not None:
            reader.close()
        store_a.close()
        store_b.close()

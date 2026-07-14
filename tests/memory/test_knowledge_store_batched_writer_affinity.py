"""Guardrail: KnowledgeStore._batched() enforces single-writer-thread affinity.

``KnowledgeStore`` and ``MemoryStore`` have near-textually-identical
``_batched()`` methods but rely on DIFFERENT invariants to avoid the PR #368
AB-BA deadlock (two writers sharing one DB):

* ``MemoryStore._batched()`` holds ``_locked()`` for the whole batch.
* ``KnowledgeStore._batched()`` holds NO such lock — it relies on
  single-writer-thread affinity, raising if invoked off its writer thread.

That affinity check is the ONLY thing standing between the current (safe,
single-writable-store) contract and the exact AB-BA the MemoryStore fix
addresses. This test pins the check so the invariant is explicit and a
refactor that drops it (or that quietly makes the batched path callable off the
writer thread) fails here rather than in a campaign stall.

See ``tests/memory/test_knowledge_store_two_writer_deadlock.py`` for the
defense-in-depth concurrency test, and the ``KnowledgeStore`` class docstring
for the full divergence rationale.
"""

from __future__ import annotations

import threading

import pytest

from kcsi.memory.knowledge_store import KnowledgeStore


def test_batched_off_writer_thread_raises(tmp_path):
    """A writable store's ``_batched()`` must raise when entered off the writer thread.

    The test's own (main) thread is never the store's writer thread, so entering
    ``_batched()`` here should trip the affinity guard. The error message names
    the writer-thread requirement so the contract is greppable.
    """
    store = KnowledgeStore(str(tmp_path / "knowledge.sqlite"))
    try:
        # Sanity: this is a real writable store with a live writer thread that
        # is a DIFFERENT thread than the one running the test.
        assert store._writer_queue is not None
        assert store._writer_thread_id is not None
        assert threading.get_ident() != store._writer_thread_id

        with pytest.raises(RuntimeError, match="writer thread"):
            with store._batched():
                pass
    finally:
        store.close()


def test_batched_on_writer_thread_is_allowed(tmp_path):
    """``_batched()`` entered ON the writer thread does not raise.

    Public ``record_*`` methods reach ``_batched()`` via ``_run_write``, which
    dispatches the closure onto the writer thread; exercising the affinity path
    from that thread must succeed. This guards against an over-eager guard that
    would reject the legitimate batched-write path.
    """
    store = KnowledgeStore(str(tmp_path / "knowledge.sqlite"))
    result: dict[str, object] = {}

    def _op():
        # Runs on the writer thread inside _run_write; entering _batched() here
        # is the legitimate path and must not raise.
        with store._batched():
            result["thread_matched"] = threading.get_ident() == store._writer_thread_id
        return True

    try:
        assert store._run_write(_op) is True
        assert result["thread_matched"] is True
    finally:
        store.close()


def test_read_only_store_batched_does_not_require_writer_thread(tmp_path):
    """Read-only stores have no writer thread; ``_batched()`` must not raise.

    A ``read_only=True`` store is the only concurrently-opened KnowledgeStore in
    production. It has no ``_writer_queue``, so the affinity guard is inert and
    entering ``_batched()`` (e.g. via internal read helpers) is a no-op that
    commits nothing.
    """
    db_path = str(tmp_path / "knowledge.sqlite")
    writer = KnowledgeStore(db_path)
    writer.close()  # materialize the schema, then release the single writer

    reader = KnowledgeStore(db_path, read_only=True)
    try:
        assert reader._writer_queue is None
        with reader._batched():
            pass  # must not raise off any writer thread
    finally:
        reader.close()

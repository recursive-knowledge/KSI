# tests/memory/test_memory_store.py
"""Tests for src/ksi/memory/store.py — SQLite memory store."""

import gc
import sqlite3
import threading
import time
import weakref
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_store(tmp_path):
    from ksi.memory.store import MemoryStore

    db_path = str(tmp_path / "test_memory.sqlite")
    return MemoryStore(db_path)


class TestMemoryStoreSchema:
    def test_creates_tables(self, tmp_path):
        store = _make_store(tmp_path)
        tables = store._execute("SELECT name FROM sqlite_master WHERE type='table'", fetchall=True)
        names = {r["name"] for r in tables}
        assert "memory_docs" in names
        assert "attempt_artifacts" in names

    def test_ensure_compat_schema_creates_views(self, tmp_path):
        """_ensure_compat_schema_locked should create task_summaries and raw_transcripts views."""
        store = _make_store(tmp_path)
        row = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name='task_summaries'"
        ).fetchone()
        assert row is not None
        row = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name='raw_transcripts'"
        ).fetchone()
        assert row is not None
        store.close()

    def test_read_only_does_not_run_compat_schema_migrations(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "readonly.sqlite")
        from ksi.memory.store import MemoryStore

        writer = MemoryStore(db_path, default_experiment="ro")
        writer.close()

        import ksi.memory.store as store_mod

        def fail_if_called(_self: object) -> None:
            raise AssertionError("compat schema migration must not run in read-only mode")

        monkeypatch.setattr(store_mod.MemoryStore, "_ensure_compat_schema_locked", fail_if_called)
        reader = store_mod.MemoryStore(db_path, read_only=True)

        try:
            row = reader._execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'",
                fetchone=True,
            )
            assert row is not None
            assert row["name"] == "runs"
        finally:
            reader.close()

    def test_read_only_does_not_cleanup_stale_locks_or_create_parent(self, tmp_path):
        import os

        db_path = str(tmp_path / "readonly.sqlite")
        from ksi.memory.store import MemoryStore

        writer = MemoryStore(db_path, default_experiment="ro")
        writer.close()
        lock_path = Path(f"{db_path}.lock")
        lock_path.write_text("stale", encoding="utf-8")
        old_time = time.time() - 7200
        os.utime(lock_path, (old_time, old_time))

        reader = MemoryStore(db_path, read_only=True)
        try:
            assert lock_path.exists()
            assert lock_path.read_text(encoding="utf-8") == "stale"
        finally:
            reader.close()

        missing_db = tmp_path / "missing-parent" / "readonly.sqlite"
        with pytest.raises(sqlite3.OperationalError):
            MemoryStore(str(missing_db), read_only=True)
        assert not missing_db.parent.exists()

    def test_forum_error_message_type_persists(self, tmp_path):
        store = _make_store(tmp_path)
        store.insert_forum_message(
            generation=1,
            agent_id="agent-0",
            message_type="error",
            content={"phase": "cross_task_forum", "error": "provider unavailable"},
            round_num=0,
            experiment="default",
        )
        rows = store._execute(
            "SELECT message_type, content FROM forum_events ORDER BY id DESC LIMIT 1",
            fetchall=True,
        )
        assert rows
        assert rows[0]["message_type"] == "error"
        assert "provider unavailable" in str(rows[0]["content"])

    def test_locked_is_reentrant_without_repeated_advisory_flock(self, tmp_path, monkeypatch):
        """Nested store helpers should reuse the outer lock.

        Large ARC runs call compound write APIs whose helpers enter _locked()
        several times. Re-taking the advisory file lock for every nested helper
        can stall the single writer thread under train-50 scale.
        """
        import ksi.memory._store_common as store_common_mod
        from ksi.memory.store import MemoryStore

        db_path = str(tmp_path / "nested.sqlite")
        store = MemoryStore(db_path, default_experiment="test")
        calls: list[int] = []

        class FakeFcntl:
            LOCK_EX = 1
            LOCK_NB = 2
            LOCK_UN = 8

            @staticmethod
            def flock(_fd, flags):
                calls.append(flags)

        # The advisory flock now lives in the shared ``_store_common._locked_guard``
        # (the lock helper was deduped out of store.py/knowledge_store.py).
        monkeypatch.setattr(store_common_mod, "fcntl", FakeFcntl)
        try:
            with store._locked():
                with store._locked():
                    store._conn.execute("SELECT 1").fetchone()

            assert calls == [FakeFcntl.LOCK_EX | FakeFcntl.LOCK_NB, FakeFcntl.LOCK_UN]
            assert int(getattr(store._lock_state, "depth", 0) or 0) == 0
        finally:
            store.close()


class TestRawTranscripts:
    def test_insert_raw_transcript_creates_all_rows(self, tmp_path):
        """insert_raw_transcript should create run, gen, agent, task, attempt, and a
        single transcript artifact in attempt_artifacts (the canonical store).

        It must NOT also write a duplicate memory_docs row with scope='transcript':
        all production readers go through attempt_artifacts — a
        memory_docs[transcript] row is dead weight that bloats memory_docs.
        """
        from ksi.memory.store import MemoryStore

        db_path = str(tmp_path / "test.sqlite")
        store = MemoryStore(db_path, default_experiment="test")
        try:
            store.insert_raw_transcript(
                experiment="test",
                agent_id="agent-1",
                generation=1,
                task_id="task-1",
                content="fixed the bug",
                model_output="patch output",
                native_score=1.0,
            )
            run = store._execute("SELECT id FROM runs WHERE experiment = 'test'", fetchone=True)
            assert run is not None, "run row missing"
            gen = store._execute("SELECT id FROM generations WHERE run_id = ?", (run["id"],), fetchone=True)
            assert gen is not None, "generation row missing"
            agent = store._execute(
                "SELECT id FROM agents WHERE run_id = ? AND agent_id = 'agent-1'", (run["id"],), fetchone=True
            )
            assert agent is not None, "agent row missing"
            task = store._execute(
                "SELECT id FROM tasks WHERE run_id = ? AND task_id = 'task-1'", (run["id"],), fetchone=True
            )
            assert task is not None, "task row missing"
            attempt = store._execute("SELECT id FROM attempts", fetchone=True)
            assert attempt is not None, "attempt row missing"
            # Canonical transcript store: attempt_artifacts
            artifact = store._execute(
                "SELECT id, content FROM attempt_artifacts WHERE artifact_type = 'transcript'",
                fetchone=True,
            )
            assert artifact is not None, "attempt_artifacts[transcript] row missing"
            assert artifact["content"] == "fixed the bug"
            # Regression guard: no duplicate transcript in memory_docs.
            # (Prior behavior double-wrote the same body; see commit that introduced this test.)
            dup = store._execute(
                "SELECT COUNT(*) AS n FROM memory_docs WHERE scope = 'transcript'",
                fetchone=True,
            )
            assert dup["n"] == 0, (
                "memory_docs should NOT contain scope='transcript' rows — the transcript "
                "lives in attempt_artifacts. Double-writing bloats the DB."
            )
        finally:
            store.close()

    def test_insert_raw_transcript_attaches_to_existing_attempt(self, tmp_path):
        """A task trace and its transcript are one logical attempt."""
        from ksi.memory.store import MemoryStore

        db_path = str(tmp_path / "test.sqlite")
        store = MemoryStore(db_path, default_experiment="test")
        try:
            store.insert_task_trace(
                experiment="test",
                generation=1,
                agent_id="agent-1",
                task_id="task-1",
                model_output="patch output",
                eval_result={"resolved": False},
                native_score=0.0,
                tool_trace=[{"tool_name": "arc_submit_trial"}],
                runtime_meta={"status": "success"},
            )
            attempt = store._execute("SELECT id FROM attempts", fetchone=True)
            assert attempt is not None

            store.insert_raw_transcript(
                experiment="test",
                agent_id="agent-1",
                generation=1,
                task_id="task-1",
                content="transcript content",
                model_output="patch output",
                native_score=0.0,
            )

            attempt_count = store._execute("SELECT COUNT(*) AS n FROM attempts", fetchone=True)
            artifact = store._execute(
                "SELECT attempt_id, content FROM attempt_artifacts WHERE artifact_type = 'transcript'",
                fetchone=True,
            )
            assert attempt_count["n"] == 1
            assert artifact["attempt_id"] == attempt["id"]
            assert artifact["content"] == "transcript content"
        finally:
            store.close()

    def test_batch_mode_suppresses_intermediate_commits(self, tmp_path):
        """Verify _batched() keeps _batch_mode=True during helpers so _commit() is a no-op."""
        from ksi.memory.store import MemoryStore

        db_path = str(tmp_path / "batch.sqlite")
        store = MemoryStore(db_path, default_experiment="test")
        try:
            # Track whether _batch_mode was True every time _commit was called
            batch_states: list[bool] = []
            original_commit = store._commit

            def tracking_commit():
                batch_states.append(store._batch_mode)
                original_commit()

            store._commit = tracking_commit
            store.insert_raw_transcript(
                experiment="test",
                agent_id="agent-1",
                generation=1,
                task_id="task-1",
                content="test content",
                model_output="output",
                native_score=1.0,
            )
            # Every _commit() call from inside _batched() should see _batch_mode=True,
            # meaning no intermediate conn.commit() was issued.
            assert len(batch_states) > 0, "Expected _commit() to be called at least once"
            assert all(batch_states), f"Expected all _commit() calls to see _batch_mode=True, got {batch_states}"
            # After _batched() exits, batch_mode should be back to False
            assert not store._batch_mode, "_batch_mode should be False after _batched() exits"
        finally:
            store.close()

    def test_insert_and_retrieve(self, tmp_path):
        store = _make_store(tmp_path)
        store.insert_raw_transcript(
            experiment="exp1",
            agent_id="agent-0",
            generation=1,
            task_id="task-1",
            content='{"event":"message"}',
        )
        result = store.get_raw_transcript(task_id="task-1")
        assert result is not None
        assert result["content"] == '{"event":"message"}'
        assert result["agent_id"] == "agent-0"

    def test_get_latest_when_multiple(self, tmp_path):
        store = _make_store(tmp_path)
        store.insert_raw_transcript(
            experiment="exp1",
            agent_id="agent-0",
            generation=1,
            task_id="task-1",
            content="old",
        )
        store.insert_raw_transcript(
            experiment="exp1",
            agent_id="agent-1",
            generation=2,
            task_id="task-1",
            content="new",
        )
        result = store.get_raw_transcript(task_id="task-1")
        assert result["content"] == "new"
        assert result["generation"] == 2

    def test_get_with_generation_filter(self, tmp_path):
        store = _make_store(tmp_path)
        store.insert_raw_transcript(
            experiment="exp1",
            agent_id="agent-0",
            generation=1,
            task_id="task-1",
            content="gen1",
        )
        store.insert_raw_transcript(
            experiment="exp1",
            agent_id="agent-1",
            generation=2,
            task_id="task-1",
            content="gen2",
        )
        result = store.get_raw_transcript(task_id="task-1", generation=1)
        assert result["content"] == "gen1"


class TestTaskSummaries:
    def test_insert_and_list(self, tmp_path):
        store = _make_store(tmp_path)
        store.insert_task_summary(
            id="s1",
            experiment="exp1",
            agent_id="agent-0",
            generation=1,
            task_id="task-1",
            repo="django/django",
            approach="Fixed migration cache",
            key_files=["loader.py"],
            outcome="resolved",
            score=1.0,
            lessons=["Check cache invalidation"],
        )
        rows = store.list_task_summaries(experiment="exp1")
        assert len(rows) == 1
        assert rows[0]["approach"] == "Fixed migration cache"
        assert rows[0]["outcome"] == "resolved"

    def test_insert_task_summary_duplicate_id_replaces_row(self, tmp_path):
        """Re-inserting a summary with the same ID must upsert, not duplicate."""
        store = _make_store(tmp_path)
        common = dict(
            experiment="exp",
            agent_id="a1",
            generation=1,
            task_id="t1",
            repo="r",
            key_files=[],
            outcome="resolved",
            score=1.0,
            lessons=["l1"],
        )
        store.insert_task_summary(id="s1", approach="first approach", **common)
        store.insert_task_summary(id="s1", approach="second approach", **common)
        rows = store._execute(
            "SELECT body FROM memory_docs WHERE id = 's1'",
            fetchall=True,
        )
        assert len(rows) == 1, "duplicate summary id must upsert a single memory_docs row"
        assert "second approach" in rows[0]["body"]
        assert "first approach" not in rows[0]["body"]
        summaries = store.list_task_summaries(experiment="exp")
        assert len(summaries) == 1
        assert summaries[0]["approach"] == "second approach"
        store.close()

    def test_close_is_idempotent(self, tmp_path):
        store = _make_store(tmp_path)
        store.close()
        store.close()  # should not raise

    def test_list_task_summaries_respects_limit(self, tmp_path):
        """list_task_summaries should not return more rows than the limit."""
        from ksi.memory.store import MemoryStore

        db_path = str(tmp_path / "test.sqlite")
        store = MemoryStore(db_path, default_experiment="test")
        try:
            for i in range(10):
                store.insert_task_summary(
                    id=f"sum-{i}",
                    experiment="test",
                    agent_id=f"agent-{i}",
                    generation=1,
                    task_id=f"task-{i}",
                    repo=None,
                    approach=f"approach {i}",
                    outcome="pass",
                    score=1.0,
                    key_files=[],
                    lessons=[],
                )
            results = store.list_task_summaries(experiment="test", limit=5)
            assert len(results) <= 5, f"Expected at most 5 results, got {len(results)}"
        finally:
            store.close()


class TestDefaultSchemaHasAllTables:
    """Test that the default MemoryStore schema creates all tables."""

    def _tables(self, store):
        rows = store._execute(
            "SELECT name FROM sqlite_master WHERE type='table'",
            fetchall=True,
        )
        return {r["name"] for r in rows}

    def test_default_has_all_tables(self, tmp_path):
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "default.sqlite"))
        tables = self._tables(store)
        assert "runs" in tables
        assert "memory_docs" in tables
        assert "forum_events" in tables
        assert "task_memory_records" in tables
        store.close()


class TestWriterThreadAssertion:
    """Test that _ensure_run works via writer thread."""

    def test_ensure_run_works_inside_run_write(self, tmp_path):
        store = _make_store(tmp_path)
        result = [None]

        def _write():
            result[0] = store._ensure_run("test_exp")

        store._run_write(_write)
        assert isinstance(result[0], int)
        assert result[0] > 0
        store.close()


class TestTaskMemoryRecords:
    def test_upsert_and_query_task_memory(self, tmp_path):
        store = _make_store(tmp_path)
        store.upsert_task_memory_record(
            experiment="exp1",
            generation=1,
            agent_id="agent-0",
            task_id="task-1",
            eval_results={"resolved": False, "status": "ok", "native_score": 0.0},
            final_model_output="out-1",
            full_memory_trace="trace-1",
            full_memory_trace_condensed="condensed-1",
            task_specific_insights=["insight-a"],
            attempt_event={"status": "ok", "resolved": False},
        )
        rows = store.query_task_memory(task_id="task-1", experiment="exp1")
        assert len(rows) == 1
        row = rows[0]
        assert row["task_id"] == "task-1"
        assert row["agent_id"] == "agent-0"
        assert row["gen"] == 1
        assert row["final_model_output"] == "out-1"
        assert row["full_memory_trace"] == "trace-1"
        assert row["full_memory_trace_condensed"] == "condensed-1"
        assert row["task_specific_insights"] == ["insight-a"]

    def test_upsert_same_key_updates_and_appends_history(self, tmp_path):
        store = _make_store(tmp_path)
        kwargs = dict(
            experiment="exp1",
            generation=1,
            agent_id="agent-0",
            task_id="task-1",
        )
        store.upsert_task_memory_record(
            **kwargs,
            eval_results={"resolved": False},
            final_model_output="out-1",
            full_memory_trace="trace-1",
            full_memory_trace_condensed="condensed-1",
            task_specific_insights=["insight-a"],
            attempt_event={"status": "first"},
        )
        store.upsert_task_memory_record(
            **kwargs,
            eval_results={"resolved": True},
            final_model_output="out-2",
            full_memory_trace="trace-2",
            full_memory_trace_condensed="condensed-2",
            task_specific_insights=["insight-b"],
            attempt_event={"status": "second"},
        )
        rows = store.query_task_memory(task_id="task-1", experiment="exp1")
        assert len(rows) == 1
        row = rows[0]
        assert row["final_model_output"] == "out-2"
        assert row["task_specific_insights"] == ["insight-b"]
        assert isinstance(row["attempt_history"], list)
        assert len(row["attempt_history"]) == 2


class TestGetBestScores:
    def test_empty_store_returns_empty_dict(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get_best_scores(experiment="exp1") == {}
        store.close()

    def test_returns_best_score_per_task(self, tmp_path):
        store = _make_store(tmp_path)
        store.upsert_task_memory_record(
            experiment="exp1",
            generation=1,
            agent_id="agent-0",
            task_id="task-1",
            eval_results={"native_score": 0.3, "resolved": False},
            final_model_output="out-1",
            full_memory_trace="trace-1",
            full_memory_trace_condensed="condensed-1",
            task_specific_insights=[],
            attempt_event={"status": "ok"},
        )
        store.upsert_task_memory_record(
            experiment="exp1",
            generation=2,
            agent_id="agent-1",
            task_id="task-1",
            eval_results={"native_score": 0.8, "resolved": False},
            final_model_output="out-2",
            full_memory_trace="trace-2",
            full_memory_trace_condensed="condensed-2",
            task_specific_insights=[],
            attempt_event={"status": "ok"},
        )
        best = store.get_best_scores(experiment="exp1")
        assert best["task-1"] == 0.8
        store.close()

    def test_experiment_filter_isolates_results(self, tmp_path):
        store = _make_store(tmp_path)
        store.upsert_task_memory_record(
            experiment="exp1",
            generation=1,
            agent_id="agent-0",
            task_id="task-1",
            eval_results={"native_score": 1.0, "resolved": True},
            final_model_output="out-1",
            full_memory_trace="trace-1",
            full_memory_trace_condensed="condensed-1",
            task_specific_insights=[],
            attempt_event={"status": "ok"},
        )
        store.upsert_task_memory_record(
            experiment="exp2",
            generation=1,
            agent_id="agent-0",
            task_id="task-2",
            eval_results={"native_score": 0.5, "resolved": False},
            final_model_output="out-2",
            full_memory_trace="trace-2",
            full_memory_trace_condensed="condensed-2",
            task_specific_insights=[],
            attempt_event={"status": "ok"},
        )
        best_exp1 = store.get_best_scores(experiment="exp1")
        assert "task-1" in best_exp1
        assert "task-2" not in best_exp1

        best_exp2 = store.get_best_scores(experiment="exp2")
        assert "task-2" in best_exp2
        assert "task-1" not in best_exp2
        store.close()


class TestConcurrentRunWrite:
    """Stress test: multiple threads calling _run_write-wrapped methods concurrently."""

    def test_concurrent_inserts_all_succeed(self, tmp_path):
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "concurrent.sqlite"))
        num_threads = 10
        errors: list[Exception] = []

        def _insert(i: int) -> None:
            try:
                store.insert_raw_transcript(
                    experiment="exp1",
                    agent_id=f"agent-{i}",
                    generation=1,
                    task_id=f"task-{i}",
                    content=f"transcript content {i}",
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_insert, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Concurrent inserts raised errors: {errors}"

        # Verify all rows landed
        rows = store._execute(
            "SELECT COUNT(*) as cnt FROM attempt_artifacts WHERE artifact_type = 'transcript'",
            fetchone=True,
        )
        assert rows["cnt"] == num_threads
        store.close()

    def test_concurrent_task_summary_inserts(self, tmp_path):
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "concurrent_summary.sqlite"))
        num_threads = 10
        errors: list[Exception] = []

        def _insert(i: int) -> None:
            try:
                store.insert_task_summary(
                    id=f"s{i}",
                    experiment="exp1",
                    agent_id=f"agent-{i}",
                    generation=1,
                    task_id=f"task-{i}",
                    repo="repo/test",
                    approach=f"Approach {i}",
                    key_files=[f"file{i}.py"],
                    outcome="resolved",
                    score=1.0,
                    lessons=[f"lesson {i}"],
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_insert, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Concurrent summary inserts raised errors: {errors}"

        # Verify row count via memory_docs (each summary creates one memory_doc)
        rows = store._execute(
            "SELECT COUNT(*) as cnt FROM memory_docs WHERE scope = 'task_summary'",
            fetchone=True,
        )
        assert rows["cnt"] == num_threads
        store.close()


class TestRunWriteTimeout:
    """Test _run_write timeout and normal-case behavior."""

    def test_run_write_succeeds_normally(self, tmp_path):
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "rw.sqlite"))
        result_box = [None]

        def _op():
            result_box[0] = "done"
            return 42

        ret = store._run_write(_op)
        assert ret == 42
        assert result_box[0] == "done"
        store.close()

    def test_run_write_propagates_exception(self, tmp_path):
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "rw_err.sqlite"))

        def _op():
            raise ValueError("test error inside writer")

        with pytest.raises(ValueError, match="test error inside writer"):
            store._run_write(_op)
        store.close()

    def test_run_write_timeout_raises_runtime_error(self, tmp_path):
        """When the writer thread is dead/stuck, _run_write should raise RuntimeError."""
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "rw_timeout.sqlite"))

        # Drain the writer thread by sending the sentinel (None) so it exits
        store._writer_queue.put(None)
        store._writer_thread.join(timeout=5)

        # Now the writer thread is dead, but the queue still exists.
        # Enqueue an operation — the done event will never be set.
        # Patch done.wait timeout to be very short to avoid slow tests.
        original_run_write = store._run_write

        def _op():
            return "should never complete"

        # We need to shorten the timeout for this test
        done_orig = threading.Event

        class FastTimeoutEvent(threading.Event):
            def wait(self, timeout=None):
                return super().wait(timeout=0.1)

        with patch("threading.Event", FastTimeoutEvent):
            with pytest.raises(RuntimeError, match="writer thread did not respond"):
                store._run_write(_op)

        # Cleanup: writer thread is already dead, just close the connection
        store._writer_queue = None
        store._writer_thread = None
        store.close()


class _FastWaitEvent(threading.Event):
    """Event whose wait() blocks at most 0.3s (still returning immediately
    once set) — compresses _run_write's 180s stall floor for tests."""

    def wait(self, timeout=None):
        return super().wait(timeout=0.3)


class TestRunWriteCancellation:
    """_run_write must be at-most-once: a write whose stall timeout fires is
    cancelled, not left in the queue to apply later (issue #767). Before the
    fix, a timed-out closure stayed queued and executed on writer recovery, so
    the sidecar retry-once-then-drop guards could write the same row twice."""

    @staticmethod
    def _occupy_writer(store, claimed: threading.Event, release: threading.Event) -> threading.Thread:
        """Block the single writer thread until ``release`` is set. Returns
        only after the worker has actually claimed the blocker closure, so
        subsequent writes deterministically queue behind it. The events must
        be real (created before the _FastWaitEvent patch)."""

        def _blocker():
            claimed.set()
            release.wait(5.0)

        def _target():
            try:
                store._run_write(_blocker)
            except RuntimeError:
                pass  # under _FastWaitEvent the blocker submission itself times out

        t = threading.Thread(target=_target)
        t.start()
        assert claimed.wait(2.0), "worker never claimed the blocker closure"
        return t

    def test_timed_out_write_is_cancelled_and_never_applies(self, tmp_path):
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "cancel.sqlite"))
        executed = []
        claimed, release = threading.Event(), threading.Event()
        with patch("threading.Event", _FastWaitEvent):
            blocker = self._occupy_writer(store, claimed, release)
            with pytest.raises(RuntimeError, match="write cancelled"):
                store._run_write(lambda: executed.append(1))
        release.set()
        blocker.join(timeout=5)
        store._run_write(lambda: None)  # barrier: queue fully drained
        assert executed == []
        store.close()

    def test_retry_after_stall_does_not_duplicate_rows(self, tmp_path):
        """The exact issue-#767 shape: a caller that retries after the stall
        error (SqlitePersistence's retry-once-then-drop guards) must not end
        up with the row applied twice once the writer recovers."""
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "dup.sqlite"))
        kwargs = dict(
            generation=1,
            round_num=2,
            agent_id="agent-1",
            message_type="error",
            content={"text": "hi"},
            experiment="exp",
        )
        claimed, release = threading.Event(), threading.Event()
        with patch("threading.Event", _FastWaitEvent):
            blocker = self._occupy_writer(store, claimed, release)
            for _attempt in (1, 2):  # mirror the sidecar guard's retry-once
                with pytest.raises(RuntimeError, match="write cancelled"):
                    store.insert_forum_message(**kwargs)
        release.set()
        blocker.join(timeout=5)
        store._run_write(lambda: None)  # barrier: queue fully drained
        rows = store._execute("SELECT COUNT(*) AS cnt FROM forum_events", fetchone=True)
        assert rows["cnt"] == 0
        # After recovery the same write succeeds exactly once — the cancelled
        # attempts left nothing behind to double-apply.
        store.insert_forum_message(**kwargs)
        rows = store._execute("SELECT COUNT(*) AS cnt FROM forum_events", fetchone=True)
        assert rows["cnt"] == 1
        store.close()

    def test_running_write_gets_grace_period_instead_of_cancel(self, tmp_path):
        """A closure the worker is already executing cannot be cancelled; the
        caller waits one more window and consumes the result normally."""
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "grace.sqlite"))
        with patch("threading.Event", _FastWaitEvent):
            ret = store._run_write(lambda: time.sleep(0.45) or 42)
        assert ret == 42
        store.close()

    def test_unclaimable_running_write_raises_indeterminate(self, tmp_path):
        """If the closure is still executing after the grace window, the error
        must be the dedicated WriteIndeterminateError so best-effort callers
        can drop instead of retrying (a generic retry could duplicate rows)."""
        from ksi.errors import WriteIndeterminateError
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "indet.sqlite"))
        finished = []

        def _slow():
            time.sleep(1.2)
            finished.append(1)

        with patch("threading.Event", _FastWaitEvent):
            with pytest.raises(WriteIndeterminateError, match="may still be applied"):
                store._run_write(_slow)
        store._run_write(lambda: None)  # barrier: waits for _slow to finish
        assert finished == [1]  # the message was truthful — it did apply
        store.close()


class TestCloseWithActiveWriter:
    """Test that close() handles a busy writer thread gracefully."""

    def test_close_after_writes_completes_cleanly(self, tmp_path):
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "close_clean.sqlite"))
        # Do some writes first
        store.insert_raw_transcript(
            experiment="exp1",
            agent_id="agent-0",
            generation=1,
            task_id="task-1",
            content="content",
        )
        # close() should complete without error
        store.close()
        # After close, connection should be None
        assert store._conn is None
        assert store._writer_thread is None

    def test_close_with_stuck_writer_cleans_up(self, tmp_path):
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "close_stuck.sqlite"))

        # Replace the writer thread with a mock that stays alive
        original_thread = store._writer_thread

        class StuckThread:
            def join(self, timeout=None):
                pass  # Pretend to join but never actually finish

            def is_alive(self):
                return True  # Always report alive

        # Stop the real writer thread first
        store._writer_queue.put(None)
        original_thread.join(timeout=5)

        # Swap in the stuck mock
        store._writer_thread = StuckThread()

        # close() should complete without error even with a stuck writer
        store.close()
        # After close, references should be cleared
        assert store._writer_queue is None
        assert store._writer_thread is None
        assert store._conn is None

    def test_double_close_is_safe(self, tmp_path):
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "double_close.sqlite"))
        store.close()
        store.close()  # second close should not raise

    def test_store_is_collectible_without_explicit_close(self, tmp_path):
        from ksi.memory.store import MemoryStore

        def _fd_count() -> int | None:
            fd_root = Path("/dev/fd")
            if not fd_root.exists():
                return None
            try:
                return len(list(fd_root.iterdir()))
            except Exception:
                return None

        baseline_fds = _fd_count()
        refs = []

        for idx in range(40):
            store = MemoryStore(str(tmp_path / f"leak_{idx}.sqlite"))
            store.insert_forum_message(
                generation=1,
                round_num=1,
                agent_id=f"agent-{idx}",
                message_type="insight",
                content="payload",
                experiment="exp",
            )
            refs.append(weakref.ref(store))
            store = None

        for _ in range(5):
            gc.collect()

        assert all(ref() is None for ref in refs)

        final_fds = _fd_count()
        if baseline_fds is not None and final_fds is not None:
            assert final_fds - baseline_fds < 20


class TestAllTablesPresentInDefaultSchema:
    """Verify all table types are accessible in the default (full) schema."""

    def test_transcript_insert_succeeds(self, tmp_path):
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "full.sqlite"))
        store.insert_raw_transcript(
            experiment="exp1",
            agent_id="agent-0",
            generation=1,
            task_id="task-1",
            content="transcript content",
        )
        result = store.get_raw_transcript(task_id="task-1")
        assert result is not None
        assert result["content"] == "transcript content"
        store.close()

    def test_forum_message_insert_succeeds(self, tmp_path):
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "full.sqlite"))
        store.insert_forum_message(
            generation=1,
            agent_id="agent-0",
            message_type="insight",
            round_num=1,
            experiment="exp1",
            content={"text": "test insight"},
        )
        rows = store.list_forum_messages(1, experiment="exp1")
        assert len(rows) >= 1
        store.close()

    def test_forum_round_payload_insert_succeeds(self, tmp_path):
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "full.sqlite"))
        store.insert_forum_message(
            generation=1,
            agent_id="agent-0",
            message_type="forum_round",
            round_num=1,
            experiment="exp1",
            content={"parsed_item_count": 2, "status": "ok"},
        )
        rows = store.list_forum_messages(1, experiment="exp1")
        assert any(r["message_type"] == "forum_round" for r in rows)
        store.close()

    def test_task_memory_upsert_succeeds(self, tmp_path):
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "full.sqlite"))
        store.upsert_task_memory_record(
            experiment="exp1",
            generation=1,
            agent_id="agent-0",
            task_id="task-1",
            eval_results={"resolved": False},
            final_model_output="out",
            full_memory_trace="trace",
            full_memory_trace_condensed="condensed",
            task_specific_insights=["insight"],
            attempt_event={"status": "ok"},
        )
        rows = store.query_task_memory(task_id="task-1", experiment="exp1")
        assert len(rows) == 1
        store.close()


class TestAssignmentTimestamps:
    """Regression: assignments.started_at / ended_at must be populated.

    Before this fix, the live Haiku baseline sweep produced 179 assignment rows
    with 0/179 started_at and 0/179 ended_at populated, making per-attempt
    wall-clock durations unrecoverable without scraping tool_trace timestamps.
    """

    def _assignment_row(self, store, task_id: str):
        return store._execute(
            "SELECT asg.id, asg.status, asg.created_at, asg.started_at, asg.ended_at "
            "FROM assignments asg "
            "JOIN tasks t ON t.id = asg.task_ref "
            "WHERE t.task_id = ?",
            (task_id,),
            fetchone=True,
        )

    def test_mark_assignment_started_sets_started_at(self, tmp_path):
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "ts.sqlite"), default_experiment="exp1")
        try:
            store.mark_assignment_started(
                experiment="exp1",
                generation=1,
                agent_id="agent-0",
                task_id="task-1",
            )
            row = self._assignment_row(store, "task-1")
            assert row is not None, "assignment row should exist after mark_assignment_started"
            assert row["started_at"] is not None, "started_at should be populated"
            assert row["ended_at"] is None, "ended_at should still be NULL"
            assert row["status"] == "started"
        finally:
            store.close()

    def test_mark_assignment_ended_sets_both_timestamps(self, tmp_path):
        """End-to-end: dispatch → complete → both timestamps present, ended >= started."""
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "ts.sqlite"), default_experiment="exp1")
        try:
            store.mark_assignment_started(
                experiment="exp1",
                generation=1,
                agent_id="agent-0",
                task_id="task-1",
            )
            # Small delay so started_at and ended_at are distinguishable at second
            # granularity (datetime('now') returns whole seconds). This is a real
            # invariant for analytics (duration >= 0), not just cosmetic.
            time.sleep(1.1)
            store.mark_assignment_ended(
                experiment="exp1",
                generation=1,
                agent_id="agent-0",
                task_id="task-1",
                status="completed",
            )
            row = self._assignment_row(store, "task-1")
            assert row is not None
            assert row["started_at"] is not None, "started_at must be populated"
            assert row["ended_at"] is not None, "ended_at must be populated"
            assert row["status"] == "completed"
            # datetime('now') returns ISO-8601-ish 'YYYY-MM-DD HH:MM:SS' — lexicographic
            # comparison matches chronological comparison.
            assert row["ended_at"] >= row["started_at"], (
                f"ended_at ({row['ended_at']}) must be >= started_at ({row['started_at']})"
            )
        finally:
            store.close()

    def test_mark_assignment_started_is_idempotent(self, tmp_path):
        """Retries must not clobber the original start time."""
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "ts.sqlite"), default_experiment="exp1")
        try:
            store.mark_assignment_started(
                experiment="exp1",
                generation=1,
                agent_id="agent-0",
                task_id="task-1",
            )
            first = self._assignment_row(store, "task-1")["started_at"]
            time.sleep(1.1)
            store.mark_assignment_started(
                experiment="exp1",
                generation=1,
                agent_id="agent-0",
                task_id="task-1",
            )
            second = self._assignment_row(store, "task-1")["started_at"]
            assert first == second, "started_at must be preserved across retry-style calls"
        finally:
            store.close()

    def test_mark_assignment_ended_backfills_started_at(self, tmp_path):
        """If ended is called without a prior start (e.g., persistence was wired
        late), started_at is back-filled so duration analytics are non-NULL."""
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "ts.sqlite"), default_experiment="exp1")
        try:
            store.mark_assignment_ended(
                experiment="exp1",
                generation=1,
                agent_id="agent-0",
                task_id="task-1",
                status="failed",
            )
            row = self._assignment_row(store, "task-1")
            assert row is not None
            assert row["started_at"] is not None, "started_at back-fill missing"
            assert row["ended_at"] is not None
            assert row["status"] == "failed"
        finally:
            store.close()

    def test_sqlite_persistence_on_task_status_populates_timestamps(self, tmp_path):
        """Covers the cli.py SqlitePersistence wiring end-to-end.

        Regression for the 0/0 assignments timestamps observed in the live
        Haiku baseline sweep (results/baseline_sweep_haiku/...).
        """
        from ksi.cli import SqlitePersistence

        db_path = str(tmp_path / "persist.sqlite")
        persist = SqlitePersistence(runtime_db_path=db_path, experiment_name="exp1")
        try:
            # Engine-style event sequence.
            persist.on_task_status(
                generation=1,
                agent_id="agent-0",
                task_id="task-1",
                status="started",
            )
            time.sleep(1.1)
            persist.on_task_status(
                generation=1,
                agent_id="agent-0",
                task_id="task-1",
                status="completed",
            )
            store = persist._ensure_store()
            row = store._execute(
                "SELECT asg.started_at, asg.ended_at, asg.status "
                "FROM assignments asg "
                "JOIN tasks t ON t.id = asg.task_ref "
                "WHERE t.task_id = ?",
                ("task-1",),
                fetchone=True,
            )
            assert row is not None
            assert row["started_at"] is not None
            assert row["ended_at"] is not None
            assert row["ended_at"] >= row["started_at"]
            assert row["status"] == "completed"
        finally:
            store = persist._store
            if store is not None:
                store.close()

    def test_insert_task_trace_then_status_events_populate_timestamps(self, tmp_path):
        """Integration-ish: mimic the engine flow — 'started' then insert_task_trace
        (which already calls _ensure_assignment) then 'completed'. Both timestamps
        should be present on the single assignment row."""
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "ts.sqlite"), default_experiment="exp1")
        try:
            store.mark_assignment_started(
                experiment="exp1",
                generation=1,
                agent_id="agent-0",
                task_id="task-1",
            )
            store.insert_task_trace(
                experiment="exp1",
                generation=1,
                agent_id="agent-0",
                task_id="task-1",
                model_output="output",
                eval_result={"native_score": 1.0},
                native_score=1.0,
                tool_trace=[],
                runtime_meta={},
            )
            store.mark_assignment_ended(
                experiment="exp1",
                generation=1,
                agent_id="agent-0",
                task_id="task-1",
                status="completed",
            )
            # Exactly one assignment row for this (gen, agent, task) triple.
            rows = store._execute(
                "SELECT id, status, started_at, ended_at FROM assignments",
                fetchall=True,
            )
            assert len(rows) == 1
            r = rows[0]
            assert r["started_at"] is not None
            assert r["ended_at"] is not None
            assert r["status"] == "completed"
            # attempts.created_at is already auto-populated by the schema default —
            # regression guard so that auto-population is not broken by this change.
            att_rows = store._execute(
                "SELECT COUNT(*) AS total, COUNT(created_at) AS filled FROM attempts",
                fetchone=True,
            )
            assert att_rows["total"] >= 1
            assert att_rows["filled"] == att_rows["total"]
        finally:
            store.close()


class TestTasksRepoPreservation:
    """Regression tests for the data-integrity bug where tasks.repo was
    being left empty for silent-failing swepro tasks (tutao/tutanota,
    internetarchive/openlibrary, flipt-io/flipt).

    Forensic evidence: half the rows in the runtime knowledge DB had
    tasks.repo='' because the only writer that reliably plumbed the repo
    through (insert_task_summary) was skipped on traces with
    error!=None — which is exactly the silent-failure case.

    Root-cause fix is callers passing TaskSpec.repo; these tests pin
    the store-level invariant as defense-in-depth.
    """

    def _repo_of(self, store, task_id: str) -> str:
        row = store._execute(
            "SELECT repo FROM tasks WHERE task_id = ?",
            (task_id,),
            fetchone=True,
        )
        assert row is not None, f"tasks row for {task_id} should exist"
        return row["repo"] or ""

    def test_repeated_upsert_preserves_existing_repo_when_empty_comes_second(self, tmp_path):
        """Write task twice: first with repo='owner/name', then with repo='' —
        the final row MUST still have repo='owner/name' (the COALESCE preserve)."""
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "repo_preserve.sqlite"), default_experiment="exp1")
        try:
            run_id = store._ensure_run("exp1")
            # First write: real repo (mimics the loader- or summary-path).
            store._ensure_task(run_id, "task-1", repo="tutao/tutanota")
            assert self._repo_of(store, "task-1") == "tutao/tutanota"
            # Second write: empty repo (mimics on_task_trace on a silent-failure).
            store._ensure_task(run_id, "task-1", repo="")
            assert self._repo_of(store, "task-1") == "tutao/tutanota", (
                "COALESCE(NULLIF(?, ''), repo) must preserve the non-empty repo"
            )
            # Third write: None (mimics mark_assignment_started with default repo=None).
            store._ensure_task(run_id, "task-1", repo=None)
            assert self._repo_of(store, "task-1") == "tutao/tutanota"
        finally:
            store.close()

    def test_repeated_upsert_sets_repo_when_empty_came_first(self, tmp_path):
        """Write task twice: first with repo='' (or None), then with real repo —
        the final row MUST have the real repo value.

        This is the bug we're actually fixing: mark_assignment_started fires
        first with repo=None, and insert_task_trace used to hard-code repo=''.
        Once the caller passes the real repo, the second write must update
        the blank row to the real value.
        """
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "repo_set_after.sqlite"), default_experiment="exp1")
        try:
            run_id = store._ensure_run("exp1")
            # First write: blank (mimics mark_assignment_started).
            store._ensure_task(run_id, "task-2", repo=None)
            assert self._repo_of(store, "task-2") == ""
            # Second write: real repo (mimics on_task_trace with trace.repo populated).
            store._ensure_task(run_id, "task-2", repo="internetarchive/openlibrary")
            assert self._repo_of(store, "task-2") == "internetarchive/openlibrary"
        finally:
            store.close()

    def test_insert_task_trace_with_repo_populates_tasks_row(self, tmp_path):
        """End-to-end: after insert_task_trace(..., repo='owner/name'), the
        tasks row must contain that repo. Pins the fix for
        SqlitePersistence.on_task_trace which used to hardcode repo=''.
        """
        from ksi.memory.store import MemoryStore

        store = MemoryStore(str(tmp_path / "trace_repo.sqlite"), default_experiment="exp1")
        try:
            # Mimic engine flow: started (no repo known to this call site) → trace (repo known).
            store.mark_assignment_started(
                experiment="exp1",
                generation=1,
                agent_id="agent-0",
                task_id="flipt",
            )
            assert self._repo_of(store, "flipt") == "", "started writer has no repo info"
            store.insert_task_trace(
                experiment="exp1",
                generation=1,
                agent_id="agent-0",
                task_id="flipt",
                repo="flipt-io/flipt",
                model_output=None,
                eval_result={},
                native_score=None,
                tool_trace=[],
                runtime_meta={},
                error_text="Silent agent-runner failure",
            )
            assert self._repo_of(store, "flipt") == "flipt-io/flipt", (
                "insert_task_trace must populate tasks.repo even on silent-failure traces"
            )
        finally:
            store.close()

    def test_composite_persistence_propagates_experiment_rename(self, tmp_path):
        """Engine collision suffixing must reach wrapped SQLite observers."""
        from ksi.cli import CollectingPersistence, CompositePersistence, SqlitePersistence

        db_path = str(tmp_path / "persist.sqlite")
        sqlite = SqlitePersistence(runtime_db_path=db_path, experiment_name="original")
        persist = CompositePersistence([CollectingPersistence(), sqlite])
        persist.experiment_name = "original_2"
        try:
            persist.on_task_status(
                generation=1,
                agent_id="agent-0",
                task_id="task-1",
                status="started",
            )
            store = sqlite._ensure_store()
            row = store._execute(
                "SELECT experiment FROM runs",
                fetchone=True,
            )
            assert row["experiment"] == "original_2"
        finally:
            sqlite.close()

    def test_composite_persistence_keeps_local_experiment_name_without_observers(self):
        """CompositePersistence should retain renamed state even with no named observers."""
        from ksi.cli import CompositePersistence

        persist = CompositePersistence([])
        persist.experiment_name = "renamed"

        assert persist.experiment_name == "renamed"

    def test_sqlite_persistence_rename_updates_open_store_default_experiment(self, tmp_path):
        """Renaming after store creation should update the open store's experiment."""
        from ksi.cli import SqlitePersistence

        db_path = str(tmp_path / "persist.sqlite")
        persist = SqlitePersistence(runtime_db_path=db_path, experiment_name="original")
        try:
            persist._ensure_store()
            persist.set_experiment_name("renamed")
            persist.on_task_status(
                generation=1,
                agent_id="agent-0",
                task_id="task-1",
                status="started",
            )
            store = persist._ensure_store()
            row = store._execute(
                "SELECT experiment FROM runs",
                fetchone=True,
            )
            assert row["experiment"] == "renamed"
        finally:
            persist.close()

    def test_sqlite_persistence_on_task_trace_plumbs_trace_repo(self, tmp_path):
        """Pin that SqlitePersistence.on_task_trace reads trace.repo (not ''),
        so the 3 always-silent-failing swepro tasks get their repo column
        populated on the first attempt.
        """
        from ksi.cli import SqlitePersistence
        from ksi.models import TaskTrace, TokenUsage

        db_path = str(tmp_path / "persist.sqlite")
        persist = SqlitePersistence(runtime_db_path=db_path, experiment_name="exp1")
        try:
            trace = TaskTrace(
                generation=1,
                agent_id="agent-0",
                task_id="tutanota-silent",
                model_output=None,
                eval_result={},
                native_score=None,
                tool_trace=[],
                runtime_meta={},
                token_usage=TokenUsage(),
                error="Silent agent-runner failure for task tutanota-silent",
                repo="tutao/tutanota",
            )
            persist.on_task_trace(trace)
            store = persist._ensure_store()
            row = store._execute(
                "SELECT repo FROM tasks WHERE task_id = ?",
                ("tutanota-silent",),
                fetchone=True,
            )
            assert row is not None
            assert row["repo"] == "tutao/tutanota", "TaskTrace.repo must flow through on_task_trace into tasks.repo"
        finally:
            persist.close()


class TestBatchedRollback:
    """_batched() must roll back partial writes on exception (#703)."""

    def _count_runs(self, store, experiment):
        rows = store._execute(
            "SELECT COUNT(*) AS n FROM runs WHERE experiment = ?",
            (experiment,),
            fetchall=True,
        )
        return int(rows[0]["n"])

    def test_exception_mid_batch_rolls_back_partial_write(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            assert self._count_runs(store, "rb-fail") == 0

            class _Boom(Exception):
                pass

            with pytest.raises(_Boom):
                with store._batched():
                    # A valid write that succeeds inside the batch...
                    store._conn.execute(
                        "INSERT OR IGNORE INTO runs (experiment) VALUES (?)",
                        ("rb-fail",),
                    )
                    # ...then the block raises before the batch commits.
                    raise _Boom()

            # The partial write must have been rolled back.
            assert self._count_runs(store, "rb-fail") == 0
        finally:
            store.close()

    def test_successful_batch_commits(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            with store._batched():
                store._conn.execute(
                    "INSERT OR IGNORE INTO runs (experiment) VALUES (?)",
                    ("rb-ok",),
                )
            assert self._count_runs(store, "rb-ok") == 1
        finally:
            store.close()

    def test_inner_batch_defers_commit_and_rollback_to_outer(self, tmp_path):
        store = _make_store(tmp_path)
        try:

            class _Boom(Exception):
                pass

            with pytest.raises(_Boom):
                with store._batched():  # outer owner
                    store._conn.execute(
                        "INSERT OR IGNORE INTO runs (experiment) VALUES (?)",
                        ("rb-outer",),
                    )
                    # Inner batch (was_batch=True) must not commit/rollback;
                    # it defers to the outer owner, which rolls back on raise.
                    with store._batched():
                        store._conn.execute(
                            "INSERT OR IGNORE INTO runs (experiment) VALUES (?)",
                            ("rb-inner",),
                        )
                    raise _Boom()

            assert self._count_runs(store, "rb-outer") == 0
            assert self._count_runs(store, "rb-inner") == 0
        finally:
            store.close()

    def test_exception_inside_inner_batch_rolls_back_both(self, tmp_path):
        store = _make_store(tmp_path)
        try:

            class _Boom(Exception):
                pass

            # An exception raised INSIDE the inner batch must propagate through
            # the inner finally (which only restores _batch_mode, never commits
            # or rolls back) up to the outer owner's except BaseException, which
            # rolls back BOTH the inner (W2) and outer (W1) writes. No partial
            # commit may survive.
            with pytest.raises(_Boom):
                with store._batched():  # outer owner
                    store._conn.execute(  # W1
                        "INSERT OR IGNORE INTO runs (experiment) VALUES (?)",
                        ("rb-outer-w1",),
                    )
                    with store._batched():  # inner (was_batch=True)
                        store._conn.execute(  # W2
                            "INSERT OR IGNORE INTO runs (experiment) VALUES (?)",
                            ("rb-inner-w2",),
                        )
                        raise _Boom()

            assert self._count_runs(store, "rb-outer-w1") == 0
            assert self._count_runs(store, "rb-inner-w2") == 0
        finally:
            store.close()

"""Regression tests for issues #728/#756: sidecar persistence must be best-effort.

The runtime DB is a non-authoritative audit sidecar. A failed sidecar write
(``SqlitePersistence.on_task_trace`` / ``on_forum_message``) is retried once
in-method, then logged and dropped; a single failing observer inside
``CompositePersistence.on_task_trace`` / ``on_forum_message`` /
``on_task_status`` must log a warning and continue — it must never propagate
and abort the generation/run. The one exception is ``AuthenticationFailure``,
which the ``CompositePersistence`` layer deliberately re-raises so the run
aborts loudly instead of silently producing 0/N solved. The Sqlite-level
in-method guards catch bare ``Exception`` and therefore swallow it at that
layer — unreachable in production, since the store layer never raises it.
"""

import contextlib
import json
import logging
import threading
from unittest.mock import MagicMock, patch

import pytest
from conftest import _build_make_tasks, _build_mock_evaluator, _build_mock_llm, _build_mock_runtime

from ksi.cli import CompositePersistence, SqlitePersistence
from ksi.errors import AuthenticationFailure, WriteIndeterminateError
from ksi.memory.store import MemoryStore
from ksi.models import GenerationConfig, TaskTrace
from ksi.orchestrator.engine import GenerationalOrchestrator
from ksi.runtime.types import RuntimeResult
from ksi.tokens import LLMResponse, TokenUsage
from tests.orchestrator_phase_helpers import per_task_forum


def _make_trace() -> TaskTrace:
    return TaskTrace(generation=1, agent_id="agent-1", task_id="task-1")


def test_sqlite_persistence_on_task_trace_retries_once_then_succeeds(caplog):
    persist = SqlitePersistence(runtime_db_path="/nonexistent/unused.sqlite", experiment_name="exp")
    store = MagicMock()
    store.insert_task_trace.side_effect = [RuntimeError("writer thread stalled"), None]
    persist._store = store

    with caplog.at_level(logging.WARNING):
        assert persist.on_task_trace(_make_trace()) is None

    # The in-method retry persisted the trace on the second attempt.
    assert store.insert_task_trace.call_count == 2
    warnings = [rec for rec in caplog.records if "failed to record task trace" in rec.message]
    assert len(warnings) == 1
    rendered = warnings[0].getMessage()
    assert warnings[0].levelno == logging.WARNING
    assert "agent-1" in rendered
    assert "task-1" in rendered
    assert "writer thread stalled" in rendered
    assert "retrying once" in rendered
    # Fall-through: the ended_at back-stop still runs.
    store.mark_assignment_ended.assert_called_once()


def test_sqlite_persistence_on_task_trace_persistent_failure_swallowed_backstop_runs(caplog):
    persist = SqlitePersistence(runtime_db_path="/nonexistent/unused.sqlite", experiment_name="exp")
    store = MagicMock()
    store.insert_task_trace.side_effect = RuntimeError("writer thread stalled")
    persist._store = store

    with caplog.at_level(logging.WARNING):
        assert persist.on_task_trace(_make_trace()) is None

    # One retry only: two attempts total, then the failure is dropped.
    assert store.insert_task_trace.call_count == 2
    warnings = [rec for rec in caplog.records if "failed to record task trace" in rec.message]
    assert len(warnings) == 2
    assert "retrying once" in warnings[0].getMessage()
    assert "after retry; dropping" in warnings[1].getMessage()
    for rec in warnings:
        assert rec.levelno == logging.WARNING
        rendered = rec.getMessage()
        assert "agent-1" in rendered
        assert "task-1" in rendered
        assert "writer thread stalled" in rendered
    # No early return: the ended_at back-stop is still attempted even though
    # the insert failed (it may succeed for a data-dependent insert failure).
    store.mark_assignment_ended.assert_called_once()


def test_sqlite_persistence_on_task_trace_swallows_authentication_failure(caplog):
    """Pin: the Sqlite-level guard catches bare ``Exception``, so
    ``AuthenticationFailure`` (a RuntimeError subclass) is swallowed at this
    layer — only ``CompositePersistence`` re-raises it. Production-unreachable
    here because the store layer never raises it."""
    persist = SqlitePersistence(runtime_db_path="/nonexistent/unused.sqlite", experiment_name="exp")
    store = MagicMock()
    store.insert_task_trace.side_effect = AuthenticationFailure("invalid api key")
    persist._store = store

    with caplog.at_level(logging.WARNING):
        assert persist.on_task_trace(_make_trace()) is None

    # Two attempts (one retry), then the auth failure is dropped like any other.
    assert store.insert_task_trace.call_count == 2
    warnings = [rec for rec in caplog.records if "failed to record task trace" in rec.message]
    assert len(warnings) == 2
    assert "retrying once" in warnings[0].getMessage()
    assert "after retry; dropping" in warnings[1].getMessage()
    assert "invalid api key" in warnings[1].getMessage()


def test_composite_persistence_one_failing_observer_does_not_skip_others(caplog):
    failing = MagicMock()
    failing.on_task_trace.side_effect = RuntimeError("sidecar DB unavailable")
    healthy = MagicMock()
    composite = CompositePersistence(observers=[failing, healthy])
    trace = _make_trace()

    with caplog.at_level(logging.WARNING):
        composite.on_task_trace(trace)

    healthy.on_task_trace.assert_called_once_with(trace)
    warning = next(rec for rec in caplog.records if "failed on_task_trace" in rec.message)
    rendered = warning.getMessage()
    assert "agent-1" in rendered
    assert "task-1" in rendered
    assert "sidecar DB unavailable" in rendered


def test_composite_persistence_reraises_authentication_failure():
    failing = MagicMock()
    failing.on_task_trace.side_effect = AuthenticationFailure("invalid api key")
    healthy = MagicMock()
    composite = CompositePersistence(observers=[failing, healthy])

    with pytest.raises(AuthenticationFailure):
        composite.on_task_trace(_make_trace())


def _forum_kwargs() -> dict:
    return {
        "generation": 1,
        "round_num": 2,
        "agent_id": "agent-1",
        "message_type": "statement",
        "content_json": {"text": "hi"},
        "token_usage": {"total": 1},
    }


def test_sqlite_persistence_on_forum_message_retries_once_then_succeeds(caplog):
    persist = SqlitePersistence(runtime_db_path="/nonexistent/unused.sqlite", experiment_name="exp")
    store = MagicMock()
    store.insert_forum_message.side_effect = [RuntimeError("writer thread stalled"), None]
    persist._store = store

    with caplog.at_level(logging.WARNING):
        assert persist.on_forum_message(**_forum_kwargs()) is None

    # The in-method retry persisted the message on the second attempt.
    assert store.insert_forum_message.call_count == 2
    warnings = [rec for rec in caplog.records if "failed to record forum message" in rec.message]
    assert len(warnings) == 1
    rendered = warnings[0].getMessage()
    assert warnings[0].levelno == logging.WARNING
    assert "agent-1" in rendered
    assert "gen 1" in rendered
    assert "round 2" in rendered
    assert "writer thread stalled" in rendered
    assert "retrying once" in rendered


def test_sqlite_persistence_on_forum_message_persistent_failure_swallowed(caplog):
    persist = SqlitePersistence(runtime_db_path="/nonexistent/unused.sqlite", experiment_name="exp")
    store = MagicMock()
    store.insert_forum_message.side_effect = RuntimeError("writer thread stalled")
    persist._store = store

    with caplog.at_level(logging.WARNING):
        assert persist.on_forum_message(**_forum_kwargs()) is None

    # One retry only: two attempts total, then the failure is dropped.
    assert store.insert_forum_message.call_count == 2
    warnings = [rec for rec in caplog.records if "failed to record forum message" in rec.message]
    assert len(warnings) == 2
    assert "retrying once" in warnings[0].getMessage()
    assert "after retry; dropping" in warnings[1].getMessage()
    for rec in warnings:
        assert rec.levelno == logging.WARNING
        rendered = rec.getMessage()
        assert "agent-1" in rendered
        assert "gen 1" in rendered
        assert "round 2" in rendered
        assert "writer thread stalled" in rendered


def test_sqlite_persistence_on_forum_message_store_construction_failure_retried(caplog):
    """The retry loop wraps ``_ensure_store()`` itself: a store-construction
    failure on attempt 1 is retried, and the message persists on attempt 2."""
    persist = SqlitePersistence(runtime_db_path="/nonexistent/unused.sqlite", experiment_name="exp")
    store = MagicMock()

    with patch("ksi.memory.store.MemoryStore", side_effect=[RuntimeError("disk full"), store]) as ctor:
        with caplog.at_level(logging.WARNING):
            assert persist.on_forum_message(**_forum_kwargs()) is None

    # Construction failed once, succeeded on the in-method retry.
    assert ctor.call_count == 2
    store.insert_forum_message.assert_called_once()
    # The successfully constructed store is cached for subsequent callbacks.
    assert persist._store is store
    warnings = [rec for rec in caplog.records if "failed to record forum message" in rec.message]
    assert len(warnings) == 1
    rendered = warnings[0].getMessage()
    assert warnings[0].levelno == logging.WARNING
    assert "agent-1" in rendered
    assert "disk full" in rendered
    assert "retrying once" in rendered


def test_sqlite_persistence_on_forum_message_store_construction_persistent_failure(caplog):
    """Store construction failing on both attempts drops the message — and
    ``_store`` stays None: ``_ensure_store`` assigns it only on successful
    construction, so a broken store is never cached."""
    persist = SqlitePersistence(runtime_db_path="/nonexistent/unused.sqlite", experiment_name="exp")

    with patch("ksi.memory.store.MemoryStore", side_effect=RuntimeError("disk full")) as ctor:
        with caplog.at_level(logging.WARNING):
            assert persist.on_forum_message(**_forum_kwargs()) is None

    # One retry only: two construction attempts, then the failure is dropped.
    assert ctor.call_count == 2
    assert persist._store is None
    warnings = [rec for rec in caplog.records if "failed to record forum message" in rec.message]
    assert len(warnings) == 2
    assert "retrying once" in warnings[0].getMessage()
    assert "after retry; dropping" in warnings[1].getMessage()
    for rec in warnings:
        assert rec.levelno == logging.WARNING
        assert "disk full" in rec.getMessage()


def test_sqlite_persistence_on_forum_message_swallows_authentication_failure(caplog):
    """Pin: the Sqlite-level guard catches bare ``Exception``, so
    ``AuthenticationFailure`` (a RuntimeError subclass) is swallowed at this
    layer — only ``CompositePersistence`` re-raises it. Production-unreachable
    here because the store layer never raises it."""
    persist = SqlitePersistence(runtime_db_path="/nonexistent/unused.sqlite", experiment_name="exp")
    store = MagicMock()
    store.insert_forum_message.side_effect = AuthenticationFailure("invalid api key")
    persist._store = store

    with caplog.at_level(logging.WARNING):
        assert persist.on_forum_message(**_forum_kwargs()) is None

    # Two attempts (one retry), then the auth failure is dropped like any other.
    assert store.insert_forum_message.call_count == 2
    warnings = [rec for rec in caplog.records if "failed to record forum message" in rec.message]
    assert len(warnings) == 2
    assert "retrying once" in warnings[0].getMessage()
    assert "after retry; dropping" in warnings[1].getMessage()
    assert "invalid api key" in warnings[1].getMessage()


def test_sqlite_on_task_trace_does_not_retry_indeterminate_write(caplog):
    """A WriteIndeterminateError means the write may still apply later
    (issue #767): retrying could duplicate the row, so the guard must drop
    immediately instead of attempting a second write."""
    persist = SqlitePersistence(runtime_db_path="/nonexistent/unused.sqlite", experiment_name="exp")
    store = MagicMock()
    store.insert_task_trace.side_effect = WriteIndeterminateError("write may still be applied")
    persist._store = store

    with caplog.at_level(logging.WARNING):
        assert persist.on_task_trace(_make_trace()) is None

    assert store.insert_task_trace.call_count == 1
    warnings = [rec for rec in caplog.records if "not retrying" in rec.message]
    assert len(warnings) == 1
    rendered = warnings[0].getMessage()
    assert "agent-1" in rendered
    assert "task-1" in rendered
    assert "write may still be applied" in rendered


def test_sqlite_on_forum_message_does_not_retry_indeterminate_write(caplog):
    """Same contract for the forum-message guard (issue #767)."""
    persist = SqlitePersistence(runtime_db_path="/nonexistent/unused.sqlite", experiment_name="exp")
    store = MagicMock()
    store.insert_forum_message.side_effect = WriteIndeterminateError("write may still be applied")
    persist._store = store

    with caplog.at_level(logging.WARNING):
        assert persist.on_forum_message(**_forum_kwargs()) is None

    assert store.insert_forum_message.call_count == 1
    warnings = [rec for rec in caplog.records if "not retrying" in rec.message]
    assert len(warnings) == 1
    rendered = warnings[0].getMessage()
    assert "agent-1" in rendered
    assert "round 2" in rendered
    assert "write may still be applied" in rendered


def test_sqlite_on_run_end_summarizes_dropped_sidecar_writes(caplog):
    """Issue #771: dropped sidecar writes must surface as ONE aggregate
    summary warning at on_run_end, naming the per-family counts."""
    persist = SqlitePersistence(runtime_db_path="/nonexistent/unused.sqlite", experiment_name="exp")
    store = MagicMock()
    store.insert_task_trace.side_effect = RuntimeError("writer thread stalled")
    store.insert_forum_message.side_effect = WriteIndeterminateError("write may still be applied")
    persist._store = store

    with caplog.at_level(logging.WARNING):
        persist.on_task_trace(_make_trace())
        persist.on_forum_message(**_forum_kwargs())
        persist.on_forum_message(**_forum_kwargs())
        assert persist.on_run_end(token_summary=MagicMock()) is None

    summaries = [rec for rec in caplog.records if "dropped" in rec.message and "sidecar writes" in rec.message]
    assert len(summaries) == 1
    assert summaries[0].levelno == logging.WARNING
    rendered = summaries[0].getMessage()
    assert "[SqlitePersistence] dropped 3 sidecar writes this run" in rendered
    assert "(1 task traces, 2 forum messages, 0 task statuses)" in rendered
    assert "the runtime DB is incomplete" in rendered


def test_sqlite_on_run_end_silent_when_no_drops(caplog):
    """Zero drops -> on_run_end stays silent (no summary warning)."""
    persist = SqlitePersistence(runtime_db_path="/nonexistent/unused.sqlite", experiment_name="exp")
    persist._store = MagicMock()

    with caplog.at_level(logging.WARNING):
        persist.on_task_trace(_make_trace())
        persist.on_forum_message(**_forum_kwargs())
        assert persist.on_run_end(token_summary=MagicMock()) is None

    assert not [rec for rec in caplog.records if "sidecar writes" in rec.message]


def test_sqlite_on_run_end_does_not_count_retried_then_succeeded_writes(caplog):
    """A write that fails once and succeeds on the in-method retry is NOT a
    drop — it must not appear in the on_run_end summary."""
    persist = SqlitePersistence(runtime_db_path="/nonexistent/unused.sqlite", experiment_name="exp")
    store = MagicMock()
    store.insert_task_trace.side_effect = [RuntimeError("writer thread stalled"), None]
    store.insert_forum_message.side_effect = [RuntimeError("writer thread stalled"), None]
    persist._store = store

    with caplog.at_level(logging.WARNING):
        persist.on_task_trace(_make_trace())
        persist.on_forum_message(**_forum_kwargs())
        assert persist.on_run_end(token_summary=MagicMock()) is None

    assert not [rec for rec in caplog.records if "sidecar writes" in rec.message]
    # Explicit counter pin (#788 review nit): retried-then-succeeded must
    # leave every drop counter at zero, not merely keep the summary silent.
    assert persist._dropped_task_traces == 0
    assert persist._dropped_forum_messages == 0
    assert persist._dropped_task_statuses == 0


class _CountingLock:
    """Context-manager spy that wraps a real ``threading.Lock`` and counts

    every ``with lock:`` acquisition. Substituted for ``SqlitePersistence.
    _dropped_counts_lock`` so the test below can deterministically prove
    each drop-counter increment goes through the lock, rather than trying
    to time a lost-update race directly: empirically, plain ``int += 1`` on
    an instance attribute essentially never loses updates under CPython's
    GIL in this environment even at millions of unguarded concurrent
    increments (verified by hand while writing this test — flipping
    ``sys.setswitchinterval`` to 1e-6 and hammering 16 threads x 200k
    iterations each still landed on the exact expected count). A timing
    race is not a reliable regression signal here; call-site instrumentation is.
    (The lock's correctness value is realest on free-threaded — no-GIL —
    builds, where unguarded ``+= 1`` genuinely loses updates.)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.acquire_count = 0

    def __enter__(self) -> "_CountingLock":
        self._lock.acquire()
        self.acquire_count += 1  # only ever mutated while self._lock is held
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._lock.release()


def test_dropped_counters_are_lock_guarded_at_every_increment_site():
    """Issue #981: on_task_trace/on_forum_message/on_task_status are invoked

    from concurrent eval-worker threads (execution_phase.py's eval
    ThreadPoolExecutor); an unguarded ``self._dropped_* += 1`` can lose
    increments under real thread interleaving. Substitute a spy lock and
    assert it's acquired exactly once per drop event. The ``RuntimeError``
    side-effects here route through the retry-then-drop paths only, so this
    test covers three of the five increment sites (on_task_status, and the
    after-retry drop branches of on_task_trace / on_forum_message); the two
    ``except WriteIndeterminateError`` no-retry sites are covered by
    ``test_dropped_counters_are_lock_guarded_at_indeterminate_write_sites``
    below. This fails deterministically if a future edit removes the
    ``with self._dropped_counts_lock:`` guard from any exercised site,
    unlike a timing-based race test (see ``_CountingLock`` above).
    """
    persist = SqlitePersistence(runtime_db_path="/nonexistent/unused.sqlite", experiment_name="exp")
    store = MagicMock()
    store.insert_task_trace.side_effect = RuntimeError("writer thread stalled")
    store.insert_forum_message.side_effect = RuntimeError("writer thread stalled")
    store.mark_assignment_started.side_effect = RuntimeError("writer thread stalled")
    store.mark_assignment_ended.side_effect = RuntimeError("writer thread stalled")
    persist._store = store
    spy_lock = _CountingLock()
    persist._dropped_counts_lock = spy_lock

    n_per_kind = 20
    for i in range(n_per_kind):
        persist.on_task_trace(_make_trace())
        persist.on_forum_message(**_forum_kwargs())
        persist.on_task_status(generation=1, agent_id=f"agent-{i}", task_id=f"task-{i}", status="started")

    assert persist._dropped_task_traces == n_per_kind
    assert persist._dropped_forum_messages == n_per_kind
    assert persist._dropped_task_statuses == n_per_kind
    # One acquisition per drop event, no more and no fewer: 3 counters x
    # n_per_kind drops each. (mark_assignment_ended also fails inside
    # on_task_trace's ended_at back-stop, but that failure is only logged,
    # never counted — confirming the spy count isn't inflated by it.)
    assert spy_lock.acquire_count == 3 * n_per_kind


def test_dropped_counters_are_lock_guarded_at_indeterminate_write_sites():
    """Deep review 2026-07-03 (PR #1092 M2): the spy-lock test above only
    exercises the retry-then-drop paths; the two ``except
    WriteIndeterminateError`` no-retry drop sites (on_task_trace /
    on_forum_message "not retrying" branches) were false-green — stripping
    their guard left the file green. Drive those two sites with a
    ``WriteIndeterminateError`` side-effect and assert the spy lock is
    acquired exactly once per drop event, so removing either site's
    ``with self._dropped_counts_lock:`` guard fails deterministically.
    """
    persist = SqlitePersistence(runtime_db_path="/nonexistent/unused.sqlite", experiment_name="exp")
    store = MagicMock()
    store.insert_task_trace.side_effect = WriteIndeterminateError("write may still be applied")
    store.insert_forum_message.side_effect = WriteIndeterminateError("write may still be applied")
    persist._store = store
    spy_lock = _CountingLock()
    persist._dropped_counts_lock = spy_lock

    n_per_kind = 20
    for _ in range(n_per_kind):
        persist.on_task_trace(_make_trace())
        persist.on_forum_message(**_forum_kwargs())

    assert persist._dropped_task_traces == n_per_kind
    assert persist._dropped_forum_messages == n_per_kind
    assert persist._dropped_task_statuses == 0
    # One acquisition per drop event: 2 counters x n_per_kind drops each.
    # (on_task_trace's ended_at back-stop succeeds here — the MagicMock's
    # mark_assignment_ended has no side-effect — so it cannot inflate the
    # count either way; only the two indeterminate-write sites acquire.)
    assert spy_lock.acquire_count == 2 * n_per_kind


def test_dropped_counters_survive_concurrent_writers_without_corruption():
    """Integration-level smoke test alongside the deterministic lock-guard

    check above: drive every drop path from a large number of real threads
    hitting a persistently-failing store and confirm the run completes
    without hanging or raising, and the final counts still land on the
    exact expected totals under real concurrent load.
    """
    persist = SqlitePersistence(runtime_db_path="/nonexistent/unused.sqlite", experiment_name="exp")
    store = MagicMock()
    store.insert_task_trace.side_effect = RuntimeError("writer thread stalled")
    store.insert_forum_message.side_effect = RuntimeError("writer thread stalled")
    store.mark_assignment_started.side_effect = RuntimeError("writer thread stalled")
    store.mark_assignment_ended.side_effect = RuntimeError("writer thread stalled")
    persist._store = store

    n_per_kind = 100
    barrier = threading.Barrier(3 * n_per_kind)

    def _trace_writer(i: int) -> None:
        barrier.wait()
        persist.on_task_trace(_make_trace())

    def _forum_writer(i: int) -> None:
        barrier.wait()
        persist.on_forum_message(**_forum_kwargs())

    def _status_writer(i: int) -> None:
        barrier.wait()
        persist.on_task_status(generation=1, agent_id=f"agent-{i}", task_id=f"task-{i}", status="started")

    threads = (
        [threading.Thread(target=_trace_writer, args=(i,)) for i in range(n_per_kind)]
        + [threading.Thread(target=_forum_writer, args=(i,)) for i in range(n_per_kind)]
        + [threading.Thread(target=_status_writer, args=(i,)) for i in range(n_per_kind)]
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)
    assert not any(t.is_alive() for t in threads), "writer threads did not finish within 30s"

    assert persist._dropped_task_traces == n_per_kind
    assert persist._dropped_forum_messages == n_per_kind
    assert persist._dropped_task_statuses == n_per_kind


def test_composite_on_forum_message_failing_observer_does_not_skip_others(caplog):
    failing = MagicMock()
    failing.on_forum_message.side_effect = RuntimeError("sidecar DB unavailable")
    healthy = MagicMock()
    composite = CompositePersistence(observers=[failing, healthy])

    with caplog.at_level(logging.WARNING):
        composite.on_forum_message(**_forum_kwargs())

    healthy.on_forum_message.assert_called_once_with(**_forum_kwargs())
    warning = next(rec for rec in caplog.records if "failed on_forum_message" in rec.message)
    rendered = warning.getMessage()
    assert "agent-1" in rendered
    assert "gen 1" in rendered
    assert "round 2" in rendered
    assert "sidecar DB unavailable" in rendered


def test_composite_on_forum_message_reraises_authentication_failure():
    failing = MagicMock()
    failing.on_forum_message.side_effect = AuthenticationFailure("invalid api key")
    healthy = MagicMock()
    composite = CompositePersistence(observers=[failing, healthy])

    with pytest.raises(AuthenticationFailure):
        composite.on_forum_message(**_forum_kwargs())


def test_composite_on_task_status_failing_observer_does_not_skip_others(caplog):
    failing = MagicMock()
    failing.on_task_status.side_effect = RuntimeError("sidecar DB unavailable")
    healthy = MagicMock()
    composite = CompositePersistence(observers=[failing, healthy])

    with caplog.at_level(logging.WARNING):
        composite.on_task_status(generation=1, agent_id="agent-1", task_id="task-1", status="started")

    healthy.on_task_status.assert_called_once_with(generation=1, agent_id="agent-1", task_id="task-1", status="started")
    warning = next(rec for rec in caplog.records if "failed on_task_status" in rec.message)
    rendered = warning.getMessage()
    assert "agent-1" in rendered
    assert "task-1" in rendered
    assert "sidecar DB unavailable" in rendered


def test_composite_on_task_status_reraises_authentication_failure():
    failing = MagicMock()
    failing.on_task_status.side_effect = AuthenticationFailure("invalid api key")
    healthy = MagicMock()
    composite = CompositePersistence(observers=[failing, healthy])

    with pytest.raises(AuthenticationFailure):
        composite.on_task_status(generation=1, agent_id="agent-1", task_id="task-1", status="started")


def test_raising_on_task_trace_observer_does_not_abort_run():
    """End-to-end: a persistence observer whose on_task_trace raises must not
    abort the generation — the run completes and other observers still fire."""
    config = GenerationConfig(num_generations=1, num_agents=1, per_task_forum_rounds=0)
    config.cross_task_forum_rounds = 0
    config.distill_enabled = False
    tasks = _build_make_tasks(1)

    failing = MagicMock()
    failing.on_task_trace.side_effect = RuntimeError("sidecar DB write failed")
    healthy = MagicMock()
    persistence = CompositePersistence(observers=[failing, healthy])

    orch = GenerationalOrchestrator(
        config=config,
        runtime=_build_mock_runtime(),
        evaluator=_build_mock_evaluator(),
        llm=_build_mock_llm(),
        persistence=persistence,
    )
    traces = orch.run(tasks)

    assert len(traces) >= 1
    assert failing.on_task_trace.called
    assert healthy.on_task_trace.called
    assert healthy.on_run_end.called


def _make_forum_traces(generation: int, agent_id: str, task_ids: list[str]) -> list[TaskTrace]:
    return [
        TaskTrace(
            generation=generation,
            agent_id=agent_id,
            task_id=tid,
            model_output="patch",
            eval_result={"resolved": True},
            native_score=1.0,
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        for tid in task_ids
    ]


def test_raising_on_forum_message_observer_does_not_abort_forum_phase(tmp_path, caplog):
    """End-to-end: the engine invokes ``persistence.on_forum_message`` for forum
    ERROR events (``message_type="error"``, runtime_meta ``forum_error``); a
    raising observer must not abort the forum phase. One of two agents fails
    terminally (emitting the error event), the failing observer raises on it,
    and the phase still completes with the healthy observer recording the
    event — the forum error record is not lost."""
    db_path = str(tmp_path / "knowledge.sqlite")
    runtime = MagicMock()

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        if agent_id == "agent-0":
            raise RuntimeError("provider unavailable")
        return RuntimeResult(
            output="discussed via MCP tools",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=50, output_tokens=30),
        )

    runtime.run_task.side_effect = fake_run_task
    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0}
    llm = MagicMock()
    llm.call.return_value = LLMResponse(text=json.dumps({}), usage=TokenUsage())

    failing = MagicMock()
    failing.on_forum_message.side_effect = RuntimeError("sidecar DB write failed")
    healthy = MagicMock()
    persistence = CompositePersistence(observers=[failing, healthy])

    config = GenerationConfig(
        num_generations=1,
        num_agents=2,
        per_task_forum_rounds=1,
        knowledge_db_path=db_path,
        max_task_retries=0,  # agent-0's failure is terminal on the first attempt
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=persistence,
    )

    # Two agents, same task so the forum phase dispatches both agents; only
    # agent-0 fails, so the all-agents-failed ForumValidationError is not hit.
    traces = _make_forum_traces(1, "agent-0", ["task-0"]) + _make_forum_traces(1, "agent-1", ["task-0"])
    with caplog.at_level(logging.WARNING):
        per_task_forum(orch, 1, traces)

    # The engine emitted the forum ERROR event; the raising observer saw it...
    failing_error_calls = [
        call for call in failing.on_forum_message.call_args_list if call.kwargs.get("message_type") == "error"
    ]
    assert len(failing_error_calls) == 1
    # ...and despite it raising, the healthy observer still recorded it.
    healthy_error_calls = [
        call for call in healthy.on_forum_message.call_args_list if call.kwargs.get("message_type") == "error"
    ]
    assert len(healthy_error_calls) == 1
    assert healthy_error_calls[0].kwargs["agent_id"] == "agent-0"
    assert healthy_error_calls[0].kwargs["content_json"]["phase"] == "per_task_forum"
    warning = next(rec for rec in caplog.records if "failed on_forum_message" in rec.message)
    assert "sidecar DB write failed" in warning.getMessage()


# ---------------------------------------------------------------------------
# share_runtime_store: single-owner adoption of the engine's runtime store
# (deep-review abstractions.md H2/H3; root cause of the PR #368 AB-BA deadlock).
# ---------------------------------------------------------------------------


def test_share_runtime_store_adopts_engine_store_no_second_instance(tmp_path):
    """The shared engine store is adopted; _ensure_store returns the SAME
    object — no second MemoryStore is opened on the runtime DB."""
    db = tmp_path / "runtime.sqlite"
    store = MemoryStore(str(db), default_experiment="exp")
    try:
        persist = SqlitePersistence(runtime_db_path=str(db), experiment_name="exp")
        # No MemoryStore must be constructed during the share path.
        with patch("ksi.memory.store.MemoryStore", side_effect=AssertionError("should not construct")):
            persist.share_runtime_store(store)
            assert persist._store is store
            assert persist._borrowed_store is True
            # _ensure_store returns the borrowed store, never a new instance.
            assert persist._ensure_store() is store
    finally:
        store.close()


def test_share_runtime_store_sets_default_experiment(tmp_path):
    """Adopting the shared store retargets its default experiment, matching
    what the lazy-open path did via the MemoryStore constructor."""
    db = tmp_path / "runtime.sqlite"
    store = MemoryStore(str(db), default_experiment="old")
    try:
        persist = SqlitePersistence(runtime_db_path=str(db), experiment_name="new-exp")
        persist.share_runtime_store(store)
        assert store._default_experiment == "new-exp"
    finally:
        store.close()


def test_close_does_not_close_borrowed_store(tmp_path):
    """close() must NOT close a borrowed store — the engine owns and closes
    it. It only drops the reference and clears the borrowed flag."""
    db = tmp_path / "runtime.sqlite"
    store = MemoryStore(str(db), default_experiment="exp")
    try:
        persist = SqlitePersistence(runtime_db_path=str(db), experiment_name="exp")
        persist.share_runtime_store(store)
        with patch.object(store, "close", wraps=store.close) as spy_close:
            persist.close()
            spy_close.assert_not_called()
        assert persist._store is None
        assert persist._borrowed_store is False
        # The store is still open and usable (engine closes it later).
        store.set_default_experiment("still-alive")
        assert store._default_experiment == "still-alive"
    finally:
        store.close()


def test_close_still_closes_self_opened_store():
    """A self-opened (non-borrowed) store is still closed by close()."""
    persist = SqlitePersistence(runtime_db_path="/nonexistent/unused.sqlite", experiment_name="exp")
    store = MagicMock()
    persist._store = store  # simulate lazy self-open
    persist.close()
    store.close.assert_called_once()
    assert persist._store is None


def test_share_runtime_store_path_mismatch_not_adopted(tmp_path, caplog):
    """A store pointing at a different DB file is NOT adopted; lazy-open
    behavior is preserved (_store stays None)."""
    other_db = tmp_path / "other.sqlite"
    store = MemoryStore(str(other_db), default_experiment="exp")
    try:
        persist = SqlitePersistence(runtime_db_path=str(tmp_path / "runtime.sqlite"), experiment_name="exp")
        with caplog.at_level(logging.WARNING):
            persist.share_runtime_store(store)
        assert persist._store is None
        assert persist._borrowed_store is False
        assert any("path mismatch" in rec.getMessage() for rec in caplog.records)
    finally:
        store.close()


def test_share_runtime_store_ignored_when_store_already_open(tmp_path, caplog):
    """If a store is already set (self-opened), sharing is a no-op — no swap,
    no double-instance leak."""
    db = tmp_path / "runtime.sqlite"
    existing = MagicMock()
    shared = MemoryStore(str(db), default_experiment="exp")
    try:
        persist = SqlitePersistence(runtime_db_path=str(db), experiment_name="exp")
        persist._store = existing
        with caplog.at_level(logging.DEBUG):
            persist.share_runtime_store(shared)
        assert persist._store is existing
        assert persist._borrowed_store is False
    finally:
        shared.close()


def test_share_runtime_store_replaces_a_borrowed_store(tmp_path):
    """A previously *borrowed* store must be replaceable by a fresh share.

    On programmatic reuse of one persistence across two engine runs, engine #1
    closes the store it lent; engine #2 then shares its own. The observer must
    adopt the new store rather than keep pointing at the closed one.
    """
    db = tmp_path / "runtime.sqlite"
    first = MemoryStore(str(db), default_experiment="exp")
    second = MemoryStore(str(db), default_experiment="exp")
    try:
        persist = SqlitePersistence(runtime_db_path=str(db), experiment_name="exp")
        persist.share_runtime_store(first)
        assert persist._store is first
        assert persist._borrowed_store is True
        # Engine #1 closes its store; engine #2 shares a fresh one.
        first.close()
        persist.share_runtime_store(second)
        assert persist._store is second, "a borrowed store must be replaced, not kept"
        assert persist._borrowed_store is True
    finally:
        with contextlib.suppress(Exception):
            first.close()
        second.close()


def test_composite_share_runtime_store_fans_out(tmp_path):
    """CompositePersistence forwards share_runtime_store to wrapped observers
    that implement it, and skips those that don't."""
    db = tmp_path / "runtime.sqlite"
    store = MemoryStore(str(db), default_experiment="exp")
    try:
        sqlite_obs = SqlitePersistence(runtime_db_path=str(db), experiment_name="exp")

        class _NoStoreObserver:
            pass

        composite = CompositePersistence([sqlite_obs, _NoStoreObserver()])
        composite.share_runtime_store(store)  # must not raise on the no-store observer
        assert sqlite_obs._store is store
        assert sqlite_obs._borrowed_store is True
    finally:
        store.close()

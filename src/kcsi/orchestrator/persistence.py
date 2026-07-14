"""Persistence observer classes for the generational orchestrator.

These observers implement the persistence callback contract consumed by
:class:`~kcsi.orchestrator.engine.GenerationalOrchestrator`:

- :class:`CollectingPersistence` -- in-memory progress logger / trace collector.
- :class:`CompositePersistence` -- best-effort fan-out to multiple observers.
- :class:`SqlitePersistence` -- primary SQLite audit sidecar writer.

Extracted verbatim from ``kcsi.cli`` to shrink ``cli.py``.
``kcsi.cli`` re-imports these names, so existing call sites and
``from kcsi.cli import SqlitePersistence`` continue to work.

This module imports only from ``..errors``, ``..models``, ``..tokens`` and
(lazily) ``..memory.store`` -- never from ``engine`` or ``cli`` -- so there is
no import cycle.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, cast

from ..errors import AuthenticationFailure, WriteIndeterminateError
from ..models import AgentState, Insight, TaskTrace

if TYPE_CHECKING:
    from ..memory.store import MemoryStore
    from ..protocols import ForumMessageContent, PersistenceObserver
    from ..tokens import TokenUsage, TokenUsageDict

log = logging.getLogger(__name__)


@dataclass
class CollectingPersistence:
    """In-memory observer that prints progress and collects traces."""

    assignments: list[dict[str, object]] = field(default_factory=list)
    generation_end: list[dict[str, object]] = field(default_factory=list)
    token_summary: TokenUsageDict | None = None
    traces: list[TaskTrace] = field(default_factory=list)
    # Best-effort incremental --output-json snapshot hook: fired
    # at the end of each generation with the traces accumulated so far, so a
    # mid-run crash doesn't lose everything the final post-run() write would
    # have captured. A failing callback logs a warning and never aborts the
    # run -- see the try/except around its call in on_generation_end.
    on_generation_snapshot: Callable[[int, list[TaskTrace]], None] | None = None

    def on_generation_start(self, *, generation: int, agents: list[AgentState]) -> None:
        log.info("[gen %s] start agents=%d", generation, len(agents))

    def on_assignment(self, *, generation: int, assigned: dict[str, list[str]], total_tasks: int = 0) -> None:
        counts = {agent: len(tasks) for agent, tasks in assigned.items()}
        self.assignments.append({"generation": generation, "counts": counts, "total_tasks": total_tasks})
        log.info("[gen %s] assignment=%s total_tasks=%d", generation, counts, total_tasks)

    def on_task_trace(self, trace: TaskTrace) -> None:
        self.traces.append(trace)

    def on_task_status(self, *, generation: int, agent_id: str, task_id: str, status: str) -> None:
        return None

    def on_forum_message(
        self,
        *,
        generation: int,
        round_num: int,
        agent_id: str,
        message_type: str,
        content_json: ForumMessageContent,
        token_usage: TokenUsageDict,
    ) -> None:
        return None

    def on_native_memory(self, *, generation: int, agent_id: str, content: str) -> None:
        return None

    def on_insight(self, *, generation: int, agent_id: str, insight: Insight) -> None:
        return None

    def on_generation_end(self, *, generation: int, agents: list[AgentState]) -> None:
        workstreams = {a.id: a.workstream for a in agents}
        payload = {"generation": generation, "workstreams": workstreams}
        self.generation_end.append(payload)
        log.info("[gen %s] end workstreams=%s", generation, workstreams)
        if self.on_generation_snapshot is not None:
            try:
                self.on_generation_snapshot(generation, list(self.traces))
            except Exception:
                log.warning("[gen %s] incremental output-json snapshot failed", generation, exc_info=True)

    def on_run_end(self, *, token_summary: "TokenUsage") -> None:
        self.token_summary = token_summary.to_dict()
        log.info(
            "[tokens] total=%s cached_input=%s uncached_input=%s output=%s cache_create=%s",
            f"{token_summary.total:,}",
            f"{token_summary.cache_read_input_tokens:,}",
            f"{token_summary.uncached_input_tokens:,}",
            f"{token_summary.output_tokens:,}",
            f"{token_summary.cache_creation_input_tokens:,}",
        )


@dataclass
class CompositePersistence:
    """Fan-out observer that broadcasts events to multiple observers.

    Best-effort semantics: the callbacks that perform sidecar writes —
    ``on_task_trace``, ``on_forum_message``, and ``on_task_status`` — are
    guarded per observer: a failing observer logs a warning and the
    remaining observers still fire. The one exception
    is ``AuthenticationFailure``, which always propagates so the run
    aborts loudly. Callbacks whose known implementations cannot fail
    (no-op/collecting) are deliberately left unguarded.
    """

    observers: list[PersistenceObserver]
    _experiment_name: str = ""

    def __post_init__(self) -> None:
        if self.observers is None:
            self.observers = []

    @property
    def experiment_name(self) -> str:
        if self._experiment_name:
            return self._experiment_name
        for observer in self.observers:
            value = getattr(observer, "experiment_name", None)
            if isinstance(value, str) and value:
                return value
        return ""

    @experiment_name.setter
    def experiment_name(self, value: str) -> None:
        self.set_experiment_name(value)

    def set_experiment_name(self, value: str) -> None:
        """Propagate a late experiment-name change across wrapped observers."""
        self._experiment_name = value
        for o in self.observers:
            setter = getattr(o, "set_experiment_name", None)
            if callable(setter):
                setter(value)
            elif hasattr(o, "experiment_name"):
                try:
                    setattr(o, "experiment_name", value)
                except Exception:
                    log.warning(
                        "CompositePersistence could not update experiment_name on %s",
                        type(o).__name__,
                    )

    def share_runtime_store(self, store: MemoryStore) -> None:
        """Forward the engine's shared runtime store to wrapped observers.

        hasattr-guarded fan-out (mirrors ``set_experiment_name``): only
        observers that implement ``share_runtime_store`` (i.e.
        ``SqlitePersistence``) adopt the store; the rest are skipped.
        """
        for o in self.observers:
            fn = getattr(o, "share_runtime_store", None)
            if callable(fn):
                fn(store)

    def on_generation_start(self, *, generation: int, agents: list[AgentState]) -> None:
        for o in self.observers:
            fn = getattr(o, "on_generation_start", None)
            if callable(fn):
                fn(generation=generation, agents=agents)

    def on_assignment(self, *, generation: int, assigned: dict[str, list[str]], total_tasks: int = 0) -> None:
        for o in self.observers:
            fn = getattr(o, "on_assignment", None)
            if callable(fn):
                fn(generation=generation, assigned=assigned, total_tasks=total_tasks)

    def on_task_trace(self, trace: TaskTrace) -> None:
        for o in self.observers:
            fn = getattr(o, "on_task_trace", None)
            if not callable(fn):
                continue
            try:
                fn(trace)
            except AuthenticationFailure:
                # Auth failures are fatal — the engine deliberately keeps
                # them so the run aborts loudly; never swallow them here.
                raise
            except Exception as exc:
                log.warning(
                    "CompositePersistence observer %s failed on_task_trace for %s/%s (gen %s): %s",
                    type(o).__name__,
                    trace.agent_id,
                    trace.task_id,
                    trace.generation,
                    exc,
                )

    def on_task_status(self, *, generation: int, agent_id: str, task_id: str, status: str) -> None:
        for o in self.observers:
            fn = getattr(o, "on_task_status", None)
            if not callable(fn):
                continue
            try:
                fn(generation=generation, agent_id=agent_id, task_id=task_id, status=status)
            except AuthenticationFailure:
                # Auth failures are fatal — the engine deliberately keeps
                # them so the run aborts loudly; never swallow them here.
                raise
            except Exception as exc:
                log.warning(
                    "CompositePersistence observer %s failed on_task_status for %s/%s (gen %s): %s",
                    type(o).__name__,
                    agent_id,
                    task_id,
                    generation,
                    exc,
                )

    def on_forum_message(
        self,
        *,
        generation: int,
        round_num: int,
        agent_id: str,
        message_type: str,
        content_json: ForumMessageContent,
        token_usage: TokenUsageDict,
    ) -> None:
        for o in self.observers:
            fn = getattr(o, "on_forum_message", None)
            if not callable(fn):
                continue
            try:
                fn(
                    generation=generation,
                    round_num=round_num,
                    agent_id=agent_id,
                    message_type=message_type,
                    content_json=content_json,
                    token_usage=token_usage,
                )
            except AuthenticationFailure:
                # Auth failures are fatal — the engine deliberately keeps
                # them so the run aborts loudly; never swallow them here.
                raise
            except Exception as exc:
                log.warning(
                    "CompositePersistence observer %s failed on_forum_message for %s (gen %s round %s): %s",
                    type(o).__name__,
                    agent_id,
                    generation,
                    round_num,
                    exc,
                )

    def on_native_memory(self, *, generation: int, agent_id: str, content: str) -> None:
        for o in self.observers:
            fn = getattr(o, "on_native_memory", None)
            if callable(fn):
                fn(generation=generation, agent_id=agent_id, content=content)

    def on_insight(self, *, generation: int, agent_id: str, insight: Insight) -> None:
        for o in self.observers:
            fn = getattr(o, "on_insight", None)
            if callable(fn):
                fn(generation=generation, agent_id=agent_id, insight=insight)

    def on_generation_end(self, *, generation: int, agents: list[AgentState]) -> None:
        for o in self.observers:
            fn = getattr(o, "on_generation_end", None)
            if callable(fn):
                fn(generation=generation, agents=agents)

    def on_run_end(self, *, token_summary: "TokenUsage") -> None:
        for o in self.observers:
            fn = getattr(o, "on_run_end", None)
            if callable(fn):
                fn(token_summary=token_summary)

    def close(self) -> None:
        for o in self.observers:
            fn = getattr(o, "close", None)
            if callable(fn):
                fn()


@dataclass
class SqlitePersistence:
    """Primary SQLite observer for evaluated task/forum persistence."""

    runtime_db_path: str
    experiment_name: str
    _store: MemoryStore | None = field(default=None, init=False, repr=False)
    # True when ``_store`` was adopted from the engine via
    # ``share_runtime_store`` (single-owner). A borrowed store is
    # owned and closed by the engine, so ``close()`` must NOT close it.
    _borrowed_store: bool = field(default=False, init=False, repr=False)
    # Dropped sidecar writes per callback family: incremented on
    # the terminal guard arms only (after-retry drops, indeterminate-write
    # drops, single-attempt assignment-guard failures), never on
    # retried-then-succeeded writes. Summarized once in on_run_end.
    _dropped_task_traces: int = field(default=0, init=False, repr=False)
    _dropped_forum_messages: int = field(default=0, init=False, repr=False)
    _dropped_task_statuses: int = field(default=0, init=False, repr=False)
    # Guards the three counters above: on_task_status/on_task_trace/
    # on_forum_message are invoked from concurrent eval-worker threads
    # (execution_phase.py's eval ThreadPoolExecutor), and an unguarded
    # `+= 1` loses increments under concurrency.
    _dropped_counts_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def set_experiment_name(self, value: str) -> None:
        self.experiment_name = value
        if self._store is not None:
            self._store.set_default_experiment(value)

    def share_runtime_store(self, store: MemoryStore) -> None:
        """Adopt the engine's runtime ``MemoryStore`` as the single owner.

        Having the engine and this observer each open their own
        ``MemoryStore`` on the same DB file is the root cause of the
        AB-BA deadlock. The engine calls this during run-setup (before any
        ``on_task_status`` fires) so we can reuse its store instead of lazily
        opening a second one.

        Conservative guards (behavior-preserving on mismatch):
        - PATH-GUARD: only adopt when ``store`` points at the same DB file as
          ``self.runtime_db_path``; otherwise keep the lazy-open path.
        - ADOPT unless we have already SELF-opened a store. A previously
          *borrowed* store may belong to an engine that has since closed it
          (programmatic reuse of one persistence across two engine runs), so a
          fresh share must be allowed to replace it — otherwise the observer
          keeps pointing at a closed store. A self-opened store is left intact.
        """
        if self._store is not None and not self._borrowed_store:
            log.debug(
                "[SqlitePersistence] share_runtime_store ignored: a self-opened store is already in use",
            )
            return
        try:
            shared_path = os.path.realpath(str(store.db_path))
            own_path = os.path.realpath(str(self.runtime_db_path))
        except Exception as exc:
            log.warning(
                "[SqlitePersistence] could not resolve store paths for sharing; keeping lazy-open: %s",
                exc,
            )
            return
        if shared_path != own_path:
            log.warning(
                "[SqlitePersistence] not adopting shared store: path mismatch (%s != %s)",
                shared_path,
                own_path,
            )
            return
        self._store = store
        self._borrowed_store = True
        store.set_default_experiment(self.experiment_name)

    def _ensure_store(self) -> MemoryStore:
        if self._store is not None:
            return self._store
        from ..memory.store import MemoryStore

        self._store = MemoryStore(self.runtime_db_path, default_experiment=self.experiment_name)
        return self._store

    def on_generation_start(self, *, generation: int, agents: list[AgentState]) -> None:
        return None

    def on_assignment(self, *, generation: int, assigned: dict[str, list[str]], total_tasks: int = 0) -> None:
        return None

    def on_task_status(self, *, generation: int, agent_id: str, task_id: str, status: str) -> None:
        # Populate assignments.started_at / ended_at so per-attempt wall-clock
        # durations can be reconstructed without scraping tool_trace timestamps.
        # Status values emitted by the engine:
        #   'started'  -> first dispatch of the assignment
        #   'retrying' -> transient failure, about to retry (keep existing started_at)
        #   'completed'-> eval succeeded (terminal)
        #   'failed'   -> eval failed or runtime error (terminal)
        if status not in {"started", "retrying", "completed", "failed"}:
            return None
        try:
            store = self._ensure_store()
            if status in {"started", "retrying"}:
                store.mark_assignment_started(
                    experiment=self.experiment_name,
                    generation=generation,
                    agent_id=agent_id,
                    task_id=task_id,
                )
            else:
                store.mark_assignment_ended(
                    experiment=self.experiment_name,
                    generation=generation,
                    agent_id=agent_id,
                    task_id=task_id,
                    status=status,
                )
        except Exception as exc:
            with self._dropped_counts_lock:
                self._dropped_task_statuses += 1
            log.warning(
                "[SqlitePersistence] failed to record assignment %s timestamp for %s/%s: %s",
                status,
                agent_id,
                task_id,
                exc,
            )

    def on_task_trace(self, trace: TaskTrace) -> None:
        runtime_meta = dict(trace.runtime_meta or {})
        runtime_meta["token_usage"] = trace.token_usage.to_dict()
        # The runtime DB is a non-authoritative audit sidecar: a failed write
        # here must never propagate and abort the run.
        #
        # Best-effort semantics (deliberate): failures are retried once here
        # and then dropped — the engine's late-persist path will NOT retry,
        # because this call returning normally sets
        # _task_trace_persisted_early. No sleep between attempts: the store
        # layer already retries "database is locked" internally, so an
        # immediate retry here covers whatever escapes it.
        for attempt in (1, 2):
            try:
                store = self._ensure_store()
                # Carry the TaskSpec.repo through so tasks.repo is populated even on
                # silent-failure traces where downstream insert_task_summary
                # (the other path that writes repo) is skipped.
                store.insert_task_trace(
                    experiment=self.experiment_name,
                    generation=trace.generation,
                    agent_id=trace.agent_id,
                    task_id=trace.task_id,
                    repo=trace.repo or None,
                    model_output=trace.model_output,
                    # ``trace.eval_result`` is the ``EvalResult`` TypedDict, which
                    # mypy does not treat as assignable to the store's
                    # ``dict[str, Any]`` param (TypedDict invariance). The store
                    # only reads/serializes it, so the cast is a no-op at runtime.
                    eval_result=cast("dict[str, Any]", trace.eval_result) or {},
                    native_score=trace.native_score,
                    tool_trace=trace.tool_trace or [],
                    runtime_meta=runtime_meta,
                    error_text=trace.error,
                )
                break
            except WriteIndeterminateError as exc:
                # The store could not cancel the timed-out write — it may
                # still apply later. Retrying would risk exactly the
                # duplicate rows, so drop instead.
                with self._dropped_counts_lock:
                    self._dropped_task_traces += 1
                log.warning(
                    "[SqlitePersistence] failed to record task trace for %s/%s (gen %s); not retrying: %s",
                    trace.agent_id,
                    trace.task_id,
                    trace.generation,
                    exc,
                )
                break
            except Exception as exc:
                if attempt == 1:
                    log.warning(
                        "[SqlitePersistence] failed to record task trace for %s/%s (gen %s): %s; retrying once",
                        trace.agent_id,
                        trace.task_id,
                        trace.generation,
                        exc,
                    )
                else:
                    with self._dropped_counts_lock:
                        self._dropped_task_traces += 1
                    log.warning(
                        "[SqlitePersistence] failed to record task trace for %s/%s (gen %s) after retry; dropping: %s",
                        trace.agent_id,
                        trace.task_id,
                        trace.generation,
                        exc,
                    )
        # Back-stop: ensure ended_at is populated even if on_task_status was
        # never delivered (e.g., a third-party persistence composite that
        # short-circuits status events). This is idempotent with on_task_status.
        # Reached even when the trace insert above failed: the back-stop may
        # still succeed for a data-dependent insert failure, avoiding NULL
        # ended_at rows.
        try:
            status = "failed" if trace.error else "completed"
            store = self._ensure_store()
            store.mark_assignment_ended(
                experiment=self.experiment_name,
                generation=trace.generation,
                agent_id=trace.agent_id,
                task_id=trace.task_id,
                status=status,
            )
        except Exception as exc:
            log.warning(
                "[SqlitePersistence] failed to back-fill ended_at for %s/%s: %s",
                trace.agent_id,
                trace.task_id,
                exc,
            )

    def on_forum_message(
        self,
        *,
        generation: int,
        round_num: int,
        agent_id: str,
        message_type: str,
        content_json: ForumMessageContent,
        token_usage: TokenUsageDict,
    ) -> None:
        # The runtime DB is a non-authoritative audit sidecar: a failed write
        # here must never propagate and abort the forum phase.
        #
        # Best-effort semantics (deliberate): failures are retried once here
        # and then dropped. No sleep between attempts: the store layer already
        # retries "database is locked" internally, so an immediate retry here
        # covers whatever escapes it.
        for attempt in (1, 2):
            try:
                store = self._ensure_store()
                store.insert_forum_message(
                    generation=generation,
                    round_num=round_num,
                    agent_id=agent_id,
                    message_type=message_type,
                    content={**(content_json or {}), "token_usage": token_usage or {}},
                    experiment=self.experiment_name,
                )
                break
            except WriteIndeterminateError as exc:
                # The store could not cancel the timed-out write — it may
                # still apply later. Retrying would risk exactly the
                # duplicate rows, so drop instead.
                with self._dropped_counts_lock:
                    self._dropped_forum_messages += 1
                log.warning(
                    "[SqlitePersistence] failed to record forum message for %s (gen %s round %s); not retrying: %s",
                    agent_id,
                    generation,
                    round_num,
                    exc,
                )
                break
            except Exception as exc:
                if attempt == 1:
                    log.warning(
                        "[SqlitePersistence] failed to record forum message for %s (gen %s round %s): %s; retrying once",
                        agent_id,
                        generation,
                        round_num,
                        exc,
                    )
                else:
                    with self._dropped_counts_lock:
                        self._dropped_forum_messages += 1
                    log.warning(
                        "[SqlitePersistence] failed to record forum message for %s (gen %s round %s) after retry; dropping: %s",
                        agent_id,
                        generation,
                        round_num,
                        exc,
                    )

    def on_native_memory(self, *, generation: int, agent_id: str, content: str) -> None:
        return None

    def on_insight(self, *, generation: int, agent_id: str, insight: Insight) -> None:
        return None

    def on_generation_end(self, *, generation: int, agents: list[AgentState]) -> None:
        return None

    def on_run_end(self, *, token_summary: "TokenUsage") -> None:
        dropped = self._dropped_task_traces + self._dropped_forum_messages + self._dropped_task_statuses
        if dropped:
            log.warning(
                "[SqlitePersistence] dropped %d sidecar writes this run "
                "(%d task traces, %d forum messages, %d task statuses); the runtime DB is incomplete",
                dropped,
                self._dropped_task_traces,
                self._dropped_forum_messages,
                self._dropped_task_statuses,
            )
        return None

    def close(self) -> None:
        if self._borrowed_store:
            # The engine owns and closes the shared store (engine.py finally
            # block, AFTER on_run_end). Never double-close it here.
            self._store = None
            self._borrowed_store = False
            return
        if self._store is not None:
            try:
                self._store.close()
            except Exception:
                pass
            self._store = None

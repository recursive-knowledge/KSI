"""Carry-forward / resume phase-service boundary for the orchestrator.

On ``--resume`` (and any non-``--drop-solved`` run) a task that was already
solved at or above ``solved_threshold`` in an earlier generation is *replayed*
from its preserved trace instead of re-executed in a container. This module
owns that cluster behind an explicit service boundary used by the generation
loop, following the established phase-service idiom (per-call
``_collaborators`` factory; the moved bodies depend on ``collab``, never on the
engine directly).

The tiny score/preserve helpers (``is_carried_forward_trace``,
``trace_preserve_score``, ``trace_meets_preserve_threshold``,
``trace_preserve_rank``, ``carry_forward_payload``) are pure module-level
functions because they are shared between this resume cluster and the engine's
``run()`` / ``_update_score_tracking`` bookkeeping, which stay on the engine.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from ..models import TaskSpec, TaskTrace
from ..tokens import TokenUsage
from .attempt_events import _coerce_float, _coerce_int, _score_from_eval

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .engine import GenerationalOrchestrator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure shared helpers (used by BOTH this service and engine.run() /
# engine._update_score_tracking). Module-level so neither side keeps a copy.
# ---------------------------------------------------------------------------
def is_carried_forward_trace(trace: TaskTrace) -> bool:
    return bool((trace.runtime_meta or {}).get("carry_forward"))


def trace_preserve_score(trace: TaskTrace | None, *, task: TaskSpec | None = None) -> float | None:
    if trace is None:
        return None
    if trace.native_score is not None:
        try:
            return float(trace.native_score)
        except (TypeError, ValueError):
            return None
    eval_result = trace.eval_result if isinstance(trace.eval_result, dict) else {}
    return _score_from_eval(eval_result, task=task)


def trace_meets_preserve_threshold(
    trace: TaskTrace | None,
    *,
    task: TaskSpec | None = None,
    solved_threshold: float,
) -> bool:
    if trace is None or trace.error:
        return False
    score = trace_preserve_score(trace, task=task)
    if score is None:
        return False
    return score >= float(solved_threshold)


def trace_preserve_rank(
    trace: TaskTrace | None,
    *,
    task: TaskSpec | None = None,
) -> tuple[float, int, int]:
    score = trace_preserve_score(trace, task=task)
    real_execution = 0 if trace is not None and is_carried_forward_trace(trace) else 1
    generation = int(getattr(trace, "generation", 0) or 0) if trace is not None else 0
    return (float(score) if score is not None else float("-inf"), real_execution, generation)


def carry_forward_payload(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(meta, dict) or not meta.get("carry_forward"):
        return None
    payload: dict[str, Any] = {
        "carry_forward": True,
        "carry_forward_reason": str(meta.get("carry_forward_reason") or "best_score_preserved"),
    }
    source_generation = _coerce_int(meta.get("carry_forward_source_generation"))
    if source_generation is not None:
        payload["carry_forward_source_generation"] = source_generation
    source_agent_id = str(meta.get("carry_forward_source_agent_id") or "").strip()
    if source_agent_id:
        payload["carry_forward_source_agent_id"] = source_agent_id
    source_score = _coerce_float(meta.get("carry_forward_source_score"))
    if source_score is not None:
        payload["carry_forward_source_score"] = source_score
    threshold = _coerce_float(meta.get("carry_forward_threshold"))
    if threshold is not None:
        payload["carry_forward_threshold"] = threshold
    return payload


def _history_runtime_meta(
    *,
    history_source: str,
    carry_forward_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    runtime_meta: dict[str, Any] = {"history_source": history_source}
    if carry_forward_payload is None:
        return runtime_meta
    runtime_meta.update(carry_forward_payload)
    runtime_meta["status"] = "carry_forward"
    return runtime_meta


def _memory_record_carry_forward_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    history = record.get("attempt_history")
    if not isinstance(history, list):
        return None
    for item in reversed(history):
        payload = carry_forward_payload(item if isinstance(item, dict) else None)
        if payload is not None:
            return payload
    return None


@dataclass(frozen=True)
class ResumeCollaborators:
    """Explicit dependencies for the carry-forward / resume service body."""

    config: Any
    knowledge: Any
    memory_store: Any
    best_preserved_traces: dict[str, TaskTrace]  # live dict ref (mutated in place)
    persistence: Any
    accumulator: Any
    is_holdout: Callable[[str], bool]
    safe_on_task_trace: Callable[[TaskTrace], None]
    # Shared persisters kept on the engine (also used by the eval pipeline);
    # the carried-trace body reaches them only through this object.
    persist_task_memory_record: Callable[..., None]
    persist_task_summary: Callable[..., None]


@dataclass
class EngineResumePhaseService:
    """Engine-backed carry-forward / resume phase adapter.

    The carry-forward split + preserved-trace lookup + carried-trace
    persistence bodies live here behind an explicit service boundary used by
    the generation loop and tests.
    """

    engine: "GenerationalOrchestrator"

    def _collaborators(self) -> ResumeCollaborators:
        engine = self.engine
        return ResumeCollaborators(
            config=engine.config,
            knowledge=engine._knowledge,
            memory_store=engine._memory_store,
            best_preserved_traces=engine._best_preserved_traces,
            persistence=engine.persistence,
            accumulator=engine.accumulator,
            is_holdout=engine._is_holdout,
            safe_on_task_trace=engine._safe_on_task_trace,
            persist_task_memory_record=engine._persist_task_memory_record,
            persist_task_summary=engine._persist_task_summary,
        )

    # ------------------------------------------------------------------
    # Preserved-trace reconstruction from history (KnowledgeStore + runtime DB)
    # ------------------------------------------------------------------
    @staticmethod
    def _trace_from_knowledge_attempt(
        *,
        task: TaskSpec,
        attempt: dict[str, Any],
    ) -> TaskTrace | None:
        if not isinstance(attempt, dict):
            return None
        content = attempt.get("content") if isinstance(attempt.get("content"), dict) else {}
        eval_result = dict(content.get("eval_results") or {})
        score = attempt.get("score")
        if score is None:
            score = _score_from_eval(eval_result, task=task)
        try:
            native_score = float(score) if score is not None else None
        except (TypeError, ValueError):
            native_score = None
        runtime_meta = _history_runtime_meta(
            history_source="knowledge_store",
            carry_forward_payload=carry_forward_payload(content.get("attempt_meta")),
        )
        return TaskTrace(
            generation=int(attempt.get("gen") or 0),
            agent_id=str(attempt.get("agent_id") or ""),
            task_id=task.id,
            model_output=str(content.get("model_output") or ""),
            eval_result=eval_result,
            native_score=native_score,
            tool_trace=[],
            runtime_meta=runtime_meta,
            token_usage=TokenUsage(),
            repo=task.repo or "",
        )

    @staticmethod
    def _trace_from_memory_record(
        *,
        task: TaskSpec,
        record: dict[str, Any],
    ) -> TaskTrace | None:
        if not isinstance(record, dict):
            return None
        eval_result = dict(record.get("eval_results") or {})
        runtime_meta = _history_runtime_meta(
            history_source="runtime_store",
            carry_forward_payload=_memory_record_carry_forward_payload(record),
        )
        return TaskTrace(
            generation=int(record.get("gen") or 0),
            agent_id=str(record.get("agent_id") or ""),
            task_id=task.id,
            model_output=str(record.get("final_model_output") or ""),
            eval_result=eval_result,
            native_score=_score_from_eval(eval_result, task=task),
            tool_trace=[],
            runtime_meta=runtime_meta,
            token_usage=TokenUsage(),
            repo=task.repo or "",
        )

    def _load_best_preserved_trace_from_history(self, task: TaskSpec) -> TaskTrace | None:
        collab = self._collaborators()
        solved_threshold = collab.config.solved_threshold
        best: TaskTrace | None = None
        if collab.knowledge is not None:
            try:
                page = collab.knowledge.query_task(
                    task.id,
                    entry_types=["attempt"],
                    experiment=collab.config.experiment_name,
                    limit=20,
                )
            except Exception:
                log.warning("[ENGINE] Failed to load knowledge attempts for %s", task.id, exc_info=True)
            else:
                for attempt in page.get("attempts") or []:
                    candidate = self._trace_from_knowledge_attempt(task=task, attempt=attempt)
                    if not trace_meets_preserve_threshold(candidate, task=task, solved_threshold=solved_threshold):
                        continue
                    if best is None or trace_preserve_rank(candidate, task=task) > trace_preserve_rank(best, task=task):
                        best = candidate
        if best is not None:
            return best
        if collab.memory_store is None:
            return None
        try:
            rows = collab.memory_store.query_task_memory(
                task_id=task.id,
                experiment=collab.config.experiment_name or None,
                limit=20,
            )
        except Exception:
            log.warning("[ENGINE] Failed to load runtime memory for %s", task.id, exc_info=True)
            return None
        for row in rows:
            candidate = self._trace_from_memory_record(task=task, record=row)
            if not trace_meets_preserve_threshold(candidate, task=task, solved_threshold=solved_threshold):
                continue
            if best is None or trace_preserve_rank(candidate, task=task) > trace_preserve_rank(best, task=task):
                best = candidate
        return best

    def _best_preserved_trace_for_task(self, task: TaskSpec) -> TaskTrace | None:
        collab = self._collaborators()
        cached = collab.best_preserved_traces.get(task.id)
        if trace_meets_preserve_threshold(cached, task=task, solved_threshold=collab.config.solved_threshold):
            return cached
        loaded = self._load_best_preserved_trace_from_history(task)
        if loaded is not None:
            collab.best_preserved_traces[task.id] = loaded
        return loaded

    def _make_carried_forward_trace(
        self,
        *,
        generation: int,
        agent_id: str,
        task: TaskSpec,
        source_trace: TaskTrace,
    ) -> TaskTrace:
        collab = self._collaborators()
        preserved_score = trace_preserve_score(source_trace, task=task)
        source_payload = carry_forward_payload(source_trace.runtime_meta)
        source_generation = _coerce_int((source_payload or {}).get("carry_forward_source_generation")) or int(
            source_trace.generation or 0
        )
        source_agent_id = str(
            (source_payload or {}).get("carry_forward_source_agent_id") or source_trace.agent_id or ""
        )
        source_score = _coerce_float((source_payload or {}).get("carry_forward_source_score"))
        threshold = _coerce_float((source_payload or {}).get("carry_forward_threshold"))
        return TaskTrace(
            generation=generation,
            agent_id=agent_id,
            task_id=task.id,
            model_output=source_trace.model_output,
            eval_result=copy.deepcopy(source_trace.eval_result or {}),
            native_score=preserved_score,
            tool_trace=[],
            runtime_meta={
                "status": "carry_forward",
                "carry_forward": True,
                "carry_forward_reason": "best_score_preserved",
                "carry_forward_source_generation": source_generation,
                "carry_forward_source_agent_id": source_agent_id,
                "carry_forward_source_score": source_score if source_score is not None else preserved_score,
                "carry_forward_threshold": threshold
                if threshold is not None
                else float(collab.config.solved_threshold),
            },
            token_usage=TokenUsage(),
            repo=task.repo or "",
        )

    # ------------------------------------------------------------------
    # Public phase-service surface
    # ------------------------------------------------------------------
    def split_assignments(
        self,
        generation: int,
        assigned_map: dict[str, list[str]],
        task_by_id: dict[str, TaskSpec],
    ) -> tuple[dict[str, list[str]], list[TaskTrace]]:
        collab = self._collaborators()
        if collab.config.drop_solved:
            return dict(assigned_map), []
        execute_map: dict[str, list[str]] = {}
        carried: list[TaskTrace] = []
        for agent_id, task_ids in assigned_map.items():
            fresh_task_ids: list[str] = []
            for task_id in task_ids:
                if collab.is_holdout(task_id):
                    # Hold-out probe tasks are always attempted fresh — never
                    # replayed from a preserved trace, even when solved before.
                    fresh_task_ids.append(task_id)
                    continue
                task = task_by_id.get(task_id)
                if task is None:
                    fresh_task_ids.append(task_id)
                    continue
                source_trace = self._best_preserved_trace_for_task(task)
                if source_trace is None or int(source_trace.generation or 0) >= generation:
                    fresh_task_ids.append(task_id)
                    continue
                carried.append(
                    self._make_carried_forward_trace(
                        generation=generation,
                        agent_id=agent_id,
                        task=task,
                        source_trace=source_trace,
                    )
                )
            if fresh_task_ids:
                execute_map[agent_id] = fresh_task_ids
        return execute_map, carried

    def persist_carried(
        self,
        trace: TaskTrace,
        task_by_id: dict[str, TaskSpec],
    ) -> None:
        collab = self._collaborators()
        collab.persistence.on_task_status(
            generation=trace.generation,
            agent_id=trace.agent_id,
            task_id=trace.task_id,
            status="started",
        )
        collab.persistence.on_task_status(
            generation=trace.generation,
            agent_id=trace.agent_id,
            task_id=trace.task_id,
            status="completed",
        )
        collab.safe_on_task_trace(trace)
        if not collab.config.no_memory:
            collab.persist_task_memory_record(trace=trace, insight=None, lessons=[])
            collab.persist_task_summary(trace, task_by_id, lessons=[])
        collab.accumulator.record_task(trace.generation, trace.agent_id, trace.task_id, trace.token_usage)

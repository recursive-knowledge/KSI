"""Execution/evaluation phase-service boundary for the orchestrator."""

from __future__ import annotations

import copy
import json
import logging
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from typing import TYPE_CHECKING, Any, Callable, cast

from ..discussion.prompts import (
    build_task_reflection_and_lessons_prompt,
    parse_task_reflection_and_lessons_response,
)
from ..errors import AuthenticationFailure, ContainerRegistryError, find_container_registry_error
from ..errors import is_auth_error as _is_auth_error
from ..models import AgentState, EvalResult, Insight, TaskSpec, TaskTrace
from ..runtime import RuntimeResult
from ..runtime.normalize import SilentAgentRuntimeError, build_error_runtime_meta
from ..tasks.registry import resolve_source
from ..tokens import TokenUsage
from .attempt_events import _knowledge_attempt_external_id, _score_from_eval
from .task_retry import (
    _accumulate_failed_attempt_tokens,
    _cap_native_memory_fields,
    _is_retryable_task_error,
    _runtime_retry_meta,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.knowledge_store import KnowledgeStore
    from ..memory.store import MemoryStore
    from ..models import GenerationConfig
    from ..protocols import Evaluator, PersistenceObserver, RuntimeExecutor
    from ..tokens import TokenAccumulator
    from .engine import GenerationalOrchestrator

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionPhaseInput:
    """Inputs required to execute and evaluate task attempts for one generation."""

    generation: int
    tasks: "list[TaskSpec]"
    assigned_map: dict[str, list[str]]


@dataclass(frozen=True)
class ExecutionPhaseResult:
    """Output produced by the task execution/evaluation phase."""

    traces: "list[TaskTrace]"


type _IntermediateResult = tuple[AgentState, TaskSpec | None, Any | None, Exception | None]


@dataclass(frozen=True)
class ExecutionCollaborators:
    """Explicit dependencies for the execution phase service body."""

    config: "GenerationConfig"
    runtime: "RuntimeExecutor"
    persistence: "PersistenceObserver"
    evaluator: "Evaluator"
    accumulator: "TokenAccumulator"
    knowledge: "KnowledgeStore | None"
    memory_store: "MemoryStore | None"
    agents: list[Any]  # read-only snapshot
    best_scores: dict[str, float]  # live dict ref (mutated in place by engine, never reassigned)
    set_last_task_by_id: Callable[[dict[str, Any]], None]
    safe_on_task_trace: Callable[[Any], None]
    record_task_tokens: Callable[[Any], None]
    persist_task_summary: Callable[..., None]
    persist_task_memory_record: Callable[..., None]  # keyword-only: trace, insight, lessons, agent
    get_or_build_kt_adapter_memo: Callable[..., Any]
    llm_call: Callable[..., Any]
    maybe_embed: Callable[[str], "list[float] | None"]
    is_holdout: Callable[[str], bool]
    tag_holdout_meta: Callable[..., Any]
    merge_optional_meta: Callable[..., Any]
    # Shared engine helpers kept on the engine (also used by carry-forward /
    # forum paths); the eval body reaches them only through this object.
    merge_attempt_meta: Callable[..., Any]
    retrieved_distillation_ids: Callable[..., Any]
    knowledge_trace_condensed: Callable[..., str]


@dataclass
class EngineExecutionPhaseService:
    """Engine-backed execution phase adapter.

    The task execution/evaluation body lives here behind an explicit service
    boundary used by the generation loop and tests.
    """

    engine: "GenerationalOrchestrator"

    def _collaborators(self) -> ExecutionCollaborators:
        engine = self.engine
        return ExecutionCollaborators(
            config=engine.config,
            runtime=engine.runtime,
            persistence=engine.persistence,
            evaluator=engine.evaluator,
            accumulator=engine.accumulator,
            knowledge=engine._knowledge,
            memory_store=engine._memory_store,
            agents=engine.agents,
            best_scores=engine._best_scores,
            set_last_task_by_id=lambda value: setattr(engine, "_last_task_by_id", value),
            safe_on_task_trace=engine._safe_on_task_trace,
            record_task_tokens=engine._record_task_tokens,
            persist_task_summary=engine._persist_task_summary,
            persist_task_memory_record=engine._persist_task_memory_record,
            get_or_build_kt_adapter_memo=engine._kt_adapter_service.get_or_build_memo,
            llm_call=engine._llm_call,
            maybe_embed=engine._maybe_embed,
            is_holdout=engine._is_holdout,
            tag_holdout_meta=engine._tag_holdout_meta,
            merge_optional_meta=engine._merge_optional_meta,
            merge_attempt_meta=engine._merge_attempt_meta,
            retrieved_distillation_ids=engine._retrieved_distillation_ids,
            knowledge_trace_condensed=engine._knowledge_trace_condensed,
        )

    def run(self, phase_input: ExecutionPhaseInput) -> ExecutionPhaseResult:
        traces = self._execute_default(phase_input)
        dispatched_count = sum(len(task_ids) for task_ids in phase_input.assigned_map.values())
        if (
            dispatched_count
            and len(traces) == dispatched_count
            and all((trace.runtime_meta or {}).get("error_origin") == "container_registry" for trace in traces)
        ):
            raise ContainerRegistryError(
                f"all {dispatched_count} dispatched tasks failed during container-registry acquisition "
                f"in generation {phase_input.generation}; see persisted task traces",
                reason="generation_registry_failure",
            )
        return ExecutionPhaseResult(traces=traces)

    def _execute_default(self, phase_input: ExecutionPhaseInput) -> list[TaskTrace]:
        collab = self._collaborators()
        generation = phase_input.generation
        tasks = phase_input.tasks
        assigned_map = phase_input.assigned_map
        task_by_id = {t.id: t for t in tasks}
        collab.set_last_task_by_id(task_by_id)  # Exposed for per-task forum descriptions
        agent_by_id = {a.id: a for a in collab.agents}
        # A failure of the *authoritative* KnowledgeStore write is fatal, but we
        # must not abort the collection loop mid-flight — that would discard
        # in-flight eval results that already consumed API tokens and container
        # time, and leave _best_scores stale (next gen re-attempts solved tasks).
        # Capture the first such failure and re-raise AFTER all in-flight results
        # are collected and recorded.
        deferred_persist_error: BaseException | None = None
        # Per-task wall-clock start times (monotonic) keyed by (agent_id, task_id),
        # so the per-task "done" INFO line can report elapsed seconds. Written once
        # per task from the agent worker thread; read once on the main thread.
        task_start_times: dict[tuple[str, str], float] = {}

        # ------------------------------------------------------------------
        # Stage-1: run the agent in a container (the expensive part).
        # Returns an "intermediate" result — everything EXCEPT the eval.
        # ------------------------------------------------------------------
        def _run_agent_stage(
            agent: AgentState,
            task_id: str,
        ) -> _IntermediateResult:
            """Run the shared task container — no evaluation yet.

            This function MUST NOT raise — all exceptions are returned as the
            4th element of the tuple so the caller always gets a result.
            """
            task = None
            try:
                task = task_by_id[task_id]
                task_start_times[(agent.id, task.id)] = time.monotonic()
                log.info("[gen %s] task=%s agent=%s start", generation, task.id, agent.id)
                collab.persistence.on_task_status(
                    generation=generation,
                    agent_id=agent.id,
                    task_id=task.id,
                    status="started",
                )
                best_score_val = collab.best_scores.get(task.id)
                task_meta = {**task.metadata, "best_score": best_score_val}
                task_for_run = dataclass_replace(task, metadata=task_meta)
                agent_seed_package = copy.deepcopy(agent.seed_package or {})
                task_source = str((task.metadata or {}).get("task_source") or "").strip().lower()
                if (
                    str(agent_seed_package.get("_kt_mode") or "") == "adapter_transfer"
                    and task_source in {"arc", "polyglot"}
                    and "kt_adapter_memo" not in agent_seed_package
                    and isinstance(agent_seed_package.get("cross_task_bundle"), dict)
                ):
                    memo = collab.get_or_build_kt_adapter_memo(
                        generation=generation,
                        agent=agent,
                        task=task_for_run,
                        cross_task=agent_seed_package["cross_task_bundle"],
                    )
                    if isinstance(memo, dict) and memo:
                        agent_seed_package["kt_adapter_memo"] = memo
                        agent_seed_package["_kt_task_source"] = task_source
                attempts = max(1, int(getattr(collab.config, "max_task_retries", 0) or 0) + 1)
                last_exc: Exception | None = None
                attempt_errors: list[dict[str, Any]] = []
                failed_runtime_metas: list[dict[str, Any]] = []
                for attempt_idx in range(attempts):
                    try:
                        run_result = collab.runtime.run_task(
                            generation=generation,
                            agent_id=agent.id,
                            task=task_for_run,
                            agent_seed_package=copy.deepcopy(agent_seed_package),
                            experiment_name=collab.config.experiment_name,
                        )
                        if attempt_errors and isinstance(run_result, RuntimeResult):
                            runtime_meta = dict(run_result.runtime_meta or {})
                            runtime_meta.update(
                                _runtime_retry_meta(
                                    attempt_errors,
                                    terminal_failure=False,
                                    failed_runtime_metas=failed_runtime_metas,
                                    # The direct _accumulate_failed_attempt_tokens
                                    # call just below recomputes this same sum
                                    # from the same input and owns the
                                    # drop-count WARNING, so it's suppressed
                                    # here to avoid double-logging one drop.
                                    log_dropped_tokens=False,
                                )
                            )
                            run_result.runtime_meta = runtime_meta
                            # Add the failed attempts' billable cost back into
                            # ``run_result.token_usage`` so the eval-stage
                            # ``trace.token_usage`` and the per-task
                            # ``token_phases`` row reflect the real cost of
                            # the run, not just the success attempt. SDK-race
                            # failures with ``tokens_source=per_turn_sum``
                            # routinely consume 30k+ cache tokens before the
                            # silent error fires.
                            run_result.token_usage = run_result.token_usage + _accumulate_failed_attempt_tokens(
                                failed_runtime_metas
                            )
                            # Surface the retry-success attempt count so operators
                            # watching a campaign log can distinguish a first-try
                            # success from a task that needed retries.
                            log.warning(
                                "[ENGINE] task succeeded after %d retry/retries generation=%s agent=%s task=%s attempts=%d/%d",
                                len(attempt_errors),
                                generation,
                                agent.id,
                                task.id,
                                attempt_idx + 1,
                                attempts,
                            )
                        return (agent, task, run_result, None)
                    except Exception as exc:
                        last_exc = exc
                        attempt_error: dict[str, Any] = {
                            "attempt": attempt_idx + 1,
                            "max_attempts": attempts,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                        registry_error = find_container_registry_error(exc)
                        if registry_error is not None:
                            attempt_error.update(
                                {
                                    "error_origin": "container_registry",
                                    "registry_failure_reason": registry_error.reason,
                                    "registry_failure_retryable": registry_error.retryable,
                                    "registry_image": registry_error.image,
                                }
                            )
                        attempt_errors.append(attempt_error)
                        # Preserve the on-host session transcript from
                        # SilentAgentRuntimeError across the retry boundary so a
                        # later-succeeding attempt's runtime_meta still contains
                        # the forensic evidence from the failed attempts.
                        if isinstance(exc, SilentAgentRuntimeError) and isinstance(exc.runtime_meta, dict):
                            failed_runtime_metas.append(dict(exc.runtime_meta))
                        if attempt_idx >= attempts - 1 or not _is_retryable_task_error(exc):
                            break
                        log.warning(
                            "[ENGINE] transient task failure, retrying generation=%s agent=%s task=%s attempt=%s/%s: %s",
                            generation,
                            agent.id,
                            task.id,
                            attempt_idx + 1,
                            attempts,
                            exc,
                        )
                        collab.persistence.on_task_status(
                            generation=generation,
                            agent_id=agent.id,
                            task_id=task.id,
                            status="retrying",
                        )
                        delay = min(60, 0.5 * 2**attempt_idx) * (0.5 + random.random())
                        time.sleep(delay)
                if last_exc is not None and attempt_errors:
                    setattr(
                        last_exc,
                        "runtime_retry_meta",
                        _runtime_retry_meta(
                            attempt_errors,
                            terminal_failure=True,
                            failed_runtime_metas=failed_runtime_metas,
                            # The eval stage folds the same failed-attempt sum
                            # into ``trace.token_usage`` via a direct
                            # ``_accumulate_failed_attempt_tokens`` call and owns
                            # the drop-count WARNING, so suppress it here to
                            # avoid double-logging one drop.
                            log_dropped_tokens=False,
                        ),
                    )
                    # Carry the raw failed-attempt runtime_meta list across the
                    # raise boundary so the eval stage can fold the real billed
                    # cost of the failed attempts into the persisted
                    # ``trace.token_usage`` instead of zeroing it.
                    setattr(last_exc, "failed_runtime_metas", failed_runtime_metas)
                return (agent, task, None, last_exc or RuntimeError(f"task execution failed for {task.id}"))
            except Exception as exc:
                fallback_task = task_by_id.get(task_id) if task is None else task
                return (agent, fallback_task, None, exc)

        # ------------------------------------------------------------------
        # Two-pool pipeline: agent runs and eval/reflect run concurrently.
        #
        # As each agent future completes, its eval is submitted to a second
        # pool.  This keeps agent workers free to start the next task while
        # evaluation (often equally expensive) runs in parallel.
        # ------------------------------------------------------------------
        all_task_pairs: list[tuple[AgentState, str]] = []
        for agent in collab.agents:
            for tid in assigned_map.get(agent.id, []):
                all_task_pairs.append((agent, tid))

        traces: list[TaskTrace] = []
        cap = collab.config.max_concurrent_tasks if collab.config.max_concurrent_tasks > 0 else 50
        agent_workers = min(max(1, len(all_task_pairs)), cap)
        eval_workers = min(max(1, len(all_task_pairs)), cap)
        log.info(
            "[ENGINE] gen=%d dispatching %d tasks (agent_workers=%d, eval_workers=%d, cap=%d)",
            generation,
            len(all_task_pairs),
            agent_workers,
            eval_workers,
            cap,
        )
        with (
            ThreadPoolExecutor(max_workers=agent_workers) as agent_pool,
            ThreadPoolExecutor(max_workers=eval_workers) as eval_pool,
        ):
            # Submit all agent runs.
            agent_future_map = {
                agent_pool.submit(_run_agent_stage, agent, tid): (agent.id, tid) for agent, tid in all_task_pairs
            }
            # As agent futures complete, submit eval futures.
            eval_future_map: dict[Any, tuple[str, str]] = {}
            for agent_fut in as_completed(agent_future_map):
                agent_id, task_id = agent_future_map[agent_fut]
                try:
                    agent_state, task_obj, run_result, error = agent_fut.result()
                except Exception as exc:
                    log.error("[ENGINE] agent stage task %s (agent %s) failed: %s", task_id, agent_id, exc)
                    _task_obj = task_by_id.get(task_id)
                    # Same silent-failure meta preservation as _eval_one_attempt —
                    # this branch should not normally fire for SilentAgentRuntimeError
                    # (`_run_agent_stage` catches and returns it via the 4th tuple
                    # element) but handle it defensively.
                    preserved_meta_agent: dict[str, Any] = {}
                    if isinstance(exc, SilentAgentRuntimeError) and isinstance(exc.runtime_meta, dict):
                        preserved_meta_agent = _cap_native_memory_fields(exc.runtime_meta)
                    else:
                        # Ensure ``runtime_meta_json`` carries a discoverable
                        # status + error message even when no run_result was
                        # produced.  Without this overlay, analytics see
                        # ``runtime_meta_json = "{}"`` and can't distinguish
                        # this branch from a genuine silent failure.
                        preserved_meta_agent = build_error_runtime_meta(exc)
                    trace = TaskTrace(
                        generation=generation,
                        agent_id=agent_id,
                        task_id=task_id,
                        model_output=None,
                        eval_result={},
                        native_score=None,
                        tool_trace=[],
                        runtime_meta=preserved_meta_agent,
                        token_usage=TokenUsage(),
                        error=f"agent_stage_exception: {exc}",
                        repo=getattr(_task_obj, "repo", "") or "",
                    )
                    collab.persistence.on_task_status(
                        generation=generation,
                        agent_id=agent_id,
                        task_id=task_id,
                        status="failed",
                    )
                    traces.append(trace)
                    _started_at = task_start_times.get((agent_id, task_id))
                    _elapsed_text = "n/a" if _started_at is None else f"{time.monotonic() - _started_at:.1f}s"
                    log.info(
                        "[gen %s] task=%s agent=%s done elapsed=%s score=n/a error",
                        generation,
                        task_id,
                        agent_id,
                        _elapsed_text,
                    )
                    collab.safe_on_task_trace(trace)
                    continue
                eval_fut = eval_pool.submit(
                    self._eval_one_attempt,
                    agent_state,
                    cast(TaskSpec, task_obj),
                    run_result,
                    error,
                    generation,
                    task_by_id,
                )
                eval_future_map[eval_fut] = (agent_id, task_id)

            # Collect eval results.
            for eval_fut in as_completed(eval_future_map):
                agent_id, task_id = eval_future_map[eval_fut]
                try:
                    trace, insight, lessons, extra_tokens = eval_fut.result()
                except AuthenticationFailure:
                    # Auth failures are fatal — do NOT wrap into a failed
                    # trace; re-raise so the run aborts loudly rather than
                    # silently producing 0/N solved.
                    raise
                except Exception as exc:
                    log.error("[ENGINE] eval stage task %s (agent %s) failed: %s", task_id, agent_id, exc)
                    # Create a failed trace so the task isn't silently dropped.
                    # Overlay status=error onto runtime_meta so the attempt row
                    # never lands with an empty ``runtime_meta_json``.
                    _task_obj = task_by_id.get(task_id)
                    trace = TaskTrace(
                        generation=generation,
                        agent_id=agent_id,
                        task_id=task_id,
                        model_output=None,
                        eval_result={},
                        native_score=None,
                        tool_trace=[],
                        runtime_meta=build_error_runtime_meta(exc),
                        token_usage=TokenUsage(),
                        error=f"eval_stage_exception: {exc}",
                        repo=getattr(_task_obj, "repo", "") or "",
                    )
                    collab.persistence.on_task_status(
                        generation=generation,
                        agent_id=agent_id,
                        task_id=task_id,
                        status="failed",
                    )
                    insight, lessons, extra_tokens = None, [], 0
                traces.append(trace)
                # Per-task progress line so long generations are monitorable at
                # INFO (one start + one done per task). Pairs with the "start"
                # line emitted in _run_agent_stage.
                _started_at = task_start_times.get((agent_id, task_id))
                _elapsed_text = "n/a" if _started_at is None else f"{time.monotonic() - _started_at:.1f}s"
                _score_text = "n/a" if trace.native_score is None else f"{trace.native_score:.4f}"
                log.info(
                    "[gen %s] task=%s agent=%s done elapsed=%s score=%s%s",
                    generation,
                    task_id,
                    agent_id,
                    _elapsed_text,
                    _score_text,
                    " error" if trace.error is not None else "",
                )
                if not bool((trace.runtime_meta or {}).get("_task_trace_persisted_early")):
                    collab.safe_on_task_trace(trace)
                # Log tool call counts for observability
                runtime_meta = trace.runtime_meta or {}
                tool_counts = runtime_meta.get("tool_call_counts")
                if isinstance(tool_counts, dict) and tool_counts:
                    memory_tools = runtime_meta.get("memory_tool_call_counts")
                    forum_tools = runtime_meta.get("forum_tool_call_counts")
                    arc_tools = runtime_meta.get("arc_tool_call_counts")
                    if not isinstance(memory_tools, dict):
                        memory_tools = {
                            k: v
                            for k, v in tool_counts.items()
                            if isinstance(k, str) and (k.startswith("mcp__memory__") or k in {"query", "forum_read"})
                        }
                    if not isinstance(forum_tools, dict):
                        forum_tools = {
                            k: v
                            for k, v in tool_counts.items()
                            if isinstance(k, str) and (k.startswith("mcp__memory__forum_") or k == "forum_read")
                        }
                    if not isinstance(arc_tools, dict):
                        arc_tools = {
                            k: v for k, v in tool_counts.items() if isinstance(k, str) and k.startswith("arc_")
                        }
                    log.info(
                        "[ENGINE] task=%s agent=%s tools=%d memory_tools=%s forum_tools=%s arc_tools=%s",
                        trace.task_id,
                        trace.agent_id,
                        sum(tool_counts.values()),
                        memory_tools or "none",
                        forum_tools or "none",
                        arc_tools or "none",
                    )
                # Accumulate token usage and task count into agent state (main thread only)
                trace_agent = agent_by_id.get(trace.agent_id)
                if trace_agent is not None:
                    trace_agent.token_usage += trace.token_usage.total + extra_tokens
                    if trace.error is None:
                        trace_agent.tasks_completed += 1
                if not collab.config.no_memory:
                    try:
                        collab.persist_task_memory_record(
                            trace=trace,
                            insight=insight,
                            lessons=lessons,
                            agent=trace_agent,
                        )
                    except Exception as exc:
                        # Authoritative-store failure is fatal, but defer the
                        # abort until the loop has collected every in-flight
                        # result (see deferred_persist_error above).
                        if deferred_persist_error is None:
                            deferred_persist_error = exc
                        log.error(
                            "[ENGINE] authoritative persist failed for %s; "
                            "aborting after in-flight results are collected: %s",
                            trace.task_id,
                            exc,
                        )
                    # Embed task summary in the runtime DB for vector search.
                    collab.persist_task_summary(trace, task_by_id, lessons=lessons)
                # Persist per-task reflection insight (generated in worker thread).
                if insight is not None and trace_agent is not None:
                    self._record_r0_insight(
                        generation=generation,
                        agent=trace_agent,
                        trace=trace,
                        insight=insight,
                    )
                # Accumulate task tokens in main thread (thread-safe).
                # Hold-out attempts land under phase ``task_execution_holdout``.
                collab.record_task_tokens(trace)
        if deferred_persist_error is not None:
            raise RuntimeError(
                "authoritative KnowledgeStore persist failed during generation "
                f"{generation}; aborted after collecting in-flight task results"
            ) from deferred_persist_error
        return traces

    def _eval_one_attempt(
        self,
        agent: AgentState,
        task: TaskSpec,
        run_result: Any | None,
        error: Exception | None,
        generation: int,
        task_by_id: dict[str, TaskSpec],
    ) -> tuple[TaskTrace, Insight | None, list | None, int]:
        """Evaluate the agent output, persist transcript, generate insight.

        Returns (trace, insight, lessons, extra_tokens) where extra_tokens
        is the total LLM token usage from reflection/lesson calls that must
        be accumulated in the main thread (not in worker threads).
        """
        collab = self._collaborators()
        extra_tokens = 0
        # Phase-1 reflection tokens are accumulated separately because
        # ``extra_tokens`` is *reassigned* (not ``+=``) further down at the
        # no_memory / insight+lesson branches; folding p1 in at the single
        # ``return`` keeps it from being clobbered AND off the worker thread.
        p1_extra_tokens = 0
        # Polyglot test-feedback retry-round tokens, accumulated the same
        # way as p1_extra_tokens above (off the worker thread, folded in at
        # the single ``return`` so they aren't clobbered).
        polyglot_tf_extra_tokens = 0
        task_id = task.id
        try:
            if error is not None:
                raise error
            # run_result is Any (collab.runtime is Any); the RuntimeResult branch
            # narrows model_output to str while the else carries the raw output —
            # declare the union so mypy doesn't pin it to str.
            model_output: Any
            if isinstance(run_result, RuntimeResult):
                model_output = run_result.output
                tool_trace = run_result.tool_trace
                runtime_meta = run_result.runtime_meta
                token_usage = run_result.token_usage
            else:
                model_output = run_result
                tool_trace = []
                runtime_meta = {}
                token_usage = TokenUsage()
            # Phase-1 reflection (Path a): when the host-side
            # ``BarrierWatcher`` already invoked
            # ``evaluator.evaluate(...)`` on the agent's submission to
            # produce the score the agent reflected on, the executor
            # caches that ``EvalResult`` into
            # ``runtime_meta.phase1_eval_result``. Reusing it here
            # avoids paying for a SECOND Docker subprocess on
            # polyglot / swebench_pro and prevents score-disagreement
            # between the watcher- and engine-side calls (the
            # reflection text is written from the watcher's score; the
            # DB stored the engine's). The watcher passes the full
            # model_output through the sentinel (8MB cap; see
            # ``index.ts`` ``runPhase1Reflection``) so the inputs
            # match. If the watcher errored OR the flag was off, the
            # cached value is absent and we fall through to the
            # in-process evaluate() call as before.
            # Polyglot test-feedback (Path b): mirrors phase1's reuse above,
            # but ONLY when the host-side watcher reported
            # ``polyglot_test_feedback_reuse_eligible`` — set in
            # ``_postprocess_runner_output`` exclusively when the TS side's
            # ``final_eval_matches_output`` confirmed the last cached eval
            # is the state actually being graded (no agent turn ran after
            # it). A retry round that edited files after its last barrier
            # evaluation must NOT reuse that (now-stale) cached value.
            cached_eval = None
            if (
                isinstance(runtime_meta, dict)
                and runtime_meta.get("phase1_reflection_enabled")
                and "phase1_eval_result" in runtime_meta
            ):
                cached_eval = runtime_meta.get("phase1_eval_result")
            elif (
                isinstance(runtime_meta, dict)
                and runtime_meta.get("polyglot_test_feedback_reuse_eligible")
                and "polyglot_test_feedback_eval_result" in runtime_meta
            ):
                cached_eval = runtime_meta.get("polyglot_test_feedback_eval_result")
            if cached_eval is not None:
                # Phase-1 evaluates before the final tool trace exists. Never
                # reuse an empty-patch result when the completed trace can
                # supply mutation/capture evidence.
                cached_status = cached_eval.get("swebench_status") if isinstance(cached_eval, dict) else None
                if tool_trace and cached_status in {"no_patch", "capture_failed"}:
                    eval_result = collab.evaluator.evaluate(
                        task=task, model_output=model_output, runtime_meta=runtime_meta, tool_trace=tool_trace
                    )
                else:
                    eval_result = cached_eval
            else:
                eval_result = collab.evaluator.evaluate(
                    task=task,
                    model_output=model_output,
                    runtime_meta=runtime_meta,
                    tool_trace=tool_trace,
                )
            score = _score_from_eval(eval_result, task=task)
            # Record the phase-1-reflection and polyglot-test-feedback
            # lifecycle token phases from ``runtime_meta`` sub-dicts. The
            # returned totals are folded into the single ``return`` on the
            # main thread (see the ``p1_extra_tokens`` comment above).
            p1_extra_tokens, polyglot_tf_extra_tokens = self._record_reflection_token_phases(
                collab=collab,
                runtime_meta=runtime_meta,
                generation=generation,
                agent=agent,
                task_id=task_id,
            )
            trace = TaskTrace(
                generation=generation,
                agent_id=agent.id,
                task_id=task_id,
                model_output=model_output,
                eval_result=cast(EvalResult, eval_result),
                native_score=score,
                tool_trace=tool_trace,
                runtime_meta=runtime_meta,
                token_usage=token_usage,
                repo=getattr(task, "repo", "") or "",
            )
            collab.persistence.on_task_status(
                generation=generation,
                agent_id=agent.id,
                task_id=task_id,
                status="completed",
            )
        except AuthenticationFailure:
            raise
        except Exception as exc:
            if _is_auth_error(exc):
                log.error(
                    "[ENGINE] LLM auth failure in _eval_one_attempt (agent=%s, task=%s) — aborting run: %s",
                    agent.id,
                    task_id,
                    exc,
                )
                raise AuthenticationFailure(
                    f"LLM authentication failed in eval stage ({agent.id}, {task_id}): {exc}"
                ) from exc
            # Preserve runtime_meta (including ``native_session_memory`` /
            # ``raw_native_session_memory``) across the silent-failure raise
            # so the attempt row carries forensics evidence. Without this,
            # ``trace.runtime_meta = {}`` strips ~134 KB of session transcript
            # on DB write, leaving the failed attempt rows with empty
            # runtime_meta_json.
            #
            # Three sources feed ``preserved_meta`` in precedence order:
            #   1. ``SilentAgentRuntimeError.runtime_meta`` — carries the
            #      silent-failure status + native session transcript.
            #   2. ``run_result.runtime_meta`` — container ran successfully
            #      but the evaluator raised (e.g. polyglot's pretask
            #      path-traversal guard raising ValueError).  Preserving
            #      this means we don't drop ``duration_ms`` / token counts /
            #      session memory that the container runner already
            #      emitted.  ``build_error_runtime_meta`` then overlays
            #      ``status='error'`` + the exception message so the row is
            #      distinguishable from a clean success in analytics.
            #   3. Neither — container never ran (or ran but emitted no
            #      meta).  Still emit a minimal ``{status: 'error', ...}``
            #      dict so ``runtime_meta_json`` never lands status-less.
            preserved_meta: dict[str, Any] = {}
            if isinstance(exc, SilentAgentRuntimeError) and isinstance(exc.runtime_meta, dict):
                preserved_meta = _cap_native_memory_fields(exc.runtime_meta)
                # Route the retry forensics (``retry_failed_attempts_token_usage``,
                # ``runtime_attempt_errors``, ``attempt_N_native_session_memory``)
                # into the persisted meta on THIS branch too. The ``else`` branch
                # already merges it via ``build_error_runtime_meta``, but the
                # terminal retryable-failure path lands here — without this merge
                # the failed-attempt accounting is dropped from the row.
                # ``_cap_native_memory_fields`` returns a fresh dict, so ``update``
                # never mutates ``exc.runtime_meta``.
                retry_meta = getattr(exc, "runtime_retry_meta", None)
                if isinstance(retry_meta, dict):
                    preserved_meta.update(retry_meta)
            else:
                # The container may have run to completion before the eval
                # stage raised — carry its meta forward and overlay the
                # error marker.  ``run_result`` is closed over from the
                # enclosing function scope; it may be ``None`` if the run
                # stage itself failed.
                container_meta: dict[str, Any] = {}
                if isinstance(run_result, RuntimeResult) and isinstance(run_result.runtime_meta, dict):
                    container_meta = _cap_native_memory_fields(run_result.runtime_meta)
                preserved_meta = build_error_runtime_meta(exc, base=container_meta)
            # A terminally-failed retryable task consumed real billable tokens
            # on each failed attempt (often 30k+ cache reads on Haiku SDK-race
            # retries). ``_run_agent_stage`` attaches the failed attempts'
            # runtime_meta list to the terminal exception; fold their aggregated
            # cost into the persisted ``trace.token_usage`` — mirrors the
            # success-after-retry path — instead of zeroing it, so
            # ``record_task_tokens`` surfaces the real cost. Absent the
            # attribute (non-retry failures) this is a no-op zero TokenUsage.
            failed_attempt_tokens = _accumulate_failed_attempt_tokens(getattr(exc, "failed_runtime_metas", None))
            trace = TaskTrace(
                generation=generation,
                agent_id=agent.id,
                task_id=task_id,
                model_output=None,
                eval_result={},
                native_score=None,
                tool_trace=[],
                runtime_meta=preserved_meta,
                token_usage=failed_attempt_tokens,
                error=str(exc),
                repo=getattr(task, "repo", "") or "",
            )
            collab.persistence.on_task_status(
                generation=generation,
                agent_id=agent.id,
                task_id=task_id,
                status="failed",
            )
        # Persist the three eager attempt side-effects (early KnowledgeStore
        # attempt, early task trace, raw transcript) and stamp the two
        # ``runtime_meta`` flags read later by ``_execute_default``'s
        # skip-check.
        self._persist_attempt_side_effects(
            collab=collab,
            trace=trace,
            agent=agent,
            task_id=task_id,
        )
        lessons: list[str]
        if collab.config.no_memory:
            insight, lessons, extra_tokens = None, [], 0
        else:
            # Generate the per-task reflection insight and reusable lessons in
            # one merged LLM call.
            task_obj = task_by_id.get(trace.task_id)
            insight, lessons, extra_tokens = self._generate_reflection_and_lessons(
                generation=generation,
                agent=agent,
                trace=trace,
                task=task_obj,
            )
        # Fold in phase-1 reflection and polyglot test-feedback tokens
        # (accumulated off the worker thread above) so the main thread
        # applies them exactly once.
        return (trace, insight, lessons, extra_tokens + p1_extra_tokens + polyglot_tf_extra_tokens)

    def _record_reflection_token_phases(
        self,
        *,
        collab: ExecutionCollaborators,
        runtime_meta: Any,
        generation: int,
        agent: AgentState,
        task_id: str,
    ) -> tuple[int, int]:
        """Record the phase-1-reflection and polyglot-test-feedback lifecycle
        token phases from ``runtime_meta`` sub-dicts.

        Returns ``(p1_extra_tokens, polyglot_tf_extra_tokens)`` — the per-agent
        token totals the caller folds into its single ``return`` on the main
        thread (this runs inside an eval *worker* thread, so it must NOT mutate
        the shared ``agent.token_usage``).
        """
        p1_extra_tokens = 0
        polyglot_tf_extra_tokens = 0
        # Phase-1 reflection (Path a): when the in-container
        # follow-up SDK turn ran, the agent-runner surfaces its
        # ``result``-event token usage in
        # ``runtime_meta.phase1_reflection_token_usage``. Record
        # it as a dedicated ``phase1_reflection`` lifecycle phase
        # so it lands in the ``token_phases`` table — without
        # this the reflection turn's input/output/cache tokens
        # silently vanish from cost reports. Pattern mirrors the
        # ``forum_round_*`` and ``cross_task_forum_round_*``
        # phases already recorded via ``record_lifecycle``.
        if isinstance(runtime_meta, dict):
            p1_usage_raw = runtime_meta.get("phase1_reflection_token_usage")
            if isinstance(p1_usage_raw, dict):
                try:
                    p1_usage = TokenUsage(
                        input_tokens=int(p1_usage_raw.get("input_tokens") or 0),
                        output_tokens=int(p1_usage_raw.get("output_tokens") or 0),
                        cache_creation_input_tokens=int(p1_usage_raw.get("cache_creation_input_tokens") or 0),
                        cache_read_input_tokens=int(p1_usage_raw.get("cache_read_input_tokens") or 0),
                    )
                    if p1_usage.total > 0:
                        collab.accumulator.record_lifecycle(
                            generation,
                            agent.id,
                            "phase1_reflection",
                            p1_usage,
                        )
                        # Accumulate into ``p1_extra_tokens`` so per-agent
                        # totals match the phase-row aggregates. This runs
                        # inside an eval *worker* thread, so we must NOT
                        # mutate the shared ``agent.token_usage`` here (it
                        # races the main thread's bump in the eval-future
                        # collection loop). The value is returned and
                        # applied once by the main thread. We add the
                        # scalar ``.total`` (``agent.token_usage`` is an
                        # ``int``; ``int += TokenUsage`` raises TypeError —
                        # no ``__radd__``).
                        p1_extra_tokens += p1_usage.total
                except (TypeError, ValueError):
                    log.warning(
                        "[ENGINE] phase1_reflection_token_usage had non-numeric fields for agent=%s task=%s — skipping",
                        agent.id,
                        task_id,
                    )
        # Polyglot test-feedback: the same lifecycle-recording pattern as
        # phase1_reflection above, for the retry loop's extra SDK-turn
        # tokens (``runtime_meta.polyglot_test_feedback_token_usage``).
        # Without this, every retry round's tokens silently vanish from
        # ``token_phases`` cost reports.
        if isinstance(runtime_meta, dict):
            tf_usage_raw = runtime_meta.get("polyglot_test_feedback_token_usage")
            if isinstance(tf_usage_raw, dict):
                try:
                    tf_usage = TokenUsage(
                        input_tokens=int(tf_usage_raw.get("input_tokens") or 0),
                        output_tokens=int(tf_usage_raw.get("output_tokens") or 0),
                        cache_creation_input_tokens=int(tf_usage_raw.get("cache_creation_input_tokens") or 0),
                        cache_read_input_tokens=int(tf_usage_raw.get("cache_read_input_tokens") or 0),
                    )
                    if tf_usage.total > 0:
                        collab.accumulator.record_lifecycle(
                            generation,
                            agent.id,
                            "polyglot_test_feedback",
                            tf_usage,
                        )
                        polyglot_tf_extra_tokens += tf_usage.total
                except (TypeError, ValueError):
                    log.warning(
                        "[ENGINE] polyglot_test_feedback_token_usage had non-numeric "
                        "fields for agent=%s task=%s — skipping",
                        agent.id,
                        task_id,
                    )
        return p1_extra_tokens, polyglot_tf_extra_tokens

    def _persist_attempt_side_effects(
        self,
        *,
        collab: ExecutionCollaborators,
        trace: TaskTrace,
        agent: AgentState,
        task_id: str,
    ) -> None:
        """Persist the three eager attempt side-effects and stamp the two
        ``runtime_meta`` flags read later by ``_execute_default``'s skip-check.

        Ordering + idempotency + stamp-flag contract live here in one place:
        (1) early KnowledgeStore attempt → ``_knowledge_attempt_persisted_early``,
        (2) early task trace → ``_task_trace_persisted_early``,
        (3) raw transcript. Each side-effect swallows-and-warns so a sidecar
        fault never aborts the attempt (see the per-block comments); the flags
        are set only when their write actually succeeded.
        """
        # Persist the canonical KnowledgeStore attempt before reflection.
        # Reflection/lesson extraction and legacy runtime DB persistence can
        # lag behind execution at higher concurrency; forum/distillation must
        # still see attempts before seed phases begin.
        if collab.knowledge is not None:
            try:
                if self._persist_knowledge_attempt_early(trace, agent=agent):
                    runtime_meta = dict(trace.runtime_meta or {})
                    runtime_meta["_knowledge_attempt_persisted_early"] = True
                    trace.runtime_meta = runtime_meta
            except Exception as exc:
                # Don't raise: the late-path _persist_task_memory_record
                # will retry, and even if both fail, losing resume-cursor
                # state is preferable to corrupting gen_traces with a
                # phantom failure (the eval-future as_completed handler
                # at the call site converts any raise here into a
                # `eval_stage_exception` failed trace, which would
                # silently mark a successfully-solved task as failed).
                log.warning(
                    "[ENGINE] early KnowledgeStore record_attempt failed for agent=%s task=%s: %s "
                    "— continuing; late-path persistence will retry",
                    agent.id,
                    task_id,
                    exc,
                    exc_info=True,
                )
        # Persist the evaluated task trace immediately so scored attempts land
        # in the runtime DB before slower reflection/lesson phases complete.
        try:
            collab.persistence.on_task_trace(trace)
            runtime_meta = dict(trace.runtime_meta or {})
            runtime_meta["_task_trace_persisted_early"] = True
            trace.runtime_meta = runtime_meta
        except AuthenticationFailure:
            raise
        except Exception as exc:
            log.warning(
                "[ENGINE] early task-trace persistence failed for agent=%s task=%s: %s",
                agent.id,
                task_id,
                exc,
            )
        # Persist transcript immediately so it's queryable by other tasks.
        if collab.memory_store is not None and trace.error is None:
            native_mem = (trace.runtime_meta or {}).get("native_session_memory", "")
            transcript_content = native_mem or (trace.model_output or "")
            if transcript_content:
                try:
                    tool_trace_json = json.dumps(trace.tool_trace) if trace.tool_trace else "[]"
                    collab.memory_store.insert_raw_transcript(
                        experiment=collab.config.experiment_name,
                        agent_id=trace.agent_id,
                        generation=trace.generation,
                        task_id=trace.task_id,
                        content=transcript_content,
                        tool_trace=tool_trace_json,
                        model_output=trace.model_output,
                        eval_result_json=json.dumps(trace.eval_result) if trace.eval_result else "{}",
                        native_score=trace.native_score,
                        runtime_meta_json=json.dumps(trace.runtime_meta) if trace.runtime_meta else "{}",
                    )
                except Exception as exc:
                    log.warning("Failed to persist transcript: %s", exc)

    def _record_r0_insight(
        self,
        *,
        generation: int,
        agent: AgentState,
        trace: TaskTrace,
        insight: Insight,
    ) -> None:
        """Persist the per-task R0 reflection insight (audit + knowledge).

        Hold-out probe attempts produce NO knowledge insight rows — insight
        rows are untagged and would otherwise leak hold-out content into
        forum query retrieval. The audit-side ``on_insight`` callback still
        fires for observability.
        """
        collab = self._collaborators()
        collab.persistence.on_insight(
            generation=generation,
            agent_id=agent.id,
            insight=insight,
        )
        # Record R0 insight in unified knowledge store
        if collab.knowledge is not None and not collab.is_holdout(trace.task_id):
            try:
                insight_text = insight.text or ""
                insight_embedding = collab.maybe_embed(insight_text)
                collab.knowledge.record_insight(
                    task_id=trace.task_id,
                    agent_id=agent.id,
                    generation=generation,
                    text=insight_text,
                    scope="task",
                    confidence=insight.confidence or "medium",
                    evidence_task_ids=[insight.source_task_id] if insight.source_task_id else [],
                    round_num=0,
                    experiment=collab.config.experiment_name,
                    embedding=insight_embedding,
                )
            except Exception as exc:
                log.warning("[ENGINE] KnowledgeStore record_insight failed for R0 %s: %s", trace.task_id, exc)

    def _persist_knowledge_attempt_early(
        self,
        trace: TaskTrace,
        agent: AgentState | None = None,
    ) -> bool:
        """Persist a KnowledgeStore attempt before slower reflection work.

        Returns ``True`` when an attempt row was written. Runtime failures are
        still authoritative attempts for resume cursors and failure-mode audit,
        especially when ``--no-memory`` or ``--no-runtime-db`` disables later
        sidecar persistence.

        ``agent`` (when supplied) provides the seed package so the attempt
        row can record which distillation entries fed the agent — needed for
        knowledge → solve provenance joins.
        """
        collab = self._collaborators()
        if collab.knowledge is None:
            return False
        eval_results = dict(trace.eval_result or {})
        if trace.error:
            eval_results.setdefault("status", "error")
            eval_results.setdefault("error", trace.error)
        attempt_meta = collab.merge_attempt_meta(
            None,
            collab.retrieved_distillation_ids(agent),
        )
        attempt_meta = collab.tag_holdout_meta(attempt_meta, trace.task_id)
        # Phase-1 self-reflection (Path a) is shipped via runtime_meta from
        # the in-container barrier round-trip. Empty string when the
        # feature flag is off OR the agent didn't produce one — distill
        # consumers handle empty gracefully (see distillation/distiller.py).
        runtime_meta = trace.runtime_meta or {}
        reflection_text = ""
        if isinstance(runtime_meta, dict):
            reflection_text = str(runtime_meta.get("phase1_reflection") or "").strip()
        # A task source may register an ``attempt_meta_builder`` on its spec
        # (wired in attempt_events._attach_engine_source_formatters); sources
        # without one contribute no extra attempt_meta. No per-source dispatch here.
        early_task_source = str(((trace.runtime_meta or {}).get("task_source") or "")).strip().lower()
        early_spec = resolve_source(early_task_source)
        early_attempt_meta_builder = early_spec.attempt_meta_builder if early_spec is not None else None
        collab.knowledge.record_attempt(
            task_id=trace.task_id,
            agent_id=trace.agent_id,
            generation=trace.generation,
            eval_results=eval_results,
            model_output=trace.model_output or "",
            trace_condensed=collab.knowledge_trace_condensed(trace),
            insights=[],
            native_score=trace.native_score,
            experiment=collab.config.experiment_name,
            embedding=collab.maybe_embed(collab.knowledge_trace_condensed(trace)),
            attempt_meta=collab.merge_optional_meta(
                attempt_meta,
                early_attempt_meta_builder(trace) if early_attempt_meta_builder is not None else None,
            ),
            reflection=reflection_text,
            repo=trace.repo,
            # Stable id shared with the engine's later, richer write for the
            # same (task_id, agent_id, generation) execution attempt — lets
            # that write supersede this placeholder in place instead of
            # being silently skipped.
            external_id=_knowledge_attempt_external_id(
                task_id=trace.task_id,
                agent_id=trace.agent_id,
                generation=trace.generation,
            ),
        )
        return True

    def _generate_reflection_and_lessons(
        self,
        *,
        generation: int,
        agent: AgentState,
        trace: TaskTrace,
        task: TaskSpec | None,
    ) -> tuple[Insight | None, list[str], int]:
        """Generate the per-task reflection insight AND reusable lessons in a
        SINGLE LLM call.

        Both deliverables mine the same attempt excerpt, so merging them removes
        one LLM round-trip per attempt. Returns (insight, lessons, token_total)
        so callers accumulate token usage on the main thread rather than mutating
        agent state from worker threads. Token usage is recorded once under the
        ``task_reflection`` phase (the former ``lesson_extraction`` phase folds
        into it). A malformed/partial response degrades gracefully — a missing
        insight does not discard valid lessons and vice-versa.

        Skips the call entirely on an errored trace OR an empty/whitespace
        model output: an empty excerpt has nothing to reflect on, and the
        former two-call path already gated lesson extraction on non-empty
        output (restoring that guard avoids soliciting hallucinated lessons —
        and a wasted LLM call — from an empty attempt).
        """
        collab = self._collaborators()
        if trace.error is not None or not (trace.model_output or "").strip():
            return None, [], 0
        eval_result = trace.eval_result or {}
        status = str(eval_result.get("status") or eval_result.get("swebench_status") or "n/a")
        resolved = eval_result.get("resolved")
        resolved_text = "unknown" if resolved is None else str(bool(resolved)).lower()
        eval_summary = (
            f"- status: {status}\n- resolved: {resolved_text}\n"
            f"- native_score: {'n/a' if trace.native_score is None else f'{trace.native_score:.4f}'}"
        )
        outcome = "resolved" if eval_result.get("resolved") or trace.native_score == 1.0 else "unresolved"
        score_text = "n/a" if trace.native_score is None else f"{trace.native_score:.2f}"
        task_prompt_preview = ((task.prompt if task else "") or "")[:300].replace("\n", " ")
        # First 1200 chars (approach/reasoning) + last 1200 chars (result/grid)
        # captures both strategy and outcome for insight AND lesson generation.
        _raw = trace.model_output or ""
        if len(_raw) > 2500:
            model_output_excerpt = (_raw[:1200] + "\n...(truncated)...\n" + _raw[-1200:]).replace("\n", " ")
        else:
            model_output_excerpt = _raw.replace("\n", " ")
        system, prompt = build_task_reflection_and_lessons_prompt(
            agent_id=agent.id,
            agent_workstream=agent.workstream,
            task_id=trace.task_id,
            task_repo=(task.repo if task else ""),
            task_prompt_preview=task_prompt_preview,
            eval_summary=eval_summary,
            outcome=outcome,
            score_text=score_text,
            model_output_excerpt=model_output_excerpt,
        )
        try:
            resp = collab.llm_call(
                system=system,
                user=prompt,
                context={
                    "phase": "task_reflection",
                    "generation": generation,
                    "agent_id": agent.id,
                    "task_id": trace.task_id,
                },
            )
            raw, usage = resp.text, resp.usage
            collab.accumulator.record_lifecycle(generation, agent.id, "task_reflection", usage)
            parsed = parse_task_reflection_and_lessons_response(raw)
            insight_data = parsed.get("insight")
            insight = None
            if insight_data:
                insight = Insight(
                    id=f"task_insight_{uuid.uuid4().hex}",
                    text=insight_data["text"],
                    author_agent_id=agent.id,
                    generation=generation,
                    workstream=insight_data["workstream"],
                    source_task_id=trace.task_id,
                    confidence=insight_data["confidence"],
                )
            lessons = parsed.get("lessons") or []
            return insight, lessons, usage.total
        except Exception as exc:
            if _is_auth_error(exc):
                log.error(
                    "[ENGINE] LLM auth failure during task_reflection — aborting run. agent=%s task=%s error=%s",
                    agent.id,
                    trace.task_id,
                    exc,
                )
                raise AuthenticationFailure(f"LLM authentication failed for task_reflection: {exc}") from exc
            log.warning(
                "[ENGINE] reflection+lessons generation failed for agent=%s task=%s: %s",
                agent.id,
                trace.task_id,
                exc,
            )
            return None, [], 0

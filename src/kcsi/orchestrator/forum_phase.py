"""Forum phase-service boundary for improvement strategies."""

from __future__ import annotations

import copy
import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, cast

from ..errors import AuthenticationFailure, KcsiError
from ..forum.prompt import build_per_task_discussion_parts
from ..runtime.native_memory import collect_native_session_memory
from .forum_runtime import (
    _coerce_round_usage,
    _CrossTaskR1Coordinator,
    _drain_forum_bus,
    _forum_container_prefix,
    _ForumEarlyExitWatcher,
    _run_retryable_forum_task,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.knowledge_store import KnowledgeStore
    from ..memory.store import MemoryStore
    from ..models import AgentState, GenerationConfig, TaskTrace
    from ..protocols import PersistenceObserver, RuntimeExecutor
    from ..tokens import TokenAccumulator, TokenUsage, TokenUsageDict
    from .engine import GenerationalOrchestrator

log = logging.getLogger(__name__)


def _resolve_forum_worker_cap(config: GenerationConfig, pool_size: int) -> int:
    """Resolve the effective forum worker count for a discussion phase.

    An explicit positive ``max_concurrent_forum_tasks`` wins; 0/unset (the
    default) follows ``max_concurrent_tasks`` so forum phases can never
    silently exceed the task-execution cap. Bounding this prevents a large
    forum fan-out from mass-failing on Docker network setup under
    default-on egress isolation (which would yield zero forum posts and an
    empty cross-task distillation).

    ``max_concurrent_tasks`` defaults to 25, so full-default behavior
    is a 25-way forum; the follow-the-task-cap default binds whenever the user
    lowered ``--max-concurrent-tasks`` below that.
    """
    explicit = int(getattr(config, "max_concurrent_forum_tasks", 0) or 0)
    if explicit > 0:
        cap = explicit
    else:
        # Mirror the execution phase's guard (execution_phase.py): a
        # non-positive task cap (0 unset *or* a negative value) falls back
        # to 50, so a negative --max-concurrent-tasks can't reach
        # ThreadPoolExecutor(max_workers=-1) and crash the forum phase.
        task_cap = int(getattr(config, "max_concurrent_tasks", 0) or 0)
        cap = task_cap if task_cap > 0 else 50
    return min(max(1, pool_size), cap)


def _cross_task_coordinator_timeout_sec(forum_timeout_sec: float) -> float:
    """Coordinator's max wait for the R0->R1 barrier, bounded strictly below
    the container's own hard external kill deadline.

    Three timeouts govern a cross-task shared-container round; they must nest
    ``coord_timeout < poll_timeout <= container_timeout`` so the coordinator
    stops waiting before any agent's container gives up, and each container's
    own graceful R0-only fallback fires before it is externally killed.
    All three are derived from the same ``max(x - 15, 300)`` base (mirrored in
    ``container_host._build_runner_env``'s ``CONTAINER_TIMEOUT`` and
    ``container_host._maybe_setup_cross_task_r1``'s ``response_poll_timeout_ms``)
    so the constants can't drift apart across the two modules.

    Timeline for the default 900s ``cross_task_forum_timeout_sec``:

    * container hard-kill (CONTAINER_TIMEOUT) = ``max(900 - 15, 300)`` = 885s
    * in-container poll self-timeout          = ``max(885 - 5, 30)``   = 880s
    * coordinator wait (this value)           = ``max(885 - 30, 60)``  = 855s

    The 30s drain margin reserves time for the coordinator's own post-barrier
    work -- draining R0 events and building each agent's R1 prompt -- after the
    barrier resolves but before the containers are killed.
    """
    container_timeout_sec = max(
        forum_timeout_sec - 15, 300
    )  # mirror container_host._build_runner_env CONTAINER_TIMEOUT
    drain_margin_sec = 30  # reserve time to drain R0 + build R1 prompts after the barrier resolves
    return max(container_timeout_sec - drain_margin_sec, 60)


def _run_forum_round(
    *,
    round_num: int,
    generation: int,
    workers: int,
    debate_agents: list[Any],
    run_agent: Callable[[Any, int], Any],
    on_result: Callable[[Any, Any, dict[str, str], int], None],
    watcher_factory: Callable[[int, threading.Event, threading.Event], "_ForumEarlyExitWatcher | None"],
    record_agent_failure: Callable[[], None],
    future_failure_label: str,
    phase_display: str,
    all_failed_message: str,
    drain_round: Callable[[int], None],
) -> None:
    """Shared per-round forum skeleton for the per-task and cross-task phases.

    Runs one discussion round: spins up an early-exit watcher (via
    ``watcher_factory``, which returns ``None`` to disable early exit),
    dispatches one ``run_agent(agent, round_num)`` future per debate agent
    through a bounded :class:`ThreadPoolExecutor`, and processes futures in
    completion order. Each future's success payload is handed to
    ``on_result(agent, result, round_failures, round_num)``; a raised
    :class:`AuthenticationFailure` aborts the round, and any other exception is
    recorded as an agent failure. The watcher's ``stop_event`` is always set in
    the ``finally`` so the watcher thread exits regardless of early-exit state.
    After the pool drains, an all-agents-failed round raises
    :class:`ForumValidationError`; otherwise ``drain_round(round_num)`` flushes
    the round's ForumBus events into the KnowledgeStore.

    The three callers differ only in per-agent work (``run_agent``), per-future
    reduction (``on_result``), watcher construction (``watcher_factory``), the
    log/exception wording (``future_failure_label`` / ``phase_display`` /
    ``all_failed_message``), and the drain body (``drain_round``); the
    concurrency lifecycle is identical and lives here.
    """
    round_failures: dict[str, str] = {}
    # Spin up one watcher per round. The stop_event is set in the ``finally``
    # once all futures are drained, so the watcher exits cleanly regardless of
    # whether early-exit fired.
    stop_event = threading.Event()
    triggered_event = threading.Event()
    watcher = watcher_factory(round_num, stop_event, triggered_event)
    if watcher is not None:
        watcher.start()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(run_agent, agent, round_num): agent for agent in debate_agents}
            for fut in as_completed(future_map):
                agent = future_map[fut]
                try:
                    result = fut.result()
                    on_result(agent, result, round_failures, round_num)
                except AuthenticationFailure:
                    raise
                except Exception as exc:
                    round_failures[agent.id] = str(exc)
                    # A raised future is the same lost-perspective degradation
                    # as the runtime_meta.forum_error branch — count it too so
                    # the degradation counter isn't blind to crash/timeout-shaped failures.
                    record_agent_failure()
                    log.error(
                        "[ENGINE] %s future failed for agent=%s gen=%s round=%s: %s",
                        future_failure_label,
                        agent.id,
                        generation,
                        round_num,
                        exc,
                    )
    finally:
        stop_event.set()
        if watcher is not None:
            watcher.join(timeout=10.0)
            if triggered_event.is_set():
                log.info(
                    "[ENGINE] %s round=%d ended early on all-done signal (gen=%s)",
                    phase_display,
                    round_num,
                    generation,
                )
    if round_failures:
        log.warning(
            "[ENGINE] %s round=%d had %d/%d failed agent(s)",
            phase_display,
            round_num,
            len(round_failures),
            len(debate_agents),
        )
        if len(round_failures) >= len(debate_agents):
            raise ForumValidationError(
                all_failed_message,
                deficient_agents={agent_id: 1 for agent_id in round_failures},
            )
    drain_round(round_num)


class ForumValidationError(KcsiError, RuntimeError):
    """Forum round validation failure carrying per-agent deficit info."""

    def __init__(self, message: str, *, deficient_agents: dict[str, int] | None = None) -> None:
        super().__init__(message)
        self.deficient_agents: dict[str, int] = deficient_agents or {}


@dataclass(frozen=True)
class PerTaskForumPhaseInput:
    """Inputs required to run the per-task forum phase."""

    generation: int
    traces: "list[TaskTrace]"
    next_task_pool_size: int | None = None


@dataclass(frozen=True)
class CrossTaskForumPhaseInput:
    """Inputs required to run the cross-task forum phase."""

    generation: int
    traces: "list[TaskTrace]"


@dataclass(frozen=True)
class ForumCollaborators:
    """Explicit dependencies for the forum phase service bodies."""

    config: "GenerationConfig"
    persistence: "PersistenceObserver"
    runtime: "RuntimeExecutor"
    accumulator: "TokenAccumulator"
    knowledge: "KnowledgeStore | None"
    memory_store: "MemoryStore | None"
    agents: list[Any]  # read-only snapshot of engine.agents at build time
    record_phase_failure: Callable[..., None]
    maybe_embed: Callable[[str], list[float] | None]
    maybe_embed_batch: Callable[[list[str]], list[list[float] | None]]
    should_count_knowledge_drain_drop: Callable[[int, str], bool]
    non_holdout: Callable[[list[Any]], list[Any]]
    last_task_by_id: dict[str, Any]


@dataclass
class EngineForumPhaseService:
    """Engine-backed forum phase adapter.

    The concurrency-heavy forum bodies live here behind explicit service
    methods used by strategies and tests.
    """

    engine: "GenerationalOrchestrator"

    def _collaborators(self) -> ForumCollaborators:
        engine = self.engine
        return ForumCollaborators(
            config=engine.config,
            persistence=engine.persistence,
            runtime=engine.runtime,
            accumulator=engine.accumulator,
            knowledge=engine._knowledge,
            memory_store=engine._memory_store,
            agents=engine.agents,
            record_phase_failure=engine._record_knowledge_phase_failure,
            maybe_embed=engine._maybe_embed,
            maybe_embed_batch=engine._maybe_embed_batch,
            should_count_knowledge_drain_drop=engine._should_count_knowledge_drain_drop,
            non_holdout=engine._non_holdout,
            last_task_by_id=getattr(engine, "_last_task_by_id", {}),
        )

    def per_task_forum(self, phase_input: PerTaskForumPhaseInput) -> None:
        collab = self._collaborators()
        self._per_task_forum_default(phase_input, collab)

    def cross_task_forum(self, phase_input: CrossTaskForumPhaseInput) -> None:
        collab = self._collaborators()
        self._cross_task_forum_default(phase_input, collab)

    def _per_task_forum_default(self, phase_input: PerTaskForumPhaseInput, collab: ForumCollaborators) -> None:
        generation = phase_input.generation
        traces = phase_input.traces
        if collab.config.no_memory:
            log.info("[ENGINE] Skipping per-task discussion phase (--no-memory disables discussion phases)")
            return
        if collab.knowledge is None:
            log.info("[ENGINE] Skipping per-task discussion phase (knowledge DB not configured)")
            return
        per_task_rounds = int(collab.config.per_task_forum_rounds or 0)
        if per_task_rounds <= 0:
            log.info(
                "[ENGINE] Skipping per-task discussion phase (per_task_forum_rounds=%d)",
                per_task_rounds,
            )
            return
        # Hold-out probe tasks are excluded from learning: their traces never
        # enter the per-task discussion.
        traces = collab.non_holdout(traces)
        if not traces:
            return
        from ..memory.forum_bus import ForumBus
        from ..models import TaskSpec

        traces_by_agent: dict[str, list[TaskTrace]] = defaultdict(list)
        for t in traces:
            traces_by_agent[t.agent_id].append(t)
        # Post author's own task score, keyed by (task_id, agent_id), so the
        # per-task distiller can weight high-score authors over low-score
        # authors when their claims conflict (threaded into the drain below).
        native_score_by_task_agent: dict[tuple[str, str], float | None] = {
            (str(t.task_id), t.agent_id): t.native_score for t in traces
        }
        debate_agents = [agent for agent in collab.agents if traces_by_agent.get(agent.id)]
        if not debate_agents:
            log.info("[ENGINE] Skipping per-task discussion (no agent has task traces in generation %s)", generation)
            return

        # Phase 2 always runs in task-mode under the V2 design: agents produce
        # structured single-agent post-mortems (one JSON post per task) that
        # cross-reference prior-generation posts on the same task. The
        # historical monologue-skip optimization conflated "no
        # current-gen reply chains" with "no value", but cross-generation
        # threading was never measured and is now load-bearing for the V2
        # contrast pipeline.
        forum_bus = ForumBus(
            db_path=collab.config.knowledge_db_path,
            experiment=collab.config.experiment_name or "default",
            generation=generation,
        )
        # Start each generation forum with a clean bus to avoid stale events
        # from aborted runs.
        forum_bus.clear()

        def _run_forum_agent(agent: AgentState, round_num: int) -> tuple[TokenUsage, dict, str]:
            my_traces = traces_by_agent.get(agent.id, [])
            forum_task_ids = sorted({str(t.task_id).strip() for t in my_traces if str(t.task_id).strip()})
            # Collect task descriptions for this agent's tasks.
            task_descriptions: dict[str, str] = {}
            task_map = cast("dict[str, TaskSpec]", collab.last_task_by_id)
            for tid in forum_task_ids:
                task_obj = task_map.get(tid)
                if task_obj is not None and task_obj.prompt:
                    task_descriptions[tid] = task_obj.prompt

            # Fetch prior-generation posts for ALL of this agent's tasks so
            # the agent can thread replies across every task it owns.
            # V2: read per-task forum posts across ALL prior generations on
            # this task — agent reflects on its ancestors' discussions, not
            # just last gen. Chronological append-only order (no top/trailing
            # truncation) keeps the prompt prefix byte-stable across gens so
            # Anthropic / OpenAI prompt caching can hit. Per-item excerpt
            # caps (already in kcsi/forum/prompt.py) bound per-post size; total
            # growth at realistic scale (10 gens × handful posts/gen on this
            # task) stays well under context budget.
            prior_posts: list[dict] = []
            if generation > 0 and collab.knowledge is not None and forum_task_ids:
                deduped: dict = {}
                ordered_keys: list = []
                # One batched IN-query for all of this agent's tasks instead of
                # a query_task() per task — each acquired the store's process
                # RLock, so N tasks × M agents fully serialized concurrent forum
                # agents. query_tasks() returns byte-identical
                # per-task pages.
                try:
                    pages = collab.knowledge.query_tasks(
                        forum_task_ids,
                        generation=None,
                        entry_types=["post"],
                        limit=500,
                        experiment=collab.config.experiment_name,
                    )
                except Exception as exc:
                    log.warning(
                        "[ENGINE] Failed to load prior-gen posts for agent=%s: %s",
                        agent.id,
                        exc,
                    )
                    pages = {}
                for tid in forum_task_ids:
                    page = pages.get(tid) or {}
                    # query_tasks returns each task's rows ORDER BY id ASC, so
                    # iteration is chronological. Deduplicate preserving order.
                    for post in page.get("discussion") or []:
                        if not isinstance(post, dict):
                            continue
                        post_gen = post.get("generation")
                        if post_gen is None or int(post_gen) >= int(generation):
                            continue
                        key = post.get("id")
                        if key is None:
                            key = id(post)
                        if key not in deduped:
                            ordered_keys.append(key)
                        deduped[key] = post
                prior_posts = [deduped[k] for k in ordered_keys]

            # Fetch same-generation peer posts from rounds before this one,
            # on this agent's tasks — mirrors the cross-task forum's
            # peer_posts_this_gen (see run_cross_task_shared_container's
            # _fetch_peer_posts_this_gen). Round 0 has no earlier same-gen
            # round, so it's always []. Posts only become visible here once
            # the round's ForumBus drain has run (drain-per-round), so this is
            # a no-op until that lands for --per-task-forum-rounds > 1.
            peer_posts_this_gen: list[dict] = []
            if round_num > 0 and collab.knowledge is not None and forum_task_ids:
                try:
                    peer_pages = collab.knowledge.query_tasks(
                        forum_task_ids,
                        generation=generation,
                        entry_types=["post"],
                        limit=500,
                        experiment=collab.config.experiment_name,
                    )
                except Exception as exc:
                    log.warning(
                        "[ENGINE] Failed to load this-gen peer posts for agent=%s: %s",
                        agent.id,
                        exc,
                    )
                    peer_pages = {}
                for tid in forum_task_ids:
                    page = peer_pages.get(tid) or {}
                    for post in page.get("discussion") or []:
                        if not isinstance(post, dict):
                            continue
                        r_tag = post.get("round_num")
                        if r_tag is None or int(r_tag) >= int(round_num):
                            continue
                        peer_posts_this_gen.append(post)

            # Collect the agent's Phase-1 native memory for the task set.
            # runtime_meta["native_session_memory"] is populated during
            # Phase 1 (EngineExecutionPhaseService eval). We concatenate across task traces so
            # the forum prompt can reference the execution-time notes.
            #
            # To avoid the prompt builder's final 8000-char cap silently
            # dropping later task chunks for multi-task agents, we cap each
            # per-task chunk to an equal share of the budget BEFORE joining.
            # The prompt builder's final safety cap is preserved for edge
            # cases.
            per_chunk_cap = 8000 // max(1, len(forum_task_ids))
            native_mem_chunks: list[str] = []
            for _tr in my_traces:
                _rm = getattr(_tr, "runtime_meta", None) or {}
                _chunk = _rm.get("native_session_memory")
                if isinstance(_chunk, str) and _chunk.strip():
                    _cleaned = _chunk.strip()
                    if len(_cleaned) > per_chunk_cap:
                        _cleaned = _cleaned[:per_chunk_cap] + "\n...(truncated)"
                    native_mem_chunks.append(f"### task={_tr.task_id}\n{_cleaned}")
            native_memory = "\n\n".join(native_mem_chunks) if native_mem_chunks else None

            # Per-task discussion prompt (formerly R0). Includes prior-gen
            # posts (for threaded replies), same-generation peer posts from
            # earlier rounds this generation, and Phase-1 native memory.
            #
            # Built as a cache-stable split: cacheable_prefix (agent-,
            # generation-, and round-stable header / instructions / task
            # descriptions) vs variable_suffix (this-gen outcomes, prior
            # posts, peer posts, native memory). The full body is the
            # concatenation, surfaced to legacy adapters via task.prompt /
            # task_md_override; the cache-aware direct forum adapter
            # consumes the split fields via metadata to place cache_control
            # on the prefix only.
            prompt_parts = build_per_task_discussion_parts(
                agent_id=agent.id,
                generation=generation,
                traces=my_traces,
                task_ids=forum_task_ids,
                task_descriptions=task_descriptions,
                prior_gen_posts=prior_posts,
                native_memory=native_memory,
                round_num=round_num,
                peer_posts_this_gen=peer_posts_this_gen,
            )
            prompt = prompt_parts.as_text()
            forum_task = TaskSpec(
                id=f"__forum__g{generation}_r{round_num}_{agent.id}",
                repo="",
                prompt=prompt,
                metadata={
                    # Per-task forum containers. Historically tagged
                    # "forum_debate" (the original single-forum feature); the
                    # wire tag now matches the canonical phase name.
                    "task_source": "per_task_forum",
                    "task_md_override": prompt,
                    "forum_generation": generation,
                    "forum_round": round_num,
                    "forum_agent_id": agent.id,
                    "forum_expected_agents": len(debate_agents),
                    "forum_task_ids": forum_task_ids,
                    # Cache-stable split for the direct forum adapter.
                    "forum_cacheable_prefix": prompt_parts.cacheable_prefix,
                    "forum_variable_suffix": prompt_parts.variable_suffix,
                },
            )
            attempts = max(1, int(getattr(collab.config, "max_task_retries", 0) or 0) + 1)
            return _run_retryable_forum_task(
                run_once=lambda: collab.runtime.run_task(
                    generation=generation,
                    agent_id=agent.id,
                    task=forum_task,
                    agent_seed_package=copy.deepcopy(agent.seed_package),
                    experiment_name=collab.config.experiment_name,
                    spawn_method="task",
                    workstream_access_method="inline",
                ),
                generation=generation,
                agent_id=agent.id,
                phase_label=f"per-task discussion task {forum_task.id}",
                attempts=attempts,
                forum_bus=forum_bus,
                forum_round=round_num,
            )

        workers = _resolve_forum_worker_cap(collab.config, len(debate_agents))

        # Build the "expected to signal done" map: for each task a debate
        # agent owns this generation, the set of agent_ids that must signal
        # done on that task before the phase can exit early.  An empty map
        # disables early-exit for this phase (watcher will never trigger).
        expected_done: dict[str, set[str]] = defaultdict(set)
        for agent in debate_agents:
            for t in traces_by_agent.get(agent.id, []):
                tid = str(getattr(t, "task_id", "") or "").strip()
                if tid:
                    expected_done[tid].add(agent.id)
        early_exit_enabled = bool(collab.config.forum_early_exit) and bool(expected_done)
        # Loop over the configured per-task forum rounds so
        # ``--per-task-forum-rounds N`` is honoured (N>1 was previously a
        # dead knob — only round 0 ran).
        forum_token_usage: dict[str, TokenUsageDict] = {}
        forum_outputs: dict[str, str] = {}

        def _per_task_watcher_factory(
            round_num: int,
            stop_event: threading.Event,
            triggered_event: threading.Event,
        ) -> "_ForumEarlyExitWatcher | None":
            if not early_exit_enabled:
                return None
            return _ForumEarlyExitWatcher(
                forum_bus=forum_bus,
                expected=dict(expected_done),
                stop_event=stop_event,
                triggered_event=triggered_event,
                container_name_prefixes=[
                    _forum_container_prefix(
                        experiment_name=collab.config.experiment_name or "default",
                        generation=generation,
                        phase="per_task",
                    ),
                ],
                poll_interval_sec=float(collab.config.forum_early_exit_poll_sec),
                quorum_pct=float(collab.config.forum_early_exit_quorum_pct),
                quorum_grace_sec=float(collab.config.forum_early_exit_quorum_grace_sec),
                knowledge=collab.knowledge,
                experiment=collab.config.experiment_name,
                generation=generation,
                round_num=round_num,
                phase_label=f"per_task_r{round_num}",
            )

        def _on_per_task_result(
            agent: AgentState,
            result: Any,
            round_failures: dict[str, str],
            round_num: int,
        ) -> None:
            usage, runtime_meta, output_text = result
            collab.accumulator.record_lifecycle(
                generation,
                agent.id,
                f"forum_round_{round_num}",
                usage,
            )
            agent.token_usage += usage.total
            if runtime_meta.get("forum_error"):
                error_text = str(runtime_meta.get("forum_error") or "")
                round_failures[agent.id] = error_text
                collab.record_phase_failure(generation, "forum_agent_failures")
                collab.persistence.on_forum_message(
                    generation=generation,
                    round_num=round_num,
                    agent_id=agent.id,
                    message_type="error",
                    content_json={
                        "phase": "per_task_forum",
                        "error": error_text,
                        "error_type": str(runtime_meta.get("forum_error_type") or ""),
                    },
                    token_usage=usage.to_dict(),
                )
                return
            forum_token_usage[agent.id] = usage.to_dict()
            forum_outputs[agent.id] = output_text
            workspace_key = runtime_meta.get("workspace_key") or runtime_meta.get("group_folder")
            native = ""
            if isinstance(workspace_key, str) and workspace_key.strip():
                native = collect_native_session_memory(
                    workspace_key,
                    max_chars=collab.config.native_memory_max_chars,
                    max_files=collab.config.native_memory_max_files,
                    max_chars_per_file=collab.config.native_memory_max_chars_per_file,
                )
            if not native and isinstance(runtime_meta.get("native_session_memory"), str):
                native = runtime_meta.get("native_session_memory") or ""
            if native:
                collab.persistence.on_native_memory(
                    generation=generation,
                    agent_id=agent.id,
                    content=native,
                )

        def _drain_per_task_round(round_num: int) -> None:
            # Drain ForumBus events into KnowledgeStore INSIDE the per-round
            # loop (mirrors the cross-task forum's per-round drain below) so
            # a later round's failure cannot discard earlier rounds' real
            # discussion posts. Tag per-task forum posts with
            # source_phase="per_task_forum" so the distiller can
            # differentiate them from cross-task forum posts.
            if collab.knowledge is None:
                return
            try:
                drained = _drain_forum_bus(
                    forum_bus=forum_bus,
                    knowledge=collab.knowledge,
                    generation=generation,
                    experiment=collab.config.experiment_name,
                    source_phase="per_task_forum",
                    embed_fn=collab.maybe_embed,
                    batch_embed_fn=collab.maybe_embed_batch,
                    forum_store=collab.memory_store,
                    on_drop=lambda n: collab.record_phase_failure(generation, "drain_failures", n),
                    drop_dedupe_fn=lambda event_id: collab.should_count_knowledge_drain_drop(generation, event_id),
                    native_score_by_task_agent=native_score_by_task_agent,
                )
                log.info(
                    "[ENGINE] Drained %d ForumBus events into KnowledgeStore for gen=%s round=%s",
                    drained,
                    generation,
                    round_num,
                )
            except Exception as exc:
                # Match the cross-task drain handlers: a per-task drain
                # failure is otherwise swallowed by the Phase-3
                # try/except in the generation loop and looks identical
                # to a healthy run.
                log.warning(
                    "[ENGINE] per-task forum drain failed for gen=%s round=%s: %s",
                    generation,
                    round_num,
                    exc,
                )
                collab.record_phase_failure(generation, "drain_failures")

        for round_num in range(per_task_rounds):
            _run_forum_round(
                round_num=round_num,
                generation=generation,
                workers=workers,
                debate_agents=debate_agents,
                run_agent=_run_forum_agent,
                on_result=_on_per_task_result,
                watcher_factory=_per_task_watcher_factory,
                record_agent_failure=lambda: collab.record_phase_failure(generation, "forum_agent_failures"),
                future_failure_label="forum",
                phase_display="per-task discussion",
                all_failed_message="all per-task discussion agents failed",
                drain_round=_drain_per_task_round,
            )

    def run_cross_task_shared_container(
        self,
        *,
        generation: int,
        debate_agents: list[AgentState],
        forum_bus: Any,
        cross_task_history: list[dict],
        phase1_by_agent: dict[str, dict],
        cross_task_evidence_ids: list[str],
        expected_agents_set: set[str],
        workers: int,
    ) -> None:
        collab = self._collaborators()
        self._run_cross_task_shared_container_default(
            generation=generation,
            debate_agents=debate_agents,
            forum_bus=forum_bus,
            cross_task_history=cross_task_history,
            phase1_by_agent=phase1_by_agent,
            cross_task_evidence_ids=cross_task_evidence_ids,
            expected_agents_set=expected_agents_set,
            workers=workers,
            collab=collab,
        )

    def _run_cross_task_shared_container_default(
        self,
        *,
        generation: int,
        debate_agents: list[AgentState],
        forum_bus: Any,
        cross_task_history: list[dict],
        phase1_by_agent: dict[str, dict],
        cross_task_evidence_ids: list[str],
        expected_agents_set: set[str],
        workers: int,
        collab: ForumCollaborators,
    ) -> None:
        knowledge = collab.knowledge
        if knowledge is None:
            log.info("[ENGINE] Skipping cross-task shared-container path (knowledge DB not configured)")
            return
        from ..forum.prompt import build_cross_task_discussion_parts
        from ..memory.knowledge_store import CROSS_TASK_SENTINEL  # noqa: F401  # re-imported for closure
        from ..models import TaskSpec

        # Build a per-agent prompt builder for the R1 response (closure
        # over the latest drain state + history). The coordinator calls
        # this AFTER the on_drain callback runs, so peer_posts_this_gen
        # already reflects the freshly-drained round-0 bus.
        def _r1_prompt_for_agent(agent_id: str) -> str:
            phase1_context = phase1_by_agent.get(agent_id)
            peer_posts_this_gen: list[dict] = []
            try:
                page = knowledge.query_task(
                    CROSS_TASK_SENTINEL,
                    generation=generation,
                    entry_types=["post"],
                    limit=100_000,
                    experiment=collab.config.experiment_name,
                )
            except Exception as exc:
                log.warning(
                    "[cross_task_r1] failed to load this-gen peer posts: %s",
                    exc,
                )
                page = {"discussion": []}
            for post in page.get("discussion") or []:
                if not isinstance(post, dict):
                    continue
                r_tag = post.get("round_num")
                if r_tag is None or int(r_tag) >= 1:
                    # Round-1 prompt only includes posts from rounds < 1
                    # (i.e., round 0 — the freshly-drained set).
                    continue
                text = post.get("text", "")
                peer_posts_this_gen.append(
                    {
                        "id": post.get("id"),
                        "round_num": int(r_tag),
                        "agent_id": post.get("agent_id"),
                        "text": text,
                    }
                )
            r1_parts = build_cross_task_discussion_parts(
                agent_id=agent_id,
                generation=generation,
                round_num=1,
                phase1_context=phase1_context,
                cross_task_history=cross_task_history,
                peer_posts_this_gen=peer_posts_this_gen,
            )
            # Send the FULL round-1 prompt (prefix + suffix), not the suffix
            # alone. The round directive is ROUND-DEPENDENT and lives in the
            # prefix: round 0 says "post a single JSON object", round 1 says
            # "respond to peers" and introduces the `transfer_claim` field. The
            # R0 session only cached the round-0 directive, so a suffix-only R1
            # turn would omit the round-1 instructions and the agent would keep
            # following the round-0 directive. History is empty on this path,
            # so as_text() is just schema/protocol/round-1-directive +
            # this-round peer posts — cheap to re-send and correct.
            return r1_parts.as_text()

        def _drain_r0() -> None:
            try:
                drained = _drain_forum_bus(
                    forum_bus=forum_bus,
                    knowledge=knowledge,
                    generation=generation,
                    experiment=collab.config.experiment_name,
                    source_phase="cross_task_forum",
                    embed_fn=collab.maybe_embed,
                    batch_embed_fn=collab.maybe_embed_batch,
                    forum_store=collab.memory_store,
                    on_drop=lambda n: collab.record_phase_failure(generation, "drain_failures", n),
                    drop_dedupe_fn=lambda event_id: collab.should_count_knowledge_drain_drop(generation, event_id),
                )
                log.info(
                    "[cross_task_r1] coordinator drained %d cross-task R0 events for gen=%s",
                    drained,
                    generation,
                )
            except Exception as exc:
                log.warning(
                    "[cross_task_r1] coordinator drain failed for gen=%s: %s",
                    generation,
                    exc,
                )
                collab.record_phase_failure(generation, "drain_failures")

        # Coordinator timeout is bounded strictly below the cross-task
        # forum container's hard external kill deadline (with margin for the
        # coordinator's own post-barrier R0 drain + R1 prompt build), since
        # the coordinator's window is bounded by how long any agent's
        # container can run before its in-container poll gives up and ships
        # an R0-only envelope. See _cross_task_coordinator_timeout_sec.
        coord_timeout = _cross_task_coordinator_timeout_sec(float(collab.config.cross_task_forum_timeout_sec or 900))
        coordinator = _CrossTaskR1Coordinator(
            expected_agent_ids=[a.id for a in debate_agents],
            prompt_builder=_r1_prompt_for_agent,
            timeout_sec=coord_timeout,
            on_drain=_drain_r0,
        )
        coordinator.start()

        def _run_shared_agent(agent: AgentState) -> tuple[TokenUsage, dict, str]:
            phase1_context = phase1_by_agent.get(agent.id)
            # Round 0 prompt — same shape as legacy round-0 dispatch.
            prompt_parts = build_cross_task_discussion_parts(
                agent_id=agent.id,
                generation=generation,
                round_num=0,
                phase1_context=phase1_context,
                cross_task_history=cross_task_history,
                peer_posts_this_gen=[],
            )
            prompt = prompt_parts.as_text()
            forum_task = TaskSpec(
                id=f"__cross_task_forum__g{generation}_shared_{agent.id}",
                repo="",
                prompt=prompt,
                metadata={
                    "task_source": "cross_task_forum",
                    "task_md_override": prompt,
                    "forum_generation": generation,
                    "forum_round": 0,
                    "forum_agent_id": agent.id,
                    "forum_expected_agents": len(debate_agents),
                    "cross_task": True,
                    # §2.2 grounding: keep the shared-container path aligned
                    # with the legacy cross-task path so the MCP server can
                    # reject unknown evidence_task_ids in R0 and R1 posts.
                    "forum_task_ids": cross_task_evidence_ids,
                    "forum_cacheable_prefix": prompt_parts.cacheable_prefix,
                    "forum_variable_suffix": prompt_parts.variable_suffix,
                },
            )
            attempts = max(1, int(getattr(collab.config, "max_task_retries", 0) or 0) + 1)
            return _run_retryable_forum_task(
                run_once=lambda: collab.runtime.run_task(
                    generation=generation,
                    agent_id=agent.id,
                    task=forum_task,
                    agent_seed_package=copy.deepcopy(agent.seed_package),
                    experiment_name=collab.config.experiment_name,
                    spawn_method="task",
                    workstream_access_method="inline",
                    cross_task_shared_container=True,
                    cross_task_r1_callback=coordinator.on_sentinel,
                ),
                generation=generation,
                agent_id=agent.id,
                phase_label=f"cross-task shared-container task {forum_task.id}",
                attempts=attempts,
                forum_bus=forum_bus,
                forum_round=0,
            )

        round_failures: dict[str, str] = {}
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_map = {pool.submit(_run_shared_agent, agent): agent for agent in debate_agents}
                for fut in as_completed(future_map):
                    agent = future_map[fut]
                    try:
                        usage, runtime_meta, _output_text = fut.result()
                    except AuthenticationFailure:
                        raise
                    except Exception as exc:
                        round_failures[agent.id] = str(exc)
                        # See per-task handler: a raised future is the same
                        # lost-perspective degradation as forum_error.
                        collab.record_phase_failure(generation, "forum_agent_failures")
                        log.error(
                            "[ENGINE] cross-task shared-container future failed agent=%s gen=%s: %s",
                            agent.id,
                            generation,
                            exc,
                        )
                        continue
                    # Per-round token attribution. The container envelope
                    # carries per-round usage in
                    # ``cross_task_round_<n>_result.tokenUsage``; when
                    # those fields are absent (legacy / R0-only outputs),
                    # the entire ``usage`` aggregate goes to round 0.
                    r0_block = runtime_meta.get("cross_task_round_0_result") if isinstance(runtime_meta, dict) else None
                    r1_block = runtime_meta.get("cross_task_round_1_result") if isinstance(runtime_meta, dict) else None
                    r0_usage = _coerce_round_usage(r0_block) if r0_block else usage
                    r1_usage = _coerce_round_usage(r1_block) if r1_block else None
                    collab.accumulator.record_lifecycle(
                        generation,
                        agent.id,
                        "cross_task_forum_round_0",
                        r0_usage,
                    )
                    if r1_usage is not None:
                        collab.accumulator.record_lifecycle(
                            generation,
                            agent.id,
                            "cross_task_forum_round_1",
                            r1_usage,
                        )
                    agent.token_usage += usage.total
                    if isinstance(runtime_meta, dict) and runtime_meta.get("forum_error"):
                        error_text = str(runtime_meta.get("forum_error") or "")
                        round_failures[agent.id] = error_text
                        collab.record_phase_failure(generation, "forum_agent_failures")
                        collab.persistence.on_forum_message(
                            generation=generation,
                            round_num=0,
                            agent_id=agent.id,
                            message_type="error",
                            content_json={
                                "phase": "cross_task_forum",
                                "error": error_text,
                                "error_type": str(runtime_meta.get("forum_error_type") or ""),
                            },
                            token_usage=usage.to_dict(),
                        )
                    shared_meta = (
                        runtime_meta.get("cross_task_shared_container_meta") if isinstance(runtime_meta, dict) else None
                    )
                    if (
                        isinstance(shared_meta, dict)
                        and shared_meta.get("enabled") is True
                        and shared_meta.get("r1_captured") is not True
                    ):
                        note = str(shared_meta.get("note") or "R1 was not captured")
                        if shared_meta.get("timed_out") is True:
                            # Graceful degrade: the R0->R1 barrier simply never
                            # produced a response in time (documented contract —
                            # see _CrossTaskR1Coordinator's docstring and the TS
                            # "R1 absence is OK" comment). R0 already ran to
                            # completion; this is not a forum-agent failure.
                            log.info(
                                "[ENGINE] cross-task shared-container R1 skipped (barrier timeout) for agent=%s: %s",
                                agent.id,
                                note,
                            )
                        else:
                            error_text = f"cross-task shared-container skipped round 1: {note}"
                            round_failures[agent.id] = error_text
                            collab.record_phase_failure(generation, "forum_agent_failures")
                            collab.persistence.on_forum_message(
                                generation=generation,
                                round_num=1,
                                agent_id=agent.id,
                                message_type="error",
                                content_json={
                                    "phase": "cross_task_forum",
                                    "error": error_text,
                                    "error_type": "cross_task_r1_not_captured",
                                },
                                token_usage=usage.to_dict(),
                            )
        finally:
            coordinator.stop()

        if round_failures:
            log.warning(
                "[ENGINE] cross-task shared-container had %d/%d failed agent(s)",
                len(round_failures),
                len(debate_agents),
            )
            if len(round_failures) >= len(debate_agents):
                raise ForumValidationError(
                    "all cross-task forum agents failed",
                    deficient_agents={agent_id: 1 for agent_id in round_failures},
                )

        # Drain round-1 forum events. The coordinator already drained R0
        # mid-flight (via on_drain), so this final pass picks up R1's
        # forum_post / forum_signal_done events written after the
        # synthetic R1 user turn fired in-container.
        try:
            drained = _drain_forum_bus(
                forum_bus=forum_bus,
                knowledge=knowledge,
                generation=generation,
                experiment=collab.config.experiment_name,
                source_phase="cross_task_forum",
                embed_fn=collab.maybe_embed,
                batch_embed_fn=collab.maybe_embed_batch,
                forum_store=collab.memory_store,
                on_drop=lambda n: collab.record_phase_failure(generation, "drain_failures", n),
                drop_dedupe_fn=lambda event_id: collab.should_count_knowledge_drain_drop(generation, event_id),
            )
            log.info(
                "[cross_task_r1] post-R1 drained %d additional cross-task events for gen=%s",
                drained,
                generation,
            )
        except Exception as exc:
            log.warning(
                "[cross_task_r1] post-R1 drain failed for gen=%s: %s",
                generation,
                exc,
            )
            collab.record_phase_failure(generation, "drain_failures")

    def _cross_task_forum_default(self, phase_input: CrossTaskForumPhaseInput, collab: ForumCollaborators) -> None:
        generation = phase_input.generation
        traces = phase_input.traces
        if collab.config.no_memory:
            log.info("[ENGINE] Skipping cross-task discussion (--no-memory disables discussion phases)")
            return
        if collab.knowledge is None:
            log.info("[ENGINE] Skipping cross-task discussion (knowledge DB not configured)")
            return
        rounds = int(collab.config.cross_task_forum_rounds or 0)
        if rounds <= 0:
            log.info("[ENGINE] Skipping cross-task discussion (cross_task_forum_rounds<=0)")
            return
        # Hold-out probe tasks are excluded from learning: their traces never
        # feed the cross-task discussion (agent set, evidence ids, context).
        traces = collab.non_holdout(traces)
        if not traces:
            return

        from ..forum.prompt import build_cross_task_discussion_parts
        from ..memory.forum_bus import ForumBus
        from ..memory.knowledge_store import CROSS_TASK_SENTINEL
        from ..models import TaskSpec

        # One agent per unique agent_id that produced traces this generation.
        agent_ids = sorted({t.agent_id for t in traces if t.agent_id})
        agents_by_id = {a.id: a for a in collab.agents}
        debate_agents = [agents_by_id[aid] for aid in agent_ids if aid in agents_by_id]
        # Evidence-map for §2.2 grounding: every task_id that produced a trace
        # this generation is a valid evidence citation for cross-task posts.
        # Populating this lets `mcp_server.handle_forum_post`'s membership
        # check (`if allowed_task_ids:`) actually fire for cross-task rooms;
        # the prior unset path made the unknown-id rejection dead code.
        cross_task_evidence_ids = sorted({str(t.task_id).strip() for t in traces if str(t.task_id).strip()})
        if not debate_agents:
            log.info(
                "[ENGINE] Skipping cross-task discussion (no matching agents for gen %s)",
                generation,
            )
            return

        # V2: build phase1_context per agent — the Phase 1 trace gives the
        # agent its just-attempted task back as prompt context (Path A
        # simulation of container persistence). Reflection captured by
        # Phase 1 reflection step (item 1) is the rich-signal field.
        phase1_by_agent: dict[str, dict] = {}
        for tr in traces:
            aid = str(getattr(tr, "agent_id", "") or "")
            if not aid:
                continue
            runtime_meta = getattr(tr, "runtime_meta", None) or {}
            reflection = ""
            if isinstance(runtime_meta, dict):
                reflection = str(runtime_meta.get("phase1_reflection") or "").strip()
            phase1_by_agent[aid] = {
                "task_id": str(getattr(tr, "task_id", "") or ""),
                "native_score": getattr(tr, "native_score", None),
                "eval_result": getattr(tr, "eval_result", None) or {},
                "reflection": reflection,
            }

        # Cross-task forum agents see ONLY this-generation posts. Those arrive
        # via peer_posts_this_gen on rounds > 0; cross_task_history (prior-gen
        # history) is intentionally left empty. Cross-generation knowledge flows
        # forward through distillation -> seeding, not through raw forum history.
        # Dropping prior-gen history also removes the unbounded-growth 200K
        # overflow that motivated the render-aware history budget: this-gen posts
        # are bounded by (rounds-1) * num_agents.
        # Trade-off (memory horizon): that distillation -> seeding path is bounded
        # by the distiller's generation window (KCSI_CROSS_TASK_DISTILL_GEN_WINDOW,
        # default 6) and bundles are NOT re-distilled, so an insight older than the
        # window decays unless an agent re-surfaces it in a recent forum post — the
        # forum no longer provides the unbounded prior-gen backstop it once did.
        cross_task_history: list[dict] = []

        forum_bus = ForumBus(
            db_path=collab.config.knowledge_db_path,
            experiment=collab.config.experiment_name or "default",
            generation=generation,
        )
        # Truncate the shared JSONL before dispatching cross-task agents.
        # The per-task forum phase drains its events but does not truncate
        # the file; without this clear, read_events(after_seq=0) would
        # re-read every per-task post and persist duplicates under
        # source_phase="cross_task_forum", corrupting the distiller's view.
        forum_bus.clear()

        def _fetch_peer_posts_this_gen(current_round: int) -> list[dict]:
            """Pull this-gen cross-task posts from rounds before current_round.

            Used at round > 0 to give agents their peers' posts from earlier
            rounds. Drained between rounds so KS has the previous round's
            posts by the time round N+1's prompt builds (see drain inside
            the per-round loop below).
            """
            if current_round <= 0 or collab.knowledge is None:
                return []
            try:
                page = collab.knowledge.query_task(
                    CROSS_TASK_SENTINEL,
                    generation=generation,
                    entry_types=["post"],
                    limit=100_000,
                    experiment=collab.config.experiment_name,
                )
            except Exception as exc:
                log.warning("[ENGINE] failed to load this-gen peer posts: %s", exc)
                return []
            out: list[dict] = []
            for post in page.get("discussion") or []:
                if not isinstance(post, dict):
                    continue
                r_tag = post.get("round_num")
                if r_tag is None or int(r_tag) >= int(current_round):
                    continue
                text = post.get("text", "")
                out.append(
                    {
                        "id": post.get("id"),
                        "round_num": int(r_tag),
                        "agent_id": post.get("agent_id"),
                        "text": text,
                    }
                )
            return out

        # Per-round shared cache of this-gen peer posts. The page returned by
        # _fetch_peer_posts_this_gen is byte-identical for every agent in a
        # given round (it queries the fixed CROSS_TASK_SENTINEL page filtered
        # only by generation/round_num, with NO agent-specific predicate), so
        # having each concurrent agent re-issue it just serializes N identical
        # reads on the KnowledgeStore process RLock. We compute it ONCE per
        # round in the round-dispatch loop below (single-threaded, before the
        # ThreadPoolExecutor fan-out) and every agent reads the same list here.
        # Mirrors the per-task forum's cross-agent read batching.
        _peer_posts_by_round: dict[int, list[dict]] = {}

        def _run_cross_task_agent(agent: AgentState, round_num: int) -> tuple[TokenUsage, dict, str]:
            phase1_context = phase1_by_agent.get(agent.id)
            # Read the round's shared page (populated before dispatch). Fall
            # back to a fresh fetch if the key is somehow absent so behavior is
            # never worse than the pre-hoist per-agent query.
            if round_num in _peer_posts_by_round:
                peer_posts_this_gen = _peer_posts_by_round[round_num]
            else:
                peer_posts_this_gen = _fetch_peer_posts_this_gen(round_num)
            # Cache-stable split: prefix carries header, cross-task history,
            # round-specific instructions, MCP protocol; suffix carries
            # this-agent's phase1_context + this-round's peer posts.
            prompt_parts = build_cross_task_discussion_parts(
                agent_id=agent.id,
                generation=generation,
                round_num=round_num,
                phase1_context=phase1_context,
                cross_task_history=cross_task_history,
                peer_posts_this_gen=peer_posts_this_gen,
            )
            prompt = prompt_parts.as_text()
            forum_task = TaskSpec(
                id=f"__cross_task_forum__g{generation}_r{round_num}_{agent.id}",
                repo="",
                prompt=prompt,
                metadata={
                    "task_source": "cross_task_forum",
                    "task_md_override": prompt,
                    "forum_generation": generation,
                    "forum_round": round_num,
                    "forum_agent_id": agent.id,
                    "forum_expected_agents": len(debate_agents),
                    "cross_task": True,
                    # §2.2 grounding: populate the evidence map so the MCP
                    # server's unknown-id rejection check actually fires.
                    "forum_task_ids": cross_task_evidence_ids,
                    # Cache-stable split for the direct forum adapter.
                    "forum_cacheable_prefix": prompt_parts.cacheable_prefix,
                    "forum_variable_suffix": prompt_parts.variable_suffix,
                },
            )
            attempts = max(1, int(getattr(collab.config, "max_task_retries", 0) or 0) + 1)
            return _run_retryable_forum_task(
                run_once=lambda: collab.runtime.run_task(
                    generation=generation,
                    agent_id=agent.id,
                    task=forum_task,
                    agent_seed_package=copy.deepcopy(agent.seed_package),
                    experiment_name=collab.config.experiment_name,
                    spawn_method="task",
                    workstream_access_method="inline",
                ),
                generation=generation,
                agent_id=agent.id,
                phase_label=f"cross-task discussion task {forum_task.id}",
                attempts=attempts,
                forum_bus=forum_bus,
                forum_round=round_num,
            )

        workers = _resolve_forum_worker_cap(collab.config, len(debate_agents))
        expected_agents_set = {agent.id for agent in debate_agents}
        early_exit_enabled = bool(collab.config.forum_early_exit) and bool(expected_agents_set)

        # Shared-container R0->R1 fast path. When the feature flag is on
        # and we have at least 2 rounds configured, run a single dispatch
        # per agent that covers BOTH round 0 and round 1 inside the same
        # SDK / Anthropic-Messages-API session (see
        # ``runtime_runner/agent-runner/src/anthropic_direct_forum.ts``).
        # The host coordinates the R0->R1 transition via
        # :class:`_CrossTaskR1Coordinator` + per-agent BarrierWatcher.
        # Rounds beyond round 1 still take the legacy multi-dispatch path.
        shared_container_enabled = bool(collab.config.cross_task_shared_container) and rounds >= 2
        if shared_container_enabled:
            self.run_cross_task_shared_container(
                generation=generation,
                debate_agents=debate_agents,
                forum_bus=forum_bus,
                cross_task_history=cross_task_history,
                phase1_by_agent=phase1_by_agent,
                cross_task_evidence_ids=cross_task_evidence_ids,
                expected_agents_set=expected_agents_set,
                workers=workers,
            )
            # Run any remaining rounds (>= 2) via the legacy per-round
            # dispatch path. The shared-container handled rounds 0 and 1.
            if rounds <= 2:
                return

        def _cross_task_watcher_factory(
            round_num: int,
            stop_event: threading.Event,
            triggered_event: threading.Event,
        ) -> "_ForumEarlyExitWatcher | None":
            if not early_exit_enabled:
                return None
            return _ForumEarlyExitWatcher(
                forum_bus=forum_bus,
                expected_agents=expected_agents_set,
                agent_only=True,
                stop_event=stop_event,
                triggered_event=triggered_event,
                container_name_prefixes=[
                    _forum_container_prefix(
                        experiment_name=collab.config.experiment_name or "default",
                        generation=generation,
                        phase="cross_task",
                    ),
                ],
                poll_interval_sec=float(collab.config.forum_early_exit_poll_sec),
                quorum_pct=float(collab.config.forum_early_exit_quorum_pct),
                quorum_grace_sec=float(collab.config.forum_early_exit_quorum_grace_sec),
                knowledge=collab.knowledge,
                experiment=collab.config.experiment_name,
                generation=generation,
                round_num=round_num,
                phase_label=f"cross_task_r{round_num}",
            )

        def _on_cross_task_result(
            agent: AgentState,
            result: Any,
            round_failures: dict[str, str],
            round_num: int,
        ) -> None:
            usage, _runtime_meta, _output_text = result
            collab.accumulator.record_lifecycle(
                generation,
                agent.id,
                f"cross_task_forum_round_{round_num}",
                usage,
            )
            agent.token_usage += usage.total
            if _runtime_meta.get("forum_error"):
                error_text = str(_runtime_meta.get("forum_error") or "")
                round_failures[agent.id] = error_text
                collab.record_phase_failure(generation, "forum_agent_failures")
                collab.persistence.on_forum_message(
                    generation=generation,
                    round_num=round_num,
                    agent_id=agent.id,
                    message_type="error",
                    content_json={
                        "phase": "cross_task_forum",
                        "error": error_text,
                        "error_type": str(_runtime_meta.get("forum_error_type") or ""),
                    },
                    token_usage=usage.to_dict(),
                )

        def _drain_cross_task_round(round_num: int) -> None:
            # V2: drain INSIDE the per-round loop so round N+1 prompts
            # can find round-N posts in KnowledgeStore. Pre-V2 drained
            # only after all rounds — fine for the legacy 1-round
            # default but breaks multi-round Phase 3 since
            # query_task(CROSS_TASK_SENTINEL, generation=None) only
            # returns drained content. Drains land under
            # CROSS_TASK_SENTINEL / source_phase=cross_task_forum.
            if collab.knowledge is None:
                return
            try:
                drained = _drain_forum_bus(
                    forum_bus=forum_bus,
                    knowledge=collab.knowledge,
                    generation=generation,
                    experiment=collab.config.experiment_name,
                    source_phase="cross_task_forum",
                    embed_fn=collab.maybe_embed,
                    batch_embed_fn=collab.maybe_embed_batch,
                    forum_store=collab.memory_store,
                    on_drop=lambda n: collab.record_phase_failure(generation, "drain_failures", n),
                    drop_dedupe_fn=lambda event_id: collab.should_count_knowledge_drain_drop(generation, event_id),
                )
                log.info(
                    "[ENGINE] Drained %d cross-task ForumBus events for gen=%s round=%s",
                    drained,
                    generation,
                    round_num,
                )
            except Exception as exc:
                log.warning(
                    "[ENGINE] cross-task drain failed for gen=%s round=%s: %s",
                    generation,
                    round_num,
                    exc,
                )
                collab.record_phase_failure(generation, "drain_failures")

        start_round = 2 if shared_container_enabled else 0
        for round_num in range(start_round, rounds):
            # Hoist the agent-independent peer-posts read out of the per-agent
            # worker: compute it ONCE here (single-threaded, before dispatch) so
            # the round's N agents share one page instead of each re-issuing the
            # identical CROSS_TASK_SENTINEL query under the KnowledgeStore RLock.
            # No-op query for round 0 (guarded inside _fetch_peer_posts_this_gen).
            _peer_posts_by_round[round_num] = _fetch_peer_posts_this_gen(round_num)
            _run_forum_round(
                round_num=round_num,
                generation=generation,
                workers=workers,
                debate_agents=debate_agents,
                run_agent=_run_cross_task_agent,
                on_result=_on_cross_task_result,
                watcher_factory=_cross_task_watcher_factory,
                record_agent_failure=lambda: collab.record_phase_failure(generation, "forum_agent_failures"),
                future_failure_label="cross-task forum",
                phase_display="cross-task discussion",
                all_failed_message="all cross-task forum agents failed",
                drain_round=_drain_cross_task_round,
            )

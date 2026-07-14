"""Forum-phase runtime machinery extracted from :mod:`ksi.orchestrator.engine`.

This module holds the module-level forum
helpers and thread classes that drive the per-task and cross-task discussion
phases -- draining the ForumBus into the KnowledgeStore, the early-exit watcher,
the cross-task R0->R1 barrier coordinator, and the retryable forum-task runner.

Extracted verbatim from ``engine``. ``engine`` re-imports the public-to-it names
(``_drain_forum_bus``, ``_ForumEarlyExitWatcher``, ``_CrossTaskR1Coordinator``,
``_run_retryable_forum_task``, ``_forum_container_prefix``, ``_coerce_post_ref``)
so existing call sites and ``from ksi.orchestrator.engine import
_drain_forum_bus`` keep working.

This module imports its shared retry/token helpers from :mod:`.task_retry` and
otherwise only from ``..errors``, ``..runtime`` and ``..tokens`` -- never from
``engine`` -- so there is no import cycle.
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable

from ..errors import AuthenticationFailure
from ..errors import is_auth_error as _is_auth_error
from ..runtime import RuntimeResult
from ..runtime.normalize import SilentAgentRuntimeError
from ..tokens import TokenUsage
from .task_retry import (
    _accumulate_failed_attempt_tokens,
    _is_retryable_task_error,
    _runtime_retry_meta,
)

if TYPE_CHECKING:
    from ..memory.forum_bus import ForumBus
    from ..memory.knowledge_store import KnowledgeStore
    from ..runtime.barrier import BarrierEvent

log = logging.getLogger(__name__)


def _coerce_round_usage(round_block: Any) -> TokenUsage:
    """Extract a :class:`TokenUsage` from a ``CrossTaskRoundResult`` dict.

    The container envelope ships the per-round token usage as a dict
    under ``round_block.tokenUsage`` with the four standard keys. We
    coerce defensively: missing keys default to 0, non-numeric values
    silently drop. Always returns a :class:`TokenUsage` so the caller
    never has to None-check.
    """
    if not isinstance(round_block, dict):
        return TokenUsage()
    usage_dict = round_block.get("tokenUsage")
    if not isinstance(usage_dict, dict):
        return TokenUsage()

    def _i(key: str) -> int:
        try:
            return int(usage_dict.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0

    return TokenUsage(
        input_tokens=_i("input_tokens"),
        output_tokens=_i("output_tokens"),
        cache_creation_input_tokens=_i("cache_creation_input_tokens"),
        cache_read_input_tokens=_i("cache_read_input_tokens"),
    )


def _coerce_post_ref(value: Any) -> int | None:
    """Normalize a parent/reply-to post reference to ``int`` or ``None``.

    Forum tool calls occasionally send the literal string ``"null"``/``"none"``
    (from LLM-rendered JSON) or a quoted integer like ``"5"`` instead of a
    real integer.  ``parent_id``/``reply_to`` are INTEGER columns in the
    ``knowledge`` table, but SQLite's type affinity silently accepts any
    value -- so without coercion these junk strings land in the DB and
    orphan the thread (no join will ever resolve a TEXT 'null' to a row id).
    """
    if value is None:
        return None
    if isinstance(value, bool):  # bool subclasses int; reject explicitly
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value or value != value:  # 0.0 or NaN → drop
            return None
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"null", "none", "nil", "undefined"}:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _drain_forum_bus(
    *,
    forum_bus: "ForumBus",
    knowledge: "KnowledgeStore",
    generation: int,
    experiment: str | None = None,
    source_phase: str | None = None,
    embed_fn: Callable[[str], list[float] | None] | None = None,
    batch_embed_fn: Callable[[list[str]], list[list[float] | None]] | None = None,
    forum_store: Any | None = None,
    on_drop: Callable[[int], None] | None = None,
    drop_dedupe_fn: Callable[[str], bool] | None = None,
    native_score_by_task_agent: dict[tuple[str, str], float | None] | None = None,
) -> int:
    """Drain ForumBus JSONL events into KnowledgeStore.

    Called by the orchestrator after R0 discussion completes.
    Returns the number of entries drained.

    ``on_drop`` is invoked once at the end with the number of drain
    degradations: knowledge-row events (insight/post/comment) whose
    KnowledgeStore write raised and was swallowed, plus stale-sidecar read
    failures that force stale-event filtering off. Lets the caller surface a
    partial drain as a degraded generation instead of it reading as
    healthy. Sidecar (``forum_store``) and ``signal_done`` write failures are
    best-effort by design and are NOT counted here.

    ``drop_dedupe_fn`` may return ``False`` for a failed knowledge-row event id
    that was already counted by an earlier drain. The event is still retried on
    later drains; only the health counter is deduped.

    ``source_phase`` overrides the default ``source_phase`` recorded on
    post entries (insights always use the store default). Pass
    ``"per_task_forum"`` or ``"cross_task_forum"`` to tag phase-of-origin
    so the distiller can query each phase's posts separately.

    ``native_score_by_task_agent`` maps ``(task_id, agent_id)`` to the post
    author's own task score this generation. When provided, each post is
    recorded with the author's score so the per-task distiller can weight
    high-score authors over low-score authors when claims conflict
    (``distiller._load_per_task_posts`` reads it back via ``query_task``).
    Cross-task drains leave it ``None`` (their ``__forum__`` posts have no
    single per-task score).

    ``embed_fn`` optionally maps post/insight text to an embedding vector
    that gets stored in ``knowledge_vec`` alongside the row so semantic
    search can retrieve these entries.  ``None`` from the fn (or no fn at
    all) is treated as "skip embedding for this row" — the main row is
    still written.

    ``batch_embed_fn`` is the preferred path: it receives a single list
    of texts and returns the parallel list of embeddings (or ``None`` per
    text). The drain calls it ONCE per drain after dedup/stale filtering,
    so a single ``SentenceTransformer.encode([...])`` invocation amortizes
    model overhead across all events. If both ``batch_embed_fn`` and
    ``embed_fn`` are provided, the batch path is used. Per-event fallback
    via ``embed_fn`` is preserved for callers that don't supply a batch fn.
    """
    events = forum_bus.read_events()
    dropped = 0  # drain degradations swallowed by this best-effort path

    def _count_failed_event_once(event_id: str) -> bool:
        if not event_id or drop_dedupe_fn is None:
            return True
        try:
            return bool(drop_dedupe_fn(event_id))
        except Exception:
            log.warning("[ENGINE] forum drain drop_dedupe_fn failed for event=%s", event_id, exc_info=True)
            return True

    # ``stale_event_ids`` are events written by failed forum-task retry
    # attempts. Their content is non-deterministic across
    # attempts (LLM ``temperature>0``), so the existing bulk_has_external_ids
    # dedup catches nothing — the retry's events have fresh fb-uuids.
    # Skipping by event_id at drain time is the single chokepoint that
    # keeps duplicates out of every downstream surface (knowledge rows,
    # forum_messages audit table, discussion_done).
    try:
        stale_event_ids = forum_bus.read_stale_event_ids()
    except Exception:
        # Read errors must not abort the drain — fall back to empty set
        # (pre-fix behavior) so a corrupt sidecar can't lose real data.
        stale_event_ids = set()
        dropped += 1
        log.warning("[ENGINE] forum_bus.read_stale_event_ids failed; skipping stale-skip", exc_info=True)
    # Bulk-prefetch existing external IDs to avoid per-event SQL round-trips.
    candidate_ids = [
        str(getattr(e, "event_id", "") or "").strip()
        for e in events
        if getattr(e, "message_type", "") in {"insight", "post", "comment"}
    ]
    candidate_ids = [eid for eid in candidate_ids if eid]
    existing_external_ids = (
        knowledge.bulk_has_external_ids(candidate_ids, experiment=experiment) if candidate_ids else set()
    )

    # Batch-embed: collect texts for all events that will actually be
    # persisted (post/insight/comment with non-empty text, not stale, not
    # already ingested) and call `batch_embed_fn` ONCE. Each per-event
    # `SentenceTransformer.encode([text])` historically cost 20–80ms of
    # single-threaded CPU; for a 50-task forum with 100 posts/round/task
    # = 5000 events, that's 100–400s of serialized embed time on the
    # drain path before the next round can start. Batching collapses
    # that to a single encode call.
    embedding_by_event_id: dict[str, list[float] | None] = {}
    if batch_embed_fn is not None:
        embeddable_pairs: list[tuple[str, str]] = []
        for ev in events:
            mtype = getattr(ev, "message_type", "")
            if mtype not in {"insight", "post", "comment"}:
                continue
            eid = str(getattr(ev, "event_id", "") or "").strip()
            if not eid:
                continue
            if eid in stale_event_ids:
                continue
            if eid in existing_external_ids:
                continue
            content = ev.content if isinstance(ev.content, dict) else {}
            text = str(content.get("text", "") or "")
            if not text.strip():
                continue
            embeddable_pairs.append((eid, text))
        if embeddable_pairs:
            try:
                texts_to_embed = [t for _, t in embeddable_pairs]
                vectors = batch_embed_fn(texts_to_embed)
                if isinstance(vectors, list) and len(vectors) == len(embeddable_pairs):
                    for (eid, _), vec in zip(embeddable_pairs, vectors):
                        embedding_by_event_id[eid] = vec
                else:
                    log.warning(
                        "[ENGINE] batch_embed_fn returned %s items for %s texts; "
                        "skipping batch embeddings for this drain",
                        "non-list" if not isinstance(vectors, list) else len(vectors),
                        len(embeddable_pairs),
                    )
            except Exception:
                log.warning(
                    "[ENGINE] batch_embed_fn raised; falling back to per-event embed_fn",
                    exc_info=True,
                )

    count = 0
    # Collect EVERY knowledge-DB write (insight/post/comment rows
    # AND ``done``/signal_done control rows) and apply them in a SINGLE
    # batched transaction via run_drain_batch, rather than one writer-queue
    # round-trip + COMMIT fsync per event. ``forum_store`` sidecar writes
    # target a different DB, so they remain best-effort per event.
    #
    # ``ops`` are zero-arg closures; ``op_meta`` is aligned 1:1 and tags each
    # op so the post-batch pass can do the right accounting:
    #   ("row", event_id)         -> counts toward count/dropped
    #   ("done", group_idx, tid)  -> control row: log only, never counted
    # ``done_groups`` tracks per-``done``-event tallies for the summary log.
    ops: list[Callable[[], Any]] = []
    op_meta: list[tuple] = []
    done_groups: list[dict] = []
    for event in events:
        event_round = int(getattr(event, "round_num", 0) or 0)
        event_id = str(getattr(event, "event_id", "") or "").strip()
        if event_id and event_id in stale_event_ids:
            continue
        if event.message_type in {"insight", "post", "comment"} and event_id:
            if event_id in existing_external_ids:
                continue
        if forum_store is not None:
            try:
                forum_store.insert_forum_message(
                    generation=generation,
                    agent_id=event.agent_id,
                    message_type=event.message_type,
                    content=event.content if isinstance(event.content, dict) else {},
                    round_num=event_round,
                    experiment=experiment,
                )
            except Exception:
                log.warning(
                    "[ENGINE] raw forum_event persist failed event=%s type=%s",
                    getattr(event, "event_id", "?"),
                    event.message_type,
                    exc_info=True,
                )
        try:
            if event.message_type == "insight":
                content = event.content or {}
                evidence = content.get("evidence_task_ids") or []
                task_id = evidence[0] if evidence else "__forum__"
                text = content.get("text", "")
                # Prefer the batch-embedded vector when available; fall
                # back to per-event embed_fn for backward compatibility
                # with callers that didn't supply batch_embed_fn.
                if event_id and event_id in embedding_by_event_id:
                    embedding = embedding_by_event_id[event_id]
                elif embed_fn and text:
                    embedding = embed_fn(text)
                else:
                    embedding = None
                insight_kwargs = dict(
                    task_id=task_id,
                    agent_id=event.agent_id,
                    generation=generation,
                    text=text,
                    scope=content.get("scope", "task"),
                    confidence=content.get("confidence", "medium"),
                    evidence_task_ids=evidence,
                    round_num=event_round,
                    experiment=experiment,
                    embedding=embedding,
                    external_id=event_id or None,
                )
                ops.append(lambda kw=insight_kwargs: knowledge._record_insight_locked(**kw))
                op_meta.append(("row", event_id))
            elif event.message_type in ("post", "comment"):
                content = event.content if isinstance(event.content, dict) else {}
                task_id = content.get("task_id") or "__forum__"
                # Cross-task forum posts MUST land under CROSS_TASK_SENTINEL so
                # the cross-task distiller (which reads
                # ``query_task(CROSS_TASK_SENTINEL)``) finds them — matching the
                # documented drain contract ("Drains land under
                # CROSS_TASK_SENTINEL", forum_phase.py) and the done-signal path
                # below, which already forces the sentinel. Otherwise an agent
                # that ignores the "post under __cross_task__" instruction (seen
                # with gpt-5.4-mini, which posts under the real task_id) has its
                # cross-task posts silently excluded from distillation.
                if source_phase == "cross_task_forum":
                    from ..memory.knowledge_store import CROSS_TASK_SENTINEL

                    task_id = CROSS_TASK_SENTINEL
                # Coerce both reference fields so junk inputs (string "null",
                # quoted integers) never pollute the INTEGER columns in
                # ``knowledge.parent_id``/``reply_to``.
                reply_to = _coerce_post_ref(content.get("reply_to"))
                parent_id = _coerce_post_ref(content.get("parent_post_id"))
                text = content.get("text", "")
                if event_id and event_id in embedding_by_event_id:
                    embedding = embedding_by_event_id[event_id]
                elif embed_fn and text:
                    embedding = embed_fn(text)
                else:
                    embedding = None
                post_kwargs = dict(
                    task_id=task_id,
                    agent_id=event.agent_id,
                    generation=generation,
                    text=text,
                    parent_id=parent_id,
                    round_num=event_round,
                    experiment=experiment,
                    reply_to=reply_to,
                    embedding=embedding,
                )
                if native_score_by_task_agent is not None:
                    post_kwargs["native_score"] = native_score_by_task_agent.get((task_id, event.agent_id))
                if source_phase is not None:
                    post_kwargs["source_phase"] = source_phase
                if event_id:
                    post_kwargs["external_id"] = event_id
                ops.append(lambda kw=post_kwargs: knowledge._record_post_locked(**kw))
                op_meta.append(("row", event_id))
            elif event.message_type == "done":
                # ``forum_signal_done`` events are control signals, not
                # knowledge rows — but we still want to land them in
                # ``discussion_done`` so audits can tell "agent finished
                # early" apart from "agent timed out".  Containers run
                # KnowledgeStore in read-only mode, so the persisted
                # writes happen here during the host-side drain.
                #
                # Per-task signals carry ``content.task_ids=[tid, ...]``; the
                # cross-task forum room carries an empty list (agents in the
                # cross-task room don't have a scoped ``FORUM_TASK_IDS``).
                # Both need to land in ``discussion_done`` — cross-task rows
                # are written under ``CROSS_TASK_SENTINEL`` so analytics can
                # distinguish them from per-task signals.
                from ..memory.knowledge_store import CROSS_TASK_SENTINEL

                content = event.content if isinstance(event.content, dict) else {}
                task_ids = content.get("task_ids") or []
                if not isinstance(task_ids, list):
                    task_ids = []
                is_cross_task = source_phase == "cross_task_forum"
                normalized = [str(t).strip() for t in task_ids if str(t).strip()]
                if is_cross_task:
                    targets = [CROSS_TASK_SENTINEL]
                else:
                    targets = normalized if normalized else [CROSS_TASK_SENTINEL]
                # Fold each signal_done into the shared batch (one txn for the
                # whole drain). discussion_done writes are best-effort control
                # rows: a failing target rolls back to its own SAVEPOINT and is
                # logged but never counted in ``dropped`` (which only tracks
                # knowledge rows). ``persisted`` is tallied post-batch.
                group_idx = len(done_groups)
                done_groups.append({"agent_id": event.agent_id, "n_targets": len(targets), "persisted": 0})
                for tid in targets:
                    ops.append(
                        lambda tid=tid, aid=event.agent_id: knowledge._signal_done_locked(
                            task_id=tid,
                            agent_id=aid,
                            generation=generation,
                            experiment=experiment,
                        )
                    )
                    op_meta.append(("done", group_idx, tid))
        except Exception:
            # Building a knowledge-row op (coercion / per-event embed) raised:
            # the event never reaches the batch, so count it as a drop here so
            # a partial drain stays visible; control/sidecar events
            # (done, forum_store) are best-effort and not counted.
            if getattr(event, "message_type", "") in {"insight", "post", "comment"}:
                if _count_failed_event_once(event_id):
                    dropped += 1
            log.warning(
                "[ENGINE] Failed to drain ForumBus event %s: %s",
                getattr(event, "event_id", "?"),
                event.message_type,
                exc_info=True,
            )

    # Apply every collected knowledge-DB write in ONE transaction. Each op
    # runs inside its own SAVEPOINT, so a single failing event rolls back to
    # its savepoint and the rest still commit (partial-drain preserved).
    if ops:
        try:
            # experiment + generation are constant across every op, so the store
            # resolves the run/generation refs once instead of per event.
            results = knowledge.run_drain_batch(ops, experiment=experiment, generation=generation)
        except Exception:
            # Whole-batch infra failure (lock/commit): no rows landed. Count
            # each row event as dropped so the degradation is visible.
            log.warning(
                "[ENGINE] forum drain batch failed; %d event(s) not persisted",
                len(ops),
                exc_info=True,
            )
            results = [(False, None)] * len(ops)
        for (ok, _res), meta in zip(results, op_meta):
            if meta[0] == "row":
                eid = meta[1]
                if ok:
                    count += 1
                else:
                    if _count_failed_event_once(eid):
                        dropped += 1
                    log.warning(
                        "[ENGINE] Failed to drain ForumBus event %s: knowledge-row write failed",
                        eid or "?",
                    )
            else:  # ("done", group_idx, tid) — control row, log only
                _, group_idx, tid = meta
                if ok:
                    done_groups[group_idx]["persisted"] += 1
                else:
                    log.warning(
                        "[ENGINE] signal_done persist failed agent=%s task=%s gen=%s",
                        done_groups[group_idx]["agent_id"],
                        tid,
                        generation,
                    )
        for group in done_groups:
            log.info(
                "[ENGINE] Drained forum_signal_done agent=%s gen=%s task_count=%d persisted=%d",
                group["agent_id"],
                generation,
                group["n_targets"],
                group["persisted"],
            )

    if dropped and on_drop is not None:
        on_drop(dropped)
    return count


# ---------------------------------------------------------------------------
# Discussion-phase early-exit helpers
#
# The write path for ``forum_signal_done`` persists agents' signal
# events to both the ForumBus JSONL and the ``discussion_done``
# table).  The helpers below read that signal during a live discussion phase and
# short-circuit the container timeout when every expected agent has signalled
# done.  The existing ``--forum-timeout-sec`` / ``--cross-task-forum-timeout-sec``
# remain the hard cap -- the watcher only ends discussion phases *early*, never
# late.
# ---------------------------------------------------------------------------


def _read_done_signals(
    forum_bus: "ForumBus",
    round_num: int | None = None,
) -> dict[str, set[str]]:
    """Return a mapping ``{task_id: {agent_id, ...}}`` of observed done signals.

    Reads every ``done`` event from the ForumBus JSONL (optionally constrained by
    round) and expands each event's ``content.task_ids`` list into per-task done
    expands each event's ``content.task_ids`` list into per-task done sets.
    Used by the early-exit watcher to decide when a discussion phase can end.
    """
    done_events = forum_bus.read_events(
        after_seq=0,
        round_num=round_num,
        message_types={"done"},
    )
    by_task: dict[str, set[str]] = defaultdict(set)
    for ev in done_events:
        if round_num is not None and getattr(ev, "round_num", None) is None:
            continue
        agent_id = str(getattr(ev, "agent_id", "") or "").strip()
        if not agent_id:
            continue
        content = ev.content if isinstance(ev.content, dict) else {}
        task_ids = content.get("task_ids") or []
        if not isinstance(task_ids, list):
            continue
        for tid in task_ids:
            key = str(tid).strip()
            if key:
                by_task[key].add(agent_id)
    return by_task


def _read_done_agent_ids(
    forum_bus: "ForumBus",
    round_num: int | None = None,
) -> set[str]:
    """Return the set of agent_ids that have emitted any ``done`` event.

    Used by cross-task early-exit: the cross-task ``forum_signal_done``
    call carries an empty ``task_ids`` payload because the tool's done-event
    semantics are room-scoped, not task-scoped (FORUM_TASK_IDS is populated
    for cross-task rooms now, but it drives
    the §2.2 grounding check for ``forum_post``, not the done-event content).
    Any ``done`` event from an agent is treated as "that agent is done with
    the cross-task room."
    """
    events = forum_bus.read_events(
        after_seq=0,
        round_num=round_num,
        message_types={"done"},
    )
    return {
        str(getattr(ev, "agent_id", "") or "").strip()
        for ev in events
        if str(getattr(ev, "agent_id", "") or "").strip()
        and (round_num is None or getattr(ev, "round_num", None) is not None)
    }


def _all_expected_signalled(
    *,
    forum_bus: "ForumBus",
    expected: dict[str, set[str]],
    agent_only: bool = False,
    expected_agents: set[str] | None = None,
    round_num: int | None = None,
) -> bool:
    """True iff every expected signal has been observed on the ForumBus.

    Two modes:
    - ``agent_only=False`` (default, per-task): ``expected`` maps a task_id
      to the set of agent_ids that must signal done on that task. Matched
      against the per-task ``done`` event payloads.
    - ``agent_only=True`` (cross-task): every agent in ``expected_agents`` must
      emit any ``done`` event. ``expected`` is ignored in this branch.
    """
    if agent_only:
        agents = expected_agents or set()
        if not agents:
            return False
        observed = _read_done_agent_ids(forum_bus, round_num=round_num)
        return agents.issubset(observed)
    if not expected:
        return False
    observed_by_task = _read_done_signals(forum_bus, round_num=round_num)
    for task_id, agents in expected.items():
        if not agents:
            continue
        if not agents.issubset(observed_by_task.get(task_id, set())):
            return False
    return True


def _stop_forum_containers(container_name_substrings: list[str]) -> int:
    """Best-effort ``docker stop`` for containers whose names match any substring.

    ``container_name_substrings`` are matched against ``docker ps --format
    '{{.Names}}'`` output.  Returns the number of containers stopped (0 if
    docker is unavailable or no matches).  Mirrors the per-experiment cleanup
    pattern in ``ksi.cli._cleanup_containers`` but scoped to a single forum
    phase so sibling experiments on the same host are never touched.
    """
    if not container_name_substrings:
        return 0
    try:
        import subprocess as _subprocess

        ps = _subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        log.warning("[ENGINE] early-exit docker ps failed: %s", exc)
        return 0
    if ps.returncode != 0:
        return 0
    names = [n.strip() for n in ps.stdout.splitlines() if n.strip()]
    targets = [n for n in names if any(sub and sub in n for sub in container_name_substrings)]
    if not targets:
        return 0
    try:
        log.info(
            "[ENGINE] early-exit: stopping %d forum container(s): %s",
            len(targets),
            targets,
        )
        _subprocess.run(
            ["docker", "stop", *targets],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        log.warning("[ENGINE] early-exit docker stop failed: %s", exc)
        return 0
    return len(targets)


def _forum_container_prefix(
    *,
    experiment_name: str,
    generation: int,
    phase: str,
) -> str:
    """Derive the docker container-name prefix unique to this forum phase.

    Mirrors the composition in ``container_runner.ts`` (``ksi-runtime-{folder}``)
    and ``layout.task_workspace_key``.  For ``phase="per_task"`` the prefix
    matches per-task forum containers; for ``phase="cross_task"`` it matches
    cross-task forum containers.  The value has no trailing ``*`` -- callers
    pass it straight to substring matching.
    """
    from ..layout import sanitize_key as _sanitize_key

    experiment_part = _sanitize_key(experiment_name or "default", fallback="default", max_len=24)
    js_safe_exp = re.sub(r"[^a-zA-Z0-9-]", "-", experiment_part)
    phase_tag = "cross-task-forum" if phase == "cross_task" else "forum"
    # ``task_workspace_key`` renders ``task__<exp>__<sanitized_task_id>__<digest>``;
    # the TS layer then replaces every non-[a-zA-Z0-9-] char with ``-`` to form
    # the container name.  Leading ``__`` on the forum task id (e.g.
    # ``__forum__g1_r0_agent-0``) is stripped by ``sanitize_key``, so the
    # post-replace rendering is ``task--<exp>--<phase_tag>--g{gen}--``.
    return f"ksi-runtime-task--{js_safe_exp}--{phase_tag}--g{int(generation)}"


class _ForumEarlyExitWatcher(threading.Thread):
    """Background poller that short-circuits forum phases on all-done signals.

    Runs alongside the forum ``ThreadPoolExecutor``.  Every ``poll_interval_sec``
    the watcher reads ``forum_bus`` done events (and optionally consults
    ``knowledge`` for host-side ``discussion_done`` rows) and, once every
    expected ``(task_id, agent_id)`` pair has signalled, sets
    ``triggered_event`` and issues a targeted ``docker stop`` against the
    remaining forum containers for this phase.  The safety backstop is the
    outer ``--forum-timeout-sec`` / ``--cross-task-forum-timeout-sec`` that
    gates ``run_task`` itself -- this watcher can only finish *early*.

    Requiring 100% of agents to signal means a straggler
    minority (concentrated in generation-1 cold starts) permanently disables
    early-exit for that round, forcing the
    full hard-timeout wait. ``quorum_pct`` (default 100.0, matching the
    all-required behavior exactly) lets callers opt into exiting
    once only a threshold fraction of expected agents have signalled, after
    waiting ``quorum_grace_sec`` from the moment that fraction was first
    reached (giving near-quorum stragglers a short window to catch up before
    their containers are cancelled).
    """

    def __init__(
        self,
        *,
        forum_bus: "ForumBus",
        expected: dict[str, set[str]] | None = None,
        expected_agents: set[str] | None = None,
        agent_only: bool = False,
        round_num: int | None = None,
        stop_event: threading.Event,
        triggered_event: threading.Event,
        container_name_prefixes: list[str],
        poll_interval_sec: float = 3.0,
        knowledge: Any = None,
        experiment: str | None = None,
        generation: int | None = None,
        phase_label: str = "forum",
        quorum_pct: float = 100.0,
        quorum_grace_sec: float = 0.0,
    ) -> None:
        super().__init__(daemon=True, name=f"forum-early-exit-{phase_label}")
        self._forum_bus = forum_bus
        # Shallow-copy the expected structures so later mutations in the caller
        # never change our exit condition mid-run.
        self._expected = {tid: set(agents) for tid, agents in (expected or {}).items() if agents}
        self._expected_agents = set(expected_agents or set())
        self._agent_only = bool(agent_only)
        self._stop_event = stop_event
        self._triggered_event = triggered_event
        self._container_name_prefixes = list(container_name_prefixes)
        # Clamp to a minimum of 50ms so the watcher never busy-loops on
        # pathological configs, but still runs fast enough for unit tests.
        self._poll_interval_sec = max(0.05, float(poll_interval_sec))
        self._knowledge = knowledge
        self._experiment = experiment
        self._generation = generation
        self._round_num = round_num
        self._phase_label = phase_label
        # Clamp to [0, 100]; a value >= 100 disables the quorum path entirely
        # (the ``_all_signalled`` check above already covers exact 100%).
        self._quorum_pct = min(100.0, max(0.0, float(quorum_pct)))
        self._quorum_grace_sec = max(0.0, float(quorum_grace_sec))

    def _all_signalled(self) -> bool:
        # Cross-task: any ``done`` event per agent is enough.
        if self._agent_only:
            if not self._expected_agents:
                return False
            return _all_expected_signalled(
                forum_bus=self._forum_bus,
                expected={},
                agent_only=True,
                expected_agents=self._expected_agents,
                round_num=self._round_num,
            )
        if not self._expected:
            return False
        if _all_expected_signalled(
            forum_bus=self._forum_bus,
            expected=self._expected,
            round_num=self._round_num,
        ):
            return True
        if self._round_num is not None:
            return False
        # Fallback: if the host-side KnowledgeStore is writable (rare for
        # in-container MCP, but used in unit tests where signal_done is called
        # directly), consult ``discussion_done`` as a second source of truth.
        if self._knowledge is None or self._generation is None:
            return False
        try:
            for task_id, agents in self._expected.items():
                if not agents:
                    continue
                status = self._knowledge.get_done_status(
                    task_id=task_id,
                    generation=int(self._generation),
                    expected_agents=len(agents),
                    experiment=self._experiment,
                )
                if not status.get("all_done", False):
                    return False
            return True
        except Exception as exc:
            log.debug(
                "[ENGINE] early-exit watcher KnowledgeStore probe failed: %s",
                exc,
            )
            return False

    def _quorum_progress(self) -> tuple[int, int]:
        """Return ``(signalled, expected)`` agent counts for the quorum check.

        Mirrors the two modes in :meth:`_all_signalled` but counts *partial*
        progress instead of requiring every ``(task_id, agent_id)`` pair to
        match. Does not consult the ``KnowledgeStore`` fallback -- that path
        exists only for the rare host-writable-KnowledgeStore case and adds
        no signal a quorum decision needs beyond the ForumBus.
        """
        if self._agent_only:
            expected = self._expected_agents
            if not expected:
                return (0, 0)
            observed = _read_done_agent_ids(self._forum_bus, round_num=self._round_num)
            return (len(expected & observed), len(expected))
        if not self._expected:
            return (0, 0)
        observed_by_task = _read_done_signals(self._forum_bus, round_num=self._round_num)
        expected_total = sum(len(agents) for agents in self._expected.values())
        signalled_total = sum(
            len(agents & observed_by_task.get(task_id, set())) for task_id, agents in self._expected.items()
        )
        return (signalled_total, expected_total)

    def run(self) -> None:
        # Monotonic timestamp of the first poll where the quorum threshold
        # was met; reset to None whenever a poll observes it's no longer met
        # (defensive -- signalled counts are monotonic in practice, but this
        # keeps the grace window honest if that ever changes).
        quorum_met_at: float | None = None
        # Minimum startup delay so short test runs that already have all
        # signals pre-written still exercise the "trigger immediately" path
        # without flakiness on slow filesystems.
        while not self._stop_event.is_set():
            try:
                if self._all_signalled():
                    log.info(
                        "[ENGINE] early-exit: all expected agents signalled done "
                        "(phase=%s, tasks=%d); cancelling remaining forum containers",
                        self._phase_label,
                        len(self._expected),
                    )
                    self._triggered_event.set()
                    stopped = _stop_forum_containers(self._container_name_prefixes)
                    log.info(
                        "[ENGINE] early-exit: stopped %d container(s) for phase=%s",
                        stopped,
                        self._phase_label,
                    )
                    return
                if self._quorum_pct < 100.0:
                    signalled, expected = self._quorum_progress()
                    quorum_reached = expected > 0 and (signalled / expected) * 100.0 >= self._quorum_pct
                    if not quorum_reached:
                        quorum_met_at = None
                    elif quorum_met_at is None:
                        quorum_met_at = time.monotonic()
                    elif time.monotonic() - quorum_met_at >= self._quorum_grace_sec:
                        log.info(
                            "[ENGINE] early-exit: quorum %.1f%% reached (%d/%d agents signalled, "
                            "phase=%s) and %.1fs grace window elapsed; cancelling remaining forum "
                            "containers (%d straggler(s) left behind)",
                            self._quorum_pct,
                            signalled,
                            expected,
                            self._phase_label,
                            self._quorum_grace_sec,
                            expected - signalled,
                        )
                        self._triggered_event.set()
                        stopped = _stop_forum_containers(self._container_name_prefixes)
                        log.info(
                            "[ENGINE] early-exit: stopped %d container(s) for phase=%s",
                            stopped,
                            self._phase_label,
                        )
                        return
            except Exception as exc:
                log.debug("[ENGINE] early-exit watcher iteration failed: %s", exc)
            # ``Event.wait`` is interrupted when the stop_event is set, so the
            # watcher exits immediately when the forum phase finishes normally.
            if self._stop_event.wait(self._poll_interval_sec):
                return


def _collect_failed_attempt_event_ids(
    forum_bus: Any | None,
    *,
    after_seq: int,
    agent_id: str,
    forum_round: int | None,
) -> list[str]:
    """Snapshot bus event_ids written by ``agent_id`` during a failed
    attempt so the retry helper can mark them stale.

    Filters by ``agent_id`` AND ``round_num`` so concurrent agents'
    events on the same shared bus are never collateral-marked.  Reads
    are best-effort: any failure returns an empty list — the bus
    sidecar is an optimization, not a correctness gate, so a transient
    read error must never abort the retry path.
    """
    if forum_bus is None:
        return []
    try:
        events = forum_bus.read_events(after_seq=after_seq)
    except Exception:
        log.warning("[ENGINE] forum_bus.read_events failed during stale-collection", exc_info=True)
        return []
    out: list[str] = []
    for ev in events:
        ev_agent = str(getattr(ev, "agent_id", "") or "").strip()
        if ev_agent != str(agent_id or "").strip():
            continue
        if forum_round is not None:
            ev_round = getattr(ev, "round_num", None)
            if ev_round is None or int(ev_round) != int(forum_round):
                continue
        ev_id = str(getattr(ev, "event_id", "") or "").strip()
        if ev_id:
            out.append(ev_id)
    return out


def _bus_seq_count(forum_bus: Any | None) -> int:
    """Return the current event count on ``forum_bus`` (0 on error)."""
    if forum_bus is None:
        return 0
    try:
        return len(forum_bus.read_events())
    except Exception:
        return 0


class _CrossTaskR1Coordinator:
    """Synchronizes N cross-task forum agents at the R0->R1 barrier.

    Used by the cross-task forum service when the
    ``cross_task_shared_container`` feature flag is on. The flow:

      1. Engine dispatches one container per agent for what would have
         been the legacy R0+R1 sequence. Each container runs R0, writes a
         barrier sentinel, waits for the host's R1-prompt response.
      2. Each agent's container has a host-side ``BarrierWatcher`` whose
         callback is :meth:`on_sentinel`. The callback BLOCKS until this
         coordinator declares "go" (after all expected agents have
         signalled OR the timeout fires), then returns the per-agent
         response dict that the watcher writes back.
      3. After all agents either signalled or timed out,
         :meth:`compute_responses` runs once: it drains the forum bus and
         calls ``prompt_builder(agent_id) -> str`` for each ready agent
         to produce its R1 prompt suffix, packed into the response dict.

    Failure handling:
      * Slow agent: when ``timeout_sec`` elapses, the coordinator unblocks
        with whatever sentinels arrived; agents that never signalled
        either get nothing back (their containers will hit their own
        in-container poll timeout and emit R0-only envelopes) or get a
        synthetic ``{"error": "..."}`` if the watcher fired late.
      * Coordinator exception: re-thrown by ``on_sentinel`` so each
        watcher writes ``{"error": ...}`` and the agent emits R0-only.
    """

    def __init__(
        self,
        *,
        expected_agent_ids: list[str],
        prompt_builder: Callable[[str], str],
        timeout_sec: float,
        on_drain: Callable[[], None] | None = None,
    ) -> None:
        self._expected = list(expected_agent_ids)
        self._prompt_builder = prompt_builder
        self._timeout_sec = max(1.0, float(timeout_sec))
        self._on_drain = on_drain
        self._lock = threading.Lock()
        self._sentinels: dict[str, dict[str, Any]] = {}
        self._all_or_timeout = threading.Event()
        self._responses_ready = threading.Event()
        self._responses: dict[str, dict[str, Any]] = {}
        self._timed_out: set[str] = set()
        # Background thread that watches for the all-ready/timeout
        # transition then computes responses once.
        self._coordinator_thread = threading.Thread(
            target=self._run_coordinator,
            daemon=True,
            name="cross-task-r1-coordinator",
        )
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._coordinator_thread.start()

    def stop(self) -> None:
        """Force-unblock any pending watchers if the engine is tearing down."""
        self._stop_event.set()
        self._all_or_timeout.set()
        if self._coordinator_thread.is_alive() and threading.current_thread() is not self._coordinator_thread:
            # Bound the join: _run_coordinator's _on_drain() (KnowledgeStore
            # writes + embedding batch) has no timeout of its own, so a slow
            # drain here would otherwise hang the main engine thread
            # indefinitely at generation teardown. Mirror the +30s grace the
            # per-agent on_sentinel callers already give the drain.
            join_timeout = self._timeout_sec + 30
            self._coordinator_thread.join(timeout=join_timeout)
            if self._coordinator_thread.is_alive():
                log.warning(
                    "cross-task-r1 coordinator thread did not stop within %.0fs; "
                    "a forum-bus drain (KnowledgeStore write / embedding batch) may "
                    "still be running in the background",
                    join_timeout,
                )
        self._responses_ready.set()

    def on_sentinel(self, event: "BarrierEvent") -> dict[str, Any]:
        """BarrierWatcher callback: register this agent's R0 sentinel and
        block until the coordinator computes responses."""
        # Decode the agent_id from the sentinel payload (the schema sets
        # ``agent_id``); fall back to extracting from the response file
        # name if absent (defensive — should never happen).
        payload = event.payload or {}
        agent_id = str(payload.get("agent_id") or "")
        if not agent_id:
            try:
                # ``.barrier.<name>.<agent_id>.response`` -> agent_id
                stem = event.response_path.name
                parts = stem.split(".")
                if len(parts) >= 4:
                    agent_id = parts[-2]
            except Exception:
                pass
        if not agent_id:
            return {"error": "cross_task_r1: sentinel missing agent_id"}
        with self._lock:
            self._sentinels[agent_id] = dict(payload)
            ready_count = len(self._sentinels)
            expected_count = len(self._expected)
        if ready_count >= expected_count:
            self._all_or_timeout.set()
        # Block until coordinator unblocks (drain + compute responses).
        self._responses_ready.wait(timeout=self._timeout_sec + 30)
        with self._lock:
            response = self._responses.get(agent_id)
        if response is None:
            return {
                "error": f"cross_task_r1: coordinator returned no response for agent={agent_id}",
            }
        return response

    def _run_coordinator(self) -> None:
        """Wait for all agents (or timeout), drain, compute responses."""
        # Wait for all_ready or timeout. Note: the BarrierWatcher per
        # agent has its OWN poll timeout; this coordinator timeout is the
        # upper bound on how long we wait for ALL agents to signal.
        self._all_or_timeout.wait(timeout=self._timeout_sec)
        with self._lock:
            ready_agents = list(self._sentinels.keys())
            for aid in self._expected:
                if aid not in self._sentinels:
                    self._timed_out.add(aid)
        if self._timed_out:
            log.warning(
                "[cross_task_r1] coordinator: %d/%d agents timed out at R0->R1 barrier (%s)",
                len(self._timed_out),
                len(self._expected),
                ",".join(sorted(self._timed_out)),
            )
        # Drain forum bus + any other engine-side housekeeping.
        if self._on_drain is not None:
            try:
                self._on_drain()
            except Exception:
                log.exception("[cross_task_r1] coordinator drain failed")
        # Compute per-agent R1 prompts. Build defensively: a builder
        # exception for one agent should not poison the others.
        responses: dict[str, dict[str, Any]] = {}
        for aid in ready_agents:
            try:
                prompt_text = self._prompt_builder(aid)
            except Exception as exc:
                log.exception("[cross_task_r1] prompt_builder failed for agent=%s", aid)
                responses[aid] = {
                    "error": f"cross_task_r1: prompt_builder raised: {type(exc).__name__}: {exc}",
                }
                continue
            responses[aid] = {
                "schema": "cross_task_r1.v1",
                "agent_id": aid,
                "r1_prompt_text": str(prompt_text or ""),
            }
        with self._lock:
            self._responses = responses
        self._responses_ready.set()

    @property
    def timed_out_agents(self) -> set[str]:
        return set(self._timed_out)


def _run_retryable_forum_task(
    *,
    run_once: Callable[[], Any],
    generation: int,
    agent_id: str,
    phase_label: str,
    attempts: int,
    forum_bus: Any | None = None,
    forum_round: int | None = None,
) -> tuple[TokenUsage, dict[str, Any], str]:
    """Run a discussion task with bounded retries on transient failures.

    When ``forum_bus`` and ``forum_round`` are provided, events written
    by failed attempts (which may have appended ``forum_post`` /
    ``insight`` / ``comment`` / ``done`` events to the bus before the
    SDK iterator drained) are marked stale so ``_drain_forum_bus``
    skips them.  Without this, the bus would carry duplicate posts
    after a successful retry.
    """
    attempts = max(1, int(attempts))
    last_exc: Exception | None = None
    attempt_errors: list[dict[str, Any]] = []
    failed_runtime_metas: list[dict[str, Any]] = []

    for attempt_idx in range(attempts):
        pre_attempt_seq = _bus_seq_count(forum_bus)
        try:
            result = run_once()
            if isinstance(result, RuntimeResult):
                runtime_meta = dict(result.runtime_meta or {})
                token_usage = result.token_usage
                if attempt_errors:
                    runtime_meta.update(
                        _runtime_retry_meta(
                            attempt_errors,
                            terminal_failure=False,
                            failed_runtime_metas=failed_runtime_metas,
                            # The direct _accumulate_failed_attempt_tokens
                            # call just below recomputes this same sum and
                            # owns the drop-count WARNING here, to avoid
                            # double-logging one real drop.
                            log_dropped_tokens=False,
                        )
                    )
                    token_usage = token_usage + _accumulate_failed_attempt_tokens(failed_runtime_metas)
                    log.warning(
                        "[ENGINE] %s succeeded after %d retry/retries generation=%s agent=%s attempts=%d/%d",
                        phase_label,
                        len(attempt_errors),
                        generation,
                        agent_id,
                        attempt_idx + 1,
                        attempts,
                    )
                return token_usage, runtime_meta, str(result.output or "")
            return TokenUsage(), {}, str(result or "")
        except Exception as exc:
            if isinstance(exc, AuthenticationFailure) or _is_auth_error(exc):
                raise AuthenticationFailure(
                    f"LLM authentication failed for {phase_label}: {exc}",
                ) from exc
            last_exc = exc
            attempt_errors.append(
                {
                    "attempt": attempt_idx + 1,
                    "max_attempts": attempts,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if isinstance(exc, SilentAgentRuntimeError) and isinstance(exc.runtime_meta, dict):
                failed_runtime_metas.append(dict(exc.runtime_meta))
            # Mark this failed attempt's bus events stale so
            # the drain skips them when the retry produces fresh content.
            # Done on every failed attempt — including the terminal one —
            # because partial bus writes from a crashed run are not
            # canonical knowledge and should never land in
            # KnowledgeStore alongside the retry's content.
            stale_ids = _collect_failed_attempt_event_ids(
                forum_bus,
                after_seq=pre_attempt_seq,
                agent_id=agent_id,
                forum_round=forum_round,
            )
            if stale_ids and forum_bus is not None:
                try:
                    forum_bus.mark_stale(stale_ids, reason="failed_attempt")
                except Exception:
                    log.warning(
                        "[ENGINE] forum_bus.mark_stale failed agent=%s gen=%s round=%s attempt=%d",
                        agent_id,
                        generation,
                        forum_round,
                        attempt_idx + 1,
                        exc_info=True,
                    )
            if attempt_idx >= attempts - 1 or not _is_retryable_task_error(exc):
                break
            log.warning(
                "[ENGINE] transient forum failure, retrying phase=%s generation=%s agent=%s attempt=%s/%s: %s",
                phase_label,
                generation,
                agent_id,
                attempt_idx + 1,
                attempts,
                exc,
            )
            delay = min(60, 0.5 * 2**attempt_idx) * (0.5 + random.random())
            time.sleep(delay)

    error_text = str(last_exc or RuntimeError(f"{phase_label} failed"))
    log.warning(
        "[ENGINE] %s failed for agent=%s gen=%s after %d attempt(s): %s",
        phase_label,
        agent_id,
        generation,
        len(attempt_errors) or 1,
        error_text,
    )
    failure_runtime_meta: dict[str, Any] = {
        "forum_error": error_text,
        "forum_error_type": type(last_exc).__name__ if last_exc is not None else "RuntimeError",
    }
    if attempt_errors:
        failure_runtime_meta["forum_retry_meta"] = _runtime_retry_meta(
            attempt_errors,
            terminal_failure=True,
            failed_runtime_metas=failed_runtime_metas,
            # failed_token_total below recomputes this same sum and owns the
            # drop-count WARNING here, to avoid double-logging one real drop.
            log_dropped_tokens=False,
        )
    # Surface aggregated failed-attempt tokens so the caller's
    # ``record_lifecycle`` /  ``agent.token_usage`` accounting reflects the
    # real cost even when every attempt failed. Without this, terminal forum
    # failures look free in token_phases despite consuming real billable
    # tokens during e.g. SDK race retries.
    failed_token_total = _accumulate_failed_attempt_tokens(failed_runtime_metas)
    return failed_token_total, failure_runtime_meta, ""

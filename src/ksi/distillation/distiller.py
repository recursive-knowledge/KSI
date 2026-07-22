"""Orchestrates per-task and cross-task distillation."""

from __future__ import annotations

import concurrent.futures
import logging
import os
from typing import Any

from ksi.memory.knowledge_store import CROSS_TASK_SENTINEL
from ksi.memory.parity import redact_solver_hidden_eval_fields

from ..errors import AuthenticationFailure
from ._removed_env import assert_no_removed_channel_env
from .cross_task import distill_cross_task, select_shared_cross_posts_for_targets
from .per_task import distill_one_task, truncate_at_boundary
from .types import CrossTaskBundle, DistillInput, DistillOutput, PerTaskBundle

log = logging.getLogger(__name__)


# Effectively-uncapped per-bucket query limit. The knowledge_store applies
# `limit` per entry_type bucket with ORDER BY k.id ASC, so capping at 500
# silently dropped NEWER posts at high N. Set to a value far above any
# realistic per-bucket count for our experiment + production scales (50
# tasks × 10 gens × 2 rounds ≈ 1000 cross-task posts, well under). If we
# ever hit this, switch to streaming/pagination — but for now uncapped is
# the right semantic.
_UNCAPPED_LIMIT = 100_000

# Sliding window for cross-task distill input. Without it, cross-task forum
# history accumulates across all gens and at gen 10 of a 50-task ARC2 c=50
# run hit Anthropic's 200K context (210,659 tokens > 200,000 maximum).
# Default 6 means each gen's cross-task distill sees the last 6 generations
# of cross-task posts. Older cross-task insights are subsumed by per-task
# distills (which retain full per-task history). Tunable via
# KSI_CROSS_TASK_DISTILL_GEN_WINDOW; set to 0 to disable windowing.
_CROSS_TASK_DISTILL_GEN_WINDOW_DEFAULT = 6

# Sliding window for per-task distill attempt input. Same failure
# mode as cross-task above: a task that stays unsolved across many
# generations accumulates one attempt (with reflection/trace_condensed) per
# generation with no cap, risking the same 200K-context overflow and paying
# O(G^2) distillation cost per stuck task across G
# generations. Default matches the cross-task window for consistency.
# Tunable via KSI_PER_TASK_DISTILL_GEN_WINDOW; set to 0 to disable windowing.
_PER_TASK_DISTILL_GEN_WINDOW_DEFAULT = 6


def distill(
    inp: DistillInput,
    *,
    unsolved_task_ids: list[str] | None = None,
    newly_solved_task_ids: list[str] | None = None,
) -> DistillOutput:
    """Run per-task and cross-task distillation for a generation.

    Returns a DistillOutput with whatever bundles succeeded. The orchestrator
    writes successful bundles to KnowledgeStore; missing tasks are simply
    absent from the per_task dict.

    ``unsolved_task_ids``: optional pre-filtered list of task IDs that are
    still unsolved at the end of this generation. Per-task distillation is
    SKIPPED for solved tasks because next-gen agents won't see those tasks
    (under ``--drop-solved``), so the distilled bundle would never be read.
    When None, fall back to ``inp.task_ids``.

    ``newly_solved_task_ids``: tasks solved THIS generation. Consumed only
    when the transfer bridge is on (``KSI_TRANSFER_BRIDGE``): each gets one
    per-task distill in win mode so the winning technique lands in
    ``transferable_insights``. Ignored when the flag is off.

    Strategy: ``window`` (recompute from the full windowed forum history every
    generation). Per-task input = attempts + per-task posts across all gens;
    cross-task input = cross-task forum history across the gen-window.
    """
    assert_no_removed_channel_env()
    bridge = _transfer_bridge_enabled()
    per_task_results: dict[str, PerTaskBundle] = {}
    failures = 0  # silent-degradation counter surfaced via DistillOutput

    # Skip per-task distill for solved tasks — next-gen --drop-solved means
    # the bundle would be wasted spend. Engine passes pre-computed list.
    per_task_target_ids = list(unsolved_task_ids) if unsolved_task_ids is not None else list(inp.task_ids)
    if not per_task_target_ids:
        log.info("distill: no unsolved task ids to distill per-task bundles for")

    # Transfer bridge (flag on): newly-solved tasks ALSO get a per-task
    # distill, in win mode. Defensively intersect with inp.task_ids (the
    # engine already strips hold-outs upstream) and exclude anything already
    # scheduled for a normal distill.
    win_task_ids: list[str] = []
    if bridge and newly_solved_task_ids:
        in_scope = set(inp.task_ids) - set(per_task_target_ids)
        win_task_ids = [tid for tid in newly_solved_task_ids if tid in in_scope]
        if win_task_ids:
            log.info(
                "distill gen=%d: transfer bridge running win-mode distill for %d newly-solved task(s)",
                inp.generation,
                len(win_task_ids),
            )

    # Per-task distillation in parallel (win-mode tasks share the batch)
    if per_task_target_ids or win_task_ids:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            per_task_futures = {ex.submit(_one_task, tid, inp): tid for tid in per_task_target_ids}
            per_task_futures.update({ex.submit(_one_task, tid, inp, win_mode=True): tid for tid in win_task_ids})
            for per_task_future in concurrent.futures.as_completed(per_task_futures):
                tid = per_task_futures[per_task_future]
                try:
                    bundle, task_failed = per_task_future.result()
                except AuthenticationFailure:
                    raise
                except Exception as exc:
                    log.warning("per-task distill for %s raised: %r", tid, exc)
                    failures += 1
                    continue
                # task_failed=True means the task HAD input but distillation
                # produced no fresh bundle (LLM/parse failure). Distinct from a
                # benign "no attempts" None, which reports task_failed=False so a
                # healthy run still yields {}.
                if task_failed:
                    failures += 1
                if bundle is not None:
                    per_task_results[tid] = bundle

    # Transfer bridge (flag on): gather success-derived transferables for the
    # cross-task channel. None (not []) when off so the prompt builders stay
    # byte-identical and no store queries happen.
    per_task_transferables: list[dict[str, Any]] | None = None
    if bridge:
        per_task_transferables = _collect_per_task_transferables(inp, per_task_results)

    # Count of cross-task posts that fed the distill (for the
    # silent-degradation visibility log below).
    # Cross-task distill (window): input is ONLY the full cross-task forum
    # history. No per-task posts (Phase 3 is structurally independent of
    # Phase 2). No prior bundle (consumed at seed time, doesn't feed back).
    # There is no enable_cross_task gate — the channel is unconditional.
    cross_task_history = _load_cross_task_posts_all_gens(inp)
    cross_post_count = len(cross_task_history)
    cross_llm = inp.llm_cross_task or inp.llm

    cross_bundle: CrossTaskBundle | None = None
    cross_by_task: dict[str, CrossTaskBundle] | None = None
    cross_target_ids: list[str] = []

    if inp.cross_task_target_conditioning:
        # One cross-task distill per downstream seed target, conditioned on that
        # task's prompt. Callers may provide an explicit target list so hold-out
        # probes and --no-drop-solved retained tasks can receive cross-task
        # guidance even when they are not per-task learning targets.
        cross_target_ids = target_task_ids_for_conditioning(inp, unsolved_task_ids)
        prompts = inp.target_task_prompts or {}
        missing_prompts = [tid for tid in cross_target_ids if tid not in prompts]
        if missing_prompts:
            log.warning(
                "cross-task target conditioning: skipping %d target(s) with missing prompt(s): %s",
                len(missing_prompts),
                ", ".join(missing_prompts[:10]),
            )
            failures += len(missing_prompts)
            cross_target_ids = [tid for tid in cross_target_ids if tid in prompts]
        cross_by_task = {}
        if cross_post_count and cross_target_ids:
            # Choose the post set each target distills from. Two mutually
            # exclusive regimes, controlled by ``cross_task_per_target_selection``:
            #
            # DEFAULT (per_target_selection=False) — SHARED SET, CACHE-OPTIMAL:
            #   Trim the shared forum history ONCE (against the largest target)
            #   so every target distills from the SAME post set → byte-identical
            #   cache_prefix → the shared-history cache actually reads across
            #   targets. Per-target trimming would pick different subsets and
            #   silently defeat the cache. Falls back to the
            #   full untrimmed history only in the pathological case where the
            #   largest target's fixed section alone exceeds budget. This is the
            #   default published behavior.
            #
            # OPT-IN (per_target_selection=True) — PER-TARGET SET, RELEVANCE-OPTIMAL:
            #   Hand each target the FULL forum history; its own
            #   ``distill_cross_task`` re-trims to the posts most relevant to
            #   THAT target (relevance-ranked selection). Under the shared set,
            #   every target except the largest gets posts selected for the
            #   largest target's vocabulary. COST: each
            #   target now selects a DIFFERENT post subset → a different
            #   ``cache_prefix`` per target → the cross-target prompt cache is
            #   defeated (``cache_creation>0, cache_read=0`` for every target but
            #   the first), i.e. every over-budget target pays a full cache write.
            #   This is the fundamental cache (tokens) vs. relevance (quality)
            #   tradeoff; it is opt-in precisely because it changes the
            #   default published cost profile.
            if inp.cross_task_per_target_selection:
                # Per-target selection: full history in, each target trims its own.
                posts_for_targets = cross_task_history
            else:
                shared_cross_posts = select_shared_cross_posts_for_targets(
                    cross_posts=cross_task_history,
                    task_source=inp.task_source,
                    per_task_transferables=per_task_transferables,
                    target_tasks=[{"id": tid, "prompt": str(prompts.get(tid) or "")} for tid in cross_target_ids],
                )
                if cross_task_history and not shared_cross_posts:
                    shared_cross_posts = cross_task_history
                posts_for_targets = shared_cross_posts

            def _distill_for(tid: str) -> tuple[str, CrossTaskBundle | None]:
                target = {"id": tid, "prompt": str(prompts.get(tid) or "")}
                try:
                    bundle = distill_cross_task(
                        cross_posts=posts_for_targets,
                        llm=cross_llm,
                        task_source=inp.task_source,
                        bundle_schema=inp.bundle_schema,
                        per_task_transferables=per_task_transferables,
                        target_task=target,
                    )
                except AuthenticationFailure:
                    raise
                except Exception as exc:
                    log.warning("cross-task distill (task=%s) raised: %r", tid, exc)
                    return tid, None
                return tid, bundle

            # Warm the shared-history prompt cache with the FIRST target
            # synchronously, then fan out the rest. In the DEFAULT (shared-set)
            # mode every target re-sends the same ~120K-token windowed forum
            # history as a cache_prefix; if all workers
            # start concurrently they race on a cold cache and each pays the
            # full write. Running one to completion first populates the cache so
            # the remaining targets cache-READ it. Under per-target selection
            # each target sends a DIFFERENT post set, so this warm-first ordering
            # yields no cross-target cache benefit (that is the documented cost
            # of the opt-in); it stays harmless. Order is deterministic
            # (cross_target_ids is a list).
            ordered_ids = list(cross_target_ids)
            first_tid, first_bundle = _distill_for(ordered_ids[0])
            if first_bundle is not None:
                cross_by_task[first_tid] = first_bundle
            else:
                failures += 1
            rest_ids = ordered_ids[1:]
            if rest_ids:
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                    cross_futures = {ex.submit(_distill_for, tid): tid for tid in rest_ids}
                    for cross_future in concurrent.futures.as_completed(cross_futures):
                        try:
                            cross_tid, cross_bundle_result = cross_future.result()
                        except AuthenticationFailure:
                            raise
                        if cross_bundle_result is not None:
                            cross_by_task[cross_tid] = cross_bundle_result
                        else:
                            failures += 1
    else:
        try:
            cross_bundle = distill_cross_task(
                cross_posts=cross_task_history,
                llm=cross_llm,
                task_source=inp.task_source,
                bundle_schema=inp.bundle_schema,
                per_task_transferables=per_task_transferables,
            )
        except AuthenticationFailure:
            raise
        except Exception as exc:
            # A transient failure on the (most expensive) cross-task LLM call
            # must not discard the already-completed per-task bundles — honor the
            # docstring contract "returns whatever bundles succeeded". Degrade to
            # no cross-task bundle; the no-bundle warning + ``failures`` increment
            # below (cross_post_count > 0) surfaces the degradation, so we do not
            # double-count it here.
            log.warning("cross-task distill raised: %r", exc)
            cross_bundle = None

    # Aggregate visibility for the silent-degradation failure mode: a
    # distillation that returns no bundle is otherwise invisible (it just looks
    # like "the system isn't learning much"). Escalate to WARNING when nothing
    # was produced at all.
    # Win-mode tasks count toward attempted too — their bundles land in
    # per_task_results, and when ALL tasks are solved they are the only
    # attempts, so excluding them would both skew the no-bundle delta
    # negative and hide total win-distill failure from the warning below.
    attempted = len(per_task_target_ids) + len(win_task_ids)
    produced = len(per_task_results)
    if attempted:
        log_fn = log.warning if produced == 0 else log.info
        log_fn(
            "distill gen=%d: produced %d/%d per-task bundle(s); %d task(s) yielded "
            "no bundle (no attempts or distill failure)",
            inp.generation,
            produced,
            attempted,
            attempted - produced,
        )
    # Window signals cross-task failure by returning no bundle despite having
    # posts to distill; count it so a degraded cross-task distill isn't
    # invisible. Under target-conditioning "produced" means at least one
    # per-task bundle succeeded; the per-task loop above already incremented
    # ``failures`` for each failed target, so we don't double-count here.
    produced_cross = cross_bundle is not None or bool(cross_by_task)
    # Under target-conditioning, "no bundle" is only a degradation when there
    # were target tasks to distill for; an empty target set (e.g. all tasks
    # solved) is a benign no-op, not a failure to warn about.
    cross_work_expected = bool(cross_target_ids) if inp.cross_task_target_conditioning else True
    if not produced_cross and cross_post_count and cross_work_expected:
        log.warning(
            "distill gen=%d: cross-task distill produced no bundle despite %d cross-task post(s) in window",
            inp.generation,
            cross_post_count,
        )
        if not inp.cross_task_target_conditioning:
            failures += 1

    return DistillOutput(
        per_task=per_task_results,
        cross_task=cross_bundle,
        failures=failures,
        cross_task_by_task=cross_by_task,
    )


def _one_task(
    task_id: str,
    inp: DistillInput,
    *,
    win_mode: bool = False,
) -> tuple[PerTaskBundle | None, bool]:
    """Returns ``(bundle, failed)``. ``failed`` is True when the task had input
    but distillation produced no fresh bundle (LLM/parse failure); a benign
    "no attempts" return reports ``(None, False)`` so a healthy generation
    still yields an empty health block."""
    per_task_llm = inp.llm_per_task or inp.llm
    attempts = _load_attempts(inp, task_id)
    if not attempts:
        return None, False
    posts = _load_per_task_posts(inp, task_id)
    bundle = distill_one_task(
        task_id=task_id,
        attempts=attempts,
        posts=posts,
        llm=per_task_llm,
        task_source=inp.task_source,
        bundle_schema=inp.bundle_schema,
        win_mode=win_mode,
    )
    return bundle, bundle is None


# Transfer bridge (KSI_TRANSFER_BRIDGE) collection caps: deterministic and
# small — the rendered section is never trimmed by the cross-task budget
# machinery, so the caps here are the only thing bounding its size.
_TRANSFERABLES_PER_TASK_CAP = 2
_TRANSFERABLES_TEXT_CAP = 480
# Mirrors the renderer's 200-char clip (_fmt_transferables_section) so the
# stored entry and the rendered line agree.
_TRANSFERABLES_APPLIES_WHEN_CAP = 200
_TRANSFERABLES_TASK_CAP = 20


def _collect_per_task_transferables(
    inp: DistillInput,
    fresh_bundles: dict[str, PerTaskBundle],
) -> list[dict[str, Any]]:
    """Gather success-derived transferables for the cross-task distill prompt.

    Precedence per task: this generation's freshly produced bundle (win-mode
    ones included) supersedes the store; tasks without a fresh bundle fall
    back to the LATEST stored per_task_distill bundle. Bundles whose
    ``transferable_insights`` is empty contribute nothing. Caps: <=2 items
    per task, each text boundary-truncated to 480 chars; when more than 20
    tasks have transferables, the 20 with the highest bundle generation win.
    """
    per_task: list[tuple[int, list[dict[str, Any]]]] = []  # (bundle_gen, entries)
    for tid in inp.task_ids:
        fresh = fresh_bundles.get(tid)
        if fresh is not None:
            insights: list[Any] = list(fresh.transferable_insights or [])
            bundle_gen = int(inp.generation)
        else:
            insights, bundle_gen = _latest_stored_transferables(inp, tid)
        entries: list[dict[str, Any]] = []
        for item in insights:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                applies_when = str(item.get("applies_when") or "").strip()
            else:
                text = str(item or "").strip()
                applies_when = ""
            if not text:
                continue
            entries.append(
                {
                    "task_id": tid,
                    "text": truncate_at_boundary(text, _TRANSFERABLES_TEXT_CAP),
                    "applies_when": truncate_at_boundary(applies_when, _TRANSFERABLES_APPLIES_WHEN_CAP),
                }
            )
            if len(entries) >= _TRANSFERABLES_PER_TASK_CAP:
                break
        if entries:
            per_task.append((bundle_gen, entries))
    if len(per_task) > _TRANSFERABLES_TASK_CAP:
        # Keep the freshest tasks by bundle generation (stable: ties keep
        # inp.task_ids order), then restore the original task order.
        keep = set(sorted(range(len(per_task)), key=lambda i: (-per_task[i][0], i))[:_TRANSFERABLES_TASK_CAP])
        per_task = [pt for i, pt in enumerate(per_task) if i in keep]
    return [entry for _, entries in per_task for entry in entries]


def _latest_stored_transferables(inp: DistillInput, task_id: str) -> tuple[list[Any], int]:
    """Latest stored per_task_distill bundle's transferable_insights and its
    generation. Reuses the query_task distillation bucket (the store already
    parses content tolerantly; rows come back ordered by id ASC, so the last
    scoped per-task entry is the latest)."""
    data = inp.knowledge_store.query_task(
        task_id,
        generation=None,
        entry_types=["distillation"],
        limit=_UNCAPPED_LIMIT,
        experiment=inp.experiment,
    )
    latest: dict[str, Any] | None = None
    for entry in data.get("distilled") or []:
        if entry.get("source_phase") != "per_task_distill":
            continue
        bundle = entry.get("bundle")
        if not isinstance(bundle, dict):
            continue
        # Rows carrying a non-bundle format have no transferable_insights; skip
        # them so the latest bundle-format row wins.
        if bundle.get("format"):
            continue
        gen = entry.get("gen")
        if gen is not None and int(gen) > int(inp.generation):
            continue
        latest = entry
    if latest is None:
        return [], 0
    insights = latest["bundle"].get("transferable_insights")
    gen = latest.get("gen")
    return (insights if isinstance(insights, list) else []), (int(gen) if gen is not None else 0)


def _load_attempts(inp: DistillInput, task_id: str, *, current_gen_only: bool = False) -> list[dict[str, Any]]:
    """Load attempts on this task within the gen-window (default last 6 gens).

    V2: drop ``native_session_memory`` (typically 50-150KB per attempt;
    redundant with ``reflection`` from the Phase 1 reflection step which
    is the agent's structured summary written with full session memory +
    eval result in working context). Load ``reflection`` instead.

    Chronological append-only order keeps the prompt prefix byte-stable
    across gens for prompt-cache hits.

    Sliding window: a task stuck unsolved across many generations
    otherwise accumulates one attempt per generation with no cap, risking
    the same context overflow the cross-task window (above) was added to
    fix. At gen N with window W, only attempts from generations
    [N-W+1 .. N] are returned.
    """
    window = _per_task_distill_gen_window()
    min_gen = (int(inp.generation) - window + 1) if window > 0 else None

    data = inp.knowledge_store.query_task(
        task_id,
        generation=None,
        entry_types=["attempt"],
        limit=_UNCAPPED_LIMIT,
    )
    # Fail-safe redaction at the load boundary: apply the canonical upstream-
    # strict policy so hidden test-runner tails, grader answer keys, ARC
    # per-test details, instance-report internals, hidden verifier transcripts
    # (attempt_meta), and hidden-marked trace/reflection text are stripped
    # before distillation — regardless of what the prompt renderer emits. The
    # renderer (prompts._fmt_eval_results) is an allow-list, but this keeps the
    # contract holding even if a future edit dumps raw fields. query_task
    # returns a fresh per-call json.loads'd page, so mutating it in place is safe.
    redact_solver_hidden_eval_fields(data)
    out: list[dict[str, Any]] = []
    dropped_pre_window = 0
    for a in data.get("attempts") or []:
        # NOTE: the "attempt" bucket keys the generation as "gen", not
        # "generation" (see KnowledgeStore._append_task_page_row) — matches
        # how _latest_stored_transferables reads the "distillation" bucket's
        # "gen" key below. Reading the wrong key here silently made every
        # attempt's generation None, which would make the sliding window
        # below a no-op.
        gen = a.get("gen")
        if gen is not None and int(gen) > int(inp.generation):
            continue
        if current_gen_only and (gen is None or int(gen) != int(inp.generation)):
            continue
        if min_gen is not None and gen is not None and int(gen) < min_gen:
            dropped_pre_window += 1
            continue
        content = a.get("content") or {}
        eval_results = dict(content.get("eval_results") or {})
        out.append(
            {
                "agent_id": a.get("agent_id"),
                "generation": gen,
                "native_score": a.get("score"),
                "model_output": content.get("model_output", ""),
                "eval_results": eval_results,
                "trace_condensed": content.get("trace_condensed", ""),
                "attempt_meta": content.get("attempt_meta") or {},
                "reflection": content.get("reflection", ""),
                "insights": content.get("insights") or [],
            }
        )
    if dropped_pre_window:
        log.info(
            "per_task_distill(%s): dropped %d attempt(s) pre-window (gen<%d) at gen=%d, "
            "kept %d attempt(s). window=%d (env KSI_PER_TASK_DISTILL_GEN_WINDOW)",
            task_id,
            dropped_pre_window,
            min_gen,
            inp.generation,
            len(out),
            window,
        )
    return out


def _load_per_task_posts(
    inp: DistillInput,
    task_id: str,
    *,
    current_gen_only: bool = False,
) -> list[dict[str, Any]]:
    """Load per-task forum posts on this task across ALL prior generations.

    V2: only ``post`` entries (legacy ``insight`` merge removed — V2 emits
    forum posts only; insight rows were a pre-V2 artifact of the deprecated
    R1 forum). Adds ``round_num`` and ``native_score`` (post author's task
    score, set by the engine) so the distiller can weight by post author's
    success and distinguish round 0 from round 1+ posts.
    """
    data = inp.knowledge_store.query_task(
        task_id,
        generation=None,
        entry_types=["post"],
        limit=_UNCAPPED_LIMIT,
    )
    posts: list[dict[str, Any]] = []
    for p in data.get("discussion") or []:
        gen = p.get("generation")
        if gen is not None and int(gen) > int(inp.generation):
            continue
        if current_gen_only and (gen is None or int(gen) != int(inp.generation)):
            continue
        posts.append(
            {
                "id": p.get("id"),
                "generation": gen,
                "round_num": p.get("round_num"),
                "native_score": p.get("native_score"),
                "agent_id": p.get("agent_id"),
                "text": p.get("text", ""),
                "reply_to": p.get("reply_to") or p.get("parent_id"),
            }
        )
    return posts


def _cross_task_distill_gen_window() -> int:
    """Resolve the sliding-window size for cross-task distill input.

    0 means uncapped (legacy behaviour); positive values cap to the last N
    generations.
    """
    raw = os.environ.get("KSI_CROSS_TASK_DISTILL_GEN_WINDOW")
    if raw is None or raw.strip() == "":
        return _CROSS_TASK_DISTILL_GEN_WINDOW_DEFAULT
    try:
        return max(0, int(raw))
    except ValueError:
        return _CROSS_TASK_DISTILL_GEN_WINDOW_DEFAULT


def _per_task_distill_gen_window() -> int:
    """Resolve the sliding-window size for per-task distill attempt input.

    0 means uncapped (legacy behaviour); positive values cap to the last N
    generations. Mirrors ``_cross_task_distill_gen_window`` above.
    """
    raw = os.environ.get("KSI_PER_TASK_DISTILL_GEN_WINDOW")
    if raw is None or raw.strip() == "":
        return _PER_TASK_DISTILL_GEN_WINDOW_DEFAULT
    try:
        return max(0, int(raw))
    except ValueError:
        return _PER_TASK_DISTILL_GEN_WINDOW_DEFAULT


_TRANSFER_BRIDGE_ON_VALUES = ("1", "true", "yes", "on")


def _transfer_bridge_enabled() -> bool:
    """Resolve the transfer-bridge flag from the environment.

    Off (default) keeps current behavior bit-for-bit. On
    (``KSI_TRANSFER_BRIDGE=1/true/yes/on``, case-insensitive), newly-solved
    tasks get a win-mode per-task distill and the cross-task distill prompt
    gains the success-derived transferables section. Unknown values fall back
    to off so a typo never silently changes behavior.
    """
    raw = (os.environ.get("KSI_TRANSFER_BRIDGE") or "").strip().lower()
    return raw in _TRANSFER_BRIDGE_ON_VALUES


def target_task_ids_for_conditioning(
    inp: DistillInput,
    unsolved_task_ids: list[str] | None,
) -> list[str]:
    """Task ids to distill cross-task bundles for under target-conditioning.

    Prefer the caller's explicit seed-target set; fall back to the old
    per-task/unsolved set for programmatic callers that do not provide it.
    """
    if inp.cross_task_target_ids is not None:
        return list(dict.fromkeys(inp.cross_task_target_ids))
    return list(unsolved_task_ids) if unsolved_task_ids is not None else list(inp.task_ids)


def _load_cross_task_posts_all_gens(inp: DistillInput) -> list[dict[str, Any]]:
    """Load cross-task forum posts within the gen-window (default last 6 gens).

    V2: cross-task knowledge accumulates across generations, so distillation
    sees the cross-task forum history. Chronological order preserved for
    prompt-cache stability. Adds ``round_num`` so the distiller can
    distinguish round-0 opinions from round-1 responses.

    Sliding window: at gen N with window W, only posts from generations
    [N-W+1 .. N] are returned. Without this cap, gen 10 of a 50-task c=50
    run hit Anthropic's 200K context (210,659 tokens). Per-task distills
    retain their own full-history view, so older cross-task themes still
    propagate via per-task bundles.
    """
    window = _cross_task_distill_gen_window()
    min_gen = (int(inp.generation) - window + 1) if window > 0 else None

    page = inp.knowledge_store.query_task(
        CROSS_TASK_SENTINEL,
        generation=None,
        entry_types=["post"],
        limit=_UNCAPPED_LIMIT,
    )
    out: list[dict[str, Any]] = []
    dropped_pre_window = 0
    for p in page.get("discussion") or []:
        gen = p.get("generation")
        if gen is not None and int(gen) > int(inp.generation):
            continue
        if min_gen is not None and gen is not None and int(gen) < min_gen:
            dropped_pre_window += 1
            continue
        out.append(
            {
                "id": p.get("id"),
                "generation": gen,
                "round_num": p.get("round_num"),
                "agent_id": p.get("agent_id"),
                "task_id": p.get("task_id"),
                "text": p.get("text", ""),
                "reply_to": p.get("reply_to") or p.get("parent_id"),
            }
        )
    if not out:
        log.info(
            "cross_task_forum cross-gen query returned 0 rows up to gen=%d "
            "(window=%d) — no cross-task posts yet. Expected when the cross-task "
            "forum is disabled (--cross-task-forum-rounds 0), at gen 1, or when no "
            "posts were produced in the window.",
            inp.generation,
            window,
        )
    elif dropped_pre_window:
        log.info(
            "cross_task_distill: dropped %d posts pre-window (gen<%d) at gen=%d, "
            "kept %d posts. window=%d (env KSI_CROSS_TASK_DISTILL_GEN_WINDOW)",
            dropped_pre_window,
            min_gen,
            inp.generation,
            len(out),
            window,
        )
    return out

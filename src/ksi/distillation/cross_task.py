"""Cross-task distillation: one pure function, one LLM call per generation."""

from __future__ import annotations

import logging
import re
from typing import Any

from ..errors import AuthenticationFailure, is_auth_error
from ..memory.cross_task_context import (  # noqa: F401  re-export for back-compat
    _backbone_indices,
    _norm_tokens,
    _post_generation,
    _post_priority,
    _post_round,
    target_relevance,
)
from ..memory.cross_task_context import approx_token_count as _approx_token_count
from .per_task import (
    _BUNDLE_ITEM_FIELDS,
    _as_insight_list,
    _as_int_list,
    _call_llm,
    _parse_json_lenient,
    _post_id_set,
    dedupe_bundle_items,
)
from .prompts import (
    _fmt_posts,
    _fmt_target_task_section,
    build_cross_task_distill_prompt,
    build_cross_task_distill_prompt_parts,
)
from .types import CrossTaskBundle, LLMCallable

log = logging.getLogger(__name__)

_DEFAULT_CONTEXT_LIMIT_TOKENS = 200_000
_PROMPT_HEADROOM_TOKENS = 8_192
_DEFAULT_INPUT_BUDGET_FRACTION = 0.70
_RETRY_INPUT_BUDGET_FRACTION = 0.60
_PROMPT_TOO_LONG_RE = re.compile(
    r"prompt is too long:\s*(?P<observed>\d+)\s*tokens\s*>\s*(?P<limit>\d+)\s*maximum",
    re.IGNORECASE,
)


def _prompt_budget_tokens(
    max_context_tokens: int,
    *,
    fraction: float,
) -> int:
    return max(8_000, int(max_context_tokens * fraction) - _PROMPT_HEADROOM_TOKENS)


def _estimate_prompt_tokens(
    *,
    cross_posts: list[dict[str, Any]],
    task_source: str | None,
    per_task_transferables: list[dict[str, Any]] | None = None,
    target_task: dict[str, Any] | None = None,
) -> int:
    # The transferables section (KSI_TRANSFER_BRIDGE) counts for budget
    # estimation — it is never trimmed itself, so posts must trim to compensate.
    # The target-task section (target-conditioning) is likewise counted but
    # never trimmed.
    system_prompt, user_prompt = build_cross_task_distill_prompt(
        cross_posts=cross_posts,
        task_source=task_source,
        per_task_transferables=per_task_transferables,
        target_task=target_task,
    )
    return _approx_token_count(system_prompt) + _approx_token_count(user_prompt)


def _select_cross_posts_for_budget(
    *,
    cross_posts: list[dict[str, Any]],
    task_source: str | None,
    max_input_tokens: int,
    per_task_transferables: list[dict[str, Any]] | None = None,
    target_task: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not cross_posts:
        return []

    full_estimate = _estimate_prompt_tokens(
        cross_posts=cross_posts,
        task_source=task_source,
        per_task_transferables=per_task_transferables,
        target_task=target_task,
    )
    if full_estimate <= max_input_tokens:
        return list(cross_posts)

    log.info(
        "distill_cross_task: trimming cross-task history from %d post(s), approx_prompt_tokens=%d, budget=%d",
        len(cross_posts),
        full_estimate,
        max_input_tokens,
    )

    base_tokens = _estimate_prompt_tokens(
        cross_posts=[],
        task_source=task_source,
        per_task_transferables=per_task_transferables,
        target_task=target_task,
    )
    if base_tokens > max_input_tokens:
        target_id = (target_task or {}).get("id") if target_task else None
        log.warning(
            "distill_cross_task: target/task fixed prompt section exceeds budget before posts "
            "(task=%s, approx_prompt_tokens=%d, budget=%d); selecting no cross-task posts",
            target_id,
            base_tokens,
            max_input_tokens,
        )
        return []
    per_post_tokens = [_approx_token_count(_fmt_posts([post])) for post in cross_posts]
    selected_indices: list[int] = []
    selected_set: set[int] = set()
    running_tokens = base_tokens

    def try_add(index: int) -> None:
        nonlocal running_tokens
        if index in selected_set:
            return
        cost = per_post_tokens[index]
        if selected_indices and running_tokens + cost > max_input_tokens:
            return
        if not selected_indices and base_tokens + cost > max_input_tokens:
            return
        selected_indices.append(index)
        selected_set.add(index)
        running_tokens += cost

    # Rank by relevance to the target task (target-conditioning) instead of
    # recency: keep the posts whose vocabulary best covers the target task's
    # prompt. The backbone still guarantees one post per generation, but now
    # picks each generation's MOST relevant post. Falls back to the recency-based
    # _post_priority when there is no target (broadcast/ablation mode), preserving
    # the legacy ordering.
    target_tokens = _norm_tokens(str((target_task or {}).get("prompt") or ""))
    if target_tokens:

        def priority_key(post: dict[str, Any], ordinal: int) -> tuple[Any, ...]:
            text = str(post.get("text") or post.get("content") or "")
            return (
                target_relevance(post, target_tokens),
                1 if _post_round(post) >= 1 else 0,
                1 if post.get("reply_to") else 0,
                min(len(text), 4_000),
                ordinal,
            )
    else:

        def priority_key(post: dict[str, Any], ordinal: int) -> tuple[Any, ...]:
            return _post_priority(post, ordinal=ordinal)

    for index in _backbone_indices(cross_posts, priority=priority_key):
        try_add(index)

    ranked_indices = sorted(
        range(len(cross_posts)),
        key=lambda idx: priority_key(cross_posts[idx], idx),
        reverse=True,
    )
    for index in ranked_indices:
        try_add(index)

    if not selected_indices:
        fallback = min(
            ranked_indices,
            key=lambda idx: (per_post_tokens[idx], -priority_key(cross_posts[idx], idx)[0], idx),
        )
        selected_indices = [fallback]

    selected_order = sorted(selected_indices)
    while len(selected_order) > 1:
        selected_estimate = _estimate_prompt_tokens(
            cross_posts=[cross_posts[idx] for idx in selected_order],
            task_source=task_source,
            per_task_transferables=per_task_transferables,
            target_task=target_task,
        )
        if selected_estimate <= max_input_tokens:
            break
        # Estimator slack can leave the selected set a post or two over budget.
        # Evict the LOWEST-priority post (least target-relevant under
        # target-conditioning, least-recent otherwise) rather than always the
        # oldest-by-index, so a high-relevance early post is not dropped in
        # favour of an irrelevant later one. priority_key breaks ties on the
        # original ordinal, so this stays deterministic → prompt-cache stable,
        # and selected_order remains chronological (remove preserves order).
        drop = min(selected_order, key=lambda idx: priority_key(cross_posts[idx], idx))
        selected_order.remove(drop)
    selected = [cross_posts[idx] for idx in selected_order]

    log.info(
        "distill_cross_task: selected %d/%d cross-task post(s) for budget=%d",
        len(selected),
        len(cross_posts),
        max_input_tokens,
    )
    return selected


def select_shared_cross_posts_for_targets(
    *,
    cross_posts: list[dict[str, Any]],
    task_source: str | None,
    per_task_transferables: list[dict[str, Any]] | None = None,
    target_tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Trim the shared forum history ONCE for a whole target-conditioned
    generation, budgeting against the LARGEST target section.

    Target-conditioning re-sends the same forum history to every unsolved
    target as a ``cache_prefix``. ``distill_cross_task``
    trims per target (``_select_cross_posts_for_budget`` counts the per-target
    target-task section in the budget), so two targets with different section
    sizes would pick DIFFERENT post subsets → different ``cache_prefix`` →
    the shared-history cache is never read (``cache_creation>0, cache_read=0``),
    which is a net cost regression exactly in the over-budget regime the
    optimization targets. Trimming once against the largest target yields a set
    that fits every (smaller) target — because ``system``/``cache_prefix`` are
    target-independent and only the ``suffix`` target section grows with the
    target — so the per-target re-trim inside ``distill_cross_task`` is a
    no-op and every target shares a byte-identical prefix.

    "Largest" is measured by the RENDERED target section
    (``_fmt_target_task_section``), and the actual largest target dict is used
    as the budget probe — NOT a synthetic probe id + longest raw prompt. The
    rendered section includes ``Task ID: {id}`` (real ids, e.g. swebench_pro
    instance ids, are far longer than any placeholder) and applies prompt
    sanitization (``\\r\\n`` collapse), both of which break a raw ``len(prompt)``
    proxy: a target that is largest by raw chars can render a SMALLER section
    than another, letting the "largest" probe under-budget the true largest
    target so its per-target re-trim drops a post and diverges the prefix.
    Probing with the real largest rendered target closes that gap.

    Returns ``[]`` only when the LARGEST target's fixed section alone exceeds
    budget (a pathological giant-prompt case); the caller falls back to
    per-target trimming there (the pre-caching behavior).
    """
    budget = _prompt_budget_tokens(_DEFAULT_CONTEXT_LIMIT_TOKENS, fraction=_DEFAULT_INPUT_BUDGET_FRACTION)
    largest_target = max(
        target_tasks,
        key=lambda t: len(_fmt_target_task_section(t)),
        default={"id": "", "prompt": ""},
    )
    return _select_cross_posts_for_budget(
        cross_posts=cross_posts,
        task_source=task_source,
        max_input_tokens=budget,
        per_task_transferables=per_task_transferables,
        target_task=largest_target,
    )


def _prompt_limit_from_error(exc: Exception) -> tuple[int, int] | None:
    match = _PROMPT_TOO_LONG_RE.search(str(exc))
    if not match:
        return None
    return int(match.group("observed")), int(match.group("limit"))


def distill_cross_task(
    *,
    cross_posts: list[dict[str, Any]],
    llm: LLMCallable,
    task_source: str | None = None,
    bundle_schema: dict[str, Any] | None = None,
    per_task_transferables: list[dict[str, Any]] | None = None,
    target_task: dict[str, Any] | None = None,
) -> CrossTaskBundle | None:
    """Distill a cross-task bundle. Returns None if LLM fails or returns
    unparseable output.

    V2: input is ONLY the cross-task forum history (across all gens). No
    per-task posts (Phase 3 is structurally independent of Phase 2 in V2,
    so distill mirrors that separation). No prior bundle (consumed by
    next-gen at seed time, doesn't feed back into next gen's distill).

    ``task_source`` is an optional domain hint ("arc", "swebench_pro",
    "polyglot", ...) that biases the prompt toward benchmark-specific
    concrete primitives. When omitted, a generic domain hint is used.

    ``per_task_transferables`` (KSI_TRANSFER_BRIDGE): success-derived
    candidates rendered as an extra prompt section. Counted by the budget
    estimator but never trimmed — forum posts trim first, as today.

    ``target_task`` (target-conditioning): ``{"id", "prompt"}`` for the
    downstream task; forwarded to the prompt builder (relevance directive +
    a target-task section after the forum posts) and counted by the budget
    estimator (never trimmed — posts trim to make room). When None, behavior
    is byte-identical to the non-conditioned path.
    """
    input_budget = _prompt_budget_tokens(
        _DEFAULT_CONTEXT_LIMIT_TOKENS,
        fraction=_DEFAULT_INPUT_BUDGET_FRACTION,
    )
    selected_posts = _select_cross_posts_for_budget(
        cross_posts=cross_posts,
        task_source=task_source,
        max_input_tokens=input_budget,
        per_task_transferables=per_task_transferables,
        target_task=target_task,
    )
    if cross_posts and not selected_posts:
        selected_tokens = _estimate_prompt_tokens(
            cross_posts=[],
            task_source=task_source,
            per_task_transferables=per_task_transferables,
            target_task=target_task,
        )
        if selected_tokens > input_budget:
            target_id = (target_task or {}).get("id") if target_task else None
            log.warning(
                "distill_cross_task: skipping LLM call because fixed prompt section exceeds budget "
                "(task=%s, approx_prompt_tokens=%d, budget=%d)",
                target_id,
                selected_tokens,
                input_budget,
            )
            return None

    def _call(posts: list[dict[str, Any]]) -> CrossTaskBundle | None:
        sys_prompt, cache_prefix, suffix = build_cross_task_distill_prompt_parts(
            cross_posts=posts,
            task_source=task_source,
            per_task_transferables=per_task_transferables,
            target_task=target_task,
        )
        # Only cache the shared forum history when target-conditioned: that is
        # the only path where the SAME prefix is re-sent across many targets in
        # a generation, so a cache write pays for itself. For the single
        # non-conditioned call, send it as a plain user string to avoid a
        # pointless cache-write premium. ``cache_prefix +
        # suffix`` is the user message byte-for-byte in both branches.
        if target_task is not None:
            user_prompt = suffix
            call_cache_prefix: str | None = cache_prefix
        else:
            user_prompt = cache_prefix + suffix
            call_cache_prefix = None
        try:
            raw, structured = _call_llm(
                llm,
                sys_prompt,
                user_prompt,
                bundle_schema=bundle_schema,
                cache_prefix=call_cache_prefix,
            )
        except AuthenticationFailure:
            raise
        except Exception as exc:
            if is_auth_error(exc):
                raise AuthenticationFailure(f"LLM authentication failed for cross-task distill: {exc}") from exc
            prompt_limit = _prompt_limit_from_error(exc)
            if prompt_limit is not None and posts:
                observed_tokens, max_context_tokens = prompt_limit
                if len(posts) > 1:
                    retry_budget = _prompt_budget_tokens(
                        max_context_tokens,
                        fraction=_RETRY_INPUT_BUDGET_FRACTION,
                    )
                    trimmed_posts = _select_cross_posts_for_budget(
                        cross_posts=posts,
                        task_source=task_source,
                        max_input_tokens=retry_budget,
                        per_task_transferables=per_task_transferables,
                        target_task=target_task,
                    )
                    if len(trimmed_posts) < len(posts):
                        log.warning(
                            "distill_cross_task: prompt overflow at ~%d>%d tokens; "
                            "retrying with %d/%d post(s) and budget=%d",
                            observed_tokens,
                            max_context_tokens,
                            len(trimmed_posts),
                            len(posts),
                            retry_budget,
                        )
                        return _call(trimmed_posts)
                else:
                    # A single post that overflows real context cannot be trimmed
                    # further — the one-post budget fallback keeps it even when it
                    # exceeds the (soft, conservative) selection budget, so the
                    # retry above never fires for len==1. Drop the oversized post
                    # and distill from the target/task section alone rather than
                    # silently yielding no bundle for this target. (_call([]) can
                    # only overflow again if the base section itself exceeds
                    # context, in which case posts is empty and no further retry
                    # fires — no loop.)
                    log.warning(
                        "distill_cross_task: single post overflowed context (~%d>%d tokens); "
                        "retrying with 0 posts (target-only)",
                        observed_tokens,
                        max_context_tokens,
                    )
                    return _call([])
            log.warning("distill_cross_task: LLM raised %r", exc)
            return None

        if not structured and not (raw or "").strip():
            # Empty response with no structured payload — likely a tool-call
            # decline, not malformed JSON. Logged distinctly.
            log.warning("distill_cross_task: LLM returned an empty response (possible structured-output decline)")
            return None

        if structured:
            # Non-empty schema-constrained dict; an empty ``{}`` is falsy and
            # falls through to the lenient parser instead of yielding a silent
            # all-empty bundle.
            payload: dict | None = structured
        else:
            payload = _parse_json_lenient(raw, label="distill_cross_task")
        if payload is None:
            return None

        valid_post_ids = _post_id_set(posts)

        fields = dedupe_bundle_items(
            {
                field: _as_insight_list(payload.get(field), allowed_post_ids=valid_post_ids)
                for field in _BUNDLE_ITEM_FIELDS
            }
        )

        return CrossTaskBundle(
            **fields,
            # Provenance trust boundary: trusted post-id evidence is
            # membership-filtered against the posts loaded for this round;
            # out-of-range ids are dropped, never trusted. See the matching
            # note in per_task.py.
            evidence_post_ids=_as_int_list(
                payload.get("evidence_post_ids"),
                allowed_values=valid_post_ids,
            ),
        )

    return _call(selected_posts)

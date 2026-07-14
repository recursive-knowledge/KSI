"""Selection + budgeting helpers for the cross-task distiller.

Used by the cross-task distiller (distillation/distiller.py) to choose which
forum posts enter the target-conditioned distill prompt under a token budget.
The cross-task forum no longer loads prior-generation history, so it
does not consume this module. Selection is a pure function of its inputs so the
rendered prefix is byte-stable across the N target consumers within a generation
(prompt-cache safe).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

_APPROX_CHARS_PER_TOKEN = 3


def approx_token_count(text: str) -> int:
    n = len(text)
    return n // _APPROX_CHARS_PER_TOKEN + (1 if n % _APPROX_CHARS_PER_TOKEN else 0)


def _norm_tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9_]+", str(text).lower()))


def _post_generation(post: dict[str, Any]) -> int:
    value = post.get("generation")
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _post_round(post: dict[str, Any]) -> int:
    value = post.get("round_num")
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _post_priority(post: dict[str, Any], *, ordinal: int) -> tuple[int, int, int, int, int]:
    text = str(post.get("text") or post.get("content") or "")
    return (
        1 if _post_round(post) >= 1 else 0,
        1 if post.get("reply_to") else 0,
        _post_generation(post),
        min(len(text), 4_000),
        ordinal,
    )


def target_relevance(post: dict[str, Any], target_tokens: frozenset[str]) -> float:
    """Lexical relevance of a post to the target task, in [0, 1].

    Fraction of the target task's vocabulary covered by the post:
    ``|post_tokens ∩ target_tokens| / |target_tokens|``. Deterministic (no
    embeddings), so relevance-ranked selection stays prompt-cache stable.
    Returns 0.0 when either side has no tokens.
    """
    if not target_tokens:
        return 0.0
    post_tokens = _norm_tokens(str(post.get("text") or post.get("content") or ""))
    if not post_tokens:
        return 0.0
    return len(post_tokens & target_tokens) / len(target_tokens)


def _backbone_indices(
    cross_posts: list[dict[str, Any]],
    *,
    priority: Callable[[dict[str, Any], int], tuple[Any, ...]] | None = None,
) -> list[int]:
    """One representative post per generation, chosen by ``priority``.

    ``priority(post, ordinal)`` returns the sort key (higher = preferred) and
    defaults to the recency-based ``_post_priority``. The distiller passes a
    relevance-based key so each generation's most target-relevant post is kept.
    """
    key = priority if priority is not None else (lambda post, ordinal: _post_priority(post, ordinal=ordinal))
    by_generation: dict[int, list[int]] = {}
    for idx, post in enumerate(cross_posts):
        by_generation.setdefault(_post_generation(post), []).append(idx)

    selected: list[int] = []
    seen: set[int] = set()
    for generation in sorted(by_generation):
        indices = by_generation[generation]
        best_any = max(indices, key=lambda idx: key(cross_posts[idx], idx))
        best_reply = None
        reply_candidates = [idx for idx in indices if _post_round(cross_posts[idx]) >= 1]
        if reply_candidates:
            best_reply = max(
                reply_candidates,
                key=lambda idx: key(cross_posts[idx], idx),
            )
        for candidate in (best_any, best_reply):
            if candidate is None or candidate in seen:
                continue
            seen.add(candidate)
            selected.append(candidate)
    return selected

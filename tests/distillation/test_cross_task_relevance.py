"""Distiller cross-task selection ranks by target relevance, not recency.

Exercises _select_cross_posts_for_budget under a budget tight enough to admit
exactly one post, so which post survives reveals the ranking signal.
"""

from ksi.distillation.cross_task import (
    _estimate_prompt_tokens,
    _select_cross_posts_for_budget,
)
from ksi.distillation.prompts import _fmt_posts
from ksi.memory.cross_task_context import approx_token_count as _approx_token_count

_TASK_SOURCE = "polyglot"
_TARGET = {"id": "target", "prompt": "optimize the sorting algorithm for large arrays quickly"}

# Same generation / round, and padded to identical length so every post costs
# the same number of tokens (approx_token_count is char-based) -- the budget
# below then admits exactly one, whichever ranks highest. Posts differ only in
# relevance to the target and in ordinal (recency). Padding uses '.' which
# _norm_tokens ignores, so it adds cost without adding overlap.
_RAW = {
    1: "optimize the sorting algorithm for large arrays quickly",  # target-relevant
    2: "choose calm pastel colours for the report header banner",  # irrelevant
    3: "remember to water the office plants every wednesday now",  # irrelevant
}
_WIDTH = max(len(v) for v in _RAW.values())
_POSTS = [
    {
        "id": i,
        "generation": 3,
        "round_num": 0,
        "reply_to": None,
        "text": _RAW[i].ljust(_WIDTH, "."),
    }
    for i in (1, 2, 3)
]


def _budget_for_one_post(target_task) -> int:
    base = _estimate_prompt_tokens(cross_posts=[], task_source=_TASK_SOURCE, target_task=target_task)
    one_post_cost = _approx_token_count(_fmt_posts([_POSTS[0]]))
    # Enough for the fixed section + exactly one post, not two (posts are
    # equal-cost, so half a post of margin cannot admit a second).
    return base + one_post_cost + one_post_cost // 2


def test_selection_prefers_relevant_post_over_more_recent_ones():
    selected = _select_cross_posts_for_budget(
        cross_posts=_POSTS,
        task_source=_TASK_SOURCE,
        max_input_tokens=_budget_for_one_post(_TARGET),
        target_task=_TARGET,
    )
    assert [p["id"] for p in selected] == [1], "the target-relevant post must win under a one-post budget"


def test_selection_falls_back_to_recency_without_a_target():
    selected = _select_cross_posts_for_budget(
        cross_posts=_POSTS,
        task_source=_TASK_SOURCE,
        max_input_tokens=_budget_for_one_post(None),
        target_task=None,
    )
    # No target -> recency ordering; equal length means the last (highest
    # ordinal) post wins.
    assert [p["id"] for p in selected] == [3], "without a target, the most recent post must win"


def test_safety_trim_evicts_least_relevant_not_oldest(monkeypatch):
    """The post-selection safety-trim drops the least target-relevant post, not
    merely the oldest-by-index, so a high-relevance early post survives under
    target-conditioning. (Old behavior popped the front / oldest post.)"""
    import ksi.distillation.cross_task as ct

    budget = 1000

    # Force the trim loop: the initial full estimate is over budget (so no fast
    # return), the base section is tiny, and the selected set only fits once a
    # single post remains. Keyed on how many posts are being estimated so the
    # loop trims 3 -> 2 -> 1. per_post_tokens uses the real estimator, and the
    # large max_input_tokens lets try_add admit all three first.
    def fake_estimate(*, cross_posts, task_source, per_task_transferables=None, target_task=None):
        n = len(cross_posts)
        if n == 0:
            return 10
        return budget + 100 if n >= 2 else budget - 10

    monkeypatch.setattr(ct, "_estimate_prompt_tokens", fake_estimate)

    selected = ct._select_cross_posts_for_budget(
        cross_posts=_POSTS,
        task_source=_TASK_SOURCE,
        max_input_tokens=budget,
        target_task=_TARGET,
    )
    # Old pop(0) trim would leave the newest irrelevant post (id 3); relevance-
    # aware eviction keeps the target-relevant post (id 1).
    assert [p["id"] for p in selected] == [1]

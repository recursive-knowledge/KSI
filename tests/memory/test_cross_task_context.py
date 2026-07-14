from kcsi.memory.cross_task_context import (
    _backbone_indices,
    _norm_tokens,
    approx_token_count,
    target_relevance,
)


def test_approx_token_count_matches_3_chars_per_token():
    assert approx_token_count("") == 0
    assert approx_token_count("abc") == 1
    assert approx_token_count("abcd") == 2  # 4//3 + 1


def test_target_relevance_is_target_coverage_fraction():
    target = _norm_tokens("verify the output grid shape before submitting")
    # Post shares 3 of the target's tokens (verify, grid, shape).
    post = {"text": "always verify the grid shape twice"}
    # target tokens: {verify, the, output, grid, shape, before, submitting} = 7
    # intersection: {verify, the, grid, shape} = 4
    assert target_relevance(post, target) == 4 / 7


def test_target_relevance_zero_when_no_overlap_or_empty_target():
    target = _norm_tokens("rust ownership and borrow checker")
    assert target_relevance({"text": "python list comprehension"}, target) == 0.0
    assert target_relevance({"text": "anything"}, frozenset()) == 0.0
    assert target_relevance({"text": ""}, target) == 0.0


def test_backbone_default_is_recency_based_and_one_per_generation():
    posts = [
        {"id": 1, "generation": 1, "text": "gen one post"},
        {"id": 2, "generation": 1, "text": "gen one other post"},
        {"id": 3, "generation": 2, "text": "gen two post"},
    ]
    idxs = _backbone_indices(posts)
    gens = {posts[i]["generation"] for i in idxs}
    assert gens == {1, 2}, "every generation must be represented"


def test_backbone_priority_key_selects_most_relevant_post_per_generation():
    target = _norm_tokens("optimize the sorting algorithm for large arrays")
    posts = [
        # gen 1: second post is far more relevant to the target
        {"id": 1, "generation": 1, "text": "unrelated note about colors"},
        {"id": 2, "generation": 1, "text": "optimize the sorting algorithm for large arrays"},
        # gen 2: single post
        {"id": 3, "generation": 2, "text": "sorting arrays quickly"},
    ]

    def priority_key(post, ordinal):
        return (target_relevance(post, target), 0, 0, min(len(str(post.get("text") or "")), 4_000), ordinal)

    idxs = _backbone_indices(posts, priority=priority_key)
    chosen_ids = {posts[i]["id"] for i in idxs}
    # gen 1 backbone must be the relevant post (id=2), NOT the recency default (id=1 or 2 by ordinal).
    assert 2 in chosen_ids
    assert 1 not in chosen_ids
    assert 3 in chosen_ids  # gen 2 still represented


def test_backbone_reexport_still_importable_from_cross_task():
    from kcsi.distillation.cross_task import _backbone_indices, _select_cross_posts_for_budget

    assert callable(_backbone_indices)
    assert callable(_select_cross_posts_for_budget)

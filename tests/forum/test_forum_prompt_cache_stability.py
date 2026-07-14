"""Cache-stability tests for the forum prompt builders.

PR #564 introduced V2 forum prompts that embed three categories of
per-agent / per-generation content into the prompt body that downstream
adapters attach `cache_control` to:

  * `prior_gen_posts` — grows monotonically across generations
  * `native_memory` — per-agent Phase-1 working memory
  * `peer_posts_this_gen` — varies per agent per round in cross-task

When that content sits inside the cache_control'd block the cache key
changes every call, breaking prompt caching (cache_read=0 across all
forum_round_* and cross_task_forum_round_* phases). The fix splits
each forum prompt into a cache-stable ``cacheable_prefix`` and a
``variable_suffix``, with the cache_control marker placed only on the
prefix block.

These tests pin the invariant: the cacheable prefix must be byte-stable
across calls that vary ONLY the per-agent / per-generation suffix
inputs.

See: feedback_cache_stability_invariant, project_pr564_cache_regression.
"""

from __future__ import annotations

from ksi.forum import (
    build_cross_task_discussion_parts,
    build_per_task_discussion_parts,
)

# ---------------------------------------------------------------------------
# Per-task: prefix is stable across varying prior_gen_posts / native_memory
# ---------------------------------------------------------------------------


def test_per_task_prefix_stable_across_prior_posts_growth():
    """Adding posts to prior_gen_posts must NOT change cacheable_prefix.

    This is the load-bearing invariant for cross-generation prompt
    caching: gen N's prefix must equal gen N+1's prefix when the only
    difference is appended posts.
    """
    base_kwargs = dict(
        agent_id="a1",
        generation=3,
        traces=[],
        task_ids=["t1", "t2"],
        task_descriptions={"t1": "alpha task", "t2": "beta task"},
    )
    parts_few = build_per_task_discussion_parts(
        **base_kwargs,
        prior_gen_posts=[
            {"id": 1, "agent_id": "a2", "generation": 1, "text": "older post"},
        ],
    )
    parts_many = build_per_task_discussion_parts(
        **base_kwargs,
        prior_gen_posts=[
            {"id": 1, "agent_id": "a2", "generation": 1, "text": "older post"},
            {"id": 2, "agent_id": "a3", "generation": 1, "text": "another post"},
            {"id": 3, "agent_id": "a4", "generation": 2, "text": "newer post"},
        ],
    )
    assert parts_few.cacheable_prefix == parts_many.cacheable_prefix
    # The prior posts MUST land in the suffix (otherwise growth would
    # silently leak into the cached prefix).
    assert "older post" in parts_few.variable_suffix
    assert "newer post" in parts_many.variable_suffix
    assert "older post" not in parts_few.cacheable_prefix
    assert "newer post" not in parts_many.cacheable_prefix


def test_per_task_prefix_stable_across_native_memory_changes():
    """Per-agent native memory must NOT change cacheable_prefix.

    Two sibling agents in the same generation/task get different
    Phase-1 native memory but share the rest of the prompt — the
    prefix must be byte-identical so they share cache entries.
    """
    base_kwargs = dict(
        agent_id="a1",
        generation=1,
        traces=[],
        task_ids=["t1"],
        task_descriptions={"t1": "alpha task"},
    )
    parts_a = build_per_task_discussion_parts(
        **base_kwargs,
        native_memory="agent A's notes: X failed because of Y",
    )
    parts_b = build_per_task_discussion_parts(
        **base_kwargs,
        native_memory="agent B's notes: completely different observation Z",
    )
    assert parts_a.cacheable_prefix == parts_b.cacheable_prefix
    assert "agent A's notes" in parts_a.variable_suffix
    assert "agent B's notes" in parts_b.variable_suffix


def test_per_task_prefix_stable_across_repeated_calls():
    """Calling the parts builder twice with identical inputs MUST produce
    byte-identical prefixes.

    This is a sanity check that the prefix is deterministic — no
    embedded timestamps, run-IDs, or other non-determinism leaks into
    the cached block.
    """
    kwargs = dict(
        agent_id="a1",
        generation=2,
        traces=[],
        task_ids=["t1"],
        task_descriptions={"t1": "alpha task"},
        prior_gen_posts=[
            {"id": 1, "agent_id": "a2", "generation": 1, "text": "older post"},
        ],
        native_memory="phase-1 notes",
    )
    p1 = build_per_task_discussion_parts(**kwargs)
    p2 = build_per_task_discussion_parts(**kwargs)
    assert p1.cacheable_prefix == p2.cacheable_prefix
    assert p1.variable_suffix == p2.variable_suffix


# ---------------------------------------------------------------------------
# Cross-task: prefix is stable across varying peer_posts_this_gen / phase1
# ---------------------------------------------------------------------------


def test_cross_task_prefix_stable_across_peer_posts():
    """peer_posts_this_gen varies per agent per round and MUST be in suffix."""
    base_kwargs = dict(
        agent_id="a1",
        generation=2,
        round_num=1,
        cross_task_history=[
            {"id": 1, "agent_id": "a2", "generation": 1, "round_num": 0, "text": "history"},
        ],
    )
    parts_no_peers = build_cross_task_discussion_parts(
        **base_kwargs,
        peer_posts_this_gen=[],
    )
    parts_with_peers = build_cross_task_discussion_parts(
        **base_kwargs,
        peer_posts_this_gen=[
            {"id": 9, "agent_id": "a2", "round_num": 0, "text": "PEER_POST_1"},
            {"id": 10, "agent_id": "a3", "round_num": 0, "text": "PEER_POST_2"},
        ],
    )
    assert parts_no_peers.cacheable_prefix == parts_with_peers.cacheable_prefix
    # Peer content MUST appear in the suffix.
    assert "PEER_POST_1" in parts_with_peers.variable_suffix
    assert "PEER_POST_2" in parts_with_peers.variable_suffix
    # And MUST NOT leak into the prefix.
    assert "PEER_POST_1" not in parts_with_peers.cacheable_prefix


def test_cross_task_prefix_stable_across_phase1_context():
    """phase1_context is per-agent (each agent attempted a different task)
    and MUST sit in the suffix for cross-agent prefix sharing."""
    base_kwargs = dict(
        agent_id="a1",
        generation=2,
        round_num=0,
        cross_task_history=[
            {"id": 1, "agent_id": "a2", "generation": 1, "round_num": 0, "text": "history"},
        ],
    )
    parts_a = build_cross_task_discussion_parts(
        **base_kwargs,
        phase1_context={
            "task_id": "task-alpha",
            "native_score": 1.0,
            "eval_result": {"resolved": True},
            "reflection": "AGENT_A_REFLECTION_MARKER",
        },
    )
    parts_b = build_cross_task_discussion_parts(
        **base_kwargs,
        phase1_context={
            "task_id": "task-beta",
            "native_score": 0.0,
            "eval_result": {"resolved": False},
            "reflection": "AGENT_B_REFLECTION_MARKER",
        },
    )
    assert parts_a.cacheable_prefix == parts_b.cacheable_prefix
    assert "AGENT_A_REFLECTION_MARKER" in parts_a.variable_suffix
    assert "AGENT_B_REFLECTION_MARKER" in parts_b.variable_suffix


def test_cross_task_prefix_stable_across_two_round_calls():
    """Round 0 and round 1 produce different prefixes (round-specific
    instructions live in the prefix), but two same-round calls with
    different peer posts share a prefix."""
    history = [
        {"id": 1, "agent_id": "a2", "generation": 1, "round_num": 0, "text": "h1"},
    ]
    r1_a = build_cross_task_discussion_parts(
        agent_id="a1",
        generation=2,
        round_num=1,
        cross_task_history=history,
        peer_posts_this_gen=[
            {"id": 9, "agent_id": "a2", "round_num": 0, "text": "p1"},
        ],
    )
    r1_b = build_cross_task_discussion_parts(
        agent_id="a1",
        generation=2,
        round_num=1,
        cross_task_history=history,
        peer_posts_this_gen=[
            {"id": 9, "agent_id": "a2", "round_num": 0, "text": "p1"},
            {"id": 10, "agent_id": "a3", "round_num": 0, "text": "p2"},
        ],
    )
    r0 = build_cross_task_discussion_parts(
        agent_id="a1",
        generation=2,
        round_num=0,
        cross_task_history=history,
        peer_posts_this_gen=[],
    )
    # Same round → same prefix.
    assert r1_a.cacheable_prefix == r1_b.cacheable_prefix
    # Different round → different prefix (round-specific instructions
    # legitimately differ).
    assert r0.cacheable_prefix != r1_a.cacheable_prefix


# ---------------------------------------------------------------------------
# Empty inputs: parts builder must still produce non-trivial prefix
# ---------------------------------------------------------------------------


def test_per_task_prefix_nonempty_with_no_optional_inputs():
    parts = build_per_task_discussion_parts(
        agent_id="a1",
        generation=1,
        traces=[],
        task_ids=["t1"],
    )
    assert "PER-TASK POST-MORTEM" in parts.cacheable_prefix
    assert "MUST call `forum_signal_done" in parts.cacheable_prefix


def test_cross_task_prefix_nonempty_with_no_optional_inputs():
    parts = build_cross_task_discussion_parts(
        agent_id="a1",
        generation=1,
        round_num=0,
    )
    assert "CROSS-TASK FORUM" in parts.cacheable_prefix
    assert "MUST call `forum_signal_done" in parts.cacheable_prefix

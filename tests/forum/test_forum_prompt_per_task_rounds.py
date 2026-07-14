"""Round-threading tests for the per-task discussion prompt builder (#1043).

`build_per_task_discussion_parts` never received `round_num`, and
`prior_gen_posts` only ever loaded strictly-earlier-*generation* posts — so
within a single generation, round 1+ reproduced the exact same prompt as
round 0 (no peer-post wiring like the cross-task forum prompt builder
already has). These tests pin the fix: `round_num` and same-generation
`peer_posts_this_gen` must both be threaded through and must change the
rendered prompt, mirroring `build_cross_task_discussion_parts`.
"""

from kcsi.forum import build_per_task_discussion_parts


def _full(parts) -> str:
    return parts.cacheable_prefix + parts.variable_suffix


def test_round0_and_round1_prompts_differ_with_peer_posts():
    """Round 1's prompt must differ from round 0's when there are
    same-generation peer posts to include — the core bug: before the fix,
    round_num wasn't threaded at all, so round 1 reproduced round 0's
    prompt byte-for-byte."""
    base_kwargs = dict(
        agent_id="agent-1",
        generation=2,
        traces=[],
        task_ids=["t1"],
        task_descriptions={"t1": "alpha task"},
    )
    round0 = build_per_task_discussion_parts(**base_kwargs, round_num=0, peer_posts_this_gen=[])
    round1 = build_per_task_discussion_parts(
        **base_kwargs,
        round_num=1,
        peer_posts_this_gen=[
            {"id": 42, "agent_id": "agent-2", "round_num": 0, "text": "PEER_POST_MARKER"},
        ],
    )
    assert _full(round0) != _full(round1)
    assert "PEER_POST_MARKER" in _full(round1)
    assert "PEER_POST_MARKER" not in _full(round0)


def test_round_num_appears_in_header():
    parts = build_per_task_discussion_parts(
        agent_id="agent-1",
        generation=1,
        traces=[],
        task_ids=["t1"],
        round_num=2,
    )
    assert "round 2" in _full(parts)


def test_peer_posts_land_in_variable_suffix_not_prefix():
    """Peer posts vary per round/agent and must not poison the cacheable
    prefix (same invariant as cross-task's peer_posts_this_gen)."""
    parts = build_per_task_discussion_parts(
        agent_id="agent-1",
        generation=1,
        traces=[],
        task_ids=["t1"],
        round_num=1,
        peer_posts_this_gen=[
            {"id": 7, "agent_id": "agent-2", "round_num": 0, "text": "SUFFIX_ONLY_MARKER"},
        ],
    )
    assert "SUFFIX_ONLY_MARKER" in parts.variable_suffix
    assert "SUFFIX_ONLY_MARKER" not in parts.cacheable_prefix


def test_prefix_stable_across_peer_posts_growth_same_round():
    """Two agents in the same round with different peer-post visibility
    must still share a cacheable prefix."""
    base_kwargs = dict(
        agent_id="agent-1",
        generation=3,
        traces=[],
        task_ids=["t1"],
        round_num=1,
    )
    few = build_per_task_discussion_parts(
        **base_kwargs,
        peer_posts_this_gen=[{"id": 1, "agent_id": "a2", "round_num": 0, "text": "p1"}],
    )
    many = build_per_task_discussion_parts(
        **base_kwargs,
        peer_posts_this_gen=[
            {"id": 1, "agent_id": "a2", "round_num": 0, "text": "p1"},
            {"id": 2, "agent_id": "a3", "round_num": 0, "text": "p2"},
        ],
    )
    assert few.cacheable_prefix == many.cacheable_prefix


def test_prefix_differs_across_rounds():
    """Round-specific instructions legitimately differ, mirroring
    `test_cross_task_prefix_stable_across_two_round_calls`."""
    base_kwargs = dict(
        agent_id="agent-1",
        generation=1,
        traces=[],
        task_ids=["t1"],
    )
    r0 = build_per_task_discussion_parts(**base_kwargs, round_num=0, peer_posts_this_gen=[])
    r1 = build_per_task_discussion_parts(**base_kwargs, round_num=1, peer_posts_this_gen=[])
    assert r0.cacheable_prefix != r1.cacheable_prefix


def test_round0_defaults_when_round_num_omitted():
    """`round_num` must default to 0 so existing single-round callers are
    unaffected (behavior-preserving at the default `--per-task-forum-rounds
    1`)."""
    parts = build_per_task_discussion_parts(
        agent_id="agent-1",
        generation=1,
        traces=[],
        task_ids=["t1"],
    )
    assert "round 0" in _full(parts)


def test_gen1_disclaimer_suppressed_when_same_gen_peer_posts_exist():
    """The gen-1 'no prior posts' hallucination-guard disclaimer should not
    fire when round 1+ genuinely has same-generation peer posts to cite —
    otherwise it would tell agents not to cite IDs that legitimately exist."""
    parts = build_per_task_discussion_parts(
        agent_id="agent-1",
        generation=1,
        traces=[],
        task_ids=["t1"],
        round_num=1,
        peer_posts_this_gen=[{"id": 1, "agent_id": "a2", "round_num": 0, "text": "real peer post"}],
    )
    text = _full(parts)
    assert "no prior posts exist" not in text.lower() or "First generation" not in text

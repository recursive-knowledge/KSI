"""Cache-split test: agent_id must not embed into the cacheable prefix.

Task 5: the per-agent identity line was previously the first line of
`cacheable_prefix` in `build_cross_task_discussion_parts`, which meant every
agent hashed a different prefix and cross-agent prompt caching never fired.
This pins the fix: the identity line moves to the top of `variable_suffix`,
and the prefix becomes byte-identical across agents in a (generation, round).

The parametrization exercises every prefix-shaping branch (gen-1 empty-history
disclaimer, round>=1 directive + peers, phase1_context present) so a future
regression that reintroduces per-agent content into ANY branch of the prefix
is caught — not just the single gen-2/round-0 case.
"""

from __future__ import annotations

import pytest

from kcsi.forum.prompt import build_cross_task_discussion_parts

HISTORY = [{"id": 1, "generation": 1, "round_num": 0, "agent_id": "a0", "text": "primitive X"}]
PHASE1 = {"task_id": "t-solo", "native_score": 1.0, "reflection": "used cargo test --nocapture"}
PEERS = [{"id": 9, "round_num": 0, "agent_id": "a3", "text": "peer primitive Y"}]

# (generation, round_num, phase1_context, cross_task_history, peers) — one row
# per prefix-shaping branch.
CONDITIONS = [
    pytest.param(2, 0, None, HISTORY, [], id="gen2-round0"),
    pytest.param(1, 0, None, [], [], id="gen1-empty-history-disclaimer"),
    pytest.param(3, 1, None, HISTORY, PEERS, id="round1-with-peers"),
    pytest.param(2, 0, PHASE1, HISTORY, [], id="with-phase1-context"),
    pytest.param(3, 2, PHASE1, HISTORY, PEERS, id="round2-phase1-peers"),
]


def _parts(agent_id, *, generation, round_num, phase1_context, cross_task_history, peers):
    return build_cross_task_discussion_parts(
        agent_id=agent_id,
        generation=generation,
        round_num=round_num,
        phase1_context=phase1_context,
        cross_task_history=cross_task_history,
        peer_posts_this_gen=peers,
    )


@pytest.mark.parametrize("generation,round_num,phase1_context,history,peers", CONDITIONS)
def test_cacheable_prefix_is_identical_across_agents(generation, round_num, phase1_context, history, peers):
    kw = dict(
        generation=generation,
        round_num=round_num,
        phase1_context=phase1_context,
        cross_task_history=history,
        peers=peers,
    )
    a = _parts("agent-0", **kw)
    b = _parts("agent-7", **kw)
    assert a.cacheable_prefix == b.cacheable_prefix, "prefix must not embed agent_id"
    assert "agent-0" not in a.cacheable_prefix
    assert "agent-7" not in b.cacheable_prefix


def test_identity_line_moved_to_suffix():
    a = _parts(
        "agent-0",
        generation=2,
        round_num=0,
        phase1_context=None,
        cross_task_history=HISTORY,
        peers=[],
    )
    assert "You are agent agent-0" in a.variable_suffix


@pytest.mark.parametrize("generation,round_num,phase1_context,history,peers", CONDITIONS)
def test_identity_appears_exactly_once_in_as_text(generation, round_num, phase1_context, history, peers):
    # Guards both loss (count 0) and duplication (count 2) of the identity
    # line in the rendered prompt after the prefix->suffix relocation.
    a = _parts(
        "agent-0",
        generation=generation,
        round_num=round_num,
        phase1_context=phase1_context,
        cross_task_history=history,
        peers=peers,
    )
    assert a.as_text().count("You are agent agent-0 in generation") == 1

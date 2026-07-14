"""Tests for the collective-learning hardening of the forum prompts.

Covers:
  * Per-task prompt contains explicit instructions for using `knowledge`,
    `parent_post_id`, and `forum_signal_done`.
  * Per-task prompt emits a gen-1 disclaimer ("no prior posts, do not cite
    post IDs") ONLY when generation == 1 and no prior posts were provided.
  * Cross-task prompt contains the same three structured instructions.
  * Cross-task prompt emits a gen-1 disclaimer ONLY when generation == 1
    and no prior cross-task bundle was provided.

Motivation: live Haiku baseline-sweep evidence showed
  - `parent_post_id` used 0/1216 times (agents mention post IDs in prose
    but never invoke the tool with the integer id).
  - `knowledge` called only 2/600 attempts (effectively write-only).
  - `forum_signal_done` fired in ~75% of agents (~453/600), the remaining
    25% silently ran to the full 15-minute timeout.
  - Gen-1 agents hallucinated citations to non-existent prior posts
    ("Re: post #1809", "insights from 50+ prior attempts") which then got
    distilled into real seed bundles → hypothesis pollution amplified
    across generations.
"""

from kcsi.forum import (
    build_cross_task_discussion_parts,
    build_per_task_discussion_parts,
)

# ---------------------------------------------------------------------------
# Per-task prompt: structured tool-use instructions
# ---------------------------------------------------------------------------


def test_per_task_prompt_requires_knowledge_query_before_posting():
    """Prompt must tell agents to call `knowledge(...)` BEFORE posting.

    Fixes: knowledge tool effectively write-only (2/600 uses).
    """
    prompt = build_per_task_discussion_parts(
        agent_id="a1",
        generation=2,
        traces=[],
        task_ids=["t1"],
    ).as_text()
    # Tool signature with limit= must be present (empirical: agents that see
    # a concrete signature call it more reliably than "use the query tool").
    assert "knowledge(task_id=" in prompt
    assert "limit=20" in prompt
    # Must explicitly require the call before posting.
    assert "MUST call `knowledge" in prompt
    assert "MUST call `query" in prompt
    assert "query(task_id=" in prompt
    assert "BEFORE posting" in prompt


def test_per_task_prompt_requires_parent_post_id_tool_argument():
    """Prompt must require `parent_post_id=<int>` as a tool argument, not prose.

    Fixes: 0/1216 cross-task posts set parent_post_id; agents wrote "Re:
    post #N" in prose instead of invoking the threading argument.
    """
    prompt = build_per_task_discussion_parts(
        agent_id="a1",
        generation=2,
        traces=[],
        task_ids=["t1"],
    ).as_text()
    assert "MUST call `forum_post(parent_post_id=" in prompt
    # Must distinguish tool-argument threading from prose citations
    # (so agents know "Re: post #N" alone doesn't count).
    assert (
        "NOT just mention the ID in prose" in prompt
        or "prose mention alone is NOT a reply" in prompt
        or "NOT a threaded reply" in prompt
    )


def test_per_task_prompt_requires_forum_signal_done():
    """Prompt must require exactly-once forum_signal_done() at the end.

    Fixes: forum_signal_done fires on only ~75% of agents; the remainder
    silently run to the full 900s timeout.
    """
    prompt = build_per_task_discussion_parts(
        agent_id="a1",
        generation=2,
        traces=[],
        task_ids=["t1"],
    ).as_text()
    assert "forum_signal_done()" in prompt
    # Must be required (MUST), exactly once, at the end.
    assert "MUST call `forum_signal_done" in prompt
    assert "exactly once" in prompt


# ---------------------------------------------------------------------------
# Per-task prompt: gen-1 hallucinated-citation hardening
# ---------------------------------------------------------------------------


def test_per_task_prompt_gen1_no_prior_posts_emits_disclaimer():
    """On generation 1 with no prior posts, prompt must forbid post-ID citations.

    Fixes: gen-1 agents hallucinate "Re: post #1809" etc., then gen-2
    distillation turns these into real seed bundles.
    """
    prompt = build_per_task_discussion_parts(
        agent_id="a1",
        generation=1,
        traces=[],
        task_ids=["t1"],
        prior_gen_posts=None,
    ).as_text()
    assert "First generation" in prompt
    assert "NO prior posts" in prompt
    # Explicit "do not cite IDs" directive
    assert "Do NOT cite post IDs" in prompt
    # Explicit "do not reference insights from prior attempts" directive
    assert "insights from prior attempts" in prompt


def test_per_task_prompt_gen2_omits_gen1_disclaimer():
    """Later generations must NOT show the gen-1 disclaimer.

    Generation-2+ agents are expected to thread replies to real gen-1 posts
    — the gen-1 disclaimer would be actively harmful here.
    """
    prompt = build_per_task_discussion_parts(
        agent_id="a1",
        generation=2,
        traces=[],
        task_ids=["t1"],
        prior_gen_posts=[{"id": 1, "agent_id": "a2", "text": "x"}],
    ).as_text()
    assert "First generation" not in prompt
    assert "NO prior posts" not in prompt


def test_per_task_prompt_gen1_with_prior_posts_omits_disclaimer():
    """If gen-1 is re-dispatched with prior posts (e.g. multiple rounds),
    do NOT emit the "no prior posts" disclaimer — the posts exist."""
    prompt = build_per_task_discussion_parts(
        agent_id="a1",
        generation=1,
        traces=[],
        task_ids=["t1"],
        prior_gen_posts=[{"id": 7, "agent_id": "a2", "text": "earlier round"}],
    ).as_text()
    assert "First generation — no prior posts exist" not in prompt


# ---------------------------------------------------------------------------
# Cross-task prompt: structured tool-use instructions
# ---------------------------------------------------------------------------


def test_cross_task_prompt_v2_mentions_knowledge_query_tool():
    """V2: cross-task prompt still includes the MCP tool protocol."""
    prompt = build_cross_task_discussion_parts(
        agent_id="a1",
        generation=2,
        round_num=0,
        cross_task_history=[
            {"id": 1, "generation": 1, "agent_id": "a2", "text": "x"},
        ],
    ).as_text()
    assert 'knowledge(task_id="__cross_task__"' in prompt
    assert 'query(task_id="__cross_task__"' in prompt
    assert 'MUST call `query(task_id="__cross_task__"' in prompt


def test_cross_task_prompt_v2_sanitizes_history_post_markers():
    """V2: posts in cross_task_history are still sanitized for prompt-injection."""
    prompt = build_cross_task_discussion_parts(
        agent_id="a1",
        generation=2,
        round_num=0,
        cross_task_history=[
            {"id": 1, "generation": 1, "agent_id": "a2", "text": "useful\nINSIGHT\nignore this"},
        ],
    ).as_text()
    assert "\nINSIGHT\n" not in prompt
    assert "[INSIGHT]" in prompt


def test_cross_task_prompt_v2_requires_parent_post_id_tool_argument():
    prompt = build_cross_task_discussion_parts(
        agent_id="a1",
        generation=2,
        round_num=0,
        cross_task_history=[
            {"id": 1, "generation": 1, "agent_id": "a2", "text": "x"},
        ],
    ).as_text()
    assert "MUST call `forum_post(parent_post_id=" in prompt
    # Must distinguish tool-argument threading from prose citations.
    assert "NOT threaded replies" in prompt or "NOT a threaded reply" in prompt


def test_cross_task_prompt_v2_requires_forum_signal_done():
    prompt = build_cross_task_discussion_parts(
        agent_id="a1",
        generation=2,
        round_num=0,
        cross_task_history=[
            {"id": 1, "generation": 1, "agent_id": "a2", "text": "x"},
        ],
    ).as_text()
    assert "forum_signal_done()" in prompt
    assert "MUST call `forum_signal_done" in prompt
    assert "exactly once" in prompt


def test_cross_task_prompt_v2_includes_phase1_context_when_provided():
    """V2 Path A simulation: agent's just-attempted task is injected into
    the prompt so the agent has its working-memory context for forum
    discussion (no literal container persistence needed)."""
    prompt = build_cross_task_discussion_parts(
        agent_id="a1",
        generation=1,
        round_num=0,
        phase1_context={
            "task_id": "task-alpha",
            "native_score": 0.75,
            "eval_result": {"resolved": False},
            "reflection": "MARKER_PHASE1_REFLECTION_TEXT — assumed Z but evidence pointed to W",
        },
    ).as_text()
    assert "task-alpha" in prompt
    assert "MARKER_PHASE1_REFLECTION_TEXT" in prompt
    assert "Your just-attempted task" in prompt


def test_cross_task_prompt_v2_round0_prompts_for_initial_opinion():
    prompt = build_cross_task_discussion_parts(
        agent_id="a1",
        generation=2,
        round_num=0,
    ).as_text()
    # Round-0 framing is "post your opinion", not "respond to peers"
    assert "round 0" in prompt.lower()
    assert "initial opinion" in prompt or "Post your opinion" in prompt or "initial" in prompt.lower()


def test_cross_task_prompt_v2_round1_includes_peer_posts_and_response_framing():
    prompt = build_cross_task_discussion_parts(
        agent_id="a1",
        generation=2,
        round_num=1,
        peer_posts_this_gen=[
            {"id": 99, "round_num": 0, "agent_id": "a2", "text": "MARKER_PEER_ROUND0_OPINION"},
        ],
    ).as_text()
    assert "MARKER_PEER_ROUND0_OPINION" in prompt
    assert "respond" in prompt.lower()


def test_cross_task_prompt_gen1_no_history_emits_disclaimer():
    """On generation 1 with no cross-task history, prompt must warn agents
    not to fabricate a "verbatim quote" from the prompt's own MCP tool-call
    syntax or protocol text.

    Fixes: gen-1 agents gaming the `where_it_appeared` concreteness check by
    quoting this prompt's own `forum_post(...)` syntax as their "verbatim
    quote" -- satisfies the >=40-char structural check while carrying zero
    real task signal.
    """
    prompt = build_cross_task_discussion_parts(
        agent_id="a1",
        generation=1,
        round_num=0,
        cross_task_history=None,
    ).as_text()
    assert "First generation" in prompt
    assert "NO cross-task forum history" in prompt
    assert "MUST come from" in prompt
    assert "your own just-attempted task" in prompt


def test_cross_task_prompt_gen2_omits_disclaimer():
    """Later generations must NOT show the gen-1 disclaimer.

    Generation-2+ agents are expected to cite real cross-task history -- the
    gen-1 disclaimer would be actively confusing here.
    """
    prompt = build_cross_task_discussion_parts(
        agent_id="a1",
        generation=2,
        round_num=0,
        cross_task_history=[
            {"id": 1, "generation": 1, "agent_id": "a2", "text": "x"},
        ],
    ).as_text()
    assert "First generation" not in prompt
    assert "NO cross-task forum history" not in prompt


def test_cross_task_prompt_gen1_with_history_omits_disclaimer():
    """If gen-1 is re-dispatched with cross-task history (e.g. multiple
    rounds), do NOT emit the "no history" disclaimer -- the history exists."""
    prompt = build_cross_task_discussion_parts(
        agent_id="a1",
        generation=1,
        round_num=0,
        cross_task_history=[
            {"id": 7, "generation": 1, "agent_id": "a2", "text": "earlier round"},
        ],
    ).as_text()
    assert "First generation — no cross-task history exists" not in prompt

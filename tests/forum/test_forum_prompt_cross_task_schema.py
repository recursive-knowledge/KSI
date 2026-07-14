"""Pin the structured JSON schema for cross-task forum posts (Plan A).

Cross-task forum agents must emit a single JSON object per post with a
mandatory `concrete_primitive` slot. The schema is what blocks meta-advice
at write time. Tests pin the schema-shaping clauses so a future refactor
that quietly drops them trips CI."""

from kcsi.forum import build_cross_task_discussion_parts


def _full(parts) -> str:
    return parts.cacheable_prefix + parts.variable_suffix


def test_round0_prompt_demands_structured_json():
    parts = build_cross_task_discussion_parts(
        agent_id="agent-1",
        generation=1,
        round_num=0,
        phase1_context={
            "task_id": "rust__wordy",
            "native_score": 1.0,
            "eval_result": {"resolved": True},
            "reflection": "tokenization split on whitespace, used i64 overflow check",
        },
        cross_task_history=[],
        peer_posts_this_gen=[],
    )
    text = _full(parts)
    # New schema markers
    assert "concrete_primitive" in text
    assert "task_grounding" in text
    assert "transfer_claim" in text
    assert "anti_meta_self_check" in text
    assert "evidence_task_ids" in text
    # JSON-only output discipline
    assert "single JSON object" in text
    # Forces concrete shape (one of the clauses naming code-shaped tokens)
    assert "API call" in text or "function" in text
    # The shared anti-meta block must render in the prompt
    assert "STRICT ANTI-META RULES" in text


def test_round0_prompt_drops_old_freeform_directive():
    parts = build_cross_task_discussion_parts(
        agent_id="agent-1",
        generation=1,
        round_num=0,
        phase1_context={
            "task_id": "rust__wordy",
            "native_score": 1.0,
            "eval_result": {"resolved": True},
            "reflection": "x",
        },
    )
    text = _full(parts)
    # Old prompt said "Post 1-3 short messages" — replaced with single JSON.
    # If both phrases co-exist agents will pick the easier path. Pin the change.
    assert "Post 1-3 short messages" not in text


def test_round1_response_prompt_keeps_schema():
    parts = build_cross_task_discussion_parts(
        agent_id="agent-2",
        generation=1,
        round_num=1,
        phase1_context={
            "task_id": "javascript__queen-attack",
            "native_score": 0.0,
            "eval_result": {"resolved": False},
            "reflection": "missed |Δrow| == |Δcol| diagonal check",
        },
        cross_task_history=[],
        peer_posts_this_gen=[
            {"id": 42, "agent_id": "agent-1", "round_num": 0, "text": "x"},
        ],
    )
    text = _full(parts)
    # Round-1 response must still be structured.
    assert "concrete_primitive" in text
    # Round-1 must add response semantics (agree/disagree/synthesize).
    assert "AGREE" in text or "DISAGREE" in text or "SYNTHESIZE" in text
    # Round 1 enforcement requires evidence_task_ids — the prompt must name it
    # so a prompt-faithful post is not rejected at the MCP boundary (621-1).
    assert "evidence_task_ids" in text


def test_anti_meta_block_imported_not_inlined():
    """The forum prompt and distiller must reference the same block."""
    from kcsi.discussion.concreteness import ANTI_META_BLOCK

    parts = build_cross_task_discussion_parts(
        agent_id="agent-1",
        generation=1,
        round_num=0,
        phase1_context={"task_id": "x", "reflection": "y"},
    )
    text = _full(parts)
    # The full anti-meta block must appear verbatim in the prompt
    # (cache-stable; same bytes as the distiller sees).
    assert ANTI_META_BLOCK in text


def test_schema_example_is_concrete_not_meta():
    """The prompt must show a concrete example, not a meta example."""
    parts = build_cross_task_discussion_parts(
        agent_id="agent-1",
        generation=1,
        round_num=0,
        phase1_context={"task_id": "x", "reflection": "y"},
    )
    text = _full(parts)
    # The schema description must explicitly reject meta wording.
    # If the example shows "separation of concerns" agents will copy it.
    assert "NOT 'separation of concerns'" in text or "NOT 'two-phase'" in text


def test_empty_history_text_is_honest_about_generation():
    """With empty cross-task history (the live forum path for ALL generations,
    #1258), gen 1 may say no prior posts exist, but gen 2+ must NOT — prior-gen
    posts exist and are merely not loaded."""
    gen1 = _full(
        build_cross_task_discussion_parts(
            agent_id="agent-1",
            generation=1,
            round_num=0,
            phase1_context=None,
            cross_task_history=[],
            peer_posts_this_gen=[],
        )
    )
    gen3 = _full(
        build_cross_task_discussion_parts(
            agent_id="agent-1",
            generation=3,
            round_num=0,
            phase1_context=None,
            cross_task_history=[],
            peer_posts_this_gen=[],
        )
    )
    # Gen 1: the "none yet" phrasing is truthful.
    assert "no cross-task posts have been written in any prior generation" in gen1
    # Gen 3: must NOT claim none were ever written (that is false); it should say
    # prior-gen posts are simply not shown here.
    assert "no cross-task posts have been written in any prior generation" not in gen3
    assert "prior-generation posts are not shown here" in gen3

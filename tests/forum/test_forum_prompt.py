"""Tests for forum prompt builders."""


def test_per_task_prompt_encourages_threaded_replies():
    from kcsi.forum import build_per_task_discussion_parts

    p = build_per_task_discussion_parts(
        agent_id="a1",
        generation=1,
        traces=[],
        task_ids=["t1"],
        task_descriptions={"t1": "desc"},
        prior_gen_posts=[{"id": 99, "agent_id": "a2", "text": "prior"}],
    ).as_text()
    # Must instruct threaded replies
    assert "reply_to" in p or "parent_post_id" in p
    # Must allow multiple insights
    assert "multiple" in p.lower() or "one or more" in p.lower()


def test_cross_task_prompt_builder_exists():
    """V2: builder takes phase1_context + cross_task_history (no per_task_posts)."""
    from kcsi.forum import build_cross_task_discussion_parts

    p = build_cross_task_discussion_parts(
        agent_id="a1",
        generation=2,
        round_num=0,
        cross_task_history=[
            {"id": 1, "generation": 1, "agent_id": "a2", "text": "x from gen 1"},
        ],
    ).as_text()
    assert "cross" in p.lower() or "across" in p.lower()


def test_cross_task_prompt_renders_history_chronologically():
    """V2: cross_task_history rendered in caller-provided order with no
    truncation. Truncating breaks prompt-cache prefix stability across
    generations.
    """
    from kcsi.forum import build_cross_task_discussion_parts

    history = [{"id": i, "generation": 1 + (i // 50), "agent_id": "a1", "text": f"post-{i}"} for i in range(100)]
    prompt = build_cross_task_discussion_parts(
        agent_id="a1",
        generation=3,
        round_num=0,
        cross_task_history=history,
    ).as_text()
    assert "post-0" in prompt and "post-99" in prompt
    assert prompt.index("post-0") < prompt.index("post-99"), "posts should appear in chronological (oldest-first) order"


# ---------------------------------------------------------------------------
# NOTE: Tests for ``GenerationalOrchestrator._condense_forum_assets`` were
# removed in Plan Task 15.  That method was the legacy R3 asset-bundle
# condenser inside the per-task forum phase; it has been replaced by the
# three-phase split: per-task forum (discussion only), cross-task forum
# (shared room), and distillation into per-task + cross-task bundles. See
# ``tests/test_distill_phase.py`` for the new coverage.
#
# The legacy structured R1/R2 forum-round builders and
# ``parse_forum_round_response`` were removed in the dead-code cleanup; their
# tests went with them. The live three-phase discussion path is covered by the
# ``build_*_discussion_parts`` tests above and in tests/test_forum_prompt_*.py.
# ---------------------------------------------------------------------------

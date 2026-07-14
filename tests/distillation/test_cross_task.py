import json

from kcsi.distillation.cross_task import (
    _select_cross_posts_for_budget,
    distill_cross_task,
)
from kcsi.distillation.types import CrossTaskBundle


def test_distill_cross_task_builds_bundle():
    posts = [{"id": 2, "agent_id": "a1", "task_id": "t1", "text": "pattern"}]

    def fake_llm(s, u):
        return json.dumps(
            {
                "transferable_insights": ["General pattern P applies to grids at least 3x3"],
                "confirmed_constraints": ["shape errors recur across tasks"],
                "rejected_hypotheses": ["blind tiling fails repeatedly"],
                "pitfalls": ["Pattern Q is misleading"],
                "checks": [],
                "next_steps": ["validate shape before color fill"],
                "evidence_post_ids": [2],
            }
        )

    b = distill_cross_task(
        cross_posts=posts,
        # V2: prior_bundle removed
        llm=fake_llm,
    )
    assert isinstance(b, CrossTaskBundle)
    assert b.transferable_insights == ["General pattern P applies to grids at least 3x3"]
    assert b.confirmed_constraints == ["shape errors recur across tasks"]
    assert b.rejected_hypotheses == ["blind tiling fails repeatedly"]
    assert b.next_steps == ["validate shape before color fill"]


def _capturing_llm(captured: dict):
    """A schema+cache_prefix-accepting fake LLM that records the kwargs it was
    called with and returns a minimal valid bundle."""

    def fake_llm(s, u, *, json_schema=None, cache_prefix=None):
        captured["system"] = s
        captured["user"] = u
        captured["cache_prefix"] = cache_prefix
        return (
            json.dumps({"transferable_insights": ["general pattern P applies to grids"]}),
            None,
        )

    return fake_llm


def test_target_conditioned_distill_sends_history_as_cache_prefix():
    """Target-conditioning must hand the shared forum history to the LLM as
    ``cache_prefix`` (cache-read across targets) while the per-target user
    message carries only the varying suffix — issue #1252 item 3."""
    posts = [{"id": 2, "agent_id": "a1", "task_id": "t1", "text": "grid pattern insight"}]
    captured: dict = {}
    distill_cross_task(
        cross_posts=posts,
        llm=_capturing_llm(captured),
        target_task={"id": "tX", "prompt": "solve the downstream task"},
    )
    assert captured["cache_prefix"] is not None
    # The shared history lives in the cached prefix, not the per-target user msg.
    assert "grid pattern insight" in captured["cache_prefix"]
    assert "grid pattern insight" not in captured["user"]
    # The varying target lives in the user suffix, not the cached prefix.
    assert "solve the downstream task" in captured["user"]
    assert "solve the downstream task" not in captured["cache_prefix"]


def test_non_conditioned_distill_sends_no_cache_prefix():
    """Without a target (single call per generation), the history is a plain
    user string — caching a one-shot prefix would only pay the write premium."""
    posts = [{"id": 2, "agent_id": "a1", "task_id": "t1", "text": "grid pattern insight"}]
    captured: dict = {}
    distill_cross_task(cross_posts=posts, llm=_capturing_llm(captured))
    assert captured["cache_prefix"] is None
    assert "grid pattern insight" in captured["user"]


def test_distill_cross_task_filters_evidence_ids_to_supplied_posts():
    posts = [{"id": 2, "agent_id": "a1", "task_id": "t1", "text": "pattern"}]

    def fake_llm(s, u):
        return json.dumps(
            {
                "transferable_insights": ["General pattern P applies"],
                "confirmed_constraints": [],
                "rejected_hypotheses": [],
                "pitfalls": [],
                "checks": [],
                "next_steps": [],
                "evidence_post_ids": [2, 999, "bad", True],
            }
        )

    b = distill_cross_task(
        cross_posts=posts,
        # V2: prior_bundle removed
        llm=fake_llm,
    )

    assert isinstance(b, CrossTaskBundle)
    assert b.evidence_post_ids == [2]


def test_distill_cross_task_none_on_failure():
    def bad(s, u):
        return "nope"

    assert (
        distill_cross_task(
            cross_posts=[],
            # V2: prior_bundle removed
            llm=bad,
        )
        is None
    )


def test_distill_cross_task_forwards_task_source_to_prompt():
    """When task_source is provided the rendered prompt should carry the
    benchmark-specific DOMAIN HINT. distill_cross_task must still return a
    valid CrossTaskBundle."""
    posts = [{"id": 3, "agent_id": "a1", "task_id": "t1", "text": "flood-fill 8-neighborhood on color 7 regions"}]
    captured: dict[str, str] = {}

    def capturing_llm(sys_prompt: str, user_prompt: str) -> str:
        captured["full"] = f"{sys_prompt}\n{user_prompt}"
        return json.dumps(
            {
                "transferable_insights": [
                    "BFS flood-fill on 8-neighborhood to isolate color-7 regions before applying any shape rule"
                ],
                "pitfalls": [],
                "checks": [],
                "evidence_post_ids": [3],
            }
        )

    bundle = distill_cross_task(
        cross_posts=posts,
        # V2: prior_bundle removed
        llm=capturing_llm,
        task_source="arc",
    )
    assert isinstance(bundle, CrossTaskBundle)
    assert bundle.transferable_insights and bundle.evidence_post_ids == [3]
    # Prompt received the ARC domain hint and the anti-meta block (now in
    # the system message after the cache-stability prefix move).
    assert "DOMAIN HINT (ARC-AGI)" in captured["full"]
    assert "STRICT ANTI-META RULES" in captured["full"]


def test_distill_cross_task_without_task_source_uses_generic_hint():
    captured: dict[str, str] = {}

    def capturing_llm(sys_prompt: str, user_prompt: str) -> str:
        captured["full"] = f"{sys_prompt}\n{user_prompt}"
        return json.dumps(
            {
                "transferable_insights": [],
                "pitfalls": [],
                "checks": [],
                "evidence_post_ids": [],
            }
        )

    bundle = distill_cross_task(
        cross_posts=[],
        # V2: prior_bundle removed
        llm=capturing_llm,
    )
    assert isinstance(bundle, CrossTaskBundle)
    assert "DOMAIN HINT:" in captured["full"]
    # No benchmark-specific block leaked in.
    assert "DOMAIN HINT (ARC-AGI)" not in captured["full"]


def test_distill_cross_task_caps_output_to_5_items():
    """When the LLM over-produces bullets, the parse-side cap should trim
    each list to 5 to match the prompt's hard cap. This prevents regressions
    where a model ignores the '<= 5' instruction and floods the bundle."""

    def over_producing_llm(sys_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "transferable_insights": [f"insight_{i}" for i in range(12)],
                "pitfalls": [f"pitfall_{i}" for i in range(8)],
                "checks": [f"check_{i}" for i in range(7)],
                "evidence_post_ids": [1, 2, 3],
            }
        )

    bundle = distill_cross_task(
        cross_posts=[{"id": 1, "agent_id": "a", "text": "x"}],
        # V2: prior_bundle removed
        llm=over_producing_llm,
    )
    assert isinstance(bundle, CrossTaskBundle)
    assert len(bundle.transferable_insights) == 5
    assert len(bundle.pitfalls) == 5
    assert len(bundle.checks) == 5


def test_cross_task_role_directive_mentions_json_posts():
    """Plan A: cross-task posts now arrive as JSON with concrete_primitive."""
    from kcsi.distillation.prompts import _CROSS_TASK_ROLE_DIRECTIVE

    assert "concrete_primitive" in _CROSS_TASK_ROLE_DIRECTIVE, (
        "Distiller must know posts now have a concrete_primitive field — "
        "without this hint it treats JSON posts as opaque prose and the "
        "structural slot is wasted."
    )
    # The directive must still tell the distiller to drop meta bullets.
    assert "verbatim" in _CROSS_TASK_ROLE_DIRECTIVE.lower()


def test_select_cross_posts_for_budget_preserves_generation_coverage():
    posts = []
    for generation in range(1, 5):
        posts.append(
            {
                "id": generation * 10 + 1,
                "agent_id": f"a{generation}",
                "task_id": f"t{generation}",
                "generation": generation,
                "round_num": 0,
                "text": f"gen {generation} round 0 " + ("x" * 1800),
            }
        )
        posts.append(
            {
                "id": generation * 10 + 2,
                "agent_id": f"a{generation}",
                "task_id": f"t{generation}",
                "generation": generation,
                "round_num": 1,
                "reply_to": generation * 10 + 1,
                "text": f"gen {generation} round 1 " + ("y" * 1800),
            }
        )

    selected = _select_cross_posts_for_budget(
        cross_posts=posts,
        task_source="polyglot",
        max_input_tokens=6_000,
    )

    selected_generations = {int(post["generation"]) for post in selected}
    assert selected_generations == {1, 2, 3, 4}
    assert any(int(post.get("round_num", 0)) >= 1 for post in selected)
    assert len(selected) < len(posts)


def test_shared_cross_posts_give_identical_prefix_across_target_sizes():
    """Over budget, per-target trimming counts the per-target target section so
    differently-sized targets pick DIFFERENT post subsets → different
    cache_prefix (the cache-defeating bug). The shared once-per-generation trim
    must yield a set that re-trims to a no-op for every target, so every target
    gets a byte-identical cache_prefix (issue #1252 item 3)."""
    from kcsi.distillation.cross_task import select_shared_cross_posts_for_targets
    from kcsi.distillation.prompts import build_cross_task_distill_prompt_parts

    # ~200K tokens of forum history — comfortably over the ~131.8K budget.
    posts = [
        {
            "id": idx + 1,
            "agent_id": f"a{idx}",
            "task_id": f"t{idx % 5}",
            "generation": 1 + (idx // 5),
            "round_num": 1 if idx % 2 == 0 else 0,
            "reply_to": idx if idx % 2 == 0 else None,
            "text": f"cross-task pattern {idx} " + ("z" * 2000),
        }
        for idx in range(300)
    ]
    small = {"id": "t_small", "prompt": "solve the task"}
    large = {"id": "t_large", "prompt": "solve the task " + ("q" * 40_000)}

    # The bug: independent per-target trimming diverges.
    sel_small = _select_cross_posts_for_budget(
        cross_posts=posts, task_source="polyglot", max_input_tokens=131_808, target_task=small
    )
    sel_large = _select_cross_posts_for_budget(
        cross_posts=posts, task_source="polyglot", max_input_tokens=131_808, target_task=large
    )
    assert sel_small != sel_large, "expected per-target trimming to diverge (the bug this guards)"

    # The fix: one shared trim → a no-op re-trim for BOTH targets → same posts.
    shared = select_shared_cross_posts_for_targets(
        cross_posts=posts, task_source="polyglot", target_tasks=[small, large]
    )
    assert shared, "shared trim should keep at least one post here"
    resel_small = _select_cross_posts_for_budget(
        cross_posts=shared, task_source="polyglot", max_input_tokens=131_808, target_task=small
    )
    resel_large = _select_cross_posts_for_budget(
        cross_posts=shared, task_source="polyglot", max_input_tokens=131_808, target_task=large
    )
    assert resel_small == shared == resel_large

    # …hence identical cache_prefix delivered to every target.
    _, prefix_small, _ = build_cross_task_distill_prompt_parts(
        cross_posts=shared, task_source="polyglot", target_task=small
    )
    _, prefix_large, _ = build_cross_task_distill_prompt_parts(
        cross_posts=shared, task_source="polyglot", target_task=large
    )
    assert prefix_small == prefix_large


def test_shared_cross_posts_probe_uses_largest_rendered_target():
    """The budget probe must pick the target with the largest RENDERED section
    (``_fmt_target_task_section``), not the longest raw prompt. The rendered
    section includes ``Task ID: {id}`` and applies ``\\r\\n`` collapse, so a
    target that is longest by raw ``len(prompt)`` can render a SMALLER section
    than one with a much longer id — and probing against the raw-longest would
    under-budget the true largest target, diverging its cache_prefix
    (issue #1252 item 3)."""
    from kcsi.distillation.cross_task import select_shared_cross_posts_for_targets
    from kcsi.distillation.prompts import _fmt_target_task_section

    posts = [
        {
            "id": idx + 1,
            "agent_id": f"a{idx}",
            "task_id": f"t{idx % 5}",
            "generation": 1 + (idx // 5),
            "round_num": 1 if idx % 2 == 0 else 0,
            "reply_to": idx if idx % 2 == 0 else None,
            "text": f"cross-task pattern {idx} " + ("z" * 2000),
        }
        for idx in range(300)
    ]
    # ``raw_longest`` wins on len(prompt) only because it is padded with CRLF
    # pairs; ``_sanitize_target_prompt`` collapses ``\r\n``→``\n`` so its RENDERED
    # section is far smaller. ``rendered_longest`` is plain text: shorter raw
    # length, but the larger rendered section (~10K chars ≈ several posts of
    # budget). A raw-``len`` probe picks the wrong (smaller-rendered) target and
    # under-budgets the true largest, diverging its prefix.
    raw_longest = {"id": "a", "prompt": "x\r\n" * 20_000}  # raw 60K, rendered ~40K
    rendered_longest = {"id": "b", "prompt": "y" * 50_000}  # raw 50K, rendered ~50K
    assert len(raw_longest["prompt"]) > len(rendered_longest["prompt"])
    assert len(_fmt_target_task_section(rendered_longest)) > len(_fmt_target_task_section(raw_longest))

    shared = select_shared_cross_posts_for_targets(
        cross_posts=posts, task_source="polyglot", target_tasks=[raw_longest, rendered_longest]
    )
    # The selector must budget against the rendered-largest target: match a
    # direct trim probed with ``rendered_longest`` (the fix) and DIFFER from one
    # probed with ``raw_longest`` (the raw-``len`` bug, which would keep more
    # posts than fit the true largest target and diverge its prefix).
    probe_rendered = _select_cross_posts_for_budget(
        cross_posts=posts, task_source="polyglot", max_input_tokens=131_808, target_task=rendered_longest
    )
    probe_raw = _select_cross_posts_for_budget(
        cross_posts=posts, task_source="polyglot", max_input_tokens=131_808, target_task=raw_longest
    )
    assert shared == probe_rendered
    assert probe_raw != probe_rendered, "constructed targets must trim differently to make this discriminating"


def test_distill_cross_task_retries_after_prompt_too_long():
    posts = []
    for idx in range(240):
        posts.append(
            {
                "id": idx + 1,
                "agent_id": f"a{idx}",
                "task_id": f"t{idx % 4}",
                "generation": 1 + (idx // 4),
                "round_num": 1 if idx % 3 == 0 else 0,
                "reply_to": idx if idx % 3 == 0 else None,
                "text": f"cross-task pattern {idx} " + ("z" * 1800),
            }
        )

    call_sizes: list[int] = []

    def flaky_llm(system_prompt: str, user_prompt: str) -> str:
        rendered_posts = user_prompt.count("- id=")
        call_sizes.append(rendered_posts)
        if len(call_sizes) == 1:
            raise RuntimeError("prompt is too long: 200513 tokens > 200000 maximum")
        return json.dumps(
            {
                "transferable_insights": ["Use the surviving cross-task pattern from post 1"],
                "confirmed_constraints": [],
                "rejected_hypotheses": [],
                "pitfalls": [],
                "checks": [],
                "next_steps": [],
                "evidence_post_ids": [posts[0]["id"]],
            }
        )

    bundle = distill_cross_task(
        cross_posts=posts,
        llm=flaky_llm,
        task_source="polyglot",
    )

    assert isinstance(bundle, CrossTaskBundle)
    assert call_sizes[0] > call_sizes[1]
    assert bundle.transferable_insights == ["Use the surviving cross-task pattern from post 1"]


def test_distill_cross_task_single_post_overflow_retries_target_only():
    # A single oversized post that overflows real context cannot be trimmed
    # further (the one-post budget fallback keeps it). The distiller should drop
    # it and retry with 0 posts (target-only) rather than yielding no bundle.
    posts = [
        {
            "id": 1,
            "agent_id": "a1",
            "task_id": "t1",
            "generation": 1,
            "round_num": 0,
            "reply_to": None,
            "text": "huge cross-task post " + ("z" * 4000),
        }
    ]
    call_sizes: list[int] = []

    # Accept the cache_prefix kwarg: under target-conditioning (#1252 item 3) the
    # posts render in cache_prefix, not user_prompt, so count across both to see
    # what the model actually receives.
    def flaky_llm(system_prompt: str, user_prompt: str, *, json_schema=None, cache_prefix=None) -> str:
        call_sizes.append(((cache_prefix or "") + user_prompt).count("- id="))
        if len(call_sizes) == 1:
            raise RuntimeError("prompt is too long: 200513 tokens > 200000 maximum")
        return json.dumps(
            {
                "transferable_insights": ["distilled from the target task alone"],
                "confirmed_constraints": [],
                "rejected_hypotheses": [],
                "pitfalls": [],
                "checks": [],
                "next_steps": [],
                "evidence_post_ids": [],
            }
        )

    bundle = distill_cross_task(
        cross_posts=posts,
        llm=flaky_llm,
        task_source="polyglot",
        target_task={"id": "tgt", "prompt": "optimize the sorting algorithm"},
    )

    # The retry with 0 posts is the behavior under test: first call renders the
    # single post (1) and overflows, second call renders none (0) and succeeds
    # with a valid bundle rather than the pre-fix None.
    assert isinstance(bundle, CrossTaskBundle)
    assert call_sizes == [1, 0], f"expected retry with 0 posts after single-post overflow, got {call_sizes}"


def test_distill_cross_task_skips_llm_when_target_alone_exceeds_budget():
    calls = {"n": 0}

    def llm(system_prompt: str, user_prompt: str) -> str:
        calls["n"] += 1
        return json.dumps({"transferable_insights": ["unexpected"], "evidence_post_ids": []})

    bundle = distill_cross_task(
        cross_posts=[{"id": 1, "agent_id": "a1", "task_id": "t1", "text": "small post"}],
        llm=llm,
        target_task={"id": "too-big", "prompt": "x" * 500_000},
    )

    assert bundle is None
    assert calls["n"] == 0


def test_distill_cross_task_forwards_target_task_to_prompt():
    posts = [{"id": 2, "agent_id": "a1", "task_id": "t1", "text": "pattern"}]
    seen = {}

    def fake_llm(s, u):
        seen["system"] = s
        seen["user"] = u
        return json.dumps(
            {
                "transferable_insights": ["P"],
                "confirmed_constraints": [],
                "rejected_hypotheses": [],
                "pitfalls": [],
                "checks": [],
                "next_steps": [],
                "evidence_post_ids": [2],
            }
        )

    b = distill_cross_task(
        cross_posts=posts,
        llm=fake_llm,
        target_task={"id": "tX", "prompt": "TARGET-PROMPT-MARKER"},
    )
    assert isinstance(b, CrossTaskBundle)
    assert "TARGET-PROMPT-MARKER" in seen["user"]
    assert "TARGET TASK" in seen["system"]

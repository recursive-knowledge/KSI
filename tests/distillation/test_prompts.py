from kcsi.discussion.concreteness import (
    assert_anti_meta_rules_present as _assert_anti_meta_rules_present,
)
from kcsi.distillation.prompts import (
    _ATTEMPT_OUTPUT_EXCERPT_CHARS,
    build_cross_task_distill_prompt,
    build_per_task_distill_prompt,
)

# --- Anti-meta / concreteness guard ---------------------------------------
#
# The cross-task distill bundle in live baseline sweeps was measured to be
# ~95% process meta-advice ("validate first", "decompose", "boundary"). The
# tests below pin the new prompt so regressions (e.g. a future refactor
# silently dropping the STRICT rules block) trip CI instead of silently
# re-polluting the bundle. The shared `_assert_anti_meta_rules_present`
# helper lives in src/kcsi/discussion/concreteness.py — single source of truth
# alongside the block content.


def _full(prompt_tuple: tuple[str, str]) -> str:
    """Return the full text the LLM sees: system + user concatenated.

    Most assertions in this file are about content visibility, not about
    which message it lives in. Stable content was moved from user → system
    in the prompt-cache prefix-stability change (see
    `_build_distill_system` in `src/kcsi/distillation/prompts.py`); using
    this helper keeps those visibility assertions robust to future moves
    in either direction. Tests that specifically pin "this content must
    be in system for cache eligibility" assert against the system half
    directly.
    """
    system, user = prompt_tuple
    return f"{system}\n{user}"


def test_per_task_prompt_bans_generic_process_advice():
    sys_p, _ = build_per_task_distill_prompt(task_id="t1", attempts=[], posts=[])
    assert "task-discriminating" in sys_p
    assert "Do NOT emit" in sys_p


def test_cross_task_prompt_bans_generic_process_advice():
    sys_p, _ = build_cross_task_distill_prompt(cross_posts=[])
    assert "task-discriminating" in sys_p


def test_rejected_hypotheses_require_parameterization():
    sys_p, _ = build_per_task_distill_prompt(task_id="t1", attempts=[], posts=[])
    assert "FALSIFIED:" in sys_p
    assert "UNTRIED" in sys_p


def test_per_task_prompt_includes_attempts_and_posts():
    attempts = [
        {"agent_id": "a1", "native_score": 0.0, "model_output": "tried X, failed"},
    ]
    posts = [
        {"id": 1, "agent_id": "a1", "text": "X fails because of Y"},
    ]
    full = _full(
        build_per_task_distill_prompt(
            task_id="t1",
            attempts=attempts,
            posts=posts,
            # V2: prior_bundle removed
        )
    )
    assert "t1" in full
    assert "tried X" in full
    assert "X fails because of Y" in full
    # Output format instructions present (moved to system in cache-stability fix)
    assert "transferable_insights" in full
    assert "confirmed_constraints" in full
    assert "rejected_hypotheses" in full
    assert "pitfalls" in full
    assert "checks" in full
    assert "next_steps" in full


def test_per_task_prompt_includes_structured_swebench_eval_details():
    attempts = [
        {
            "agent_id": "a1",
            "native_score": 0.25,
            "model_output": "patched parser",
            "eval_results": {
                "swebench_status": "ok",
                "resolved": False,
                "native_score": 0.25,
                "instance_report": {
                    "tests_status": {
                        "FAIL_TO_PASS": {
                            "success": ["test_fixed"],
                            "failure": ["test_still_bad"],
                            "skipped": ["test_skip"],
                            "unknown": ["test_missing"],
                        },
                        "PASS_TO_PASS": {
                            "success": ["test_existing"],
                            "failure": ["test_regressed"],
                        },
                    },
                },
            },
        },
    ]

    _, user = build_per_task_distill_prompt(
        task_id="t1",
        attempts=attempts,
        posts=[],
        # V2: prior_bundle removed
    )

    assert "eval=status=ok" in user
    assert "resolved=false" in user
    assert "native_score=0.25" in user
    # Test names must NOT appear — only anonymized counts (leak-fix D).
    assert "test_fixed" not in user
    assert "test_still_bad" not in user
    assert "test_skip" not in user
    assert "test_missing" not in user
    assert "test_existing" not in user
    assert "test_regressed" not in user
    # Counts should still be present.
    assert "FAIL_TO_PASS success=1 failure=1 skipped=1 unknown=1" in user
    assert "PASS_TO_PASS success=1 failure=1" in user


def test_per_task_prompt_includes_structured_arc_eval_details():
    attempts = [
        {
            "agent_id": "a1",
            "native_score": 0.5,
            "model_output": "submitted two grids",
            "eval_results": {
                "status": "evaluated",
                "resolved": False,
                "native_score": 0.5,
                "arc_correct_count": 1,
                "arc_total_count": 2,
                "arc_pass_ratio": 0.5,
                "arc_per_test": [
                    {"test_index": 0, "correct": True, "detail": "matched"},
                    {
                        "test_index": 1,
                        "correct": False,
                        "detail": {
                            "reason": "cell_mismatch",
                            "submitted_shape": [2, 2],
                            "expected_shape": [3, 3],
                            "first_mismatch": {
                                "row": 2,
                                "col": 3,
                                "expected": 7,
                                "submitted": 0,
                            },
                        },
                    },
                ],
            },
        },
    ]

    _, user = build_per_task_distill_prompt(
        task_id="arc-task",
        attempts=attempts,
        posts=[],
        # V2: prior_bundle removed
        task_source="arc",
    )

    assert "ARC 1/2 correct" in user
    assert "pass_ratio=0.5" in user
    assert "ARC per_test_summary: observed=2 correct=1 wrong=1 unknown=0" in user
    assert "first_mismatch" not in user
    assert "expected=7" not in user
    assert "expected_shape" not in user
    assert "cell_mismatch" not in user
    assert "matched" not in user


def test_per_task_prompt_includes_trace_and_attempt_meta_for_tb2():
    attempts = [
        {
            "agent_id": "a1",
            "native_score": 0.0,
            "model_output": "configured nginx",
            "trace_condensed": (
                "TB2 attempt summary: reward=0.0 agent_exit=0 verifier_exit=0; "
                "failure_signature=Expected benchmark-access.log to exist; "
                "verifier_clues=['Expected benchmark-access.log to exist']; "
                "tool_count=12"
            ),
            "reflection": (
                "I confirmed reward=0.0; "
                "verifier_stdout_tail=Expected benchmark-access.log to exist; "
                "proposed change: configure nginx access logging"
            ),
            "attempt_meta": {
                "reward": 0.0,
                "agent_exit_code": 0,
                "verifier_exit_code": 0,
                "tool_count": 12,
                "verified_outcome": "Verifier unresolved with reward 0.",
                "failure_signature": "Expected benchmark-access.log to exist",
                "last_state_change": "nginx -t && nginx -s reload",
                "recent_commands": [
                    "apt-get update && apt-get install -y nginx",
                    "nginx -t",
                ],
                "verifier_clues": [
                    "Expected benchmark-access.log to exist",
                    "Connection refused",
                ],
                "verifier_stdout_tail": "Expected benchmark-access.log to exist",
            },
        },
    ]

    system, user = build_per_task_distill_prompt(
        task_id="nginx-request-logging",
        attempts=attempts,
        posts=[],
        task_source="terminal_bench_2",
    )

    # The agent's OWN observations flow forward.
    assert "trace=TB2 attempt summary" in user
    assert "reward=0.0" in user
    assert "tool_count=12" in user
    assert "proposed change: configure nginx access logging" in user
    assert "verified_outcome=Verifier unresolved with reward 0." in user
    assert "last_state_change=nginx -t && nginx -s reload" in user
    assert "apt-get update && apt-get install -y nginx" in user
    assert "DOMAIN HINT (Terminal-Bench 2)" in system
    # Held-out verifier content (hidden pytest output) must NOT reach the distill
    # LLM: the TB2 verifier runs hidden tests the benchmark forbids reading, so
    # failure_signature / verifier_clues / verifier_*_tail echo held-out test
    # assertions — an information-parity leak into the distilled seed.
    assert "failure_signature=" not in user
    assert "verifier_clues=" not in user
    assert "verifier_stdout_tail=" not in user
    assert "Expected benchmark-access.log to exist" not in user


def test_per_task_prompt_excerpt_survives_early_isolated_boundary():
    # Regression for issue #1057 review: a realistic error/trace payload with
    # one early boundary (a short "Error:" label) followed by a long
    # unbroken tail must not collapse the 2000-char excerpt budget down to a
    # handful of characters via truncate_at_boundary's boundary fallback.
    long_output = "Error: " + "x" * 5000
    attempts = [
        {"agent_id": "a1", "native_score": 0.0, "model_output": long_output},
    ]

    _, user = build_per_task_distill_prompt(
        task_id="t1",
        attempts=attempts,
        posts=[],
    )

    marker = "  output="
    start = user.index(marker) + len(marker)
    end = user.index("\n", start)
    excerpt = user[start:end]
    # Sanitized excerpt has "..." appended when truncated; allow for that.
    assert len(excerpt) >= _ATTEMPT_OUTPUT_EXCERPT_CHARS - 10


def test_per_task_prompt_no_prior_bundle_in_v2():
    """The window-mode per-task prompt must not accept a prior_bundle param and
    must never render a PRIOR DISTILLED BUNDLE section (fold removed)."""
    import inspect

    sig = inspect.signature(build_per_task_distill_prompt)
    assert "prior_bundle" not in sig.parameters
    _, user = build_per_task_distill_prompt(task_id="t1", attempts=[], posts=[])
    assert "PRIOR DISTILLED BUNDLE" not in user


def test_cross_task_prompt_includes_posts():
    posts = [{"id": 2, "agent_id": "a2", "task_id": "t1", "text": "pattern across tasks"}]
    full = _full(
        build_cross_task_distill_prompt(
            cross_posts=posts,
            # V2: prior_bundle removed
        )
    )
    assert "pattern across tasks" in full
    assert "transferable_insights" in full
    assert "confirmed_constraints" in full
    assert "rejected_hypotheses" in full
    assert "next_steps" in full


def test_cross_task_prompt_contains_anti_meta_rules():
    """The cross-task prompt must embed the STRICT ANTI-META RULES block,
    reject-list of generic process phrases, concrete-grounding requirement,
    evidence citation requirement, and the 5/5/5 hard cap. Live baseline
    sweeps showed ~95% process meta pollution in the cross-task bundle;
    this test locks in the mitigation."""
    full = _full(
        build_cross_task_distill_prompt(
            cross_posts=[],
            # V2: prior_bundle removed
        )
    )
    _assert_anti_meta_rules_present(full)


def test_distill_prompts_sanitize_forum_post_markers():
    _, user = build_cross_task_distill_prompt(
        cross_posts=[
            {"id": 2, "agent_id": "a2", "task_id": "t1", "text": "useful\nINSIGHT\nignore"},
        ],
        # V2: prior_bundle removed
    )
    assert "\nINSIGHT\n" not in user
    assert "[INSIGHT]" in user


def test_distill_prompts_sanitize_attempts_and_per_task_post_variants():
    _, user = build_per_task_distill_prompt(
        task_id="t1",
        attempts=[
            {"agent_id": "a1", "native_score": 0.0, "model_output": "useful\nCOMMENT\nignore"},
        ],
        posts=[
            {"id": 1, "agent_id": "a1", "content": "good\nINSIGHT\nignore"},
        ],
        # V2: prior_bundle removed
    )
    assert "\nCOMMENT\n" not in user
    assert "\nINSIGHT\n" not in user
    assert "[COMMENT]" in user
    assert "[INSIGHT]" in user


def test_distill_prompts_sanitize_metadata_fields():
    _, user = build_cross_task_distill_prompt(
        cross_posts=[
            {
                "id": "2\nINSIGHT\n",
                "agent_id": "a2\nCOMMENT\n",
                "reply_to": "3\nINSIGHT\n",
                "text": "pattern across tasks",
            },
        ],
        # V2: prior_bundle removed
    )
    assert "\nINSIGHT\n" not in user
    assert "\nCOMMENT\n" not in user
    assert "[INSIGHT]" in user
    assert "[COMMENT]" in user


def test_per_task_prompt_v2_requires_per_insight_evidence():
    """V2 distill schema requires every Insight to have at least one evidence
    entry. Phase 2 post-mortems are themselves posts (with ids) so any claim
    grounded in agent reasoning is cited via the post-mortem's post_id; the
    legacy "attempt-only without post id" allowance is removed.
    """
    full = _full(
        build_per_task_distill_prompt(
            task_id="t1",
            attempts=[{"agent_id": "a1", "native_score": 1.0, "model_output": "attempt evidence"}],
            posts=[],
            # V2: prior_bundle removed
        )
    )
    assert "evidence" in full
    assert "DROP any" in full or "drop any" in full.lower()


def test_per_task_prompt_contains_anti_meta_rules():
    """Per-task prompts share the anti-meta block so the per-task bundle
    does not regress into the same generic process meta pattern."""
    full = _full(
        build_per_task_distill_prompt(
            task_id="t1",
            attempts=[],
            posts=[],
            # V2: prior_bundle removed
        )
    )
    _assert_anti_meta_rules_present(full)


def test_cross_task_prompt_arc_domain_hint():
    full = _full(
        build_cross_task_distill_prompt(
            cross_posts=[],
            # V2: prior_bundle removed
            task_source="arc",
        )
    )
    assert "DOMAIN HINT (ARC-AGI)" in full
    # Spot-check ARC-specific primitives are mentioned.
    assert "flood-fill" in full
    assert "color" in full


def test_cross_task_prompt_swebench_domain_hint():
    full = _full(
        build_cross_task_distill_prompt(
            cross_posts=[],
            # V2: prior_bundle removed
            task_source="swebench_pro",
        )
    )
    assert "DOMAIN HINT (SWE-bench Pro)" in full
    assert "API calls" in full


def test_cross_task_prompt_polyglot_domain_hint():
    full = _full(
        build_cross_task_distill_prompt(
            cross_posts=[],
            # V2: prior_bundle removed
            task_source="polyglot",
        )
    )
    assert "DOMAIN HINT (polyglot / Exercism)" in full
    assert "pytest" in full


def test_cross_task_prompt_tb2_domain_hint():
    system, user = build_cross_task_distill_prompt(
        cross_posts=[],
        task_source="terminal_bench_2",
    )
    assert "DOMAIN HINT (Terminal-Bench 2)" in system
    assert "/etc" in system
    assert "outcome/reward signal" in system
    assert "exact shell command or path" in system
    assert "make that behavior persist for a fresh shell and the verifier" in system
    assert "DOMAIN HINT (Terminal-Bench 2)" not in user


def test_per_task_prompt_tb2_requires_command_and_path_grounding():
    system, user = build_per_task_distill_prompt(
        task_id="git-multibranch",
        attempts=[],
        posts=[],
        task_source="terminal_bench_2",
    )
    assert "For Terminal-Bench 2" in system
    assert "exact shell command" in system
    assert "file path" in system
    assert "verifier-aligned checks" in system
    assert "For Terminal-Bench 2" not in user


def test_cross_task_prompt_unknown_task_source_falls_back_to_generic():
    full = _full(
        build_cross_task_distill_prompt(
            cross_posts=[],
            # V2: prior_bundle removed
            task_source="mystery-bench",
        )
    )
    # Generic hint used, no benchmark-specific block.
    assert "DOMAIN HINT:" in full
    assert "DOMAIN HINT (ARC-AGI)" not in full
    assert "DOMAIN HINT (SWE-bench Pro)" not in full
    assert "DOMAIN HINT (polyglot" not in full


def test_cross_task_prompt_aliases_arc1_arc2():
    """arc1/arc2/arc_agi_1/arc_agi_2 should all route to the ARC hint."""
    for alias in ("arc1", "arc2", "arc_agi_1", "arc_agi_2"):
        full = _full(
            build_cross_task_distill_prompt(
                cross_posts=[],
                # V2: prior_bundle removed
                task_source=alias,
            )
        )
        assert "DOMAIN HINT (ARC-AGI)" in full, f"alias {alias!r} should route to the ARC domain hint"


# --- Prompt-cache prefix stability ----------------------------------------
#
# The system prompt is the cache-stable prefix. The tests below pin two
# properties that together make the system prompt eligible for prompt
# caching on Anthropic Sonnet/Opus and OpenAI (both have a 1024-token
# minimum prefix to enable caching):
#
#   1. The system text is byte-identical across calls of the same builder
#      with the same task_source — varying user-side data must not leak
#      into the system message, or the prefix changes per call and the
#      cache never fires.
#   2. The system text is at least 1024 tokens for every supported
#      task_source — short prefixes are silently rejected by both
#      providers' caches, so anything below the floor renders the
#      `cache_control` / `prompt_cache_key` plumbing inert.
#
# (Anthropic Haiku 4.5 has a higher floor of 2048 tokens; this test
# pins the lower 1024 floor that unlocks Sonnet/Opus and OpenAI. Closing
# the Haiku gap is a separate, larger prompt-engineering decision.)


def _approx_token_count(text: str) -> int:
    """Conservative chars/3.5 estimate. tiktoken would be more precise but
    isn't a hard dependency of the test suite."""
    return len(text) // 3 + (1 if len(text) % 3 else 0)


def test_per_task_system_is_stable_across_varying_user_data():
    """Same task_source + different per-call data → identical system text.
    This is the precondition for prompt-cache hits across calls."""
    sys_a, _ = build_per_task_distill_prompt(
        task_id="task-A",
        attempts=[{"agent_id": "a1", "native_score": 0.0, "model_output": "first"}],
        posts=[{"id": 1, "agent_id": "a1", "text": "post A"}],
        # V2: prior_bundle removed
        task_source="arc",
    )
    sys_b, _ = build_per_task_distill_prompt(
        task_id="task-B",
        attempts=[{"agent_id": "a2", "native_score": 1.0, "model_output": "second"}],
        posts=[{"id": 2, "agent_id": "a2", "text": "post B"}],
        # V2: prior_bundle removed
        task_source="arc",
    )
    assert sys_a == sys_b, (
        "System prompt must be identical across distill calls with the "
        "same task_source, or the prompt cache cannot reuse the prefix."
    )


def test_cross_task_system_is_stable_across_varying_user_data():
    sys_a, _ = build_cross_task_distill_prompt(
        cross_posts=[{"id": 1, "agent_id": "a1", "text": "post one"}],
        # V2: prior_bundle removed
        task_source="swebench_pro",
    )
    sys_b, _ = build_cross_task_distill_prompt(
        cross_posts=[{"id": 2, "agent_id": "a2", "text": "post two — much longer"}],
        # V2: prior_bundle removed
        task_source="swebench_pro",
    )
    assert sys_a == sys_b


def test_per_task_system_above_cache_threshold_for_every_task_source():
    """System prompt must clear OpenAI/Sonnet/Opus 1024-token cache floor
    for every supported task_source. Below the floor, both providers'
    caches silently no-op and the routing-pin work in PR #547 is inert
    on this code path. Closing the Haiku 4.5 floor (2048 tokens) is
    deferred."""
    for source in ("arc", "swebench_pro", "polyglot", None):
        system, _ = build_per_task_distill_prompt(
            task_id="t",
            attempts=[],
            posts=[],
            # V2: prior_bundle removed
            task_source=source,
        )
        approx = _approx_token_count(system)
        assert approx >= 1024, (
            f"per-task system prompt for task_source={source!r} is "
            f"~{approx} tokens; needs >= 1024 for prompt-cache eligibility "
            "on OpenAI / Anthropic Sonnet / Anthropic Opus"
        )


def test_cross_task_system_above_cache_threshold_for_every_task_source():
    for source in ("arc", "swebench_pro", "polyglot", None):
        system, _ = build_cross_task_distill_prompt(
            cross_posts=[],
            # V2: prior_bundle removed
            task_source=source,
        )
        approx = _approx_token_count(system)
        assert approx >= 1024, (
            f"cross-task system prompt for task_source={source!r} is "
            f"~{approx} tokens; needs >= 1024 for prompt-cache eligibility"
        )


def test_distill_system_includes_moved_blocks():
    """Anti-meta rules + output schema + domain hint must live in the
    system message (not the user message). This is the property that
    makes the cache prefix stable across calls — varying user data after
    a stable system prefix lets the prefix cache."""
    system, user = build_per_task_distill_prompt(
        task_id="t",
        attempts=[],
        posts=[],
        # V2: prior_bundle removed
        task_source="arc",
    )
    assert "STRICT ANTI-META RULES" in system
    assert "DOMAIN HINT (ARC-AGI)" in system
    assert "Output schema (strict JSON)" in system
    # User must NOT carry duplicates of the same blocks (would defeat the
    # point of moving them, and would inflate per-call input tokens).
    assert "STRICT ANTI-META RULES" not in user
    assert "DOMAIN HINT (ARC-AGI)" not in user
    assert "Output schema (strict JSON)" not in user


def test_cross_task_prompt_target_none_is_unchanged():
    posts = [{"id": 1, "agent_id": "a1", "task_id": "t1", "text": "pattern"}]
    base_sys, base_user = build_cross_task_distill_prompt(cross_posts=posts)
    tgt_sys, tgt_user = build_cross_task_distill_prompt(cross_posts=posts, target_task=None)
    assert (tgt_sys, tgt_user) == (base_sys, base_user)


def test_cross_task_prompt_includes_target_after_posts():
    posts = [{"id": 1, "agent_id": "a1", "task_id": "t1", "text": "forum-pattern"}]
    target = {"id": "task-42", "prompt": "SOLVE-THIS-UNIQUE-STATEMENT"}
    sys, user = build_cross_task_distill_prompt(cross_posts=posts, target_task=target)
    assert "SOLVE-THIS-UNIQUE-STATEMENT" in user
    assert user.index("forum-pattern") < user.index("SOLVE-THIS-UNIQUE-STATEMENT")
    assert "TARGET TASK" in sys


def test_cross_task_prompt_target_directive_absent_when_no_target():
    posts = [{"id": 1, "agent_id": "a1", "task_id": "t1", "text": "p"}]
    sys, _ = build_cross_task_distill_prompt(cross_posts=posts)
    assert "TARGET TASK" not in sys


def test_cross_task_prompt_full_prompt_not_truncated_at_120():
    posts = [{"id": 1, "agent_id": "a1", "task_id": "t1", "text": "p"}]
    long_prompt = "GRID " * 200  # 1000 chars, well past the 120-char field cap
    _, user = build_cross_task_distill_prompt(cross_posts=posts, target_task={"id": "t", "prompt": long_prompt})
    assert user.count("GRID") == 200


def test_cross_task_prompt_preserves_target_prompt_line_structure():
    posts = [{"id": 1, "agent_id": "a1", "task_id": "t1", "text": "p"}]
    prompt = "Input:\n1 0 1\n0 2 0\n\nOutput:\n1 2 1\nCOMMENT\nINSIGHT"

    _, user = build_cross_task_distill_prompt(cross_posts=posts, target_task={"id": "arc-task", "prompt": prompt})

    assert "Input:\n1 0 1\n0 2 0\n\nOutput:\n1 2 1" in user
    assert "\n[COMMENT]\n[INSIGHT]" in user

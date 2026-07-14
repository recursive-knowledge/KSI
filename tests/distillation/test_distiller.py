import json
import tempfile
from pathlib import Path

from kcsi.distillation import (
    CrossTaskBundle,
    DistillInput,
    PerTaskBundle,
    distill,
)
from kcsi.memory.knowledge_store import CROSS_TASK_SENTINEL, KnowledgeStore


def _seed_db(tmp: Path) -> KnowledgeStore:
    ks = KnowledgeStore(str(tmp / "k.sqlite"), default_experiment="exp")
    ks.record_attempt(
        task_id="t1",
        agent_id="a1",
        generation=0,
        model_output="attempted t1",
        native_score=0.0,
    )
    ks.record_post(
        task_id="t1",
        agent_id="a1",
        generation=0,
        text="insight on t1",
    )
    ks.record_post(
        task_id=CROSS_TASK_SENTINEL,
        agent_id="a1",
        generation=0,
        text="cross-task pattern",
    )
    return ks


def test_distill_orchestrates_per_task_and_cross_task():
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_db(Path(tmp))

        def fake_llm(sys_prompt, user_prompt):
            return json.dumps(
                {
                    "transferable_insights": ["i"],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                }
            )

        out = distill(
            DistillInput(
                generation=0,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=fake_llm,
            )
        )
        assert set(out.per_task.keys()) == {"t1"}
        assert isinstance(out.per_task["t1"], PerTaskBundle)
        assert isinstance(out.cross_task, CrossTaskBundle)


def test_distill_skips_task_with_no_attempts():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(
            str(Path(tmp) / "k.sqlite"),
            default_experiment="exp",
        )

        def fake_llm(s, u):
            return json.dumps(
                {
                    "transferable_insights": [],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                }
            )

        out = distill(
            DistillInput(
                generation=0,
                task_ids=["missing_task"],
                knowledge_store=ks,
                llm=fake_llm,
            )
        )
        # Task with no attempts is skipped entirely (not just None)
        assert "missing_task" not in out.per_task
        # A benign "no attempts" skip is NOT a failure (#740 clean-run invariant).
        assert out.failures == 0


def test_distill_handles_partial_llm_failure():
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_db(Path(tmp))
        calls = {"n": 0}

        def flaky_llm(s, u):
            calls["n"] += 1
            if calls["n"] == 1:
                return json.dumps(
                    {
                        "transferable_insights": ["ok"],
                        "pitfalls": [],
                        "checks": [],
                        "evidence_post_ids": [],
                    }
                )
            return "bad json"

        out = distill(
            DistillInput(
                generation=0,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=flaky_llm,
            )
        )
        # Either per-task or cross-task may fail; the other should succeed
        assert out.per_task.get("t1") is not None or out.cross_task is not None


def test_distill_cross_task_exception_preserves_per_task_bundles(monkeypatch):
    """A non-LLM exception escaping distill_cross_task (e.g. a context-length
    error) must NOT discard the already-completed per-task bundles — the
    docstring promises 'returns whatever bundles succeeded'."""
    import kcsi.distillation.distiller as distiller_mod

    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_db(Path(tmp))  # t1 has an attempt; cross-task has a post

        def good_llm(s, u):
            return json.dumps(
                {
                    "transferable_insights": ["i"],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                }
            )

        def boom_cross(*_a, **_k):
            raise RuntimeError("simulated context-length overflow")

        monkeypatch.setattr(distiller_mod, "distill_cross_task", boom_cross)

        out = distill(
            DistillInput(
                generation=0,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=good_llm,
            )
        )
        # Per-task bundle survives despite the cross-task call raising.
        assert out.per_task.get("t1") is not None
        assert out.cross_task is None
        # The degradation is still visible (cross_post_count > 0 -> failures++).
        assert out.failures >= 1


def test_distill_cross_task_auth_failure_propagates(monkeypatch):
    """AuthenticationFailure from the cross-task call is fatal and must NOT be
    swallowed by the per-task-bundle-preserving guard."""
    import kcsi.distillation.distiller as distiller_mod
    from kcsi.errors import AuthenticationFailure

    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_db(Path(tmp))

        def good_llm(s, u):
            return json.dumps({"transferable_insights": [], "pitfalls": [], "checks": [], "evidence_post_ids": []})

        def auth_boom(*_a, **_k):
            raise AuthenticationFailure("bad key")

        monkeypatch.setattr(distiller_mod, "distill_cross_task", auth_boom)

        import pytest

        with pytest.raises(AuthenticationFailure):
            distill(
                DistillInput(
                    generation=0,
                    task_ids=["t1"],
                    knowledge_store=ks,
                    llm=good_llm,
                )
            )


def test_distill_forwards_reply_to_to_prompt():
    """_load_per_task_posts should surface threading (via reply_to) so the
    threaded structure is visible in the distill prompt. This exercises the
    production MCP forum_post path which writes to the reply_to column."""
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(
            str(Path(tmp) / "k.sqlite"),
            default_experiment="exp",
        )
        ks.record_attempt(
            task_id="t1",
            agent_id="a1",
            generation=0,
            model_output="attempted t1",
            native_score=0.0,
        )
        parent_id = ks.record_post(
            task_id="t1",
            agent_id="a1",
            generation=0,
            text="parent post",
        )
        # Use the NEW path (reply_to kwarg) -- the production MCP flow
        # populates the reply_to column, not parent_id.
        ks.record_post(
            task_id="t1",
            agent_id="a2",
            generation=0,
            text="child reply",
            reply_to=parent_id,
        )

        captured_prompts: list[str] = []

        def capturing_llm(sys_prompt: str, user_prompt: str) -> str:
            captured_prompts.append(user_prompt)
            return json.dumps(
                {
                    "transferable_insights": [],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                }
            )

        distill(
            DistillInput(
                generation=0,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=capturing_llm,
            )
        )

        # The per-task prompt should mention reply_to=<parent_id> somewhere
        per_task_prompt = next(
            (p for p in captured_prompts if "t1" in p and "parent post" in p),
            None,
        )
        assert per_task_prompt is not None
        assert f"reply_to={parent_id}" in per_task_prompt


def test_distill_forwards_post_author_native_score_to_prompt():
    """_load_per_task_posts must surface each post author's native_score so
    the distill prompt renders ``author_score=<score>`` — the signal the
    per-task distiller uses to weight high-score authors over low-score
    authors when their claims conflict. Without native_score threaded through
    record_post -> query_task, the field is None and the weighting never
    fires."""
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(
            str(Path(tmp) / "k.sqlite"),
            default_experiment="exp",
        )
        ks.record_attempt(
            task_id="t1",
            agent_id="solver",
            generation=0,
            model_output="attempted t1",
            native_score=1.0,
        )
        ks.record_post(
            task_id="t1",
            agent_id="solver",
            generation=0,
            text="the winning approach",
            native_score=1.0,
        )

        captured_prompts: list[str] = []

        def capturing_llm(sys_prompt: str, user_prompt: str) -> str:
            captured_prompts.append(user_prompt)
            return json.dumps(
                {
                    "transferable_insights": [],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                }
            )

        distill(
            DistillInput(
                generation=0,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=capturing_llm,
            )
        )

        per_task_prompt = next(
            (p for p in captured_prompts if "t1" in p and "winning approach" in p),
            None,
        )
        assert per_task_prompt is not None
        assert "author_score=1.0" in per_task_prompt


def test_distill_forwards_reply_to_on_cross_task_posts():
    """_load_cross_task_posts must forward reply_to so the cross-task distill
    prompt preserves threading, matching the per-task loader's behavior.
    """
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(
            str(Path(tmp) / "k.sqlite"),
            default_experiment="exp",
        )
        # Seed one per-task attempt so the per-task phase runs successfully.
        ks.record_attempt(
            task_id="t1",
            agent_id="a1",
            generation=0,
            model_output="attempted t1",
            native_score=0.0,
        )
        # Cross-task forum thread: parent post + threaded reply via reply_to.
        parent_id = ks.record_post(
            task_id=CROSS_TASK_SENTINEL,
            agent_id="a1",
            generation=0,
            text="parent cross-task post",
            source_phase="cross_task_forum",
        )
        ks.record_post(
            task_id=CROSS_TASK_SENTINEL,
            agent_id="a2",
            generation=0,
            text="child cross-task reply",
            source_phase="cross_task_forum",
            reply_to=parent_id,
        )

        captured_prompts: list[str] = []

        def capturing_llm(sys_prompt: str, user_prompt: str) -> str:
            captured_prompts.append(user_prompt)
            return json.dumps(
                {
                    "transferable_insights": [],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                }
            )

        distill(
            DistillInput(
                generation=0,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=capturing_llm,
            )
        )

        # The cross-task prompt should include reply_to=<parent_id>.
        cross_prompt = next(
            (p for p in captured_prompts if "parent cross-task post" in p and "child cross-task reply" in p),
            None,
        )
        assert cross_prompt is not None, f"cross-task prompt not captured; prompts: {captured_prompts}"
        assert f"reply_to={parent_id}" in cross_prompt


def test_distill_forwards_task_source_hint_end_to_end():
    """DistillInput.task_source should propagate through both per-task and
    cross-task prompts so the distiller emits benchmark-aware bullets."""
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_db(Path(tmp))

        captured_prompts: list[str] = []

        def capturing_llm(sys_prompt: str, user_prompt: str) -> str:
            # Capture system + user — the ARC domain hint moved into the
            # system message in the prompt-cache prefix-stability change.
            captured_prompts.append(f"{sys_prompt}\n{user_prompt}")
            return json.dumps(
                {
                    "transferable_insights": [],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                }
            )

        distill(
            DistillInput(
                generation=0,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=capturing_llm,
                task_source="arc",
            )
        )

        # Both phases should have received the ARC domain hint.
        arc_hint_prompts = [p for p in captured_prompts if "DOMAIN HINT (ARC-AGI)" in p]
        assert len(arc_hint_prompts) >= 2, (
            f"Expected ARC hint in both per-task and cross-task prompts, "
            f"got {len(arc_hint_prompts)}; prompts={captured_prompts}"
        )


def test_distill_per_phase_llm_overrides():
    """When ``DistillInput.llm_per_task`` or ``llm_cross_task`` is set, the
    distiller must route the corresponding phase through that callable
    rather than ``llm``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_db(Path(tmp))

        default_calls = {"n": 0}
        per_task_calls = {"n": 0}
        cross_task_calls = {"n": 0}

        def default_llm(s, u):
            default_calls["n"] += 1
            return json.dumps(
                {
                    "transferable_insights": ["default"],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                }
            )

        def per_task_llm(s, u):
            per_task_calls["n"] += 1
            return json.dumps(
                {
                    "transferable_insights": ["per-task"],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                }
            )

        def cross_task_llm(s, u):
            cross_task_calls["n"] += 1
            return json.dumps(
                {
                    "transferable_insights": ["cross-task"],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                }
            )

        distill(
            DistillInput(
                generation=0,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=default_llm,
                llm_per_task=per_task_llm,
                llm_cross_task=cross_task_llm,
            )
        )

        # The default llm should not have been used at all.
        assert default_calls["n"] == 0
        assert per_task_calls["n"] >= 1
        assert cross_task_calls["n"] >= 1


# ---------------------------------------------------------------------------
# Cross-task distill sliding-window (default last 6 generations).
#
# Without windowing, a 50-task c=50 ARC2 run hit Anthropic 200K context at
# gen 10 (210,659 tokens). The default 6-gen window keeps the input bounded
# while letting per-task distill (which uses full per-task history) carry
# older themes forward. Tunable via KCSI_CROSS_TASK_DISTILL_GEN_WINDOW.
# ---------------------------------------------------------------------------


def _seed_cross_task_history(tmp: Path, gens: list[int]) -> KnowledgeStore:
    ks = KnowledgeStore(str(tmp / "k.sqlite"), default_experiment="exp")
    ks.record_attempt(
        task_id="t1",
        agent_id="a1",
        generation=0,
        model_output="attempted t1",
        native_score=0.0,
    )
    for g in gens:
        ks.record_post(
            task_id=CROSS_TASK_SENTINEL,
            agent_id=f"a{g}",
            generation=g,
            text=f"cross-task post from gen {g}",
            source_phase="cross_task_forum",
        )
    return ks


def test_cross_task_distill_default_window_drops_old_gens(monkeypatch):
    """Default 6-gen window: at gen 10, keep gens 5-10, drop gens 0-4."""
    monkeypatch.delenv("KCSI_CROSS_TASK_DISTILL_GEN_WINDOW", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_cross_task_history(Path(tmp), gens=list(range(0, 11)))

        captured: list[str] = []

        def capturing_llm(sys_prompt, user_prompt):
            captured.append(user_prompt)
            return json.dumps(
                {
                    "transferable_insights": [],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                }
            )

        distill(
            DistillInput(
                generation=10,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=capturing_llm,
            )
        )
        cross_prompt = next(
            (p for p in captured if "cross-task post from gen" in p),
            None,
        )
        assert cross_prompt is not None
        # Match the full unique post text so "gen 1" doesn't substring-match "gen 10".
        for g in range(0, 5):
            assert f"cross-task post from gen {g}\n" not in cross_prompt, (
                f"gen {g} should be outside the 6-gen window at gen 10"
            )
        for g in range(5, 11):
            assert f"cross-task post from gen {g}\n" in cross_prompt, (
                f"gen {g} should be inside the 6-gen window at gen 10"
            )


def test_cross_task_distill_window_zero_disables_capping(monkeypatch):
    """Setting window=0 reverts to legacy uncapped behaviour."""
    monkeypatch.setenv("KCSI_CROSS_TASK_DISTILL_GEN_WINDOW", "0")
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_cross_task_history(Path(tmp), gens=list(range(0, 11)))

        captured: list[str] = []

        def capturing_llm(sys_prompt, user_prompt):
            captured.append(user_prompt)
            return json.dumps(
                {
                    "transferable_insights": [],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                }
            )

        distill(
            DistillInput(
                generation=10,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=capturing_llm,
            )
        )
        cross_prompt = next(
            (p for p in captured if "cross-task post from gen" in p),
            None,
        )
        assert cross_prompt is not None
        # All 11 gens should be present when window is disabled.
        for g in range(0, 11):
            assert f"cross-task post from gen {g}\n" in cross_prompt, f"gen {g} dropped despite window=0"


def test_cross_task_distill_custom_window(monkeypatch):
    """Window=3 at gen 10 keeps gens 8, 9, 10 only."""
    monkeypatch.setenv("KCSI_CROSS_TASK_DISTILL_GEN_WINDOW", "3")
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_cross_task_history(Path(tmp), gens=list(range(0, 11)))

        captured: list[str] = []

        def capturing_llm(sys_prompt, user_prompt):
            captured.append(user_prompt)
            return json.dumps(
                {
                    "transferable_insights": [],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                }
            )

        distill(
            DistillInput(
                generation=10,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=capturing_llm,
            )
        )
        cross_prompt = next(
            (p for p in captured if "cross-task post from gen" in p),
            None,
        )
        assert cross_prompt is not None
        for g in range(0, 8):
            assert f"cross-task post from gen {g}\n" not in cross_prompt
        for g in range(8, 11):
            assert f"cross-task post from gen {g}\n" in cross_prompt


# ---------------------------------------------------------------------------
# Per-task distill sliding-window (default last 6 generations) (#1014).
#
# Same failure mode as the cross-task window above: without a cap, a task
# stuck unsolved across many generations accumulates one attempt (with
# reflection) per generation with no bound, risking the same 200K-context
# overflow. Tunable via KCSI_PER_TASK_DISTILL_GEN_WINDOW.
# ---------------------------------------------------------------------------


def _seed_per_task_attempt_history(tmp: Path, gens: list[int]) -> KnowledgeStore:
    ks = KnowledgeStore(str(tmp / "k.sqlite"), default_experiment="exp")
    for g in gens:
        ks.record_attempt(
            task_id="t1",
            agent_id=f"a{g}",
            generation=g,
            model_output=f"attempt output gen {g}",
            native_score=0.0,
            reflection=f"reflection from gen {g}",
        )
    return ks


def test_per_task_distill_default_window_drops_old_gens(monkeypatch):
    """Default 6-gen window: at gen 10, keep gens 5-10, drop gens 0-4."""
    monkeypatch.delenv("KCSI_PER_TASK_DISTILL_GEN_WINDOW", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_per_task_attempt_history(Path(tmp), gens=list(range(0, 11)))

        captured: list[str] = []

        def capturing_llm(sys_prompt, user_prompt):
            captured.append(user_prompt)
            return json.dumps(
                {
                    "transferable_insights": [],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                }
            )

        distill(
            DistillInput(
                generation=10,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=capturing_llm,
            )
        )
        per_task_prompt = next(
            (p for p in captured if "reflection from gen" in p),
            None,
        )
        assert per_task_prompt is not None
        # Match the full unique reflection text so "gen 1" doesn't
        # substring-match "gen 10".
        for g in range(0, 5):
            assert f"reflection from gen {g}\n" not in per_task_prompt, (
                f"gen {g} should be outside the 6-gen window at gen 10"
            )
        for g in range(5, 11):
            assert f"reflection from gen {g}\n" in per_task_prompt, (
                f"gen {g} should be inside the 6-gen window at gen 10"
            )


def test_per_task_distill_window_zero_disables_capping(monkeypatch):
    """Setting window=0 reverts to legacy uncapped behaviour."""
    monkeypatch.setenv("KCSI_PER_TASK_DISTILL_GEN_WINDOW", "0")
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_per_task_attempt_history(Path(tmp), gens=list(range(0, 11)))

        captured: list[str] = []

        def capturing_llm(sys_prompt, user_prompt):
            captured.append(user_prompt)
            return json.dumps(
                {
                    "transferable_insights": [],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                }
            )

        distill(
            DistillInput(
                generation=10,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=capturing_llm,
            )
        )
        per_task_prompt = next(
            (p for p in captured if "reflection from gen" in p),
            None,
        )
        assert per_task_prompt is not None
        # All 11 gens should be present when window is disabled.
        for g in range(0, 11):
            assert f"reflection from gen {g}\n" in per_task_prompt, f"gen {g} dropped despite window=0"


def test_per_task_distill_custom_window(monkeypatch):
    """Window=3 at gen 10 keeps gens 8, 9, 10 only."""
    monkeypatch.setenv("KCSI_PER_TASK_DISTILL_GEN_WINDOW", "3")
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_per_task_attempt_history(Path(tmp), gens=list(range(0, 11)))

        captured: list[str] = []

        def capturing_llm(sys_prompt, user_prompt):
            captured.append(user_prompt)
            return json.dumps(
                {
                    "transferable_insights": [],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                }
            )

        distill(
            DistillInput(
                generation=10,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=capturing_llm,
            )
        )
        per_task_prompt = next(
            (p for p in captured if "reflection from gen" in p),
            None,
        )
        assert per_task_prompt is not None
        for g in range(0, 8):
            assert f"reflection from gen {g}\n" not in per_task_prompt
        for g in range(8, 11):
            assert f"reflection from gen {g}\n" in per_task_prompt


def test_per_task_distill_prompt_bounded_at_realistic_generation_count(monkeypatch):
    """Regression for #1014: 30 generations of a stuck task must not make the
    per-task distill prompt grow unboundedly. With the default 6-gen window,
    the prompt only ever carries ~6 gens' worth of attempts regardless of how
    many generations the task has been attempted across."""
    monkeypatch.delenv("KCSI_PER_TASK_DISTILL_GEN_WINDOW", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_per_task_attempt_history(Path(tmp), gens=list(range(0, 30)))

        captured: list[str] = []

        def capturing_llm(sys_prompt, user_prompt):
            captured.append(user_prompt)
            return json.dumps(
                {
                    "transferable_insights": [],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                }
            )

        distill(
            DistillInput(
                generation=29,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=capturing_llm,
            )
        )
        per_task_prompt = next(
            (p for p in captured if "reflection from gen" in p),
            None,
        )
        assert per_task_prompt is not None
        kept = [g for g in range(0, 30) if f"reflection from gen {g}\n" in per_task_prompt]
        assert kept == list(range(24, 30)), (
            f"expected only the last 6 of 30 generations to survive windowing, got {kept}"
        )
        # Prompt length must stay bounded near a single-window's worth of
        # attempts, not grow proportionally with all 30 generations attempted.
        assert len(per_task_prompt) < 20_000, (
            f"per-task distill prompt grew unbounded ({len(per_task_prompt)} chars) despite windowing"
        )


# ---------------------------------------------------------------------------
# Free-text fallback + no-repair-LLM contract (#645).
#
# Structured outputs (Anthropic tool-forcing / OpenAI Responses json_schema)
# now guarantee parseable JSON for providers that support them. The paid
# second "repair-LLM" call was removed: a plain (system, user) -> str stub
# (which does NOT accept json_schema) exercises the lenient free-text fallback
# path, and broken JSON simply yields None — with NO second LLM round-trip.
# ---------------------------------------------------------------------------


def test_distill_parses_free_text_json_via_fallback(monkeypatch):
    """A legacy (system, user) -> str stub goes through the lenient parser."""
    monkeypatch.delenv("KCSI_CROSS_TASK_DISTILL_GEN_WINDOW", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_db(Path(tmp))

        def fenced_json(sys_prompt, user_prompt):
            # Wrapped in a code fence + commentary: the lenient parser must
            # extract the first balanced object.
            return (
                "Here is the bundle:\n```json\n"
                + json.dumps(
                    {
                        "transferable_insights": ["recovered"],
                        "pitfalls": [],
                        "checks": [],
                        "evidence_post_ids": [],
                    }
                )
                + "\n```\ndone"
            )

        out = distill(
            DistillInput(
                generation=0,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=fenced_json,
            )
        )
        assert out.per_task.get("t1") is not None


def test_distill_broken_json_returns_none_without_second_call():
    """Broken free-text JSON yields None and does NOT trigger a repair call."""
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_db(Path(tmp))

        calls = {"n": 0}

        def always_broken(s, u):
            calls["n"] += 1
            # Mid-string unescaped quote: passes the brace counter, fails
            # json.loads, and no deterministic repair candidate fixes it.
            return '{"transferable_insights": ["bad "quote" inside]}'

        out = distill(
            DistillInput(
                generation=0,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=always_broken,
            )
        )
        assert out.per_task.get("t1") is None
        assert out.cross_task is None
        # Exactly one call per phase (per-task + cross-task) — no paid repair
        # round-trip. The removed repair path would have doubled this.
        assert calls["n"] == 2, f"expected 1 call per distill phase (no repair retry); saw {calls['n']}"


def test_distill_consumes_structured_payload_without_regex_path(monkeypatch):
    """When the caller supports json_schema, distill uses the provider's parsed
    dict directly and never touches the free-text brace-matcher / repair path."""
    import kcsi.distillation.per_task as per_task_mod

    monkeypatch.delenv("KCSI_CROSS_TASK_DISTILL_GEN_WINDOW", raising=False)

    def _boom(*_a, **_k):  # pragma: no cover - must never run
        raise AssertionError("free-text parser must not run for structured output")

    monkeypatch.setattr(per_task_mod, "_parse_json", _boom)
    monkeypatch.setattr(per_task_mod, "_parse_json_lenient", _boom)

    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_db(Path(tmp))

        seen_schema = {"n": 0}

        def structured_llm(sys_prompt, user_prompt, *, json_schema=None):
            # Caller advertised support by passing the schema through.
            assert json_schema is not None
            assert json_schema["name"] == "distill_bundle"
            seen_schema["n"] += 1
            parsed = {
                "transferable_insights": ["from structured output on post 1"],
                "pitfalls": [],
                "checks": [],
                "evidence_post_ids": [],
            }
            # Provider contract: (text, usage, parsed_dict). Text is raw JSON
            # but the distill path must prefer the parsed dict.
            return ("RAW TEXT THAT WOULD NOT PARSE", None, parsed)

        out = distill(
            DistillInput(
                generation=0,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=structured_llm,
            )
        )
        assert seen_schema["n"] >= 1, "json_schema was never passed to the caller"
        bundle = out.per_task.get("t1")
        assert bundle is not None
        assert bundle.transferable_insights == ["from structured output on post 1"]


def test_call_llm_reraises_internal_typeerror_instead_of_silent_fallback():
    """A TypeError raised *inside* a schema-capable callable is a real bug and
    must surface — it must NOT be misclassified as 'callable rejects json_schema'
    and silently retried without the schema (which masks the bug and double-calls).
    """
    from kcsi.distillation.per_task import _call_llm

    calls = {"n": 0}

    def buggy(sys_prompt, user_prompt, *, json_schema=None):
        calls["n"] += 1
        raise TypeError("internal bug: NoneType is not subscriptable")

    try:
        _call_llm(buggy, "sys", "user")
    except TypeError as exc:
        assert "internal bug" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("internal TypeError was swallowed instead of raised")
    # Exactly one call: no silent schema-less retry that would mask the bug.
    assert calls["n"] == 1, f"expected no fallback retry on internal TypeError; saw {calls['n']}"


def test_call_llm_falls_back_when_callable_rejects_json_schema_kwarg():
    """A callable that genuinely doesn't accept json_schema (raises the standard
    'unexpected keyword argument' TypeError) falls back to a schema-less call."""
    from kcsi.distillation.per_task import _call_llm

    def legacy(sys_prompt, user_prompt):
        return '{"transferable_insights": ["legacy"]}'

    raw, parsed = _call_llm(legacy, "sys", "user")
    assert parsed is None  # legacy callable returns no structured dict
    assert "legacy" in raw


def test_call_llm_folds_cache_prefix_back_when_callable_rejects_it():
    """A callable that accepts json_schema but NOT cache_prefix must still see
    the full prompt: the cross-task caller passes only the per-target suffix as
    user_prompt (the forum history lives entirely in cache_prefix), so the
    fallback must fold cache_prefix back into the user message rather than drop
    it — otherwise the history silently vanishes (issue #1252 item 3)."""
    from kcsi.distillation.per_task import _call_llm

    seen: dict[str, str] = {}

    def rejects_cache_prefix(sys_prompt, user_prompt, *, json_schema=None):
        # No cache_prefix kwarg → raises the standard TypeError when one is passed.
        seen["user"] = user_prompt
        return ('{"transferable_insights": ["ok"]}', None, None)

    raw, _parsed = _call_llm(rejects_cache_prefix, "sys", "SUFFIX-ONLY", cache_prefix="FORUM-HISTORY-PREFIX\n")
    assert "ok" in raw
    assert seen["user"] == "FORUM-HISTORY-PREFIX\nSUFFIX-ONLY", (
        "cache_prefix must be folded back into the user message, not dropped"
    )


def test_call_llm_empty_dict_payload_falls_back_to_lenient_parser():
    """An empty {} structured payload must NOT be accepted as a successful parse
    (which would silently produce an all-empty bundle). It falls through to the
    lenient free-text parser on the raw text instead."""
    from kcsi.distillation.per_task import _call_llm, distill_one_task

    def empty_struct(sys_prompt, user_prompt, *, json_schema=None):
        # Provider returned a syntactically-valid but empty object as the parsed
        # dict; the raw text carries the real content.
        return ('{"transferable_insights": ["recovered from raw at test_index 1"]}', None, {})

    raw, parsed = _call_llm(empty_struct, "sys", "user")
    assert parsed == {}  # _call_llm surfaces it; the consumer must treat it as falsy

    bundle = distill_one_task(task_id="t1", attempts=[], posts=[], llm=empty_struct)
    assert bundle is not None
    assert bundle.transferable_insights == ["recovered from raw at test_index 1"], (
        "empty {} structured payload should fall back to lenient parse of raw text, "
        "not produce a silent all-empty bundle"
    )


def test_call_llm_uses_injected_bundle_schema():
    """A custom bundle_schema is forwarded to the callable; absent it, the
    default DISTILL_BUNDLE_JSON_SCHEMA is used."""
    from kcsi.distillation.per_task import _call_llm

    seen = {}

    def llm(s, u, *, json_schema=None):
        seen["schema"] = json_schema
        return ("{}", None, None)

    custom = {"name": "custom_bundle", "schema": {"type": "object"}}
    _call_llm(llm, "s", "u", bundle_schema=custom)
    assert seen["schema"] is custom

    _call_llm(llm, "s", "u")
    assert seen["schema"]["name"] == "distill_bundle"


def test_distill_one_task_empty_response_returns_none_distinctly():
    """An empty LLM response (e.g. a tool-call decline) returns None and does
    not crash in the lenient parser."""
    from kcsi.distillation.per_task import distill_one_task

    def empty(s, u, *, json_schema=None):
        return ("", None, None)

    assert distill_one_task(task_id="t1", attempts=[], posts=[], llm=empty) is None


def test_distill_counts_failures_when_llm_yields_no_bundle():
    """#740 M4/H2: a per-task or cross-task distill that returns no bundle WITHOUT
    raising (LLM unavailable -> distill_one_task/distill_cross_task return None)
    must increment DistillOutput.failures, so a degraded generation isn't invisible."""
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_db(Path(tmp))  # t1 has an attempt + post; cross-task has a post

        def boom_llm(s, u):
            raise RuntimeError("LLM unavailable")

        out = distill(
            DistillInput(
                generation=0,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=boom_llm,
            )
        )
        # Both the per-task (t1 had an attempt) and the cross-task distill (had a
        # post) yielded no bundle without raising -> both counted.
        assert out.per_task == {}
        assert out.cross_task is None
        assert out.failures == 2


def test_distill_no_failures_on_healthy_run():
    """The clean-run invariant: a fully successful generation records 0 failures
    (so knowledge_phase_health stays empty for healthy runs)."""
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_db(Path(tmp))

        def good_llm(s, u):
            return json.dumps(
                {
                    "transferable_insights": ["i"],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                }
            )

        out = distill(
            DistillInput(
                generation=0,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=good_llm,
            )
        )
        assert out.failures == 0


class _StubKnowledgeStore:
    """Minimal query_task stub returning a single attempt with eval_results."""

    def __init__(self, eval_results: dict) -> None:
        self._eval_results = eval_results

    def query_task(self, task_id, *, generation=None, entry_types=None, limit=None):
        return {
            "attempts": [
                {
                    "agent_id": "a1",
                    "generation": 0,
                    "score": 0.0,
                    "content": {
                        "model_output": "out",
                        "eval_results": dict(self._eval_results),
                    },
                }
            ]
        }


def test_load_attempts_strips_hidden_eval_fields():
    """Fail-safe: _load_attempts applies the canonical upstream-strict redaction
    at the load boundary, so hidden test-runner tails, grader answer keys, and
    ARC per-test details cannot reach the distill prompt even if a future
    renderer edit dumps raw eval_results."""
    from kcsi.distillation.distiller import _load_attempts

    eval_results = {
        "resolved": False,
        "native_score": 0.0,
        "test_stdout_tail": "CANARY_POLY_TAIL",
        "test_stderr_tail": "CANARY_POLY_STDERR",
        "swebench_stdout_tail": "CANARY_SWE_TAIL",
        "swebench_stderr_tail": "CANARY_SWE_STDERR",
        # Grader answer keys (HIDDEN_EVAL_ANSWER_KEYS) must also be stripped.
        "detail": "CANARY_GOLD_DETAIL",
        "expected": "CANARY_EXPECTED",
        # ARC per-test entries projected to the safe {test_index, correct} keys.
        "arc_per_test": [{"test_index": 0, "correct": True, "detail": "CANARY_ARC_GOLD"}],
    }
    inp = DistillInput(
        generation=0,
        task_ids=["t1"],
        knowledge_store=_StubKnowledgeStore(eval_results),
        llm=lambda s, u: "{}",
    )
    loaded = _load_attempts(inp, "t1")
    assert len(loaded) == 1
    loaded_eval = loaded[0]["eval_results"]
    for hidden_key in (
        "test_stdout_tail",
        "test_stderr_tail",
        "swebench_stdout_tail",
        "swebench_stderr_tail",
        "detail",
        "expected",
    ):
        assert hidden_key not in loaded_eval
    # ARC per-test details are projected to the safe key subset.
    assert loaded_eval["arc_per_test"] == [{"test_index": 0, "correct": True}]
    # Declared experience scalars are retained.
    assert loaded_eval["native_score"] == 0.0
    assert loaded_eval["resolved"] is False


def test_load_attempts_strips_hidden_attempt_meta_and_trace():
    """The load-boundary redaction also covers hidden verifier transcripts in
    attempt_meta and hidden-marked text in the condensed trace."""
    from kcsi.distillation.distiller import _load_attempts

    class _RichStub:
        def query_task(self, task_id, *, generation=None, entry_types=None, limit=None):
            return {
                "attempts": [
                    {
                        "agent_id": "a1",
                        "generation": 0,
                        "score": 0.0,
                        "content": {
                            "model_output": "out",
                            "eval_results": {"native_score": 0.0},
                            "attempt_meta": {
                                "verifier_clues": "CANARY_CLUES",
                                "failure_signature": "CANARY_SIG",
                                "agent_exit_code": 0,
                            },
                            "trace_condensed": "tried X; verifier_clues=CANARY_TRACE; reward=0",
                        },
                    }
                ]
            }

    inp = DistillInput(
        generation=0,
        task_ids=["t1"],
        knowledge_store=_RichStub(),
        llm=lambda s, u: "{}",
    )
    loaded = _load_attempts(inp, "t1")
    assert len(loaded) == 1
    meta = loaded[0]["attempt_meta"]
    assert "verifier_clues" not in meta
    assert "failure_signature" not in meta
    # Declared scalar in attempt_meta is retained.
    assert meta["agent_exit_code"] == 0
    # Hidden-marked segment stripped from the condensed trace.
    assert "CANARY_TRACE" not in loaded[0]["trace_condensed"]


def _seed_cross_only(tmp: Path) -> KnowledgeStore:
    """Two attempted tasks + a cross-task post; no per-task attempts needed for
    the cross-task branch under test."""
    ks = KnowledgeStore(str(tmp / "kc.sqlite"), default_experiment="exp")
    ks.record_post(
        task_id=CROSS_TASK_SENTINEL,
        agent_id="a1",
        generation=0,
        text="cross-task pattern",
    )
    return ks


def _bundle_llm(seen_users):
    def fake_llm(s, u):
        seen_users.append(u)
        return json.dumps(
            {
                "transferable_insights": ["i"],
                "pitfalls": [],
                "checks": [],
                "evidence_post_ids": [],
            }
        )

    return fake_llm


def test_distill_conditioning_on_produces_bundle_per_task():
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_cross_only(Path(tmp))
        seen_users: list[str] = []
        out = distill(
            DistillInput(
                generation=0,
                task_ids=["t1", "t2"],
                knowledge_store=ks,
                llm=_bundle_llm(seen_users),
                cross_task_target_conditioning=True,
                target_task_prompts={"t1": "PROMPT-ONE", "t2": "PROMPT-TWO"},
            ),
            unsolved_task_ids=["t1", "t2"],
        )
        assert out.cross_task is None
        assert set(out.cross_task_by_task or {}) == {"t1", "t2"}
        joined = "\n".join(seen_users)
        assert "PROMPT-ONE" in joined and "PROMPT-TWO" in joined


def test_distill_conditioning_uses_explicit_cross_task_targets():
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_cross_only(Path(tmp))
        seen_users: list[str] = []
        out = distill(
            DistillInput(
                generation=0,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=_bundle_llm(seen_users),
                cross_task_target_conditioning=True,
                target_task_prompts={"t1": "PROMPT-ONE", "h1": "HOLDOUT-PROMPT"},
                cross_task_target_ids=["t1", "h1", "h1"],
            ),
            unsolved_task_ids=["t1"],
        )

        assert out.cross_task is None
        assert set(out.cross_task_by_task or {}) == {"t1", "h1"}
        joined = "\n".join(seen_users)
        assert "PROMPT-ONE" in joined
        assert "HOLDOUT-PROMPT" in joined


def test_distill_conditioning_skips_missing_target_prompt():
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_cross_only(Path(tmp))
        seen_users: list[str] = []
        out = distill(
            DistillInput(
                generation=0,
                task_ids=["t1"],
                knowledge_store=ks,
                llm=_bundle_llm(seen_users),
                cross_task_target_conditioning=True,
                target_task_prompts={"t1": "PROMPT-ONE"},
                cross_task_target_ids=["t1", "missing"],
            ),
            unsolved_task_ids=["t1"],
        )

        assert set(out.cross_task_by_task or {}) == {"t1"}
        assert out.failures == 1
        joined = "\n".join(seen_users)
        assert "PROMPT-ONE" in joined
        assert "Task ID: missing" not in joined


def test_distill_conditioning_off_keeps_single_bundle():
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_cross_only(Path(tmp))
        out = distill(
            DistillInput(
                generation=0,
                task_ids=["t1", "t2"],
                knowledge_store=ks,
                llm=_bundle_llm([]),
                cross_task_target_conditioning=False,
            ),
            unsolved_task_ids=["t1", "t2"],
        )
        assert out.cross_task is not None
        assert out.cross_task_by_task is None


# --- Per-target relevance selection (opt-in, deep-review H4) ------------------
#
# These exercise distiller.distill()'s new shared-vs-per-target branch under
# target-conditioning + an over-budget forum history (the #1258 89-task regime).
# The forum posts split into two disjoint vocabularies: half share TARGET_ALPHA's
# words, half share TARGET_OMEGA's. Over budget, per-target trimming keeps each
# target's own relevant posts; the shared-set default trims once against the
# largest target so both targets get a byte-identical post set (cache_prefix).

_A_VOCAB = "alpha bravo charlie delta echo foxtrot golf hotel"
_B_VOCAB = "omega psi chi phi upsilon sigma tau rho"


def _seed_relevance_split_cross(tmp: Path, *, n_each: int = 150) -> KnowledgeStore:
    """Seed ``n_each`` ALPHA-relevant + ``n_each`` OMEGA-relevant cross-task
    posts, all gen 0, so the whole history is comfortably over the cross-task
    distill budget (~131.8K tokens) — forcing trimming. The renderer clips each
    post to ``_POST_TEXT_EXCERPT_CHARS`` (2000), so ~300 posts (not a handful of
    huge ones) are what push the rendered prompt over budget.

    The unique ``POSTA<i>``/``POSTB<i>`` marker leads each post's text so it
    survives the renderer's 2000-char clip, letting a test detect which posts
    were delivered to each target. Filler words are disjoint from both target
    vocabularies so only the leading vocab words drive ``target_relevance``.
    """
    ks = KnowledgeStore(str(tmp / "kc.sqlite"), default_experiment="exp")
    # ~2.0K chars/post (clip boundary); 2*150 posts ~= 200K rendered tokens.
    filler = " concept context detail example " * 63
    for i in range(n_each):
        ks.record_post(
            task_id=CROSS_TASK_SENTINEL,
            agent_id=f"aa{i}",
            generation=0,
            text=f"POSTA{i} {_A_VOCAB} {filler}",
        )
        ks.record_post(
            task_id=CROSS_TASK_SENTINEL,
            agent_id=f"bb{i}",
            generation=0,
            text=f"POSTB{i} {_B_VOCAB} {filler}",
        )
    return ks


def _capture_prefix_llm(captured: dict[str, str]):
    """LLM stub that accepts ``cache_prefix`` and records the exact forum-post
    prefix delivered to each target (keyed by the target sentinel), returning a
    concrete bundle so the distiller yields a real CrossTaskBundle."""

    def llm(system, user, *, json_schema=None, cache_prefix=None):
        prefix = cache_prefix or ""
        full = prefix + user
        for key in ("TARGET_ALPHA", "TARGET_OMEGA"):
            if key in full:
                captured[key] = prefix
        return json.dumps(
            {
                "transferable_insights": [
                    "When solving the task, verify the output shape before submitting the attempt."
                ],
                "pitfalls": [],
                "checks": [],
                "evidence_post_ids": [],
            }
        )

    return llm


def _count_markers(prefix: str, letter: str, n_each: int) -> set[int]:
    return {i for i in range(n_each) if f"POST{letter}{i} " in prefix}


def _run_split_distill(ks, *, per_target: bool, captured: dict[str, str]):
    return distill(
        DistillInput(
            generation=0,
            task_ids=["t_a", "t_b"],
            knowledge_store=ks,
            llm=_capture_prefix_llm(captured),
            cross_task_target_conditioning=True,
            cross_task_per_target_selection=per_target,
            target_task_prompts={
                "t_a": f"TARGET_ALPHA {_A_VOCAB}",
                "t_b": f"TARGET_OMEGA {_B_VOCAB}",
            },
            cross_task_target_ids=["t_a", "t_b"],
        ),
        unsolved_task_ids=["t_a", "t_b"],
    )


def test_per_target_selection_off_gives_identical_post_set_across_targets():
    """DEFAULT (per_target_selection=False): over budget, the shared once-per-gen
    trim yields a byte-identical cache_prefix for every target — the #1258
    published, cache-optimal behavior."""
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_relevance_split_cross(Path(tmp))
        captured: dict[str, str] = {}
        out = _run_split_distill(ks, per_target=False, captured=captured)

        assert set(out.cross_task_by_task or {}) == {"t_a", "t_b"}
        prefix_alpha = captured["TARGET_ALPHA"]
        prefix_omega = captured["TARGET_OMEGA"]
        # The invariant: identical forum-post prefix delivered to both targets.
        assert prefix_alpha == prefix_omega
        # ...and trimming actually happened (not the trivial "all posts fit").
        n_delivered = len(_count_markers(prefix_alpha, "A", 150)) + len(_count_markers(prefix_alpha, "B", 150))
        assert 0 < n_delivered < 300


def test_per_target_selection_on_diverges_by_relevance():
    """OPT-IN (per_target_selection=True): each target trims its own
    relevance-ranked post set, so the two targets receive DIFFERENT posts
    (defeating the cross-target cache) — each biased toward its own vocabulary."""
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_relevance_split_cross(Path(tmp))
        captured: dict[str, str] = {}
        out = _run_split_distill(ks, per_target=True, captured=captured)

        assert set(out.cross_task_by_task or {}) == {"t_a", "t_b"}
        prefix_alpha = captured["TARGET_ALPHA"]
        prefix_omega = captured["TARGET_OMEGA"]

        a_in_alpha = _count_markers(prefix_alpha, "A", 150)
        b_in_alpha = _count_markers(prefix_alpha, "B", 150)
        a_in_omega = _count_markers(prefix_omega, "A", 150)
        b_in_omega = _count_markers(prefix_omega, "B", 150)

        # Each target keeps ALL of its own relevant posts (they rank first) ...
        assert len(a_in_alpha) == 150
        assert len(b_in_omega) == 150
        # ... and strictly fewer of the other target's posts (relevance bias) ...
        assert len(a_in_alpha) > len(a_in_omega)
        assert len(b_in_omega) > len(b_in_alpha)
        # ... so the delivered post sets differ per target: the cross-target
        # cache_prefix is DEFEATED (the documented cost of the opt-in).
        assert prefix_alpha != prefix_omega
        # Trimming happened for the ALPHA target (not all 300 posts fit).
        assert len(a_in_alpha) + len(b_in_alpha) < 300

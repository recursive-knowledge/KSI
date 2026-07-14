"""Tests for the KCSI_TRANSFER_BRIDGE flag (success-derived transfer bridge).

Covers the transfer-bridge design spec:
- resolver semantics (default off);
- flag off = behavior identical (no win-mode calls, no new prompt section,
  no extra store queries);
- flag on = win-mode per-task distill for newly-solved tasks;
- transferables collection caps (2/task, 480-char boundary, 20 tasks,
  freshest-bundle precedence, empty-insights skip, task_ids scoping);
- the cross-task prompt section in all three builders;
- budget estimator counts the section (posts trim first, transferables never).
"""

import json
import logging
import tempfile
from pathlib import Path

import pytest

from kcsi.distillation import DistillInput, PerTaskBundle, distill
from kcsi.distillation.cross_task import _select_cross_posts_for_budget, distill_cross_task
from kcsi.distillation.distiller import (
    _collect_per_task_transferables,
    _transfer_bridge_enabled,
)
from kcsi.distillation.per_task import truncate_at_boundary
from kcsi.distillation.prompts import (
    _TRANSFERABLES_DIRECTIVE,
    _TRANSFERABLES_SECTION_TITLE,
    build_cross_task_distill_prompt,
    build_per_task_distill_prompt,
)
from kcsi.memory.knowledge_store import CROSS_TASK_SENTINEL, KnowledgeStore
from tests.orchestrator_phase_helpers import run_distill

_WIN_DIRECTIVE_MARK = "This task was SOLVED this generation"
_SECTION_TITLE = "## Per-task transferable candidates (distilled from per-task bundles)"
_SECTION_DIRECTIVE = (
    "These are transferable-insight candidates distilled from individual "
    "per-task bundles (both solved and still-unsolved tasks); treat them as "
    "candidate transferable insights — generalize and merge them with forum "
    "evidence rather than restating them verbatim."
)


def _bundle_json(insights=()):
    return json.dumps(
        {
            "transferable_insights": list(insights),
            "pitfalls": [],
            "checks": [],
            "evidence_post_ids": [],
        }
    )


def _seed_two_task_db(tmp: Path) -> KnowledgeStore:
    """t1 = solved this gen, t2 = still unsolved; one cross-task post."""
    ks = KnowledgeStore(str(tmp / "k.sqlite"), default_experiment="exp")
    ks.record_attempt(task_id="t1", agent_id="a1", generation=0, model_output="won t1", native_score=1.0)
    ks.record_attempt(task_id="t2", agent_id="a2", generation=0, model_output="failed t2", native_score=0.0)
    ks.record_post(
        task_id=CROSS_TASK_SENTINEL,
        agent_id="a1",
        generation=0,
        text="cross-task pattern",
        source_phase="cross_task_forum",
    )
    return ks


class _QueryCountingStore:
    """Delegating proxy that counts distillation-bucket query_task calls."""

    def __init__(self, ks: KnowledgeStore):
        self._ks = ks
        self.distillation_queries = 0

    def query_task(self, *args, **kwargs):
        if kwargs.get("entry_types") == ["distillation"]:
            self.distillation_queries += 1
        return self._ks.query_task(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._ks, name)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "On"])
def test_transfer_bridge_resolver_on_values(monkeypatch, raw):
    monkeypatch.setenv("KCSI_TRANSFER_BRIDGE", raw)
    assert _transfer_bridge_enabled() is True


@pytest.mark.parametrize("raw", [None, "", "0", "off", "false", "bridge"])
def test_transfer_bridge_resolver_off_values(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv("KCSI_TRANSFER_BRIDGE", raising=False)
    else:
        monkeypatch.setenv("KCSI_TRANSFER_BRIDGE", raw)
    assert _transfer_bridge_enabled() is False


# ---------------------------------------------------------------------------
# Flag off: behavior identical
# ---------------------------------------------------------------------------


def test_flag_off_no_win_calls_no_section_no_extra_queries(monkeypatch):
    monkeypatch.delenv("KCSI_TRANSFER_BRIDGE", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        ks = _QueryCountingStore(_seed_two_task_db(Path(tmp)))
        captured: list[tuple[str, str]] = []

        def llm(sys_prompt, user_prompt):
            captured.append((sys_prompt, user_prompt))
            return _bundle_json(["something"])

        out = distill(
            DistillInput(
                generation=0,
                task_ids=["t1", "t2"],
                knowledge_store=ks,
                llm=llm,
            ),
            unsolved_task_ids=["t2"],
            newly_solved_task_ids=["t1"],
        )
        # Solved task stays skipped; no win-mode call happened.
        assert "t1" not in out.per_task
        assert "t2" in out.per_task
        full = "\n".join(s + u for s, u in captured)
        assert _WIN_DIRECTIVE_MARK not in full
        assert _SECTION_TITLE not in full
        # No new store queries against the distillation bucket.
        assert ks.distillation_queries == 0


# ---------------------------------------------------------------------------
# Flag on: win-mode per-task distill
# ---------------------------------------------------------------------------


def test_flag_on_newly_solved_triggers_one_win_mode_distill(monkeypatch):
    monkeypatch.setenv("KCSI_TRANSFER_BRIDGE", "1")
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_two_task_db(Path(tmp))
        captured: list[tuple[str, str]] = []

        def llm(sys_prompt, user_prompt):
            captured.append((sys_prompt, user_prompt))
            return _bundle_json([{"text": "winning move", "applies_when": "always"}])

        out = distill(
            DistillInput(
                generation=0,
                task_ids=["t1", "t2"],
                knowledge_store=ks,
                llm=llm,
            ),
            unsolved_task_ids=["t2"],
            newly_solved_task_ids=["t1"],
        )
        # The win bundle lands in per_task_results like a normal bundle.
        assert isinstance(out.per_task.get("t1"), PerTaskBundle)
        assert isinstance(out.per_task.get("t2"), PerTaskBundle)
        win_calls = [(s, u) for s, u in captured if _WIN_DIRECTIVE_MARK in s]
        assert len(win_calls) == 1
        assert "Task ID: t1" in win_calls[0][1]
        # The unsolved task's call carries no win directive.
        t2_calls = [(s, u) for s, u in captured if "Task ID: t2" in u]
        assert t2_calls and all(_WIN_DIRECTIVE_MARK not in s for s, _ in t2_calls)


def test_flag_on_newly_solved_intersected_with_task_ids(monkeypatch):
    """A task id outside inp.task_ids (e.g. a hold-out) never gets a win call."""
    monkeypatch.setenv("KCSI_TRANSFER_BRIDGE", "1")
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_two_task_db(Path(tmp))
        captured: list[str] = []

        def llm(sys_prompt, user_prompt):
            captured.append(user_prompt)
            return _bundle_json()

        out = distill(
            DistillInput(
                generation=0,
                task_ids=["t1", "t2"],
                knowledge_store=ks,
                llm=llm,
            ),
            unsolved_task_ids=["t2"],
            newly_solved_task_ids=["t1", "holdout_x"],
        )
        assert "holdout_x" not in out.per_task
        assert not any("Task ID: holdout_x" in u for u in captured)


# ---------------------------------------------------------------------------
# Transferables collection
# ---------------------------------------------------------------------------


def _inp(ks, task_ids, generation=1) -> DistillInput:
    return DistillInput(
        generation=generation,
        task_ids=task_ids,
        knowledge_store=ks,
        llm=lambda s, u: _bundle_json(),
    )


def _record_bundle(ks, task_id, generation, insights):
    ks.record_distillation(
        task_id=task_id,
        generation=generation,
        bundle={"transferable_insights": insights, "pitfalls": [], "checks": [], "evidence_post_ids": []},
        scope="per_task",
    )


def test_collect_fresh_bundle_supersedes_stored(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        _record_bundle(ks, "t1", 0, [{"text": "stale stored", "applies_when": "old"}])
        fresh = {
            "t1": PerTaskBundle(
                task_id="t1",
                transferable_insights=[{"text": "fresh win", "applies_when": "new"}],
                pitfalls=[],
                checks=[],
                evidence_post_ids=[],
            )
        }
        out = _collect_per_task_transferables(_inp(ks, ["t1"]), fresh)
        assert out == [{"task_id": "t1", "text": "fresh win", "applies_when": "new"}]


def test_collect_latest_stored_bundle_wins():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        _record_bundle(ks, "t1", 0, [{"text": "old insight", "applies_when": ""}])
        _record_bundle(ks, "t1", 1, [{"text": "new insight", "applies_when": "later"}])
        out = _collect_per_task_transferables(_inp(ks, ["t1"]), {})
        assert out == [{"task_id": "t1", "text": "new insight", "applies_when": "later"}]


def test_collect_skips_empty_insights_and_caps_two_per_task():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        _record_bundle(ks, "t_empty", 0, [])
        _record_bundle(
            ks,
            "t_many",
            0,
            [{"text": f"insight {i}", "applies_when": ""} for i in range(4)],
        )
        out = _collect_per_task_transferables(_inp(ks, ["t_empty", "t_many"]), {})
        assert [e["task_id"] for e in out] == ["t_many", "t_many"]
        assert [e["text"] for e in out] == ["insight 0", "insight 1"]


def test_collect_truncates_text_at_boundary_480():
    long_text = ("alpha beta gamma delta " * 40).strip()  # ~920 chars
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        _record_bundle(ks, "t1", 0, [{"text": long_text, "applies_when": ""}])
        out = _collect_per_task_transferables(_inp(ks, ["t1"]), {})
        assert len(out) == 1
        assert out[0]["text"] == truncate_at_boundary(long_text, 480)
        assert len(out[0]["text"]) <= 480


def test_collect_caps_at_20_tasks_keeping_highest_generation():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        task_ids = [f"t{g:02d}" for g in range(25)]
        for g, tid in enumerate(task_ids):
            _record_bundle(ks, tid, g, [{"text": f"win on {tid}", "applies_when": ""}])
        out = _collect_per_task_transferables(_inp(ks, task_ids, generation=25), {})
        kept = {e["task_id"] for e in out}
        assert len(kept) == 20
        # The 20 highest-generation bundles (gens 5..24) survive.
        assert kept == {f"t{g:02d}" for g in range(5, 25)}


def test_collect_never_includes_tasks_outside_task_ids():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        _record_bundle(ks, "t1", 0, [{"text": "in scope", "applies_when": ""}])
        _record_bundle(ks, "holdout_x", 0, [{"text": "out of scope", "applies_when": ""}])
        out = _collect_per_task_transferables(_inp(ks, ["t1"]), {})
        assert {e["task_id"] for e in out} == {"t1"}


def test_collect_accepts_legacy_string_insights():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        _record_bundle(ks, "t1", 0, ["legacy bare string insight"])
        out = _collect_per_task_transferables(_inp(ks, ["t1"]), {})
        assert out == [{"task_id": "t1", "text": "legacy bare string insight", "applies_when": ""}]


def test_transferables_prompt_matches_broad_sourcing():
    """Prompt-vs-behavior contract (deep-review #1264 High 2c).

    Post-#1247 (revert of the #1230 win-mode filter — deliberate: the owner is
    "not pursuing the transfer-bridge success-derived route"),
    ``_collect_per_task_transferables`` sources from EVERY fresh per-task
    bundle — solved AND still-unsolved — plus stored bundles. The rendered
    section framing must therefore NOT claim the candidates are "verified
    solves" from "solved tasks", or it mislabels failure-derived insights.
    Pins both sides so they can't drift apart again.
    """
    # (behavior) a still-unsolved task's fresh bundle DOES reach the collector.
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        unsolved_fresh = {
            "t_unsolved": PerTaskBundle(
                task_id="t_unsolved",
                transferable_insights=[{"text": "from a still-failing task", "applies_when": "x"}],
                pitfalls=[],
                checks=[],
                evidence_post_ids=[],
            )
        }
        out = _collect_per_task_transferables(_inp(ks, ["t_unsolved"]), unsolved_fresh)
        assert out == [{"task_id": "t_unsolved", "text": "from a still-failing task", "applies_when": "x"}]

    # (prompt) the source-of-truth section framing does not overclaim provenance
    # with the reverted "verified solves" / "from solved tasks" wording.
    framing = f"{_TRANSFERABLES_SECTION_TITLE} {_TRANSFERABLES_DIRECTIVE}".lower()
    assert "verified solves" not in framing
    assert "from solved tasks" not in framing
    assert "winning techniques" not in framing
    # test-local copies stay in sync with the real prompt constants.
    assert _SECTION_TITLE == _TRANSFERABLES_SECTION_TITLE
    assert _SECTION_DIRECTIVE == _TRANSFERABLES_DIRECTIVE


# ---------------------------------------------------------------------------
# Prompt builders: the success-derived section
# ---------------------------------------------------------------------------

_TRANSFERABLES = [
    {"task_id": "t1", "text": "use flood-fill", "applies_when": "grid has symmetric halves"},
    {"task_id": "t2", "text": "pin the fixture", "applies_when": ""},
]


def test_cross_task_prompt_renders_section_when_nonempty():
    _, user = build_cross_task_distill_prompt(
        cross_posts=[{"id": 1, "agent_id": "a", "text": "p"}],
        per_task_transferables=_TRANSFERABLES,
    )
    assert _SECTION_TITLE in user
    assert "- [t1] use flood-fill (applies when: grid has symmetric halves)" in user
    assert "- [t2] pin the fixture" in user
    assert _SECTION_DIRECTIVE in user


def test_cross_task_prompt_omits_section_when_empty():
    posts = [{"id": 1, "agent_id": "a", "text": "p"}]
    base_sys, base_user = build_cross_task_distill_prompt(cross_posts=posts)
    for empty in (None, []):
        sys_p, user_p = build_cross_task_distill_prompt(cross_posts=posts, per_task_transferables=empty)
        assert (sys_p, user_p) == (base_sys, base_user)
    assert _SECTION_TITLE not in base_user


def test_per_task_prompt_win_directive():
    sys_on, _ = build_per_task_distill_prompt(task_id="t1", attempts=[], posts=[], win_mode=True)
    sys_off, _ = build_per_task_distill_prompt(task_id="t1", attempts=[], posts=[])
    assert _WIN_DIRECTIVE_MARK in sys_on
    assert "Emit 1-3 transferable_insights items" in sys_on
    assert _WIN_DIRECTIVE_MARK not in sys_off


# ---------------------------------------------------------------------------
# Budget: estimator counts the section; posts trim first
# ---------------------------------------------------------------------------


def test_budget_estimator_counts_transferables_section():
    posts = [{"id": i, "agent_id": f"a{i}", "generation": 0, "text": "x" * 600} for i in range(30)]
    transferables = [{"task_id": f"t{i}", "text": "y" * 470, "applies_when": "z" * 150} for i in range(20)]
    budget = 8_000
    without = _select_cross_posts_for_budget(
        cross_posts=posts,
        task_source=None,
        max_input_tokens=budget,
    )
    with_t = _select_cross_posts_for_budget(
        cross_posts=posts,
        task_source=None,
        max_input_tokens=budget,
        per_task_transferables=transferables,
    )
    # The section consumes budget, so MORE posts must be trimmed to
    # compensate — transferables themselves are never trimmed.
    assert len(with_t) < len(without)
    _, user = build_cross_task_distill_prompt(
        cross_posts=with_t,
        per_task_transferables=transferables,
    )
    assert _SECTION_TITLE in user
    assert user.count("- [t") == 20


# ---------------------------------------------------------------------------
# End-to-end: flag on routes fresh win transferables into the cross-task prompt
# ---------------------------------------------------------------------------


def test_flag_on_cross_task_prompt_carries_fresh_win_transferables(monkeypatch):
    monkeypatch.setenv("KCSI_TRANSFER_BRIDGE", "1")
    with tempfile.TemporaryDirectory() as tmp:
        ks = _seed_two_task_db(Path(tmp))
        captured: list[str] = []

        def llm(sys_prompt, user_prompt):
            captured.append(user_prompt)
            return _bundle_json([{"text": "winning move via grid rotation", "applies_when": "always"}])

        distill(
            DistillInput(
                generation=0,
                task_ids=["t1", "t2"],
                knowledge_store=ks,
                llm=llm,
            ),
            unsolved_task_ids=["t2"],
            newly_solved_task_ids=["t1"],
        )
        cross_prompts = [u for u in captured if "cross-task pattern" in u]
        assert cross_prompts, f"cross-task prompt not captured: {captured}"
        assert _SECTION_TITLE in cross_prompts[-1]
        assert "- [t1] winning move via grid rotation (applies when: always)" in cross_prompts[-1]


# ---------------------------------------------------------------------------
# Visibility log: win tasks count toward attempted/produced
# ---------------------------------------------------------------------------

_PRODUCED_LOG_MARK = "per-task bundle(s)"


def test_attempted_accounting_counts_win_tasks(monkeypatch, caplog):
    """1 unsolved + 2 win tasks, one win fails: the no-bundle delta counts the
    failed win task and never goes negative."""
    monkeypatch.setenv("KCSI_TRANSFER_BRIDGE", "1")
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        ks.record_attempt(task_id="t_u", agent_id="a1", generation=0, model_output="failed", native_score=0.0)
        ks.record_attempt(task_id="t_w1", agent_id="a2", generation=0, model_output="won", native_score=1.0)
        ks.record_attempt(task_id="t_w2", agent_id="a3", generation=0, model_output="won", native_score=1.0)

        def llm(sys_prompt, user_prompt):
            if "Task ID: t_w2" in user_prompt:
                return "not json"  # this win distill fails
            return _bundle_json(["x"])

        with caplog.at_level(logging.INFO, logger="kcsi.distillation.distiller"):
            out = distill(
                DistillInput(
                    generation=0,
                    task_ids=["t_u", "t_w1", "t_w2"],
                    knowledge_store=ks,
                    llm=llm,
                ),
                unsolved_task_ids=["t_u"],
                newly_solved_task_ids=["t_w1", "t_w2"],
            )
        assert set(out.per_task) == {"t_u", "t_w1"}
        msgs = [r.getMessage() for r in caplog.records if _PRODUCED_LOG_MARK in r.getMessage()]
        assert msgs, f"produced/attempted log not emitted: {[r.getMessage() for r in caplog.records]}"
        assert "produced 2/3 per-task bundle(s); 1 task(s) yielded" in msgs[0]


def test_only_win_tasks_total_failure_still_warns(monkeypatch, caplog):
    """All tasks solved (target empty, only win distills ran) and every win
    fails: the silent-degradation WARNING still fires."""
    monkeypatch.setenv("KCSI_TRANSFER_BRIDGE", "1")
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        ks.record_attempt(task_id="t_w", agent_id="a1", generation=0, model_output="won", native_score=1.0)

        with caplog.at_level(logging.INFO, logger="kcsi.distillation.distiller"):
            out = distill(
                DistillInput(
                    generation=0,
                    task_ids=["t_w"],
                    knowledge_store=ks,
                    llm=lambda s, u: "not json",
                ),
                unsolved_task_ids=[],
                newly_solved_task_ids=["t_w"],
            )
        assert out.per_task == {}
        recs = [r for r in caplog.records if _PRODUCED_LOG_MARK in r.getMessage()]
        assert recs, "produced/attempted log must run when only win tasks were attempted"
        assert recs[0].levelno == logging.WARNING
        assert "produced 0/1 per-task bundle(s); 1 task(s) yielded" in recs[0].getMessage()


# ---------------------------------------------------------------------------
# Engine: win extraction requires --drop-solved
# ---------------------------------------------------------------------------


def _make_engine_orch(tmp_path, *, drop_solved: bool):
    from unittest.mock import MagicMock

    from kcsi.models import GenerationConfig
    from kcsi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
    from kcsi.tokens import LLMResponse, TokenUsage

    llm = MagicMock()
    llm.call.return_value = LLMResponse(text=_bundle_json(), usage=TokenUsage(input_tokens=1, output_tokens=1))
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        knowledge_db_path=str(tmp_path / f"k_drop_{drop_solved}.sqlite"),
        drop_solved=drop_solved,
    )
    return GenerationalOrchestrator(
        config=config,
        runtime=MagicMock(),
        evaluator=MagicMock(),
        llm=llm,
        persistence=NoopPersistence(),
    )


@pytest.mark.parametrize("drop_solved", [True, False])
def test_engine_gates_newly_solved_on_drop_solved(tmp_path, monkeypatch, drop_solved):
    """With --no-drop-solved a solved task stays dispatched every generation,
    so win extraction would re-run unbounded — the engine must pass None."""
    import kcsi.distillation as dist_pkg
    from kcsi.distillation import DistillOutput

    orch = _make_engine_orch(tmp_path, drop_solved=drop_solved)
    orch._knowledge.record_attempt(task_id="t1", agent_id="a1", generation=0, model_output="won", native_score=1.0)

    captured = {}

    def fake_distill(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
        captured["newly"] = newly_solved_task_ids
        return DistillOutput(per_task={}, cross_task=None)

    monkeypatch.setattr(dist_pkg, "distill", fake_distill)
    run_distill(orch, generation=0, task_ids=["t1"])
    assert captured["newly"] == (["t1"] if drop_solved else None)


def test_no_drop_solved_keeps_solved_tasks_as_cross_task_targets(tmp_path, monkeypatch):
    import kcsi.distillation as dist_pkg
    from kcsi.distillation import DistillOutput

    orch = _make_engine_orch(tmp_path, drop_solved=False)
    orch._tasks_by_id = {
        "t1": type("Task", (), {"prompt": "prompt one", "metadata": {}})(),
        "t2": type("Task", (), {"prompt": "prompt two", "metadata": {}})(),
    }
    orch._knowledge.record_attempt(task_id="t1", agent_id="a1", generation=0, model_output="won", native_score=1.0)

    captured = {}

    def fake_distill(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
        captured["task_ids"] = list(inp.task_ids)
        captured["unsolved"] = list(unsolved_task_ids or [])
        captured["cross_task_target_ids"] = list(inp.cross_task_target_ids or [])
        captured["target_task_prompts"] = dict(inp.target_task_prompts or {})
        return DistillOutput(per_task={}, cross_task=None)

    monkeypatch.setattr(dist_pkg, "distill", fake_distill)
    run_distill(orch, generation=0, task_ids=["t1", "t2"])

    assert captured["task_ids"] == ["t1", "t2"]
    assert captured["unsolved"] == ["t2"]
    assert captured["cross_task_target_ids"] == ["t1", "t2"]
    assert captured["target_task_prompts"] == {"t1": "prompt one", "t2": "prompt two"}


def test_distill_phase_leaves_missing_target_prompts_absent(tmp_path, monkeypatch):
    import kcsi.distillation as dist_pkg
    from kcsi.distillation import DistillOutput
    from kcsi.models import TaskSpec

    orch = _make_engine_orch(tmp_path, drop_solved=True)
    orch._tasks_by_id = {"t1": TaskSpec(id="t1", prompt="prompt one")}
    captured = {}

    def fake_distill(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
        captured["cross_task_target_ids"] = list(inp.cross_task_target_ids or [])
        captured["target_task_prompts"] = dict(inp.target_task_prompts or {})
        return DistillOutput(per_task={}, cross_task=None)

    monkeypatch.setattr(dist_pkg, "distill", fake_distill)
    run_distill(orch, generation=0, task_ids=["t1", "missing"])

    assert captured["cross_task_target_ids"] == ["t1", "missing"]
    assert captured["target_task_prompts"] == {"t1": "prompt one"}


# ---------------------------------------------------------------------------
# Collection: cap ordering
# ---------------------------------------------------------------------------


def test_collect_cap_output_preserves_task_ids_order():
    """After the 20-task cap, output order follows inp.task_ids, not bundle
    generation order."""
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        task_ids = [f"t{i:02d}" for i in range(25)]
        gens = [(i * 7) % 25 for i in range(25)]  # scrambled permutation of 0..24
        for tid, g in zip(task_ids, gens):
            _record_bundle(ks, tid, g, [{"text": f"win on {tid}", "applies_when": ""}])
        out = _collect_per_task_transferables(_inp(ks, task_ids, generation=25), {})
        expected = [tid for tid, g in zip(task_ids, gens) if g >= 5]  # 20 highest gens
        assert [e["task_id"] for e in out] == expected


def test_collect_truncates_applies_when_at_200():
    long_aw = ("alpha beta gamma " * 30).strip()  # ~500 chars
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        _record_bundle(ks, "t1", 0, [{"text": "win", "applies_when": long_aw}])
        out = _collect_per_task_transferables(_inp(ks, ["t1"]), {})
        assert out[0]["applies_when"] == truncate_at_boundary(long_aw, 200)
        assert len(out[0]["applies_when"]) <= 200


# ---------------------------------------------------------------------------
# Overflow retry keeps the transferables section
# ---------------------------------------------------------------------------


def test_overflow_retry_reselection_keeps_transferables():
    """When the provider rejects the prompt as too long, the retry's
    re-selection must still pass per_task_transferables: the section stays in
    the retry prompt and the trimmer compensates by dropping more posts."""
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

    call_prompts: list[str] = []

    def flaky_llm(system_prompt: str, user_prompt: str) -> str:
        call_prompts.append(user_prompt)
        if len(call_prompts) == 1:
            raise RuntimeError("prompt is too long: 200513 tokens > 200000 maximum")
        return _bundle_json(["survivor"])

    bundle = distill_cross_task(
        cross_posts=posts,
        llm=flaky_llm,
        task_source="polyglot",
        per_task_transferables=_TRANSFERABLES,
    )
    assert bundle is not None
    assert len(call_prompts) == 2
    # The section survives BOTH the first selection and the overflow retry.
    for prompt in call_prompts:
        assert _SECTION_TITLE in prompt
        assert "- [t1] use flood-fill (applies when: grid has symmetric halves)" in prompt
    # The retry trimmed more posts to compensate — transferables intact.
    assert call_prompts[1].count("- id=") < call_prompts[0].count("- id=")

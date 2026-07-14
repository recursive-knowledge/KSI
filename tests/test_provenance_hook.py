"""Tests for the knowledge -> solve provenance hook.

The hook spans three pieces:

1. ``KnowledgeStore.load_distillation`` stamps ``_knowledge_id`` on the
   returned bundle dict so seed packages carry a backref.
2. ``GenerationalOrchestrator._retrieved_distillation_ids`` reads those
   IDs back from ``agent.seed_package``.
3. ``GenerationalOrchestrator._merge_attempt_meta`` folds them into the
   carry-forward payload (or builds a fresh dict) for ``record_attempt``.

These tests cover (1) directly and (2)+(3) at the unit level so the
end-to-end provenance edge can be verified without standing up a full
orchestrator + container runtime.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from ksi.memory.knowledge_store import (
    CROSS_TASK_SENTINEL,
    KnowledgeStore,
)
from ksi.orchestrator.engine import GenerationalOrchestrator

# ---------------------------------------------------------------------------
# load_distillation — stamps _knowledge_id
# ---------------------------------------------------------------------------


def test_load_distillation_stamps_knowledge_id_per_task():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            rid = ks.record_distillation(
                task_id="t1",
                generation=1,
                bundle={
                    "transferable_insights": ["i1"],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                },
                scope="per_task",
            )
            loaded = ks.load_distillation(generation=1, task_id="t1", scope="per_task")
            assert loaded is not None
            assert loaded["_knowledge_id"] == rid
            # Original content fields still intact
            assert loaded["transferable_insights"] == ["i1"]
            assert loaded["scope"] == "per_task"
        finally:
            ks.close()


def test_load_distillation_stamps_knowledge_id_cross_task():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            rid = ks.record_distillation(
                task_id=CROSS_TASK_SENTINEL,
                generation=2,
                bundle={
                    "transferable_insights": ["x1"],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                },
                scope="cross_task",
            )
            loaded = ks.load_distillation(
                generation=2,
                task_id=CROSS_TASK_SENTINEL,
                scope="cross_task",
            )
            assert loaded is not None
            assert loaded["_knowledge_id"] == rid
            assert loaded["scope"] == "cross_task"
        finally:
            ks.close()


def test_load_distillation_picks_latest_when_overwritten():
    """Later distillation for the same (gen, task, scope) should win — and the
    returned ``_knowledge_id`` must point to the *latest* row, since that's
    what the seeder will inject and what attempt_meta will reference."""
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            rid_first = ks.record_distillation(
                task_id="t1",
                generation=1,
                bundle={"transferable_insights": ["v1"], "pitfalls": [], "checks": [], "evidence_post_ids": []},
                scope="per_task",
            )
            rid_second = ks.record_distillation(
                task_id="t1",
                generation=1,
                bundle={"transferable_insights": ["v2"], "pitfalls": [], "checks": [], "evidence_post_ids": []},
                scope="per_task",
            )
            assert rid_second > rid_first
            loaded = ks.load_distillation(generation=1, task_id="t1", scope="per_task")
            assert loaded is not None
            assert loaded["_knowledge_id"] == rid_second
            assert loaded["transferable_insights"] == ["v2"]
        finally:
            ks.close()


# ---------------------------------------------------------------------------
# _retrieved_distillation_ids — reads from agent.seed_package
# ---------------------------------------------------------------------------


def _agent(seed_package):
    return SimpleNamespace(seed_package=seed_package)


def test_retrieved_ids_returns_none_for_missing_agent():
    assert GenerationalOrchestrator._retrieved_distillation_ids(None) is None


def test_retrieved_ids_returns_none_for_empty_seed():
    assert GenerationalOrchestrator._retrieved_distillation_ids(_agent({})) is None
    assert GenerationalOrchestrator._retrieved_distillation_ids(_agent(None)) is None


def test_retrieved_ids_reads_cross_task_only():
    pkg = {"cross_task_bundle": {"_knowledge_id": 42, "transferable_insights": ["x"]}}
    out = GenerationalOrchestrator._retrieved_distillation_ids(_agent(pkg))
    assert out == {"cross_task": 42}


def test_retrieved_ids_reads_per_task_only():
    pkg = {"per_task_bundle": {"_knowledge_id": 17, "transferable_insights": ["x"]}}
    out = GenerationalOrchestrator._retrieved_distillation_ids(_agent(pkg))
    assert out == {"per_task": 17}


def test_retrieved_ids_reads_both():
    pkg = {
        "cross_task_bundle": {"_knowledge_id": 42},
        "per_task_bundle": {"_knowledge_id": 17},
    }
    out = GenerationalOrchestrator._retrieved_distillation_ids(_agent(pkg))
    assert out == {"cross_task": 42, "per_task": 17}


def test_retrieved_ids_skips_bundles_without_knowledge_id():
    """A bundle dict without _knowledge_id (e.g. legacy seed) shouldn't crash."""
    pkg = {
        "cross_task_bundle": {"transferable_insights": ["x"]},  # no _knowledge_id
        "per_task_bundle": {"_knowledge_id": 99},
    }
    out = GenerationalOrchestrator._retrieved_distillation_ids(_agent(pkg))
    assert out == {"per_task": 99}


def test_retrieved_ids_ignores_non_int_knowledge_id():
    """Defensive: corrupted seed shouldn't propagate string IDs into provenance."""
    pkg = {
        "cross_task_bundle": {"_knowledge_id": "bogus"},
        "per_task_bundle": {"_knowledge_id": 5},
    }
    out = GenerationalOrchestrator._retrieved_distillation_ids(_agent(pkg))
    assert out == {"per_task": 5}


# ---------------------------------------------------------------------------
# _merge_attempt_meta — folds retrieved IDs into carry-forward base
# ---------------------------------------------------------------------------


def test_merge_returns_none_when_both_empty():
    assert GenerationalOrchestrator._merge_attempt_meta(None, None) is None


def test_merge_returns_base_unchanged_when_no_retrieval():
    base = {"carry_forward": True, "carry_forward_source_score": 0.9}
    out = GenerationalOrchestrator._merge_attempt_meta(base, None)
    assert out is base  # short-circuit, no copy
    out2 = GenerationalOrchestrator._merge_attempt_meta(base, {})
    assert out2 is base


def test_merge_builds_fresh_dict_when_only_retrieval():
    out = GenerationalOrchestrator._merge_attempt_meta(None, {"cross_task": 16, "per_task": 14})
    assert out == {"retrieved_distillation_ids": {"cross_task": 16, "per_task": 14}}


def test_merge_keeps_carry_forward_and_adds_retrieval():
    """Both fields must coexist — carry-forward replays a prior best while
    retrieval records what knowledge the (replayed) agent had access to."""
    base = {
        "carry_forward": True,
        "carry_forward_reason": "best_score_preserved",
        "carry_forward_source_generation": 3,
        "carry_forward_source_score": 0.85,
    }
    retrieved = {"cross_task": 100, "per_task": 200}
    out = GenerationalOrchestrator._merge_attempt_meta(base, retrieved)
    assert out is not None
    # base fields preserved
    assert out["carry_forward"] is True
    assert out["carry_forward_reason"] == "best_score_preserved"
    assert out["carry_forward_source_generation"] == 3
    assert out["carry_forward_source_score"] == 0.85
    # retrieval added
    assert out["retrieved_distillation_ids"] == retrieved
    # base dict not mutated
    assert "retrieved_distillation_ids" not in base


# ---------------------------------------------------------------------------
# Integration: end-to-end through KnowledgeStore.record_attempt
# ---------------------------------------------------------------------------


def test_record_attempt_persists_retrieved_ids_in_content_json():
    """Round-trip: write an attempt with attempt_meta containing retrieved_distillation_ids,
    read it back from the knowledge content JSON."""
    import json

    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            attempt_id = ks.record_attempt(
                task_id="t1",
                agent_id="agent-0",
                generation=2,
                eval_results={"native_score": 0.5},
                model_output="...",
                trace_condensed="...",
                native_score=0.5,
                attempt_meta={
                    "retrieved_distillation_ids": {"cross_task": 16, "per_task": 14},
                },
            )
            assert attempt_id > 0
            row = (
                ks._connection()
                .execute(
                    "SELECT content FROM knowledge WHERE id = ?",
                    (attempt_id,),
                )
                .fetchone()
            )
            content = json.loads(row[0])
            assert content["attempt_meta"]["retrieved_distillation_ids"] == {
                "cross_task": 16,
                "per_task": 14,
            }
        finally:
            ks.close()

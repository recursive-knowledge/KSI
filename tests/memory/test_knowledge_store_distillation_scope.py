"""Tests for scoped distillation bundles and load_distillation API."""

import tempfile
from pathlib import Path

from ksi.memory.knowledge_store import (
    CROSS_TASK_SENTINEL,
    VALID_SOURCE_PHASES,
    KnowledgeStore,
)


def test_record_distillation_per_task_scope():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            rid = ks.record_distillation(
                task_id="t1",
                generation=1,
                bundle={
                    "transferable_insights": ["i1"],
                    "pitfalls": ["p1"],
                    "checks": ["c1"],
                    "evidence_post_ids": [42],
                },
                scope="per_task",
            )
            assert rid > 0
            loaded = ks.load_distillation(generation=1, task_id="t1", scope="per_task")
            assert loaded is not None
            assert loaded["transferable_insights"] == ["i1"]
            assert loaded["scope"] == "per_task"
        finally:
            ks.close()


def test_record_distillation_cross_task_uses_sentinel():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            ks.record_distillation(
                task_id=CROSS_TASK_SENTINEL,
                generation=1,
                bundle={
                    "transferable_insights": ["x1"],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                },
                scope="cross_task",
            )
            loaded = ks.load_distillation(
                generation=1,
                task_id=CROSS_TASK_SENTINEL,
                scope="cross_task",
            )
            assert loaded is not None
            assert loaded["scope"] == "cross_task"
        finally:
            ks.close()


def test_load_distillation_missing_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            assert ks.load_distillation(generation=99, task_id="nope", scope="per_task") is None
        finally:
            ks.close()


def test_load_distillations_batch_matches_singular():
    """Batched load returns per-task bundles identical to load_distillation."""
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            for tid in ("t1", "t2", "t3"):
                ks.record_distillation(
                    task_id=tid,
                    generation=1,
                    bundle={"transferable_insights": [tid], "evidence_post_ids": []},
                    scope="per_task",
                )
            got = ks.load_distillations_batch(
                generation=1,
                task_ids=["t1", "t2", "t3", "missing"],
                scope="per_task",
            )
            # Only present tasks appear (absent == None per the singular method).
            assert set(got) == {"t1", "t2", "t3"}
            for tid in ("t1", "t2", "t3"):
                singular = ks.load_distillation(generation=1, task_id=tid, scope="per_task")
                assert got[tid] == singular  # includes the _knowledge_id backref
        finally:
            ks.close()


def test_load_distillations_batch_picks_latest_and_scopes():
    """Batched load picks the newest row per task and honors generation/scope."""
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            ks.record_distillation(
                task_id="t1",
                generation=1,
                bundle={"transferable_insights": ["old"], "evidence_post_ids": []},
                scope="per_task",
            )
            ks.record_distillation(
                task_id="t1",
                generation=1,
                bundle={"transferable_insights": ["new"], "evidence_post_ids": []},
                scope="per_task",
            )
            got = ks.load_distillations_batch(generation=1, task_ids=["t1"], scope="per_task")
            assert got["t1"]["transferable_insights"] == ["new"]
            # Wrong generation / scope return nothing for this task.
            assert ks.load_distillations_batch(generation=2, task_ids=["t1"], scope="per_task") == {}
            assert ks.load_distillations_batch(generation=1, task_ids=["t1"], scope="cross_task") == {}
        finally:
            ks.close()


def test_load_distillations_batch_rejects_invalid_scope():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            import pytest as _pytest

            with _pytest.raises(ValueError, match="invalid scope"):
                ks.load_distillations_batch(generation=1, task_ids=["t1"], scope="bogus")
            # Empty id list short-circuits to an empty map without a query.
            assert ks.load_distillations_batch(generation=1, task_ids=[], scope="per_task") == {}
        finally:
            ks.close()


def test_new_source_phase_values_present():
    for v in (
        "per_task_forum",
        "cross_task_forum",
        "per_task_distill",
        "cross_task_distill",
    ):
        assert v in VALID_SOURCE_PHASES


def test_record_distillation_rejects_invalid_scope():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            import pytest as _pytest

            with _pytest.raises(ValueError, match="invalid scope"):
                ks.record_distillation(
                    task_id="t1",
                    generation=1,
                    bundle={},
                    scope="bogus",
                )
        finally:
            ks.close()


def test_record_distillation_rejects_both_forms():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            import pytest as _pytest

            with _pytest.raises(ValueError, match="not both"):
                ks.record_distillation(
                    task_id="t1",
                    generation=1,
                    assets=[{"x": 1}],
                    bundle={"transferable_insights": []},
                    scope="per_task",
                )
        finally:
            ks.close()


def test_record_distillation_rejects_empty_call():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            import pytest as _pytest

            with _pytest.raises(ValueError, match="must pass"):
                ks.record_distillation(task_id="t1", generation=1)
        finally:
            ks.close()


def test_record_distillation_uses_distiller_agent_id():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            ks.record_distillation(
                task_id="t1",
                generation=1,
                bundle={
                    "transferable_insights": ["x"],
                    "pitfalls": [],
                    "checks": [],
                    "evidence_post_ids": [],
                },
                scope="per_task",
            )
            row = (
                ks._connection()
                .execute("SELECT agent_id FROM knowledge WHERE entry_type='distillation' AND generation=1")
                .fetchone()
            )
            assert row[0] == "__distiller__"
        finally:
            ks.close()

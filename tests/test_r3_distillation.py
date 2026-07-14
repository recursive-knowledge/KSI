"""Tests for simplified R3 distillation (no clustering, no open_hypothesis)."""

from __future__ import annotations

import tempfile

from ksi.memory.knowledge_store import KnowledgeStore
from ksi.orchestrator import engine as eng


class TestRemovedMethods:
    """Verify that dead-code clustering methods have been deleted."""

    def test_cluster_forum_pages_not_callable(self):
        """_cluster_forum_pages should be removed from the engine module."""
        assert not hasattr(eng.GenerationalOrchestrator, "_cluster_forum_pages"), (
            "_cluster_forum_pages should be removed"
        )

    def test_apply_cluster_results_not_callable(self):
        """_apply_cluster_results should be removed from the engine module."""
        assert not hasattr(eng.GenerationalOrchestrator, "_apply_cluster_results"), (
            "_apply_cluster_results should be removed"
        )


class TestKnowledgeStoreDistillation:
    """Basic round-trip for distillation entries in KnowledgeStore."""

    def test_record_and_query_distillation(self):
        """KnowledgeStore can store and retrieve distillation entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(f"{tmpdir}/test.sqlite", default_experiment="test-exp")
            store.record_distillation(
                task_id="task-1",
                generation=1,
                assets=[
                    {
                        "asset_id": "asset-1",
                        "asset_type": "transferable_insight",
                        "text": "Always check edge cases",
                        "source_insight_ids": ["ins-1"],
                    }
                ],
                experiment="test-exp",
            )
            page = store.query_task("task-1", experiment="test-exp")
            distilled = page.get("distilled", [])
            assert len(distilled) >= 1
            # query_task unpacks assets into individual distilled entries
            assert distilled[0]["text"] == "Always check edge cases"
            assert distilled[0]["asset_type"] == "transferable_insight"

    def test_ablation_distillation_entries(self):
        """Ablated raw insights can be stored as distillation entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(f"{tmpdir}/test.sqlite", default_experiment="test-exp")
            raw_items = [
                {"insight_id": "ins-1", "text": "Insight A"},
                {"insight_id": "ins-2", "text": "Insight B"},
            ]
            for item in raw_items:
                store.record_distillation(
                    task_id=str(item.get("insight_id", "__distillation__")),
                    generation=1,
                    assets=[item],
                    experiment="test-exp",
                )
            # Both entries should be queryable
            page1 = store.query_task("ins-1", experiment="test-exp")
            assert len(page1.get("distilled", [])) >= 1
            page2 = store.query_task("ins-2", experiment="test-exp")
            assert len(page2.get("distilled", [])) >= 1

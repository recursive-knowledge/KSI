"""Tests for distilled knowledge handling in seed packages.

After Task 15, the legacy ``distilled_knowledge`` injection path in the engine
(which dumped legacy ``condensation`` assets into seed packages) is removed.
Per-task and cross-task bundles now flow through ``PopulationSeeder.seed`` via
``per_task_bundle`` and ``cross_task_bundle`` keys.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def knowledge_store(tmp_path):
    from kcsi.memory.knowledge_store import KnowledgeStore

    ks = KnowledgeStore(str(tmp_path / "k.sqlite"), default_experiment="test")
    yield ks
    ks.close()


class TestDistillationStorage:
    """KnowledgeStore-level distillation storage is still intact."""

    def test_gen2_gets_gen1_distillations(self, knowledge_store):
        """Generation 2 can query distillation entries from gen 1."""
        knowledge_store.record_distillation(
            task_id="task-1",
            generation=1,
            assets=[
                {
                    "asset_type": "transferable_insight",
                    "text": "Use connection pooling for Django DB",
                    "title": "Connection pooling",
                    "source_insight_ids": ["ins-1"],
                }
            ],
            experiment="test",
        )
        knowledge_store.record_distillation(
            task_id="task-2",
            generation=1,
            assets=[
                {
                    "asset_type": "pitfall",
                    "text": "Avoid raw SQL in migrations",
                    "title": "Migration safety",
                    "source_insight_ids": ["ins-2"],
                }
            ],
            experiment="test",
        )
        distillations = knowledge_store.query_generation(1, entry_types=["distillation"])
        assert len(distillations) == 2
        for d in distillations:
            content = d["content"]
            assert "assets" in content
            assets = content["assets"]
            assert len(assets) == 1
            assert "text" in assets[0]

    def test_gen1_has_no_distillations(self, knowledge_store):
        """Generation 1 should have no prior distillations (gen 0 doesn't exist)."""
        distillations = knowledge_store.query_generation(0, entry_types=["distillation"])
        assert len(distillations) == 0


class TestEnrichPathRemoved:
    """The engine's legacy `distilled_knowledge` seed-package injection is gone."""

    def test_enrich_path_removed_from_engine(self):
        """The legacy ``distilled_knowledge`` key is no longer injected in
        the seed-package enrichment path. It has been replaced by the
        per-task and cross-task bundles threaded through the seeder."""
        import inspect

        from kcsi.orchestrator import engine

        source = inspect.getsource(engine)
        # The injection line (seed_package["distilled_knowledge"] = ...) is
        # removed entirely.
        assert 'seed_package["distilled_knowledge"]' not in source


class TestRendererBackwardCompat:
    """The MEMORY.md renderer should still be resilient to pkgs without bundles."""

    def test_empty_seed_package_renders_cleanly(self):
        from kcsi.runtime.seeding import seed_package_to_memory_md

        md = seed_package_to_memory_md({"workstream_name": "solver"}, current_task_id="t1")
        # Must not crash; the renderer may include "None available" or simply
        # skip missing sections.
        assert isinstance(md, str)

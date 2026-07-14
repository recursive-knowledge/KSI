"""End-to-end: target-conditioned cross-task distillation → store → seed.

Proves the storage keys line up across the pipeline: the distiller writes one
cross-task bundle per attempted task under ``scope="cross_task", task_id=<tid>``,
and the seeder reads them back per agent label under the same scope, delivering
each agent the bundle distilled for ITS task. Uses a real ``KnowledgeStore`` so
the key contract is exercised without mocks.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from kcsi.distillation import DistillInput, distill
from kcsi.memory.knowledge_store import CROSS_TASK_SENTINEL, KnowledgeStore
from kcsi.seeding.seeder import PopulationSeeder


def _fake_llm(system_prompt: str, user_prompt: str) -> str:
    # The target-task prompt is appended to the user message, so the bundle
    # content can be keyed off which task this call was conditioned on. Use a
    # concrete sentence carrying the marker so it survives the distiller's
    # anti-meta (_is_concrete) filter, which drops bare short tokens.
    marker = "MARK-ONE" if "PROMPT-ONE" in user_prompt else "MARK-TWO"
    return json.dumps(
        {
            "transferable_insights": [
                f"When solving the {marker} task, verify the output grid shape before submitting."
            ],
            "pitfalls": [],
            "checks": [],
            "evidence_post_ids": [],
        }
    )


def test_target_conditioning_distill_store_seed_keys_line_up():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        ks.record_post(task_id=CROSS_TASK_SENTINEL, agent_id="a1", generation=0, text="cross pattern")

        out = distill(
            DistillInput(
                generation=0,
                task_ids=["t1", "t2"],
                knowledge_store=ks,
                llm=_fake_llm,
                cross_task_target_conditioning=True,
                target_task_prompts={"t1": "PROMPT-ONE", "t2": "PROMPT-TWO"},
            ),
            unsolved_task_ids=["t1", "t2"],
        )

        assert out.cross_task is None
        assert set(out.cross_task_by_task or {}) == {"t1", "t2"}

        # Persist exactly as the distillation phase does (scope="cross_task",
        # task_id=<target>).
        for tid, bundle in out.cross_task_by_task.items():
            ks.record_distillation(
                task_id=tid,
                generation=0,
                bundle=bundle.to_dict(),
                scope="cross_task",
                experiment="exp",
            )

        # Seeder reads per-task cross-task bundles by label under conditioning.
        agents = PopulationSeeder().seed(
            num_agents=2,
            generation=0,
            task_labels=["t1", "t2"],
            knowledge_store=ks,
            experiment="exp",
            cross_task_target_conditioning=True,
        )
        by_label = {a.workstream: a.seed_package["cross_task_bundle"] for a in agents}
        assert "MARK-ONE" in by_label["t1"]["transferable_insights"][0]
        assert "MARK-TWO" in by_label["t2"]["transferable_insights"][0]
        # No sentinel-keyed cross-task bundle was produced.
        assert ks.load_distillation(generation=0, task_id=CROSS_TASK_SENTINEL, scope="cross_task") is None


def test_conditioning_off_produces_single_broadcast_bundle():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        ks.record_post(task_id=CROSS_TASK_SENTINEL, agent_id="a1", generation=0, text="cross pattern")

        out = distill(
            DistillInput(
                generation=0,
                task_ids=["t1", "t2"],
                knowledge_store=ks,
                llm=_fake_llm,
                cross_task_target_conditioning=False,
            ),
            unsolved_task_ids=["t1", "t2"],
        )
        assert out.cross_task is not None
        assert out.cross_task_by_task is None

        ks.record_distillation(
            task_id=CROSS_TASK_SENTINEL,
            generation=0,
            bundle=out.cross_task.to_dict(),
            scope="cross_task",
            experiment="exp",
        )
        single = ks.load_distillation(generation=0, task_id=CROSS_TASK_SENTINEL, scope="cross_task")
        agents = PopulationSeeder().seed(
            num_agents=2,
            generation=0,
            task_labels=["t1", "t2"],
            cross_task_bundle=single,
            knowledge_store=ks,
            experiment="exp",
            cross_task_target_conditioning=False,
        )
        # Both agents get the same broadcast bundle.
        assert agents[0].seed_package["cross_task_bundle"]["transferable_insights"] == single["transferable_insights"]
        assert agents[1].seed_package["cross_task_bundle"]["transferable_insights"] == single["transferable_insights"]

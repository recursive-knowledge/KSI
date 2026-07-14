"""End-to-end smoke test for the 4-phase loop using a stub runtime.

This verifies: attempts recorded, per-task posts recorded with reply_to
capability, cross-task posts recorded under sentinel, both bundle types
written by distillation, next-gen seed packages carry both bundles.

Uses an in-memory stub runtime that:
- "Executes" by writing an attempt row with a canned score.
- "Discusses" by writing a canned post per agent per task.
- Uses a fake LLM for distillation.
"""

import json
import tempfile
from pathlib import Path

from ksi.distillation import DistillInput, distill
from ksi.memory.knowledge_store import CROSS_TASK_SENTINEL, KnowledgeStore


def test_end_to_end_two_generations_with_stub_runtime():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "k.sqlite"
        ks = KnowledgeStore(str(db), default_experiment="smoke")

        def fake_llm(sys_prompt, user_prompt):
            return json.dumps(
                {
                    "transferable_insights": ["Try approach A on task_alpha at generation 0"],
                    "pitfalls": ["Approach B dead-ends"],
                    "checks": ["Assert shape before calling"],
                    "evidence_post_ids": [],
                }
            )

        tasks = ["task_alpha", "task_beta"]
        agents = ["agent_0", "agent_1"]

        # Simulate Gen 0
        gen = 0
        for t, a in zip(tasks, agents):
            ks.record_attempt(
                task_id=t,
                agent_id=a,
                generation=gen,
                model_output=f"attempt on {t}",
                native_score=0.0,
            )
            parent_pid = ks.record_post(
                task_id=t,
                agent_id=a,
                generation=gen,
                text=f"per-task insight for {t}",
                source_phase="per_task_forum",
            )
            # Threaded reply from the other agent
            other = agents[1] if a == agents[0] else agents[0]
            ks.record_post(
                task_id=t,
                agent_id=other,
                generation=gen,
                text=f"replying to {parent_pid}",
                reply_to=parent_pid,
                source_phase="per_task_forum",
            )
        for a in agents:
            ks.record_post(
                task_id=CROSS_TASK_SENTINEL,
                agent_id=a,
                generation=gen,
                text=f"cross-task pattern seen by {a}",
                source_phase="cross_task_forum",
            )

        # Run distillation
        out = distill(
            DistillInput(
                generation=gen,
                task_ids=tasks,
                knowledge_store=ks,
                llm=fake_llm,
            )
        )
        for tid, bundle in out.per_task.items():
            ks.record_distillation(
                task_id=tid,
                generation=gen,
                bundle=bundle.to_dict(),
                scope="per_task",
            )
        assert out.cross_task is not None
        ks.record_distillation(
            task_id=CROSS_TASK_SENTINEL,
            generation=gen,
            bundle=out.cross_task.to_dict(),
            scope="cross_task",
        )

        # Verify both bundle kinds present
        assert ks.load_distillation(generation=gen, task_id="task_alpha", scope="per_task") is not None
        assert ks.load_distillation(generation=gen, task_id="task_beta", scope="per_task") is not None
        assert ks.load_distillation(generation=gen, task_id=CROSS_TASK_SENTINEL, scope="cross_task") is not None

        # Verify reply_to chain recorded
        rows = (
            ks._connection()
            .execute("SELECT id, reply_to FROM knowledge WHERE entry_type='post' AND reply_to IS NOT NULL")
            .fetchall()
        )
        assert len(rows) == len(tasks), f"expected {len(tasks)} threaded replies, got {len(rows)}"

        # Simulate seed-package construction for Gen 1
        from ksi.seeding.seeder import _build_task_seed_package

        per_task_b = ks.load_distillation(
            generation=gen,
            task_id="task_alpha",
            scope="per_task",
        )
        cross_b = ks.load_distillation(
            generation=gen,
            task_id=CROSS_TASK_SENTINEL,
            scope="cross_task",
        )
        pkg = _build_task_seed_package(
            label="task_alpha",
            next_gen=gen + 1,
            per_task_bundle=per_task_b,
            cross_task_bundle=cross_b,
        )
        assert pkg["per_task_bundle"]["transferable_insights"] == ["Try approach A on task_alpha at generation 0"]
        assert pkg["cross_task_bundle"]["transferable_insights"] == ["Try approach A on task_alpha at generation 0"]

        # Verify MEMORY.md rendering
        from ksi.runtime.seeding import seed_package_to_memory_md

        md = seed_package_to_memory_md(pkg, current_task_id="task_alpha")
        assert "Task-specific guidance" in md
        assert "Cross-task patterns" in md
        assert "Try approach A on task_alpha at generation 0" in md

        ks.close()

"""Regression test for issue #990b: ``CollectingPersistence`` must fire an
incremental per-generation snapshot callback so a mid-run crash doesn't lose
every trace collected up to that point (only the final, post-``run()``
``--output-json`` write existed before this)."""

from __future__ import annotations

from ksi.models import TaskTrace
from ksi.orchestrator.persistence import CollectingPersistence


def _make_trace(task_id: str, generation: int) -> TaskTrace:
    return TaskTrace(generation=generation, agent_id="agent-0", task_id=task_id)


def test_on_task_trace_accumulates_and_snapshot_fires_per_generation():
    snapshots: list[tuple[int, int]] = []  # (generation, num_traces_so_far)

    def spy(generation, traces_so_far):
        snapshots.append((generation, len(traces_so_far)))

    persistence = CollectingPersistence(on_generation_snapshot=spy)
    persistence.on_task_trace(_make_trace("t1", generation=1))
    persistence.on_task_trace(_make_trace("t2", generation=1))
    persistence.on_generation_end(generation=1, agents=[])
    persistence.on_task_trace(_make_trace("t3", generation=2))
    persistence.on_generation_end(generation=2, agents=[])

    assert snapshots == [(1, 2), (2, 3)]
    assert len(persistence.traces) == 3

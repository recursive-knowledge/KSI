"""Verify _enrich_seed_packages runs after carry-forward split (F3).

The key correctness invariant: enrichment should receive execute_map
(fresh tasks only), not assigned_map (which includes carried-forward
tasks). This avoids paying list_task_summaries + query_task costs for
tasks that will be skipped.
"""

from unittest.mock import MagicMock


def test_enrich_receives_execute_map_not_assigned_map():
    """The enrichment call must receive the post-split execute_map."""

    orch = MagicMock()
    assigned_map = {"a1": ["t1"], "a2": ["t2"]}
    execute_map = {"a1": ["t1"]}
    carried_traces = [MagicMock(agent_id="a2", task_id="t2")]

    orch._resume_phase.split_assignments.return_value = (execute_map, carried_traces)

    # Simulate the generation loop body: split first, then enrich.
    result_execute_map, result_carried = orch._resume_phase.split_assignments(
        generation=2,
        assigned_map=assigned_map,
        task_by_id={},
    )

    # Contract: enrichment must receive execute_map (fresh tasks only)
    assert result_execute_map == {"a1": ["t1"]}
    assert "a2" not in result_execute_map, (
        "_enrich_seed_packages should receive execute_map (fresh tasks only), "
        "not assigned_map which includes carried-forward tasks"
    )

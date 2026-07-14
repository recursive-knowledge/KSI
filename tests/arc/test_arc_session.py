"""Tests pinning the REMOVAL of the ARC text-output fallback (issue #944).

Canonical ARC-AGI scoring credits only a formally submitted grid, reconstructed
from the tool trace. The old lenient/strict text-output recovery paths (parsing
``model_output`` JSON, fenced blocks, trailing prose, ASCII grids, multi-candidate
"take the last", and the ``arc_set_output_grid`` recovery) have been removed.

These tests assert that the inputs those paths used to recover from now score 0:

* No tool trace at all -> infra-failure 0 (``scored_from_runtime_trials`` absent),
  regardless of how clean ``model_output`` is.
* A tool trace that sets a grid but never submits -> canonical 0
  (``scored_from_runtime_trials`` True): the agent ran but did not submit.
* A proper set_output_grid + submit trace still scores (regression guard).
"""

from __future__ import annotations

from ksi.benchmarks.arc_session import ArcSessionEvaluator
from ksi.models import TaskSpec


def _task_single(output_grid=None):
    return TaskSpec(
        id="arc-fallback-1",
        repo="",
        prompt="arc",
        metadata={
            "task_source": "arc",
            "arc_test_pairs": [{"input": [[0, 0], [0, 0]], "output": output_grid or [[1, 1], [1, 1]]}],
        },
    )


def _set_grid_entry(grid):
    """Build a tool_trace entry shaped like the one emitted by the runtime."""
    return {
        "type": "tool_call",
        "tool_name": "arc_set_output_grid",
        "tool_input": {"grid": grid},
        "tool_output": '{"status": "ok"}',
    }


def _submit_entry():
    return {
        "type": "tool_call",
        "tool_name": "arc_submit_trial",
        "tool_input": {},
        "tool_output": '{"status": "ok", "trial_count": 1, "trials_remaining": 1, "test_index": 0}',
    }


# ---------------------------------------------------------------------------
# No tool trace -> infra-failure 0, no matter how clean model_output is.
# The old paths would have recovered each of these to a 1.0.
# ---------------------------------------------------------------------------


def _assert_infra_zero(result):
    assert result["status"] == "no_runtime_submission"
    assert result["resolved"] is False
    assert result["native_score"] == 0.0
    assert "scored_from_runtime_trials" not in result


def test_direct_json_model_output_no_longer_recovered():
    evaluator = ArcSessionEvaluator()
    result = evaluator.evaluate(task=_task_single([[1, 1], [1, 1]]), model_output="[[1,1],[1,1]]")
    _assert_infra_zero(result)


def test_fenced_json_model_output_no_longer_recovered():
    evaluator = ArcSessionEvaluator()
    fenced = "Here is my final answer:\n```json\n[[1, 1], [1, 1]]\n```"
    result = evaluator.evaluate(task=_task_single([[1, 1], [1, 1]]), model_output=fenced)
    _assert_infra_zero(result)


def test_trailing_prose_model_output_no_longer_recovered():
    evaluator = ArcSessionEvaluator()
    output = "[[1, 1], [1, 1]] — this matches the pattern I inferred from train."
    result = evaluator.evaluate(task=_task_single([[1, 1], [1, 1]]), model_output=output)
    _assert_infra_zero(result)


def test_ascii_grid_model_output_no_longer_recovered():
    evaluator = ArcSessionEvaluator()
    output = "I examined the inputs carefully and decided on my final answer.\n\n1 1\n1 1\n"
    result = evaluator.evaluate(task=_task_single([[1, 1], [1, 1]]), model_output=output)
    _assert_infra_zero(result)


def test_multi_candidate_model_output_no_longer_recovered():
    evaluator = ArcSessionEvaluator()
    output = "[[[9, 9], [9, 9]], [[5, 5], [5, 5]], [[1, 1], [1, 1]]]"
    result = evaluator.evaluate(task=_task_single([[1, 1], [1, 1]]), model_output=output)
    _assert_infra_zero(result)


def test_empty_model_output_no_longer_recovered():
    evaluator = ArcSessionEvaluator()
    result = evaluator.evaluate(task=_task_single(), model_output="")
    _assert_infra_zero(result)


def test_pure_prose_model_output_no_longer_recovered():
    evaluator = ArcSessionEvaluator()
    result = evaluator.evaluate(task=_task_single(), model_output="I could not solve this puzzle.")
    _assert_infra_zero(result)


# ---------------------------------------------------------------------------
# set_output_grid WITHOUT a submit was the old "empty-output recovery" path.
# It now gets a distinct "no_submission" status (agent ran, never submitted)
# — NOT a recovery, and NOT the same status as a genuine wrong submission
# ("no_submission" is still scored as a failed attempt by score_arc_from_eval).
# ---------------------------------------------------------------------------


def test_set_output_grid_without_submit_scores_canonical_zero():
    evaluator = ArcSessionEvaluator()
    task = _task_single([[1, 1], [1, 1]])
    tool_trace = [
        _set_grid_entry([[9, 9], [9, 9]]),
        _set_grid_entry([[1, 1], [1, 1]]),  # correct grid, but never submitted
    ]
    result = evaluator.evaluate(task=task, model_output="", tool_trace=tool_trace)
    assert result["status"] == "no_submission"
    assert result["resolved"] is False
    assert result["native_score"] == 0.0
    assert result["scored_from_runtime_trials"] is True


def test_set_output_grid_without_submit_via_runtime_meta_trace():
    evaluator = ArcSessionEvaluator()
    task = _task_single([[1, 1], [1, 1]])
    result = evaluator.evaluate(
        task=task,
        model_output="",
        runtime_meta={"tool_trace": [_set_grid_entry([[1, 1], [1, 1]])]},
    )
    assert result["status"] == "no_submission"
    assert result["resolved"] is False
    assert result["native_score"] == 0.0
    assert result["scored_from_runtime_trials"] is True


# ---------------------------------------------------------------------------
# Regression: a proper set_output_grid + submit trace still scores correctly.
# ---------------------------------------------------------------------------


def test_canonical_submit_trace_still_scores():
    evaluator = ArcSessionEvaluator()
    task = _task_single([[1, 1], [1, 1]])
    tool_trace = [_set_grid_entry([[1, 1], [1, 1]]), _submit_entry()]
    result = evaluator.evaluate(task=task, model_output="", tool_trace=tool_trace)
    assert result["status"] == "ok"
    assert result["resolved"] is True
    assert result["native_score"] == 1.0
    assert result["scored_from_runtime_trials"] is True


def test_missing_reference_still_reported():
    evaluator = ArcSessionEvaluator()
    task = TaskSpec(id="arc-x", repo="", prompt="arc", metadata={"task_source": "arc"})
    result = evaluator.evaluate(task=task, model_output="[[1]]")
    assert result["status"] == "missing_reference"
    assert result["scored_from_runtime_trials"] is False

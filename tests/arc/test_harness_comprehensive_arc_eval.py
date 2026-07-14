"""Comprehensive tests for ARC Session Evaluator.

Covers reference-validation paths of ArcSessionEvaluator.evaluate end-to-end and
normalize_grid edge cases. Canonical exact-match scoring (trace reconstruction)
is exercised in tests/arc/test_arc_session_evaluator.py; the removed text-output
fallback contract is pinned in tests/arc/test_arc_session.py (issue #944).
"""

from __future__ import annotations

import pytest

from kcsi.benchmarks.arc_session import ArcSessionEvaluator
from kcsi.memory.arc_semantics import normalize_grid
from kcsi.models import TaskSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_with_tests(test_pairs, *, meta_key: str = "arc_test_pairs") -> TaskSpec:
    return TaskSpec(
        id="arc-test-task",
        repo="",
        prompt="arc",
        metadata={
            "task_source": "arc",
            meta_key: test_pairs,
        },
    )


# ===================================================================
# Reference-validation paths (run before any scoring).
# ===================================================================


class TestArcSessionEvaluatorEvaluate:
    """End-to-end evaluator reference-validation tests."""

    def setup_method(self):
        self.evaluator = ArcSessionEvaluator()

    def test_missing_test_pairs_in_metadata(self):
        task = TaskSpec(id="t1", metadata={})
        result = self.evaluator.evaluate(task=task, model_output="[[1]]")
        assert result["status"] == "missing_reference"
        assert result["resolved"] is False
        assert result["native_score"] == 0.0

    def test_empty_test_pairs_list(self):
        task = _task_with_tests([])
        result = self.evaluator.evaluate(task=task, model_output="[[1]]")
        assert result["status"] == "missing_reference"
        assert result["resolved"] is False

    def test_none_metadata(self):
        task = TaskSpec(id="t2", metadata=None)
        result = self.evaluator.evaluate(task=task, model_output="[[1]]")
        assert result["status"] == "missing_reference"

    def test_test_pair_without_output_key(self):
        task = _task_with_tests([{"input": [[0]]}])
        result = self.evaluator.evaluate(task=task, model_output="[[1]]")
        assert result["status"] == "missing_reference_output"
        assert result["resolved"] is False

    def test_invalid_reference_output_non_rectangular(self):
        task = _task_with_tests([{"input": [[0]], "output": [[1, 2], [3]]}])
        result = self.evaluator.evaluate(task=task, model_output="[[1,2],[3,4]]")
        assert result["status"] == "invalid_reference_output"
        assert result["resolved"] is False

    def test_non_dict_test_pair_element(self):
        task = _task_with_tests(["not_a_dict"])
        result = self.evaluator.evaluate(task=task, model_output="[[1]]")
        assert result["status"] == "missing_reference_output"


# ===================================================================
# normalize_grid edge cases
# ===================================================================


class TestNormalizeGrid:
    """Edge cases for grid normalization."""

    def test_empty_outer_list(self):
        with pytest.raises(ValueError, match="non-empty"):
            normalize_grid([])

    def test_inner_list_with_empty_row(self):
        with pytest.raises(ValueError, match="non-empty"):
            normalize_grid([[1, 2], []])

    def test_non_rectangular_grid(self):
        with pytest.raises(ValueError, match="rectangular"):
            normalize_grid([[1, 2], [3]])

    def test_negative_cell_values(self):
        with pytest.raises(ValueError, match="0-9"):
            normalize_grid([[-1, 0]])

    def test_float_cells_whole_numbers(self):
        result = normalize_grid([[1.0, 2.0], [3.0, 4.0]])
        assert result == [[1, 2], [3, 4]]
        # Verify they are actually ints
        assert all(isinstance(cell, int) for row in result for cell in row)

    def test_float_cells_not_whole_numbers(self):
        with pytest.raises(ValueError, match="integers"):
            normalize_grid([[1.5, 2.0]])

    def test_string_cell_values(self):
        with pytest.raises(ValueError, match="integers"):
            normalize_grid([["a", "b"]])

    def test_very_large_grid_30x30(self):
        grid = [[(i + j) % 10 for j in range(30)] for i in range(30)]
        result = normalize_grid(grid)
        assert len(result) == 30
        assert len(result[0]) == 30
        assert result[0][0] == 0
        assert result[29][29] == (29 + 29) % 10

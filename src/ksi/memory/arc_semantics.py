"""Deterministic ARC grid semantics used by the native ARC scorer.

Provides grid normalization and exact-match comparison. This mirrors core
behavior from dataset/ARC-AGI/apps/js/testing_interface.js without UI concerns
(DOM, rendering events). ``normalize_grid`` and ``compare_grids`` are imported
by ``ksi.benchmarks.arc_session`` (the native attempt-file scorer).
"""

from __future__ import annotations

from typing import Any


def normalize_grid(value: Any) -> list[list[int]]:
    if not isinstance(value, list) or not value:
        raise ValueError("grid must be a non-empty 2D array")
    out: list[list[int]] = []
    width: int | None = None
    for row in value:
        if not isinstance(row, list) or not row:
            raise ValueError("grid rows must be non-empty arrays")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("grid must be rectangular")
        out_row: list[int] = []
        for cell in row:
            if isinstance(cell, bool):
                raise ValueError("grid cells must be integers, not booleans")
            if isinstance(cell, float) and cell == int(cell):
                cell = int(cell)
            elif not isinstance(cell, int):
                raise ValueError("grid cells must be integers")
            if cell < 0 or cell > 9:
                raise ValueError("grid cells must be integers in 0-9")
            out_row.append(cell)
        out.append(out_row)
    return out


def _grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def compare_grids(expected: list[list[int]], submitted: list[list[int]]) -> tuple[bool, dict[str, Any]]:
    exp_h, exp_w = _grid_shape(expected)
    sub_h, sub_w = _grid_shape(submitted)
    if exp_h != sub_h or exp_w != sub_w:
        return False, {
            "reason": "shape_mismatch",
            "expected_shape": [exp_h, exp_w],
            "submitted_shape": [sub_h, sub_w],
        }
    for i in range(exp_h):
        for j in range(exp_w):
            if expected[i][j] != submitted[i][j]:
                return False, {
                    "reason": "cell_mismatch",
                    "first_mismatch": {
                        "row": i,
                        "col": j,
                        "expected": expected[i][j],
                        "submitted": submitted[i][j],
                    },
                }
    return True, {"reason": "exact_match"}

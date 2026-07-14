from __future__ import annotations

import pytest

from kcsi.memory.arc_semantics import (
    compare_grids,
    normalize_grid,
)


def test_normalize_grid_rejects_booleans():
    """bool is a subclass of int in Python; without an explicit guard,
    `True`/`False` would be silently coerced to 1/0 and scored as valid
    model output. Reject them so a broken model surface is loud, not silent."""
    with pytest.raises(ValueError, match="booleans"):
        normalize_grid([[True, False]])
    with pytest.raises(ValueError, match="booleans"):
        normalize_grid([[1, 2], [3, True]])


def test_normalize_grid_accepts_floats_that_are_whole():
    """Floats equal to their int cast (e.g. 1.0) are coerced — this matches
    the existing contract for models that emit `1.0` instead of `1`."""
    assert normalize_grid([[1.0, 2.0], [3.0, 4.0]]) == [[1, 2], [3, 4]]


def test_normalize_grid_rejects_out_of_range():
    """ARC symbols are integers in 0-9. Values outside this range must be rejected
    to stay consistent with benchmarks/scripts/arc_prep/_common.py::validate_grid."""
    with pytest.raises(ValueError, match="0-9"):
        normalize_grid([[0, 1, 10]])
    with pytest.raises(ValueError, match="0-9"):
        normalize_grid([[-1]])


def test_compare_grids_shape_and_value():
    ok, detail = compare_grids([[1, 2]], [[1, 2]])
    assert ok is True
    assert detail["reason"] == "exact_match"

    ok, detail = compare_grids([[1, 2]], [[1]])
    assert ok is False
    assert detail["reason"] == "shape_mismatch"

    ok, detail = compare_grids([[1, 2]], [[1, 9]])
    assert ok is False
    assert detail["reason"] == "cell_mismatch"

"""Regression tests for ARC TASK.md inline-grid restoration (post-0610767d fix).

Commit 0610767d removed inline train/test grids from the ARC TASK.md, forcing
the model to burn budget before it could even try to solve. These tests lock in
the restored behavior: the full train/test grids are embedded inline in the
native (attempt-file) ARC TASK.md.
"""

from __future__ import annotations

import glob
import json
import os

import pytest

from kcsi.models import TaskSpec
from kcsi.prompts import build_task_markdown

_ARC2_EVAL_DIR = "benchmarks/arc2/source/data/evaluation"


def _first_real_arc_task() -> TaskSpec:
    """Load one ARC task from the on-disk evaluation set. Skip if unavailable."""
    pattern = os.path.join(_ARC2_EVAL_DIR, "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        pytest.skip(f"No ARC2 evaluation tasks at {_ARC2_EVAL_DIR}")
    path = files[0]
    with open(path) as f:
        data = json.load(f)
    train_pairs = data.get("train") or []
    test_pairs = data.get("test") or []
    test_inputs = [{"input": p.get("input")} for p in test_pairs]
    return TaskSpec(
        id=os.path.splitext(os.path.basename(path))[0],
        repo="",
        prompt="Infer the transformation rule from train pairs and solve the test grid.",
        metadata={
            "task_source": "arc",
            "arc_split": "evaluation",
            "arc_train_pairs": train_pairs,
            "arc_test_inputs": test_inputs,
            "arc_eval_test_pairs": test_pairs,
            "arc_max_trials": 2,
        },
    )


def _tiny_arc_task() -> TaskSpec:
    """A minimal in-memory ARC task, independent of the dataset being present."""
    return TaskSpec(
        id="arc_tiny",
        repo="",
        prompt="",
        metadata={
            "task_source": "arc",
            "arc_split": "training",
            "arc_train_pairs": [
                {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
                {"input": [[5]], "output": [[6]]},
            ],
            "arc_test_inputs": [{"input": [[7, 8]]}],
            "arc_max_trials": 2,
        },
    )


def test_arc_task_md_contains_inline_training_grids():
    task = _first_real_arc_task()
    md = build_task_markdown(task)

    # The raw template placeholder must not leak through.
    assert "{training_data}" not in md
    assert "{test_data}" not in md

    # JSON-encoded grids mean we should see "input":" substrings at least 2x
    # (one per train pair minimum for a typical task; real ARC tasks have 3+).
    assert md.count('"input":') >= 2, (
        f"Expected inline JSON grids (>=2 'input':' substrings); got {md.count(chr(34) + 'input' + chr(34) + ':')}"
    )

    # Typical ARC tasks (even small ones) inflate well beyond 5000 chars when
    # grids are embedded; the regressed (HEAD-at-0610767d) version was ~900.
    assert len(md) > 5000, f"Expected TASK.md > 5000 chars with inline grids; got {len(md)}"


def test_arc_task_md_inline_grids_are_compact():
    """Inline grids must be compact JSON (no indent=2 whitespace) to keep the
    cached ARC prefix small — issue #1252 item 2. Pretty-printing the grids
    wasted ~78% of the block on whitespace, re-paid on every cached-read turn."""
    md = build_task_markdown(_tiny_arc_task())

    # Compact JSON has no space after the key colon and no newline-indented
    # grid rows; the pretty form (indent=2) produced `"input": [` followed by
    # deeply-indented rows.
    assert '"input":[[' in md, "Expected compact JSON grids (no space after colon)"
    assert '"input": [' not in md, "Inline grids must not be pretty-printed (indent=2)"
    # No run of indentation spaces from indent=2 nesting inside the grid block.
    assert "\n      " not in md, "Inline grids must not contain indent=2 whitespace"


def test_arc_task_md_handles_empty_test_output_correctly():
    """Test-input block must render with the pre-0610767d `"output": [[]]` shape."""
    md = build_task_markdown(_tiny_arc_task())
    # The placeholder shape [[]] should appear in the rendered test block
    # (empty output list, matching the pre-regression template).
    assert '"output": [\n      []\n    ]' in md or '"output":[[]]' in md or "[[]]" in md, (
        "Expected empty-grid placeholder ([[]]) in rendered test input block"
    )


def test_arc_task_md_metadata_section_preserved():
    md = build_task_markdown(_tiny_arc_task())
    assert "train_examples: 2" in md
    assert "test_inputs: 1" in md

"""Regression tests for prominent multi-test guidance in the native ARC
TASK.md / execution prompt.

Context (2026-04-21): Haiku arc2-eval telemetry showed 48% of multi-test
attempts never advanced to the next test input. These tests lock in the
conditional multi-test section (native attempt-file variant) so per-test
answers are named explicitly.
"""

from __future__ import annotations

from ksi.models import TaskSpec
from ksi.prompts import (
    _build_arc_no_mcp_execution_prompt,
    _build_arc_no_mcp_task_markdown,
    build_execution_prompt,
    build_task_markdown,
)


def _arc_task(test_inputs: list[list[list[int]]]) -> TaskSpec:
    return TaskSpec(
        id="arc_multi_test_demo",
        repo="",
        prompt="",
        metadata={
            "task_source": "arc",
            "arc_split": "evaluation",
            "arc_max_trials": 2,
            "arc_train_pairs": [{"input": [[1]], "output": [[2]]}],
            "arc_test_inputs": [{"input": inp} for inp in test_inputs],
        },
    )


def test_single_test_task_omits_multi_test_section():
    md = build_task_markdown(_arc_task([[[1]]]))
    assert "## Multi-Test Tasks" not in md
    assert "YOU HAVE" not in md


def test_multi_test_dispatcher_announces_count_and_upper_bound():
    """The public ``build_task_markdown`` dispatcher routes an ARC task to the
    native multi-test TASK.md, which announces the per-test count and the
    0..N-1 loop bound."""
    md = build_task_markdown(_arc_task([[[1]], [[2]]]))
    assert "2 test inputs" in md
    # Flow loop announces the right upper bound (0..N-1).
    assert "0..1" in md


def test_multi_test_task_md_token_budget_bounded():
    """The multi-test section must not runaway — inline grids dominate, but our
    additions should stay bounded (~300 tokens) over the single-test baseline.
    Bound raised to 1200 chars after #593 mandated two trials per test input."""
    single = build_task_markdown(_arc_task([[[1]]]))
    multi = build_task_markdown(_arc_task([[[1]], [[2]]]))
    delta = len(multi) - len(single)
    assert 0 < delta < 1200, f"Multi-test section delta={delta} chars out of expected (0, 1200) bound"


def test_arc_execution_prompt_has_explicit_multi_test_step():
    prompt = build_execution_prompt(_arc_task([[[1]], [[2]]]), has_memory=False, generation=1)
    assert "Multi-test task (2 test inputs)" in prompt
    assert "attempt_i_1.txt" in prompt


# ---------------------------------------------------------------------------
# --arc-no-mcp (native default) multi-test TASK.md / execution prompt (#694)
# ---------------------------------------------------------------------------


def test_no_mcp_single_test_task_md_uses_legacy_files_only():
    md = _build_arc_no_mcp_task_markdown(_arc_task([[[1]]]))
    # No per-test file naming, no multi-test announcement.
    assert "attempt_0_1.txt" not in md
    assert "test inputs. You must produce a" not in md
    # Legacy single-test wording preserved verbatim.
    assert "Overwrite `attempt_1.txt` and `attempt_2.txt`" in md


def test_no_mcp_multi_test_task_md_names_per_test_files():
    md = _build_arc_no_mcp_task_markdown(_arc_task([[[1]], [[2]]]))
    # Per-test files are enumerated for both tests, both trials.
    for fname in (
        "attempt_0_1.txt",
        "attempt_0_2.txt",
        "attempt_1_1.txt",
        "attempt_1_2.txt",
    ):
        assert fname in md, f"expected {fname} named in multi-test no-MCP TASK.md"
    # Multi-test section announces the per-test count.
    assert "2 test inputs" in md
    # Per-test workflow instructs writing both per-test files for each test i.
    assert "attempt_i_1.txt" in md
    assert "attempt_i_2.txt" in md


def test_no_mcp_task_md_includes_validation_discipline_single_and_multi():
    """#690 follow-up: bake the train-pair validation discipline into the
    scaffold (single- AND multi-test) instead of re-distilling it every gen."""
    for test_inputs in ([[[1]]], [[[1]], [[2]]]):
        md = _build_arc_no_mcp_task_markdown(_arc_task(test_inputs))
        assert "## Validation discipline" in md
        assert "every training pair" in md


def test_no_mcp_execution_prompt_appends_multi_test_paragraph():
    single = _build_arc_no_mcp_execution_prompt(has_memory=False, generation=1, test_count=1)
    multi = _build_arc_no_mcp_execution_prompt(has_memory=False, generation=1, test_count=2)
    assert "Multi-test task" not in single
    assert "Multi-test task (2 test inputs)" in multi
    assert "attempt_i_1.txt" in multi

from __future__ import annotations

from kcsi.models import TaskSpec
from kcsi.prompts import build_execution_prompt, build_task_markdown


def _arc_task() -> TaskSpec:
    return TaskSpec(
        id="arc_demo",
        repo="",
        prompt="",
        metadata={
            "task_source": "arc",
            "arc_split": "training",
            "arc_max_trials": 2,
            "arc_train_pairs": [{"input": [[1]], "output": [[2]]}],
            "arc_test_pairs": [{"input": [[1]], "output": [[2]]}],
        },
    )


def test_arc_execution_prompt_dispatches_to_native_attempt_files():
    """The legacy ARC MCP toolset is removed: ``build_execution_prompt`` for an
    ARC task must route to the native attempt-file prompt (Read/Edit/Write of
    attempt_1.txt / attempt_2.txt), not the deleted MCP builder."""
    prompt = build_execution_prompt(_arc_task(), has_memory=True, generation=1)
    assert "attempt_1.txt" in prompt
    assert "attempt_2.txt" in prompt
    assert "ASCII" in prompt
    assert "Do NOT write JSON" in prompt
    # No MCP tool-call instructions survive.
    assert "arc_load_task" not in prompt
    assert "arc_submit_trial" not in prompt


def test_arc_execution_prompt_does_not_promise_correctness_feedback():
    """The native path scores blind (host-side, from attempt files). The prompt
    must not instruct the agent to check a ``correct`` field or iterate on
    feedback."""
    prompt = build_execution_prompt(_arc_task(), has_memory=True, generation=1)
    assert "check `correct`" not in prompt
    assert "check correct" not in prompt.lower().replace("`", "")


def test_arc_task_markdown_dispatches_to_native():
    """``build_task_markdown`` for an ARC task must produce the native TASK.md
    (attempt-file submission, inlined grids), not the deleted MCP TASK.md."""
    md = build_task_markdown(_arc_task())
    assert "attempt_1.txt" in md
    assert "## Training Examples and Test Input" in md
    # Native output does not instruct JSON submission or MCP tools.
    assert "Return JSON only" not in md
    assert "arc_load_task" not in md


def test_arc_task_markdown_frames_inline_grids_as_primary_inspection_surface():
    """Inline grids belong in the cached prompt prefix and are the model's
    primary view of the task data."""
    md = build_task_markdown(_arc_task()).lower()
    assert "## training examples and test input" in md
    assert "primary inspection surface" in md

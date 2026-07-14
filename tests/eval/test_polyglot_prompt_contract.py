from __future__ import annotations

from kcsi.models import TaskSpec
from kcsi.prompts import build_execution_prompt, build_task_markdown


def test_polyglot_prompt_contract_surfaces_verification_command() -> None:
    task = TaskSpec(
        id="python__dot-dsl",
        repo="",
        prompt="Implement the dot-dsl exercise in python.",
        metadata={
            "task_source": "polyglot",
            "language": "python",
            "exercise_name": "dot-dsl",
            "starter_code": {"dot_dsl.py": "class Graph: ..."},
            "test_files": {"dot_dsl_test.py": "def test_empty_graph(): ..."},
            "test_command": "python -m pytest -rA --tb=long",
        },
    )

    prompt = build_execution_prompt(task, has_memory=False, generation=1)
    md = build_task_markdown(task)

    assert "Do not expect hidden benchmark tests to be present" in prompt
    assert "run the verification command from `TASK.md`" in prompt
    assert "write and run a small smoke check" in prompt
    assert "do not create, stub, or guess those test files" in prompt
    assert "real compile/type/signature failure" in prompt
    assert "## Required Verification" in md
    assert "python -m pytest -rA --tb=long" in md
    assert "Do not use ad hoc snippets in place of this command" in md
    assert "NOT seeded into your workspace" in md
    assert "write and run your own ad hoc smoke check" in md
    assert "dot_dsl_test.py" not in md

from __future__ import annotations

from ksi.models import TaskSpec
from ksi.prompts import build_execution_prompt, build_task_markdown


def test_terminal_bench_2_prompt_contract_uses_native_files() -> None:
    task = TaskSpec(
        id="git-multibranch",
        repo="",
        prompt="Solve the native Terminal-Bench 2 task described in instruction.md.",
        metadata={
            "task_source": "terminal_bench_2",
            "category": "system-administration",
            "difficulty": "medium",
            "agent_timeout_sec": 900.0,
            "verifier_timeout_sec": 900.0,
            "task_files": {
                "tb2/instruction.md": "Native instruction.\n",
                "tb2/task.toml": 'version = "1.0"\n',
            },
        },
    )

    prompt = build_execution_prompt(task, has_memory=False, generation=1)
    md = build_task_markdown(task)

    assert "Read the native `tb2/instruction.md` first" in prompt
    assert "No prior memory is provided for this run." in prompt
    assert "Read `tb2/task.toml` for task metadata" in prompt
    assert "Treat the mounted KSI workspace as task specification only" in prompt
    assert "identify the real task surface quickly" in prompt
    assert "Do not spend many steps re-reading the overlay" in prompt
    assert "Prefer short mutate-then-check cycles" in prompt
    assert "Avoid giant here-doc scripts" in prompt
    assert "If `MEMORY.md` names a concrete failure" in prompt
    assert "make that fix persistent so a fresh shell and the verifier also pass" in prompt
    assert "verify it from a fresh command after reload/restart" in prompt
    assert "exercise the concrete artifact, service, build output, or runtime behavior" in prompt
    assert "TASK.md" not in prompt
    assert "Do not rely on `solution/` or hidden verifier assets" in prompt
    assert "Leave a concise final summary" in prompt
    assert md == ""

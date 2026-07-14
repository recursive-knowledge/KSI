from __future__ import annotations

import pytest

from kcsi.models import TaskSpec
from kcsi.prompts import build_execution_prompt, build_task_markdown
from kcsi.runtime.seeding import workspace_task_files


def test_swebench_pro_prompt_contract_matches_patch_flow() -> None:
    task = TaskSpec(
        id="instance_demo__repo-123",
        repo="demo/repo",
        prompt="Fix the regression in widget normalization.",
        metadata={
            "task_source": "swebench_pro",
            "instance_id": "instance_demo__repo-123",
            "base_commit": "abc123",
        },
    )

    prompt = build_execution_prompt(task, has_memory=False, generation=1)
    md = build_task_markdown(task)

    # The execution prompt no longer mandates a `<patch>` block. The runtime
    # captures the workspace diff as the canonical patch; instructing the
    # agent to also hand-write a `<patch>` produces stale-diff scoring.
    assert "<patch>...</patch>" not in prompt
    assert "Patch format (required)" not in prompt
    assert "Edit files in place" in prompt
    assert "runtime captures the workspace diff" in prompt
    assert "short checklist of the required behavior changes" in prompt
    assert "Read the verification section in `TASK.md`" in prompt
    assert "identify the exact assertion or behavior it is checking" in prompt
    assert "code path that must change and the interfaces it affects" in prompt
    assert "Stop broad searching once the likely fix path is clear" in prompt
    assert "Use the regression targets and selected tests from `TASK.md`" in prompt
    assert "If `TASK.md` lists named failing tests without a runnable command" in prompt
    assert "use the first failing assertion or error" in prompt
    assert "Read `MEMORY.md`" not in prompt
    assert "No prior memory is provided" in prompt
    assert "mcp__arc__arc_load_task()" not in prompt
    assert "- task_source: swebench_pro" in md
    assert "- instance_id: instance_demo__repo-123" in md


def test_swebench_task_markdown_upstream_strict_omits_test_names() -> None:
    """In upstream-strict mode (default, swebench_pro_seed_tests absent/False),
    test names must NOT appear in the TASK.md — only anonymized counts."""
    task = TaskSpec(
        id="instance_demo__repo-456",
        repo="demo/repo",
        prompt="Fix the regression in widget normalization.",
        metadata={
            "task_source": "swebench_pro",
            "instance_id": "instance_demo__repo-456",
            "selected_test_files_to_run": '["tests/test_widget.py", "tests/test_api.py"]',
            "fail_to_pass": "['tests/test_widget.py::test_fix_regression']",
            "pass_to_pass": '["tests/test_widget.py::test_old_case", "tests/test_api.py::test_smoke"]',
        },
    )

    md = build_task_markdown(task)

    assert "## Verification" in md
    # Test names must NOT appear in upstream-strict mode
    assert "tests/test_widget.py::test_fix_regression" not in md
    assert "tests/test_widget.py::test_old_case" not in md
    assert "tests/test_api.py::test_smoke" not in md
    assert "tests/test_widget.py" not in md
    assert "tests/test_api.py" not in md
    assert "Inspect these exact failing tests" not in md
    assert "Selected test files or suites to inspect first" not in md
    # Count-based guidance should still be present
    assert "1 target test(s) that must change from failing to passing" in md
    assert "2 benchmark checks" in md
    assert "Benchmark test script command" in md
    assert "After each targeted run, use the first failing assertion or error to decide the next edit." in md
    assert "runtime captures the workspace diff as the canonical patch" in md
    assert "Do not hand-write a `<patch>` block" in md


def test_swebench_task_markdown_seeded_mode_surfaces_test_names() -> None:
    """In seeded mode (swebench_pro_seed_tests=True), test names ARE surfaced."""
    task = TaskSpec(
        id="instance_demo__repo-456s",
        repo="demo/repo",
        prompt="Fix the regression in widget normalization.",
        metadata={
            "task_source": "swebench_pro",
            "instance_id": "instance_demo__repo-456s",
            "selected_test_files_to_run": '["tests/test_widget.py", "tests/test_api.py"]',
            "fail_to_pass": "['tests/test_widget.py::test_fix_regression']",
            "pass_to_pass": '["tests/test_widget.py::test_old_case", "tests/test_api.py::test_smoke"]',
            "swebench_pro_seed_tests": True,
        },
    )

    md = build_task_markdown(task)

    assert "## Verification" in md
    assert "Inspect these exact failing tests before editing" in md
    assert "`tests/test_widget.py::test_fix_regression`" in md
    assert "Selected test files or suites to inspect first" in md
    assert "`tests/test_widget.py`" in md
    assert (
        "cd '/workspace/task/workspace/repo' && bash /workspace/task/workspace/run_script.sh "
        "'tests/test_widget.py,tests/test_api.py'"
    ) in md
    assert "After each targeted run, use the first failing assertion or error to decide the next edit." in md
    assert "Preserve existing passing behavior covered by 2 benchmark checks." in md
    assert "`tests/test_widget.py::test_old_case`" in md
    assert "runtime captures the workspace diff as the canonical patch" in md
    assert "Do not hand-write a `<patch>` block" in md


def test_swebench_task_markdown_uses_official_repo_container_path() -> None:
    task = TaskSpec(
        id="instance_demo__repo-457",
        repo="demo/repo",
        prompt="Fix the regression.",
        metadata={
            "task_source": "swebench_pro",
            "instance_id": "instance_demo__repo-457",
            "official_repo_container_path": "/app",
            "selected_test_files_to_run": ["tests/test_widget.py"],
            # seeded mode so selected_tests arg is included in the command
            "swebench_pro_seed_tests": True,
        },
    )

    md = build_task_markdown(task)

    assert "cd '/app' && bash /workspace/task/workspace/run_script.sh 'tests/test_widget.py'" in md


def test_swebench_workspace_task_files_upstream_strict_omits_run_script(tmp_path, monkeypatch) -> None:
    """In upstream-strict mode, run_script.sh must NOT be copied to agent workspace."""
    instance_id = "instance_demo__repo-789"
    instance_dir = tmp_path / "scripts" / instance_id
    instance_dir.mkdir(parents=True)
    (instance_dir / "run_script.sh").write_text("#!/bin/bash\necho selected:$@\n", encoding="utf-8")
    (instance_dir / "parser.py").write_text("print('parse')\n", encoding="utf-8")
    monkeypatch.setenv("SWEBENCH_PRO_SCRIPTS_DIR", str(tmp_path / "scripts"))
    task = TaskSpec(
        id=instance_id,
        repo="demo/repo",
        prompt="Fix the issue.",
        metadata={
            "task_source": "swebench_pro",
            "instance_id": instance_id,
            # No swebench_pro_seed_tests → upstream-strict, no run_script.sh
        },
    )

    files = workspace_task_files(task)

    assert "run_script.sh" not in files
    assert "parser.py" not in files


def test_swebench_workspace_task_files_include_instance_test_wrapper_when_seeded(tmp_path, monkeypatch) -> None:
    """In seeded mode (swebench_pro_seed_tests=True), run_script.sh is provided."""
    instance_id = "instance_demo__repo-789s"
    instance_dir = tmp_path / "scripts" / instance_id
    instance_dir.mkdir(parents=True)
    (instance_dir / "run_script.sh").write_text("#!/bin/bash\necho selected:$@\n", encoding="utf-8")
    (instance_dir / "parser.py").write_text("print('parse')\n", encoding="utf-8")
    monkeypatch.setenv("SWEBENCH_PRO_SCRIPTS_DIR", str(tmp_path / "scripts"))
    task = TaskSpec(
        id=instance_id,
        repo="demo/repo",
        prompt="Fix the issue.",
        metadata={
            "task_source": "swebench_pro",
            "instance_id": instance_id,
            "swebench_pro_seed_tests": True,
        },
    )

    files = workspace_task_files(task)

    assert "run_script.sh" in files
    assert "echo selected" in files["run_script.sh"]
    assert "parser.py" in files
    assert "parse" in files["parser.py"]


def test_swebench_workspace_task_files_ignores_dataset_scripts_dir(tmp_path, monkeypatch) -> None:
    """Dataset-controlled `swebench_scripts_dir` must NOT override the scripts
    root. Honoring it would let a malicious dataset row point the seeder at
    any path the kcsi user can read, embedding arbitrary script content
    into the agent's TASK.md and shell."""
    instance_id = "instance_demo__repo-790"
    attacker_dir = tmp_path / "attacker" / instance_id
    attacker_dir.mkdir(parents=True)
    (attacker_dir / "run_script.sh").write_text("#!/bin/bash\necho pwn\n", encoding="utf-8")
    monkeypatch.delenv("SWEBENCH_PRO_SCRIPTS_DIR", raising=False)
    task = TaskSpec(
        id=instance_id,
        repo="demo/repo",
        prompt="Fix the issue.",
        metadata={
            "task_source": "swebench_pro",
            "instance_id": instance_id,
            "swebench_scripts_dir": str(tmp_path / "attacker"),
            "scripts_dir": str(tmp_path / "attacker"),
        },
    )

    files = workspace_task_files(task)

    # The attacker-controlled directory must be ignored. Without an env-var
    # override and without a real per-instance script under the repo's default
    # benchmarks/swebench_pro/evaluator/run_scripts/<id>/, the seeder yields
    # nothing.
    assert "run_script.sh" not in files
    assert "parser.py" not in files


@pytest.mark.parametrize("task_source", ["swebench_pro", "arc", "polyglot", "terminal_bench_2", "unknown"])
def test_execution_prompt_memory_block_is_consistent_across_task_sources(task_source: str) -> None:
    task = TaskSpec(
        id=f"{task_source}-1",
        repo="demo/repo",
        prompt="Solve the task.",
        metadata={"task_source": task_source},
    )

    without_memory = build_execution_prompt(task, has_memory=False, generation=1)
    with_memory = build_execution_prompt(task, has_memory=True, generation=1)

    if task_source == "arc":
        # ARC is always native (attempt-file): its execution prompt is fixed and
        # memory-agnostic. Prior memory reaches the agent via MEMORY.md and the
        # system-prompt seed context, not this prompt, so the body carries no
        # "No prior memory" phrase and does not vary with has_memory.
        assert "No prior memory is provided" not in without_memory
        assert without_memory == with_memory
    elif task_source == "terminal_bench_2":
        # TB2 uses its own step list (not `_memory_block`): the memory arm gains
        # only a neutral "Read `MEMORY.md` next." pointer, the control arm none.
        assert "Read `MEMORY.md` next." in with_memory
        assert "Read `MEMORY.md` next." not in without_memory
        assert "Review prior attempts" not in with_memory
    else:
        # The memory arm gains only a neutral knowledge pointer; the control arm
        # has none. Neither carries a strategy directive (issue #1151 parity).
        assert "Review prior attempts" not in without_memory
        assert "Review prior attempts" in with_memory

    # Parity (issue #1151): the memory-arm prompt must not carry any
    # strategy-diversification directive that the control arm lacks. The two
    # branches differ only by the neutral knowledge pointer, so these removed
    # exhortations must be absent from BOTH branches for every task source.
    for prompt in (with_memory, without_memory):
        assert "avoid repeating the same approach" not in prompt
        assert "fundamentally different strategy" not in prompt
        assert "do not repeat any rejected hypothesis" not in prompt
        assert "actively reuse it before planning or repeating exploration" not in prompt

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from ksi.models import TaskSpec
from ksi.runtime.container_host import KsiContainerExecutor


def _runner_stdout(*, task_id: str) -> str:
    return json.dumps(
        {
            "result": "done",
            "tool_trace": [],
            "meta": {
                "generation": 1,
                "agent_id": "agent-0",
                "task_id": task_id,
                "status": "success",
                "input_tokens": 1,
                "output_tokens": 1,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
    )


def test_terminal_bench_2_seed_policy_uses_native_files_and_thin_instruction(tmp_path):
    captured: list[dict] = []
    instruction_path = tmp_path / "INSTRUCTION.md"
    instruction_path.write_text("DEFAULT TEMPLATE\n", encoding="utf-8")

    def fake_run(cmd, **kw):
        with open(cmd[-1]) as handle:
            captured.append(json.load(handle))
        return MagicMock(returncode=0, stdout=_runner_stdout(task_id="git-multibranch"), stderr="")

    ex = KsiContainerExecutor(
        command=["echo", "dummy"],
        working_dir=str(tmp_path),
        instruction_path=str(instruction_path),
        agent_workspace_root=str(tmp_path / "workspaces"),
        timeout_sec=60,
        env={
            "MODEL_PROVIDER": "anthropic",
            "MODEL_AUTH_MODE": "api",
            "MODEL": "claude-sonnet-4-6",
            "ANTHROPIC_API_KEY": "sk-test",
        },
    )

    task = TaskSpec(
        id="git-multibranch",
        prompt="Solve the native Terminal-Bench 2 task described in instruction.md.",
        metadata={
            "task_source": "terminal_bench_2",
            "task_files": {
                "tb2/instruction.md": "Native task statement.\n",
                "tb2/task.toml": 'version = "1.0"\n',
            },
            "category": "system-administration",
            "difficulty": "medium",
            "agent_timeout_sec": 900.0,
            "verifier_timeout_sec": 900.0,
        },
    )

    with patch("ksi.runtime.container_host._run_command_with_backstop", side_effect=fake_run):
        ex.run_task(
            generation=1,
            agent_id="agent-0",
            task=task,
            agent_seed_package={},
            experiment_name="tb2-seed-test",
        )

    assert len(captured) == 1
    seed = captured[0]["workspace_seed"]
    assert seed["instruction_md"] == "DEFAULT TEMPLATE\n"
    assert seed["task_files"]["tb2/instruction.md"] == "Native task statement."
    assert seed["task_files"]["tb2/task.toml"] == 'version = "1.0"'

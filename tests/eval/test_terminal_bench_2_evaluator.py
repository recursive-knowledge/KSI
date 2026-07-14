from __future__ import annotations

import math
from pathlib import Path

import pytest

from kcsi.benchmarks.terminal_bench_2 import (
    TB2_VERIFIER_FAIL_CLOSED_STATUS,
    TB2_VERIFIER_MISSING_STATUS,
    TerminalBench2ContractError,
    TerminalBench2Evaluator,
    resolve_terminal_bench_2_task_contract,
)
from kcsi.models import TaskSpec


def _write_tb2_task(tmp_path: Path, task_id: str = "demo-task") -> Path:
    task_root = tmp_path / task_id
    (task_root / "environment").mkdir(parents=True)
    (task_root / "solution").mkdir()
    (task_root / "tests").mkdir()
    (task_root / "instruction.md").write_text("Solve the task.\n", encoding="utf-8")
    (task_root / "task.toml").write_text(
        """\
version = "1.0"

[metadata]
author_name = "Example"
author_email = "example@example.com"
difficulty = "medium"
category = "software-engineering"

[verifier]
timeout_sec = 900.0

[agent]
timeout_sec = 1200.0

[environment]
docker_image = "example/demo-task:latest"
""",
        encoding="utf-8",
    )
    (task_root / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n", encoding="utf-8")
    (task_root / "solution" / "solve.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (task_root / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n",
        encoding="utf-8",
    )
    return task_root


def _task_spec(task_root: Path, task_id: str = "demo-task") -> TaskSpec:
    return TaskSpec(
        id=task_id,
        prompt="demo",
        metadata={
            "task_source": "terminal_bench_2",
            "task_root": str(task_root),
        },
    )


def test_resolve_terminal_bench_2_contract_reads_layout(tmp_path: Path):
    task_root = _write_tb2_task(tmp_path)
    contract = resolve_terminal_bench_2_task_contract(_task_spec(task_root))

    assert contract.task_id == "demo-task"
    assert contract.task_root == task_root.resolve()
    assert contract.docker_image == "example/demo-task:latest"
    assert contract.build_timeout_sec == 1800.0
    assert contract.agent_timeout_sec == 1200.0
    assert contract.verifier_timeout_sec == 900.0
    assert contract.cpus is None
    assert contract.memory == ""
    assert contract.storage == ""
    assert contract.category == "software-engineering"
    assert contract.difficulty == "medium"


def test_resolve_terminal_bench_2_contract_uses_task_toml_timeouts_over_metadata(tmp_path: Path):
    task_root = _write_tb2_task(tmp_path)
    task = _task_spec(task_root)
    task.metadata["agent_timeout_sec"] = 30.0
    task.metadata["verifier_timeout_sec"] = 45.0

    contract = resolve_terminal_bench_2_task_contract(task)

    assert contract.agent_timeout_sec == 1200.0
    assert contract.verifier_timeout_sec == 900.0


def test_resolve_contract_build_timeout_honors_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_root = _write_tb2_task(tmp_path)
    monkeypatch.setenv("KCSI_TB2_BUILD_TIMEOUT_SEC", "3600")
    contract = resolve_terminal_bench_2_task_contract(_task_spec(task_root))
    assert contract.build_timeout_sec == 3600.0


def test_resolve_contract_build_timeout_uses_max_of_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_root = tmp_path / "demo-task"
    (task_root / "environment").mkdir(parents=True)
    (task_root / "solution").mkdir()
    (task_root / "tests").mkdir()
    (task_root / "instruction.md").write_text("Solve.\n", encoding="utf-8")
    (task_root / "task.toml").write_text(
        """\
version = "1.0"
[metadata]
author_name = "x"
author_email = "x@x"
difficulty = "medium"
category = "software-engineering"
[verifier]
timeout_sec = 900.0
[agent]
timeout_sec = 1200.0
[environment]
docker_image = "example/demo-task:latest"
build_timeout_sec = 7200.0
""",
        encoding="utf-8",
    )
    (task_root / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n", encoding="utf-8")
    (task_root / "solution" / "solve.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (task_root / "tests" / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    monkeypatch.delenv("KCSI_TB2_BUILD_TIMEOUT_SEC", raising=False)

    contract = resolve_terminal_bench_2_task_contract(_task_spec(task_root))
    assert contract.build_timeout_sec == 7200.0


def test_resolve_terminal_bench_2_contract_rejects_missing_files(tmp_path: Path):
    task_root = _write_tb2_task(tmp_path)
    (task_root / "tests" / "test.sh").unlink()

    with pytest.raises(TerminalBench2ContractError, match="missing required files"):
        resolve_terminal_bench_2_task_contract(_task_spec(task_root))


def test_evaluator_preflight_returns_runtime_metadata(tmp_path: Path):
    task_root = _write_tb2_task(tmp_path)
    evaluator = TerminalBench2Evaluator()

    result = evaluator.preflight(task=_task_spec(task_root))

    assert result["status"] == "preflight_ok"
    assert result["instance_id"] == "demo-task"
    assert result["task_root"] == str(task_root.resolve())
    assert result["docker_image"] == "example/demo-task:latest"
    assert result["build_timeout_sec"] == 1800.0
    assert result["agent_timeout_sec"] == 1200.0
    assert result["verifier_timeout_sec"] == 900.0
    assert result["timeout_source"] == "task.toml"


def test_evaluator_scores_from_reward_runtime_meta(tmp_path: Path):
    task_root = _write_tb2_task(tmp_path)
    evaluator = TerminalBench2Evaluator()

    result = evaluator.evaluate(
        task=_task_spec(task_root),
        model_output="changed nginx config",
        runtime_meta={
            "reward": 1.0,
            "agent_exit_code": 127,
            "verifier_exit_code": 0,
            "reward_path": "/tmp/reward.txt",
            "ctrf_path": "/tmp/ctrf.json",
        },
    )

    assert result["status"] == "completed"
    assert result["native_score"] == 1.0
    assert result["reward"] == 1.0
    assert result["resolved"] is True
    assert result["agent_exit_code"] == 127
    assert result["verifier_exit_code"] == 0
    assert result["reward_path"] == "/tmp/reward.txt"
    assert result["ctrf_path"] == "/tmp/ctrf.json"


def test_evaluator_handles_missing_reward_as_unsolved(tmp_path: Path):
    task_root = _write_tb2_task(tmp_path)
    evaluator = TerminalBench2Evaluator()

    result = evaluator.evaluate(
        task=_task_spec(task_root),
        model_output="no-op",
        runtime_meta={"agent_exit_code": 0, "verifier_exit_code": 1},
    )

    assert result["status"] == "verifier_failed"
    assert result["native_score"] == 0.0
    assert result["reward"] is None
    assert result["resolved"] is False


def test_evaluator_distinguishes_missing_reward_from_true_zero(tmp_path: Path):
    task_root = _write_tb2_task(tmp_path)
    evaluator = TerminalBench2Evaluator()

    result = evaluator.evaluate(
        task=_task_spec(task_root),
        model_output="timed out verifier",
        runtime_meta={
            "agent_exit_code": 0,
            "verifier_exit_code": None,
            "trial_status": "verifier_did_not_produce_reward",
        },
    )

    assert result["status"] == "verifier_did_not_produce_reward"
    # Verifier never ran -> unscored (None), NOT a fabricated genuine 0.0.
    # evaluate() itself now distinguishes missing from true-zero: a genuine
    # verifier_failed still emits 0.0 (test above), so this None never reaches
    # _best_scores / record_attempt or the distillation/seeding prose (#977).
    assert result["native_score"] is None
    assert result["reward"] is None
    assert result["resolved"] is False


def test_evaluator_treats_fail_closed_status_as_unscored(tmp_path: Path):
    """#1206: strict-mode fail-closed (verifier refused, never ran) is unscored
    (native_score None), NOT a fabricated genuine 0.0 -- same as the
    verifier-missing case, so it never contaminates _best_scores/distillation."""
    task_root = _write_tb2_task(tmp_path)
    evaluator = TerminalBench2Evaluator()

    result = evaluator.evaluate(
        task=_task_spec(task_root),
        model_output="",
        runtime_meta={
            "agent_exit_code": 0,
            "verifier_exit_code": None,
            "reward": None,
            "trial_status": TB2_VERIFIER_FAIL_CLOSED_STATUS,
            "verifier_fail_closed": True,
        },
    )

    assert result["status"] == TB2_VERIFIER_FAIL_CLOSED_STATUS
    assert result["native_score"] is None
    assert result["reward"] is None
    assert result["resolved"] is False


@pytest.mark.parametrize("reward_raw", ["inf", "-inf", "nan", "Infinity", math.inf, -math.inf, math.nan])
@pytest.mark.parametrize("verifier_exit_code", [None, 0])
def test_evaluator_rejects_non_finite_reward(tmp_path: Path, reward_raw: object, verifier_exit_code: int | None):
    """The reward file is AGENT-CONTROLLED; ``float`` accepts nan/inf without
    raising. A container writing ``inf`` must NOT become a solve, and ``nan``
    must NOT leak into native_score -- non-finite routes through the unscored
    None path, never a fabricated solve/score."""
    task_root = _write_tb2_task(tmp_path)
    evaluator = TerminalBench2Evaluator()

    result = evaluator.evaluate(
        task=_task_spec(task_root),
        model_output="adversarial container reward",
        runtime_meta={"reward": reward_raw, "agent_exit_code": 0, "verifier_exit_code": verifier_exit_code},
    )

    # inf must never mark the task solved; nan must never reach native_score.
    assert result["resolved"] is False
    assert result["reward"] is None
    native_score = result["native_score"]
    assert native_score is None or math.isfinite(native_score)
    # Verifier produced no genuine reward -> unscored (None), same contract as
    # the verifier-missing case.
    assert result["status"] == TB2_VERIFIER_MISSING_STATUS
    assert native_score is None

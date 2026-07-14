from __future__ import annotations

import csv
import json
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kcsi.benchmarks.swebench_pro import (
    SwebenchProEvaluator,
    _build_tests_status,
    _git_workspace_diff,
    _grader_test_file_paths,
    _patch_from_tool_trace,
    filter_grader_test_hunks,
)
from kcsi.benchmarks.swebench_pro_external import (
    EVALUATOR_REVISION,
    REVISION_MARKER,
    SWEBENCH_FAILURE_STATUSES,
)
from kcsi.models import TaskSpec
from kcsi.tasks.loaders import load_tasks_for_source


def _write_eval_root(path: Path, *, revision: str = EVALUATOR_REVISION) -> None:
    (path / "run_scripts").mkdir(parents=True)
    (path / "swe_bench_pro_eval.py").write_text("# stub\n", encoding="utf-8")
    (path / REVISION_MARKER).write_text(revision + "\n", encoding="utf-8")


def test_load_swebench_pro_tasks_from_jsonl(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    row = {
        "instance_id": "instance_demo__repo-123",
        "repo": "demo/repo",
        "problem_statement": "Fix widget normalization.",
        "requirements": {"behavior": "Normalize stale widgets before saving."},
        "interface": "Function: normalize_widget(widget)",
        "base_commit": "deadbeef",
        "image_name": "registry.example/sweap-images/demo:repo-123",
        "fail_to_pass": ["tests.widget.test_fix"],
        "pass_to_pass": ["tests.widget.test_existing"],
    }
    tasks_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    tasks = load_tasks_for_source(task_source="swebench_pro", tasks_path=tasks_path)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.id == "instance_demo__repo-123"
    assert task.repo == "demo/repo"
    assert "## Requirements" in task.prompt
    assert "Normalize stale widgets before saving." in task.prompt
    assert "## Interface" in task.prompt
    assert "normalize_widget" in task.prompt
    assert task.metadata["task_source"] == "swebench_pro"
    assert task.metadata["base_commit"] == "deadbeef"
    assert task.metadata["image_name"] == "registry.example/sweap-images/demo:repo-123"
    assert task.metadata["fail_to_pass"] == ["tests.widget.test_fix"]


def test_load_swebench_pro_tasks_omits_missing_requirements_and_interface(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    row = {
        "instance_id": "instance_demo__repo-456",
        "repo": "demo/repo",
        "problem_statement": "Fix widget persistence.",
        "requirements": "null",
        "interface": float("nan"),
    }
    tasks_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    task = load_tasks_for_source(task_source="swebench_pro", tasks_path=tasks_path)[0]

    assert task.prompt == "Fix widget persistence."
    assert "## Requirements" not in task.prompt
    assert "## Interface" not in task.prompt


def test_swebench_pro_evaluator_invokes_official_repo(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "SWE-bench_Pro-os"
    _write_eval_root(repo_root)

    raw_sample_path = tmp_path / "samples.csv"
    with raw_sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["instance_id", "repo", "fail_to_pass", "pass_to_pass"])
        writer.writeheader()
        writer.writerow(
            {
                "instance_id": "instance_demo__repo-123",
                "repo": "demo/repo",
                "fail_to_pass": "['tests.widget.test_fix']",
                "pass_to_pass": "['tests.widget.test_existing']",
            }
        )

    captured: dict[str, object] = {}

    def fake_run_eval(cmd, cwd, env, timeout):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        captured["timeout"] = timeout
        output_dir = Path(cmd[cmd.index("--output_dir") + 1])
        instance_dir = output_dir / "instance_demo__repo-123"
        instance_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "eval_results.json").write_text(
            json.dumps({"instance_demo__repo-123": True}),
            encoding="utf-8",
        )
        (instance_dir / "kcsi_output.json").write_text(
            json.dumps(
                {
                    "tests": [
                        {"name": "tests.widget.test_fix", "status": "PASSED"},
                        {"name": "tests.widget.test_existing", "status": "PASSED"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr=""), {}

    monkeypatch.setattr("kcsi.benchmarks.swebench_pro._run_eval_command", fake_run_eval)

    evaluator = SwebenchProEvaluator(
        raw_sample_path=str(raw_sample_path),
        repo_root=str(repo_root),
        timeout_sec=30,
        use_local_docker=True,
    )
    result = evaluator.evaluate(
        task=load_tasks_for_source(task_source="swebench_pro", tasks_path=raw_sample_path)[0],
        model_output="<patch>diff --git a/a b/a\n--- a/a\n+++ b/a\n@@\n-x\n+y\n</patch>",
    )

    assert result["swebench_status"] == "ok"
    assert result["resolved"] is True
    assert result["native_score"] == 1.0
    assert captured["cmd"][1] == str(repo_root / "swe_bench_pro_eval.py")
    assert "--use_local_docker" in captured["cmd"]
    assert "--block_network" not in captured["cmd"]
    assert captured["timeout"] == 30
    assert str(repo_root) in captured["env"]["PYTHONPATH"]
    tests_status = result["instance_report"]["tests_status"]
    assert tests_status["FAIL_TO_PASS"]["failure"] == []
    assert tests_status["PASS_TO_PASS"]["failure"] == []


def test_swebench_pro_evaluator_uses_workspace_diff_fallback(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "SWE-bench_Pro-os"
    _write_eval_root(repo_root)
    raw_sample_path = tmp_path / "samples.jsonl"
    raw_sample_path.write_text(
        json.dumps({"instance_id": "instance_demo__repo-123", "repo": "demo/repo"}) + "\n",
        encoding="utf-8",
    )
    workspace_diff = "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-x\n+y\n"
    captured: dict[str, object] = {}

    def fake_run_eval(cmd, cwd, env, timeout):  # type: ignore[no-untyped-def]
        patch_path = Path(cmd[cmd.index("--patch_path") + 1])
        captured["patch_payload"] = json.loads(patch_path.read_text(encoding="utf-8"))
        output_dir = Path(cmd[cmd.index("--output_dir") + 1])
        instance_dir = output_dir / "instance_demo__repo-123"
        instance_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "eval_results.json").write_text(
            json.dumps({"instance_demo__repo-123": True}),
            encoding="utf-8",
        )
        (instance_dir / "kcsi_output.json").write_text(json.dumps({"tests": []}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr=""), {}

    monkeypatch.setattr("kcsi.benchmarks.swebench_pro._run_eval_command", fake_run_eval)
    evaluator = SwebenchProEvaluator(
        raw_sample_path=str(raw_sample_path),
        repo_root=str(repo_root),
        timeout_sec=30,
    )

    result = evaluator.evaluate(
        task=load_tasks_for_source(task_source="swebench_pro", tasks_path=raw_sample_path)[0],
        model_output="",
        runtime_meta={"workspace_diff": workspace_diff},
    )

    assert result["swebench_status"] == "ok"
    assert result["patch_source"] == "workspace_diff"
    assert captured["patch_payload"][0]["patch"] == workspace_diff


def test_swebench_pro_evaluator_prefers_workspace_diff_over_model_output(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "SWE-bench_Pro-os"
    _write_eval_root(repo_root)
    raw_sample_path = tmp_path / "samples.jsonl"
    raw_sample_path.write_text(
        json.dumps({"instance_id": "instance_demo__repo-123", "repo": "demo/repo"}) + "\n",
        encoding="utf-8",
    )
    workspace_diff = "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-real\n+workspace\n"
    stale_model_patch = "<patch>diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-real\n+stale\n</patch>"
    captured: dict[str, object] = {}

    def fake_run_eval(cmd, cwd, env, timeout):  # type: ignore[no-untyped-def]
        patch_path = Path(cmd[cmd.index("--patch_path") + 1])
        captured["patch_payload"] = json.loads(patch_path.read_text(encoding="utf-8"))
        output_dir = Path(cmd[cmd.index("--output_dir") + 1])
        instance_dir = output_dir / "instance_demo__repo-123"
        instance_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "eval_results.json").write_text(
            json.dumps({"instance_demo__repo-123": True}),
            encoding="utf-8",
        )
        (instance_dir / "kcsi_output.json").write_text(json.dumps({"tests": []}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr=""), {}

    monkeypatch.setattr("kcsi.benchmarks.swebench_pro._run_eval_command", fake_run_eval)
    evaluator = SwebenchProEvaluator(
        raw_sample_path=str(raw_sample_path),
        repo_root=str(repo_root),
        timeout_sec=30,
    )

    result = evaluator.evaluate(
        task=load_tasks_for_source(task_source="swebench_pro", tasks_path=raw_sample_path)[0],
        model_output=stale_model_patch,
        runtime_meta={"workspace_diff": workspace_diff},
    )

    assert result["swebench_status"] == "ok"
    assert result["patch_source"] == "workspace_diff"
    assert captured["patch_payload"][0]["patch"] == workspace_diff
    assert "stale" not in captured["patch_payload"][0]["patch"]


def test_swebench_pro_evaluator_uses_runtime_meta_workspace_repo_fallback(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "SWE-bench_Pro-os"
    _write_eval_root(repo_root)
    raw_sample_path = tmp_path / "samples.jsonl"
    raw_sample_path.write_text(
        json.dumps({"instance_id": "instance_demo__repo-123", "repo": "demo/repo"}) + "\n",
        encoding="utf-8",
    )
    workspace_repo = tmp_path / "workspace" / "repo"
    workspace_repo.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=workspace_repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "tester"], cwd=workspace_repo, check=True)
    subprocess.run(["git", "config", "user.email", "tester@example.com"], cwd=workspace_repo, check=True)
    (workspace_repo / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=workspace_repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=workspace_repo, check=True, capture_output=True)
    (workspace_repo / "new_file.py").write_text("VALUE = 1\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run_eval(cmd, cwd, env, timeout):  # type: ignore[no-untyped-def]
        patch_path = Path(cmd[cmd.index("--patch_path") + 1])
        captured["patch_payload"] = json.loads(patch_path.read_text(encoding="utf-8"))
        output_dir = Path(cmd[cmd.index("--output_dir") + 1])
        instance_dir = output_dir / "instance_demo__repo-123"
        instance_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "eval_results.json").write_text(
            json.dumps({"instance_demo__repo-123": True}),
            encoding="utf-8",
        )
        (instance_dir / "kcsi_output.json").write_text(json.dumps({"tests": []}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr=""), {}

    monkeypatch.setattr("kcsi.benchmarks.swebench_pro._run_eval_command", fake_run_eval)
    evaluator = SwebenchProEvaluator(
        raw_sample_path=str(raw_sample_path),
        repo_root=str(repo_root),
        timeout_sec=30,
    )

    result = evaluator.evaluate(
        task=load_tasks_for_source(task_source="swebench_pro", tasks_path=raw_sample_path)[0],
        model_output="",
        runtime_meta={"host_workspace_repo_dir": str(workspace_repo)},
    )

    patch = captured["patch_payload"][0]["patch"]
    assert result["swebench_status"] == "ok"
    assert result["patch_source"] == "workspace_diff"
    assert "diff --git a/new_file.py b/new_file.py" in patch
    assert "+VALUE = 1" in patch


def test_swebench_pro_evaluator_timeout_grace_is_explicit(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "SWE-bench_Pro-os"
    _write_eval_root(repo_root)
    raw_sample_path = tmp_path / "samples.jsonl"
    raw_sample_path.write_text(
        json.dumps({"instance_id": "instance_demo__repo-123", "repo": "demo/repo"}) + "\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run_eval(cmd, cwd, env, timeout):  # type: ignore[no-untyped-def]
        captured["timeout"] = timeout
        output_dir = Path(cmd[cmd.index("--output_dir") + 1])
        instance_dir = output_dir / "instance_demo__repo-123"
        instance_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "eval_results.json").write_text(
            json.dumps({"instance_demo__repo-123": False}),
            encoding="utf-8",
        )
        (instance_dir / "kcsi_output.json").write_text(json.dumps({"tests": []}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr=""), {}

    monkeypatch.setattr("kcsi.benchmarks.swebench_pro._run_eval_command", fake_run_eval)

    evaluator = SwebenchProEvaluator(
        raw_sample_path=str(raw_sample_path),
        repo_root=str(repo_root),
        timeout_sec=30,
        harness_grace_sec=5,
    )
    evaluator.evaluate(
        task=load_tasks_for_source(task_source="swebench_pro", tasks_path=raw_sample_path)[0],
        model_output="<patch>diff --git a/a b/a\n--- a/a\n+++ b/a\n@@\n-x\n+y\n</patch>",
    )

    assert captured["timeout"] == 35


def test_swebench_pro_evaluator_timeout_reports_cleanup(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "SWE-bench_Pro-os"
    _write_eval_root(repo_root)
    raw_sample_path = tmp_path / "samples.jsonl"
    raw_sample_path.write_text(
        json.dumps({"instance_id": "instance_demo__repo-123", "repo": "demo/repo"}) + "\n",
        encoding="utf-8",
    )

    def fake_run_eval(cmd, cwd, env, timeout):  # type: ignore[no-untyped-def]
        return None, {
            "swebench_process_cleanup_attempted": True,
            "swebench_process_cleanup_method": "process_group",
            "swebench_process_cleanup_status": "killed",
            "swebench_container_cleanup_attempted": False,
            "swebench_container_cleanup_status": "not_attempted",
            "swebench_stdout_tail": "partial stdout",
            "swebench_stderr_tail": "partial stderr",
        }

    monkeypatch.setattr("kcsi.benchmarks.swebench_pro._run_eval_command", fake_run_eval)

    evaluator = SwebenchProEvaluator(
        raw_sample_path=str(raw_sample_path),
        repo_root=str(repo_root),
        timeout_sec=30,
        harness_grace_sec=5,
    )
    result = evaluator.evaluate(
        task=load_tasks_for_source(task_source="swebench_pro", tasks_path=raw_sample_path)[0],
        model_output="<patch>diff --git a/a b/a\n--- a/a\n+++ b/a\n@@\n-x\n+y\n</patch>",
    )

    assert result["swebench_status"] == "harness_timeout"
    assert result["instance_id"] == "instance_demo__repo-123"
    assert result["swebench_process_cleanup_attempted"] is True
    assert result["swebench_process_cleanup_method"] == "process_group"
    assert result["swebench_process_cleanup_status"] == "killed"
    assert result["swebench_container_cleanup_attempted"] is False
    assert result["swebench_container_cleanup_status"] == "not_attempted"
    assert result["swebench_stdout_tail"] == "partial stdout"
    assert result["swebench_stderr_tail"] == "partial stderr"


def test_swebench_pro_evaluator_nonzero_returncode_marks_harness_failed(monkeypatch, tmp_path: Path) -> None:
    """A non-zero harness returncode (Docker failure / OOM / upstream crash) → harness_failed.

    Covers swebench_pro.py:603-610. The result carries no native_score; the
    score path (_score_from_eval, swebench_pro branch) now returns None (unscored — no trustworthy verdict, not a genuine 0.0 failure).
    """
    repo_root = tmp_path / "SWE-bench_Pro-os"
    _write_eval_root(repo_root)
    raw_sample_path = tmp_path / "samples.jsonl"
    raw_sample_path.write_text(
        json.dumps({"instance_id": "instance_demo__repo-123", "repo": "demo/repo"}) + "\n",
        encoding="utf-8",
    )

    def fake_run_eval(cmd, cwd, env, timeout):  # type: ignore[no-untyped-def]
        # Harness ran but exited non-zero without writing eval_results.json.
        return SimpleNamespace(returncode=1, stdout="boom stdout", stderr="boom stderr"), {}

    monkeypatch.setattr("kcsi.benchmarks.swebench_pro._run_eval_command", fake_run_eval)

    evaluator = SwebenchProEvaluator(
        raw_sample_path=str(raw_sample_path),
        repo_root=str(repo_root),
        timeout_sec=30,
    )
    task = load_tasks_for_source(task_source="swebench_pro", tasks_path=raw_sample_path)[0]
    result = evaluator.evaluate(
        task=task,
        model_output="<patch>diff --git a/a b/a\n--- a/a\n+++ b/a\n@@\n-x\n+y\n</patch>",
    )

    assert result["swebench_status"] == "harness_failed"
    assert result["instance_id"] == "instance_demo__repo-123"
    assert result["swebench_returncode"] == 1
    assert result["swebench_stdout_tail"] == "boom stdout"
    assert result["swebench_stderr_tail"] == "boom stderr"
    # The harness_failed branch returns no native_score.
    assert "native_score" not in result

    # The score path treats the missing native_score + harness_failed status as unscored (None) for swebench_pro.
    from kcsi.orchestrator.engine import _score_from_eval

    assert _score_from_eval(result, task=task) is None


def test_swebench_pro_evaluator_malformed_eval_results_is_harness_failed(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "SWE-bench_Pro-os"
    _write_eval_root(repo_root)
    raw_sample_path = tmp_path / "samples.jsonl"
    raw_sample_path.write_text(
        json.dumps({"instance_id": "instance_demo__repo-123", "repo": "demo/repo"}) + "\n",
        encoding="utf-8",
    )

    def fake_run_eval(cmd, cwd, env, timeout):  # type: ignore[no-untyped-def]
        output_dir = Path(cmd[cmd.index("--output_dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "eval_results.json").write_text("{not valid json", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="stdout tail", stderr="stderr tail"), {}

    monkeypatch.setattr("kcsi.benchmarks.swebench_pro._run_eval_command", fake_run_eval)

    result = SwebenchProEvaluator(raw_sample_path=str(raw_sample_path), repo_root=str(repo_root)).evaluate(
        task=load_tasks_for_source(task_source="swebench_pro", tasks_path=raw_sample_path)[0],
        model_output="<patch>diff --git a/a b/a\n</patch>",
    )

    assert result["swebench_status"] == "harness_failed"
    assert result["swebench_malformed_report"] is True
    assert result["swebench_report_path"].endswith("eval_results.json")
    assert "not valid json" in result["swebench_parse_error"]
    assert result["swebench_stdout_tail"] == "stdout tail"
    assert result["swebench_stderr_tail"] == "stderr tail"
    assert "native_score" not in result


def test_swebench_pro_evaluator_malformed_output_json_is_harness_failed(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "SWE-bench_Pro-os"
    _write_eval_root(repo_root)
    raw_sample_path = tmp_path / "samples.jsonl"
    raw_sample_path.write_text(
        json.dumps({"instance_id": "instance_demo__repo-123", "repo": "demo/repo"}) + "\n",
        encoding="utf-8",
    )

    def fake_run_eval(cmd, cwd, env, timeout):  # type: ignore[no-untyped-def]
        output_dir = Path(cmd[cmd.index("--output_dir") + 1])
        instance_dir = output_dir / "instance_demo__repo-123"
        instance_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "eval_results.json").write_text(
            json.dumps({"instance_demo__repo-123": False}),
            encoding="utf-8",
        )
        (instance_dir / "kcsi_output.json").write_text("{not valid json", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="stdout tail", stderr="stderr tail"), {}

    monkeypatch.setattr("kcsi.benchmarks.swebench_pro._run_eval_command", fake_run_eval)

    result = SwebenchProEvaluator(raw_sample_path=str(raw_sample_path), repo_root=str(repo_root)).evaluate(
        task=load_tasks_for_source(task_source="swebench_pro", tasks_path=raw_sample_path)[0],
        model_output="<patch>diff --git a/a b/a\n</patch>",
    )

    assert result["swebench_status"] == "harness_failed"
    assert result["swebench_malformed_output_json"] is True
    assert result["swebench_output_json_path"].endswith("kcsi_output.json")
    assert "not valid json" in result["swebench_parse_error"]
    assert result["swebench_stdout_tail"] == "stdout tail"
    assert result["swebench_stderr_tail"] == "stderr tail"
    assert "native_score" not in result


def test_swebench_pro_evaluator_rejects_invalid_timeout_contract(tmp_path: Path) -> None:
    raw_sample_path = tmp_path / "samples.jsonl"
    raw_sample_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="timeout_sec"):
        SwebenchProEvaluator(raw_sample_path=str(raw_sample_path), timeout_sec=0)
    with pytest.raises(ValueError, match="harness_grace_sec"):
        SwebenchProEvaluator(raw_sample_path=str(raw_sample_path), harness_grace_sec=-1)


def test_swebench_pro_evaluator_preserves_positional_use_local_docker() -> None:
    evaluator = SwebenchProEvaluator("samples.jsonl", "", "jefzda", "", 30, False)

    assert evaluator.timeout_sec == 30
    assert evaluator.use_local_docker is False
    assert evaluator.harness_grace_sec == 0


def test_swebench_pro_evaluator_emits_block_network(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "SWE-bench_Pro-os"
    _write_eval_root(repo_root)
    raw_sample_path = tmp_path / "samples.jsonl"
    raw_sample_path.write_text(
        json.dumps({"instance_id": "instance_demo__repo-123", "repo": "demo/repo"}) + "\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run_eval(cmd, cwd, env, timeout):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        output_dir = Path(cmd[cmd.index("--output_dir") + 1])
        instance_dir = output_dir / "instance_demo__repo-123"
        instance_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "eval_results.json").write_text(
            json.dumps({"instance_demo__repo-123": False}),
            encoding="utf-8",
        )
        (instance_dir / "kcsi_output.json").write_text(json.dumps({"tests": []}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr=""), {}

    monkeypatch.setattr("kcsi.benchmarks.swebench_pro._run_eval_command", fake_run_eval)

    evaluator = SwebenchProEvaluator(
        raw_sample_path=str(raw_sample_path),
        repo_root=str(repo_root),
        timeout_sec=30,
        block_network=True,
    )
    evaluator.evaluate(
        task=load_tasks_for_source(task_source="swebench_pro", tasks_path=raw_sample_path)[0],
        model_output="<patch>diff --git a/a b/a\n--- a/a\n+++ b/a\n@@\n-x\n+y\n</patch>",
    )

    assert "--block_network" in captured["cmd"]


def test_swebench_pro_evaluator_rejects_wrong_evaluator_revision(tmp_path: Path) -> None:
    repo_root = tmp_path / "SWE-bench_Pro-os"
    _write_eval_root(repo_root, revision="wrong")
    raw_sample_path = tmp_path / "samples.jsonl"
    raw_sample_path.write_text(
        json.dumps({"instance_id": "instance_demo__repo-123", "repo": "demo/repo"}) + "\n",
        encoding="utf-8",
    )

    evaluator = SwebenchProEvaluator(
        raw_sample_path=str(raw_sample_path),
        repo_root=str(repo_root),
    )
    result = evaluator.evaluate(
        task=load_tasks_for_source(task_source="swebench_pro", tasks_path=raw_sample_path)[0],
        model_output="<patch>diff --git a/a b/a\n--- a/a\n+++ b/a\n@@\n-x\n+y\n</patch>",
    )

    assert result["swebench_status"] == "harness_failed"
    assert "expected" in result["error"]


def test_swebench_pro_tests_status_marks_missing_parser_rows_unknown() -> None:
    raw_sample = {
        "fail_to_pass": ["TestMux/SSHProxyHelloSignature", "TestMux"],
        "pass_to_pass": ["TestMux/ProxyLine"],
    }

    tests_status = _build_tests_status({"tests": []}, raw_sample)

    assert tests_status["observed_count"] == 0
    assert tests_status["FAIL_TO_PASS"]["success"] == []
    assert tests_status["FAIL_TO_PASS"]["failure"] == []
    assert tests_status["FAIL_TO_PASS"]["unknown"] == [
        "TestMux/SSHProxyHelloSignature",
        "TestMux",
    ]
    assert tests_status["PASS_TO_PASS"]["failure"] == []
    assert tests_status["PASS_TO_PASS"]["unknown"] == ["TestMux/ProxyLine"]


def test_swebench_pro_tests_status_handles_null_tests_field() -> None:
    # The grader may emit an explicit ``{"tests": null}`` on a parse/timeout
    # signal. ``dict.get("tests", [])`` returns None there, so iterating it must
    # not raise — the eval result would otherwise be lost rather than recorded.
    raw_sample = {
        "fail_to_pass": ["TestA"],
        "pass_to_pass": ["TestB"],
    }

    tests_status = _build_tests_status({"tests": None}, raw_sample)

    assert tests_status["observed_count"] == 0
    assert tests_status["FAIL_TO_PASS"]["unknown"] == ["TestA"]
    assert tests_status["PASS_TO_PASS"]["unknown"] == ["TestB"]


def test_swebench_pro_tests_status_keeps_observed_failures_separate() -> None:
    raw_sample = {
        "fail_to_pass": ["TestFix", "TestMissing"],
        "pass_to_pass": ["TestExisting"],
    }
    output = {
        "tests": [
            {"name": "TestFix", "status": "FAILED"},
            {"name": "TestExisting", "status": "PASSED"},
        ]
    }

    tests_status = _build_tests_status(output, raw_sample)

    assert tests_status["observed_count"] == 2
    assert tests_status["FAIL_TO_PASS"]["failure"] == ["TestFix"]
    assert tests_status["FAIL_TO_PASS"]["unknown"] == ["TestMissing"]
    assert tests_status["PASS_TO_PASS"]["success"] == ["TestExisting"]
    assert tests_status["PASS_TO_PASS"]["failure"] == []


def test_swebench_pro_tests_status_treats_observed_skipped_separately() -> None:
    raw_sample = {
        "fail_to_pass": ["TestSkipped", "TestXfail", "TestMissing"],
        "pass_to_pass": ["TestError"],
    }
    output = {
        "tests": [
            {"name": "TestSkipped", "status": "SKIPPED"},
            {"name": "TestXfail", "status": "XFAIL"},
            {"name": "TestError", "status": "ERROR"},
        ]
    }

    tests_status = _build_tests_status(output, raw_sample)

    assert tests_status["observed_count"] == 3
    assert tests_status["FAIL_TO_PASS"]["success"] == []
    assert tests_status["FAIL_TO_PASS"]["failure"] == []
    assert tests_status["FAIL_TO_PASS"]["skipped"] == ["TestSkipped", "TestXfail"]
    assert tests_status["FAIL_TO_PASS"]["unknown"] == ["TestMissing"]
    assert tests_status["PASS_TO_PASS"]["failure"] == []
    assert tests_status["PASS_TO_PASS"]["skipped"] == ["TestError"]


def test_swebench_pro_tests_status_matches_parametrized_ids_verbatim() -> None:
    # Guard for #1010 item 2 ("date-parametrized test-name drift"). Expected and
    # observed test names — including the ``[...]`` pytest-parametrize suffix — are
    # matched byte-for-byte. Two consequences are pinned here deliberately:
    #
    # 1. A parametrize ID that drifts year-to-year (a test parametrized on
    #    ``datetime.now().year``) would land in ``unknown`` if the dataset row and
    #    the harness run disagree on the year. This is the theoretical drift the
    #    issue describes. It is UPSTREAM-owned and diagnostic-only: ``native_score``
    #    reads the pinned grader's ``eval_results.json`` verdict, not this function.
    #    It also does not manifest in the pinned dataset — the one ``datetime.now()``
    #    test (openlibrary ``test_future_publication_dates_are_deleted``) has only
    #    its two FIXED sentinel params recorded (``[2000-11-11-True]`` /
    #    ``[9999-01-01-False]``); the drifting params were never captured.
    #
    # 2. Year-bearing but DISTINCT params must stay distinct. The dataset is full of
    #    fixed years used as test data (publication years, Amazon Linux version
    #    tags, IPv6 octets). Stripping/normalizing a year suffix to paper over (1)
    #    would silently merge these — e.g. the real vuls instance below expects both
    #    ``Test_getAmazonLinuxVersion/2025`` and ``/2027`` — so normalization is
    #    rejected. This assertion is the tripwire against re-introducing it.
    raw_sample = {
        "fail_to_pass": [
            "Test_getAmazonLinuxVersion/2025",
            "Test_getAmazonLinuxVersion/2027",
        ],
        "pass_to_pass": [
            "test_add_book.py::TestNormalizeImportRecord::test_future_publication_dates_are_deleted[2000-11-11-True]",
        ],
    }
    output = {
        "tests": [
            # Distinct fixed-year params: both observed, matched independently.
            {"name": "Test_getAmazonLinuxVersion/2025", "status": "PASSED"},
            {"name": "Test_getAmazonLinuxVersion/2027", "status": "FAILED"},
            # The recorded stable sentinel param is observed verbatim -> matched.
            {
                "name": "test_add_book.py::TestNormalizeImportRecord::"
                "test_future_publication_dates_are_deleted[2000-11-11-True]",
                "status": "PASSED",
            },
            # A drifting datetime.now() param the harness happens to also run.
            # It is NOT in the expected list, so it is simply ignored (extra
            # observed tests never create ``unknown``).
            {
                "name": "test_add_book.py::TestNormalizeImportRecord::"
                "test_future_publication_dates_are_deleted[2026-True]",
                "status": "PASSED",
            },
        ]
    }

    tests_status = _build_tests_status(output, raw_sample)

    # Distinct year params are kept apart, not merged into one bucket.
    assert tests_status["FAIL_TO_PASS"]["success"] == ["Test_getAmazonLinuxVersion/2025"]
    assert tests_status["FAIL_TO_PASS"]["failure"] == ["Test_getAmazonLinuxVersion/2027"]
    assert tests_status["FAIL_TO_PASS"]["unknown"] == []
    # The stable recorded param matches; the extra drifting observed param does
    # not inflate ``unknown`` because only expected -> observed is checked.
    assert tests_status["PASS_TO_PASS"]["success"] == [
        "test_add_book.py::TestNormalizeImportRecord::test_future_publication_dates_are_deleted[2000-11-11-True]"
    ]
    assert tests_status["PASS_TO_PASS"]["unknown"] == []
    assert tests_status["unknown_count"] == 0


def test_swebench_pro_tests_status_year_drift_is_upstream_owned_not_normalized() -> None:
    # Companion to the guard above: if a drifting ``datetime.now().year`` param
    # WERE recorded in the dataset (year Y) and the harness ran a later year
    # (Y+1), exact matching lands it in ``unknown``. We pin this as the accepted
    # behavior — reconciling it here would mean diverging from the pinned upstream
    # grader that actually sets ``native_score``, breaking leaderboard parity.
    recorded_year = "2026"
    harness_year = "2027"
    base = (
        "openlibrary/catalog/add_book/tests/test_add_book.py::"
        "TestNormalizeImportRecord::test_future_publication_dates_are_deleted"
    )
    raw_sample = {"fail_to_pass": [], "pass_to_pass": [f"{base}[{recorded_year}-True]"]}
    output = {"tests": [{"name": f"{base}[{harness_year}-True]", "status": "PASSED"}]}

    tests_status = _build_tests_status(output, raw_sample)

    assert tests_status["PASS_TO_PASS"]["success"] == []
    assert tests_status["PASS_TO_PASS"]["unknown"] == [f"{base}[{recorded_year}-True]"]
    assert tests_status["unknown_count"] == 1


# ---------------------------------------------------------------------------
# filter_grader_test_hunks: strip diff hunks the grader would overwrite
# ---------------------------------------------------------------------------

_VALID_SHA = "a" * 40
_TEST_SHA = "b" * 40


def _diff_block(path: str, content_line: str = "+stub") -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        "@@ -0,0 +1,1 @@\n"
        f"{content_line}\n"
    )


class TestGraderTestFilePaths:
    def test_picks_up_files_from_before_repo_set_cmd(self) -> None:
        raw_sample = {
            "before_repo_set_cmd": (
                "git reset --hard " + _VALID_SHA + "\n"
                "git checkout " + _TEST_SHA + " -- model/criteria/criteria_test.go test/posts.js"
            ),
        }
        paths = _grader_test_file_paths(raw_sample)
        assert "model/criteria/criteria_test.go" in paths
        assert "test/posts.js" in paths

    def test_ignores_bare_test_names_in_selected_test_files(self) -> None:
        # ``["TestCriteria"]`` is a Go test name, not a path. Filtering on it
        # would risk wiping unrelated diff hunks. The path-shape guard rejects it.
        paths = _grader_test_file_paths({"selected_test_files_to_run": ["TestCriteria", "TestHosts"]})
        assert paths == set()

    def test_extracts_pytest_path_from_pipe_form(self) -> None:
        # SWE-bench Pro NodeBB FAIL_TO_PASS values look like:
        #   "test/posts.js | <test description>"
        raw = {
            "FAIL_TO_PASS": [
                "test/posts.js | something",
                "test/other.js::should_pass",
            ],
        }
        paths = _grader_test_file_paths(raw)
        assert "test/posts.js" in paths
        assert "test/other.js" in paths


class TestFilterGraderTestHunks:
    def test_drops_hunks_for_grader_files(self) -> None:
        raw = {
            "before_repo_set_cmd": "git checkout " + _TEST_SHA + " -- model/criteria/criteria_test.go",
        }
        patch = _diff_block("model/criteria/criteria.go", "+keep") + _diff_block(
            "model/criteria/criteria_test.go", "+drop"
        )
        filtered = filter_grader_test_hunks(patch, raw)
        assert "criteria.go" in filtered
        assert "criteria_test.go" not in filtered

    def test_returns_input_when_no_matches(self) -> None:
        raw = {"before_repo_set_cmd": "git checkout " + _TEST_SHA + " -- model/elsewhere_test.go"}
        patch = _diff_block("src/foo.go", "+keep")
        assert filter_grader_test_hunks(patch, raw) == patch

    def test_handles_empty_inputs(self) -> None:
        assert filter_grader_test_hunks("", None) == ""
        assert filter_grader_test_hunks("", {"before_repo_set_cmd": ""}) == ""
        # Patch with no grader paths to filter against → unchanged.
        patch = _diff_block("a.go")
        assert filter_grader_test_hunks(patch, None) == patch

    def test_drops_appendonlydir_node_bb_style(self) -> None:
        # NodeBB campaigns kept committing redis ``appendonlydir/*.aof`` files
        # alongside source edits. This filter only fires on grader-tracked
        # files, so it does NOT clean these up — flagged here so we don't
        # mistakenly assume otherwise. (See followups for a separate noise
        # filter at the workspace_diff layer.)
        raw = {"before_repo_set_cmd": "git checkout " + _TEST_SHA + " -- test/posts.js"}
        patch = _diff_block("appendonlydir/appendonly.aof.1.base.rdb")
        # No grader path matches → noise survives.
        assert filter_grader_test_hunks(patch, raw) == patch


def test_output_prefix_constant_matches_payload_and_readback():
    """The rename broke this once (issue #758): the patch payload sent
    prefix="kcsi" while the readback still looked for swarms_output.json.
    Pin both sides to the shared constant so they cannot diverge again."""
    import inspect

    from kcsi.benchmarks import swebench_pro as sp

    src = " ".join(inspect.getsource(sp).split())  # whitespace-normalized
    assert '"prefix": _SWE_OUTPUT_PREFIX' in src
    assert 'f"{_SWE_OUTPUT_PREFIX}_output.json"' in src
    assert sp._SWE_OUTPUT_PREFIX == "kcsi"


def _write_jsonl_sample(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "instance_id": "instance_demo__repo-123",
                "repo": "demo/repo",
                "fail_to_pass": "['tests.widget.test_fix']",
                "pass_to_pass": "['tests.widget.test_existing']",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_evaluator_retries_when_output_json_missing_then_succeeds(monkeypatch, tmp_path: Path) -> None:
    """An interrupted run (returncode 0 but no output.json) is transient (#962):
    the evaluator must re-run rather than silently scoring it. On the retry the
    run completes and the instance resolves."""
    repo_root = tmp_path / "eval"
    _write_eval_root(repo_root)
    raw_sample_path = tmp_path / "samples.jsonl"
    _write_jsonl_sample(raw_sample_path)

    calls = {"n": 0}

    def fake_run_eval(cmd, cwd, env, timeout):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        output_dir = Path(cmd[cmd.index("--output_dir") + 1])
        instance_dir = output_dir / "instance_demo__repo-123"
        instance_dir.mkdir(parents=True, exist_ok=True)
        if calls["n"] >= 2:  # second attempt completes
            (output_dir / "eval_results.json").write_text(
                json.dumps({"instance_demo__repo-123": True}), encoding="utf-8"
            )
            (instance_dir / "kcsi_output.json").write_text(
                json.dumps(
                    {
                        "tests": [
                            {"name": "tests.widget.test_fix", "status": "PASSED"},
                            {"name": "tests.widget.test_existing", "status": "PASSED"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
        else:  # first attempt: interrupted run — eval_results written, output.json NOT
            (output_dir / "eval_results.json").write_text(
                json.dumps({"instance_demo__repo-123": False}), encoding="utf-8"
            )
        return SimpleNamespace(returncode=0, stdout="", stderr=""), {}

    monkeypatch.setattr("kcsi.benchmarks.swebench_pro._run_eval_command", fake_run_eval)
    evaluator = SwebenchProEvaluator(
        raw_sample_path=str(raw_sample_path), repo_root=str(repo_root), timeout_sec=30, max_eval_attempts=2
    )
    result = evaluator.evaluate(
        task=load_tasks_for_source(task_source="swebench_pro", tasks_path=raw_sample_path)[0],
        model_output="<patch>diff --git a/a b/a\n--- a/a\n+++ b/a\n@@\n-x\n+y\n</patch>",
    )

    assert calls["n"] == 2  # retried once
    assert result["swebench_status"] == "ok"
    assert result["resolved"] is True
    assert result["native_score"] == 1.0


def test_evaluator_fails_loud_when_output_json_persistently_missing(monkeypatch, tmp_path: Path) -> None:
    """If output.json is missing on every attempt, the evaluator must surface a
    retryable harness failure — never a silent scored 0 (resolved=False)."""
    repo_root = tmp_path / "eval"
    _write_eval_root(repo_root)
    raw_sample_path = tmp_path / "samples.jsonl"
    _write_jsonl_sample(raw_sample_path)

    calls = {"n": 0}

    def fake_run_eval(cmd, cwd, env, timeout):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        output_dir = Path(cmd[cmd.index("--output_dir") + 1])
        (output_dir / "eval_results.json").write_text(json.dumps({"instance_demo__repo-123": False}), encoding="utf-8")
        # never writes kcsi_output.json -> simulates an interrupted run every time
        return SimpleNamespace(returncode=0, stdout="ran tests", stderr=""), {}

    monkeypatch.setattr("kcsi.benchmarks.swebench_pro._run_eval_command", fake_run_eval)
    evaluator = SwebenchProEvaluator(
        raw_sample_path=str(raw_sample_path), repo_root=str(repo_root), timeout_sec=30, max_eval_attempts=3
    )
    result = evaluator.evaluate(
        task=load_tasks_for_source(task_source="swebench_pro", tasks_path=raw_sample_path)[0],
        model_output="<patch>diff --git a/a b/a\n--- a/a\n+++ b/a\n@@\n-x\n+y\n</patch>",
    )

    assert calls["n"] == 3  # exhausted all attempts
    assert result["swebench_status"] == "harness_failed"
    assert result.get("swebench_missing_output_json") is True
    # critically: NOT a silent resolved/ok
    assert result.get("resolved") is not True


def test_evaluator_surfaces_oom_kill_diagnostic(monkeypatch, tmp_path: Path, caplog) -> None:
    """The patched evaluator writes a kcsi_oom.json marker when Docker reports
    the container was OOM-killed by the mem_limit cap (per_instance_resource_limits.patch).
    Without this, an OOM kill is indistinguishable from a genuine test failure
    across every retry attempt. The flag must be diagnostic-only: it must not
    flip resolved/unresolved on its own."""
    repo_root = tmp_path / "eval"
    _write_eval_root(repo_root)
    raw_sample_path = tmp_path / "samples.jsonl"
    _write_jsonl_sample(raw_sample_path)

    def fake_run_eval(cmd, cwd, env, timeout):  # type: ignore[no-untyped-def]
        output_dir = Path(cmd[cmd.index("--output_dir") + 1])
        instance_dir = output_dir / "instance_demo__repo-123"
        instance_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "eval_results.json").write_text(json.dumps({"instance_demo__repo-123": False}), encoding="utf-8")
        # The container was OOM-killed before parser.py could write output.json.
        (instance_dir / "kcsi_oom.json").write_text(
            json.dumps({"oom_killed": True, "status_code": 137, "mem_limit": "8g"}), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="ran tests", stderr=""), {}

    monkeypatch.setattr("kcsi.benchmarks.swebench_pro._run_eval_command", fake_run_eval)
    evaluator = SwebenchProEvaluator(
        raw_sample_path=str(raw_sample_path), repo_root=str(repo_root), timeout_sec=30, max_eval_attempts=2
    )
    with caplog.at_level(logging.WARNING, logger="kcsi.benchmarks.swebench_pro"):
        result = evaluator.evaluate(
            task=load_tasks_for_source(task_source="swebench_pro", tasks_path=raw_sample_path)[0],
            model_output="<patch>diff --git a/a b/a\n--- a/a\n+++ b/a\n@@\n-x\n+y\n</patch>",
        )

    assert result["swebench_status"] == "harness_failed"
    assert result.get("oom_killed") is True
    # Diagnostic-only: still not a silent resolved/ok.
    assert result.get("resolved") is not True

    assert len(caplog.records) == 2  # retried once, marker present (and logged) both attempts
    for record in caplog.records:
        assert record.levelno == logging.WARNING
        assert "instance_demo__repo-123" in record.getMessage()
        assert "8g" in record.getMessage()


@pytest.mark.parametrize(
    ("resolved", "expected_status", "expected_score"),
    [
        (False, "oom_killed", None),
        (True, "ok", 1.0),
    ],
)
def test_evaluator_scores_completed_output_with_oom_marker_by_resolution(
    monkeypatch,
    tmp_path: Path,
    resolved: bool,
    expected_status: str,
    expected_score: float | None,
) -> None:
    repo_root = tmp_path / "eval"
    _write_eval_root(repo_root)
    raw_sample_path = tmp_path / "samples.jsonl"
    _write_jsonl_sample(raw_sample_path)

    def fake_run_eval(cmd, cwd, env, timeout):  # type: ignore[no-untyped-def]
        output_dir = Path(cmd[cmd.index("--output_dir") + 1])
        instance_dir = output_dir / "instance_demo__repo-123"
        instance_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "eval_results.json").write_text(
            json.dumps({"instance_demo__repo-123": resolved}), encoding="utf-8"
        )
        (instance_dir / "kcsi_output.json").write_text(
            json.dumps(
                {
                    "tests": [
                        {"name": "tests.widget.test_fix", "status": "PASSED" if resolved else "FAILED"},
                        {"name": "tests.widget.test_existing", "status": "PASSED"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        (instance_dir / "kcsi_oom.json").write_text(
            json.dumps({"oom_killed": True, "status_code": 137, "mem_limit": "8g"}), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="ran tests", stderr=""), {}

    monkeypatch.setattr("kcsi.benchmarks.swebench_pro._run_eval_command", fake_run_eval)
    evaluator = SwebenchProEvaluator(
        raw_sample_path=str(raw_sample_path), repo_root=str(repo_root), timeout_sec=30, max_eval_attempts=1
    )

    result = evaluator.evaluate(
        task=load_tasks_for_source(task_source="swebench_pro", tasks_path=raw_sample_path)[0],
        model_output="<patch>diff --git a/a b/a\n--- a/a\n+++ b/a\n@@\n-x\n+y\n</patch>",
    )

    assert result["swebench_status"] == expected_status
    assert result.get("oom_killed") is True
    assert result.get("native_score") == expected_score
    if not resolved:
        assert "resolved" not in result


# --- Regression tests: broken-submodule diff capture + tool-trace recovery ---
# Root cause: a SWE-bench Pro instance whose git submodule failed to clone
# (private repo / dead git:// URL / timeout) is left with a broken gitlink, so a
# plain ``git diff HEAD`` exits 128 and the agent's real edits are silently lost
# (scored ``no_patch``). See fix/swebench-diff-capture-submodule.


def _init_repo_with_broken_submodule(root: Path) -> Path:
    """Build a real git repo containing a broken submodule gitlink plus an
    unstaged edit to a tracked file, mirroring the failed-clone on-disk state."""
    sub = root / "sub_src"
    sub.mkdir()
    _git(sub, "init", "-q")
    _git(sub, "config", "user.email", "t@t")
    _git(sub, "config", "user.name", "t")
    (sub / "s.txt").write_text("hi\n", encoding="utf-8")
    _git(sub, "add", "-A")
    _git(sub, "commit", "-qm", "init")

    repo = root / "super"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "tracked.py").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(sub), "sub")
    _git(repo, "commit", "-qm", "addsub")
    # Break the submodule gitlink the way a failed clone leaves it.
    (repo / "sub" / ".git").write_text("gitdir: /nonexistent/broken/path\n", encoding="utf-8")
    # Agent edit to a tracked file that must survive capture.
    (repo / "tracked.py").write_text("base\nAGENT_EDIT_LINE\n", encoding="utf-8")
    return repo


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True)


def test_git_workspace_diff_ignores_broken_submodule(tmp_path: Path) -> None:
    repo = _init_repo_with_broken_submodule(tmp_path)
    diff, error = _git_workspace_diff(repo)
    # Before the --ignore-submodules=all fix this returned "" (git diff exited
    # 128 on the broken gitlink), silently dropping the agent's edit.
    assert "diff --git a/tracked.py b/tracked.py" in diff
    assert "AGENT_EDIT_LINE" in diff
    assert error is None


def test_patch_from_tool_trace_never_recovers_unverified_shell_output() -> None:
    real_diff = (
        "diff --git a/mod.py b/mod.py\n"
        "index 2d6e840..0f71272 100644\n"
        "--- a/mod.py\n+++ b/mod.py\n@@ -1 +1,2 @@\n x\n+edit\n"
    )
    trace = [
        {"tool_name": "apply_patch", "tool_output": json.dumps({"status": "completed"})},
        {"tool_name": "shell", "tool_output": json.dumps([{"stdout": real_diff}])},
    ]
    patch, edits = _patch_from_tool_trace({"tool_trace": trace})
    assert patch is None
    assert edits == 1  # the apply_patch call


def test_patch_from_tool_trace_counts_edits_without_recoverable_diff() -> None:
    trace = [
        {"tool_name": "apply_patch", "tool_output": json.dumps({"status": "completed"})},
        {"tool_name": "shell", "tool_output": json.dumps([{"stdout": "ran tests\n"}])},
    ]
    patch, edits = _patch_from_tool_trace({"tool_trace": trace})
    assert patch is None
    assert edits == 1


def _fake_grader_resolved(instance_id: str):
    def fake_run_eval(cmd, cwd, env, timeout):  # type: ignore[no-untyped-def]
        output_dir = Path(cmd[cmd.index("--output_dir") + 1])
        instance_dir = output_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "eval_results.json").write_text(json.dumps({instance_id: True}), encoding="utf-8")
        (instance_dir / "kcsi_output.json").write_text(json.dumps({"tests": []}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr=""), {}

    return fake_run_eval


def test_evaluator_refuses_unverified_patch_from_tool_trace(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "SWE-bench_Pro-os"
    _write_eval_root(repo_root)
    raw_sample_path = tmp_path / "samples.jsonl"
    raw_sample_path.write_text(
        json.dumps({"instance_id": "instance_demo__repo-123", "repo": "demo/repo"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kcsi.benchmarks.swebench_pro._run_eval_command",
        _fake_grader_resolved("instance_demo__repo-123"),
    )
    recovered = "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-x\n+y\n"
    trace = [
        {"tool_name": "apply_patch", "tool_output": json.dumps({"status": "completed"})},
        {"tool_name": "shell", "tool_output": json.dumps([{"stdout": recovered}])},
    ]
    result = SwebenchProEvaluator(
        raw_sample_path=str(raw_sample_path), repo_root=str(repo_root), timeout_sec=30
    ).evaluate(
        # Tool output is not a canonical patch source, even when it looks like
        # a valid diff.
        task=load_tasks_for_source(task_source="swebench_pro", tasks_path=raw_sample_path)[0],
        model_output="I edited the file.",
        runtime_meta={"workspace_diff": ""},
        tool_trace=trace,
    )
    assert result["swebench_status"] == "capture_failed"


def test_evaluator_capture_failed_when_edits_but_no_recoverable_patch(tmp_path: Path) -> None:
    raw_sample_path = tmp_path / "samples.jsonl"
    raw_sample_path.write_text(
        json.dumps({"instance_id": "instance_demo__repo-123", "repo": "demo/repo"}) + "\n",
        encoding="utf-8",
    )
    trace = [{"tool_name": "apply_patch", "tool_output": json.dumps({"status": "completed"})}]
    result = SwebenchProEvaluator(raw_sample_path=str(raw_sample_path)).evaluate(
        task=TaskSpec(id="instance_demo__repo-123"),
        model_output="",
        runtime_meta={"workspace_diff": ""},
        tool_trace=trace,
    )
    assert result["swebench_status"] == "capture_failed"
    assert result["swebench_capture_failed_edit_calls"] == 1
    # capture_failed is an unscored infra failure (None), like no_patch.
    assert "capture_failed" in SWEBENCH_FAILURE_STATUSES


def test_evaluator_no_patch_when_no_edits(tmp_path: Path) -> None:
    raw_sample_path = tmp_path / "samples.jsonl"
    raw_sample_path.write_text(
        json.dumps({"instance_id": "instance_demo__repo-123", "repo": "demo/repo"}) + "\n",
        encoding="utf-8",
    )
    # Genuine no-submission: no workspace_diff, no model patch, no edit tools.
    result = SwebenchProEvaluator(raw_sample_path=str(raw_sample_path)).evaluate(
        task=TaskSpec(id="instance_demo__repo-123"),
        model_output="I could not solve it.",
        runtime_meta={"workspace_diff": ""},
        tool_trace=[{"tool_name": "shell", "tool_output": json.dumps([{"stdout": "ls\n"}])}],
    )
    assert result["swebench_status"] == "no_patch"


def test_evaluator_capture_failure_is_not_reported_as_no_patch(tmp_path: Path) -> None:
    raw_sample_path = tmp_path / "samples.jsonl"
    raw_sample_path.write_text(json.dumps({"instance_id": "instance_demo__repo-123", "repo": "demo/repo"}) + "\n")
    result = SwebenchProEvaluator(raw_sample_path=str(raw_sample_path)).evaluate(
        task=TaskSpec(id="instance_demo__repo-123"),
        model_output="",
        runtime_meta={"workspace_diff": "", "workspace_diff_capture_error": "git diff failed"},
        tool_trace=[],
    )
    assert result["swebench_status"] == "capture_failed"

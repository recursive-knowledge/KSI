from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest
from conftest import REPO_ROOT

from kcsi.benchmarks.swebench_pro_external import EVALUATOR_REVISION, REVISION_MARKER

WRAPPER = REPO_ROOT / "benchmarks" / "scripts" / "run_swebench_pro_eval.py"


def _load_wrapper():
    spec = importlib.util.spec_from_file_location("run_swebench_pro_eval_under_test", WRAPPER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_eval_root(path: Path) -> None:
    (path / "run_scripts").mkdir(parents=True)
    (path / "swe_bench_pro_eval.py").write_text("# stub\n", encoding="utf-8")
    (path / REVISION_MARKER).write_text(EVALUATOR_REVISION + "\n", encoding="utf-8")


def _write_raw_sample(path: Path, instance_id: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["instance_id"])
        writer.writeheader()
        writer.writerow({"instance_id": instance_id})


def test_source_dir_controls_default_scripts_dir_for_underscore_flags(tmp_path: Path) -> None:
    module = _load_wrapper()
    eval_root = tmp_path / "custom_swebench_pro"
    _write_eval_root(eval_root)
    raw_sample = tmp_path / "samples.csv"
    _write_raw_sample(raw_sample, "instance-a")
    patch_file = tmp_path / "patches.json"
    patch_file.write_text(
        json.dumps([{"instance_id": "instance-a", "patch": "diff --git a/a b/a\n"}]),
        encoding="utf-8",
    )

    args = module.parse_args(
        [
            "--patch_path",
            str(patch_file),
            "--output_dir",
            str(tmp_path / "out"),
            "--source_dir",
            str(eval_root),
            "--raw_sample_path",
            str(raw_sample),
        ]
    )

    cmd, cwd = module.build_eval_command(args)

    assert cwd == eval_root.resolve()
    assert cmd[cmd.index("--scripts_dir") + 1] == str((eval_root / "run_scripts").resolve())


def test_missing_explicit_scripts_dir_falls_back_to_resolved_eval_root(monkeypatch, tmp_path: Path) -> None:
    module = _load_wrapper()
    fallback_eval_root = tmp_path / "third_party_swebench_pro"
    _write_eval_root(fallback_eval_root)
    monkeypatch.setattr(module, "DEFAULT_EVAL_ROOT", fallback_eval_root)
    missing_source = tmp_path / "benchmarks" / "swebench_pro" / "source"
    raw_sample = tmp_path / "samples.csv"
    _write_raw_sample(raw_sample, "instance-a")
    patch_file = tmp_path / "patches.json"
    patch_file.write_text(
        json.dumps([{"instance_id": "instance-a", "patch": "diff --git a/a b/a\n"}]),
        encoding="utf-8",
    )

    args = module.parse_args(
        [
            "--patch-path",
            str(patch_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--source-dir",
            str(missing_source),
            "--raw-sample-path",
            str(raw_sample),
            "--scripts-dir",
            str(missing_source / "run_scripts"),
        ]
    )

    cmd, cwd = module.build_eval_command(args)

    assert cwd == fallback_eval_root.resolve()
    assert cmd[cmd.index("--scripts_dir") + 1] == str((fallback_eval_root / "run_scripts").resolve())


def test_patch_bundle_without_raw_sample_match_fails_before_launch(monkeypatch, tmp_path: Path) -> None:
    module = _load_wrapper()
    eval_root = tmp_path / "custom_swebench_pro"
    _write_eval_root(eval_root)
    raw_sample = tmp_path / "samples.csv"
    _write_raw_sample(raw_sample, "raw-instance")
    patch_file = tmp_path / "patches.json"
    patch_file.write_text(
        json.dumps([{"instance_id": "other-instance", "patch": "diff --git a/a b/a\n"}]),
        encoding="utf-8",
    )
    calls = []

    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((args, kwargs))
        raise AssertionError("official evaluator should not be launched")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="no entries matching raw sample"):
        module.main(
            [
                "--patch-path",
                str(patch_file),
                "--output-dir",
                str(tmp_path / "out"),
                "--source-dir",
                str(eval_root),
                "--raw-sample-path",
                str(raw_sample),
            ]
        )

    assert calls == []


def test_wrong_evaluator_revision_fails_before_launch(tmp_path: Path) -> None:
    module = _load_wrapper()
    eval_root = tmp_path / "custom_swebench_pro"
    _write_eval_root(eval_root)
    (eval_root / REVISION_MARKER).write_text("wrong\n", encoding="utf-8")
    raw_sample = tmp_path / "samples.csv"
    _write_raw_sample(raw_sample, "instance-a")
    patch_file = tmp_path / "patches.json"
    patch_file.write_text(
        json.dumps([{"instance_id": "instance-a", "patch": "diff --git a/a b/a\n"}]),
        encoding="utf-8",
    )

    args = module.parse_args(
        [
            "--patch-path",
            str(patch_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--source-dir",
            str(eval_root),
            "--raw-sample-path",
            str(raw_sample),
        ]
    )

    with pytest.raises(RuntimeError, match="expected"):
        module.build_eval_command(args)

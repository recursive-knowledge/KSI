from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import pytest

from benchmarks.scripts.dataprep import export_swebench_pro_dataset as export_module
from benchmarks.scripts.dataprep import generate_kt_recipient_subsets as kt_module
from benchmarks.scripts.dataprep import generate_swebench_pro_task_map as task_map_module
from benchmarks.scripts.dataprep.generate_swebench_pro_task_map import build_task_map
from ksi.benchmarks.swebench_pro_external import DATASET_REVISION


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_build_task_map_is_deterministic_from_sorted_instance_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.jsonl"
    rows = [
        {"instance_id": "task-c", "repo": "org/c", "base_commit": "ccc"},
        {"instance_id": "task-a", "repo": "org/a", "base_commit": "aaa"},
        {"instance_id": "task-b", "repo": "org/b", "base_commit": "bbb"},
    ]
    _write_jsonl(dataset_path, rows)

    first = build_task_map(
        dataset_path=dataset_path,
        dataset_name="ScaleAI/SWE-bench_Pro",
        split="test",
        selection_name="demo",
        seed=7,
        count=2,
    )
    second = build_task_map(
        dataset_path=dataset_path,
        dataset_name="ScaleAI/SWE-bench_Pro",
        split="test",
        selection_name="demo",
        seed=7,
        count=2,
    )

    assert first == second
    assert first["benchmark"] == "swebench_pro"
    assert first["task_count"] == 2
    assert len(first["tasks"]) == 2
    assert all("task_id" in row for row in first["tasks"])
    assert first["source_path"] == dataset_path.as_posix()
    assert len(first["source_sha256"]) == 64


def test_build_task_map_keeps_repo_and_commit_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "instance_id": "instance_demo__repo-123",
                "repo": "demo/repo",
                "base_commit": "deadbeef",
            },
        ],
    )

    task_map = build_task_map(
        dataset_path=dataset_path,
        dataset_name="ScaleAI/SWE-bench_Pro",
        split="test",
        selection_name="demo",
        seed=0,
        count=1,
    )

    assert task_map["tasks"] == [
        {
            "index": 1,
            "task_id": "instance_demo__repo-123",
            "repo": "demo/repo",
            "base_commit": "deadbeef",
            "notes": "",
        }
    ]


def test_build_task_map_records_source_revision(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.jsonl"
    _write_jsonl(dataset_path, [{"instance_id": "task-a", "repo": "org/a", "base_commit": "aaa"}])

    with_revision = build_task_map(
        dataset_path=dataset_path,
        dataset_name="ScaleAI/SWE-bench_Pro",
        split="test",
        selection_name="demo",
        seed=0,
        count=1,
        source_revision="deadbeefrev",
    )
    assert with_revision["source_revision"] == "deadbeefrev"

    without_revision = build_task_map(
        dataset_path=dataset_path,
        dataset_name="ScaleAI/SWE-bench_Pro",
        split="test",
        selection_name="demo",
        seed=0,
        count=1,
    )
    assert without_revision["source_revision"] is None


def test_export_cli_defaults_to_pinned_dataset_revision(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_export_dataset(**kwargs):
        captured.update(kwargs)
        return 1

    monkeypatch.setattr(export_module, "export_dataset", fake_export_dataset)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_swebench_pro_dataset.py",
            "--output",
            str(tmp_path / "test.jsonl"),
        ],
    )

    export_module.main()

    assert captured["revision"] == DATASET_REVISION


def test_generate_task_map_cli_defaults_to_pinned_source_revision(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.jsonl"
    _write_jsonl(dataset_path, [{"instance_id": "task-a", "repo": "org/a", "base_commit": "aaa"}])
    output = tmp_path / "map.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_swebench_pro_task_map.py",
            "--dataset-path",
            str(dataset_path),
            "--selection-name",
            "demo",
            "--seed",
            "0",
            "--count",
            "1",
            "--output",
            str(output),
        ],
    )

    task_map_module.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["source_revision"] == DATASET_REVISION


def test_verify_swebench_task_map_source_matching_dataset_never_raises(
    tmp_path: Path,
) -> None:
    from ksi.cli import _verify_swebench_task_map_source

    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text('{"instance_id": "task-a"}\n', encoding="utf-8")
    original_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    ids_file = tmp_path / "map.json"
    ids_file.write_text(
        json.dumps({"source_sha256": original_sha, "tasks": [{"task_id": "task-a"}]}),
        encoding="utf-8",
    )

    # Matching dataset: no raise in either mode.
    _verify_swebench_task_map_source(str(ids_file), dataset_path)
    _verify_swebench_task_map_source(str(ids_file), dataset_path, strict=True)


def test_verify_swebench_task_map_source_default_warns_does_not_raise(tmp_path: Path, caplog) -> None:
    from ksi.cli import _verify_swebench_task_map_source

    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text('{"instance_id": "task-a"}\n', encoding="utf-8")
    original_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    ids_file = tmp_path / "map.json"
    ids_file.write_text(
        json.dumps({"source_sha256": original_sha, "tasks": [{"task_id": "task-a"}]}),
        encoding="utf-8",
    )

    # One-byte mutation → drift. Default (warn-mode) must NOT raise.
    dataset_path.write_text('{"instance_id": "task-b"}\n', encoding="utf-8")
    mutated_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert mutated_sha != original_sha

    with caplog.at_level(logging.WARNING, logger="ksi.cli"):
        _verify_swebench_task_map_source(str(ids_file), dataset_path)  # no strict

    warning_text = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert original_sha in warning_text
    assert mutated_sha in warning_text


def test_verify_swebench_task_map_source_strict_fails_closed(tmp_path: Path) -> None:
    from ksi.cli import _verify_swebench_task_map_source

    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text('{"instance_id": "task-a"}\n', encoding="utf-8")
    original_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    ids_file = tmp_path / "map.json"
    ids_file.write_text(
        json.dumps({"source_sha256": original_sha, "tasks": [{"task_id": "task-a"}]}),
        encoding="utf-8",
    )

    # One-byte mutation: strict mode fails closed, naming both hashes.
    dataset_path.write_text('{"instance_id": "task-b"}\n', encoding="utf-8")
    mutated_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert mutated_sha != original_sha

    with pytest.raises(SystemExit) as excinfo:
        _verify_swebench_task_map_source(str(ids_file), dataset_path, strict=True)
    message = str(excinfo.value)
    assert original_sha in message
    assert mutated_sha in message


def test_verify_swebench_task_map_source_pinned_revision_fails_closed_by_default(tmp_path: Path) -> None:
    from ksi.cli import _verify_swebench_task_map_source

    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text('{"instance_id": "task-a"}\n', encoding="utf-8")
    original_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    ids_file = tmp_path / "map.json"
    ids_file.write_text(
        json.dumps(
            {
                "source_revision": DATASET_REVISION,
                "source_sha256": original_sha,
                "tasks": [{"task_id": "task-a"}],
            }
        ),
        encoding="utf-8",
    )

    dataset_path.write_text('{"instance_id": "task-b"}\n', encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _verify_swebench_task_map_source(str(ids_file), dataset_path)
    assert DATASET_REVISION in str(excinfo.value)


def test_verify_swebench_task_map_source_strict_rejects_missing_sha(tmp_path: Path) -> None:
    from ksi.cli import _verify_swebench_task_map_source

    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text('{"instance_id": "task-a"}\n', encoding="utf-8")
    ids_file = tmp_path / "map.json"
    ids_file.write_text(json.dumps({"tasks": [{"task_id": "task-a"}]}), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _verify_swebench_task_map_source(str(ids_file), dataset_path, strict=True)

    assert "does not record source_sha256" in str(excinfo.value)


def test_kt_swebench_recipient_records_dataset_digest_and_revision(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {"instance_id": "task-a"},
            {"instance_id": "task-b"},
            {"instance_id": "task-c"},
        ],
    )
    baseline_map = tmp_path / "baseline.json"
    baseline_map.write_text(
        json.dumps(
            {
                "source_revision": DATASET_REVISION,
                "tasks": [{"task_id": "task-a"}],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "recipient.json"

    kt_module.build_swebench_pro_recipient(
        dataset_jsonl=dataset_path,
        baseline_map=baseline_map,
        output=output,
        seed=1,
        count=1,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["source_revision"] == DATASET_REVISION
    assert payload["source_sha256"] == hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert payload["tasks"][0]["task_id"] in {"task-b", "task-c"}


def test_source_path_is_relative_to_repo_root_not_cwd(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    dataset_path = repo_root / "benchmarks" / "swebench_pro" / "dataset" / "test.jsonl"
    dataset_path.parent.mkdir(parents=True)
    _write_jsonl(dataset_path, [{"instance_id": "task-a", "repo": "org/a", "base_commit": "aaa"}])
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()

    monkeypatch.setattr(task_map_module, "REPO_ROOT", repo_root)
    monkeypatch.chdir(other_cwd)

    task_map = task_map_module.build_task_map(
        dataset_path=dataset_path,
        dataset_name="ScaleAI/SWE-bench_Pro",
        split="test",
        selection_name="demo",
        seed=0,
        count=1,
    )

    assert task_map["source_path"] == "benchmarks/swebench_pro/dataset/test.jsonl"


def test_build_task_map_hashes_the_same_snapshot_it_loaded(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.jsonl"
    original = b'{"instance_id": "task-a", "repo": "org/a", "base_commit": "aaa"}\n'
    mutated = b'{"instance_id": "task-b", "repo": "org/b", "base_commit": "bbb"}\n'
    dataset_path.write_bytes(original)

    def fake_loader(path: Path, data: bytes) -> list[dict]:
        assert path == dataset_path
        assert data == original
        dataset_path.write_bytes(mutated)
        return [{"instance_id": "task-a", "repo": "org/a", "base_commit": "aaa"}]

    monkeypatch.setattr(task_map_module, "_load_rows_from_bytes", fake_loader)

    task_map = build_task_map(
        dataset_path=dataset_path,
        dataset_name="ScaleAI/SWE-bench_Pro",
        split="test",
        selection_name="demo",
        seed=0,
        count=1,
    )

    assert task_map["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert task_map["tasks"][0]["task_id"] == "task-a"

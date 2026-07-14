"""Tests for benchmarks/scripts/dataprep/validate_task_map.py --check-sources.

Covers the optional source-file verification path added alongside the
ARC prep work. The base ID-validation path is exercised indirectly via
existing CI / build scripts, so these tests focus on the new flag.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "benchmarks" / "scripts" / "dataprep" / "validate_task_map.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_task_map", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_arc_task(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "train": [{"input": [[0]], "output": [[1]]}],
                "test": [{"input": [[2]], "output": [[3]]}],
            }
        ),
        encoding="utf-8",
    )


def _write_task_map(path: Path, tasks: list[dict], *, extra: dict | None = None) -> None:
    payload = {
        "benchmark": "arc1",
        "split": "evaluation",
        "seed": 0,
        "count": len(tasks),
        "selection_name": "arc1_test",
        "tasks": tasks,
    }
    if extra:
        payload.update(extra)
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_check_sources_reports_ok_for_valid_map(tmp_path):
    source_dir = tmp_path / "source"
    source_file = source_dir / "abc123.json"
    _write_arc_task(source_file)

    task_map = tmp_path / "task_map.json"
    _write_task_map(
        task_map,
        [{"index": 1, "task_id": "abc123", "source_file": str(source_file)}],
    )

    proc = _run(
        [
            "--task-map",
            str(task_map),
            "--task-source",
            "arc",
            "--tasks-path",
            str(source_dir),
            "--check-sources",
        ]
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    summary = json.loads(proc.stdout)
    assert summary["sources"]["checked"] == 1
    assert summary["sources"]["missing_count"] == 0
    assert summary["sources"]["malformed_count"] == 0


def test_require_provenance_rejects_arc_map_without_source_commit(tmp_path):
    source_dir = tmp_path / "source"
    source_file = source_dir / "abc123.json"
    _write_arc_task(source_file)

    task_map = tmp_path / "task_map.json"
    _write_task_map(
        task_map,
        [{"index": 1, "task_id": "abc123", "source_file": str(source_file)}],
    )

    proc = _run(
        [
            "--task-map",
            str(task_map),
            "--task-source",
            "arc",
            "--tasks-path",
            str(source_dir),
            "--require-provenance",
        ]
    )

    assert proc.returncode == 6, proc.stdout
    summary = json.loads(proc.stdout)
    fields = {item["field"] for item in summary["provenance"]["missing_or_invalid"]}
    assert {"source_repo", "source_branch", "source_commit", "source_path", "selection_algorithm"} <= fields


def test_require_provenance_accepts_arc_map_with_source_commit(tmp_path):
    source_dir = tmp_path / "source"
    source_file = source_dir / "abc123.json"
    _write_arc_task(source_file)

    task_map = tmp_path / "task_map.json"
    _write_task_map(
        task_map,
        [{"index": 1, "task_id": "abc123", "source_file": str(source_file)}],
        extra={
            "source_repo": "example/arc",
            "source_branch": "main",
            "source_commit": "0" * 40,
            "source_path": "data/evaluation",
            "selection_algorithm": "fixture",
        },
    )

    proc = _run(
        [
            "--task-map",
            str(task_map),
            "--task-source",
            "arc",
            "--tasks-path",
            str(source_dir),
            "--require-provenance",
        ]
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    summary = json.loads(proc.stdout)
    assert summary["provenance"]["missing_or_invalid"] == []


def test_committed_arc_task_maps_have_required_provenance():
    mod = _load_module()
    paths = sorted((REPO_ROOT / "benchmarks").glob("arc[12]/task_maps/*.json"))
    assert paths
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        errors = mod._required_provenance_errors(payload, task_source="arc")
        assert errors == [], f"{path} provenance errors: {errors}"


def test_check_sources_detects_missing_source_file(tmp_path):
    """Exit 4 must fire when all task_ids are known but a source_file is gone.

    The earlier shape of this test let both "abc123" and "ghost" pass through,
    but ghost.json didn't exist in source_dir — so the ID-listing check fired
    exit 2 first and exit 4 was never exercised. Here we put all IDs into
    source_dir (so ID listing is green) and point the task map's source_file
    for "ghost" to an unrelated absolute path that doesn't exist on disk.
    """
    source_dir = tmp_path / "source"
    present = source_dir / "abc123.json"
    _write_arc_task(present)
    # All task_ids must resolve against the ID listing so exit 2 doesn't shadow us.
    ghost_in_listing = source_dir / "ghost.json"
    _write_arc_task(ghost_in_listing)

    # The task map's source_file for "ghost" points somewhere else entirely.
    missing_source = tmp_path / "not-on-disk" / "ghost.json"
    task_map = tmp_path / "task_map.json"
    _write_task_map(
        task_map,
        [
            {"index": 1, "task_id": "abc123", "source_file": str(present)},
            {"index": 2, "task_id": "ghost", "source_file": str(missing_source)},
        ],
    )

    proc = _run(
        [
            "--task-map",
            str(task_map),
            "--task-source",
            "arc",
            "--tasks-path",
            str(source_dir),
            "--check-sources",
        ]
    )
    assert proc.returncode == 4, f"rc={proc.returncode} out={proc.stdout}"
    summary = json.loads(proc.stdout)
    assert summary["sources"]["missing_count"] == 1
    assert summary["sources"]["missing"][0]["task_id"] == "ghost"


def test_check_sources_detects_malformed_json(tmp_path):
    source_dir = tmp_path / "source"
    source_file = source_dir / "abc123.json"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("{not valid json", encoding="utf-8")

    task_map = tmp_path / "task_map.json"
    _write_task_map(
        task_map,
        [{"index": 1, "task_id": "abc123", "source_file": str(source_file)}],
    )

    proc = _run(
        [
            "--task-map",
            str(task_map),
            "--task-source",
            "arc",
            "--tasks-path",
            str(source_dir),
            "--check-sources",
        ]
    )
    assert proc.returncode == 4, f"rc={proc.returncode} out={proc.stdout}"
    summary = json.loads(proc.stdout)
    assert summary["sources"]["malformed_count"] == 1
    assert "invalid JSON" in summary["sources"]["malformed"][0]["error"]


def test_check_sources_detects_missing_train_test(tmp_path):
    source_dir = tmp_path / "source"
    source_file = source_dir / "abc123.json"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    # Valid JSON but missing train/test keys.
    source_file.write_text(json.dumps({"some_other_key": []}), encoding="utf-8")

    task_map = tmp_path / "task_map.json"
    _write_task_map(
        task_map,
        [{"index": 1, "task_id": "abc123", "source_file": str(source_file)}],
    )

    proc = _run(
        [
            "--task-map",
            str(task_map),
            "--task-source",
            "arc",
            "--tasks-path",
            str(source_dir),
            "--check-sources",
        ]
    )
    assert proc.returncode == 4, f"rc={proc.returncode} out={proc.stdout}"
    summary = json.loads(proc.stdout)
    assert summary["sources"]["malformed_count"] == 1
    assert "train/test" in summary["sources"]["malformed"][0]["error"]


def test_check_sources_rejects_swebench(tmp_path):
    task_map = tmp_path / "task_map.json"
    task_map.write_text(
        json.dumps({"task_ids": ["django__django-1"]}),
        encoding="utf-8",
    )
    bogus_parquet = tmp_path / "bogus.parquet"
    bogus_parquet.write_bytes(b"")

    proc = _run(
        [
            "--task-map",
            str(task_map),
            "--task-source",
            "swebench",
            "--tasks-path",
            str(bogus_parquet),
            "--check-sources",
        ]
    )
    assert proc.returncode != 0
    # The swebench rejection message goes to stderr via SystemExit.
    assert "--check-sources" in (proc.stderr + proc.stdout)


def test_check_source_files_helper_direct(tmp_path):
    """Exercise the _check_source_files helper without subprocess."""
    mod = _load_module()
    present = tmp_path / "ok.json"
    _write_arc_task(present)
    missing = tmp_path / "nope.json"
    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")

    report = mod._check_source_files(
        [
            {"task_id": "ok", "source_file": str(present)},
            {"task_id": "missing", "source_file": str(missing)},
            {"task_id": "bad", "source_file": str(malformed)},
            {"task_id": "no_field"},  # missing source_file entirely
        ]
    )
    # checked counts only fully verified entries (resolved + parseable + shaped).
    assert report["checked"] == 1
    assert report["missing_count"] == 1
    # no_field (missing source_file meta) + malformed JSON = 2 malformed entries.
    assert report["malformed_count"] == 2
    missing_ids = {item["task_id"] for item in report["missing"]}
    assert "missing" in missing_ids


def test_check_sources_rejects_relative_traversal(tmp_path):
    """A relative source_file that escapes REPO_ROOT (`..` segments) is refused.

    Absolute paths outside the repo are trusted by intent, but relative paths
    must stay within the repository tree to prevent task maps of unknown origin
    from silently reading arbitrary files.
    """
    mod = _load_module()
    report = mod._check_source_files([{"task_id": "esc", "source_file": "../../../../etc/passwd"}])
    assert report["checked"] == 0
    assert report["malformed_count"] == 1
    assert "escapes repository root" in report["malformed"][0]["error"]


def test_check_sources_rejects_non_utf8(tmp_path):
    """Binary/non-UTF-8 source files are reported as malformed, not crashed on."""
    mod = _load_module()
    source_file = tmp_path / "bad_bytes.json"
    source_file.write_bytes(b"\xff\xfe not utf-8 at all \xc3\x28")

    report = mod._check_source_files([{"task_id": "bad", "source_file": str(source_file)}])
    assert report["checked"] == 0
    assert report["malformed_count"] == 1
    assert "UTF-8" in report["malformed"][0]["error"]


def test_swebench_jsonl_validation_hashes_tasks_path_not_source_path(tmp_path):
    original = tmp_path / "original.jsonl"
    drifted = tmp_path / "drifted.jsonl"
    _write_jsonl(original, [{"instance_id": "task-a", "payload": "original"}])
    _write_jsonl(drifted, [{"instance_id": "task-a", "payload": "drifted"}])
    task_map = tmp_path / "map.json"
    task_map.write_text(
        json.dumps(
            {
                "source_path": str(original),
                "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                "tasks": [{"task_id": "task-a"}],
            }
        ),
        encoding="utf-8",
    )

    proc = _run(
        [
            "--task-map",
            str(task_map),
            "--task-source",
            "swebench_pro",
            "--tasks-path",
            str(drifted),
        ]
    )

    assert proc.returncode == 5, proc.stdout
    summary = json.loads(proc.stdout)
    assert summary["missing_count"] == 0
    assert summary["source_path"] == str(original)
    assert summary["source_sha256_expected"] != summary["source_sha256_actual"]


def test_swebench_jsonl_and_csv_validation_accept_matching_sources(tmp_path):
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(jsonl, [{"instance_id": "task-a"}, {"instance_id": "task-b"}])
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("instance_id\n task-a\n task-b\n", encoding="utf-8")

    for source in (jsonl, csv_path):
        task_map = tmp_path / f"{source.stem}.json"
        task_map.write_text(
            json.dumps(
                {
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "tasks": [{"task_id": "task-a"}, {"task_id": "task-b"}],
                }
            ),
            encoding="utf-8",
        )
        proc = _run(
            [
                "--task-map",
                str(task_map),
                "--task-source",
                "swebench_pro",
                "--tasks-path",
                str(source),
            ]
        )
        assert proc.returncode == 0, proc.stderr or proc.stdout


def test_swebench_pinned_revision_requires_source_sha(tmp_path):
    source = tmp_path / "data.jsonl"
    _write_jsonl(source, [{"instance_id": "task-a"}])
    task_map = tmp_path / "map.json"
    task_map.write_text(
        json.dumps(
            {
                "source_revision": "7ab5114912baf22bb098818e604c02fe7ad2c11f",
                "tasks": [{"task_id": "task-a"}],
            }
        ),
        encoding="utf-8",
    )

    proc = _run(
        [
            "--task-map",
            str(task_map),
            "--task-source",
            "swebench_pro",
            "--tasks-path",
            str(source),
        ]
    )

    assert proc.returncode == 5
    summary = json.loads(proc.stdout)
    assert summary["source_sha256_missing"] is True


def test_swebench_unpinned_revision_marker_does_not_require_source_sha(tmp_path):
    source = tmp_path / "data.jsonl"
    _write_jsonl(source, [{"instance_id": "task-a"}])
    task_map = tmp_path / "map.json"
    task_map.write_text(
        json.dumps(
            {
                "source_revision": "none",
                "tasks": [{"task_id": "task-a"}],
            }
        ),
        encoding="utf-8",
    )

    proc = _run(
        [
            "--task-map",
            str(task_map),
            "--task-source",
            "swebench_pro",
            "--tasks-path",
            str(source),
        ]
    )

    assert proc.returncode == 0, proc.stdout
    summary = json.loads(proc.stdout)
    assert "source_sha256_missing" not in summary

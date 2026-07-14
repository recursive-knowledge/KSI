from __future__ import annotations

import json
from pathlib import Path

import pytest

from ksi.benchmarks.loaders import _resolve_tb2_source_root
from ksi.tasks.loaders import SUPPORTED_TASK_SOURCES, load_tasks_for_source


def test_terminal_bench_2_in_supported_sources() -> None:
    assert "terminal_bench_2" in SUPPORTED_TASK_SOURCES


def test_resolve_tb2_source_root_rejects_relative_traversal_escape(tmp_path: Path) -> None:
    # A task map of unknown origin must not point a RELATIVE source_path outside
    # the repo via `../..` traversal (mirrors validate_task_map's ARC guard).
    with pytest.raises(ValueError, match="escapes the repo"):
        _resolve_tb2_source_root(tmp_path / "map.json", {"source_path": "../../../../etc"})


def test_resolve_tb2_source_root_allows_explicit_absolute_path(tmp_path: Path) -> None:
    # An explicit absolute path is the caller's deliberate choice: containment
    # is not enforced, only existence.
    src = tmp_path / "tb2-src"
    src.mkdir()
    assert _resolve_tb2_source_root(tmp_path / "map.json", {"source_path": str(src)}) == src.resolve()


def _write_tb2_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "tb2-source"
    task_root = source_root / "demo-task"
    (task_root / "environment").mkdir(parents=True)
    (task_root / "solution").mkdir()
    (task_root / "tests").mkdir()
    (task_root / "instruction.md").write_text("Native task statement.\n", encoding="utf-8")
    (task_root / "task.toml").write_text(
        """\
version = "1.0"

[metadata]
author_name = "Example"
author_email = "example@example.com"
difficulty = "medium"
category = "system-administration"

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
    (task_root / "tests" / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")

    task_map = tmp_path / "tb2-map.json"
    task_map.write_text(
        json.dumps(
            {
                "selection_name": "tb2_demo",
                "benchmark": "terminal_bench_2",
                "dataset_name": "harbor-framework/terminal-bench-2",
                "source_path": str(source_root),
                "source_git_revision": "deadbeef",
                "task_count": 1,
                "tasks": [
                    {
                        "index": 1,
                        "task_id": "demo-task",
                        "difficulty": "medium",
                        "category": "system-administration",
                        "docker_image": "example/demo-task:latest",
                        "verifier_timeout_sec": 900.0,
                        "agent_timeout_sec": 1200.0,
                        "notes": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return task_map, task_root


def test_load_terminal_bench_2_task_map_preserves_native_files(tmp_path: Path) -> None:
    task_map, task_root = _write_tb2_fixture(tmp_path)
    tasks = load_tasks_for_source(task_source="terminal_bench_2", tasks_path=task_map)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.id == "demo-task"
    assert task.metadata["task_source"] == "terminal_bench_2"
    assert task.metadata["task_root"] == str(task_root.resolve())
    assert task.metadata["instruction_path"] == str((task_root / "instruction.md").resolve())
    assert task.metadata["task_toml_path"] == str((task_root / "task.toml").resolve())
    assert task.metadata["task_files"]["tb2/instruction.md"] == "Native task statement.\n"
    assert 'docker_image = "example/demo-task:latest"' in task.metadata["task_files"]["tb2/task.toml"]
    # Task-map timeouts are provenance only; the metadata marks them as such so a
    # reader does not mistake them for the runtime-authoritative task.toml values.
    assert task.metadata["agent_timeout_sec"] == 1200.0
    assert task.metadata["verifier_timeout_sec"] == 900.0
    assert task.metadata["timeout_source"] == "task_map"

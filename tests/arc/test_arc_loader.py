from __future__ import annotations

import json
from pathlib import Path

import pytest

from kcsi.tasks.loaders import load_tasks_for_source


def _write_arc_task(path: Path) -> None:
    payload = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]], "output": [[4]]}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_arc_tasks_sets_max_trials_metadata(tmp_path):
    arc_dir = tmp_path / "training"
    arc_dir.mkdir(parents=True, exist_ok=True)
    _write_arc_task(arc_dir / "abc123.json")

    tasks = load_tasks_for_source(
        task_source="arc",
        tasks_path=arc_dir,
        arc_max_trials=5,
    )
    assert len(tasks) == 1
    assert tasks[0].metadata["task_source"] == "arc"
    assert tasks[0].metadata["arc_max_trials"] == 5


def test_load_arc_tasks_clamps_max_trials_to_at_least_one(tmp_path):
    arc_dir = tmp_path / "training"
    arc_dir.mkdir(parents=True, exist_ok=True)
    _write_arc_task(arc_dir / "abc123.json")

    tasks = load_tasks_for_source(
        task_source="arc",
        tasks_path=arc_dir,
        arc_max_trials=0,
    )
    assert len(tasks) == 1
    assert tasks[0].metadata["arc_max_trials"] == 1


def test_load_arc_tasks_rejects_root_with_training_and_evaluation(tmp_path):
    arc_root = tmp_path / "arc_data"
    training_dir = arc_root / "training"
    evaluation_dir = arc_root / "evaluation"
    training_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    _write_arc_task(training_dir / "train1234.json")
    _write_arc_task(evaluation_dir / "eval1234.json")

    with pytest.raises(ValueError, match="contains multiple split directories"):
        load_tasks_for_source(
            task_source="arc",
            tasks_path=arc_root,
            arc_max_trials=2,
        )


def test_load_arc_tasks_renames_third_duplicate_stem(tmp_path):
    for dirname in ("a", "b", "c"):
        path = tmp_path / dirname
        path.mkdir(parents=True, exist_ok=True)
        _write_arc_task(path / "dup12345.json")

    tasks = load_tasks_for_source(
        task_source="arc",
        tasks_path=tmp_path,
        arc_max_trials=2,
    )
    ids = sorted(task.id for task in tasks)
    assert ids == ["arc__dup12345", "arc__dup12345__2", "dup12345"]

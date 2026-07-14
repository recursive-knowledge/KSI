#!/usr/bin/env python3
"""Create benchmark-safe ARC workspace payloads from a task map."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import sys
from pathlib import Path

# Allow `from arc_prep._common import ...` when this script is run directly.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from arc_prep._common import (
    load_json,
    require_field,
    save_json,
)
from arc_prep._common import (
    relative_to_manifest as _relative_to_manifest,
)
from arc_prep._common import (
    sanitize_task_id as _sanitize_task_id,
)
from arc_prep._common import (
    validate_grid as _validate_grid,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"


def default_output_dir(task_map_path: Path, benchmark: str) -> Path:
    return BENCHMARKS_DIR / "arc" / "workspace_payloads" / benchmark / task_map_path.stem


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def generate_payloads(task_map_path: Path, output_dir: Path | None) -> Path:
    task_map = load_json(task_map_path)
    if not isinstance(task_map, dict):
        raise ValueError("Task map must be a JSON object.")
    benchmark = task_map.get("benchmark")
    if benchmark not in {"arc1", "arc2"}:
        raise ValueError("Task map benchmark must be arc1 or arc2.")
    tasks = task_map.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Task map must contain a tasks list.")

    out_dir = output_dir or default_output_dir(task_map_path, benchmark)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_dir = out_dir

    manifest = {
        "kind": "payloads",
        "benchmark": benchmark,
        "task_map": _relative_to_manifest(task_map_path, manifest_dir),
        "count": 0,
        "payloads": [],
    }

    for task_idx, task in enumerate(tasks):
        task_origin = f"{task_map_path.name} tasks[{task_idx}]"
        task_id = require_field(task, "task_id", origin=task_origin)
        source_file_value = require_field(task, "source_file", origin=f"{task_origin} (task_id={task_id!r})")
        source_file = resolve_project_path(source_file_value)
        source_task = load_json(source_file)
        train = require_field(source_task, "train", origin=f"source {source_file}")
        test = require_field(source_task, "test", origin=f"source {source_file}")

        # Validate train pairs up-front so we fail fast with a clear origin.
        if not isinstance(train, list) or not train:
            raise ValueError(f"task_id={task_id}: source file {source_file} has no train pairs.")
        if not isinstance(test, list) or not test:
            raise ValueError(f"task {task_id!r} has empty or missing 'test' pairs in {source_file}")
        for train_index, train_pair in enumerate(train):
            if not isinstance(train_pair, dict):
                raise ValueError(f"task_id={task_id} train[{train_index}]: expected object with input/output.")
            _validate_grid(
                train_pair.get("input"),
                f"task_id={task_id} train[{train_index}].input",
            )
            _validate_grid(
                train_pair.get("output"),
                f"task_id={task_id} train[{train_index}].output",
            )

        for pair_index, pair in enumerate(test):
            if not isinstance(pair, dict):
                raise ValueError(
                    f"task_id={task_id} pair_index={pair_index}: test pair must be an object with an input."
                )
            test_input = require_field(pair, "input", origin=f"task_id={task_id} pair_index={pair_index}")
            _validate_grid(
                test_input,
                f"task_id={task_id} pair_index={pair_index} test_input",
            )
            payload = {
                "benchmark": benchmark,
                "task_id": task_id,
                "pair_index": pair_index,
                "train": train,
                "test_input": test_input,
            }
            file_name = f"{_sanitize_task_id(task_id)}_pair{pair_index}.json"
            payload_path = out_dir / file_name
            save_json(payload_path, payload)
            manifest["payloads"].append(
                {
                    "task_id": task_id,
                    "pair_index": pair_index,
                    "source_file": _relative_to_manifest(source_file, manifest_dir),
                    "payload_file": _relative_to_manifest(payload_path, manifest_dir),
                }
            )

    manifest["count"] = len(manifest["payloads"])
    save_json(manifest_path, manifest)
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare redacted ARC workspace payloads.")
    parser.add_argument("--task-map", type=Path, required=True, help="ARC task map JSON.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override payload output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = generate_payloads(args.task_map.resolve(), args.output_dir.resolve() if args.output_dir else None)
    print(f"Saved ARC workspace payloads to {out_dir}")


if __name__ == "__main__":
    main()

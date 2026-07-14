#!/usr/bin/env python3
"""Prepare the SWE-bench Pro repo cache used by swarm and baseline runners."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kcsi.tasks.loaders import load_tasks_for_source  # noqa: E402
from kcsi.tasks.repo_cache import prepare_swebench_repo_snapshots, validate_repo_cache_task_id  # noqa: E402

DEFAULT_TASKS_PATH = REPO_ROOT / "benchmarks" / "swebench_pro" / "dataset" / "test.jsonl"
DEFAULT_TASK_MAP = REPO_ROOT / "benchmarks" / "swebench_pro" / "task_maps" / "swebench_pro_test_50_seed0_v1.json"
DEFAULT_REPO_CACHE = REPO_ROOT / "benchmarks" / "swebench_pro" / "repo_cache"


def _validate_task_ids(task_ids: list[str], *, path: Path) -> list[str]:
    seen: set[str] = set()
    validated: list[str] = []
    for index, task_id in enumerate(task_ids):
        try:
            validated_id = validate_repo_cache_task_id(task_id)
        except ValueError as exc:
            raise ValueError(f"Invalid task ID at index {index} in {path}: {task_id!r}") from exc
        if validated_id in seen:
            raise ValueError(f"Duplicate task ID in {path}: {validated_id}")
        seen.add(validated_id)
        validated.append(validated_id)
    return validated


def _task_ids_from_task_objects(tasks: object, *, path: Path) -> list[str]:
    if not isinstance(tasks, list):
        raise ValueError(f"Expected 'tasks' to be a list in task map: {path}")
    task_ids: list[str] = []
    for index, item in enumerate(tasks):
        if not isinstance(item, dict):
            raise ValueError(f"Expected tasks[{index}] to be an object in task map: {path}")
        task_id = item.get("task_id")
        if not isinstance(task_id, str):
            raise ValueError(f"Expected tasks[{index}].task_id to be a string in task map: {path}")
        task_ids.append(task_id)
    return task_ids


def _load_task_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    task_ids: list[str]
    if isinstance(data, list):
        if not all(isinstance(item, str) for item in data):
            raise ValueError(f"Expected task map list entries to be strings: {path}")
        task_ids = data
        return _validate_task_ids(task_ids, path=path)
    if isinstance(data, dict):
        if "task_ids" in data:
            task_ids_value = data["task_ids"]
            if not isinstance(task_ids_value, list) or not all(isinstance(item, str) for item in task_ids_value):
                raise ValueError(f"Expected 'task_ids' to be a list of strings in task map: {path}")
            task_ids = task_ids_value
            return _validate_task_ids(task_ids, path=path)
        if "tasks" in data:
            task_ids = _task_ids_from_task_objects(data["tasks"], path=path)
            return _validate_task_ids(task_ids, path=path)
    raise ValueError(f"Unsupported task map format: {path}")


def prepare_repo_cache(
    *,
    tasks_path: Path,
    task_map: Path | None,
    repo_cache: Path,
    seed_test_files: bool = False,
) -> int:
    task_ids = _load_task_ids(task_map)
    tasks = load_tasks_for_source(task_source="swebench_pro", tasks_path=tasks_path)
    if task_ids is not None:
        by_id = {}
        for task in tasks:
            if task.id in by_id:
                raise ValueError(f"Duplicate task ID in {tasks_path}: {task.id}")
            by_id[task.id] = task
        missing = [task_id for task_id in task_ids if task_id not in by_id]
        if missing:
            raise ValueError(f"{len(missing)} task IDs missing from {tasks_path}: {missing[:10]}")
        tasks = [by_id[task_id] for task_id in task_ids]
    prepare_swebench_repo_snapshots(
        tasks=tasks,
        repos_cache_dir=repo_cache,
        seed_test_files=seed_test_files,
    )
    return len(tasks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-path", type=Path, default=DEFAULT_TASKS_PATH)
    parser.add_argument(
        "--task-map",
        default=str(DEFAULT_TASK_MAP),
        help="Task map to filter. Use an empty string to prepare every task.",
    )
    parser.add_argument("--repo-cache", type=Path, default=DEFAULT_REPO_CACHE)
    parser.add_argument(
        "--seed-tests",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Seed grader test files into each cached repo as a baseline commit on top "
            "of base_commit (DGM-equivalent harness). Default false = upstream-strict: "
            "agent works against base_commit alone, mirroring the upstream SWE-bench "
            "Pro reference protocol where before_repo_set_cmd executes only inside "
            "the grader after the agent's patch is applied. Use true only for "
            "DGM-comparable runs; results are NOT comparable to public leaderboards."
        ),
    )
    args = parser.parse_args(argv)

    task_map = None if args.task_map == "" else Path(args.task_map)
    count = prepare_repo_cache(
        tasks_path=args.tasks_path.resolve(),
        task_map=task_map.resolve() if task_map is not None else None,
        repo_cache=args.repo_cache.resolve(),
        seed_test_files=args.seed_tests,
    )
    mode = "seeded (DGM-equivalent)" if args.seed_tests else "upstream-strict (no seeded tests)"
    print(f"Prepared {count} SWE-bench Pro repo snapshots under {args.repo_cache.resolve()} [{mode}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

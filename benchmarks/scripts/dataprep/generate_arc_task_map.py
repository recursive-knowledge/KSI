#!/usr/bin/env python3
"""Generate a reproducible ARC task-map artifact from a source directory.

The selection is deterministic:
- enumerate `*.json` files under the source directory
- derive task IDs from sorted file stems
- select `count` tasks with `random.Random(seed).sample(...)`

Example:
    python3 benchmarks/scripts/dataprep/generate_arc_task_map.py \
      --benchmark arc2 \
      --source-repo arcprize/ARC-AGI-2 \
      --source-branch main \
      --source-commit f3283f727488ad98fe575ea6a5ac981e4a188e49 \
      --source-path benchmarks/arc2/source/data/training \
      --split training \
      --selection-name arc2_train_50_seed0 \
      --seed 0 \
      --count 50 \
      --output benchmarks/arc2/task_maps/arc2_train_50_seed0.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SELECTION_ALGORITHM = "random.Random(seed).sample(sorted(task_ids), count)"


def load_task_map_ids(path: Path) -> set[str]:
    """Read task ids from a task-map JSON (either `tasks` or `task_ids` shape)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("tasks"), list):
        return {str(entry["task_id"]) for entry in data["tasks"]}
    if isinstance(data, dict) and isinstance(data.get("task_ids"), list):
        return {str(task_id) for task_id in data["task_ids"]}
    raise ValueError(f"unrecognized task-map shape in {path}: expected 'tasks' or 'task_ids'")


def build_task_map(
    *,
    benchmark: str,
    source_repo: str,
    source_branch: str,
    source_commit: str,
    source_path: str,
    resolved_source_path: Path,
    split: str,
    selection_name: str,
    seed: int,
    count: int,
    exclude_ids: set[str] | None = None,
    excluded_maps: list[str] | None = None,
) -> dict:
    files = sorted(resolved_source_path.glob("*.json"))
    if not files:
        raise ValueError(f"no ARC task json files found under {resolved_source_path}")
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")

    task_ids = sorted(path.stem for path in files)
    if exclude_ids:
        task_ids = [task_id for task_id in task_ids if task_id not in exclude_ids]
    if count > len(task_ids):
        raise ValueError(
            f"requested count={count} exceeds available candidates={len(task_ids)} under "
            f"{resolved_source_path} (after exclusions)"
        )
    selected_ids = random.Random(seed).sample(task_ids, count)

    source_path_posix = source_path.replace("\\", "/").rstrip("/")

    tasks = []
    for idx, task_id in enumerate(selected_ids, start=1):
        tasks.append(
            {
                "index": idx,
                "task_id": task_id,
                "source_file": f"{source_path_posix}/{task_id}.json",
            }
        )

    return {
        "benchmark": benchmark,
        "split": split,
        "seed": seed,
        "count": count,
        "selection_name": selection_name,
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "source_path": source_path_posix,
        "selection_algorithm": SELECTION_ALGORITHM,
        "selection_notes": [
            f"Deterministic subset generated from sorted task IDs with seed={seed}.",
            "Do not modify task membership after publishing results.",
        ],
        "excluded_maps": list(excluded_maps or []),
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a reproducible ARC task-map JSON.")
    parser.add_argument("--benchmark", required=True, help="Benchmark label, e.g. arc1 or arc2")
    parser.add_argument("--source-repo", default="", help="Source repo label; defaults to --benchmark")
    parser.add_argument("--source-branch", required=True, help="Source repo branch name")
    parser.add_argument("--source-commit", required=True, help="Pinned source repo commit SHA")
    parser.add_argument(
        "--source-path",
        required=True,
        help="ARC split directory containing task json files (repo-relative, e.g. data/evaluation)",
    )
    parser.add_argument(
        "--project-root",
        default="",
        help="Project root to resolve --source-path against for file enumeration (default: cwd)",
    )
    parser.add_argument("--split", default="evaluation", help="ARC split label (default: evaluation)")
    parser.add_argument("--selection-name", required=True, help="Selection artifact name")
    parser.add_argument("--seed", type=int, required=True, help="Selection seed")
    parser.add_argument("--count", type=int, required=True, help="Number of tasks to select")
    parser.add_argument(
        "--exclude-task-map",
        action="append",
        default=[],
        help="Path to an existing task-map JSON whose task ids are removed from the "
        "candidate pool before sampling (repeatable)",
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve() if args.project_root else Path.cwd().resolve()
    source_path_raw = args.source_path
    source_path_obj = Path(source_path_raw)
    if source_path_obj.is_absolute():
        resolved_source_path = source_path_obj
    else:
        resolved_source_path = (project_root / source_path_obj).resolve()

    exclude_ids: set[str] = set()
    for exclude_path in args.exclude_task_map:
        exclude_ids |= load_task_map_ids(Path(exclude_path))

    task_map = build_task_map(
        benchmark=args.benchmark,
        source_repo=args.source_repo or args.benchmark,
        source_branch=args.source_branch,
        source_commit=args.source_commit,
        source_path=source_path_raw,
        resolved_source_path=resolved_source_path,
        split=args.split,
        selection_name=args.selection_name,
        seed=args.seed,
        count=args.count,
        exclude_ids=exclude_ids or None,
        excluded_maps=list(args.exclude_task_map),
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(task_map, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {output} | benchmark={args.benchmark} split={args.split} seed={args.seed} count={args.count}")


if __name__ == "__main__":
    main()

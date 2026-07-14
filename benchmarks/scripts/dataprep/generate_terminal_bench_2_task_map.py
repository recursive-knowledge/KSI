#!/usr/bin/env python3
"""Generate stable task maps for the Terminal-Bench 2 submodule checkout."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPO_ROOT / "benchmarks" / "terminal_bench_2" / "source"


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    difficulty: str
    category: str
    docker_image: str
    verifier_timeout_sec: float | None
    agent_timeout_sec: float | None


def _git_head(path: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected TOML object in {path}")
    return data


def _validate_task_dir(task_dir: Path) -> TaskRecord:
    required = [
        task_dir / "instruction.md",
        task_dir / "task.toml",
        task_dir / "environment" / "Dockerfile",
        task_dir / "solution" / "solve.sh",
        task_dir / "tests" / "test.sh",
    ]
    missing = [path.relative_to(task_dir) for path in required if not path.is_file()]
    if missing:
        joined = ", ".join(str(item) for item in missing)
        raise ValueError(f"{task_dir.name}: missing required files: {joined}")

    config = _load_toml(task_dir / "task.toml")
    metadata = config.get("metadata") if isinstance(config.get("metadata"), dict) else {}
    verifier = config.get("verifier") if isinstance(config.get("verifier"), dict) else {}
    agent = config.get("agent") if isinstance(config.get("agent"), dict) else {}
    environment = config.get("environment") if isinstance(config.get("environment"), dict) else {}

    return TaskRecord(
        task_id=task_dir.name,
        difficulty=str(metadata.get("difficulty") or "").strip(),
        category=str(metadata.get("category") or "").strip(),
        docker_image=str(environment.get("docker_image") or "").strip(),
        verifier_timeout_sec=_coerce_float(verifier.get("timeout_sec")),
        agent_timeout_sec=_coerce_float(agent.get("timeout_sec")),
    )


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_requested_ids(task_ids_file: Path) -> list[str]:
    payload = json.loads(task_ids_file.read_text(encoding="utf-8"))
    if isinstance(payload, list) and all(isinstance(item, str) for item in payload):
        return [item.strip() for item in payload if item.strip()]
    if isinstance(payload, dict):
        task_ids = payload.get("task_ids")
        if isinstance(task_ids, list) and all(isinstance(item, str) for item in task_ids):
            return [item.strip() for item in task_ids if item.strip()]
        tasks = payload.get("tasks")
        if isinstance(tasks, list):
            out: list[str] = []
            for item in tasks:
                if isinstance(item, dict):
                    task_id = str(item.get("task_id") or "").strip()
                    if task_id:
                        out.append(task_id)
            if out:
                return out
    raise ValueError(
        f"{task_ids_file} must contain a JSON array of task ids, a `task_ids` array, "
        "or a `tasks` array with `task_id` objects"
    )


def _select_task_ids(
    *,
    records: dict[str, TaskRecord],
    count: int,
    seed: int,
    task_ids_file: Path | None,
) -> list[str]:
    all_ids = sorted(records)
    if task_ids_file is not None:
        requested = _load_requested_ids(task_ids_file)
        missing = [task_id for task_id in requested if task_id not in records]
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(f"requested task ids not found in source checkout: {preview}")
        return requested
    if count <= 0 or count >= len(all_ids):
        return all_ids
    return sorted(random.Random(seed).sample(all_ids, count))


def build_task_map(
    *,
    source: Path,
    selection_name: str,
    count: int,
    seed: int,
    task_ids_file: Path | None,
) -> dict[str, Any]:
    if not source.is_dir():
        raise FileNotFoundError(f"source directory not found: {source}")

    task_dirs = sorted(path for path in source.iterdir() if path.is_dir() and not path.name.startswith("."))
    if not task_dirs:
        raise ValueError(f"no task directories found under {source}")

    records = {task_dir.name: _validate_task_dir(task_dir) for task_dir in task_dirs}
    selected_ids = _select_task_ids(
        records=records,
        count=count,
        seed=seed,
        task_ids_file=task_ids_file,
    )
    revision = _git_head(source)

    notes = [
        "Terminal-Bench 2 source is an upstream Harbor-native task corpus tracked as a git submodule.",
        "Each selected task was validated to contain instruction.md, task.toml, environment/Dockerfile, solution/solve.sh, and tests/test.sh.",
        "Do not modify task membership after publishing results.",
    ]
    if task_ids_file is not None:
        notes.append(f"Selection taken from explicit task ids file: {task_ids_file}.")
    else:
        notes.append(
            "Selection algorithm: random.Random(selection_seed).sample(sorted(task_ids), task_count)."
            if count > 0 and count < len(records)
            else "Selection includes the full validated task set."
        )

    return {
        "selection_name": selection_name,
        "benchmark": "terminal_bench_2",
        "dataset_name": "harbor-framework/terminal-bench-2",
        "source_path": str(source.relative_to(REPO_ROOT)),
        "source_git_revision": revision,
        "selection_seed": seed,
        "task_count": len(selected_ids),
        "selection_notes": notes,
        "tasks": [
            {
                "index": index,
                "task_id": task_id,
                "difficulty": records[task_id].difficulty,
                "category": records[task_id].category,
                "docker_image": records[task_id].docker_image,
                "verifier_timeout_sec": records[task_id].verifier_timeout_sec,
                "agent_timeout_sec": records[task_id].agent_timeout_sec,
                "notes": "",
            }
            for index, task_id in enumerate(selected_ids, start=1)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="TB2 source checkout directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON task map path.")
    parser.add_argument("--selection-name", default="", help="Optional explicit selection name.")
    parser.add_argument("--count", type=int, default=0, help="Task count to sample; 0 means all validated tasks.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampled subsets.")
    parser.add_argument(
        "--task-ids-file",
        type=Path,
        default=None,
        help="Optional JSON file specifying exact task ids to include.",
    )
    args = parser.parse_args(argv)

    if args.count < 0:
        parser.error("--count must be >= 0")
    if args.task_ids_file is not None and not args.task_ids_file.is_file():
        parser.error(f"--task-ids-file not found: {args.task_ids_file}")

    selection_name = args.selection_name.strip()
    if not selection_name:
        if args.task_ids_file is not None:
            selection_name = f"terminal_bench_2_explicit_{args.task_ids_file.stem}"
        elif args.count > 0:
            selection_name = f"terminal_bench_2_{args.count}_seed{args.seed}"
        else:
            selection_name = "terminal_bench_2_all"

    task_map = build_task_map(
        source=args.source.resolve(),
        selection_name=selection_name,
        count=args.count,
        seed=args.seed,
        task_ids_file=args.task_ids_file.resolve() if args.task_ids_file else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(task_map, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.output} | tasks={task_map['task_count']} | source_rev={task_map['source_git_revision'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

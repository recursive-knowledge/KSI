#!/usr/bin/env python3
"""Generate a reproducible SWE-bench Pro task-map artifact from a tabular dataset.

The selection is deterministic:
- load rows from jsonl/csv/parquet
- derive task IDs from sorted `instance_id` values
- select `count` tasks with `random.Random(seed).sample(...)`

Example:
    python3 benchmarks/scripts/dataprep/generate_swebench_pro_task_map.py \
      --dataset-path benchmarks/swebench_pro/dataset/test.jsonl \
      --dataset-name ScaleAI/SWE-bench_Pro \
      --split test \
      --selection-name swebench_pro_test_50_seed0_v1 \
      --seed 0 \
      --count 50 \
      --output benchmarks/swebench_pro/task_maps/swebench_pro_test_50_seed0_v1.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any

from ksi.benchmarks.swebench_pro_external import DATASET_NAME, DATASET_REVISION

REPO_ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _portable_source_path(dataset_path: Path) -> str:
    try:
        return dataset_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return dataset_path.as_posix()


def _load_rows_from_bytes(dataset_path: Path, data: bytes) -> list[dict[str, Any]]:
    suffix = dataset_path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in data.decode("utf-8").splitlines():
            payload = line.strip()
            if not payload:
                continue
            row = json.loads(payload)
            if isinstance(row, dict):
                rows.append(row)
        return rows
    if suffix == ".csv":
        return [dict(row) for row in csv.DictReader(io.StringIO(data.decode("utf-8")))]
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore

            table = pq.read_table(io.BytesIO(data))
            return [dict(row) for row in table.to_pylist()]
        except ImportError:
            import pandas as pd  # type: ignore

            return pd.read_parquet(io.BytesIO(data)).to_dict(orient="records")
    raise ValueError(f"Unsupported SWE-bench Pro dataset file type: {dataset_path}")


def _load_rows(dataset_path: Path) -> list[dict[str, Any]]:
    return _load_rows_from_bytes(dataset_path, dataset_path.read_bytes())


def build_task_map(
    *,
    dataset_path: Path,
    dataset_name: str,
    split: str,
    selection_name: str,
    seed: int,
    count: int,
    source_revision: str | None = None,
) -> dict[str, Any]:
    dataset_snapshot = dataset_path.read_bytes()
    source_sha256 = _sha256_bytes(dataset_snapshot)
    rows = _load_rows_from_bytes(dataset_path, dataset_snapshot)
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        instance_id = str(row.get("instance_id") or "").strip()
        if not instance_id:
            continue
        indexed[instance_id] = row

    task_ids = sorted(indexed)
    if not task_ids:
        raise ValueError(f"no SWE-bench Pro rows with instance_id found in {dataset_path}")
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if count > len(task_ids):
        raise ValueError(f"requested count={count} exceeds available tasks={len(task_ids)} in {dataset_path}")

    selected_ids = random.Random(seed).sample(task_ids, count)
    tasks = []
    for idx, task_id in enumerate(selected_ids, start=1):
        row = indexed[task_id]
        tasks.append(
            {
                "index": idx,
                "task_id": task_id,
                "repo": str(row.get("repo") or "").strip(),
                "base_commit": str(row.get("base_commit") or "").strip(),
                "notes": "",
            }
        )

    return {
        "selection_name": selection_name,
        "benchmark": "swebench_pro",
        "dataset_name": dataset_name,
        "source_path": _portable_source_path(dataset_path),
        "source_sha256": source_sha256,
        "source_revision": source_revision,
        "split": split,
        "selection_seed": seed,
        "task_count": count,
        "selection_notes": [
            "Deterministic subset generated from sorted instance_id values.",
            "Selection algorithm: random.Random(selection_seed).sample(sorted(instance_ids), task_count).",
            "Do not modify task membership after publishing results.",
        ],
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a reproducible SWE-bench Pro task-map JSON.")
    parser.add_argument("--dataset-path", required=True, help="Path to SWE-bench Pro dataset (.jsonl/.csv/.parquet)")
    parser.add_argument("--dataset-name", default=DATASET_NAME, help="Dataset label")
    parser.add_argument(
        "--source-revision",
        default=DATASET_REVISION,
        help=(
            "Hugging Face dataset revision the --dataset-path was exported from "
            "(git tag/branch/commit SHA). Recorded as 'source_revision' in the "
            f"map. Default: {DATASET_REVISION}. Pass an empty string only for "
            "an explicit unpinned map."
        ),
    )
    parser.add_argument("--split", default="test", help="Dataset split label (default: test)")
    parser.add_argument("--selection-name", required=True, help="Selection artifact name")
    parser.add_argument("--seed", type=int, required=True, help="Selection seed")
    parser.add_argument("--count", type=int, required=True, help="Number of tasks to select")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    task_map = build_task_map(
        dataset_path=dataset_path,
        dataset_name=args.dataset_name,
        split=args.split,
        selection_name=args.selection_name,
        seed=args.seed,
        count=args.count,
        source_revision=(args.source_revision.strip() or None) if isinstance(args.source_revision, str) else None,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(task_map, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {output} | dataset={args.dataset_name} split={args.split} seed={args.seed} count={args.count}")


if __name__ == "__main__":
    main()

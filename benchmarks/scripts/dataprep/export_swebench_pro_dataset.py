#!/usr/bin/env python3
"""Export SWE-bench Pro from Hugging Face into a local tabular file.

Usage:
    python benchmarks/scripts/dataprep/export_swebench_pro_dataset.py \
        --split test \
        --format jsonl \
        --output benchmarks/swebench_pro/dataset/test.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ksi.benchmarks.swebench_pro_external import DATASET_NAME, DATASET_REVISION


def _load_dataset_rows(*, dataset_name: str, split: str, revision: str | None = None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "The `datasets` package is required for this dataprep script. Install it with `uv sync --extra dataprep`."
        ) from exc

    dataset = load_dataset(dataset_name, split=split, revision=revision)
    return [dict(row) for row in dataset]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {
                key: json.dumps(value, ensure_ascii=True) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(normalized)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path)
        return
    except ImportError:
        pass

    import pandas as pd  # type: ignore

    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def export_dataset(*, dataset_name: str, split: str, output: Path, fmt: str, revision: str | None = None) -> int:
    rows = _load_dataset_rows(dataset_name=dataset_name, split=split, revision=revision)
    output.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "jsonl":
        _write_jsonl(output, rows)
    elif fmt == "csv":
        _write_csv(output, rows)
    elif fmt == "parquet":
        _write_parquet(output, rows)
    else:
        raise ValueError(f"unsupported format={fmt!r}")

    print(
        f"wrote {output} | dataset={dataset_name} split={split} "
        f"revision={revision or '<unpinned>'} rows={len(rows)} format={fmt}"
    )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SWE-bench Pro dataset from Hugging Face.")
    parser.add_argument(
        "--dataset",
        default=DATASET_NAME,
        help=f"Hugging Face dataset name (default: {DATASET_NAME})",
    )
    parser.add_argument("--split", default="test", help="Dataset split to export")
    parser.add_argument(
        "--revision",
        default=DATASET_REVISION,
        help=(
            "Hugging Face dataset revision to pin (git tag/branch/commit SHA). "
            f"Default: {DATASET_REVISION}. Pass an empty string only for an "
            "explicit unpinned export."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("jsonl", "csv", "parquet"),
        default="jsonl",
        help="Output format (default: jsonl)",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/swebench_pro/dataset/test.jsonl",
        help="Output file path",
    )
    args = parser.parse_args()

    export_dataset(
        dataset_name=args.dataset,
        split=args.split,
        output=Path(args.output),
        fmt=args.format,
        revision=(args.revision.strip() or None) if isinstance(args.revision, str) else args.revision,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Cap source file size at 10 MB. ARC task files are ~10-100 KB; anything larger
# is either a misconfigured path or an accidentally swapped file.
MAX_SOURCE_FILE_BYTES = 10 * 1024 * 1024
_UNPINNED_SWEBENCH_REVISION_MARKERS = {"", "none", "null", "<none>", "<null>", "<unpinned>", "unpinned"}
ARC_REQUIRED_PROVENANCE_FIELDS = (
    "benchmark",
    "split",
    "selection_name",
    "source_repo",
    "source_branch",
    "source_commit",
    "source_path",
    "selection_algorithm",
)
_GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _pinned_swebench_source_revision(payload: dict[str, Any]) -> str | None:
    raw = payload.get("source_revision")
    if not isinstance(raw, str):
        return None
    revision = raw.strip()
    if revision.lower() in _UNPINNED_SWEBENCH_REVISION_MARKERS:
        return None
    return revision


def _resolve_source_path(source_value: str) -> tuple[Path, bool]:
    """Return (resolved_path, was_relative).

    Relative paths are joined against REPO_ROOT and must stay under it after
    resolution (prevents `../..` traversal from a task map of unknown origin).
    Absolute paths are trusted — they're explicitly opt-in.
    """
    path = Path(source_value)
    if path.is_absolute():
        return path, False
    return REPO_ROOT / path, True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return False
    return True


def _check_source_files(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    missing: list[dict[str, str]] = []
    malformed: list[dict[str, str]] = []
    checked = 0
    for idx, task in enumerate(tasks):
        if not isinstance(task, dict):
            malformed.append({"index": str(idx), "error": "task is not an object"})
            continue
        task_id = task.get("task_id", f"<index {idx}>")
        source_file = task.get("source_file")
        if not isinstance(source_file, str) or not source_file:
            malformed.append({"task_id": str(task_id), "error": "missing source_file"})
            continue
        path, was_relative = _resolve_source_path(source_file)
        if was_relative and not _is_within_repo(path):
            malformed.append(
                {
                    "task_id": str(task_id),
                    "source_file": source_file,
                    "error": "relative source_file escapes repository root",
                }
            )
            continue
        if not path.is_file():
            missing.append({"task_id": str(task_id), "source_file": source_file})
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            malformed.append(
                {
                    "task_id": str(task_id),
                    "source_file": source_file,
                    "error": f"stat failed: {exc}",
                }
            )
            continue
        if size > MAX_SOURCE_FILE_BYTES:
            malformed.append(
                {
                    "task_id": str(task_id),
                    "source_file": source_file,
                    "error": f"file exceeds {MAX_SOURCE_FILE_BYTES} bytes ({size})",
                }
            )
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            malformed.append(
                {
                    "task_id": str(task_id),
                    "source_file": source_file,
                    "error": f"invalid JSON: {exc.msg} at line {exc.lineno}",
                }
            )
            continue
        except UnicodeDecodeError as exc:
            malformed.append(
                {
                    "task_id": str(task_id),
                    "source_file": source_file,
                    "error": f"not valid UTF-8: {exc.reason} at byte {exc.start}",
                }
            )
            continue
        if not isinstance(data, dict) or "train" not in data or "test" not in data:
            malformed.append(
                {
                    "task_id": str(task_id),
                    "source_file": source_file,
                    "error": "missing train/test fields",
                }
            )
            continue
        # Only count fully verified entries: resolved, parseable, and shaped correctly.
        checked += 1
    return {
        "checked": checked,
        "missing_count": len(missing),
        "malformed_count": len(malformed),
        "missing": missing[:25],
        "malformed": malformed[:25],
    }


def _required_provenance_errors(payload: dict[str, Any], *, task_source: str) -> list[dict[str, str]]:
    if task_source != "arc":
        return []

    errors: list[dict[str, str]] = []
    for field in ARC_REQUIRED_PROVENANCE_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append({"field": field, "error": "missing or empty"})
            continue
        if field == "source_commit" and not _GIT_SHA_RE.fullmatch(value.strip()):
            errors.append({"field": field, "error": "expected 40-character git commit SHA"})
    return errors


def _load_parquet_ids(path: Path, id_col: str) -> set[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(path, columns=[id_col])
        rows = table.to_pylist()
        return {str(r.get(id_col) or "").strip() for r in rows if isinstance(r, dict)}
    except ImportError:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(path, columns=[id_col])
        return {str(v).strip() for v in df[id_col].tolist()}


def _load_swebench_ids(path: Path) -> set[str]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        ids: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = line.strip()
            if not payload:
                continue
            row = json.loads(payload)
            if isinstance(row, dict):
                value = str(row.get("instance_id") or "").strip()
                if value:
                    ids.add(value)
        return ids
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return {
                str(row.get("instance_id") or "").strip() for row in csv.DictReader(handle) if row.get("instance_id")
            }
    if suffix == ".parquet":
        return _load_parquet_ids(path, "instance_id")
    raise ValueError(f"SWE-bench Pro source expects .jsonl/.csv/.parquet, got: {path}")


def _load_arc_ids(path: Path) -> set[str]:
    if path.is_file():
        if path.suffix.lower() != ".json":
            raise ValueError(f"ARC source expects .json task files, got: {path}")
        return {path.stem}
    if not path.is_dir():
        raise ValueError(f"ARC source expects dir/file, got: {path}")

    ids = {p.stem for p in path.glob("*.json")}
    if ids:
        return ids
    for split in ("training", "evaluation"):
        split_dir = path / split
        if split_dir.is_dir():
            for p in split_dir.glob("*.json"):
                ids.add(p.stem)
    if ids:
        return ids
    return {p.stem for p in path.rglob("*.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate task map IDs against source dataset.")
    parser.add_argument("--task-map", required=True, help="Path to map JSON")
    parser.add_argument(
        "--task-source",
        required=True,
        choices=("arc", "swebench_pro"),
        help="Task source for validation",
    )
    parser.add_argument("--tasks-path", required=True, help="Source dataset path")
    parser.add_argument(
        "--check-sources",
        action="store_true",
        help=(
            "Also verify each task's source_file field points to an existing, "
            "JSON-parseable file with train/test keys. ARC-only."
        ),
    )
    parser.add_argument(
        "--require-provenance",
        action="store_true",
        help="Fail when the task map omits required provenance fields for its task source. ARC requires source_commit.",
    )
    args = parser.parse_args()

    if args.check_sources and args.task_source != "arc":
        raise SystemExit(f"--check-sources is only supported for --task-source arc, not {args.task_source!r}")

    task_map_path = Path(args.task_map)
    tasks_path = Path(args.tasks_path)
    payload = json.loads(task_map_path.read_text(encoding="utf-8"))
    if args.require_provenance and not isinstance(payload, dict):
        raise SystemExit(f"{task_map_path}: --require-provenance requires a JSON object task map")
    if "task_ids" in payload:
        raw_ids = payload["task_ids"]
        tasks_entries: list[dict[str, Any]] = []
    elif "tasks" in payload:
        tasks_entries = payload["tasks"] if isinstance(payload["tasks"], list) else []
        raw_ids = [t["task_id"] for t in tasks_entries]
    else:
        raise SystemExit(f"{task_map_path}: task map has neither 'task_ids' nor 'tasks' field")
    task_ids = [str(x).strip() for x in raw_ids if str(x).strip()]
    if not task_ids:
        raise SystemExit(f"{task_map_path}: empty or missing task_ids")

    if args.task_source == "arc":
        available = _load_arc_ids(tasks_path)
    else:
        available = _load_swebench_ids(tasks_path)

    unique_task_ids = sorted(set(task_ids))
    missing = [task_id for task_id in unique_task_ids if task_id not in available]
    duplicates = len(task_ids) - len(unique_task_ids)

    summary: dict[str, Any] = {
        "task_map": str(task_map_path),
        "task_source": args.task_source,
        "tasks_path": str(tasks_path),
        "task_ids_total": len(task_ids),
        "task_ids_unique": len(unique_task_ids),
        "duplicates": duplicates,
        "missing_count": len(missing),
        "missing_ids": missing[:25],
    }

    provenance_bad = False
    if args.require_provenance:
        provenance_errors = _required_provenance_errors(payload, task_source=args.task_source)
        summary["provenance"] = {
            "required_fields": list(ARC_REQUIRED_PROVENANCE_FIELDS) if args.task_source == "arc" else [],
            "missing_or_invalid": provenance_errors,
        }
        provenance_bad = bool(provenance_errors)

    # Dataset-integrity tripwire: when the map records a source_sha256, recompute
    # it from the actual --tasks-path being validated. source_path remains
    # provenance; it must not be used as a substitute for the operator-supplied
    # dataset path (#1153).
    source_sha_mismatch = False
    source_sha_missing_for_revision = False
    expected_sha = payload.get("source_sha256") if isinstance(payload, dict) else None
    source_path_value = payload.get("source_path") if isinstance(payload, dict) else None
    source_revision = payload.get("source_revision") if isinstance(payload, dict) else None
    pinned_source_revision = _pinned_swebench_source_revision(payload) if isinstance(payload, dict) else None
    if isinstance(source_path_value, str) and source_path_value.strip():
        summary["source_path"] = source_path_value.strip()
    if isinstance(source_revision, str) and source_revision.strip():
        summary["source_revision"] = source_revision.strip()
    if isinstance(expected_sha, str) and expected_sha.strip() and tasks_path.is_file():
        actual_sha = _sha256(tasks_path)
        summary["source_sha256_expected"] = expected_sha.strip()
        summary["source_sha256_actual"] = actual_sha
        source_sha_mismatch = actual_sha != expected_sha.strip()
    elif pinned_source_revision is not None:
        summary["source_sha256_missing"] = True
        source_sha_missing_for_revision = True

    sources_bad = False
    if args.check_sources:
        if not tasks_entries:
            raise SystemExit(
                f"{task_map_path}: --check-sources requires the 'tasks' field "
                f"(with per-task source_file entries), not 'task_ids'."
            )
        sources_report = _check_source_files(tasks_entries)
        summary["sources"] = sources_report
        sources_bad = bool(sources_report["missing_count"] or sources_report["malformed_count"])

    print(json.dumps(summary, indent=2, ensure_ascii=True))

    if missing:
        raise SystemExit(2)
    if duplicates:
        raise SystemExit(3)
    if sources_bad:
        raise SystemExit(4)
    if source_sha_mismatch:
        raise SystemExit(5)
    if source_sha_missing_for_revision:
        raise SystemExit(5)
    if provenance_bad:
        raise SystemExit(6)


if __name__ == "__main__":
    main()

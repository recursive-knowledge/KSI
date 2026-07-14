"""Shared helpers for the ARC prep scripts.

Extracted from the ARC workspace payload, native prompt, and prediction
conversion scripts to avoid silent drift between the three.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-]+$")


def sanitize_task_id(task_id: Any) -> str:
    if not isinstance(task_id, str) or not _TASK_ID_PATTERN.fullmatch(task_id):
        raise ValueError(f"Unsafe task_id (must match [A-Za-z0-9_-]+): {task_id!r}")
    return task_id


def load_json(path: Path) -> object:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON ({exc.msg} at line {exc.lineno})") from exc


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def relative_to_manifest(target: Path, manifest_dir: Path) -> str:
    """Return target as a path relative to manifest_dir.

    Uses pathlib.Path.relative_to when possible; falls back to
    os.path.relpath for targets outside the manifest_dir subtree.
    Resolved absolute forms are compared so symlinks/`..` segments are
    normalized before computing the relative path.
    """
    target_abs = target.resolve()
    manifest_dir_abs = manifest_dir.resolve()
    try:
        return str(target_abs.relative_to(manifest_dir_abs))
    except ValueError:
        return os.path.relpath(str(target_abs), start=str(manifest_dir_abs))


def validate_grid(grid: object, origin: str) -> None:
    """Validate that `grid` is a non-empty 2D list of ints in [0,9].

    Raises ValueError with a message that includes `origin` for easy
    identification of the offending task/pair.
    """
    if not isinstance(grid, list) or not grid:
        raise ValueError(f"{origin}: grid must be a non-empty list of rows.")
    width = None
    for row_index, row in enumerate(grid):
        if not isinstance(row, list) or not row:
            raise ValueError(f"{origin}: row {row_index} must be a non-empty list.")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"{origin}: row {row_index} has width {len(row)}, expected {width}.")
        for col_index, value in enumerate(row):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 9:
                raise ValueError(
                    f"{origin}: cell ({row_index},{col_index}) must be an integer in [0,9], got {value!r}."
                )


def require_field(obj: Any, key: str, *, origin: str) -> Any:
    """Return obj[key] or raise ValueError with origin context."""
    if not isinstance(obj, dict):
        raise ValueError(f"{origin}: expected object, got {type(obj).__name__}.")
    if key not in obj:
        raise ValueError(f"{origin}: missing required field {key!r}.")
    return obj[key]


def coerce_int(value: Any, *, origin: str) -> int:
    """Coerce value to int or raise ValueError with origin context."""
    if isinstance(value, bool):
        raise ValueError(f"{origin}: expected integer, got bool {value!r}.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{origin}: expected integer, got {value!r}.") from exc

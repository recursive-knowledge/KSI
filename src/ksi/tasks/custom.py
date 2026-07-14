"""Built-in ``custom`` task source: run KSI on your own tasks.

Tasks are defined in a JSON array (``.json``) or JSONL (``.jsonl``) file.
Each record::

    {
      "task_id": "my-task-1",                     # required, unique
      "prompt": "…instruction for the agent…",    # required
      "workspace_dir": "path/to/starting/files",  # optional, dir; relative to the tasks file
      "files": {"relative/path.py": "content"},   # optional, inline alternative to workspace_dir
      "eval": {"command": "python3 tests.py",     # optional; graded by the `command` evaluator
               "timeout_sec": 300}
    }

``workspace_dir`` and ``files`` are mutually exclusive. Either way the task's
starting files are seeded into the agent workspace's ``repo/`` directory
(via ``metadata["repo_path"]``, the same seam the benchmark sources use), and
the ``command`` evaluator later runs ``eval.command`` host-side in the
captured post-attempt copy of that directory.
"""

from __future__ import annotations

import atexit
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..models import TaskSpec
from .registry import TaskSourceSpec, register_task_source, resolve_source

_DEFAULT_EVAL_TIMEOUT_SEC = 300.0

_WORKSPACE_GUIDANCE = (
    "\n\nYour workspace's repo/ directory contains the task's starting files "
    "(it may be empty). Create or edit files there; the evaluation command "
    "runs in that directory after your attempt."
)

_TEMP_SEED_DIRS: list[str] = []


def _cleanup_temp_seed_dirs() -> None:
    for path in _TEMP_SEED_DIRS:
        shutil.rmtree(path, ignore_errors=True)


atexit.register(_cleanup_temp_seed_dirs)


def _make_temp_seed_dir() -> Path:
    seed = Path(tempfile.mkdtemp(prefix="ksi_custom_"))
    _TEMP_SEED_DIRS.append(str(seed))
    return seed


def _record_error(index: int, task_id: str, message: str) -> ValueError:
    label = f"record {index}" + (f" (task_id={task_id!r})" if task_id else "")
    return ValueError(f"custom tasks file: {label}: {message}")


def _read_records(tasks_path: Path) -> list[Any]:
    text = tasks_path.read_text(encoding="utf-8")
    if tasks_path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"custom tasks file {tasks_path}: .json form must be a JSON array")
    return data


def _validate_files(index: int, task_id: str, raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict) or not raw:
        raise _record_error(index, task_id, "files must be a non-empty object of relative-path -> content")
    out: dict[str, str] = {}
    for key, value in raw.items():
        rel = str(key or "").strip()
        if not rel or rel.startswith(("/", "\\")) or ".." in Path(rel).parts:
            raise _record_error(index, task_id, f"files key {key!r} must be a relative path without '..'")
        if not isinstance(value, str):
            raise _record_error(index, task_id, f"files[{key!r}] content must be a string")
        out[rel] = value
    return out


def _materialize_seed_dir(files: dict[str, str]) -> Path:
    seed = _make_temp_seed_dir()
    for rel, content in files.items():
        dest = seed / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
    return seed


def _resolve_workspace_dir(index: int, task_id: str, raw: Any, *, base: Path) -> Path:
    ws = Path(str(raw)).expanduser()
    if not ws.is_absolute():
        ws = base / ws
    ws = ws.resolve()
    if not ws.is_dir():
        raise _record_error(index, task_id, f"workspace_dir {raw!r} does not exist or is not a directory")
    return ws


def _parse_eval(index: int, task_id: str, raw: Any) -> tuple[str, float]:
    if raw is None:
        return "", _DEFAULT_EVAL_TIMEOUT_SEC
    if not isinstance(raw, dict):
        raise _record_error(index, task_id, "eval must be an object with a 'command' key")
    command = str(raw.get("command") or "").strip()
    if not command:
        raise _record_error(index, task_id, "eval.command must be a non-empty string")
    timeout_raw = raw.get("timeout_sec", _DEFAULT_EVAL_TIMEOUT_SEC)
    try:
        timeout = float(timeout_raw)
    except (TypeError, ValueError):
        raise _record_error(index, task_id, f"eval.timeout_sec must be a number, got {timeout_raw!r}") from None
    if timeout <= 0:
        raise _record_error(index, task_id, "eval.timeout_sec must be > 0")
    return command, timeout


def load_custom_tasks(tasks_path: Path, **_kwargs: Any) -> list[TaskSpec]:
    tasks_path = Path(tasks_path)
    records = _read_records(tasks_path)
    if not records:
        raise ValueError(f"custom tasks file {tasks_path} contains no task records")
    base = tasks_path.resolve().parent
    tasks: list[TaskSpec] = []
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise _record_error(index, "", f"expected an object, got {type(record).__name__}")
        task_id = str(record.get("task_id") or "").strip()
        if not task_id:
            raise _record_error(index, "", "task_id is required and must be non-empty")
        if task_id in seen:
            raise _record_error(index, task_id, "duplicate task_id")
        seen.add(task_id)
        prompt = str(record.get("prompt") or "").strip()
        if not prompt:
            raise _record_error(index, task_id, "prompt is required and must be non-empty")
        unknown = set(record) - {"task_id", "prompt", "workspace_dir", "files", "eval"}
        if unknown:
            raise _record_error(index, task_id, f"unknown keys: {sorted(unknown)}")
        if record.get("files") is not None and record.get("workspace_dir") is not None:
            raise _record_error(index, task_id, "'files' and 'workspace_dir' are mutually exclusive")
        if record.get("workspace_dir") is not None:
            seed = _resolve_workspace_dir(index, task_id, record["workspace_dir"], base=base)
        elif record.get("files") is not None:
            seed = _materialize_seed_dir(_validate_files(index, task_id, record["files"]))
        else:
            seed = _make_temp_seed_dir()
        command, timeout = _parse_eval(index, task_id, record.get("eval"))
        tasks.append(
            TaskSpec(
                id=task_id,
                repo="",
                prompt=prompt + _WORKSPACE_GUIDANCE,
                metadata={
                    "task_source": "custom",
                    "repo_path": str(seed),
                    "eval_command": command,
                    "eval_timeout_sec": timeout,
                },
            )
        )
    return tasks


def validate_custom_tasks_path(tasks_path: Path, *, evals_path: Path | None = None) -> str | None:
    """Return an error string when ``tasks_path`` is unusable, else None."""
    del evals_path
    tasks_path = Path(tasks_path)
    if not tasks_path.is_file():
        return f"custom tasks file not found: {tasks_path}"
    if tasks_path.suffix.lower() not in {".json", ".jsonl"}:
        return f"custom task source expects a .json or .jsonl file, got: {tasks_path.name}"
    try:
        records = _read_records(tasks_path)
    except (ValueError, json.JSONDecodeError) as exc:
        return f"custom tasks file {tasks_path} is not valid JSON/JSONL: {exc}"
    if not records:
        return f"custom tasks file {tasks_path} contains no task records"
    return None


def register_custom_source() -> None:
    """Register the built-in ``custom`` ``TaskSourceSpec`` and wire its loader/validator.

    Called from ``ksi.tasks.loaders`` at import time, AFTER
    ``ksi.benchmarks.register_all()``, so the canonical registration order
    stays ``swebench_pro, arc, polyglot, terminal_bench_2, custom`` (pinned by
    ``tests/test_task_registry.py``). Idempotent.
    """
    from dataclasses import replace as dataclass_replace

    spec = resolve_source("custom")
    if spec is None:
        spec = register_task_source(
            TaskSourceSpec(
                name="custom",
                default_evaluator="command",
                prompt_kind="generic",
            )
        )
    if spec.loader is None:
        register_task_source(
            dataclass_replace(
                spec,
                loader=lambda tasks_path, **kwargs: load_custom_tasks(tasks_path, **kwargs),
                validate_tasks_path=validate_custom_tasks_path,
            ),
            replace=True,
        )

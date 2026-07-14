"""Generic host-side command evaluator for the ``custom`` task source.

Runs each task's ``metadata["eval_command"]`` as a shell command in the
post-attempt workspace (``runtime_meta["host_workspace_repo_dir"]``; falls
back to a fresh copy of the task's seed dir, overlaid with any captured
``runtime_meta["workspace_solution_files"]`` content, when the runtime's
captured on-disk workspace is unavailable — e.g. it was wiped by
``--wipe-workspace-per-task true``, the default). Runs on the HOST, never
inside the agent container, so grader output stays out of the agent's reach
by construction. Exit 0 scores 1.0,
nonzero 0.0; a ``score.json`` ``{"score": <0..1>}`` written by the command
overrides (partial credit). Infra failures (timeout, spawn failure, missing
workspace) return status-only dicts, which score ``None`` (unscored) under
``score_from_eval_results``.

SECURITY: the eval command comes from the user's own tasks file and runs
with the launching user's privileges. Do not run untrusted tasks files.
"""

from __future__ import annotations

import atexit
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import TaskSpec

_TEMP_EVAL_DIRS: list[str] = []


def _cleanup_temp_eval_dirs() -> None:
    for path in _TEMP_EVAL_DIRS:
        shutil.rmtree(path, ignore_errors=True)


atexit.register(_cleanup_temp_eval_dirs)


def _make_temp_eval_dir() -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="kcsi_cmd_eval_"))
    _TEMP_EVAL_DIRS.append(str(tmpdir))
    return tmpdir


_TAIL_CHARS = 2000


@dataclass
class CommandEvaluator:
    default_timeout_sec: float = 300.0

    def evaluate(self, *, task: TaskSpec, model_output: str, **kwargs: Any) -> dict[str, Any]:
        del model_output
        metadata = task.metadata or {}
        command = str(metadata.get("eval_command") or "").strip()
        if not command:
            return {"status": "no_eval_command", "instance_id": task.id}
        runtime_meta = kwargs.get("runtime_meta") or {}
        workdir = self._resolve_workdir(metadata, runtime_meta)
        if workdir is None:
            return {"status": "no_workspace", "instance_id": task.id}
        # Forgery close: a stale score.json (left over in a captured
        # host_workspace_repo_dir from an earlier retry, or one the agent
        # itself wrote directly) must not be mistaken for the eval command's
        # own override. Only the command run below may produce it.
        (workdir / "score.json").unlink(missing_ok=True)
        timeout = self._timeout(metadata)
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "eval_timeout",
                "instance_id": task.id,
                "timeout_sec": timeout,
                "eval_workdir": str(workdir),
            }
        except OSError as exc:
            return {"status": "eval_spawn_failed", "instance_id": task.id, "error": str(exc)}
        score = 1.0 if proc.returncode == 0 else 0.0
        override = self._score_json_override(workdir)
        if override is not None:
            score = override
        return {
            "status": "evaluated",
            "instance_id": task.id,
            "native_score": score,
            "resolved": score >= 1.0,
            "exit_code": proc.returncode,
            "stdout_tail": proc.stdout[-_TAIL_CHARS:],
            "stderr_tail": proc.stderr[-_TAIL_CHARS:],
            "eval_workdir": str(workdir),
        }

    def _timeout(self, metadata: dict[str, Any]) -> float:
        raw = metadata.get("eval_timeout_sec")
        try:
            value = float(raw) if raw is not None else self.default_timeout_sec
        except (TypeError, ValueError):
            return self.default_timeout_sec
        return value if value > 0 else self.default_timeout_sec

    def _resolve_workdir(self, metadata: dict[str, Any], runtime_meta: dict[str, Any]) -> Path | None:
        captured = runtime_meta.get("host_workspace_repo_dir") if isinstance(runtime_meta, dict) else None
        if isinstance(captured, str) and captured and Path(captured).is_dir():
            return Path(captured)
        seed = metadata.get("repo_path")
        if not (isinstance(seed, str) and seed and Path(seed).is_dir()):
            return None
        # No captured on-disk workspace (e.g. wiped by the default
        # --wipe-workspace-per-task true before this evaluate() call runs):
        # grade a fresh COPY of the starter state so the eval command can
        # never mutate the shared seed dir, overlaid with whatever file
        # content the runtime captured in-process — before the workspace was
        # wiped — via runtime_meta["workspace_solution_files"] (a relative
        # path -> content dict; the same channel the polyglot evaluator's
        # solution-file fallback uses).
        dest = _make_temp_eval_dir() / "repo"
        shutil.copytree(seed, dest)
        solution_files = runtime_meta.get("workspace_solution_files") if isinstance(runtime_meta, dict) else None
        if isinstance(solution_files, dict):
            for rel, content in solution_files.items():
                if not isinstance(rel, str) or not isinstance(content, str):
                    continue
                rel_path = Path(rel)
                if rel_path.is_absolute() or ".." in rel_path.parts:
                    continue
                # Defense in depth (mirrors the TS-side capture filter): never
                # overlay a captured ``score.json`` (would forge the score
                # override below) or a test file (would let a captured copy
                # of the grader clobber the seed's real one).
                base = rel_path.name.lower()
                if base == "score.json" or "test" in base:
                    continue
                target = dest / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
        return dest

    def _score_json_override(self, workdir: Path) -> float | None:
        score_file = workdir / "score.json"
        if not score_file.is_file():
            return None
        try:
            value = json.loads(score_file.read_text(encoding="utf-8")).get("score")
        except (OSError, ValueError):
            return None
        if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
            return float(value)
        return None

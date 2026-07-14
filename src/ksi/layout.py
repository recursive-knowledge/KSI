from __future__ import annotations

import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"

RUNTIME_STATE_DIR = PROJECT_ROOT / "runtime_state"
RUNTIME_KNOWLEDGE_DIR = RUNTIME_STATE_DIR / "knowledge"
RUNTIME_AUDIT_DIR = RUNTIME_STATE_DIR / "runtime"


def sanitize_key(raw: str, *, fallback: str = "task", max_len: int | None = None) -> str:
    value = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in (raw or fallback)).strip("._-")
    value = value or fallback
    if max_len is not None and max_len > 0:
        value = value[:max_len].rstrip("._-") or fallback
    return value


def scoped_task_label(experiment_name: str, task_id: str) -> str:
    # Keep task-scoped workspace keys within the runtime validator's 128-char cap.
    # The digest preserves determinism and uniqueness even after truncation.
    task_part = sanitize_key(task_id, fallback="task", max_len=80)
    experiment_part = sanitize_key(experiment_name, fallback="default", max_len=24)
    raw_task_id = task_id or "task"
    raw_experiment = experiment_name or "default"
    digest = hashlib.sha1(f"{raw_experiment}::{raw_task_id}".encode("utf-8")).hexdigest()[:10]
    return f"{experiment_part}__{task_part}__{digest}"


def task_workspace_key(task_id: str, experiment_name: str = "") -> str:
    return f"task__{scoped_task_label(experiment_name, task_id)}"


def default_knowledge_db_path(experiment_name: str) -> Path:
    # Per-experiment subdir so that /app/memory-db exposes only this
    # experiment's files. The runtime mounts individual allowlisted
    # files/subdirs under it rather than the whole
    # directory (the runtime-DB sibling and per-task snapshot derive from
    # this parent).
    stem = sanitize_key(experiment_name, fallback="ksi")
    return RUNTIME_KNOWLEDGE_DIR / stem / f"{stem}_knowledge.sqlite"


def legacy_flat_knowledge_db_path(experiment_name: str) -> Path:
    """Legacy flat layout: runtime_state/knowledge/<exp>_knowledge.sqlite.

    Retained for resolve-in-place back-compat so a pre-existing experiment
    resumes against its flat DB instead of starting fresh in the new subdir.
    """
    stem = sanitize_key(experiment_name, fallback="ksi")
    return RUNTIME_KNOWLEDGE_DIR / f"{stem}_knowledge.sqlite"


def default_runtime_db_path(experiment_name: str) -> Path:
    return RUNTIME_AUDIT_DIR / f"{sanitize_key(experiment_name, fallback='ksi')}_runtime.sqlite"


def derive_runtime_sibling(knowledge_db_path: str) -> str:
    """Derive an optional runtime audit DB beside or near a knowledge DB path."""
    path = Path(knowledge_db_path).resolve()
    parent = path.parent
    name = path.name
    if name.endswith(".sqlite"):
        stem = path.stem
        if stem.endswith("_knowledge"):
            return str(parent / f"{stem[: -len('_knowledge')]}_runtime.sqlite")
        return str(parent / f"{stem}_runtime.sqlite")
    return str(parent / "runtime.sqlite")


def derive_legacy_sibling(memory_db_path: str, kind: str) -> str:
    """Derive a per-experiment sibling DB path from a legacy runtime DB path.

    Given ``<dir>/<experiment>_runtime.sqlite`` or
    ``<dir>/<experiment>_memory.sqlite`` this returns
    ``<dir>/<experiment>_<kind>.sqlite`` so that per-experiment state
    (knowledge / forum / task_docs) stays isolated per run. Legacy generic
    names (``swarms.sqlite``, ``task_memory.sqlite`` — historical on-disk
    artifacts that pre-date the ksi rename, so the old spelling is a
    deliberate keep) collapse to ``<dir>/<kind>.sqlite`` to preserve
    backwards compatibility with the pre-existing migration layout.
    """
    path = Path(memory_db_path).resolve()
    parent = path.parent
    name = path.name
    if name in {"swarms.sqlite", "task_memory.sqlite"}:
        return str(parent / f"{kind}.sqlite")
    if name.endswith(".sqlite"):
        stem = path.stem
        for suffix in ("_runtime", "_memory"):
            if stem.endswith(suffix):
                return str(parent / f"{stem[: -len(suffix)]}_{kind}.sqlite")
        return str(parent / f"{stem}_{kind}.sqlite")
    return str(parent / f"{kind}.sqlite")


def default_swebench_repo_cache_dir(task_source: str) -> Path:
    # task_source parameter kept for call-site stability; only swebench_pro remains.
    return BENCHMARKS_DIR / "swebench_pro" / "repo_cache"

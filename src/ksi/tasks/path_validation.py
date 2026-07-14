"""Per-source ``--tasks-path`` validators.

The cli used a ``tasks_path_kind`` if/elif chain to validate ``--tasks-path``
(and, for swebench, ``--evals-path``) per task source. Those legs move here as
standalone validators attached to each ``TaskSourceSpec.validate_tasks_path``
via the same post-hoc registry wiring used for ``loader``. Each validator
returns the exact ``parser.error`` message string (byte-identical to the former
cli branch) or ``None`` when the path is acceptable.
"""

from __future__ import annotations

from pathlib import Path

# `ksi.tasks.__init__` imports this module alphabetically-before `.loaders`
# (isort/ruff pin that ordering), but the attach below needs the benchmark +
# custom TaskSourceSpecs already registered — which only happens as a side
# effect of importing `.loaders`. Import it explicitly here (not just rely on
# `tasks/__init__.py` eventually importing it) so this module's own
# registration trigger doesn't depend on import order elsewhere.
from . import loaders as _loaders  # noqa: F401
from .registry import REGISTRY, register_task_source


def _validate_arc(tasks_path: Path, *, evals_path: Path | None) -> str | None:
    if not tasks_path.exists():
        return f"--tasks-path for --task-source arc must exist (json file or directory): {tasks_path}"
    if tasks_path.is_file() and tasks_path.suffix.lower() != ".json":
        return f"--tasks-path for --task-source arc file input must be .json: {tasks_path}"
    return None


def _validate_polyglot(tasks_path: Path, *, evals_path: Path | None) -> str | None:
    if not tasks_path.exists() or not tasks_path.is_file() or tasks_path.suffix.lower() != ".json":
        return f"--tasks-path for --task-source polyglot must be an existing .json file: {tasks_path}"
    return None


def _validate_swebench_pro(tasks_path: Path, *, evals_path: Path | None) -> str | None:
    if not tasks_path.exists():
        return f"--tasks-path for --task-source swebench_pro must exist: {tasks_path}"
    if tasks_path.suffix.lower() not in {".parquet", ".csv", ".jsonl"}:
        return f"--tasks-path for --task-source swebench_pro must be .parquet, .csv, or .jsonl: {tasks_path}"
    if evals_path and evals_path.suffix.lower() != ".parquet":
        return f"--evals-path must be a parquet file (.parquet): {evals_path}"
    return None


def _validate_terminal_bench_2(tasks_path: Path, *, evals_path: Path | None) -> str | None:
    if not tasks_path.exists() or not tasks_path.is_file() or tasks_path.suffix.lower() != ".json":
        return f"--tasks-path for --task-source terminal_bench_2 must be an existing .json task map: {tasks_path}"
    return None


_VALIDATORS = {
    "arc": _validate_arc,
    "polyglot": _validate_polyglot,
    "swebench_pro": _validate_swebench_pro,
    "terminal_bench_2": _validate_terminal_bench_2,
}


def _attach_registry_path_validators() -> None:
    """Attach the built-in ``--tasks-path`` validators to their specs.

    Mirrors ``ksi.benchmarks.loaders.attach_benchmark_loaders``; idempotent.
    """
    from dataclasses import replace as dataclass_replace

    for name, validator in _VALIDATORS.items():
        spec = REGISTRY.get(name)
        if spec is not None and spec.validate_tasks_path is None:
            register_task_source(dataclass_replace(spec, validate_tasks_path=validator), replace=True)


_attach_registry_path_validators()

"""Benchmark-specific task loaders (swebench_pro, arc, polyglot, terminal_bench_2).

``ksi.benchmarks.register_all()`` calls ``attach_benchmark_loaders()`` (below)
to wire these loader callables onto the specs registered by
``ksi.benchmarks.sources``. Generic loading infra (tabular-row readers,
``load_tasks_for_source``, ``load_eval_records_for_source``, task
classification) stays in ``ksi.tasks.loaders`` — this module imports it from
there rather than duplicating it.
"""

from __future__ import annotations

import ast
import json
import logging
import math
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

from ..models import TaskSpec
from ..tasks.loaders import _load_rows, _load_tabular_rows, load_eval_records_for_source
from ..tasks.registry import register_task_source, resolve_source

log = logging.getLogger(__name__)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_tb2_source_root(tasks_path: Path, task_map: dict[str, Any]) -> Path:
    raw = str(task_map.get("source_path") or "").strip()
    if not raw:
        raise ValueError(f"terminal_bench_2 task map missing source_path: {tasks_path}")
    candidate = Path(raw).expanduser()
    was_relative = not candidate.is_absolute()
    if was_relative:
        candidate = (_repo_root() / candidate).resolve()
        # A relative source_path must resolve inside the repo — reject `../..`
        # traversal from a task map of unknown origin (mirrors the guard
        # benchmarks/scripts/dataprep/validate_task_map._is_within_repo applies for ARC).
        # An explicit absolute path stays the caller's deliberate choice.
        try:
            candidate.relative_to(_repo_root().resolve())
        except ValueError:
            raise ValueError(f"terminal_bench_2 relative source_path escapes the repo: {raw!r}") from None
    else:
        candidate = candidate.resolve()
    if not candidate.exists() or not candidate.is_dir():
        raise ValueError(f"terminal_bench_2 source_path does not exist: {candidate}")
    return candidate


def _read_tb2_task_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"terminal_bench_2 task.toml is not a table: {path}")
    return payload


def _build_terminal_bench_2_metadata(
    *,
    task_id: str,
    task_root: Path,
    task_item: dict[str, Any],
    task_map: dict[str, Any],
) -> dict[str, Any]:
    instruction_path = task_root / "instruction.md"
    task_toml_path = task_root / "task.toml"
    if not instruction_path.is_file():
        raise ValueError(f"terminal_bench_2 task missing instruction.md: {instruction_path}")
    if not task_toml_path.is_file():
        raise ValueError(f"terminal_bench_2 task missing task.toml: {task_toml_path}")

    instruction_text = instruction_path.read_text(encoding="utf-8")
    task_toml_text = task_toml_path.read_text(encoding="utf-8")
    task_toml = _read_tb2_task_toml(task_toml_path)
    env_raw = task_toml.get("environment")
    env = env_raw if isinstance(env_raw, dict) else {}

    return {
        "task_source": "terminal_bench_2",
        "task_root": str(task_root.resolve()),
        "instruction_path": str(instruction_path.resolve()),
        "task_toml_path": str(task_toml_path.resolve()),
        "selection_name": str(task_map.get("selection_name") or "").strip(),
        "dataset_name": str(task_map.get("dataset_name") or "").strip(),
        "source_path": str(task_map.get("source_path") or "").strip(),
        "source_git_revision": str(task_map.get("source_git_revision") or "").strip(),
        "task_index": task_item.get("index"),
        "category": str(task_item.get("category") or "").strip(),
        "difficulty": str(task_item.get("difficulty") or "").strip(),
        "docker_image": str(task_item.get("docker_image") or env.get("docker_image") or "").strip(),
        # Provenance only: these are the values copied from the task map at
        # map-generation time. Runtime does NOT read them back — it rereads the
        # authoritative ``task.toml`` (see ``resolve_terminal_bench_2_task_contract``)
        # and records ``timeout_source = "task.toml"`` in preflight/trial metadata.
        "agent_timeout_sec": task_item.get("agent_timeout_sec"),
        "verifier_timeout_sec": task_item.get("verifier_timeout_sec"),
        "timeout_source": "task_map",
        "notes": str(task_item.get("notes") or "").strip(),
        "task_files": {
            "tb2/instruction.md": instruction_text,
            "tb2/task.toml": task_toml_text,
        },
    }


def _load_terminal_bench_2_tasks(tasks_path: Path) -> list[TaskSpec]:
    if tasks_path.suffix.lower() != ".json":
        raise ValueError(f"terminal_bench_2 source expects a .json task map, got: {tasks_path}")
    payload = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"terminal_bench_2 task map must be a JSON object: {tasks_path}")
    tasks_raw = payload.get("tasks")
    if not isinstance(tasks_raw, list):
        raise ValueError(f"terminal_bench_2 task map must contain a tasks array: {tasks_path}")

    source_root = _resolve_tb2_source_root(tasks_path, payload)
    tasks: list[TaskSpec] = []
    for item in tasks_raw:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id") or "").strip()
        if not task_id:
            continue
        task_root = source_root / task_id
        metadata = _build_terminal_bench_2_metadata(
            task_id=task_id,
            task_root=task_root,
            task_item=item,
            task_map=payload,
        )
        tasks.append(
            TaskSpec(
                id=task_id,
                repo="",
                prompt="Solve the native Terminal-Bench 2 task described in instruction.md.",
                metadata=metadata,
            )
        )
    return tasks


def _load_swebench_pro_tasks(tasks_path: Path, eval_records: dict[str, dict[str, Any]]) -> list[TaskSpec]:
    rows = _load_tabular_rows(tasks_path)
    return _build_swebench_tasks(rows, eval_records=eval_records, task_source="swebench_pro")


_MISSING_FIELD_SENTINELS = {"", "none", "null", "nan", "na", "n/a", "<na>"}


def _is_missing_field_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        if value != value:
            return True
    except Exception:
        pass
    return False


def _format_jsonish_field(value: Any) -> str:
    if _is_missing_field_value(value):
        return ""
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in _MISSING_FIELD_SENTINELS:
            return ""
        if (
            (text.startswith("{") and text.endswith("}"))
            or (text.startswith("[") and text.endswith("]"))
            or (text.startswith('"') and text.endswith('"'))
        ):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                    break
                except Exception:
                    parsed = value
        else:
            parsed = value
    if _is_missing_field_value(parsed):
        return ""
    if isinstance(parsed, (dict, list)):
        return json.dumps(parsed, indent=2, ensure_ascii=True)
    text = str(parsed).strip()
    if text.lower() in _MISSING_FIELD_SENTINELS:
        return ""
    return text


def _build_swebench_problem(row: dict[str, Any], *, task_source: str) -> str:
    problem = str(row.get("problem_statement") or row.get("problem_description") or row.get("prompt") or "").strip()
    spec = resolve_source(task_source)
    if spec is None or not spec.uses_repo_snapshots:
        return problem

    sections: list[str] = []
    requirements = _format_jsonish_field(row.get("requirements"))
    if requirements:
        sections.extend(["## Requirements", requirements])
    interface = _format_jsonish_field(row.get("interface"))
    if interface:
        sections.extend(["## Interface", interface])
    if not sections:
        return problem
    return "\n\n".join([problem, *sections] if problem else sections).strip()


def _build_swebench_tasks(
    rows: list[dict[str, Any]],
    *,
    eval_records: dict[str, dict[str, Any]],
    task_source: str,
) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for row in rows:
        instance_id = str(row.get("instance_id") or "").strip()
        repo = str(row.get("repo") or "").strip()
        if not instance_id:
            continue
        problem = _build_swebench_problem(row, task_source=task_source)
        base_commit = row.get("base_commit")
        if not base_commit and instance_id in eval_records:
            base_commit = eval_records[instance_id].get("base_commit")
        dockerhub_tag = row.get("dockerhub_tag")
        image_name = row.get("image_name")
        tasks.append(
            TaskSpec(
                id=instance_id,
                repo=repo,
                prompt=problem,
                metadata={
                    "task_source": task_source,
                    "instance_id": instance_id,
                    "hints_text": row.get("hints_text"),
                    "image_assets": row.get("image_assets"),
                    "base_commit": base_commit,
                    "dockerhub_tag": dockerhub_tag,
                    "image_name": image_name,
                    "fail_to_pass": row.get("fail_to_pass") or row.get("FAIL_TO_PASS"),
                    "pass_to_pass": row.get("pass_to_pass") or row.get("PASS_TO_PASS"),
                    "before_repo_set_cmd": row.get("before_repo_set_cmd"),
                    "selected_test_files_to_run": row.get("selected_test_files_to_run"),
                },
            )
        )
    return tasks


def _arc_json_files(tasks_path: Path) -> list[Path]:
    """Return ARC task json files from a file or directory input."""
    if tasks_path.is_file():
        if tasks_path.suffix.lower() != ".json":
            raise ValueError(f"ARC source expects .json task files, got: {tasks_path}")
        return [tasks_path]

    if not tasks_path.is_dir():
        raise ValueError(f"ARC source expects a file or directory, got: {tasks_path}")

    # Prefer direct json files for split directories like .../training or .../evaluation.
    direct = sorted(tasks_path.glob("*.json"))
    if direct:
        return direct

    # Fallback for ARC roots like .../data containing one split subdir. Mixing
    # training and evaluation silently contaminates benchmark selection, so fail
    # closed and require the caller to pass the intended split directory.
    split_file_groups: dict[str, list[Path]] = {}
    for split in ("training", "evaluation"):
        split_dir = tasks_path / split
        if split_dir.is_dir():
            files = sorted(split_dir.glob("*.json"))
            if files:
                split_file_groups[split] = files
    if len(split_file_groups) > 1:
        splits = ", ".join(sorted(split_file_groups))
        raise ValueError(
            f"ARC source root {tasks_path} contains multiple split directories with JSON files ({splits}); "
            "pass the intended split directory, e.g. .../training or .../evaluation"
        )
    if split_file_groups:
        return next(iter(split_file_groups.values()))

    # Last-resort recursive scan.
    recursive = sorted(tasks_path.rglob("*.json"))
    if not recursive:
        raise ValueError(f"No ARC json task files found under {tasks_path}")
    return recursive


def _load_arc_tasks(tasks_path: Path, *, max_trials: int = 2) -> list[TaskSpec]:
    files = _arc_json_files(tasks_path)
    tasks: list[TaskSpec] = []
    seen: set[str] = set()
    effective_max_trials = max(1, int(max_trials))

    for file_path in files:
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping ARC task %s (invalid JSON): %s", file_path, exc)
            continue

        if not isinstance(raw, dict):
            log.warning("Skipping ARC task %s (expected object)", file_path)
            continue

        train_raw = raw.get("train", [])
        test_raw = raw.get("test", [])
        if not isinstance(train_raw, list) or not isinstance(test_raw, list):
            log.warning("Skipping ARC task %s (train/test must be arrays)", file_path)
            continue

        train_pairs = [p for p in train_raw if isinstance(p, dict)]
        test_pairs = [p for p in test_raw if isinstance(p, dict)]
        if not test_pairs:
            log.warning("Skipping ARC task %s (no test pairs)", file_path)
            continue

        split = file_path.parent.name if file_path.parent.name in {"training", "evaluation"} else ""
        stem = file_path.stem
        task_id = stem
        if task_id in seen:
            prefix = split or "arc"
            task_id = f"{prefix}__{stem}"
            suffix = 2
            while task_id in seen:
                task_id = f"{prefix}__{stem}__{suffix}"
                suffix += 1
        seen.add(stem)
        seen.add(task_id)

        test_inputs = [{"input": p.get("input")} for p in test_pairs]

        tasks.append(
            TaskSpec(
                id=task_id,
                repo="",
                prompt="Infer the transformation rule from train pairs and solve the test grid.",
                metadata={
                    "task_source": "arc",
                    "arc_split": split or None,
                    "arc_source_file": str(file_path),
                    "arc_train_pairs": train_pairs,
                    "arc_test_inputs": test_inputs,
                    "arc_eval_test_pairs": test_pairs,
                    "arc_max_trials": effective_max_trials,
                },
            )
        )

    return tasks


def _load_swebench_eval_records(evals_path: Path) -> dict[str, dict[str, Any]]:
    rows = _load_rows(evals_path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        instance_id = str(row.get("instance_id") or "").strip()
        if instance_id:
            out[instance_id] = row
    return out


def _load_polyglot_tasks(
    tasks_path: Path,
    *,
    polyglot_test_feedback_tries: int = 2,
    polyglot_test_feedback_max_lines: int = 50,
) -> list[TaskSpec]:
    """Load polyglot coding tasks from a JSON file.

    Each entry in the JSON array must have at least ``language`` and
    ``exercise_name``.  An ``instance_id`` field is used when present;
    otherwise the task id is synthesised as ``<language>__<exercise_name>``.

    Returns a list of :class:`TaskSpec` with rich metadata suitable for
    multi-language code-generation evaluation.
    """
    if tasks_path.suffix.lower() != ".json":
        raise ValueError(f"Polyglot source expects a .json file, got: {tasks_path}")

    raw = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Polyglot JSON must be an array, got {type(raw).__name__}")

    tasks: list[TaskSpec] = []
    for i, entry in enumerate(raw, start=1):
        if not isinstance(entry, dict):
            log.warning("Skipping polyglot entry %d (expected object, got %s)", i, type(entry).__name__)
            continue

        language = str(entry.get("language") or "").strip()
        exercise_name = str(entry.get("exercise_name") or "").strip()
        if not language or not exercise_name:
            log.warning("Skipping polyglot entry %d (missing language or exercise_name)", i)
            continue

        task_id = str(entry.get("instance_id") or entry.get("id") or "").strip()
        if not task_id:
            task_id = f"{language}__{exercise_name}"

        prompt = str(entry.get("problem_statement") or entry.get("prompt") or "").strip()
        if not prompt:
            prompt = f"Implement the {exercise_name!r} exercise in {language}."

        tasks.append(
            TaskSpec(
                id=task_id,
                repo="",
                prompt=str(prompt),
                metadata={
                    "task_source": "polyglot",
                    "language": language,
                    "exercise_name": exercise_name,
                    "starter_code": entry.get("starter_code", {}),
                    "test_files": entry.get("test_files", {}),
                    "reference_solution": entry.get("reference_solution", ""),
                    "build_files": entry.get("build_files", {}),
                    "test_command": entry.get("test_command", ""),
                    "meta_config": entry.get("meta_config", {}),
                    "polyglot_test_feedback_tries": polyglot_test_feedback_tries,
                    "polyglot_test_feedback_max_lines": polyglot_test_feedback_max_lines,
                },
            )
        )

    return tasks


# ── Registry loader wiring ────────────────────────────────────────────────────
#
# ``TaskSourceSpec.loader`` callables share one contract: ``load_tasks_for_source``
# invokes ``loader(tasks_path, *, task_source=..., evals_path=..., arc_max_trials=...)``
# and expects a list of TaskSpec back. The adapters below normalize the built-in
# per-source loader signatures to that contract; custom loaders should accept
# ``tasks_path`` plus ``**kwargs`` for forward compatibility.


def _load_swebench_pro_source(
    tasks_path: Path,
    *,
    task_source: str = "swebench_pro",
    evals_path: Path | None = None,
    **_kwargs: Any,
) -> list[TaskSpec]:
    eval_records = load_eval_records_for_source(task_source=task_source, evals_path=evals_path)
    return _load_swebench_pro_tasks(tasks_path, eval_records)


def _load_arc_source(tasks_path: Path, *, arc_max_trials: int = 2, **_kwargs: Any) -> list[TaskSpec]:
    return _load_arc_tasks(tasks_path, max_trials=arc_max_trials)


def _load_polyglot_source(tasks_path: Path, **_kwargs: Any) -> list[TaskSpec]:
    tries_raw = _kwargs.get("polyglot_test_feedback_tries")
    max_lines_raw = _kwargs.get("polyglot_test_feedback_max_lines")
    return _load_polyglot_tasks(
        tasks_path,
        polyglot_test_feedback_tries=int(tries_raw) if tries_raw is not None else 2,
        polyglot_test_feedback_max_lines=int(max_lines_raw) if max_lines_raw is not None else 50,
    )


def _load_terminal_bench_2_source(tasks_path: Path, **_kwargs: Any) -> list[TaskSpec]:
    return _load_terminal_bench_2_tasks(tasks_path)


def attach_benchmark_loaders() -> None:
    """Attach the built-in benchmark loader callables to their registered specs.

    Called by ``ksi.benchmarks.register_all()``. Idempotent: a spec that
    already carries a loader is left untouched.
    """
    for name, loader_fn in (
        ("swebench_pro", _load_swebench_pro_source),
        ("arc", _load_arc_source),
        ("polyglot", _load_polyglot_source),
        ("terminal_bench_2", _load_terminal_bench_2_source),
    ):
        spec = resolve_source(name)
        if spec is not None and spec.loader is None:
            register_task_source(dataclass_replace(spec, loader=loader_fn), replace=True)

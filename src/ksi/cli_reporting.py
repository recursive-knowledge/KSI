"""Run-output reporting for the KSI CLI.

Extracted from ``cli.py``: the ``--output-json`` / ``--pretask-debug-json``
result writers plus the trace-serialization helpers they use. Keeping the
reporting concern in its own module shrinks ``main()`` and lets the writers be
tested without driving the whole CLI. Behavior is unchanged — the payload
shape, atomic temp-file replace, and log lines are identical to the inline
version. ``cli`` re-exports the helper functions so existing
``from ksi.cli import ...`` test imports keep working.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import TaskSpec, TaskTrace
from .tasks import resolve_source

log = logging.getLogger("ksi.cli")


def _resolve_arc_split(tasks: list[TaskSpec]) -> str | None:
    """Best-effort ARC split label for the run-artifact JSON.

    ``TaskSpec.metadata["arc_split"]`` is set per task by ``_load_arc_tasks``
    from the source file's parent dir name. Non-ARC runs, or ARC runs whose
    tasks disagree on split (mixed task maps), return None rather than guess.
    """
    splits = {t.metadata.get("arc_split") for t in tasks if t.metadata.get("arc_split")}
    if len(splits) == 1:
        return next(iter(splits))
    return None


_TASK_MAP_METADATA_FIELDS = (
    "benchmark",
    "split",
    "selection_name",
    "source_repo",
    "source_branch",
    "source_commit",
    "source_path",
    "selection_algorithm",
    "seed",
    "selection_seed",
    "count",
    "disjoint_from",
    "excluded_maps",
    "ids_file",
    "parent_pool_file",
)


def _task_ids_from_map_payload(payload: Any) -> list[str]:
    if isinstance(payload, list):
        return [str(item).strip() for item in payload if str(item).strip()]
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("task_ids"), list):
        return [str(item).strip() for item in payload["task_ids"] if str(item).strip()]
    if isinstance(payload.get("tasks"), list):
        return [
            str(item.get("task_id") or "").strip()
            for item in payload["tasks"]
            if isinstance(item, dict) and str(item.get("task_id") or "").strip()
        ]
    return []


def _task_map_metadata(task_map_path: str | None) -> dict[str, Any] | None:
    """Load compact task-map identity/provenance for result JSON artifacts."""
    if not task_map_path:
        return None
    path = Path(task_map_path)
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    task_ids = _task_ids_from_map_payload(payload)
    meta: dict[str, Any] = {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "task_ids_count": len(task_ids),
        "task_ids_sha256": hashlib.sha256("\n".join(task_ids).encode("utf-8")).hexdigest(),
    }
    if isinstance(payload, dict):
        for field in _TASK_MAP_METADATA_FIELDS:
            if field in payload:
                meta[field] = payload[field]
    return meta


def _serialize_trace_with_retry_summary(trace) -> dict:
    """Serialize a :class:`TaskTrace` and lift retry forensics to the top level.

    ``retry_attempts`` and ``runtime_attempt_errors`` live in ``runtime_meta``
    so the runtime DB carries the retry forensics on the ``token_phases`` /
    ``attempts`` rows. That data does NOT otherwise appear on the per-experiment
    results JSON written by ``--output-json`` — analysts would have to join
    through the runtime DB to count retries.

    To make the data visible to ``jq`` and downstream analysis tooling
    without a DB join, we lift two fields onto the trace dict:

    - ``retry_attempts`` (int, defaults to 0): number of failed attempts
      before the recorded outcome. Mirrors ``runtime_meta["retry_attempts"]``
      when present; otherwise zero.
    - ``runtime_attempt_errors_count`` (int, defaults to 0): length of the
      ``runtime_attempt_errors`` list, surfaced as a count so dashboards can
      consume it cheaply without parsing the nested error dicts.

    The original ``runtime_meta`` dict is preserved untouched so existing
    consumers that already reach into it keep working.
    """
    payload = asdict(trace)
    runtime_meta = payload.get("runtime_meta")
    retry_attempts = 0
    errors_count = 0
    if isinstance(runtime_meta, dict):
        raw_retry = runtime_meta.get("retry_attempts")
        if isinstance(raw_retry, bool):
            # bool is an int subclass — guard so a stray True doesn't read as 1.
            raw_retry = int(raw_retry)
        if isinstance(raw_retry, int) and raw_retry >= 0:
            retry_attempts = raw_retry
        elif isinstance(raw_retry, (float, str)):
            try:
                coerced = int(raw_retry)
                if coerced >= 0:
                    retry_attempts = coerced
            except (TypeError, ValueError):
                pass
        errors = runtime_meta.get("runtime_attempt_errors")
        if isinstance(errors, list):
            errors_count = len(errors)
    payload["retry_attempts"] = retry_attempts
    payload["runtime_attempt_errors_count"] = errors_count
    return payload


def _tb2_recent_commands(trace: TaskTrace) -> list[str]:
    commands: list[str] = []
    for entry in trace.tool_trace or []:
        if not isinstance(entry, dict):
            continue
        command = entry.get("command")
        if isinstance(command, str) and command.strip():
            commands.append(command.strip())
            continue
        tool_args = entry.get("tool_args")
        if isinstance(tool_args, dict):
            nested = tool_args.get("command")
            if isinstance(nested, str) and nested.strip():
                commands.append(nested.strip())
                continue
        tool_input = entry.get("tool_input")
        if isinstance(tool_input, dict):
            nested = tool_input.get("command")
            if isinstance(nested, str) and nested.strip():
                commands.append(nested.strip())
    return commands[-3:]


def _build_tb2_results_summary(traces: list[TaskTrace]) -> dict[str, object]:
    """Build a compact, benchmark-facing TB2 summary for ``--output-json``."""
    by_task: dict[str, dict[str, object]] = {}
    by_generation: dict[int, dict[str, object]] = {}

    for trace in traces:
        runtime_meta = trace.runtime_meta if isinstance(trace.runtime_meta, dict) else {}
        reward_raw = runtime_meta.get("reward", trace.native_score)
        if reward_raw is None or reward_raw == "":
            reward = None
        else:
            try:
                reward = float(reward_raw)
            except (TypeError, ValueError):
                reward = 0.0
        # A legacy non-finite stored reward parses without raising;
        # keep it out of the aggregates (best_reward/reward_total/averages) by
        # treating it as 0.0, same as an unparseable value.
        if reward is not None and not math.isfinite(reward):
            reward = 0.0
        scored = reward is not None
        resolved = bool((trace.eval_result or {}).get("resolved", scored and reward >= 1.0))
        verifier_stdout_tail = str(runtime_meta.get("verifier_stdout_tail") or "").strip()
        verifier_stderr_tail = str(runtime_meta.get("verifier_stderr_tail") or "").strip()
        trial_status = str(runtime_meta.get("trial_status") or "").strip()
        task_entry = by_task.setdefault(
            trace.task_id,
            {
                "task_id": trace.task_id,
                "best_reward": None,
                "solved": False,
                "attempts": [],
            },
        )
        attempts = task_entry["attempts"]
        assert isinstance(attempts, list)
        attempts.append(
            {
                "generation": trace.generation,
                "agent_id": trace.agent_id,
                "reward": reward,
                "scored": scored,
                "unscored_reason": "" if scored else trial_status or "missing_reward",
                "resolved": resolved,
                "agent_exit_code": runtime_meta.get("agent_exit_code"),
                "verifier_exit_code": runtime_meta.get("verifier_exit_code"),
                "trial_status": trial_status,
                "tool_count": len(trace.tool_trace or []),
                "recent_commands": _tb2_recent_commands(trace),
                "verifier_stdout_tail": verifier_stdout_tail,
                "verifier_stderr_tail": verifier_stderr_tail,
                "error": trace.error,
            }
        )
        if scored:
            current_best = task_entry["best_reward"]
            task_entry["best_reward"] = reward if current_best is None else max(float(current_best), reward)
        task_entry["solved"] = bool(task_entry["solved"]) or resolved

        gen_entry = by_generation.setdefault(
            int(trace.generation),
            {
                "generation": int(trace.generation),
                "attempts": 0,
                "scored_attempts": 0,
                "resolved": 0,
                "reward_total": 0.0,
            },
        )
        gen_entry["attempts"] = int(gen_entry["attempts"]) + 1
        gen_entry["resolved"] = int(gen_entry["resolved"]) + (1 if resolved else 0)
        if scored:
            gen_entry["scored_attempts"] = int(gen_entry["scored_attempts"]) + 1
            gen_entry["reward_total"] = float(gen_entry["reward_total"]) + reward

    task_rows = []
    for task_id in sorted(by_task):
        task_entry = by_task[task_id]
        attempts = sorted(task_entry["attempts"], key=lambda item: (int(item["generation"]), str(item["agent_id"])))
        best_reward = task_entry["best_reward"]
        task_rows.append(
            {
                "task_id": task_id,
                "best_reward": float(best_reward) if best_reward is not None else None,
                "solved": bool(task_entry["solved"]),
                "attempts": attempts,
            }
        )

    generation_rows = []
    for generation in sorted(by_generation):
        entry = by_generation[generation]
        attempts = int(entry["attempts"])
        scored_attempts = int(entry["scored_attempts"])
        generation_rows.append(
            {
                "generation": generation,
                "attempts": attempts,
                "scored_attempts": scored_attempts,
                "resolved": int(entry["resolved"]),
                "average_reward": (float(entry["reward_total"]) / scored_attempts if scored_attempts else None),
            }
        )

    solved_total = sum(1 for row in task_rows if bool(row["solved"]))
    best_rewards = [float(row["best_reward"]) for row in task_rows if row["best_reward"] is not None]
    return {
        "tasks_total": len(task_rows),
        "tasks_solved": solved_total,
        "attempts_total": sum(len(row["attempts"]) for row in task_rows),
        "average_best_reward": (sum(best_rewards) / len(best_rewards) if best_rewards else None),
        "per_task": task_rows,
        "by_generation": generation_rows,
    }


def _atomic_write_json(out_path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as pretty JSON to ``out_path`` via a temp-file replace."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(out_path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_f:
            json.dump(payload, tmp_f, indent=2)
        os.replace(tmp_path, str(out_path))
    except BaseException:
        os.unlink(tmp_path)
        raise
    log.info("wrote %s", out_path)


def _write_once_run_provenance(
    args,
    *,
    code_commit: str,
    resolved_model: str,
    scoring_mode: str,
) -> dict[str, str]:
    values = {
        "code_commit": code_commit,
        "resolved_model": resolved_model,
        "scoring_mode": scoring_mode,
    }
    experiment = str(getattr(args, "experiment_name", "") or "default")
    seen: set[Path] = set()
    for raw_path in (getattr(args, "runtime_db_path", ""), getattr(args, "knowledge_db_path", "")):
        if not raw_path:
            continue
        path = Path(raw_path)
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=0.1)
            try:
                row = conn.execute(
                    "SELECT code_commit, resolved_model, scoring_mode FROM runs WHERE experiment = ?",
                    (experiment,),
                ).fetchone()
            finally:
                conn.close()
        except sqlite3.Error:
            continue
        if row is None:
            continue
        for key, value in zip(("code_commit", "resolved_model", "scoring_mode"), row, strict=True):
            if value is not None and str(value):
                values[key] = str(value)
    return values


def write_output_json(
    args,
    *,
    traces: list[TaskTrace],
    orchestrator,
    collecting,
    tasks,
    code_commit: str,
    resolved_model: str,
    scoring_mode: str,
    run_complete: bool,
) -> None:
    """Write the per-experiment results JSON for ``--output-json``.

    Materialises traces and surfaces retry forensics at the top level so
    downstream analysis tooling can read
    ``retry_attempts`` directly via ``jq`` without joining through the runtime
    DB. The retry data lives in ``runtime_meta`` but is keyed differently per
    attempt path; we lift it onto each trace dict and aggregate it for the run.
    Without this lift the ``retry_count``-style fields are invisible in the
    per-experiment JSON.

    ``run_complete`` distinguishes a best-effort per-generation snapshot
    (``False``, fired via ``collecting.on_generation_snapshot`` so a mid-run
    crash doesn't lose every trace collected up to that point) from the
    final, authoritative write after ``orchestrator.run()`` completes
    (``True``) — both otherwise have an identical payload shape.
    """
    serialized_traces = [_serialize_trace_with_retry_summary(t) for t in traces]
    total_retry_attempts = sum(int(t.get("retry_attempts", 0) or 0) for t in serialized_traces)
    traces_retried_count = sum(1 for t in serialized_traces if int(t.get("retry_attempts", 0) or 0) > 0)
    provenance = _write_once_run_provenance(
        args,
        code_commit=code_commit,
        resolved_model=resolved_model,
        scoring_mode=scoring_mode,
    )
    payload = {
        "args": vars(args),
        "num_tasks": len(tasks),
        "arc_split": _resolve_arc_split(tasks),
        # Provenance stamp: mirror the write-once runs row when it
        # exists, so resumed output JSON stays pinned to the original run stamp
        # instead of the current launch's HEAD/model.
        "code_commit": provenance["code_commit"],
        "resolved_model": provenance["resolved_model"],
        "scoring_mode": provenance["scoring_mode"],
        "run_complete": run_complete,
        "num_traces": len(traces),
        "total_retry_attempts": total_retry_attempts,
        "traces_retried_count": traces_retried_count,
        "traces": serialized_traces,
        "assignments": collecting.assignments,
        "generation_end": collecting.generation_end,
        "token_usage_total": collecting.token_summary,
        "token_usage_breakdown": orchestrator.accumulator.to_dict(),
        # Hold-out transfer probe: per-gen NON-cumulative solve rate
        # over --holdout-task-ids; {} when the feature is unused.
        "holdout_solve_rate_by_generation": orchestrator.holdout_solve_rate_by_generation(),
        # Per-generation knowledge-phase degradation counts:
        # {gen: {drain_failures, forum_agent_failures, distill_failures,
        # seed_failures}}; {} when every generation's knowledge phases
        # were healthy. ``..._measured`` disambiguates a clean {} from a
        # --no-memory run that never measured these phases.
        "knowledge_phase_health": orchestrator.knowledge_phase_health_by_generation(),
        "knowledge_phase_health_measured": orchestrator.knowledge_phase_health_measured(),
    }
    task_map_meta = _task_map_metadata(getattr(args, "task_map_path", None))
    if task_map_meta is not None:
        payload["task_map"] = task_map_meta
    tb2_spec = resolve_source(args.task_source)
    if tb2_spec is not None and tb2_spec.delegates_runtime:
        payload["tb2_summary"] = _build_tb2_results_summary(traces)
    _atomic_write_json(Path(args.output_json), payload)


def write_pretask_debug_json(args, *, orchestrator) -> None:
    """Write the pre-task claim-debug history for ``--pretask-debug-json``."""
    debug_payload = {
        "args": vars(args),
        "claim_debug_history": orchestrator.get_claim_debug_history(),
    }
    _atomic_write_json(Path(args.pretask_debug_json), debug_payload)

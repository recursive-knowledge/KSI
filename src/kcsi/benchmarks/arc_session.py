from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any

from ..memory.arc_semantics import compare_grids, normalize_grid
from ..models import TaskSpec

log = logging.getLogger(__name__)


def _is_grid_like(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(row, list) for row in value)


# --- Tool-trace reconstruction: canonical ARC blind scoring ----------------


_ARC_SET_GRID_TOOL_NAMES = frozenset({"arc_set_output_grid"})
_ARC_RESIZE_GRID_TOOL_NAMES = frozenset({"arc_resize_output_grid"})
_ARC_SUBMIT_TRIAL_TOOL_NAMES = frozenset({"arc_submit_trial"})
_ARC_NEXT_TEST_TOOL_NAMES = frozenset({"arc_next_test_input"})


def _coerce_tool_input(raw: Any) -> dict[str, Any] | None:
    """Tool inputs arrive as a dict OR as a JSON string in the trace."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _is_tool_call(entry: Any, names: frozenset[str]) -> bool:
    if not isinstance(entry, dict):
        return False
    if entry.get("type") != "tool_call":
        return False
    name = entry.get("tool_name")
    return isinstance(name, str) and name in names


def _resize_grid(old_grid: list[list[int]] | None, *, height: int, width: int) -> list[list[int]]:
    """Reproduce ARC's resize overlap-copy/zero-fill semantics.

    ``old_grid`` is ``None`` when no ``arc_set_output_grid`` call has been
    reconstructed yet — that corresponds to the runtime's default output
    grid (a 3x3 zero grid established at load / on advancing to the next
    test input), not "no grid at all", so fall back to that shape.
    """
    old = old_grid if old_grid is not None else [[0, 0, 0] for _ in range(3)]
    old_h = len(old)
    old_w = len(old[0]) if old else 0
    new_grid = [[0] * width for _ in range(height)]
    for i in range(min(old_h, height)):
        for j in range(min(old_w, width)):
            new_grid[i][j] = old[i][j]
    return new_grid


def _tool_output_status_ok(entry: dict[str, Any]) -> bool:
    """Return True if the tool's output reports status=ok.

    ``tool_output`` arrives as a JSON string OR a dict, depending on which
    wrapper emitted the trace. If it can't be parsed, be permissive —
    advancing the test index even on unknown output is safer than silently
    staying put and scoring every later submission against test 0 (which
    would pin canonical blind scoring to the first test). For
    ``arc_submit_trial`` this also controls whether the trace creates a scoring
    attempt; non-ok submits are runtime evidence but not accepted attempts.
    """
    raw = entry.get("tool_output")
    parsed: Any
    if isinstance(raw, dict):
        parsed = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except JSONDecodeError:
            return True  # permissive default (see docstring)
    else:
        return True
    if not isinstance(parsed, dict):
        return True
    status = parsed.get("status")
    if status is None:
        return True
    return str(status).strip().lower() == "ok"


def _reconstruct_submissions_from_trace(
    tool_trace: Any,
) -> list[tuple[int, list[list[int]] | None]]:
    """Walk `tool_trace` in order and reconstruct accepted ARC submissions.

    Returns a list of (test_index, submitted_grid_or_None) tuples. A None
    grid indicates the agent called submit without having set a grid —
    the caller should treat that as a wrong submission.

    Canonical ARC blind scoring means ``arc_submit_trial`` no longer carries
    a ``correct`` field, so the scorer recovers the submitted grid from the
    most recent ``arc_set_output_grid`` call preceding the submit, and
    compares it against the expected grid itself. ``arc_resize_output_grid``
    calls are replayed too (see ``_resize_grid``), so a resize-then-submit
    with no further full ``arc_set_output_grid`` call is still scored
    against the true submitted grid rather than the stale pre-resize one.
    """
    if not isinstance(tool_trace, list):
        return []

    current_grid: list[list[int]] | None = None
    current_test_index = 0
    submissions: list[tuple[int, list[list[int]] | None]] = []

    for entry in tool_trace:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "tool_call":
            continue
        name = entry.get("tool_name")
        if not isinstance(name, str):
            continue

        if name in _ARC_SET_GRID_TOOL_NAMES:
            tool_input = _coerce_tool_input(entry.get("tool_input"))
            if tool_input is None:
                continue
            grid_candidate = tool_input.get("grid")
            if not _is_grid_like(grid_candidate):
                continue
            try:
                current_grid = normalize_grid(grid_candidate)
            except Exception:
                # Malformed grid — leave current_grid as-is; the submit
                # may still succeed if the agent fixes it later.
                continue
        elif name in _ARC_RESIZE_GRID_TOOL_NAMES:
            tool_input = _coerce_tool_input(entry.get("tool_input"))
            if tool_input is None:
                continue
            try:
                height = int(tool_input.get("height"))
                width = int(tool_input.get("width"))
            except (TypeError, ValueError):
                continue
            if not (1 <= height <= 30 and 1 <= width <= 30):
                # Mirrors the runtime resize's ValueError guard — an invalid
                # resize raises there and leaves the current output grid
                # untouched; do the same here.
                continue
            current_grid = _resize_grid(current_grid, height=height, width=width)
        elif name in _ARC_NEXT_TEST_TOOL_NAMES:
            # Only advance when the tool actually reported success.
            if _tool_output_status_ok(entry):
                current_test_index += 1
                # The runtime auto-resets the output grid when advancing to
                # the next test input. Mirror that here so stale test-0 grids
                # aren't scored for test-1.
                current_grid = None
        elif name in _ARC_SUBMIT_TRIAL_TOOL_NAMES:
            # Mirror the runtime's submit behavior: over-budget submissions
            # return a non-ok status and must not create extra scoring attempts.
            if _tool_output_status_ok(entry):
                submissions.append((current_test_index, current_grid))

    return submissions


def _trace_contains_submit_call(tool_trace: Any) -> bool:
    if not isinstance(tool_trace, list):
        return False
    return any(_is_tool_call(entry, _ARC_SUBMIT_TRIAL_TOOL_NAMES) for entry in tool_trace)


def _empty_runtime_trial_score(*, task_id: str, expected_count: int) -> dict[str, Any]:
    total = int(expected_count)
    return {
        "status": "no_submission",
        "instance_id": task_id,
        "task_type": "arc",
        "resolved": False,
        "native_score": 0.0,
        "arc_pass_ratio": 0.0,
        "normalized_output_json": "",
        "arc_correct_count": 0,
        "arc_total_count": total,
        "arc_per_test": [
            {
                "test_index": idx,
                "correct": False,
                "detail": "no accepted scoring attempt",
                "source": "trace_reconstruction",
            }
            for idx in range(total)
        ],
        "scored_from_runtime_trials": True,
    }


def _normalize_max_trials(max_trials: Any | None) -> int | None:
    if max_trials is None:
        return None
    try:
        return max(1, int(max_trials))
    except Exception:
        return None


def _consume_trial_budget(
    counts: dict[int, int],
    *,
    test_index: int,
    max_trials: int | None,
) -> bool:
    if max_trials is None:
        return True
    current = int(counts.get(test_index, 0))
    if current >= max_trials:
        return False
    counts[test_index] = current + 1
    return True


def _score_trial_results(
    *,
    task_id: str,
    expected_count: int,
    runtime_meta: dict[str, Any] | None,
    tool_trace: Any | None = None,
    expected_grids: list[list[list[int]]] | None = None,
    max_trials: int | None = None,
) -> dict[str, Any] | None:
    """Score the run from runtime-side trial data.

    Walk the ``tool_trace`` to recover every submitted grid, compare against
    ``expected_grids``, and emit per-test verdicts. This is the canonical path
    for ARC's blind harness, where ``arc_submit_trial`` returns no oracle
    feedback. Return ``None`` if no submissions are available — the evaluator
    then scores 0 (canonical ARC has no text-output fallback).
    """
    effective_max_trials = _normalize_max_trials(max_trials)

    if not isinstance(tool_trace, list) or not tool_trace:
        # Check runtime_meta["tool_trace"] as a secondary source.
        if isinstance(runtime_meta, dict):
            nested = runtime_meta.get("tool_trace")
            if isinstance(nested, list):
                tool_trace = nested
    if not isinstance(tool_trace, list) or not tool_trace:
        return None
    if not expected_grids:
        return None

    submissions = _reconstruct_submissions_from_trace(tool_trace)
    if not submissions:
        if _trace_contains_submit_call(tool_trace):
            return _empty_runtime_trial_score(task_id=task_id, expected_count=expected_count)
        return None

    per_test_by_index: dict[int, dict[str, Any]] = {}
    trial_counts_by_index: dict[int, int] = {}
    for test_index, submitted in submissions:
        if test_index < 0 or test_index >= expected_count:
            continue
        if not _consume_trial_budget(
            trial_counts_by_index,
            test_index=test_index,
            max_trials=effective_max_trials,
        ):
            continue
        if submitted is None:
            # Submit with no prior set_output_grid: treat as wrong.
            correct = False
        else:
            try:
                correct, _detail = compare_grids(expected_grids[test_index], submitted)
            except Exception:
                correct = False
        existing = per_test_by_index.get(test_index)
        if existing is None or (correct and not existing.get("correct")):
            per_test_by_index[test_index] = {
                "test_index": test_index,
                "correct": bool(correct),
                "detail": "",
                "source": "trace_reconstruction",
            }

    if not per_test_by_index:
        return None

    total = int(expected_count)
    submitted_count = len(per_test_by_index)
    # Emit a row for every test input so arc_per_test always carries `total`
    # entries, matching the no-submission paths (_empty_runtime_trial_score /
    # no_runtime_submission). A test with no accepted submission is recorded as
    # wrong, exactly like a missing submission.json entry.
    for idx in range(total):
        per_test_by_index.setdefault(
            idx,
            {
                "test_index": idx,
                "correct": False,
                "detail": "no accepted scoring attempt",
                "source": "trace_reconstruction",
            },
        )
    per_test = [per_test_by_index[idx] for idx in sorted(per_test_by_index)]
    correct_count = sum(1 for item in per_test if item["correct"])
    native_score = (correct_count / total) if total else 0.0
    resolved = submitted_count == total and correct_count == total
    return {
        "status": "ok",
        "instance_id": task_id,
        "task_type": "arc",
        "resolved": resolved,
        "native_score": native_score,
        "arc_pass_ratio": native_score,
        "normalized_output_json": "",
        "arc_correct_count": int(correct_count),
        "arc_total_count": total,
        "arc_per_test": per_test,
        "scored_from_runtime_trials": True,
    }


@dataclass
class ArcSessionEvaluator:
    """Evaluator for ARC tasks using canonical exact-match session semantics.

    Only a formal ``arc_submit_trial`` submission (reconstructed from the tool
    trace) is scored, cell-for-cell against the reference grid, under the
    pass@2 per-test-input trial budget. There is no text-output fallback: if
    the agent never submits a scorable grid the task scores 0 here
    (``status="no_submission"``), exactly like a missing ``submission.json``
    entry in the official ARC-AGI grader. The orchestrator's
    ``score_arc_from_eval`` (``src/kcsi/orchestrator/scoring.py``) keeps this
    as a scored failed attempt (0.0), distinct from runtime/infrastructure
    failures such as ``status="no_runtime_submission"``.
    """

    def evaluate(
        self,
        *,
        task: TaskSpec,
        model_output: str,
        runtime_meta: dict[str, Any] | None = None,
        tool_trace: Any | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        meta = task.metadata or {}
        test_pairs = meta.get("arc_eval_test_pairs") or meta.get("arc_test_pairs") or []
        if not isinstance(test_pairs, list) or not test_pairs:
            return {
                "status": "missing_reference",
                "instance_id": task.id,
                "task_type": "arc",
                "resolved": False,
                "native_score": 0.0,
                "arc_pass_ratio": 0.0,
                "scored_from_runtime_trials": False,
            }

        expected_grids: list[list[list[int]]] = []
        for pair in test_pairs:
            if not isinstance(pair, dict) or "output" not in pair:
                return {
                    "status": "missing_reference_output",
                    "instance_id": task.id,
                    "task_type": "arc",
                    "resolved": False,
                    "native_score": 0.0,
                    "arc_pass_ratio": 0.0,
                    "scored_from_runtime_trials": False,
                }
            try:
                expected_grids.append(normalize_grid(pair.get("output")))
            except Exception:
                return {
                    "status": "invalid_reference_output",
                    "instance_id": task.id,
                    "task_type": "arc",
                    "resolved": False,
                    "native_score": 0.0,
                    "arc_pass_ratio": 0.0,
                    "scored_from_runtime_trials": False,
                }

        # Resolve the tool trace from the kwarg or the legacy runtime_meta slot.
        trace = tool_trace
        if trace is None and isinstance(runtime_meta, dict):
            nested = runtime_meta.get("tool_trace")
            if isinstance(nested, list):
                trace = nested

        # The live ``arc_load_task`` MCP tool accepts an agent-suppliable
        # ``max_trials`` override that the real session genuinely enforces
        # (arc_semantics.py), independent of the task-metadata-derived default.
        # Prefer the *effective* max_trials recorded at session-load time
        # (runtime_meta) so the two can't drift; fall back to re-deriving from
        # task metadata only for older traces that predate this field.
        effective_max_trials = None
        if isinstance(runtime_meta, dict):
            effective_max_trials = _normalize_max_trials(runtime_meta.get("arc_effective_max_trials"))
        if effective_max_trials is None:
            effective_max_trials = _normalize_max_trials(meta.get("arc_max_trials", 2)) or 2

        trial_result = _score_trial_results(
            task_id=task.id,
            expected_count=len(expected_grids),
            runtime_meta=runtime_meta,
            tool_trace=trace,
            expected_grids=expected_grids,
            max_trials=effective_max_trials,
        )
        if trial_result is not None:
            return trial_result

        # No canonical runtime-trial submission could be reconstructed. Canonical
        # ARC-AGI scoring only credits a formally submitted grid (the
        # ``submission.json`` analog); there is no text-output fallback. How the
        # 0 is recorded depends on whether the agent actually ran:
        #   * A tool trace exists but holds no accepted submission -> the agent
        #     ran and simply never submitted a scorable grid. ``status ==
        #     "no_submission"`` (``scored_from_runtime_trials`` True), mirroring
        #     an official missing ``submission.json`` entry; the scorer
        #     (score_arc_from_eval) keeps it as a scored failed attempt (0.0).
        #   * No tool trace at all -> the runtime captured nothing (infra
        #     failure). Score 0 but leave ``scored_from_runtime_trials`` absent
        #     so paper analysis filters it instead of counting it.
        have_trace = isinstance(trace, list) and bool(trace)
        if have_trace:
            log.warning(
                "arc_session: task=%s produced a tool trace but no accepted "
                "arc_submit_trial submission; scoring status=no_submission (canonical, no text fallback).",
                task.id,
            )
            return _empty_runtime_trial_score(task_id=task.id, expected_count=len(expected_grids))

        log.warning(
            "arc_session: task=%s has no tool_trace; cannot score canonically "
            "(infra failure). Recording 0 with scored_from_runtime_trials absent.",
            task.id,
        )
        return {
            "status": "no_runtime_submission",
            "instance_id": task.id,
            "task_type": "arc",
            "resolved": False,
            "native_score": 0.0,
            "arc_pass_ratio": 0.0,
            "normalized_output_json": "",
            "arc_correct_count": 0,
            "arc_total_count": len(expected_grids),
            "arc_per_test": [
                {
                    "test_index": idx,
                    "correct": False,
                    "detail": "no tool trace captured",
                    "source": "no_runtime_submission",
                }
                for idx in range(len(expected_grids))
            ],
        }

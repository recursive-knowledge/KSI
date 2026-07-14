"""Tests for surfacing ``retry_attempts`` in the per-experiment results JSON.

Context
-------
PR #491 (``feat/retry-preserve-forensics-and-log``) added ``retry_attempts``
and ``runtime_attempt_errors`` into the ``runtime_meta`` dict on the attempt
row. The data lands in the runtime DB but did NOT appear in the per-experiment
results JSON (``--output-json``). The cross-model audit at
``/tmp/swarms-audit-revalidation/04_cross_model.md`` flagged this — analysts
had to join through the runtime DB to count retries.

This PR lifts ``retry_attempts`` onto each trace dict (defaulting to 0 when
no retry happened) and adds ``total_retry_attempts`` + ``traces_retried_count``
aggregates at the top level of the JSON so dashboards / ``jq`` can read them
directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ksi.cli import _build_tb2_results_summary, _serialize_trace_with_retry_summary
from ksi.models import TaskTrace, TokenUsage

# ---------------------------------------------------------------------------
# _serialize_trace_with_retry_summary — pure function
# ---------------------------------------------------------------------------


def _make_trace(runtime_meta: dict[str, Any] | None = None) -> TaskTrace:
    return TaskTrace(
        generation=1,
        agent_id="agent-0",
        task_id="task-1",
        runtime_meta=runtime_meta or {},
    )


def test_serialize_trace_defaults_retry_attempts_to_zero_when_meta_empty() -> None:
    payload = _serialize_trace_with_retry_summary(_make_trace())
    assert payload["retry_attempts"] == 0
    assert payload["runtime_attempt_errors_count"] == 0


def test_serialize_trace_lifts_retry_attempts_from_runtime_meta() -> None:
    trace = _make_trace(runtime_meta={"retry_attempts": 2})
    payload = _serialize_trace_with_retry_summary(trace)
    assert payload["retry_attempts"] == 2
    # runtime_meta is preserved untouched.
    assert payload["runtime_meta"]["retry_attempts"] == 2


def test_serialize_trace_counts_runtime_attempt_errors_list_length() -> None:
    errors = [
        {"attempt": 1, "error_type": "SilentAgentRuntimeError", "error": "drained"},
        {"attempt": 2, "error_type": "SilentAgentRuntimeError", "error": "drained"},
        {"attempt": 3, "error_type": "SilentAgentRuntimeError", "error": "drained"},
    ]
    trace = _make_trace(runtime_meta={"retry_attempts": 2, "runtime_attempt_errors": errors})
    payload = _serialize_trace_with_retry_summary(trace)
    assert payload["retry_attempts"] == 2
    assert payload["runtime_attempt_errors_count"] == 3


def test_serialize_trace_negative_retry_attempts_clamps_to_zero() -> None:
    # Defensive: a malformed runtime_meta with a negative value shouldn't
    # bleed through as -1 in the JSON.
    trace = _make_trace(runtime_meta={"retry_attempts": -3})
    payload = _serialize_trace_with_retry_summary(trace)
    assert payload["retry_attempts"] == 0


def test_serialize_trace_string_retry_attempts_is_coerced() -> None:
    # Some upstream paths may write string values (e.g. from JSON re-parse);
    # accept them as long as they're integral and non-negative.
    trace = _make_trace(runtime_meta={"retry_attempts": "4"})
    payload = _serialize_trace_with_retry_summary(trace)
    assert payload["retry_attempts"] == 4


def test_serialize_trace_non_int_retry_attempts_falls_back_to_zero() -> None:
    trace = _make_trace(runtime_meta={"retry_attempts": "not-a-number"})
    payload = _serialize_trace_with_retry_summary(trace)
    assert payload["retry_attempts"] == 0


def test_serialize_trace_runtime_meta_not_dict_returns_zeros() -> None:
    @dataclass
    class _Trace:
        generation: int = 1
        agent_id: str = "a"
        task_id: str = "t"
        model_output: str | None = None
        eval_result: dict = field(default_factory=dict)
        native_score: float | None = None
        tool_trace: list = field(default_factory=list)
        # Intentionally non-dict runtime_meta — defensive against drift.
        runtime_meta: Any = None
        token_usage: TokenUsage = field(default_factory=TokenUsage)
        error: str | None = None
        repo: str = ""

    payload = _serialize_trace_with_retry_summary(_Trace())
    assert payload["retry_attempts"] == 0
    assert payload["runtime_attempt_errors_count"] == 0


def test_build_tb2_results_summary_groups_attempts_by_task_and_generation() -> None:
    traces = [
        TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="git-multibranch",
            native_score=0.0,
            tool_trace=[
                {"tool_name": "tb2_shell", "tool_input": {"command": "pwd"}},
                {"tool_name": "tb2_shell", "tool_input": {"command": "service nginx start"}},
            ],
            runtime_meta={
                "reward": 0.0,
                "agent_exit_code": 0,
                "verifier_exit_code": 0,
                "verifier_stdout_tail": "expected branch missing",
            },
            eval_result={"resolved": False},
        ),
        TaskTrace(
            generation=2,
            agent_id="agent-0",
            task_id="git-multibranch",
            native_score=1.0,
            tool_trace=[
                {"tool_name": "tb2_shell", "tool_input": {"command": "git branch feature"}},
                {"tool_name": "tb2_shell", "tool_input": {"command": "git push origin feature"}},
            ],
            runtime_meta={
                "reward": 1.0,
                "agent_exit_code": 0,
                "verifier_exit_code": 0,
                "verifier_stdout_tail": "all checks passed",
            },
            eval_result={"resolved": True},
        ),
    ]

    summary = _build_tb2_results_summary(traces)

    assert summary["tasks_total"] == 1
    assert summary["tasks_solved"] == 1
    assert summary["attempts_total"] == 2
    assert summary["average_best_reward"] == 1.0
    per_task = summary["per_task"]
    assert isinstance(per_task, list)
    assert per_task[0]["task_id"] == "git-multibranch"
    assert per_task[0]["best_reward"] == 1.0
    assert per_task[0]["attempts"][0]["recent_commands"] == ["pwd", "service nginx start"]
    assert per_task[0]["attempts"][1]["resolved"] is True
    by_generation = summary["by_generation"]
    assert isinstance(by_generation, list)
    assert by_generation[0]["generation"] == 1
    assert by_generation[1]["generation"] == 2
    assert by_generation[1]["average_reward"] == 1.0


def test_build_tb2_results_summary_drops_non_finite_reward() -> None:
    """A legacy (pre-#1267) non-finite stored reward must not poison the
    aggregates: ``float("inf")`` parses without raising, so an ungated inf would
    make best_reward/average_reward inf. It is normalized to 0.0 (same as an
    unparseable reward)."""
    traces = [
        TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="poisoned",
            native_score=0.0,
            tool_trace=[],
            runtime_meta={"reward": float("inf")},
            eval_result={"resolved": False},
        ),
    ]

    summary = _build_tb2_results_summary(traces)

    per_task = summary["per_task"]
    assert isinstance(per_task, list)
    assert per_task[0]["best_reward"] == 0.0
    assert summary["average_best_reward"] == 0.0
    by_generation = summary["by_generation"]
    assert isinstance(by_generation, list)
    assert by_generation[0]["average_reward"] == 0.0


def test_build_tb2_results_summary_preserves_unscored_reward_none() -> None:
    traces = [
        TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="unscored",
            native_score=None,
            tool_trace=[],
            runtime_meta={
                "reward": None,
                "trial_status": "verifier_fail_closed_untrusted_toolchain",
                "verifier_fail_closed": True,
            },
            eval_result={"resolved": False},
        ),
    ]

    summary = _build_tb2_results_summary(traces)

    per_task = summary["per_task"]
    assert isinstance(per_task, list)
    assert per_task[0]["best_reward"] is None
    assert per_task[0]["attempts"][0]["reward"] is None
    assert per_task[0]["attempts"][0]["scored"] is False
    assert per_task[0]["attempts"][0]["unscored_reason"] == "verifier_fail_closed_untrusted_toolchain"
    assert summary["average_best_reward"] is None
    by_generation = summary["by_generation"]
    assert isinstance(by_generation, list)
    assert by_generation[0]["attempts"] == 1
    assert by_generation[0]["scored_attempts"] == 0
    assert by_generation[0]["average_reward"] is None

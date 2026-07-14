"""Unit tests for _score_from_eval in the orchestrator engine.

The function signature is:
    _score_from_eval(eval_result: dict, *, task: TaskSpec | None = None) -> float | None

It determines the task source from task.metadata["task_source"].
"""

from __future__ import annotations

import pytest

from kcsi.benchmarks.swebench_pro_external import SWEBENCH_FAILURE_STATUSES
from kcsi.models import TaskSpec
from kcsi.orchestrator.engine import _score_from_eval


def _task(source: str) -> TaskSpec:
    """Create a TaskSpec with the given task_source in metadata."""
    return TaskSpec(id="test-1", prompt="test", metadata={"task_source": source})


class TestScoreFromEvalSwebench:
    def test_resolved_via_instance_report_tests(self):
        result = _score_from_eval(
            {
                "instance_report": {
                    "tests_status": {
                        "FAIL_TO_PASS": {"success": ["test_a"], "failure": []},
                        "PASS_TO_PASS": {"success": ["test_b"], "failure": []},
                    }
                }
            },
            task=_task("swebench_pro"),
        )
        assert result == 1.0

    def test_not_resolved_via_f2p_failure(self):
        result = _score_from_eval(
            {
                "instance_report": {
                    "tests_status": {
                        "FAIL_TO_PASS": {"success": [], "failure": ["test_a"]},
                        "PASS_TO_PASS": {"success": ["test_b"], "failure": []},
                    }
                }
            },
            task=_task("swebench_pro"),
        )
        assert result == 0.0

    def test_not_resolved_via_p2p_failure(self):
        result = _score_from_eval(
            {
                "instance_report": {
                    "tests_status": {
                        "FAIL_TO_PASS": {"success": ["test_a"], "failure": []},
                        "PASS_TO_PASS": {"success": [], "failure": ["test_b"]},
                    }
                }
            },
            task=_task("swebench_pro"),
        )
        assert result == 0.0

    def test_not_resolved_via_skipped_expected_test(self):
        result = _score_from_eval(
            {
                "instance_report": {
                    "tests_status": {
                        "FAIL_TO_PASS": {"success": ["test_a"], "failure": [], "skipped": ["test_b"]},
                        "PASS_TO_PASS": {"success": ["test_c"], "failure": [], "skipped": []},
                    }
                }
            },
            task=_task("swebench_pro"),
        )
        assert result == 0.0

    def test_not_resolved_via_unknown_expected_test(self):
        result = _score_from_eval(
            {
                "instance_report": {
                    "resolved": False,
                    "tests_status": {
                        "FAIL_TO_PASS": {"success": ["test_a"], "failure": [], "unknown": ["test_b"]},
                        "PASS_TO_PASS": {"success": [], "failure": [], "unknown": []},
                    },
                },
            },
            task=_task("swebench_pro"),
        )
        assert result == 0.0

    def test_harness_failed_status(self):
        result = _score_from_eval(
            {"instance_report": {"status": "harness_failed"}},
            task=_task("swebench_pro"),
        )
        assert result is None

    def test_timeout_status(self):
        result = _score_from_eval(
            {"instance_report": {"status": "timeout"}},
            task=_task("swebench_pro"),
        )
        assert result is None

    def test_instance_report_resolved_true(self):
        result = _score_from_eval(
            {"instance_report": {"resolved": True}},
            task=_task("swebench_pro"),
        )
        assert result == 1.0

    def test_instance_report_resolved_false(self):
        result = _score_from_eval(
            {"instance_report": {"resolved": False}},
            task=_task("swebench_pro"),
        )
        assert result == 0.0

    def test_run_summary_resolved(self):
        task = _task("swebench_pro")
        task.id = "my-task-1"
        result = _score_from_eval(
            {"run_summary": {"resolved_ids": ["my-task-1"], "unresolved_ids": []}},
            task=task,
        )
        assert result == 1.0

    def test_run_summary_unresolved(self):
        task = _task("swebench_pro")
        task.id = "my-task-1"
        result = _score_from_eval(
            {"run_summary": {"resolved_ids": [], "unresolved_ids": ["my-task-1"]}},
            task=task,
        )
        assert result == 0.0

    @pytest.mark.parametrize(
        "swebench_status",
        list(SWEBENCH_FAILURE_STATUSES),
    )
    def test_eval_level_swebench_status_failures(self, swebench_status):
        # swebench_pro emits failure statuses under ``swebench_status`` (and
        # uses ``harness_timeout``, not ``timeout``); the explicit check must
        # recognize that key, not just ``status``.
        result = _score_from_eval(
            {"swebench_status": swebench_status, "instance_id": "test-1"},
            task=_task("swebench_pro"),
        )
        assert result is None

    def test_empty_eval_result(self):
        result = _score_from_eval({}, task=_task("swebench_pro"))
        assert result == 0.0


class TestScoreFromEvalGenericFallback:
    """When task_source is unknown or absent, the generic fallback path applies."""

    def test_native_score(self):
        result = _score_from_eval({"native_score": 0.9}, task=None)
        assert result == 0.9

    def test_resolved_true(self):
        result = _score_from_eval({"resolved": True}, task=None)
        assert result == 1.0

    def test_resolved_false(self):
        result = _score_from_eval({"resolved": False}, task=None)
        assert result == 0.0

    def test_instance_report_resolved(self):
        result = _score_from_eval(
            {"instance_report": {"resolved": True}},
            task=None,
        )
        assert result == 1.0

    def test_pass_true(self):
        result = _score_from_eval({"pass": True}, task=None)
        assert result == 1.0

    def test_empty_returns_none(self):
        result = _score_from_eval({}, task=None)
        assert result is None


class TestScoreFromEvalPolyglot:
    """Polyglot evaluator returns dicts with native_score and resolved.

    Since task_source='polyglot' is not 'swebench_pro', these hit the generic
    fallback path which checks native_score first.
    """

    def test_score_from_eval_polyglot_resolved(self):
        task = TaskSpec(id="python__poker", metadata={"task_source": "polyglot"})
        result = {
            "status": "ok",
            "instance_id": "python__poker",
            "native_score": 1.0,
            "resolved": True,
            "language": "python",
        }
        assert _score_from_eval(result, task=task) == 1.0

    def test_score_from_eval_polyglot_failed(self):
        task = TaskSpec(id="python__poker", metadata={"task_source": "polyglot"})
        result = {
            "status": "ok",
            "instance_id": "python__poker",
            "native_score": 0.0,
            "resolved": False,
            "language": "python",
        }
        assert _score_from_eval(result, task=task) == 0.0

    def test_score_from_eval_polyglot_no_solution(self):
        task = TaskSpec(id="python__poker", metadata={"task_source": "polyglot"})
        result = {
            "status": "no_solution",
            "instance_id": "python__poker",
            "native_score": 0.0,
            "resolved": False,
        }
        assert _score_from_eval(result, task=task) is None

    def test_score_from_eval_polyglot_timeout(self):
        task = TaskSpec(id="rust__wordy", metadata={"task_source": "polyglot"})
        result = {
            "status": "timeout",
            "instance_id": "rust__wordy",
            "native_score": 0.0,
            "resolved": False,
            "language": "rust",
        }
        assert _score_from_eval(result, task=task) is None

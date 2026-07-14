"""Direct unit tests for ``score_swebench_from_eval`` (issue #870, part 2).

The function is normally exercised only through engine dispatch
(``_score_from_eval``), which always supplies ``task``. These tests call the
scorer directly to pin every branch, including the ``task=None`` default that
silently skips the ``run_summary`` leg and falls through to ``return 0.0``.

Function under test: ``src/kcsi/orchestrator/scoring.py``
``score_swebench_from_eval(eval_result, *, task=None)`` (lines 45-115).
"""

from __future__ import annotations

from kcsi.models import TaskSpec
from kcsi.orchestrator.scoring import score_swebench_from_eval


def _task(task_id: str = "task-1") -> TaskSpec:
    return TaskSpec(id=task_id, prompt="test", metadata={"task_source": "swebench_pro"})


class TestRunSummaryBranch:
    """The ``run_summary`` leg (scoring.py:106-114) requires ``task``."""

    def test_task_none_with_run_summary_falls_through_to_zero(self):
        # scoring.py:107 — ``isinstance(run_summary, dict) and task is not None``.
        # With the default ``task=None`` the run_summary leg is skipped entirely
        # and the function falls through to ``return 0.0`` (scoring.py:115),
        # EVEN when the run_summary marks this id as resolved. This documents
        # that ``task`` is required for run_summary-based scoring; calling
        # without it is a silent false-negative.
        eval_result = {"run_summary": {"resolved_ids": ["task-1"], "unresolved_ids": []}}
        assert score_swebench_from_eval(eval_result, task=None) == 0.0
        # Sanity: the SAME payload scores 1.0 once ``task`` is supplied.
        assert score_swebench_from_eval(eval_result, task=_task("task-1")) == 1.0

    def test_task_in_resolved_ids_returns_one(self):
        # scoring.py:111-112 — task_id in resolved_ids -> 1.0.
        eval_result = {
            "run_summary": {
                "resolved_ids": ["task-1", "other"],
                "unresolved_ids": [],
            }
        }
        assert score_swebench_from_eval(eval_result, task=_task("task-1")) == 1.0

    def test_task_in_unresolved_ids_returns_zero(self):
        # scoring.py:113-114 — task_id in unresolved_ids -> 0.0.
        eval_result = {
            "run_summary": {
                "resolved_ids": [],
                "unresolved_ids": ["task-1", "other"],
            }
        }
        assert score_swebench_from_eval(eval_result, task=_task("task-1")) == 0.0

    def test_task_in_neither_list_returns_zero(self):
        # task present but absent from both id sets -> final ``return 0.0``.
        eval_result = {"run_summary": {"resolved_ids": ["x"], "unresolved_ids": ["y"]}}
        assert score_swebench_from_eval(eval_result, task=_task("task-1")) == 0.0


class TestInstanceReportBranch:
    """The ``instance_report.tests_status`` leg (scoring.py:50-92)."""

    def test_all_success_returns_one(self):
        # scoring.py:57-92 — total > 0 and zero failure/skipped/unknown -> 1.0.
        eval_result = {
            "instance_report": {
                "tests_status": {
                    "FAIL_TO_PASS": {
                        "success": ["test_a"],
                        "failure": [],
                        "skipped": [],
                        "unknown": [],
                    },
                    "PASS_TO_PASS": {
                        "success": ["test_b"],
                        "failure": [],
                        "skipped": [],
                        "unknown": [],
                    },
                }
            }
        }
        # No ``task`` needed — instance_report scoring is task-independent.
        assert score_swebench_from_eval(eval_result, task=None) == 1.0

    def test_any_failure_returns_zero(self):
        # scoring.py:84-89 — a FAIL_TO_PASS failure forces 0.0.
        eval_result = {
            "instance_report": {
                "tests_status": {
                    "FAIL_TO_PASS": {"success": [], "failure": ["test_a"]},
                    "PASS_TO_PASS": {"success": ["test_b"], "failure": []},
                }
            }
        }
        assert score_swebench_from_eval(eval_result, task=None) == 0.0

    def test_known_failure_status_is_unscored(self):
        # scoring.py:53-55 — harness_failed/timeout/no_patch/missing_report -> None (unscored).
        eval_result = {"instance_report": {"status": "harness_failed"}}
        assert score_swebench_from_eval(eval_result, task=None) is None


class TestResolvedVerdictIsAuthoritative:
    """Score from the harness's ``resolved`` verdict, not the re-derived tally
    (issue #943 / #962).

    The upstream harness resolves an instance via an exact-string subset check
    ``(FAIL_TO_PASS | PASS_TO_PASS) <= passed_tests`` and emits a per-instance
    ``resolved`` boolean — the published metric. The local ``tests_status``
    tally is a re-derivation over a separately-read ``output.json`` that can
    diverge (stale/empty/name-skew), so it is DIAGNOSTIC only and must not
    override the verdict.
    """

    def test_resolved_true_with_unknown_tally_scores_one(self):
        # Re-baseline change: harness resolved the instance, but the local tally
        # marks an expected name ``unknown`` (a read divergence, #962). The old
        # binary scorer forced 0.0; the verdict now wins -> 1.0.
        eval_result = {
            "instance_report": {
                "resolved": True,
                "tests_status": {
                    "FAIL_TO_PASS": {"success": ["a"], "failure": [], "skipped": [], "unknown": ["b"]},
                    "PASS_TO_PASS": {"success": [], "failure": [], "skipped": [], "unknown": []},
                },
            }
        }
        assert score_swebench_from_eval(eval_result, task=None) == 1.0

    def test_resolved_true_with_failure_tally_scores_one(self):
        # resolved=True alongside a tally ``failure`` is also a read divergence
        # by the harness's subset rule; the verdict still wins.
        eval_result = {
            "instance_report": {
                "resolved": True,
                "tests_status": {
                    "FAIL_TO_PASS": {"success": [], "failure": ["a"], "skipped": [], "unknown": []},
                    "PASS_TO_PASS": {"success": [], "failure": [], "skipped": [], "unknown": []},
                },
            }
        }
        assert score_swebench_from_eval(eval_result, task=None) == 1.0

    def test_resolved_false_scores_zero_regardless_of_tally(self):
        # The verdict is authoritative downward too: harness did NOT resolve, so
        # 0.0 even if the local tally happens to look all-clear.
        eval_result = {
            "instance_report": {
                "resolved": False,
                "tests_status": {
                    "FAIL_TO_PASS": {"success": ["a"], "failure": [], "skipped": [], "unknown": []},
                    "PASS_TO_PASS": {"success": ["b"], "failure": [], "skipped": [], "unknown": []},
                },
            }
        }
        assert score_swebench_from_eval(eval_result, task=None) == 0.0

    def test_tally_still_scores_when_verdict_absent(self):
        # Fallback: a verdict-less report (no ``resolved`` key) still scores from
        # the all-or-nothing tally. Real evaluator output always carries
        # ``resolved`` (handled above), so this only serves malformed reports.
        eval_result = {
            "instance_report": {
                "tests_status": {
                    "FAIL_TO_PASS": {"success": ["a"], "failure": [], "skipped": [], "unknown": []},
                    "PASS_TO_PASS": {"success": ["b"], "failure": [], "skipped": [], "unknown": []},
                }
            }
        }
        assert score_swebench_from_eval(eval_result, task=None) == 1.0


class TestNonBoolResolvedIsNotAuthoritative:
    """A non-bool ``resolved`` is NOT trusted as the verdict (#966).

    The harness emits a JSON boolean, but a malformed or foreign report could
    carry a string/int. ``bool("false")`` is truthy, so scoring it via plain
    truthiness would yield a false 1.0. The scorer requires
    ``isinstance(resolved, bool)`` and otherwise falls through to the tally.
    """

    def test_string_resolved_falls_through_to_tally(self):
        # ``resolved="false"`` is a truthy string; the old ``bool(...)`` path
        # scored 1.0. The type guard makes it fall through to the tally, which
        # has a failure -> 0.0 (no longer a false resolve).
        eval_result = {
            "instance_report": {
                "resolved": "false",
                "tests_status": {
                    "FAIL_TO_PASS": {"success": [], "failure": ["a"], "skipped": [], "unknown": []},
                    "PASS_TO_PASS": {"success": [], "failure": [], "skipped": [], "unknown": []},
                },
            }
        }
        assert score_swebench_from_eval(eval_result, task=None) == 0.0

    def test_int_resolved_falls_through_to_tally(self):
        # ``isinstance(1, bool)`` is False, so an int verdict is not trusted; the
        # all-clear tally scores it instead.
        eval_result = {
            "instance_report": {
                "resolved": 1,
                "tests_status": {
                    "FAIL_TO_PASS": {"success": ["a"], "failure": [], "skipped": [], "unknown": []},
                    "PASS_TO_PASS": {"success": ["b"], "failure": [], "skipped": [], "unknown": []},
                },
            }
        }
        assert score_swebench_from_eval(eval_result, task=None) == 1.0

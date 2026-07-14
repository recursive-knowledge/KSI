"""Tests for evaluation-signal anonymization in MEMORY.md seeding.

Verify that _best_attempt_summary does NOT leak F2P/P2P test names into the
MEMORY.md that is written into the agent workspace.  Only anonymized counts
should appear.
"""

from __future__ import annotations

from kcsi.runtime.seeding import _best_attempt_summary


def _make_record(
    *,
    gen: int = 1,
    score: float = 0.5,
    resolved: bool = False,
    f2p_success: list[str] | None = None,
    f2p_failure: list[str] | None = None,
    p2p_failure: list[str] | None = None,
) -> dict:
    """Build a minimal attempt record with realistic test names."""
    return {
        "gen": gen,
        "eval_results": {
            "native_score": score,
            "resolved": resolved,
            "instance_report": {
                "tests_status": {
                    "observed_count": (len(f2p_success or []) + len(f2p_failure or [])),
                    "FAIL_TO_PASS": {
                        "success": f2p_success or [],
                        "failure": f2p_failure or [],
                    },
                    "PASS_TO_PASS": {
                        "success": [],
                        "failure": p2p_failure or [],
                    },
                }
            },
        },
    }


F2P_NAMES = [
    "tests/test_widget.py::test_fix_regression",
    "tests/test_api.py::test_endpoint_smoke",
    "tests.core.test_models.TestModel.test_save",
]
P2P_NAMES = [
    "tests/test_widget.py::test_old_case",
    "tests/test_api.py::test_auth",
]


class TestBestAttemptSummaryAnonymization:
    def test_no_test_names_in_passed_tests_field(self):
        records = [_make_record(f2p_success=F2P_NAMES, f2p_failure=[], p2p_failure=[])]
        lines = _best_attempt_summary(records)
        combined = "\n".join(lines)
        for name in F2P_NAMES:
            assert name not in combined, f"Test name leaked: {name!r}"

    def test_no_test_names_in_remaining_tests_field(self):
        records = [_make_record(f2p_success=[], f2p_failure=F2P_NAMES, p2p_failure=[])]
        lines = _best_attempt_summary(records)
        combined = "\n".join(lines)
        for name in F2P_NAMES:
            assert name not in combined, f"Test name leaked: {name!r}"

    def test_no_test_names_in_p2p_regressions_field(self):
        records = [_make_record(f2p_success=F2P_NAMES, f2p_failure=[], p2p_failure=P2P_NAMES)]
        lines = _best_attempt_summary(records)
        combined = "\n".join(lines)
        for name in P2P_NAMES:
            assert name not in combined, f"P2P test name leaked: {name!r}"

    def test_count_information_is_preserved(self):
        """Anonymized counts must still be surfaced so the swarm knows progress."""
        records = [
            _make_record(
                f2p_success=F2P_NAMES[:2],
                f2p_failure=F2P_NAMES[2:],
                p2p_failure=P2P_NAMES[:1],
            )
        ]
        lines = _best_attempt_summary(records)
        combined = "\n".join(lines)
        # Should mention the count of passed target tests (2)
        assert "2" in combined
        # Regression risk count (1)
        assert "1" in combined

    def test_anonymized_label_present_for_passed_tests(self):
        records = [_make_record(f2p_success=F2P_NAMES, f2p_failure=[])]
        lines = _best_attempt_summary(records)
        combined = "\n".join(lines)
        assert "target tests to preserve" in combined
        assert "previously-passing target test" in combined

    def test_anonymized_label_present_for_remaining_tests(self):
        records = [_make_record(f2p_success=[], f2p_failure=F2P_NAMES)]
        lines = _best_attempt_summary(records)
        combined = "\n".join(lines)
        assert "remaining failing tests" in combined
        assert "target test(s) still failing" in combined

    def test_anonymized_label_present_for_regressions(self):
        records = [_make_record(f2p_success=F2P_NAMES, f2p_failure=[], p2p_failure=P2P_NAMES)]
        lines = _best_attempt_summary(records)
        combined = "\n".join(lines)
        assert "regressed tests to preserve" in combined
        assert "previously-passing test(s) now failing" in combined

    def test_no_leakage_with_multiple_records(self):
        """Ensure best-record selection still anonymizes even across generations."""
        records = [
            _make_record(gen=1, score=0.3, f2p_success=F2P_NAMES[:1], f2p_failure=F2P_NAMES[1:]),
            _make_record(gen=2, score=0.7, f2p_success=F2P_NAMES, f2p_failure=[], p2p_failure=P2P_NAMES),
        ]
        lines = _best_attempt_summary(records)
        combined = "\n".join(lines)
        for name in F2P_NAMES + P2P_NAMES:
            assert name not in combined, f"Test name leaked across records: {name!r}"

    def test_empty_records_returns_empty(self):
        assert _best_attempt_summary([]) == []

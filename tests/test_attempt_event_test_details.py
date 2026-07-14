import json

from ksi.orchestrator.engine import _build_attempt_event


class TestAttemptEventTestDetailsLeakedModeOptIn:
    """DGM-equivalent leaked-mode behavior: `_build_attempt_event` emits raw
    test-name lists ONLY when `seed_test_files=True` is passed explicitly.

    The function-signature default is `False` (upstream-strict, no leak); the
    leaked path is opt-in. Each test below passes `seed_test_files=True`
    explicitly to exercise the leaked behavior — these are kept as a contract
    for DGM-equivalent comparisons where leakage is intentional. The
    corresponding upstream-strict behavior lives in
    `TestAttemptEventUpstreamStrictAnonymization` below.
    """

    def test_includes_still_failing_tests(self):
        eval_results = {
            "swebench_status": "ok",
            "instance_report": {
                "tests_status": {
                    "FAIL_TO_PASS": {"success": [], "failure": ["tests/test_foo.py::test_bar"]},
                    "PASS_TO_PASS": {"success": [], "failure": []},
                }
            },
        }
        event = _build_attempt_event(native_score=0.0, error="", eval_results=eval_results, seed_test_files=True)
        assert "tests/test_foo.py::test_bar" in event["tests_still_failing"]
        assert event["tests_regressed"] == []

    def test_includes_regressed_tests(self):
        eval_results = {
            "swebench_status": "ok",
            "instance_report": {
                "tests_status": {
                    "FAIL_TO_PASS": {"success": [], "failure": []},
                    "PASS_TO_PASS": {"success": [], "failure": ["tests/test_x.py::test_y"]},
                }
            },
        }
        event = _build_attempt_event(native_score=0.0, error="", eval_results=eval_results, seed_test_files=True)
        assert "tests/test_x.py::test_y" in event["tests_regressed"]

    def test_includes_newly_passing(self):
        eval_results = {
            "swebench_status": "ok",
            "instance_report": {
                "tests_status": {
                    "FAIL_TO_PASS": {"success": ["tests/test_a.py::test_b"], "failure": ["tests/test_c.py::test_d"]},
                    "PASS_TO_PASS": {"success": [], "failure": []},
                }
            },
        }
        event = _build_attempt_event(native_score=0.0, error="", eval_results=eval_results, seed_test_files=True)
        assert "tests/test_a.py::test_b" in event["tests_now_passing"]
        assert "tests/test_c.py::test_d" in event["tests_still_failing"]

    def test_empty_when_no_instance_report(self):
        event = _build_attempt_event(
            native_score=0.0,
            error="",
            eval_results={"swebench_status": "no_patch"},
            seed_test_files=True,
        )
        assert event["tests_still_failing"] == []
        assert event["tests_regressed"] == []
        assert event["tests_now_passing"] == []
        assert event["tests_skipped"] == []
        assert event["tests_unobserved"] == []

    def test_includes_unobserved_tests(self):
        eval_results = {
            "swebench_status": "ok",
            "instance_report": {
                "tests_status": {
                    "FAIL_TO_PASS": {"success": [], "failure": [], "unknown": ["TestFix"]},
                    "PASS_TO_PASS": {"success": [], "failure": [], "unknown": ["TestExisting"]},
                }
            },
        }
        event = _build_attempt_event(native_score=0.0, error="", eval_results=eval_results, seed_test_files=True)
        assert event["tests_still_failing"] == []
        assert event["tests_regressed"] == []
        assert event["tests_unobserved"] == ["TestFix", "TestExisting"]

    def test_includes_skipped_tests(self):
        eval_results = {
            "swebench_status": "ok",
            "instance_report": {
                "tests_status": {
                    "FAIL_TO_PASS": {"success": [], "failure": [], "skipped": ["TestSkipped"]},
                    "PASS_TO_PASS": {"success": [], "failure": [], "skipped": ["TestError"]},
                }
            },
        }
        event = _build_attempt_event(native_score=0.0, error="", eval_results=eval_results, seed_test_files=True)
        assert event["tests_still_failing"] == []
        assert event["tests_regressed"] == []
        assert event["tests_skipped"] == ["TestSkipped", "TestError"]

    def test_harness_failed_status_preserved(self):
        event = _build_attempt_event(
            native_score=0.0,
            error="",
            eval_results={"swebench_status": "harness_failed"},
            seed_test_files=True,
        )
        assert event["status"] == "harness_failed"

    def test_preserves_existing_fields(self):
        event = _build_attempt_event(native_score=0.5, error="timeout", eval_results={}, seed_test_files=True)
        assert event["native_score"] == 0.5
        assert event["error"] == "timeout"
        assert event["resolved"] is False


class TestAttemptEventDefaultIsUpstreamStrict:
    """Regression test for the seed_tests-default fix. The function-signature
    default for `seed_test_files` is False (upstream-strict). Callers that
    forget the kwarg must NOT silently get the leaked-mode behavior."""

    def test_default_kwarg_omits_test_name_lists(self):
        eval_results = {
            "swebench_status": "ok",
            "instance_report": {
                "tests_status": {
                    "FAIL_TO_PASS": {"success": [], "failure": ["tests/test_grader.py::test_X"]},
                    "PASS_TO_PASS": {"success": [], "failure": []},
                }
            },
        }
        # NO `seed_test_files=` kwarg — must default to False (strict mode).
        event = _build_attempt_event(native_score=0.0, error="", eval_results=eval_results)
        # Leaked-mode keys must NOT be present in the event payload.
        assert "tests_still_failing" not in event, (
            "Default behavior must NOT emit raw test-name lists; "
            "this regresses the seed_tests fix and re-enables the grader-test-name leak."
        )
        assert "tests_now_passing" not in event
        assert "tests_regressed" not in event
        # Safe count keys must be present.
        assert event["tests_still_failing_count"] == 1
        assert event["tests_now_passing_count"] == 0


class TestAttemptEventArcParity:
    def test_arc_per_test_omits_answer_detail(self):
        event = _build_attempt_event(
            native_score=0.0,
            error="",
            eval_results={
                "status": "ok",
                "arc_pass_ratio": 0.0,
                "arc_per_test": [
                    {
                        "test_index": 0,
                        "correct": False,
                        "detail": {
                            "expected_shape": [3, 3],
                            "first_mismatch": {"row": 0, "col": 0, "expected": 4242, "submitted": 3},
                        },
                    }
                ],
            },
        )

        blob = json.dumps(event)
        assert "detail" not in blob
        assert "4242" not in blob
        assert event["arc_pass_ratio"] == 0.0
        assert event["arc_per_test"] == [{"test_index": 0, "correct": False}]


class TestAttemptEventUpstreamStrictAnonymization:
    """When seed_test_files=False (upstream-strict SWE-bench Pro), the
    attempt event must NOT contain raw test name lists — only counts.

    These names are persisted in attempt_history_json and surfaced to
    the next-generation agent via the MCP query tool's
    ``compact_records[*].attempt_history`` payload, so leaking them
    invalidates upstream-strict comparability.
    """

    _eval_with_test_names = {
        "swebench_status": "ok",
        "instance_report": {
            "tests_status": {
                "FAIL_TO_PASS": {
                    "success": ["tests/test_widget.py::test_a"],
                    "failure": ["tests/test_widget.py::test_b"],
                    "skipped": ["tests/test_widget.py::test_c"],
                    "unknown": ["TestUnknownFTP"],
                },
                "PASS_TO_PASS": {
                    "success": [],
                    "failure": ["tests/test_api.py::test_z"],
                    "skipped": ["tests/test_api.py::test_skip"],
                    "unknown": ["TestUnknownPTP"],
                },
            }
        },
    }

    def test_anonymized_event_omits_test_name_lists(self):
        event = _build_attempt_event(
            native_score=0.5,
            error="",
            eval_results=self._eval_with_test_names,
            seed_test_files=False,
        )
        for forbidden in (
            "tests_still_failing",
            "tests_now_passing",
            "tests_regressed",
            "tests_skipped",
            "tests_unobserved",
        ):
            assert forbidden not in event, forbidden
        # Confirm the JSON-rendered payload itself contains no test names.
        import json

        payload = json.dumps(event)
        for needle in (
            "tests/test_widget.py",
            "tests/test_api.py",
            "TestUnknownFTP",
            "TestUnknownPTP",
        ):
            assert needle not in payload, needle

    def test_anonymized_event_preserves_counts(self):
        event = _build_attempt_event(
            native_score=0.5,
            error="",
            eval_results=self._eval_with_test_names,
            seed_test_files=False,
        )
        assert event["tests_still_failing_count"] == 1
        assert event["tests_now_passing_count"] == 1
        assert event["tests_regressed_count"] == 1
        assert event["tests_skipped_count"] == 2
        assert event["tests_unobserved_count"] == 2

    def test_seeded_mode_keeps_legacy_lists(self):
        event = _build_attempt_event(
            native_score=0.5,
            error="",
            eval_results=self._eval_with_test_names,
            seed_test_files=True,
        )
        assert event["tests_still_failing"] == ["tests/test_widget.py::test_b"]
        assert event["tests_now_passing"] == ["tests/test_widget.py::test_a"]
        assert event["tests_regressed"] == ["tests/test_api.py::test_z"]


class TestApproachDiagnosisUpstreamStrict:
    """``_build_approach_diagnosis`` must not emit test names when
    ``seed_test_files=False``. Mirrors PR #586's anonymization for the
    persisted full_memory_trace_condensed payload that surfaces to
    next-gen agents via MEMORY.md and the MCP query tool.
    """

    def _trace(self, native_score=0.5):
        from types import SimpleNamespace

        return SimpleNamespace(native_score=native_score, runtime_meta={}, model_output="")

    def _eval(self):
        return {
            "instance_report": {
                "tests_status": {
                    "FAIL_TO_PASS": {
                        "success": ["tests/test_x.py::test_a"],
                        "failure": ["tests/test_x.py::test_b"],
                    },
                    "PASS_TO_PASS": {
                        "success": [],
                        "failure": ["tests/test_y.py::test_regression"],
                    },
                }
            }
        }

    def test_strict_diagnosis_omits_test_names(self):
        from ksi.orchestrator.engine import _build_approach_diagnosis

        text = _build_approach_diagnosis(
            trace=self._trace(),
            eval_result=self._eval(),
            outcome="unresolved",
            task_source="swebench_pro",
            seed_test_files=False,
        )
        for needle in (
            "tests/test_x.py",
            "tests/test_y.py",
            "test_regression",
            "test_a",
            "test_b",
        ):
            assert needle not in text, needle
        # Counts and descriptive labels are still present.
        assert "STILL FAILING: 1" in text
        assert "REGRESSED: 1" in text
        assert "now passing (good): 1" in text

    def test_seeded_diagnosis_keeps_test_names(self):
        from ksi.orchestrator.engine import _build_approach_diagnosis

        text = _build_approach_diagnosis(
            trace=self._trace(),
            eval_result=self._eval(),
            outcome="unresolved",
            task_source="swebench_pro",
            seed_test_files=True,
        )
        assert "tests/test_x.py::test_b" in text
        assert "tests/test_y.py::test_regression" in text

    def test_tb2_diagnosis_omits_hidden_verifier_tails_but_keeps_scalars(self):
        from types import SimpleNamespace

        from ksi.orchestrator.engine import _build_approach_diagnosis

        text = _build_approach_diagnosis(
            trace=SimpleNamespace(
                native_score=0.0,
                runtime_meta={
                    "reward": 0.0,
                    "agent_exit_code": 0,
                    "verifier_exit_code": 1,
                    "verifier_stdout_tail": "secretcanarytoken_tb2_stdout",
                    "verifier_stderr_tail": "secretcanarytoken_tb2_stderr",
                },
                model_output="",
                tool_trace=[
                    {"tool_input": {"command": "python solve.py"}},
                    {"tool_input": {"command": "pytest hidden_canary.py"}},
                ],
            ),
            eval_result={},
            outcome="unresolved",
            task_source="terminal_bench_2",
        )

        assert "secretcanarytoken_tb2_stdout" not in text
        assert "secretcanarytoken_tb2_stderr" not in text
        assert "reward=0.0" in text
        assert "agent_exit=0" in text
        assert "verifier_exit=1" in text
        assert "python solve.py" in text
        assert "pytest hidden_canary.py" in text

    def test_arc_diagnosis_omits_hidden_trial_details(self):
        from types import SimpleNamespace

        from ksi.orchestrator.engine import _build_approach_diagnosis

        text = _build_approach_diagnosis(
            trace=SimpleNamespace(
                native_score=0.0,
                runtime_meta={
                    "arc_submit_trial_results": [
                        {
                            "test_index": 0,
                            "reason": "shape_mismatch",
                            "submitted_shape": [1, 1],
                            "expected_shape": [3, 3],
                        },
                        {
                            "test_index": 1,
                            "reason": "cell_mismatch",
                            "first_mismatch": {"row": 0, "col": 0, "expected": 4242, "submitted": 3},
                        },
                    ]
                },
                model_output="",
            ),
            eval_result={
                "status": "scored",
                "arc_total_count": 2,
                "arc_correct_count": 0,
                "arc_pass_ratio": 0.0,
            },
            outcome="unresolved",
            task_source="arc",
        )

        assert "Rejected hidden ARC trial(s): 2" in text
        assert "4242" not in text
        assert "expected_shape" not in text
        assert "expected [3, 3]" not in text
        assert "r0 c0" not in text

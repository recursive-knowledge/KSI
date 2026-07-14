"""Seam 4 (issue #741): per-source score extraction dispatch.

Pins ``_score_from_eval`` output for the swebench leg and the generic fallback
so collapsing the ``task_source ==`` branch onto a
``TaskSourceSpec.score_from_eval`` hook is byte-identical. The capability test
exercises the new hook and fails until the spec field + dispatch exist.
"""

from types import SimpleNamespace as NS

from kcsi.orchestrator.engine import _score_from_eval


def _task(src):
    return NS(metadata={"task_source": src}, id="task-1")


def test_swebench_all_pass():
    out = _score_from_eval(
        {
            "instance_report": {
                "tests_status": {
                    "FAIL_TO_PASS": {"success": ["a"], "failure": [], "skipped": [], "unknown": []},
                    "PASS_TO_PASS": {"success": ["b"], "failure": [], "skipped": [], "unknown": []},
                }
            }
        },
        task=_task("swebench_pro"),
    )
    assert out == 1.0


def test_swebench_has_failure():
    out = _score_from_eval(
        {"instance_report": {"tests_status": {"FAIL_TO_PASS": {"success": [], "failure": ["a"]}, "PASS_TO_PASS": {}}}},
        task=_task("swebench"),
    )
    assert out == 0.0


def test_swebench_harness_failed_status_is_unscored():
    # Harness-level infra failure: no trustworthy verdict was produced -> None.
    out = _score_from_eval({"instance_report": {"status": "harness_failed"}}, task=_task("swebench_pro"))
    assert out is None


def test_swebench_resolved_flag():
    out = _score_from_eval({"instance_report": {"resolved": True}}, task=_task("swebench_pro"))
    assert out == 1.0


def test_swebench_swebench_status_harness_timeout_is_unscored():
    out = _score_from_eval({"swebench_status": "harness_timeout"}, task=_task("swebench_pro"))
    assert out is None


def test_swebench_oom_killed_status_is_unscored():
    # An OOM-killed container (mem_limit cap) is an infra failure, not a genuine
    # agent test failure -> None, so it never feeds _best_scores/distillation as
    # a fabricated 0.0.
    out = _score_from_eval(
        {"swebench_status": "oom_killed", "oom_killed": True},
        task=_task("swebench_pro"),
    )
    assert out is None


def test_swebench_run_summary_resolved_membership():
    assert _score_from_eval({"run_summary": {"resolved_ids": ["task-1"]}}, task=_task("swebench_pro")) == 1.0
    assert _score_from_eval({"run_summary": {"unresolved_ids": ["task-1"]}}, task=_task("swebench_pro")) == 0.0


def test_swebench_empty_defaults_zero():
    out = _score_from_eval({}, task=_task("swebench_pro"))
    assert out == 0.0


def test_generic_native_score():
    assert _score_from_eval({"native_score": 0.7}, task=_task("arc")) == 0.7


def test_generic_resolved():
    assert _score_from_eval({"resolved": True}, task=_task("polyglot")) == 1.0


def test_generic_none_when_no_signal():
    assert _score_from_eval({}, task=_task("polyglot")) is None


def test_no_task_uses_generic():
    assert _score_from_eval({"native_score": 1.0}, task=None) == 1.0


def test_custom_task_source_scorer_is_dispatched():
    """A registered source's ``score_from_eval`` hook drives scoring, with no
    ``task_source ==`` edit in the engine."""
    from kcsi.tasks import registry

    def _scorer(eval_result, *, task):
        return 0.42 if eval_result.get("magic") else None

    spec = registry.TaskSourceSpec(name="score_fake_src", score_from_eval=_scorer)
    registry.register_task_source(spec)
    try:
        assert _score_from_eval({"magic": True}, task=_task("score_fake_src")) == 0.42
        # Returning None from the hook is honored (not overridden by the generic).
        assert _score_from_eval({"native_score": 1.0}, task=_task("score_fake_src")) is None
    finally:
        for key in spec.all_names():
            registry.REGISTRY.pop(key, None)


def test_swebench_skipped_present_is_unresolved():
    # A skipped (not failed) test still forces 0.0 — the strict "ALL pass" invariant.
    out = _score_from_eval(
        {
            "instance_report": {
                "tests_status": {
                    "FAIL_TO_PASS": {"success": ["a"], "skipped": ["s"]},
                    "PASS_TO_PASS": {"success": ["b"]},
                }
            }
        },
        task=_task("swebench_pro"),
    )
    assert out == 0.0


def test_swebench_unknown_present_is_unresolved():
    # An unknown-status test (parser saw neither pass nor fail) also forces 0.0.
    out = _score_from_eval(
        {
            "instance_report": {
                "tests_status": {
                    "FAIL_TO_PASS": {"success": ["a"], "unknown": ["u"]},
                    "PASS_TO_PASS": {"success": ["b"]},
                }
            }
        },
        task=_task("swebench_pro"),
    )
    assert out == 0.0


# --- terminal_bench_2 (issue #977) -------------------------------------------
# The TB2 evaluator emits ``native_score=0.0`` for BOTH a genuine agent failure
# and a verifier-never-ran infra failure; the registered scorer must treat the
# latter as unscored (``None``) so the engine preserves the prior best instead
# of recording a fabricated 0.0 into the learning signal.


def test_tb2_verifier_missing_is_unscored():
    # Verifier never ran: native_score 0.0 + verifier-missing status -> None.
    out = _score_from_eval(
        {"status": "verifier_did_not_produce_reward", "native_score": 0.0, "reward": None},
        task=_task("terminal_bench_2"),
    )
    assert out is None


def test_tb2_fail_closed_untrusted_toolchain_is_unscored():
    # #1206: strict mode refused the untrusted-toolchain fallback; the verifier
    # never ran, so this is unscored (None), never a fabricated 0.0.
    out = _score_from_eval(
        {
            "status": "verifier_fail_closed_untrusted_toolchain",
            "native_score": None,
            "reward": None,
            "resolved": False,
        },
        task=_task("terminal_bench_2"),
    )
    assert out is None


def test_tb2_genuine_failure_scores_zero():
    # Verifier ran, agent failed: a real 0.0 reward stays 0.0 (not unscored).
    out = _score_from_eval(
        {"status": "verifier_failed", "native_score": 0.0, "reward": 0.0},
        task=_task("terminal_bench_2"),
    )
    assert out == 0.0


def test_tb2_genuine_pass_scores_one():
    out = _score_from_eval(
        {"status": "completed", "native_score": 1.0, "reward": 1.0, "resolved": True},
        task=_task("terminal_bench_2"),
    )
    assert out == 1.0


# --- polyglot -----------------------------------------------------------
# The polyglot evaluator's infra-failure paths are Docker subprocess timeout
# (status="timeout") and agent-produced-nothing (status="no_solution"); both
# are unscored (None). "ok" (real pass/fail) is a genuine, trustworthy verdict.


def test_polyglot_docker_timeout_is_unscored():
    out = _score_from_eval(
        {"status": "timeout", "native_score": 0.0, "resolved": False},
        task=_task("polyglot"),
    )
    assert out is None


def test_polyglot_no_solution_is_unscored():
    # Agent produced no extractable solution at all: unscored, mirrors
    # swebench_pro's no_patch (nothing was graded, not a trustworthy 0.0).
    out = _score_from_eval(
        {"status": "no_solution", "native_score": 0.0, "resolved": False},
        task=_task("polyglot"),
    )
    assert out is None


def test_polyglot_setup_failure_is_unscored():
    # Setup step (e.g. `npm install`) itself exited nonzero before the test
    # ever ran: unscored, mirrors timeout/no_solution (issue #1042). Not a
    # trustworthy 0.0 -- indistinguishable from a genuine test failure via
    # returncode alone since setup and test are chained with `&&`.
    out = _score_from_eval(
        {"status": "setup_failed", "native_score": 0.0, "resolved": False},
        task=_task("polyglot"),
    )
    assert out is None


def test_polyglot_genuine_pass_scores_one():
    out = _score_from_eval(
        {"status": "ok", "native_score": 1.0, "resolved": True},
        task=_task("polyglot"),
    )
    assert out == 1.0


def test_polyglot_genuine_fail_scores_zero():
    out = _score_from_eval(
        {"status": "ok", "native_score": 0.0, "resolved": False},
        task=_task("polyglot"),
    )
    assert out == 0.0


# --- arc ------------------------------------------------------------------
# arc_session.py always emits a numeric native_score, but true no-verdict
# statuses stay unscored: no_runtime_submission (true infra failure — no
# tool_trace captured at all), and the missing/invalid-reference-data statuses
# (a dataset bug, not a genuine agent failure). no_submission is a real executed
# failed attempt and should keep its evaluator-emitted 0.0.


def test_arc_no_runtime_submission_is_unscored():
    out = _score_from_eval(
        {"status": "no_runtime_submission", "native_score": 0.0, "resolved": False},
        task=_task("arc"),
    )
    assert out is None


def test_arc_no_submission_keeps_canonical_zero_score():
    out = _score_from_eval(
        {
            "status": "no_submission",
            "native_score": 0.0,
            "resolved": False,
            "scored_from_runtime_trials": True,
        },
        task=_task("arc"),
    )
    assert out == 0.0


def test_arc_missing_reference_is_unscored():
    out = _score_from_eval(
        {"status": "missing_reference", "native_score": 0.0, "resolved": False},
        task=_task("arc"),
    )
    assert out is None


def test_arc_missing_reference_output_is_unscored():
    out = _score_from_eval(
        {"status": "missing_reference_output", "native_score": 0.0, "resolved": False},
        task=_task("arc"),
    )
    assert out is None


def test_arc_invalid_reference_output_is_unscored():
    out = _score_from_eval(
        {"status": "invalid_reference_output", "native_score": 0.0, "resolved": False},
        task=_task("arc"),
    )
    assert out is None


def test_arc_genuine_trial_zero_scores_zero():
    # Agent ran and got a submission accepted, but it was wrong: a real 0.0
    # (scored_from_runtime_trials=True). Distinct from status="no_submission"
    # above, where nothing was ever accepted.
    out = _score_from_eval(
        {
            "status": "ok",
            "native_score": 0.0,
            "resolved": False,
            "scored_from_runtime_trials": True,
        },
        task=_task("arc"),
    )
    assert out == 0.0


def test_arc_genuine_partial_score_preserved():
    out = _score_from_eval(
        {
            "status": "ok",
            "native_score": 0.5,
            "resolved": False,
            "scored_from_runtime_trials": True,
        },
        task=_task("arc"),
    )
    assert out == 0.5


def test_arc_genuine_pass_scores_one():
    out = _score_from_eval(
        {"status": "ok", "native_score": 1.0, "resolved": True, "scored_from_runtime_trials": True},
        task=_task("arc"),
    )
    assert out == 1.0

"""Seam 4 (issue #741): benchmark-specific approach-diagnosis formatting.

These tests pin the *exact* output of ``_build_approach_diagnosis`` for every
benchmark leg so the dispatch refactor (collapsing the ``task_source ==`` chain
onto a ``TaskSourceSpec.approach_diagnosis`` hook) is provably byte-identical.

The characterization snapshots below PASS both before and after the refactor.
``test_custom_task_source_formatter_is_dispatched`` exercises the *new*
capability — a task source supplying its own formatter — and fails until the
spec hook + dispatch exist.
"""

from types import SimpleNamespace as NS

from ksi.orchestrator.engine import _build_approach_diagnosis


def _trace(**kw):
    base = dict(native_score=0.0, runtime_meta={}, model_output="", tool_trace=[])
    base.update(kw)
    return NS(**base)


# ── Characterization snapshots (byte-identical before/after the refactor) ──────


def test_arc_resolved_snapshot():
    out = _build_approach_diagnosis(
        trace=_trace(native_score=1.0, runtime_meta={"arc_submit_trial_results": [{"test_index": 0, "reason": "ok"}]}),
        eval_result={"status": "correct", "arc_total_count": 3, "arc_correct_count": 3, "arc_pass_ratio": 1.0},
        outcome="resolved",
        task_source="arc",
    )
    assert out == (
        "Outcome: resolved (score: 1.0)\nEval status: correct\nARC score: 3/3 tests correct (pass_ratio=1.0)"
    )


def test_arc_failed_rejected_trials_snapshot():
    out = _build_approach_diagnosis(
        trace=_trace(
            runtime_meta={"arc_submit_trial_results": [{"test_index": 0, "reason": "shape_mismatch"}, {"reason": "ok"}]}
        ),
        eval_result={"status": "incorrect", "arc_total_count": 3, "arc_correct_count": 1, "arc_pass_ratio": 0.33},
        outcome="unresolved",
        task_source="arc",
    )
    assert out == (
        "Outcome: unresolved (score: 0.0)\n"
        "Eval status: incorrect\n"
        "ARC score: 1/3 tests correct (pass_ratio=0.33)\n"
        "Rejected hidden ARC trial(s): 1\n"
        "Next check: derive the transformation from visible train pairs before spending another trial.\n"
        "Next attempt: use the concrete evidence above to choose a different fix path or tighten the current one."
    )


def test_arc_parse_error_snapshot():
    out = _build_approach_diagnosis(
        trace=_trace(),
        eval_result={"status": "parse_error", "error": "bad json grid xxxxx"},
        outcome="unresolved",
        task_source="arc",
    )
    assert out == (
        "Outcome: unresolved (score: 0.0)\n"
        "Eval status: parse_error\n"
        "Rejected output format: bad json grid xxxxx\n"
        "Next check: return exactly the required grid/list-of-grids JSON.\n"
        "Next attempt: use the concrete evidence above to choose a different fix path or tighten the current one."
    )


def _swebench_eval():
    return {
        "instance_report": {
            "tests_status": {
                "FAIL_TO_PASS": {"success": ["t_a"], "failure": ["t_b"], "unknown": ["t_u"]},
                "PASS_TO_PASS": {"failure": ["t_r"], "unknown": []},
            }
        }
    }


def test_swebench_strict_counts_only_snapshot():
    out = _build_approach_diagnosis(
        trace=_trace(model_output="diff --git a/foo.py b/foo.py\n"),
        eval_result=_swebench_eval(),
        outcome="unresolved",
        task_source="swebench_pro",
        seed_test_files=False,
    )
    assert out == (
        "Outcome: unresolved (score: 0.0)\n"
        "Target tests now passing (good): 1\n"
        "Target tests STILL FAILING: 1\n"
        "Previously-passing tests REGRESSED: 1\n"
        "Expected tests not observed in parser output: 1\n"
        "Files modified: foo.py\n"
        "Next attempt: use the concrete evidence above to choose a different fix path or tighten the current one."
    )


def test_swebench_seeded_keeps_names_snapshot():
    out = _build_approach_diagnosis(
        trace=_trace(model_output="diff --git a/foo.py b/foo.py\n"),
        eval_result=_swebench_eval(),
        outcome="unresolved",
        task_source="swebench_pro",
        seed_test_files=True,
    )
    assert out == (
        "Outcome: unresolved (score: 0.0)\n"
        "Tests now passing (good): t_a\n"
        "Tests STILL FAILING (the patch did NOT fix these): t_b\n"
        "Tests REGRESSED (the patch BROKE these): t_r\n"
        "Expected tests not observed in parser output: t_u\n"
        "Files modified: foo.py\n"
        "Next attempt: use the concrete evidence above to choose a different fix path or tighten the current one."
    )


def test_swebench_all_pass_snapshot():
    out = _build_approach_diagnosis(
        trace=_trace(native_score=1.0),
        eval_result={
            "instance_report": {
                "tests_status": {
                    "FAIL_TO_PASS": {"success": ["t_a"], "failure": [], "unknown": []},
                    "PASS_TO_PASS": {"failure": [], "unknown": []},
                }
            }
        },
        outcome="resolved",
        task_source="swebench",
    )
    assert out == (
        "Outcome: resolved (score: 1.0)\nTarget tests now passing (good): 1\nAll target tests pass and no regressions."
    )


def test_swebench_no_tests_status_snapshot():
    out = _build_approach_diagnosis(
        trace=_trace(),
        eval_result={"status": "error_x"},
        outcome="unresolved",
        task_source="swebench_pro",
    )
    assert out == (
        "Outcome: unresolved (score: 0.0)\n"
        "Eval status: error_x\n"
        "Next attempt: use the concrete evidence above to choose a different fix path or tighten the current one."
    )


def test_tb2_snapshot():
    out = _build_approach_diagnosis(
        trace=_trace(
            runtime_meta={"reward": 0.0, "agent_exit_code": 0, "verifier_exit_code": 1},
            tool_trace=[{"tool_input": {"command": "ls"}}, {"tool_input": {"command": "pytest"}}],
        ),
        eval_result={},
        outcome="unresolved",
        task_source="terminal_bench_2",
    )
    assert out == (
        "Outcome: unresolved (score: 0.0)\n"
        "TB2 verifier summary: reward=0.0 agent_exit=0 verifier_exit=1\n"
        "Shell steps executed: 2\n"
        "Recent commands: ls | pytest\n"
        "Next attempt: use the concrete evidence above to choose a different fix path or tighten the current one."
    )


def test_generic_source_unresolved_snapshot():
    # polyglot/unknown sources fall through to the generic leg.
    out = _build_approach_diagnosis(
        trace=_trace(),
        eval_result={"status": "failed"},
        outcome="unresolved",
        task_source="polyglot",
    )
    assert out == (
        "Outcome: unresolved (score: 0.0)\n"
        "Eval status: failed\n"
        "Next attempt: use the concrete evidence above to choose a different fix path or tighten the current one."
    )


def test_generic_source_resolved_snapshot():
    out = _build_approach_diagnosis(
        trace=_trace(native_score=1.0),
        eval_result={},
        outcome="resolved",
        task_source="polyglot",
    )
    assert out == "Outcome: resolved (score: 1.0)"


def test_unknown_source_falls_back_to_generic():
    out = _build_approach_diagnosis(
        trace=_trace(),
        eval_result={"status": "boom"},
        outcome="unresolved",
        task_source="not_a_real_source",
    )
    assert out == (
        "Outcome: unresolved (score: 0.0)\n"
        "Eval status: boom\n"
        "Next attempt: use the concrete evidence above to choose a different fix path or tighten the current one."
    )


# ── New capability: a task source supplying its own diagnosis formatter ───────


def test_custom_task_source_formatter_is_dispatched():
    """A registered source's ``approach_diagnosis`` hook drives the per-source
    lines, with no ``task_source ==`` edit in the engine."""
    from ksi.tasks import registry

    def _fmt(*, trace, eval_result, outcome, seed_test_files):
        return [f"CUSTOM diag for {eval_result.get('status')}"]

    spec = registry.TaskSourceSpec(name="diag_fake_src", approach_diagnosis=_fmt)
    registry.register_task_source(spec)
    try:
        out = _build_approach_diagnosis(
            trace=_trace(),
            eval_result={"status": "whatever"},
            outcome="unresolved",
            task_source="diag_fake_src",
        )
    finally:
        for key in spec.all_names():
            registry.REGISTRY.pop(key, None)

    assert out == (
        "Outcome: unresolved (score: 0.0)\n"
        "CUSTOM diag for whatever\n"
        "Next attempt: use the concrete evidence above to choose a different fix path or tighten the current one."
    )


def test_arc_rejected_trials_capped_at_four_snapshot():
    # >4 rejected hidden trials: the ``[:4]`` cap means the reported count tops out at 4.
    out = _build_approach_diagnosis(
        trace=_trace(runtime_meta={"arc_submit_trial_results": [{"reason": "shape_mismatch"}] * 6}),
        eval_result={"status": "incorrect", "arc_total_count": 3, "arc_correct_count": 0, "arc_pass_ratio": 0.0},
        outcome="unresolved",
        task_source="arc",
    )
    assert out == (
        "Outcome: unresolved (score: 0.0)\n"
        "Eval status: incorrect\n"
        "ARC score: 0/3 tests correct (pass_ratio=0.0)\n"
        "Rejected hidden ARC trial(s): 4\n"
        "Next check: derive the transformation from visible train pairs before spending another trial.\n"
        "Next attempt: use the concrete evidence above to choose a different fix path or tighten the current one."
    )

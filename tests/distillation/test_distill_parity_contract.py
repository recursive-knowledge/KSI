"""Behavioral parity guard for the distillation prompt surface.

``src/kcsi/memory/parity.py`` names three agent-facing adaptive surfaces; the
distillation prompt is one of them. Unlike the MCP forum tools, the distill
renderers do NOT route ``eval_results`` through ``redact_solver_hidden_eval_fields``
-- their only protection is *render-time field selection* (``_fmt_eval_results`` /
``_fmt_arc_eval`` emit a fixed allow-list of scalars/counts; ``_fmt_attempts``
text-redacts ``trace_condensed`` / ``reflection``).

``tests/test_parity_redactor_contract.py`` scans source for hidden *key names*,
but a future edit that dumps raw ``eval_results`` (e.g. ``json.dumps(eval_results)``)
names no key and would slip past that grep while still leaking the hidden test
contract to the distill LLM -> next-generation seed. This test closes that gap
behaviorally: it feeds the real renderers an attempt loaded with every hidden
field class and asserts none of it survives in the rendered prompt text, while
the declared experience signal (outcome scalars/counts) is retained.
"""

from __future__ import annotations

from kcsi.distillation.prompts import _fmt_attempts, _fmt_eval_results

# Sentinels for each hidden field class. None may appear in rendered distill text.
_POLY_TAIL = "CANARY_POLYGLOT_TEST_RUNNER_TAIL"
_SWE_TAIL = "CANARY_SWEBENCH_TEST_RUNNER_TAIL"
_TOP_DETAIL = "CANARY_TOP_LEVEL_DETAIL"
_F2P_NAME = "secret_fail_to_pass_test_name"
_RAW_TEST_NAME = "secret_instance_report_test_name"
_VERIF_TAIL = "CANARY_VERIFIER_TAIL"
_VERIF_CLUE = "CANARY_VERIFIER_CLUE"
_FSIG = "CANARY_FAILURE_SIGNATURE"
_TRACE_FSIG = "CANARY_TRACE_FAILURE_SIGNATURE"
_REFL_TAIL = "CANARY_REFLECTION_VERIFIER_TAIL"
_ARC_GOLD_CELL = 424242  # arc_per_test[].detail.first_mismatch.expected


def _loaded_attempt() -> dict:
    """A distill-shaped attempt carrying every hidden field class + scalars."""
    return {
        "agent_id": "agent-1",
        "generation": 1,
        "native_score": 0.0,
        "model_output": "(agent's own output)",
        "eval_results": {
            "status": "unresolved",
            "resolved": False,
            "native_score": 0.0,
            # hidden test-runner tails (polyglot / SWE-bench Pro)
            "test_stdout_tail": _POLY_TAIL,
            "swebench_stderr_tail": _SWE_TAIL,
            # top-level + nested ARC grader answers
            "detail": _TOP_DETAIL,
            "arc_correct_count": 0,
            "arc_total_count": 1,
            "arc_per_test": [
                {
                    "test_index": 0,
                    "correct": False,
                    "expected_shape": [9, 9],
                    "detail": {"first_mismatch": {"expected": _ARC_GOLD_CELL}},
                }
            ],
            # SWE-bench hidden test identifiers (raw list + status names)
            "instance_report": {
                "status": "unresolved",
                "resolved": False,
                "tests": [_RAW_TEST_NAME],
                "tests_status": {"FAIL_TO_PASS": {"failure": [_F2P_NAME], "success": []}},
            },
        },
        "attempt_meta": {
            "reward": 0.0,
            "verifier_exit_code": 1,
            # terminal_bench_2 hidden verifier content
            "verifier_stdout_tail": _VERIF_TAIL,
            "verifier_clues": [_VERIF_CLUE],
            "failure_signature": _FSIG,
        },
        # stale hidden-marker fragments baked into derived text on older rows
        "trace_condensed": f"failure_signature={_TRACE_FSIG}; reward=0",
        "reflection": f"verifier_stdout_tail={_REFL_TAIL}; insight=tried X",
    }


_HIDDEN_SENTINELS = (
    _POLY_TAIL,
    _SWE_TAIL,
    _TOP_DETAIL,
    _F2P_NAME,
    _RAW_TEST_NAME,
    _VERIF_TAIL,
    _VERIF_CLUE,
    _FSIG,
    _TRACE_FSIG,
    _REFL_TAIL,
    str(_ARC_GOLD_CELL),
)


def test_fmt_attempts_renders_no_hidden_content():
    rendered = _fmt_attempts([_loaded_attempt()])
    for sentinel in _HIDDEN_SENTINELS:
        assert sentinel not in rendered, f"hidden content leaked into distill prompt: {sentinel!r}"
    # Declared experience signal is retained (else the test would pass trivially
    # by emitting nothing).
    assert "resolved=" in rendered
    assert "reward=" in rendered
    assert "native_score=" in rendered


def test_fmt_eval_results_renders_no_hidden_content():
    rendered = _fmt_eval_results(_loaded_attempt()["eval_results"])
    for sentinel in _HIDDEN_SENTINELS:
        assert sentinel not in rendered, f"hidden content leaked into eval summary: {sentinel!r}"
    # Anonymized count survives without the test name.
    assert "FAIL_TO_PASS" in rendered and "failure=1" in rendered
    assert "resolved=" in rendered

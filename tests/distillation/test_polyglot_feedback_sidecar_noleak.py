"""Canary: the polyglot test-feedback audit sidecar must never reach the agent.

``container_host.py`` stores the raw delivered/graded polyglot eval as
``runtime_meta.polyglot_test_feedback_eval_result`` (and
``polyglot_test_feedback_reuse_eligible``) — an AUDIT-only sidecar. Its docstring
claims "no code path reads ``runtime_meta_json`` back into MEMORY.md, the forum,
or distillation," but the raw eval can carry hidden test-runner tails the agent
never saw (grader-side output for reuse-eligible/final rounds).

Today nothing in the distill/seed render pipeline reads ``runtime_meta`` (grep of
``distillation/`` + ``runtime/seeding.py`` + ``enrichment_phase.py`` finds no
reference), so this test passes trivially now — its job is to FAIL if a future
change starts folding ``runtime_meta`` into an agent-facing render, forwarding a
pre-consolidated grader-side eval the agent never saw (the exact provenance
confusion documented once in ARM_G_REPORT.md).
"""

from __future__ import annotations

from kcsi.distillation.prompts import _fmt_attempts, _fmt_eval_results

_FB_STDOUT_TAIL = "CANARY_POLYGLOT_FEEDBACK_STDOUT_TAIL"
_FB_STDERR_TAIL = "CANARY_POLYGLOT_FEEDBACK_STDERR_TAIL"


def _attempt_with_feedback_sidecar() -> dict:
    """A distill-shaped attempt whose runtime_meta carries the audit sidecar."""
    return {
        "agent_id": "agent-1",
        "generation": 1,
        "native_score": 0.0,
        "model_output": "(agent's own output)",
        "eval_results": {"status": "unresolved", "resolved": False, "native_score": 0.0},
        # Audit-only sidecar — the raw grader-side eval the agent did NOT see for
        # reuse-eligible/final rounds, incl. hidden test-runner tails.
        "runtime_meta": {
            "polyglot_test_feedback_reuse_eligible": True,
            "polyglot_test_feedback_eval_result": {
                "status": "unresolved",
                "resolved": False,
                "native_score": 0.0,
                "test_stdout_tail": _FB_STDOUT_TAIL,
                "test_stderr_tail": _FB_STDERR_TAIL,
            },
        },
    }


def test_fmt_attempts_never_renders_polyglot_feedback_sidecar():
    rendered = _fmt_attempts([_attempt_with_feedback_sidecar()])
    for sentinel in (
        _FB_STDOUT_TAIL,
        _FB_STDERR_TAIL,
        "polyglot_test_feedback_eval_result",
        "polyglot_test_feedback_reuse_eligible",
    ):
        assert sentinel not in rendered, f"audit sidecar leaked into distill prompt: {sentinel!r}"
    # Not a trivial pass: the declared experience signal is still rendered.
    assert "native_score=" in rendered


def test_fmt_eval_results_strips_feedback_sidecar_tails_if_ever_rendered():
    # Defense in depth: even if the sidecar's eval dict were passed straight to
    # the eval renderer, its hidden test-runner tails are field-selected out.
    sidecar_eval = _attempt_with_feedback_sidecar()["runtime_meta"]["polyglot_test_feedback_eval_result"]
    rendered = _fmt_eval_results(sidecar_eval)
    assert _FB_STDOUT_TAIL not in rendered
    assert _FB_STDERR_TAIL not in rendered

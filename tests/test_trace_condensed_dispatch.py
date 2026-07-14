"""Seam 4 (issue #741): per-source condensed-trace dispatch.

Pins ``GenerationalOrchestrator._knowledge_trace_condensed`` for the tb2 leg and
the generic default so collapsing the ``task_source ==`` branch onto a
``TaskSourceSpec.trace_condensed`` hook is byte-identical. The capability test
exercises the new hook and fails until the spec field + dispatch exist.
"""

from types import SimpleNamespace as NS

from kcsi.orchestrator.engine import GenerationalOrchestrator

_f = GenerationalOrchestrator._knowledge_trace_condensed


def test_tb2_default_insight():
    trace = NS(
        runtime_meta={
            "task_source": "terminal_bench_2",
            "reward": 1.0,
            "agent_exit_code": 0,
            "verifier_exit_code": 0,
            "verifier_stdout_tail": "SECRET",
            "verifier_stderr_tail": "SECRET2",
        },
        model_output="x",
        native_score=1.0,
        error=None,
        tool_trace=[{"tool_input": {"command": "ls"}}],
    )
    out = _f(trace)
    assert out == (
        "TB2 attempt summary: reward=1.0 Verifier passed with reward 1.0. "
        "agent_exit=0 verifier_exit=0; tool_count=1; recent_commands=['ls']; "
        "Insight: (pending reflection)"
    )
    # Hidden verifier tails never leak into the condensed trace.
    assert "SECRET" not in out


def test_tb2_custom_insight():
    trace = NS(
        runtime_meta={
            "task_source": "terminal_bench_2",
            "reward": 1.0,
            "agent_exit_code": 0,
            "verifier_exit_code": 0,
        },
        model_output="x",
        native_score=1.0,
        error=None,
        tool_trace=[{"tool_input": {"command": "ls"}}],
    )
    out = _f(trace, insight_text="my insight")
    assert out == (
        "TB2 attempt summary: reward=1.0 Verifier passed with reward 1.0. "
        "agent_exit=0 verifier_exit=0; tool_count=1; recent_commands=['ls']; "
        "Insight: my insight"
    )


def test_generic_with_output():
    trace = NS(
        runtime_meta={"task_source": "polyglot"},
        model_output="I'll solve this.\nDef real approach here.",
        native_score=0.5,
        error=None,
        tool_trace=[],
    )
    assert _f(trace) == "Approach: Def real approach here.. Score: 0.5. Insight: (pending reflection)"


def test_generic_error_no_output():
    trace = NS(runtime_meta={"task_source": "arc"}, model_output="", native_score=0.0, error="boom", tool_trace=[])
    assert _f(trace) == "task failed: boom"


def test_generic_no_output_no_error():
    trace = NS(runtime_meta={"task_source": "arc"}, model_output="", native_score=0.0, error=None, tool_trace=[])
    assert _f(trace) == "Approach: (no output). Score: 0.0. Insight: (pending reflection)"


def test_custom_task_source_trace_condensed_is_dispatched():
    """A registered source's ``trace_condensed`` hook drives the condensed
    trace, with no ``task_source ==`` edit in the engine."""
    from kcsi.tasks import registry

    def _fmt(trace, *, insight_text):
        return f"CUSTOM tc score={trace.native_score} insight={insight_text}"

    spec = registry.TaskSourceSpec(name="tc_fake_src", trace_condensed=_fmt)
    registry.register_task_source(spec)
    try:
        trace = NS(
            runtime_meta={"task_source": "tc_fake_src"},
            model_output="x",
            native_score=0.9,
            error=None,
            tool_trace=[],
        )
        assert _f(trace, insight_text="hi") == "CUSTOM tc score=0.9 insight=hi"
    finally:
        for key in spec.all_names():
            registry.REGISTRY.pop(key, None)


def test_tb2_failed_attempt_snapshot():
    # The common (failed) TB2 path: reward<1 renders the "unresolved" verifier
    # summary that seeds the next-gen solver's MEMORY.md. Pin it byte-for-byte and
    # confirm the hidden verifier tails never leak into the condensed trace.
    trace = NS(
        runtime_meta={
            "task_source": "terminal_bench_2",
            "reward": 0.0,
            "agent_exit_code": 0,
            "verifier_exit_code": 1,
            "verifier_stdout_tail": "SECRET",
            "verifier_stderr_tail": "SECRET2",
        },
        model_output="x",
        native_score=0.0,
        error=None,
        tool_trace=[{"tool_input": {"command": "ls"}}],
    )
    out = _f(trace)
    assert out == (
        "TB2 attempt summary: reward=0.0 Verifier unresolved with reward 0. "
        "agent_exit=0 verifier_exit=1; tool_count=1; recent_commands=['ls']; "
        "Insight: (pending reflection)"
    )
    assert "SECRET" not in out

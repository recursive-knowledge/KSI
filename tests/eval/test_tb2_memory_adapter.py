from __future__ import annotations

from kcsi.models import TaskTrace
from kcsi.orchestrator.engine import GenerationalOrchestrator, _tb2_attempt_meta


def test_tb2_attempt_meta_extracts_normalized_verifier_evidence() -> None:
    trace = TaskTrace(
        generation=1,
        agent_id="agent-0",
        task_id="cancel-async-tasks",
        tool_trace=[
            {"tool_input": {"command": "cat /app/run.py"}},
            {"tool_input": {"command": "python3 /tmp/final_verification.py"}},
        ],
        runtime_meta={
            "task_source": "terminal_bench_2",
            "reward": 0.0,
            "agent_exit_code": 0,
            "verifier_exit_code": 0,
            "verifier_stdout_tail": (
                "FAILED ../tests/test_outputs.py::test_tasks_cancel_above_max_concurrent\n"
                "E       AssertionError: assert 0 == 2\n"
                "Connection refused should not appear here\n"
            ),
            "verifier_stderr_tail": "debconf warning",
        },
    )

    meta = _tb2_attempt_meta(trace)

    assert meta["verified_outcome"] == "Verifier unresolved with reward 0."
    assert meta["last_state_change"] == "python3 /tmp/final_verification.py"
    assert "AssertionError" in meta["failure_signature"]
    assert any(
        "FAILED ../tests/test_outputs.py::test_tasks_cancel_above_max_concurrent" in clue
        for clue in meta["verifier_clues"]
    )
    assert isinstance(meta["verifier_clues"], list)
    assert meta["verifier_clues"]


def test_tb2_attempt_meta_skips_traceback_boilerplate_clues() -> None:
    trace = TaskTrace(
        generation=1,
        agent_id="agent-0",
        task_id="nginx-request-logging",
        runtime_meta={
            "task_source": "terminal_bench_2",
            "reward": 0.0,
            "agent_exit_code": 0,
            "verifier_exit_code": 0,
            "verifier_stdout_tail": (
                "except LocationValueError as e:\n"
                "raise ConnectionError(e, request=request)\n"
                "requests.exceptions.ConnectionError: HTTPConnectionPool(host='localhost', port=8080): "
                "Max retries exceeded with url: /test-1 (Caused by NewConnectionError"
            ),
            "verifier_stderr_tail": "",
        },
    )

    meta = _tb2_attempt_meta(trace)

    assert "except LocationValueError as e:" not in meta["verifier_clues"]
    assert "raise ConnectionError" not in meta["verifier_clues"]
    assert "ConnectionError" in meta["failure_signature"]


def test_knowledge_trace_condensed_uses_normalized_tb2_fields() -> None:
    trace = TaskTrace(
        generation=2,
        agent_id="agent-1",
        task_id="git-multibranch",
        tool_trace=[
            {"tool_input": {"command": "git push -u origin main"}},
        ],
        runtime_meta={
            "task_source": "terminal_bench_2",
            "reward": 0.0,
            "agent_exit_code": 0,
            "verifier_exit_code": 0,
            "verifier_stdout_tail": "Permission denied, please try again.",
            "verifier_stderr_tail": "",
        },
    )

    condensed = GenerationalOrchestrator._knowledge_trace_condensed(
        trace,
        insight_text="retry deploy verification",
    )

    assert "TB2 attempt summary:" in condensed
    # Held-out verifier output (hidden TB2 pytest) must NOT be baked into the
    # condensed trace — it flows to the next-gen solver's MEMORY.md seed.
    assert "failure_signature" not in condensed
    assert "verifier_clues" not in condensed
    assert "Permission denied" not in condensed
    # The agent's own observations still carry forward.
    assert "recent_commands=['git push -u origin main']" in condensed
    assert "Insight: retry deploy verification" in condensed

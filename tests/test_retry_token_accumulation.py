"""Retry paths must surface failed-attempt token cost to accounting.

Background: ``_run_retryable_forum_task`` and ``_run_agent_stage``'s retry
loop both used to return only the success attempt's ``token_usage``. Failed
attempts whose ``tokens_source`` was ``per_turn_sum`` (SDK race after real
per-turn deltas, scheduled-task end without a result event) silently
consumed billable tokens that vanished from every accounting surface
(``token_phases`` rows, ``agent.token_usage`` counters, the campaign totals
JSON).

These tests pin the post-fix contract: failed-attempt tokens are
accumulated into the returned ``TokenUsage`` (success and terminal-failure
paths) and surfaced in ``runtime_meta`` for forensic auditability.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ksi.models import AgentState, GenerationConfig, TaskSpec
from ksi.orchestrator.engine import (
    GenerationalOrchestrator,
    NoopPersistence,
    _accumulate_failed_attempt_tokens,
    _run_retryable_forum_task,
    _runtime_retry_meta,
)
from ksi.orchestrator.execution_phase import EngineExecutionPhaseService
from ksi.runtime.normalize import SilentAgentRuntimeError
from ksi.runtime.types import RuntimeResult
from ksi.tokens import TokenUsage


def _per_turn_sum_meta(input_tokens: int, output_tokens: int, cache_read: int = 0) -> dict:
    """Build a runtime_meta dict shaped like a per_turn_sum failed attempt."""
    return {
        "status": "error",
        "tokens_source": "per_turn_sum",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": 0,
    }


def test_accumulate_failed_attempt_tokens_sums_each_meta() -> None:
    metas = [
        _per_turn_sum_meta(input_tokens=100, output_tokens=20, cache_read=14_000),
        _per_turn_sum_meta(input_tokens=200, output_tokens=10, cache_read=30_000),
    ]
    total = _accumulate_failed_attempt_tokens(metas)
    assert total.input_tokens == 300
    assert total.output_tokens == 30
    assert total.cache_read_input_tokens == 44_000


def test_accumulate_failed_attempt_tokens_handles_empty_and_garbage() -> None:
    assert _accumulate_failed_attempt_tokens(None).total == 0
    assert _accumulate_failed_attempt_tokens([]).total == 0
    # Non-dict entries are skipped, valid entries still summed.
    metas = [None, "garbage", _per_turn_sum_meta(input_tokens=5, output_tokens=3)]
    total = _accumulate_failed_attempt_tokens(metas)
    assert total.input_tokens == 5
    assert total.output_tokens == 3


def test_accumulate_failed_attempt_tokens_logs_dropped_extractions(caplog) -> None:
    """A runtime_meta that raises during extraction (beyond the
    TypeError/ValueError extract_token_usage already swallows) must be
    dropped with a WARNING log, not silently -- #981.
    """
    import logging

    class _RaisingDict(dict):
        def get(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("boom")

    metas = [_RaisingDict(_per_turn_sum_meta(input_tokens=5, output_tokens=3))]
    with caplog.at_level(logging.WARNING, logger="ksi.orchestrator.task_retry"):
        total = _accumulate_failed_attempt_tokens(metas)
    assert total.total == 0
    assert any("dropped 1/1" in record.message for record in caplog.records)


def test_accumulate_failed_attempt_tokens_log_dropped_false_suppresses_warning(caplog) -> None:
    """``log_dropped=False`` must silence the WARNING without changing the
    computed (zero, since the only entry is dropped) total -- this is the
    knob callers use to avoid double-logging a single real drop event
    across the ``_runtime_retry_meta`` + direct-call pattern.
    """
    import logging

    class _RaisingDict(dict):
        def get(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("boom")

    metas = [_RaisingDict(_per_turn_sum_meta(input_tokens=5, output_tokens=3))]
    with caplog.at_level(logging.WARNING, logger="ksi.orchestrator.task_retry"):
        total = _accumulate_failed_attempt_tokens(metas, log_dropped=False)
    assert total.total == 0
    assert not any("dropped" in record.message for record in caplog.records)


def test_runtime_retry_meta_surfaces_aggregated_failed_tokens() -> None:
    attempt_errors = [{"attempt": 1, "max_attempts": 2, "error_type": "X", "error": "x"}]
    failed_metas = [_per_turn_sum_meta(input_tokens=100, output_tokens=20, cache_read=14_000)]
    meta = _runtime_retry_meta(
        attempt_errors,
        terminal_failure=False,
        failed_runtime_metas=failed_metas,
    )
    assert "retry_failed_attempts_token_usage" in meta
    tu = meta["retry_failed_attempts_token_usage"]
    assert tu["input_tokens"] == 100
    assert tu["output_tokens"] == 20
    assert tu["cache_read_input_tokens"] == 14_000


def test_runtime_retry_meta_omits_token_field_when_no_failed_attempts() -> None:
    attempt_errors = [{"attempt": 1, "max_attempts": 2, "error_type": "X", "error": "x"}]
    meta = _runtime_retry_meta(attempt_errors, terminal_failure=False, failed_runtime_metas=[])
    assert "retry_failed_attempts_token_usage" not in meta


def test_runtime_retry_meta_plus_direct_accumulate_logs_dropped_once(caplog) -> None:
    """Regression test for the duplicate-WARNING bug: 3 of 4 call flows
    (``execution_phase.py``'s success-after-retry path, ``forum_runtime.py``'s
    success and terminal-failure paths) call
    ``_accumulate_failed_attempt_tokens`` twice on the same
    ``failed_runtime_metas`` list for the same event -- once indirectly via
    ``_runtime_retry_meta`` (populating ``retry_failed_attempts_token_usage``
    in ``runtime_meta``) and once again directly (folding the sum into the
    caller's own ``token_usage``). Before the fix, one real extraction
    failure logged the identical WARNING line twice. This pins the call
    pattern those 3 sites use -- ``_runtime_retry_meta(...,
    log_dropped_tokens=False)`` followed by a direct
    ``_accumulate_failed_attempt_tokens(...)`` call -- to exactly one
    WARNING, and would fail (2 warnings) if ``log_dropped_tokens=False``
    were dropped from a call site again.
    """
    import logging

    class _RaisingOnTokenExtractionDict(dict):
        """Raises only on the key ``extract_token_usage`` reads first, so the
        unrelated ``native_session_memory`` lookup ``_runtime_retry_meta``
        also performs on ``failed_meta`` doesn't itself raise.
        """

        def get(self, key, *args, **kwargs):  # noqa: ANN002, ANN003
            if key == "cache_creation_input_tokens":
                raise RuntimeError("boom")
            return super().get(key, *args, **kwargs)

    attempt_errors = [{"attempt": 1, "max_attempts": 2, "error_type": "X", "error": "x"}]
    metas = [_RaisingOnTokenExtractionDict(_per_turn_sum_meta(input_tokens=5, output_tokens=3))]

    with caplog.at_level(logging.WARNING, logger="ksi.orchestrator.task_retry"):
        meta = _runtime_retry_meta(
            attempt_errors,
            terminal_failure=False,
            failed_runtime_metas=metas,
            log_dropped_tokens=False,
        )
        total = _accumulate_failed_attempt_tokens(metas)

    # The extraction failure is dropped from both computed values (numeric
    # behavior is unchanged by the log-dedup fix)...
    assert "retry_failed_attempts_token_usage" not in meta
    assert total.total == 0
    # ...but the drop is logged exactly once, not once per call site.
    drop_warnings = [record for record in caplog.records if "dropped 1/1" in record.message]
    assert len(drop_warnings) == 1


def test_run_retryable_forum_task_success_after_retry_accumulates_tokens() -> None:
    """Pre-fix, the returned TokenUsage was *only* the success attempt's
    cost. Failed attempts that consumed real cache tokens were dropped from
    accounting. Post-fix, the helper must return success_tokens + sum of
    failed-attempt tokens.
    """
    success_meta = _per_turn_sum_meta(input_tokens=200, output_tokens=80)
    failed_meta = _per_turn_sum_meta(input_tokens=100, output_tokens=20, cache_read=14_000)
    call_state = {"attempt": 0}

    def run_once() -> RuntimeResult:
        call_state["attempt"] += 1
        if call_state["attempt"] == 1:
            raise SilentAgentRuntimeError(
                "service unavailable: SDK iterator drained",
                runtime_meta=failed_meta,
            )
        return RuntimeResult(
            output="discussion text",
            tool_trace=[],
            runtime_meta={"status": "success", **success_meta},
            token_usage=TokenUsage(input_tokens=200, output_tokens=80),
        )

    token_usage, runtime_meta, output = _run_retryable_forum_task(
        run_once=run_once,
        generation=1,
        agent_id="agent-0",
        phase_label="per-task discussion",
        attempts=2,
    )

    assert call_state["attempt"] == 2  # one retry consumed
    # Success attempt 200 in + failed attempt 100 in = 300 total
    assert token_usage.input_tokens == 300
    assert token_usage.output_tokens == 100
    assert token_usage.cache_read_input_tokens == 14_000
    # Forensic field on runtime_meta exposes the failed-attempt cost so
    # analytics can split first-try vs retry costs without re-walking
    # per-attempt blobs.
    assert "retry_failed_attempts_token_usage" in runtime_meta
    assert runtime_meta["retry_failed_attempts_token_usage"]["input_tokens"] == 100
    assert output == "discussion text"


def test_run_retryable_forum_task_terminal_failure_returns_failed_attempt_tokens() -> None:
    """Even when every retry fails, the helper must surface the aggregated
    failed-attempt cost so the caller's per-agent / per-generation token
    accounting reflects what was actually billed.
    """
    failed_metas = [
        _per_turn_sum_meta(input_tokens=100, output_tokens=20, cache_read=14_000),
        _per_turn_sum_meta(input_tokens=120, output_tokens=15, cache_read=20_000),
    ]
    call_state = {"attempt": 0}

    def run_once() -> RuntimeResult:
        call_state["attempt"] += 1
        meta = failed_metas[min(call_state["attempt"], len(failed_metas)) - 1]
        raise SilentAgentRuntimeError(
            "provider unavailable",
            runtime_meta=meta,
        )

    token_usage, runtime_meta, output = _run_retryable_forum_task(
        run_once=run_once,
        generation=1,
        agent_id="agent-0",
        phase_label="per-task discussion",
        attempts=2,
    )

    assert call_state["attempt"] == 2  # both attempts ran and failed
    # Both failed attempts' tokens are summed into the returned usage.
    assert token_usage.input_tokens == 220
    assert token_usage.output_tokens == 35
    assert token_usage.cache_read_input_tokens == 34_000
    # Terminal failure still records forum_error and the retry forensics.
    assert runtime_meta.get("forum_error")
    assert "forum_retry_meta" in runtime_meta
    assert runtime_meta["forum_retry_meta"]["retry_failed_attempts_token_usage"]["input_tokens"] == 220
    assert output == ""


def test_run_retryable_forum_task_first_try_success_unchanged() -> None:
    """The fix must not regress the (common) first-try-success path: no
    retry meta and the success attempt's token_usage flows through verbatim.
    """

    def run_once() -> RuntimeResult:
        return RuntimeResult(
            output="ok",
            tool_trace=[],
            runtime_meta={"status": "success"},
            token_usage=TokenUsage(input_tokens=50, output_tokens=10),
        )

    token_usage, runtime_meta, output = _run_retryable_forum_task(
        run_once=run_once,
        generation=1,
        agent_id="agent-0",
        phase_label="per-task discussion",
        attempts=3,
    )

    assert token_usage.input_tokens == 50
    assert token_usage.output_tokens == 10
    assert "retry_failed_attempts_token_usage" not in runtime_meta
    assert "retry_attempts" not in runtime_meta
    assert output == "ok"


def _make_service() -> EngineExecutionPhaseService:
    orch = GenerationalOrchestrator(
        config=GenerationConfig(num_generations=1, num_agents=1, no_memory=True),
        runtime=MagicMock(),
        evaluator=MagicMock(),
        llm=MagicMock(),
        persistence=NoopPersistence(),
    )
    return EngineExecutionPhaseService(orch)


def test_eval_one_attempt_terminal_failure_accumulates_failed_attempt_tokens() -> None:
    """The task-execution terminal-failure path (#1121): when every retryable
    attempt fails, ``_run_agent_stage`` attaches the failed attempts' runtime
    metas + retry summary to the terminal exception. ``_eval_one_attempt`` must
    fold the aggregated billed cost into ``trace.token_usage`` (not zero it) and
    surface ``retry_failed_attempts_token_usage`` in ``runtime_meta``.
    """
    service = _make_service()
    agent = AgentState(id="agent-0", workstream="w")
    task = TaskSpec(id="t1", repo="r", prompt="solve")

    # Two retryable SilentAgentRuntimeErrors carrying per-turn token counters,
    # exactly as ``_run_agent_stage`` accumulates them into
    # ``failed_runtime_metas`` before the terminal raise.
    failed_metas = [
        _per_turn_sum_meta(input_tokens=100, output_tokens=20, cache_read=14_000),
        _per_turn_sum_meta(input_tokens=120, output_tokens=15, cache_read=20_000),
    ]
    attempt_errors = [
        {"attempt": 1, "max_attempts": 2, "error_type": "SilentAgentRuntimeError", "error": "boom"},
        {"attempt": 2, "max_attempts": 2, "error_type": "SilentAgentRuntimeError", "error": "boom"},
    ]
    terminal_exc = SilentAgentRuntimeError("provider unavailable", runtime_meta=failed_metas[-1])
    # Mirror the two attributes ``_run_agent_stage`` sets on the terminal exc.
    setattr(
        terminal_exc,
        "runtime_retry_meta",
        _runtime_retry_meta(
            attempt_errors,
            terminal_failure=True,
            failed_runtime_metas=failed_metas,
            log_dropped_tokens=False,
        ),
    )
    setattr(terminal_exc, "failed_runtime_metas", failed_metas)

    trace, insight, lessons, extra_tokens = service._eval_one_attempt(agent, task, None, terminal_exc, 1, {"t1": task})

    # The task terminally failed...
    assert trace.error is not None
    assert trace.native_score is None
    # ...but its real billed cost (sum of both failed attempts) is persisted,
    # not zeroed — this is the #1121 fix.
    assert trace.token_usage.total > 0
    assert trace.token_usage.input_tokens == 220
    assert trace.token_usage.output_tokens == 35
    assert trace.token_usage.cache_read_input_tokens == 34_000
    # And the retry forensics ride along in runtime_meta on the silent branch.
    assert "retry_failed_attempts_token_usage" in trace.runtime_meta
    assert trace.runtime_meta["retry_failed_attempts_token_usage"]["input_tokens"] == 220
    assert trace.runtime_meta.get("runtime_attempt_errors") == attempt_errors
    # no_memory short-circuits reflection.
    assert insight is None
    assert lessons == []
    assert extra_tokens == 0

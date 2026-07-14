"""Unit tests for kcsi.orchestrator.engine._is_retryable_task_error.

Covers the transient markers that feed into the existing max_task_retries
loop. Specifically pins the SDK-stream-race signatures added after the
audit_3x2 sweep (3/6 polyglot attempts hit this gotcha) and the
isinstance-based ``SilentAgentRuntimeError`` path that separates real
stream races (retryable) from container auth/startup failures (not).
"""

from __future__ import annotations

import pytest

from kcsi.orchestrator.engine import _is_retryable_task_error
from kcsi.runtime.normalize import SilentAgentRuntimeError


@pytest.mark.parametrize(
    "message",
    [
        # Pre-existing transient signatures (regression guard).
        "Connection reset by peer",
        "HTTP 429 Too Many Requests",
        "timed out after 60s",
        "service unavailable",
        # SDK stream race signatures — these are the new additions.
        "agent-runner error envelope for task X (agent=a0, generation=1): "
        "SDK query loop drained without yielding any assistant/result message (messageCount=0)",
        "agent-runner error envelope for task X (agent=a0, generation=1): "
        "SDK query iterator threw AbortError: stream closed",
        "SDK emitted an empty result event (no text, zero tokens, empty tool trace)",
        "Silent agent-runner failure for task X (agent=a0, generation=1): ...",
    ],
)
def test_retryable(message: str) -> None:
    assert _is_retryable_task_error(Exception(message)) is True, message


@pytest.mark.parametrize(
    "message",
    [
        "",
        "invalid prompt: refused by usage policy",
        "parse_error:empty_model_output",
        "container exited with exit=137",  # OOM — explicit non-retryable
        "container exited with exit=139",  # segfault — explicit non-retryable
        "no patch produced",
        "missing report file",
        "task failed for unknown reasons",  # generic message, no marker — non-retryable
    ],
)
def test_not_retryable(message: str) -> None:
    assert _is_retryable_task_error(Exception(message)) is False, message


# ---------------------------------------------------------------------------
# SilentAgentRuntimeError branch — isinstance-based decision path.
# ---------------------------------------------------------------------------


def _silent(message: str, **meta) -> SilentAgentRuntimeError:
    return SilentAgentRuntimeError(message, runtime_meta=meta)


def test_silent_with_session_recovery_is_retryable() -> None:
    """SDK drained but session log recovery succeeded → CLI actually ran."""
    exc = _silent(
        "Silent agent-runner failure for task X ...",
        tokens_source="session_recovery",
        status="recovered_from_session",
    )
    assert _is_retryable_task_error(exc) is True


def test_silent_with_session_recovery_status_only_is_retryable() -> None:
    """Tolerate either tokens_source or status signalling recovery."""
    exc = _silent(
        "Silent agent-runner failure for task X ...",
        status="recovered_from_session",
    )
    assert _is_retryable_task_error(exc) is True


def test_silent_without_recovery_is_not_retryable() -> None:
    """No session-log recovery = CLI never produced output = auth/startup failure."""
    # SILENT_FAILURE_MESSAGE explicitly names auth/startup as a cause.
    exc = _silent(
        "Silent agent-runner failure for task X ... agent-runner produced no "
        "output (0 tokens, 0 tool calls, no model_output). The SDK query loop "
        "exited without emitting any messages -- likely an auth/startup failure "
        "inside the container or a claude-agent-sdk stream that drained without "
        "yielding any events.",
        tokens_source="unavailable",
        status="silent_failure",
    )
    assert _is_retryable_task_error(exc) is False


def test_silent_with_empty_meta_is_not_retryable() -> None:
    """Defensive: missing runtime_meta defaults to not-retryable (no signal = no retry)."""
    exc = _silent("Silent agent-runner failure for task X ...")
    assert _is_retryable_task_error(exc) is False


def test_silent_with_empty_result_event_recovery_failed_is_not_retryable() -> None:
    """SDK emitted an empty result event AND session-log recovery failed.

    When recovery fails, the CLI produced nothing on disk — consistent with a
    disk-full condition, broken MCP, or misconfigured container. Not transient.
    """
    exc = _silent(
        "agent-runner error envelope for task X: SDK emitted an empty result "
        "event (no text, zero tokens, empty tool trace)",
        tokens_source="unavailable",
        status="error",
    )
    assert _is_retryable_task_error(exc) is False


# ---------------------------------------------------------------------------
# Widened gate: tokens_source=per_turn_sum should retry (PR #485 follow-up).
#
# The revalidation sweep showed 0/34 retries because the surviving polyglot
# errors had tokens_source=per_turn_sum (CLI ran partially: real tool calls
# + thousands of cache-read tokens) or tokens_source=unavailable (no
# telemetry at all). The original PR #485 only allowed session_recovery,
# leaving the per_turn_sum case stranded.
#
# Pinned positive case from the actual revalidation DB:
#   results/cross_runner_sweeps/revalidate_haiku_haiku_audit/src/kcsi/knowledge/
#   polyglot_haiku_audit_revalidate_haiku_runtime.sqlite, attempt 4
#   (javascript__queen-attack, agent-0, gen 2): 14418 cache_creation +
#   45201 cache_read tokens, 2 Read tool calls, 4697 chars of session
#   memory — clearly a partial CLI execution that should retry.
# ---------------------------------------------------------------------------


def test_silent_with_per_turn_sum_is_retryable() -> None:
    """Per-turn telemetry without final result event → CLI ran partially → retry."""
    exc = _silent(
        "agent-runner error envelope for task javascript__queen-attack "
        "(agent=agent-0, generation=2): agent-runner returned status=error",
        tokens_source="per_turn_sum",
        status="error",
    )
    assert _is_retryable_task_error(exc) is True


def test_silent_per_turn_sum_with_pending_tool_calls_is_retryable() -> None:
    """Forum task ended with pending tool calls — common transient scheduling glitch."""
    exc = _silent(
        "agent-runner error envelope for task X (agent=a0, generation=1): "
        "Scheduled forum task ended with pending tool call(s): 2; "
        "refusing fallback success.",
        tokens_source="per_turn_sum",
        status="error",
    )
    assert _is_retryable_task_error(exc) is True


def test_silent_per_turn_sum_with_iterator_threw_is_retryable() -> None:
    """SDK iterator threw mid-stream after partial output — retry."""
    exc = _silent(
        "agent-runner error envelope for task Y: SDK query iterator threw AbortError: stream closed",
        tokens_source="per_turn_sum",
        status="error",
    )
    assert _is_retryable_task_error(exc) is True


def test_silent_per_turn_sum_but_auth_startup_in_meta_is_not_retryable() -> None:
    """If meta.error explicitly carries an auth/startup signature, refuse retry.

    Belt-and-braces: even a per_turn_sum meta should not retry if the runtime
    has explicitly tagged the failure as a deterministic auth/startup issue.
    """
    exc = _silent(
        "agent-runner error envelope: ...",
        tokens_source="per_turn_sum",
        status="error",
        error="auth/startup failure inside the container",
    )
    assert _is_retryable_task_error(exc) is False


def test_silent_per_turn_sum_but_usage_policy_in_meta_is_not_retryable() -> None:
    """A usage-policy refusal embedded in meta.error is deterministic."""
    exc = _silent(
        "agent-runner error envelope for task X: ...",
        tokens_source="per_turn_sum",
        status="error",
        error="Request flagged as potentially violating usage policy",
    )
    assert _is_retryable_task_error(exc) is False


def test_silent_per_turn_sum_but_oom_in_meta_is_not_retryable() -> None:
    """exit=137 (OOM) embedded in meta should still be non-retryable."""
    exc = _silent(
        "agent-runner error envelope for task X: ...",
        tokens_source="per_turn_sum",
        status="error",
        error="Container exited with exit=137 after running out of memory",
    )
    assert _is_retryable_task_error(exc) is False


def test_silent_unavailable_default_is_still_not_retryable() -> None:
    """Conservative: tokens_source=unavailable + status=error stays non-retryable.

    Pinned from the actual revalidation DB: attempt 1 (python__dot-dsl,
    agent-2, gen 1) had 0 tokens, 0 tool calls, 214 chars of session memory.
    Indistinguishable from auth/startup hard-fail without further signal.
    """
    exc = _silent(
        "agent-runner error envelope for task python__dot-dsl "
        "(agent=agent-2, generation=1): agent-runner returned status=error",
        tokens_source="unavailable",
        status="error",
    )
    assert _is_retryable_task_error(exc) is False


def test_silent_silent_failure_status_overrides_per_turn_sum() -> None:
    """status=silent_failure is always non-retryable regardless of tokens_source.

    Defensive: the silent_failure path is only emitted when the runner
    detected 0/0/0/0 + no output, so tokens_source should be unavailable.
    But pin the invariant in case a future code path mis-sets it.
    """
    exc = _silent(
        "Silent agent-runner failure for task X ...",
        tokens_source="per_turn_sum",
        status="silent_failure",
    )
    assert _is_retryable_task_error(exc) is False


def test_silent_recovered_from_session_with_unavailable_tokens_is_retryable() -> None:
    """status=recovered_from_session is sufficient on its own (tolerate odd meta)."""
    exc = _silent(
        "Recovered from on-disk session log",
        tokens_source="unavailable",  # nonsensical pairing, but recovery wins
        status="recovered_from_session",
    )
    assert _is_retryable_task_error(exc) is True


# ---------------------------------------------------------------------------
# Upstream-provider transient gate (HTTP 5xx / 429 / upstream connect reset).
#
# Distinct from the SDK-stream-race signatures: when the LLM provider itself
# returns a transient error before the SDK ever yielded telemetry, the
# tokens_source is "unavailable" but the failure is provider-side, not
# auth/startup. Pinned from a 50-task ARC2 c=50 cross-task forum run where
# agent-34 hit Anthropic 503 once and was silently dropped instead of
# retried.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "envelope",
    [
        # Real envelope shape from container_host._raise_silent at agent-runner level.
        "agent-runner error envelope for task __cross_task_forum__g1_r0_agent-34 "
        "(agent=agent-34, generation=1): Anthropic API returned non-JSON response "
        "(503): upstream connect error or disconnect/reset before headers. "
        "reset reason: connection termination.",
        "agent-runner error envelope for task X: HTTP 429 too many requests",
        "agent-runner error envelope for task X: 502 bad gateway",
        "agent-runner error envelope for task X: anthropic returned 504 gateway timeout",
        "agent-runner error envelope for task X: provider overloaded",
        # Node undici upstream-fetch failures (HeadersTimeoutError surfacing
        # as "fetch failed" in the exception text). Observed on agent-31 in
        # a 10-gen ARC2 c=50 cross-task forum.
        "agent-runner error envelope for task __cross_task_forum__g1_r1_agent-31 "
        "(agent=agent-31, generation=1): fetch failed",
        "agent-runner error envelope: fetch failed cause HeadersTimeoutError: Headers Timeout Error",
    ],
)
def test_silent_upstream_provider_blip_is_retryable(envelope: str) -> None:
    """Provider-side 5xx / 429 / upstream connect reset → retry, no telemetry needed."""
    exc = _silent(envelope, tokens_source="unavailable", status="error")
    assert _is_retryable_task_error(exc) is True, envelope


def test_silent_upstream_blip_in_meta_error_is_retryable() -> None:
    """Marker in meta.error (not just exception message) also flips the gate."""
    exc = _silent(
        "agent-runner error envelope for task X: ...",
        tokens_source="unavailable",
        status="error",
        error="Anthropic API returned non-JSON response (503): upstream connect error",
    )
    assert _is_retryable_task_error(exc) is True


def test_silent_upstream_blip_with_auth_startup_signature_is_not_retryable() -> None:
    """auth/startup hard-deny still wins over the upstream marker."""
    exc = _silent(
        "agent-runner error envelope: 503 service unavailable",
        tokens_source="unavailable",
        status="error",
        error="auth/startup failure inside the container",
    )
    assert _is_retryable_task_error(exc) is False

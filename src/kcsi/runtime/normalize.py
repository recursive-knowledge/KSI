from __future__ import annotations

import json
import logging
from typing import Any

from ..errors import KcsiError, find_container_registry_error
from ..tokens import TokenUsage

log = logging.getLogger(__name__)

# Sentinel status value written into runtime_meta when the agent-runner exits
# with zero observable activity (no tokens, no tool calls, empty output). The
# engine-side exception handler keys off this to set trace.error so the
# attempt is recorded as a failure instead of silently succeeding.
SILENT_FAILURE_STATUS = "silent_failure"
SILENT_FAILURE_MESSAGE = (
    "agent-runner produced no output (0 tokens, 0 tool calls, no model_output). "
    "The SDK query loop exited without emitting any messages -- likely an auth/startup "
    "failure inside the container or a claude-agent-sdk stream that drained without "
    "yielding any events. Treating as a runtime error instead of success."
)

# Sentinel status emitted by the in-container agent-runner when the SDK
# iterator drained without yielding events BUT the on-disk session log
# (/home/node/.claude/projects/*.jsonl) had usable content. The agent-runner
# reconstructs the last assistant message and approximate tokens from that
# log and emits status=recovered_from_session with tokens_source=session_recovery.
# Downstream normalize / host / engine code treats this as a REAL attempt
# (not a runtime failure) so the reconstructed output can be scored.
RECOVERED_FROM_SESSION_STATUS = "recovered_from_session"


class SilentAgentRuntimeError(KcsiError, RuntimeError):
    """Runtime error that preserves the silent-failure ``runtime_meta``.

    The plain ``RuntimeError`` path in the engine's exception handler writes
    ``trace.runtime_meta = {}``, which silently strips ``native_session_memory``
    and ``raw_native_session_memory`` from the attempt row. That threw away the
    one artifact forensics need -- the on-host session transcript harvested by
    ``kcsi.runtime.native_memory.collect_native_session_memory`` after the
    container returned.

    This subclass carries that meta across the raise/except boundary so the
    engine can still persist it. ``runtime_meta`` is a plain dict; it may be
    sliced later (e.g. at ``KCSI_NATIVE_MEMORY_MAX_CHARS``) before DB write.
    """

    def __init__(self, message: str, *, runtime_meta: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.runtime_meta: dict[str, Any] = dict(runtime_meta or {})


def _int_value(mapping: dict[str, Any] | None, key: str, default: int = 0) -> int:
    if not isinstance(mapping, dict):
        return default
    try:
        return int(mapping.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _cached_input_tokens(meta: dict[str, Any]) -> int:
    """Extract OpenAI/Anthropic cache-read usage from normalized metadata."""
    explicit = _int_value(meta, "cache_read_input_tokens")
    if explicit:
        return explicit
    details = meta.get("input_tokens_details")
    if not isinstance(details, dict):
        details = meta.get("prompt_tokens_details")
    return _int_value(details, "cached_tokens") or _int_value(details, "cache_read_input_tokens")


def _openai_raw_input_details(meta: dict[str, Any]) -> bool:
    return isinstance(meta.get("input_tokens_details"), dict) or isinstance(
        meta.get("prompt_tokens_details"),
        dict,
    )


def _normalized_input_tokens(meta: dict[str, Any], cache_read: int, cache_creation: int) -> int:
    raw_input = _int_value(meta, "input_tokens")
    if _openai_raw_input_details(meta):
        return max(0, raw_input - cache_read - cache_creation)
    return raw_input


def try_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


# Framing emitted by ``container/entrypoint.sh`` when ``npx tsc`` fails during
# the hash-mismatch recompile. Without this framing the host used to see only
# "Container exited with code 2: " (30 chars) because the container's stderr
# buffer had already been clipped to 200 chars by the runner, swallowing the
# actual TypeScript diagnostic.
_TSC_COMPILE_FAILED_HEADER = "====TSC_COMPILE_FAILED===="
_TSC_COMPILE_FAILED_FOOTER = "====END_TSC_COMPILE_FAILED===="
_TSC_EXCERPT_MAX_CHARS = 4000


def extract_tsc_compile_error(stderr: str | None) -> str | None:
    """Best-effort extraction of the TSC compile-error body from container stderr.

    Returns the text between ``====TSC_COMPILE_FAILED====`` and
    ``====END_TSC_COMPILE_FAILED====`` (markers stripped) when the header is
    present. If the footer is missing — e.g. stderr truncated at buffer
    boundary — returns everything from the header to the end-of-buffer, so the
    operator still sees a useful excerpt instead of nothing.

    Returns ``None`` when the header is absent (backward-compatible: callers
    should leave their existing error text untouched).

    The excerpt is capped at :data:`_TSC_EXCERPT_MAX_CHARS` to keep the DB
    ``error_text`` column bounded.
    """
    if not stderr:
        return None
    header_idx = stderr.find(_TSC_COMPILE_FAILED_HEADER)
    if header_idx < 0:
        return None
    # Start reading past the header line (skip to next newline).
    body_start = stderr.find("\n", header_idx + len(_TSC_COMPILE_FAILED_HEADER))
    if body_start < 0:
        # Header with no newline at all — take whatever follows the marker.
        body = stderr[header_idx + len(_TSC_COMPILE_FAILED_HEADER) :]
    else:
        body_start += 1  # move past the newline
        footer_idx = stderr.find(_TSC_COMPILE_FAILED_FOOTER, body_start)
        if footer_idx >= 0:
            body = stderr[body_start:footer_idx]
        else:
            # Truncated — take everything we have and soldier on.
            body = stderr[body_start:]
    body = body.strip()
    if not body:
        return None
    if len(body) > _TSC_EXCERPT_MAX_CHARS:
        body = body[:_TSC_EXCERPT_MAX_CHARS] + "\n... (truncated)"
    return body


def extract_token_usage(meta: dict[str, Any]) -> TokenUsage:
    """Extract a :class:`TokenUsage` from a runner ``meta`` dict.

    The runner may emit token counts at several locations depending on how
    the container exited:

    1. Flat per-direction fields (``input_tokens`` / ``output_tokens`` /
       ``cache_creation_input_tokens`` / ``cache_read_input_tokens``) — the
       canonical shape. Populated by both the ``result``-event and the
       scheduled-fallback paths in the agent-runner.
    2. A nested ``usage`` dict — used by some non-scheduled callers (e.g.
       OpenAI adapter summaries). Same four keys.
    3. ``total_tokens`` / ``tokens_used`` scalars — legacy aggregate.

    The companion ``tokens_source`` field on ``meta`` (set by the agent-runner
    post-2026-04 fix) records which path produced the counts:
        * ``"result_event"`` — SDK's final aggregate (trusted)
        * ``"per_turn_sum"`` — summed from per-turn assistant message deltas
          (stream ended without a result event, e.g. max-messages ceiling)
        * ``"unavailable"`` — both sources were empty; the returned zeros
          are a *reporting gap*, not a genuinely zero attempt.

    Callers that care about distinguishing "truly zero" from "lost the
    counter" should read ``meta.get("tokens_source")``. This function
    preserves the zero-TokenUsage return contract so existing call sites
    (which sum unconditionally) keep working.
    """
    try:
        if "input_tokens" in meta and "output_tokens" in meta:
            cache_creation = int(meta.get("cache_creation_input_tokens", 0))
            cache_read = _cached_input_tokens(meta)
            return TokenUsage(
                input_tokens=_normalized_input_tokens(meta, cache_read, cache_creation),
                output_tokens=int(meta["output_tokens"]),
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
            )
        usage = meta.get("usage")
        if isinstance(usage, dict):
            cache_creation = int(usage.get("cache_creation_input_tokens", 0))
            cache_read = _cached_input_tokens(usage)
            return TokenUsage(
                input_tokens=_normalized_input_tokens(usage, cache_read, cache_creation),
                output_tokens=int(usage.get("output_tokens", 0)),
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
            )
        if "tokens_used" in meta:
            return TokenUsage(output_tokens=int(meta["tokens_used"]))
        if "total_tokens" in meta:
            return TokenUsage(output_tokens=int(meta["total_tokens"]))
    except (TypeError, ValueError):
        pass
    return TokenUsage()


STRICT_RUNNER_STATUSES = {"success", "error", "recovered_from_session"}


def _validate_strict_runner_envelope(parsed: dict[str, Any], *, key: str) -> None:
    if key not in parsed:
        raise ValueError(f"runtime output envelope is missing key={key!r}")
    if "tool_trace" not in parsed or not isinstance(parsed.get("tool_trace"), list):
        raise ValueError("runtime output envelope must include tool_trace as a list")
    meta = parsed.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("runtime output envelope must include meta as an object")

    status = meta.get("status")
    if not isinstance(status, str) or status.strip().lower() not in STRICT_RUNNER_STATUSES:
        allowed = ", ".join(sorted(STRICT_RUNNER_STATUSES))
        raise ValueError(f"runtime output envelope has invalid meta.status={status!r}; expected one of: {allowed}")

    missing: list[str] = []
    for field in ("generation", "agent_id", "task_id"):
        value = meta.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field)
    if missing:
        raise ValueError("runtime output envelope meta missing required field(s): " + ", ".join(missing))


def build_runner_result(parsed: dict[str, Any], *, key: str) -> dict[str, Any]:
    value = parsed[key]
    output = ""
    if value is None:
        output = ""
    elif isinstance(value, str):
        output = value
    else:
        output = json.dumps(value, ensure_ascii=True)
    tool_trace_raw = parsed.get("tool_trace")
    tool_trace = tool_trace_raw if isinstance(tool_trace_raw, list) else []
    runtime_meta_raw = parsed.get("meta")
    runtime_meta = runtime_meta_raw if isinstance(runtime_meta_raw, dict) else {}
    token_usage = extract_token_usage(runtime_meta)
    return {"output": output, "tool_trace": tool_trace, "runtime_meta": runtime_meta, "token_usage": token_usage}


def is_silent_agent_failure(result: dict[str, Any]) -> bool:
    """Detect the "silent agent-runner exit" pattern.

    The container returns an outer JSON object with status='success' but zero
    observable activity: empty model_output, empty tool_trace, and zero tokens
    across all four token fields. This is the fingerprint of the bug where the
    claude-agent-sdk query loop exits without emitting any messages (e.g., auth
    failure at startup inside the container, or a race where the SDK stream
    closes before any event arrives). The host layer classifies these as
    success because the subprocess exited cleanly and emitted a parseable JSON
    envelope -- but the envelope is empty, so downstream evaluators get nothing
    to score.

    Args:
        result: The dict returned by ``build_runner_result`` (keys: ``output``,
            ``tool_trace``, ``runtime_meta``, ``token_usage``).

    Returns:
        True if the pattern matches and the caller should reclassify as an error.
    """
    meta = result.get("runtime_meta") or {}
    status = str(meta.get("status") or "").lower()
    # Only intervene on runs the runner claimed succeeded. If status is already
    # 'error' or one of our sentinel values, let the existing handler deal with it.
    if status not in {"", "success", "ok"}:
        return False

    output = result.get("output") or ""
    if isinstance(output, str) and output.strip():
        return False

    tool_trace = result.get("tool_trace") or []
    if isinstance(tool_trace, list) and len(tool_trace) > 0:
        return False

    token_usage = result.get("token_usage")
    if token_usage is not None:
        total = getattr(token_usage, "total", None)
        if total is None:
            total = (
                getattr(token_usage, "input_tokens", 0)
                + getattr(token_usage, "output_tokens", 0)
                + getattr(token_usage, "cache_creation_input_tokens", 0)
                + getattr(token_usage, "cache_read_input_tokens", 0)
            )
        if total and total > 0:
            return False

    # All signals point to a silent exit — intervene.
    return True


def mark_silent_failure(result: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a silent-failure result so its meta explicitly flags the failure.

    Does not mutate the caller's input; returns a new dict. The reclassification
    adds ``status='silent_failure'`` and ``error`` to ``runtime_meta`` and
    leaves every other field untouched (preserving diagnostic info like
    ``native_session_memory`` for post-mortems).
    """
    new_meta = dict(result.get("runtime_meta") or {})
    new_meta["status"] = SILENT_FAILURE_STATUS
    new_meta.setdefault("error", SILENT_FAILURE_MESSAGE)
    new_result = dict(result)
    new_result["runtime_meta"] = new_meta
    return new_result


# Status emitted into ``runtime_meta`` when the evaluator (or a guard invoked
# during evaluation / setup) raises a hard failure instead of returning a
# structured result. Writing an explicit status — instead of dropping
# ``runtime_meta`` to ``{}`` — lets downstream analytics distinguish this
# failure mode from a successful attempt and from a silent agent failure.
ERROR_STATUS = "error"


def build_error_runtime_meta(
    exc: BaseException,
    *,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a runtime_meta dict for an attempt that failed during eval/setup.

    Prior to this helper, the engine's ``_eval_stage`` exception handler wrote
    ``runtime_meta = {}`` whenever ``evaluator.evaluate`` raised.  That included
    the polyglot harness's pretask path-traversal guards (``_validate_safe_path``
    / ``_safe_write``) which raise ``ValueError`` before the evaluator can
    populate a structured ``eval_result``.  The resulting attempt row landed
    with a ``runtime_meta_json`` that had no ``status`` field — indistinguishable
    from a genuine silent failure.

    This helper overlays an ``error`` status and error message onto any
    ``runtime_meta`` that was already harvested by the container runner, so
    the failure mode is always discoverable in the DB.

    Parameters
    ----------
    exc:
        The exception whose ``str()`` becomes the ``error`` field.  The
        exception type name is also recorded as ``error_type`` to help
        analytics distinguish ``ValueError`` (guard rejection) from other
        failures.
    base:
        Optional starting ``runtime_meta`` dict — typically taken from
        ``run_result.runtime_meta`` when the container ran successfully but
        the evaluator raised.  The returned dict is a shallow copy of ``base``
        with the status / error fields overlaid (existing ``status`` keys are
        overwritten because the eval failure is the new terminal state).

    Returns
    -------
    A new dict; never mutates ``base``.
    """
    out: dict[str, Any] = dict(base) if isinstance(base, dict) else {}
    out["status"] = ERROR_STATUS
    out["error"] = str(exc)
    # Preserve the exception's type so downstream analytics can bucket
    # guard-rejection failures (ValueError) separately from e.g. OSError
    # or RuntimeError.  Use ``setdefault`` so an upstream caller that
    # pre-populated ``error_type`` wins.
    out.setdefault("error_type", type(exc).__name__)
    registry_error = find_container_registry_error(exc)
    if registry_error is not None:
        out["error_origin"] = "container_registry"
        out["registry_failure_reason"] = registry_error.reason
        out["registry_failure_retryable"] = registry_error.retryable
        if registry_error.image:
            out["registry_image"] = registry_error.image
    retry_meta = getattr(exc, "runtime_retry_meta", None)
    if isinstance(retry_meta, dict):
        out.update(retry_meta)
    return out


def parse_runner_stdout(
    stdout_text: str,
    *,
    key: str,
    strict: bool = False,
) -> dict[str, Any]:
    text = (stdout_text or "").strip()
    if not text:
        if strict:
            raise ValueError("runtime stdout is empty; expected JSON output envelope")
        return {"output": "", "tool_trace": [], "runtime_meta": {}, "token_usage": TokenUsage()}

    validation_error: ValueError | None = None

    def valid_candidate(parsed: Any) -> dict[str, Any] | None:
        nonlocal validation_error
        if parsed is None or not isinstance(parsed, dict) or key not in parsed:
            return None
        if strict:
            try:
                _validate_strict_runner_envelope(parsed, key=key)
            except ValueError as exc:
                validation_error = exc
                return None
        return parsed

    parsed = valid_candidate(try_parse_json(text))
    if parsed is not None:
        result = build_runner_result(parsed, key=key)
        if is_silent_agent_failure(result):
            return mark_silent_failure(result)
        return result

    last_match = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            parsed = valid_candidate(try_parse_json(line))
            if parsed is not None:
                last_match = parsed
    if last_match is not None:
        result = build_runner_result(last_match, key=key)
        if is_silent_agent_failure(result):
            return mark_silent_failure(result)
        return result

    if strict:
        if validation_error is not None:
            raise validation_error
        raise ValueError(f"runtime output is missing parseable envelope key={key!r}; stdout tail={text[-120:]!r}")
    log.warning("stdout has no parseable JSON, using raw text: %.80s ...", text)
    return {"output": text, "tool_trace": [], "runtime_meta": {}, "token_usage": TokenUsage()}

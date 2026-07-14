"""Task-execution retry classification and token/meta accounting helpers.

Extracted from :mod:`ksi.orchestrator.engine` so the forum-phase machinery
in :mod:`ksi.orchestrator.forum_runtime` and the main task-execution path in
``engine`` can share these helpers without an import cycle.

These are general task-retry utilities, not forum-specific:
``_is_retryable_task_error`` decides whether a runner/provider failure is worth
retrying, ``_runtime_retry_meta`` summarises retry attempts for persistence, and
``_accumulate_failed_attempt_tokens`` sums the billable cost of failed retries.
``engine`` re-imports these names, so existing call sites and
``from ksi.orchestrator.engine import _is_retryable_task_error`` keep working.

This module imports only from ``..errors``, ``..runtime.normalize`` and
``..tokens`` -- never from ``engine`` -- so there is no import cycle.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Callable, TypeVar

from ..errors import (
    AuthenticationFailure,
    exception_chain,
    find_container_registry_error,
    is_auth_error,
    non_retryable_exit_code_markers,
    non_retryable_markers,
    transient_markers,
    upstream_provider_transient_markers,
)
from ..runtime.normalize import SilentAgentRuntimeError, extract_token_usage
from ..tokens import TokenUsage

log = logging.getLogger(__name__)

# Retryable-error markers come from the single source of truth shared with the
# TypeScript agent-runner: ``runtime_runner/shared/retryable_markers.json``
# (loaded via ``ksi.errors.load_retryable_markers``). Do not hardcode the
# substrings here — a reworded provider/SDK phrase must be changed in the JSON
# (and mirrored in the agent-runner's emitted text) so Python classification
# and the runner's error envelopes stay in lockstep.
#
# Category semantics:
#   * non_retryable            — deterministic refusals/parse failures
#                                ("invalid prompt", "usage policy", "no patch").
#   * non_retryable_exit_codes — container hard-fail exit codes
#                                (137 OOM/SIGKILL, 139 SIGSEGV, 126/127 exec).
#   * upstream_provider_transient — provider-side 5xx / rate-limit / undici
#                                fetch failures; retryable even with no
#                                execution telemetry (tokens_source=unavailable).
#   * transient (= upstream + network + SDK stream-race phrases) — the full
#                                set matched for plain exceptions; the SDK
#                                stream-race phrases (e.g. "silent agent-runner
#                                failure") are emitted verbatim by
#                                runtime_runner/agent-runner/src/index.ts via
#                                runtime_runner/agent-runner/src/retryable_markers.ts.
_NON_RETRYABLE_TASK_ERROR_MARKERS = non_retryable_markers()
_NON_RETRYABLE_EXIT_CODES = non_retryable_exit_code_markers()
_UPSTREAM_PROVIDER_TRANSIENT_MARKERS = upstream_provider_transient_markers()
_TRANSIENT_TASK_ERROR_MARKERS = transient_markers()


def _is_retryable_task_error(exc: Exception) -> bool:
    """Return True only for transient runner/provider failures.

    For ``SilentAgentRuntimeError`` the decision is made by inspecting
    ``runtime_meta`` instead of (or in addition to) the exception message.
    The string-based transient markers match phrases like "silent
    agent-runner failure" and "sdk emitted an empty result event", but those
    same phrases surface in both:
      1. a real claude-agent-sdk stream race / mid-stream crash where the
         CLI subprocess actually ran and produced telemetry (recovered
         session log, or per-turn token deltas) — transient, retry helps
      2. a container auth/startup failure where the CLI never executed —
         deterministic, retrying burns LLM spend

    The distinguishing signal is the ``tokens_source`` field set by the
    in-container agent-runner (``runtime_runner/agent-runner/src/index.ts``):

      * ``session_recovery`` — emitted by ``maybeRecoverFromEmptyScheduledOutcome``
        when ``recoverFromSessionLog()`` reconstructed usable output from the
        on-disk ``/home/node/.claude/projects/*.jsonl`` log. The CLI ran to
        completion. Retryable.
      * ``per_turn_sum`` — emitted on the partial-execution paths in
        ``index.ts`` (grep ``tokens_source: 'per_turn_sum'``) when the SDK
        delivered per-turn usage deltas but no terminal result event, or
        the scheduled task ended with pending tool calls. The CLI ran at
        least partially (often thousands of cache-read tokens, real tool
        calls in the trace) before something went wrong. Retryable: the
        common cause is a forum/ARC tool-loop ordering glitch that doesn't
        reproduce.
      * ``unavailable`` — no usable telemetry. This is the silent-exit
        branch (``maybeRecoverFromEmptyScheduledOutcome`` in index.ts) where
        session-log recovery also failed. Could be
        auth/startup failure, container OOM, MCP server hang, or a true
        silent SDK drain. We refuse to retry to avoid burning spend on
        deterministic failures.

    The ``status == 'silent_failure'`` path is always non-retryable: it is
    raised explicitly when the runner detects 0-tokens / 0-tool-calls / no
    output, which by construction maps to ``tokens_source=unavailable``.

    The retry gate accepts ``session_recovery`` / ``recovered_from_session``
    and the ``per_turn_sum`` case (proven transient), but not ``unavailable``.
    """
    # Registry provenance and retry policy are separate. The nearest typed
    # registry error wins even when an adapter wrapped it in a generic exception.
    registry_error = find_container_registry_error(exc)
    if registry_error is not None:
        return registry_error.retryable

    if any(isinstance(item, AuthenticationFailure) for item in exception_chain(exc)):
        return False

    if isinstance(exc, SilentAgentRuntimeError):
        meta = exc.runtime_meta or {}
        tokens_source = str(meta.get("tokens_source") or "").strip().lower()
        status = str(meta.get("status") or "").strip().lower()
        message = str(exc or "").strip().lower()
        meta_error = str(meta.get("error") or "").strip().lower()

        # Hard deny: explicit silent-failure status (auth/startup hard-fail
        # path) — never worth retrying. ``tokens_source`` for this branch
        # is always ``unavailable`` by construction.
        if status == "silent_failure":
            return False

        # Hard deny: deterministic non-retryable markers in either the
        # exception message or any ``error`` text the runtime stashed in
        # runtime_meta. Examples: "invalid prompt", "usage policy refused",
        # "parse_error:empty_model_output", "no patch produced".
        for haystack in (message, meta_error):
            if not haystack:
                continue
            if any(marker in haystack for marker in _NON_RETRYABLE_TASK_ERROR_MARKERS):
                return False
            if any(code in haystack for code in _NON_RETRYABLE_EXIT_CODES):
                return False
            # Auth/startup signature emitted by the in-container runner's
            # silent-diagnostic envelope. Independent of SILENT_FAILURE_STATUS
            # because some non-recovery error envelopes carry the same text.
            if "auth/startup failure" in haystack:
                return False

        # Allow: positive evidence the CLI actually executed.
        if tokens_source in ("session_recovery", "per_turn_sum"):
            return True
        if status == "recovered_from_session":
            return True

        # Allow: explicit upstream/provider transient signature in the
        # error envelope (Anthropic 503, gateway timeout, rate limit,
        # upstream connect reset, etc.). These are provider-side blips,
        # not container/auth/startup failures, so retry is appropriate
        # even when the SDK never reached the model and tokens_source is
        # "unavailable". Without this gate, a single upstream connection
        # reset silently drops one agent's contribution from a forum or
        # cross-task round (observed on agent-34 in a 50-task ARC2 run).
        # Distinct from SDK-stream-race markers above, which require
        # execution evidence (per_turn_sum / session_recovery) to retry.
        for haystack in (message, meta_error):
            if not haystack:
                continue
            if any(marker in haystack for marker in _UPSTREAM_PROVIDER_TRANSIENT_MARKERS):
                return True

        # Default for SilentAgentRuntimeError: tokens_source is
        # ``unavailable`` or absent, no recovery signal — refuse to retry.
        return False

    message = str(exc or "").strip().lower()
    if not message:
        return False
    if any(marker in message for marker in _NON_RETRYABLE_TASK_ERROR_MARKERS):
        return False
    if any(code in message for code in _NON_RETRYABLE_EXIT_CODES):
        return False
    return any(marker in message for marker in _TRANSIENT_TASK_ERROR_MARKERS)


# --- Host-side distillation LLM retry --------------------------------------
#
# The forum and task-execution paths retry transient failures via
# ``max_task_retries``; the host-side distill LLM call had *no* retry, so a
# transient host->provider network blip (e.g. a DNS ``getaddrinfo EAI_AGAIN``)
# made every ``distill_one_task`` call fail once and give up, zeroing a whole
# generation's distillation with only a WARNING per task. These helpers give
# distillation its own, deliberately more generous, bounded retry so a genuine
# transient (seconds to ~1-2 min) is ridden out rather than reaching failure.

# Retries default higher than ``max_task_retries`` because a zeroed distill
# generation wastes ALL that generation's attempt compute (the knowledge it
# would have produced is gone), so paying a little extra wall-clock to ride out
# a blip is strongly worth it. Env-tunable via ``KSI_DISTILL_MAX_RETRIES``.
_DISTILL_MAX_RETRIES_DEFAULT = 6
_DISTILL_BACKOFF_BASE_SEC = 1.0
_DISTILL_BACKOFF_CAP_SEC = 30.0

# Exception *type* names that signal a transient host<->provider failure whose
# ``str()`` may be too terse to match the substring markers. The Anthropic
# SDK's ``APIConnectionError`` stringifies to just ``"Connection error."`` —
# the real signature (``getaddrinfo EAI_AGAIN``) lives in ``__cause__`` — so
# the type name is the reliable signal. Lowercased substring match.
_DISTILL_RETRYABLE_TYPE_MARKERS: tuple[str, ...] = (
    "apiconnectionerror",
    "apitimeouterror",
    "connectionerror",
    "connecttimeout",
    "readtimeout",
    "timeouterror",
)

# Auth substrings safe to match anywhere in the (type-name + message) haystack.
_AUTH_HAYSTACK_MARKERS: tuple[str, ...] = (
    "authenticationerror",
    "authentication_error",
    "invalid_api_key",
    "invalid api key",
)

_T = TypeVar("_T")


def _exc_chain_haystack(exc: BaseException) -> str:
    """Lowercased ``type-name + str`` for ``exc`` and its ``__cause__`` chain.

    Walks ``__cause__``/``__context__`` (bounded, cycle-safe) so a transient
    signature buried in the chained cause (the ``EAI_AGAIN`` case) is
    matched even when the top exception's ``str()`` is terse.
    """
    parts = [part for item in exception_chain(exc) for part in (type(item).__name__, str(item))]
    return " ".join(parts).lower()


def is_retryable_distill_error(exc: BaseException) -> bool:
    """True for transient host-side distill LLM failures worth retrying.

    Distinct from :func:`_is_retryable_task_error` (which classifies container
    task-runner error *envelopes* by their message text): the host-side distill
    call raises provider-SDK exceptions directly, whose ``str()`` is often just
    ``"Connection error."`` while the transient cause lives in ``__cause__``.
    This inspects the whole exception chain plus the exception type names, and
    never retries an auth failure (deterministic — retrying burns spend).
    """
    if isinstance(exc, AuthenticationFailure):
        return False
    haystack = _exc_chain_haystack(exc)
    # Auth signatures anywhere in the chain are deterministic — never retry.
    if is_auth_error(exc) or any(marker in haystack for marker in _AUTH_HAYSTACK_MARKERS):
        return False
    # Deterministic refusals/parse failures — never retry.
    if any(marker in haystack for marker in _NON_RETRYABLE_TASK_ERROR_MARKERS):
        return False
    # Known transient substrings (provider 5xx, fetch failed, eai_again,
    # connection reset, timeout, ...) anywhere in the chain.
    if any(marker in haystack for marker in _TRANSIENT_TASK_ERROR_MARKERS):
        return True
    # Terse SDK connection/timeout exceptions: rely on the type name.
    return any(marker in haystack for marker in _DISTILL_RETRYABLE_TYPE_MARKERS)


def _distill_max_retries() -> int:
    """Resolve the distill retry budget from ``KSI_DISTILL_MAX_RETRIES``.

    0 disables retry (single attempt); positive values are the number of
    retries *after* the first attempt. Unparseable/negative -> default.
    """
    raw = os.environ.get("KSI_DISTILL_MAX_RETRIES")
    if raw is None or raw.strip() == "":
        return _DISTILL_MAX_RETRIES_DEFAULT
    try:
        return max(0, int(raw))
    except ValueError:
        return _DISTILL_MAX_RETRIES_DEFAULT


def _sleep(delay: float) -> None:
    """Indirection over ``time.sleep`` so tests can stub the backoff wait."""
    time.sleep(delay)


def run_with_distill_retry(fn: Callable[[], _T], *, generation: int, phase: str) -> _T:
    """Run ``fn`` with bounded exponential backoff on transient distill errors.

    Retries only when :func:`is_retryable_distill_error` is True; a
    deterministic failure (auth, refusal, bug) raises on the first attempt so
    no spend is wasted. Backoff is jittered exponential capped at
    ``_DISTILL_BACKOFF_CAP_SEC``. On exhaustion the last exception propagates,
    so the caller's existing failure handling (WARNING + ``failures`` counter,
    plus the generation-level escalation) still fires.
    """
    retries = _distill_max_retries()
    last_exc: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below when terminal
            last_exc = exc
            if attempt >= retries or not is_retryable_distill_error(exc):
                raise
            delay = min(_DISTILL_BACKOFF_CAP_SEC, _DISTILL_BACKOFF_BASE_SEC * 2**attempt) * (0.5 + random.random())
            log.warning(
                "[ENGINE] transient distill LLM failure, retrying phase=%s generation=%s attempt=%d/%d: %s",
                phase,
                generation,
                attempt + 1,
                retries + 1,
                exc,
            )
            _sleep(delay)
    # Unreachable: the loop either returns or raises. Re-raise defensively.
    assert last_exc is not None
    raise last_exc


def _accumulate_failed_attempt_tokens(
    failed_runtime_metas: list[dict[str, Any]] | None,
    *,
    log_dropped: bool = True,
) -> TokenUsage:
    """Sum :class:`TokenUsage` across the runtime_meta dicts from failed retries.

    Failed attempts that had ``tokens_source=per_turn_sum`` (SDK race after
    real per-turn deltas, scheduled-task end without a result event) consumed
    real billable tokens — often 30k+ cache reads on Haiku ARC runs — even
    though the attempt raised ``SilentAgentRuntimeError`` and triggered a
    retry. Without this accumulation the success attempt's ``token_usage``
    is the only thing recorded, and the failed attempts' cost vanishes from
    every accounting surface (``token_phases`` rows, ``agent.token_usage``
    counters, the campaign totals JSON).

    Returns a zero ``TokenUsage`` when ``failed_runtime_metas`` is empty.

    A ``failed_meta`` that raises during extraction (beyond the
    ``TypeError``/``ValueError`` ``extract_token_usage`` already swallows) is
    dropped from the sum rather than propagated, so accounting can never crash
    a hot retry path -- but the drop is logged with a count so a
    systematic extraction failure shows up somewhere instead of silently
    undercounting cost with no trace anywhere.

    ``log_dropped=False`` suppresses the WARNING for this call only (the sum
    itself is unaffected). Several call flows compute this same sum twice
    from the same ``failed_runtime_metas`` list -- once indirectly via
    ``_runtime_retry_meta`` and once again directly -- to populate two
    different fields; passing ``log_dropped=False`` on the redundant call
    keeps a single real drop event to a single WARNING line.
    """
    total = TokenUsage()
    if not failed_runtime_metas:
        return total
    dropped = 0
    for failed_meta in failed_runtime_metas:
        if not isinstance(failed_meta, dict):
            continue
        try:
            total = total + extract_token_usage(failed_meta)
        except Exception:
            dropped += 1
            continue
    if dropped and log_dropped:
        log.warning(
            "_accumulate_failed_attempt_tokens: dropped %d/%d failed-attempt "
            "runtime_meta dict(s) that raised during token extraction; their "
            "token cost is undercounted in this task's accounting",
            dropped,
            len(failed_runtime_metas),
        )
    return total


def _runtime_retry_meta(
    attempt_errors: list[dict[str, Any]],
    *,
    terminal_failure: bool,
    failed_runtime_metas: list[dict[str, Any]] | None = None,
    log_dropped_tokens: bool = True,
) -> dict[str, Any]:
    """Summarise retry attempts for persistence in ``runtime_meta``.

    ``failed_runtime_metas``, when provided, carries the ``runtime_meta`` dicts
    from ``SilentAgentRuntimeError`` instances raised on failed attempts. The
    stated purpose of ``SilentAgentRuntimeError`` (see
    :mod:`ksi.runtime.normalize`) is to propagate forensic data —
    specifically ``native_session_memory`` / ``raw_native_session_memory`` —
    across the raise boundary. When a task succeeds on retry, the succeeding
    attempt's ``run_result.runtime_meta`` is used verbatim, so without
    this extraction the ~134 KB session transcripts from the failed attempts
    are silently dropped. We namespace them under ``attempt_{n}_*`` keys so
    the attempt row still carries the transcript proving the CLI subprocess
    actually ran even when the final attempt succeeded.

    The summary also records ``retry_failed_attempts_token_usage`` — the
    aggregated billable cost of the failed attempts — so analytics can
    distinguish a first-try success from one that needed retries by inspecting
    only ``runtime_meta``, without re-walking the per-attempt blobs.

    ``log_dropped_tokens=False`` suppresses the drop-count WARNING from the
    internal ``_accumulate_failed_attempt_tokens`` call. Callers that also
    call ``_accumulate_failed_attempt_tokens(failed_runtime_metas)`` directly
    afterward (to fold the same sum into ``token_usage``) should pass
    ``False`` here so a single real extraction failure logs once, not twice.
    """

    if not attempt_errors:
        return {}
    retry_attempts = len(attempt_errors)
    if terminal_failure:
        retry_attempts = max(0, retry_attempts - 1)
    meta: dict[str, Any] = {
        "retry_attempts": retry_attempts,
        "runtime_attempt_errors": attempt_errors,
    }
    failed_token_total = _accumulate_failed_attempt_tokens(failed_runtime_metas, log_dropped=log_dropped_tokens)
    if failed_token_total.total > 0:
        meta["retry_failed_attempts_token_usage"] = failed_token_total.to_dict()
    if failed_runtime_metas:
        for idx, failed_meta in enumerate(failed_runtime_metas, start=1):
            if not isinstance(failed_meta, dict):
                continue
            nsm = failed_meta.get("native_session_memory") or failed_meta.get("raw_native_session_memory")
            if nsm:
                meta[f"attempt_{idx}_native_session_memory"] = nsm
    return meta


def _cap_native_memory_fields(meta: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``meta`` with ``native_session_memory`` / raw variant
    truncated to the ``KSI_NATIVE_MEMORY_MAX_CHARS`` cap (default 240000).

    Called on the silent-failure / recovered-from-session persistence path so
    the attempt row in SQLite preserves the session transcript that proves
    the container actually did work, without bloating ``runtime_meta_json``
    beyond the configured cap. A non-positive cap disables capture entirely
    (matching the collector semantics documented in CLAUDE.md) — in that case
    both fields are dropped.
    """
    if not isinstance(meta, dict):
        return {}
    try:
        cap_raw = os.environ.get("KSI_NATIVE_MEMORY_MAX_CHARS", "").strip()
        cap = int(cap_raw) if cap_raw else 240_000
    except ValueError:
        cap = 240_000
    out = dict(meta)
    for key in ("native_session_memory", "raw_native_session_memory"):
        val = out.get(key)
        if not isinstance(val, str):
            continue
        if cap <= 0:
            out.pop(key, None)
            continue
        if len(val) > cap:
            # Tail-slice: the final turns carry the most relevant evidence
            # (last assistant message, final tool outputs).
            out[key] = val[-cap:]
    return out

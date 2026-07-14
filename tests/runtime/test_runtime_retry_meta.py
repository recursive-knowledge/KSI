"""Unit tests for ``ksi.orchestrator.engine._runtime_retry_meta``.

Context
-------
PR #485 added SDK-stream-race retryability to ``_run_agent_stage``, but the
resulting retry-meta builder (``_runtime_retry_meta``) only recorded
``retry_attempts`` + ``runtime_attempt_errors`` — it discarded the
``runtime_meta`` dicts attached to ``SilentAgentRuntimeError`` instances
raised on failed attempts. Those dicts carry ``native_session_memory`` /
``raw_native_session_memory`` — the ~134 KB on-host session transcripts that
prove the CLI subprocess actually ran. When the task then succeeded on
retry, the succeeding attempt's ``run_result.runtime_meta`` was used
verbatim and the failed attempts' transcripts were silently dropped.

This follow-up (PR A from the PR #485 review) augments ``_runtime_retry_meta``
to accept a ``failed_runtime_metas`` list and fan out
``native_session_memory`` / ``raw_native_session_memory`` under
``attempt_{n}_native_session_memory`` keys so the succeeding attempt row
still carries the failed attempts' forensic evidence.

These tests pin the pure-function behaviour. Engine-level wiring is covered
by the existing silent-failure / retry tests.
"""

from __future__ import annotations

import pytest

from ksi.orchestrator.engine import _runtime_retry_meta


def _attempt_error(idx: int, *, err: str = "SDK query loop drained") -> dict:
    return {
        "attempt": idx,
        "max_attempts": 3,
        "error_type": "SilentAgentRuntimeError",
        "error": err,
    }


# ---------------------------------------------------------------------------
# Baseline: empty attempt_errors still yields empty dict
# ---------------------------------------------------------------------------


def test_empty_attempt_errors_returns_empty_dict() -> None:
    assert _runtime_retry_meta([], terminal_failure=False) == {}
    assert _runtime_retry_meta([], terminal_failure=True) == {}
    assert _runtime_retry_meta([], terminal_failure=False, failed_runtime_metas=[{"native_session_memory": "x"}]) == {}


# ---------------------------------------------------------------------------
# retry_attempts arithmetic (retained behaviour from PR #485)
# ---------------------------------------------------------------------------


def test_retry_attempts_nonterminal_counts_all_failed_attempts() -> None:
    errs = [_attempt_error(1), _attempt_error(2)]
    out = _runtime_retry_meta(errs, terminal_failure=False)
    assert out["retry_attempts"] == 2
    assert out["runtime_attempt_errors"] == errs


def test_retry_attempts_terminal_decrements_to_exclude_final_failure() -> None:
    # Terminal-failure path: the final attempt is the one that propagated, so
    # the "retry" count is (attempts - 1).
    errs = [_attempt_error(1), _attempt_error(2), _attempt_error(3)]
    out = _runtime_retry_meta(errs, terminal_failure=True)
    assert out["retry_attempts"] == 2
    assert out["runtime_attempt_errors"] == errs


def test_retry_attempts_terminal_single_attempt_clamps_to_zero() -> None:
    # Sanity: single failure with terminal_failure=True shouldn't produce -1.
    out = _runtime_retry_meta([_attempt_error(1)], terminal_failure=True)
    assert out["retry_attempts"] == 0


# ---------------------------------------------------------------------------
# native_session_memory preservation (new behaviour)
# ---------------------------------------------------------------------------


def test_single_failed_meta_with_native_session_memory_is_preserved() -> None:
    errs = [_attempt_error(1)]
    failed = [{"native_session_memory": "SESSION_TRANSCRIPT_ATTEMPT_1"}]
    out = _runtime_retry_meta(errs, terminal_failure=False, failed_runtime_metas=failed)
    assert out["retry_attempts"] == 1
    assert out["attempt_1_native_session_memory"] == "SESSION_TRANSCRIPT_ATTEMPT_1"


def test_multiple_failed_metas_each_get_namespaced_key() -> None:
    errs = [_attempt_error(1), _attempt_error(2)]
    failed = [
        {"native_session_memory": "TRANSCRIPT_1"},
        {"native_session_memory": "TRANSCRIPT_2"},
    ]
    out = _runtime_retry_meta(errs, terminal_failure=False, failed_runtime_metas=failed)
    assert out["attempt_1_native_session_memory"] == "TRANSCRIPT_1"
    assert out["attempt_2_native_session_memory"] == "TRANSCRIPT_2"


def test_raw_native_session_memory_fallback_is_also_extracted() -> None:
    # Some call sites populate ``raw_native_session_memory`` (pre-cap) only;
    # _cap_native_memory_fields or its callers may fill the canonical key
    # later. _runtime_retry_meta should still surface the raw variant.
    errs = [_attempt_error(1)]
    failed = [{"raw_native_session_memory": "RAW_TRANSCRIPT"}]
    out = _runtime_retry_meta(errs, terminal_failure=False, failed_runtime_metas=failed)
    assert out["attempt_1_native_session_memory"] == "RAW_TRANSCRIPT"


def test_native_key_preferred_over_raw_when_both_present() -> None:
    # When both are set (canonical post-cap path), the primary key wins; the
    # test avoids relying on key-insertion order for correctness.
    errs = [_attempt_error(1)]
    failed = [
        {
            "native_session_memory": "CANONICAL",
            "raw_native_session_memory": "RAW_SHOULD_NOT_WIN",
        }
    ]
    out = _runtime_retry_meta(errs, terminal_failure=False, failed_runtime_metas=failed)
    assert out["attempt_1_native_session_memory"] == "CANONICAL"


# ---------------------------------------------------------------------------
# No-op paths
# ---------------------------------------------------------------------------


def test_empty_failed_runtime_metas_adds_no_namespaced_keys() -> None:
    errs = [_attempt_error(1), _attempt_error(2)]
    out = _runtime_retry_meta(errs, terminal_failure=False, failed_runtime_metas=[])
    assert "attempt_1_native_session_memory" not in out
    assert "attempt_2_native_session_memory" not in out
    # Core fields still present.
    assert set(out.keys()) == {"retry_attempts", "runtime_attempt_errors"}


def test_none_failed_runtime_metas_adds_no_namespaced_keys() -> None:
    errs = [_attempt_error(1)]
    out = _runtime_retry_meta(errs, terminal_failure=False, failed_runtime_metas=None)
    assert "attempt_1_native_session_memory" not in out


def test_failed_meta_without_native_session_memory_is_skipped() -> None:
    errs = [_attempt_error(1)]
    failed = [{"status": "error", "duration_ms": 123}]  # no transcript fields
    out = _runtime_retry_meta(errs, terminal_failure=False, failed_runtime_metas=failed)
    assert "attempt_1_native_session_memory" not in out
    # Unrelated meta keys are NOT propagated (only native_session_memory is).
    assert "status" not in out
    assert "duration_ms" not in out


def test_failed_meta_with_empty_string_native_session_memory_is_skipped() -> None:
    # ``nsm = failed.get(...) or failed.get(...)`` treats "" as falsy, which
    # is what we want — empty strings aren't useful forensic evidence and
    # would bloat the meta dict.
    errs = [_attempt_error(1)]
    failed = [{"native_session_memory": "", "raw_native_session_memory": ""}]
    out = _runtime_retry_meta(errs, terminal_failure=False, failed_runtime_metas=failed)
    assert "attempt_1_native_session_memory" not in out


def test_non_dict_entries_in_failed_runtime_metas_are_skipped() -> None:
    # Defensive: if a caller somehow passes a non-dict element (e.g. None),
    # don't crash — skip it and process the remaining dicts.
    errs = [_attempt_error(1), _attempt_error(2)]
    failed = [None, {"native_session_memory": "GOOD"}]  # type: ignore[list-item]
    out = _runtime_retry_meta(
        errs,
        terminal_failure=False,
        failed_runtime_metas=failed,  # type: ignore[arg-type]
    )
    # Index 2 because enumeration is 1-based over the full list.
    assert out["attempt_2_native_session_memory"] == "GOOD"


# ---------------------------------------------------------------------------
# Mixed: some attempts had transcripts, others didn't
# ---------------------------------------------------------------------------


def test_mixed_failed_metas_only_transcript_bearing_ones_surface() -> None:
    errs = [_attempt_error(1), _attempt_error(2), _attempt_error(3)]
    failed = [
        {"native_session_memory": "T1"},
        {"status": "error"},  # no transcript — e.g. non-silent failure
        {"raw_native_session_memory": "T3_RAW"},
    ]
    out = _runtime_retry_meta(errs, terminal_failure=False, failed_runtime_metas=failed)
    assert out["attempt_1_native_session_memory"] == "T1"
    assert "attempt_2_native_session_memory" not in out
    assert out["attempt_3_native_session_memory"] == "T3_RAW"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

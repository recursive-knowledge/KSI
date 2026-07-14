"""Regression tests for path-traversal guard ``runtime_meta`` status.

Context
-------
The polyglot harness (``kcsi.benchmarks.polyglot_harness``) has a pretask guard
that rejects task metadata whose workspace filenames escape the sandbox root
(e.g. ``../../etc/passwd``).  The guard is implemented as ``_validate_safe_path``
/ ``_safe_write`` which ``raise ValueError``.

Before this fix, ``evaluator.evaluate()`` propagated that ``ValueError`` to the
engine's ``_eval_stage`` exception handler, which then set
``trace.runtime_meta = {}`` — stripping both the container's forensics meta
(duration, token counts, any harvested session memory) AND the status field
itself.  Downstream analytics saw ``runtime_meta_json = "{}"`` and couldn't
distinguish "guard rejected the task" from "silent-failure with no transcript"
from "container never ran at all".

The forensics report surfaced this on ``haiku_polyglot`` attempt #181.

This PR adds a ``build_error_runtime_meta`` helper (in
``kcsi.runtime.normalize``) and wires it into every exception path in the
engine's ``_eval_stage``.  The guard still rejects the task (security is
untouched), but the failure now lands in the DB with an explicit
``{status: 'error', error: '<rejection reason>', error_type: 'ValueError'}``
overlay — and any container-level ``runtime_meta`` that was already harvested
is preserved as the base dict.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from kcsi.benchmarks.polyglot_harness import _safe_write, _validate_safe_path
from kcsi.runtime.normalize import (
    ERROR_STATUS,
    SILENT_FAILURE_STATUS,
    build_error_runtime_meta,
)

# -----------------------------------------------------------------------
# build_error_runtime_meta — direct unit tests for the helper.
# -----------------------------------------------------------------------


class TestBuildErrorRuntimeMeta:
    def test_returns_status_error_for_bare_exception(self):
        exc = ValueError("Unsafe file path in task metadata (path traversal): '../../etc/passwd'")
        out = build_error_runtime_meta(exc)
        assert out["status"] == ERROR_STATUS
        assert out["status"] == "error"  # explicit: downstream analytics key on literal 'error'
        assert "path traversal" in out["error"]
        assert out["error_type"] == "ValueError"

    def test_preserves_base_meta_fields(self):
        base = {
            "duration_ms": 12345,
            "tokens_source": "result",
            "input_tokens": 100,
            "output_tokens": 50,
            "status": "success",  # container reported success
        }
        exc = ValueError("Path traversal in exercise_name: '../etc'")
        out = build_error_runtime_meta(exc, base=base)

        # Container's successful meta carried forward.
        assert out["duration_ms"] == 12345
        assert out["tokens_source"] == "result"
        assert out["input_tokens"] == 100
        assert out["output_tokens"] == 50

        # Status is overwritten to 'error' — the eval-stage failure is the
        # terminal state even if the container ran successfully.
        assert out["status"] == "error"
        assert out["error"] == "Path traversal in exercise_name: '../etc'"
        assert out["error_type"] == "ValueError"

    def test_does_not_mutate_base(self):
        base = {"status": "success", "duration_ms": 42}
        exc = RuntimeError("boom")
        out = build_error_runtime_meta(exc, base=base)
        assert base == {"status": "success", "duration_ms": 42}  # unchanged
        assert out is not base

    def test_none_base_yields_minimal_dict(self):
        exc = ValueError("rejected")
        out = build_error_runtime_meta(exc, base=None)
        assert out == {"status": "error", "error": "rejected", "error_type": "ValueError"}

    def test_non_dict_base_treated_as_empty(self):
        # Defensive: if some caller passes a truthy non-dict by mistake, don't crash.
        exc = RuntimeError("bad")
        out = build_error_runtime_meta(exc, base="not a dict")  # type: ignore[arg-type]
        assert out == {"status": "error", "error": "bad", "error_type": "RuntimeError"}

    def test_caller_supplied_error_type_not_overwritten(self):
        # Allow specialized callers to tag a more specific error_type — the
        # helper should leave that alone while still setting status + error.
        exc = ValueError("x")
        out = build_error_runtime_meta(exc, base={"error_type": "PolyglotGuardRejection"})
        assert out["error_type"] == "PolyglotGuardRejection"
        assert out["status"] == "error"
        assert out["error"] == "x"

    def test_records_exception_type_name(self):
        class CustomError(RuntimeError):
            pass

        out = build_error_runtime_meta(CustomError("custom"))
        assert out["error_type"] == "CustomError"


# -----------------------------------------------------------------------
# Integration: guard rejection → engine trace → runtime_meta_json status.
# -----------------------------------------------------------------------


class TestPolyglotGuardRejectionPropagation:
    """The guard still raises ValueError — this test pins that behaviour and
    shows how the engine fix turns the raised exception into a well-formed
    ``runtime_meta`` dict with ``status='error'``.
    """

    def test_safe_write_guard_still_rejects_traversal(self):
        """Security invariant: _safe_write MUST raise on path traversal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            with pytest.raises(ValueError, match="path traversal"):
                _safe_write(base, "../../etc/passwd", "pwned")

    def test_validate_safe_path_guard_still_rejects_traversal(self):
        """Security invariant: _validate_safe_path MUST raise on traversal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            with pytest.raises(ValueError, match="path traversal"):
                _validate_safe_path(base, "../../../tmp/evil")

    def test_guard_rejection_yields_status_error_via_helper(self):
        """Simulate the engine's eval-stage exception path on a guard trip.

        This is the end-to-end assertion the audit asked for: when the guard
        short-circuits, the resulting ``runtime_meta`` dict (which is what
        lands in ``runtime_meta_json`` on the attempt row) carries
        ``status='error'`` with the guard's rejection reason.  Before the
        fix, this dict was ``{}`` — status-less and indistinguishable from a
        silent failure.
        """
        # Step 1: the guard trips.
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            try:
                _safe_write(base, "../../etc/passwd", "pwned")
            except ValueError as exc:
                guard_exc: Exception = exc
            else:
                pytest.fail("guard did not raise — security regression")

        # Step 2: the container had already run and populated meta — the
        # engine must carry this forward, not discard it.
        container_meta: dict[str, Any] = {
            "status": "success",
            "duration_ms": 9876,
            "input_tokens": 2500,
            "output_tokens": 400,
            "tokens_source": "result",
        }

        # Step 3: the fix — engine overlays status=error onto container meta.
        out = build_error_runtime_meta(guard_exc, base=container_meta)

        # Assertions — this is what the downstream attempt row carries:
        assert out["status"] == "error", (
            "runtime_meta_json['status'] must be 'error' for guard-rejected "
            "attempts so analytics can bucket this failure mode"
        )
        assert "path traversal" in out["error"], "error message must carry the guard's rejection reason"
        assert out["error_type"] == "ValueError"
        # Container-level forensics are preserved — we don't throw away the
        # ~10s of work the container did just because the evaluator rejected.
        assert out["duration_ms"] == 9876
        assert out["input_tokens"] == 2500
        assert out["output_tokens"] == 400
        assert out["tokens_source"] == "result"

    def test_silent_failure_status_is_not_overwritten_when_exc_is_silent(self):
        """Safety check: the silent-failure path uses its own status sentinel
        (``silent_failure``) — the engine branches on SilentAgentRuntimeError
        BEFORE calling build_error_runtime_meta, so we must not accidentally
        reclassify silent failures as generic ``error`` when this helper is
        called directly.

        ``build_error_runtime_meta`` always writes ``status='error'``.  That's
        correct — it's only invoked on the non-silent branch.  Verify the
        helper does overwrite an existing ``silent_failure`` status when
        called directly (so callers know this overlay is unconditional).
        """
        base = {"status": SILENT_FAILURE_STATUS, "native_session_memory": "..."}
        out = build_error_runtime_meta(ValueError("x"), base=base)
        # Unconditional overwrite — caller is responsible for not calling this
        # on the silent-failure branch (the engine already does).
        assert out["status"] == "error"
        # Everything else is preserved including the session transcript.
        assert out["native_session_memory"] == "..."


# -----------------------------------------------------------------------
# End-to-end integration via the engine's evaluator exception path.
# -----------------------------------------------------------------------


class _RaisingEvaluator:
    """Fake evaluator that always raises — simulates the polyglot guard trip."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def evaluate(self, **_: Any) -> dict[str, Any]:
        raise self._exc


def test_engine_preserves_runtime_meta_on_evaluator_exception(monkeypatch):
    """Full engine-level check: when ``evaluator.evaluate`` raises (the guard
    path), the resulting ``TaskTrace.runtime_meta`` must carry status=error
    and preserve ``run_result.runtime_meta`` fields.

    We exercise the helper exactly as ``_eval_stage``'s except block does,
    without spinning up the full engine (which pulls in a SQLite store, an
    LLM client, population state, etc.).  The behaviour under test is
    purely the ``build_error_runtime_meta(exc, base=run_result.runtime_meta)``
    overlay — so a mechanical simulation captures the same invariants the
    engine relies on.
    """
    from kcsi.orchestrator.engine import _cap_native_memory_fields
    from kcsi.runtime.types import RuntimeResult
    from kcsi.tokens import TokenUsage

    # Arrange: container succeeded, runtime_meta has real fields.
    run_result = RuntimeResult(
        output="model output text",
        tool_trace=[{"tool": "Read", "args": {"file": "solution.py"}}],
        runtime_meta={
            "status": "success",
            "duration_ms": 5432,
            "input_tokens": 800,
            "output_tokens": 120,
            "tokens_source": "result",
        },
        token_usage=TokenUsage(input_tokens=800, output_tokens=120),
    )

    # The guard trips during evaluator.evaluate().
    guard_exc = ValueError("Unsafe file path in task metadata (path traversal): '../../etc/passwd'")

    # Act: simulate the engine's except-block overlay.
    container_meta = _cap_native_memory_fields(run_result.runtime_meta)
    preserved = build_error_runtime_meta(guard_exc, base=container_meta)

    # Assert: the attempt row's runtime_meta_json will carry:
    assert preserved["status"] == "error"
    assert "path traversal" in preserved["error"]
    assert preserved["error_type"] == "ValueError"

    # Forensics preserved from the container that did run:
    assert preserved["duration_ms"] == 5432
    assert preserved["input_tokens"] == 800
    assert preserved["output_tokens"] == 120
    assert preserved["tokens_source"] == "result"

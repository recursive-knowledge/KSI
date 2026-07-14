"""Regression tests for silent-failure runtime_meta persistence.

Context
-------
Forensics on the 2026-04 Haiku baseline sweep showed that when the
claude-agent-sdk's async `query(...)` iterator drains without yielding events
to the Node wrapper, the underlying claude-code CLI subprocess STILL runs
to completion inside the container: 15-30 turns, thousands of tokens, real
tool calls captured in the on-host session transcript
(``runtime_state/provider_sessions/<task>/.claude/projects/*.jsonl``).

PR #351 caught the silent-success bug by reclassifying these runs as a
``silent_failure`` and raising ``RuntimeError`` inside ``container_host``.
But the engine's ``_eval_stage`` exception handler wrote ``runtime_meta={}``
on the resulting trace — stripping the ~134 KB of native session memory
that had already been harvested by ``runtime_runner/src/main.ts`` and placed
into ``parsed['runtime_meta']``. The attempt row ended up with a 129-byte
JSON blob containing only zero-filled token counters, while the file on
disk held the full session.

This PR preserves that meta across the raise/except boundary by:

  1. Defining ``SilentAgentRuntimeError`` (RuntimeError subclass) that
     carries ``runtime_meta`` as an instance attribute.

  2. Having ``KsiContainerExecutor.run_task`` raise the subclass (not a
     plain RuntimeError) when the silent-failure sentinel is observed.

  3. Having the engine's ``_eval_stage`` exception handler detect the
     subclass and restore ``runtime_meta`` on the trace (capped at
     ``KSI_NATIVE_MEMORY_MAX_CHARS`` via ``_cap_native_memory_fields``).

The cap prevents unbounded bloat if an evaluator re-runs pathologically
long sessions, and matches the semantics documented in CLAUDE.md (positive
integer → enforced cap, non-positive → capture disabled).

These tests exercise the full persistence chain end-to-end: simulate a
silent_failure attempt with non-empty ``native_session_memory``, write it
through ``MemoryStore.insert_task_trace``, read it back from the DB, and
assert the JSON round-trip preserves the payload (up to the cap).
"""

from __future__ import annotations

import json

import pytest

from ksi.memory.store import MemoryStore
from ksi.orchestrator.engine import _cap_native_memory_fields
from ksi.runtime.normalize import (
    SILENT_FAILURE_STATUS,
    SilentAgentRuntimeError,
)


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "silent_failure_meta.sqlite")
    s = MemoryStore(db_path, default_experiment="test_silent")
    try:
        yield s
    finally:
        s.close()


class TestSilentAgentRuntimeError:
    def test_carries_runtime_meta(self):
        meta = {
            "status": SILENT_FAILURE_STATUS,
            "native_session_memory": "<134 KB of turns>",
            "raw_native_session_memory": "<same raw>",
            "task_id": "55783887",
        }
        exc = SilentAgentRuntimeError("agent-runner produced no output", runtime_meta=meta)
        assert isinstance(exc, RuntimeError), (
            "must subclass RuntimeError so existing except-RuntimeError paths still catch it"
        )
        assert exc.runtime_meta == meta
        assert str(exc) == "agent-runner produced no output"

    def test_runtime_meta_defaults_to_empty_dict(self):
        exc = SilentAgentRuntimeError("no meta")
        assert exc.runtime_meta == {}

    def test_runtime_meta_is_copied_not_aliased(self):
        meta = {"key": "value"}
        exc = SilentAgentRuntimeError("msg", runtime_meta=meta)
        exc.runtime_meta["added"] = True
        # Mutations on the exception's copy must not leak to the caller's dict.
        assert "added" not in meta


class TestCapNativeMemoryFields:
    def test_preserves_memory_fields_below_cap(self, monkeypatch):
        monkeypatch.delenv("KSI_NATIVE_MEMORY_MAX_CHARS", raising=False)
        meta = {
            "native_session_memory": "short transcript",
            "raw_native_session_memory": "raw short",
            "task_id": "t1",
        }
        out = _cap_native_memory_fields(meta)
        assert out["native_session_memory"] == "short transcript"
        assert out["raw_native_session_memory"] == "raw short"
        assert out["task_id"] == "t1"

    def test_truncates_oversized_memory_to_tail(self, monkeypatch):
        monkeypatch.setenv("KSI_NATIVE_MEMORY_MAX_CHARS", "100")
        long_memory = "A" * 500 + "TAIL_MARKER_END"
        meta = {"native_session_memory": long_memory, "raw_native_session_memory": long_memory}
        out = _cap_native_memory_fields(meta)
        assert len(out["native_session_memory"]) == 100
        # Tail slice: the final turns carry the most relevant evidence.
        assert out["native_session_memory"].endswith("TAIL_MARKER_END")
        assert len(out["raw_native_session_memory"]) == 100

    def test_disables_capture_on_non_positive_cap(self, monkeypatch):
        monkeypatch.setenv("KSI_NATIVE_MEMORY_MAX_CHARS", "0")
        meta = {
            "native_session_memory": "something",
            "raw_native_session_memory": "something",
            "task_id": "t1",
        }
        out = _cap_native_memory_fields(meta)
        # Collector semantics (per CLAUDE.md): non-positive → capture disabled
        assert "native_session_memory" not in out
        assert "raw_native_session_memory" not in out
        # Other meta keys survive.
        assert out["task_id"] == "t1"

    def test_handles_non_dict_input(self):
        assert _cap_native_memory_fields(None) == {}  # type: ignore[arg-type]
        assert _cap_native_memory_fields("not a dict") == {}  # type: ignore[arg-type]

    def test_non_string_memory_field_is_untouched(self, monkeypatch):
        monkeypatch.setenv("KSI_NATIVE_MEMORY_MAX_CHARS", "100")
        # Defensive: someone sets it to None or a list by mistake — must not crash.
        meta = {"native_session_memory": None, "raw_native_session_memory": ["x"]}
        out = _cap_native_memory_fields(meta)
        assert out["native_session_memory"] is None
        assert out["raw_native_session_memory"] == ["x"]


class TestRecoveredFromSessionStatus:
    """The recovered_from_session status must flow through the full normalize
    stack WITHOUT being reclassified as silent_failure — it represents a real
    attempt that was reconstructed from the on-disk session log.
    """

    def test_recovered_from_session_is_not_flagged_silent(self):
        """parse_runner_stdout must leave status=recovered_from_session alone."""
        from ksi.runtime.normalize import RECOVERED_FROM_SESSION_STATUS, parse_runner_stdout

        stdout = json.dumps(
            {
                "result": "The grid is [[1,2,3]].",
                "tool_trace": [],
                "meta": {
                    "status": RECOVERED_FROM_SESSION_STATUS,
                    "input_tokens": 200,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 1000,
                    "tokens_source": "session_recovery",
                    "recovery_note": "Recovered from /home/node/.claude/projects/x.jsonl",
                    "task_id": "vuls-86b60e",
                },
            }
        )
        parsed = parse_runner_stdout(stdout, key="result")
        assert parsed["runtime_meta"]["status"] == RECOVERED_FROM_SESSION_STATUS
        assert parsed["runtime_meta"].get("error") is None, (
            "recovered_from_session must NOT be reclassified as silent_failure"
        )
        assert parsed["token_usage"].total == 1250
        assert parsed["output"] == "The grid is [[1,2,3]]."

    def test_recovered_status_constant_present(self):
        """Module-level constant must exist so downstream code can import it."""
        from ksi.runtime import normalize

        assert normalize.RECOVERED_FROM_SESSION_STATUS == "recovered_from_session"


class TestSilentFailureAttemptPersistence:
    """End-to-end: silent_failure meta flows from exception → engine trace →
    SqlitePersistence.on_task_trace → MemoryStore.insert_task_trace → the
    attempts.runtime_meta_json column. Read it back and confirm the payload
    survived the JSON round-trip up to the cap.
    """

    def test_native_session_memory_survives_full_persistence_chain(self, tmp_path, store, monkeypatch):
        monkeypatch.delenv("KSI_NATIVE_MEMORY_MAX_CHARS", raising=False)
        # The meta shape that runtime_runner/src/main.ts writes after a silent
        # exit: native_session_memory populated by the on-host collector, plus
        # the zero-filled token counters from the empty container envelope.
        input_memory = (
            "# file: projects/vuls-86b60e/abc-123.jsonl\n"
            "{'type':'assistant',...} × 18 turns × 4494 tokens × 6 tool calls\n" + ("turn-content " * 5000)
            # Fill a realistic size that still fits below the default 240k cap.
        )
        runtime_meta = {
            "status": SILENT_FAILURE_STATUS,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "tokens_source": "unavailable",
            "native_session_memory": input_memory,
            "raw_native_session_memory": input_memory,
            "task_id": "vuls-86b60e",
            "workspace_key": "task__baseline_haiku_arc1__vuls-86b60e__deadbeef",
        }

        # Simulate the engine's preservation step on the silent_failure trace.
        preserved = _cap_native_memory_fields(runtime_meta)
        # Sanity: meta kept the memory fields before DB write.
        assert preserved["native_session_memory"] == input_memory
        assert preserved["raw_native_session_memory"] == input_memory

        store.insert_task_trace(
            experiment="test_silent",
            generation=3,
            agent_id="agent-7",
            task_id="vuls-86b60e",
            model_output=None,
            eval_result={},
            native_score=None,
            tool_trace=[],
            runtime_meta={
                # Engine's real call layers token_usage on top before insert.
                **preserved,
                "token_usage": {"input_tokens": 0, "output_tokens": 0},
            },
            error_text="Silent agent-runner failure for task vuls-86b60e",
        )

        # Read back from the attempts table and verify the JSON blob preserved
        # the native_session_memory.
        row = store._execute(
            "SELECT runtime_meta_json, error_text FROM attempts "
            "JOIN assignments ON attempts.assignment_id = assignments.id "
            "JOIN tasks ON assignments.task_ref = tasks.id "
            "WHERE tasks.task_id = 'vuls-86b60e'",
            fetchone=True,
        )
        assert row is not None, "attempts row missing for silent-failure task"
        assert row["error_text"] == "Silent agent-runner failure for task vuls-86b60e"

        parsed_meta = json.loads(row["runtime_meta_json"])
        assert parsed_meta["native_session_memory"] == input_memory, (
            "native_session_memory was stripped or mangled on DB round-trip — "
            "this is the 2026-04 bug that Fix 2 of this PR addresses"
        )
        assert parsed_meta["raw_native_session_memory"] == input_memory
        assert parsed_meta["status"] == SILENT_FAILURE_STATUS
        assert parsed_meta["task_id"] == "vuls-86b60e"
        # Token counts still zero (aggregate counters must not be inflated by
        # the forensics preservation).
        assert parsed_meta["token_usage"]["input_tokens"] == 0
        assert parsed_meta["token_usage"]["output_tokens"] == 0

    def test_oversized_memory_is_capped_on_persistence_chain(self, tmp_path, store, monkeypatch):
        """When a session log exceeds the cap, the tail is preserved."""
        monkeypatch.setenv("KSI_NATIVE_MEMORY_MAX_CHARS", "256")
        # 10 KB of real content; cap is 256 chars.
        long_memory = "HEAD_JUNK" + ("X" * 10_000) + "TAIL_TURN_CONTENT_RIGHT_AT_END"
        runtime_meta = {
            "status": SILENT_FAILURE_STATUS,
            "native_session_memory": long_memory,
            "raw_native_session_memory": long_memory,
            "task_id": "oversized",
        }
        preserved = _cap_native_memory_fields(runtime_meta)

        store.insert_task_trace(
            experiment="test_silent",
            generation=1,
            agent_id="agent-0",
            task_id="oversized",
            model_output=None,
            eval_result={},
            native_score=None,
            tool_trace=[],
            runtime_meta=preserved,
            error_text="silent",
        )
        row = store._execute(
            "SELECT runtime_meta_json FROM attempts "
            "JOIN assignments ON attempts.assignment_id = assignments.id "
            "JOIN tasks ON assignments.task_ref = tasks.id "
            "WHERE tasks.task_id = 'oversized'",
            fetchone=True,
        )
        assert row is not None
        parsed = json.loads(row["runtime_meta_json"])
        assert len(parsed["native_session_memory"]) == 256
        # Tail slice preserves the terminal content (where the final turns live)
        assert parsed["native_session_memory"].endswith("TAIL_TURN_CONTENT_RIGHT_AT_END")
        assert "HEAD_JUNK" not in parsed["native_session_memory"]

    def test_regression_true_silent_fail_with_no_memory_persists_empty(self, tmp_path, store):
        """A TRUE silent-fail with empty native_session_memory must not crash
        the persistence path and must end up with an empty string (or missing
        field) — not an exception.

        This is the third regression from the task description: don't let the
        memory-preservation code assume memory is always present.
        """
        runtime_meta = {
            "status": SILENT_FAILURE_STATUS,
            "native_session_memory": "",
            "raw_native_session_memory": "",
            "task_id": "truly-silent",
        }
        preserved = _cap_native_memory_fields(runtime_meta)
        # Empty string is still a string — kept as-is.
        assert preserved["native_session_memory"] == ""
        assert preserved["raw_native_session_memory"] == ""

        store.insert_task_trace(
            experiment="test_silent",
            generation=1,
            agent_id="agent-0",
            task_id="truly-silent",
            model_output=None,
            eval_result={},
            native_score=None,
            tool_trace=[],
            runtime_meta=preserved,
            error_text="silent-no-log",
        )
        row = store._execute(
            "SELECT runtime_meta_json, error_text FROM attempts "
            "JOIN assignments ON attempts.assignment_id = assignments.id "
            "JOIN tasks ON assignments.task_ref = tasks.id "
            "WHERE tasks.task_id = 'truly-silent'",
            fetchone=True,
        )
        assert row is not None
        assert row["error_text"] == "silent-no-log"
        parsed = json.loads(row["runtime_meta_json"])
        assert parsed["native_session_memory"] == ""
        assert parsed["status"] == SILENT_FAILURE_STATUS

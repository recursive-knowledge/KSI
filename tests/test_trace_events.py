"""Tests for src/kcsi/trace_events.py -- trace event appending and helpers."""

from __future__ import annotations

import json

from kcsi.trace_events import _now_iso, append_trace_event, get_trace_dir


class TestNowIso:
    def test_returns_string(self):
        result = _now_iso()
        assert isinstance(result, str)

    def test_iso_format_parseable(self):
        from datetime import datetime

        result = _now_iso()
        # Should be parseable as an ISO timestamp
        dt = datetime.fromisoformat(result)
        assert dt is not None


class TestGetTraceDir:
    def test_returns_env_value(self, monkeypatch):
        monkeypatch.setenv("KCSI_TRACE_DIR", "/tmp/traces")
        assert get_trace_dir() == "/tmp/traces"

    def test_returns_empty_when_unset(self, monkeypatch):
        monkeypatch.delenv("KCSI_TRACE_DIR", raising=False)
        assert get_trace_dir() == ""

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("KCSI_TRACE_DIR", "  /tmp/traces  ")
        assert get_trace_dir() == "/tmp/traces"


class TestAppendTraceEvent:
    def test_noop_on_empty_trace_dir(self, tmp_path):
        """Should silently return when trace_dir is empty."""
        append_trace_event("", "test.jsonl", {"key": "val"})
        # No error, no files created

    def test_creates_dir_and_writes_event(self, tmp_path):
        trace_dir = str(tmp_path / "traces")
        append_trace_event(trace_dir, "events.jsonl", {"action": "test", "value": 42})

        out_file = tmp_path / "traces" / "events.jsonl"
        assert out_file.exists()

        lines = out_file.read_text().strip().split("\n")
        assert len(lines) == 1

        event = json.loads(lines[0])
        assert event["action"] == "test"
        assert event["value"] == 42
        assert "ts" in event

    def test_appends_multiple_events(self, tmp_path):
        trace_dir = str(tmp_path / "traces")
        append_trace_event(trace_dir, "multi.jsonl", {"seq": 1})
        append_trace_event(trace_dir, "multi.jsonl", {"seq": 2})
        append_trace_event(trace_dir, "multi.jsonl", {"seq": 3})

        out_file = tmp_path / "traces" / "multi.jsonl"
        lines = out_file.read_text().strip().split("\n")
        assert len(lines) == 3
        seqs = [json.loads(line)["seq"] for line in lines]
        assert seqs == [1, 2, 3]

    def test_separate_filenames(self, tmp_path):
        trace_dir = str(tmp_path / "traces")
        append_trace_event(trace_dir, "a.jsonl", {"file": "a"})
        append_trace_event(trace_dir, "b.jsonl", {"file": "b"})

        assert (tmp_path / "traces" / "a.jsonl").exists()
        assert (tmp_path / "traces" / "b.jsonl").exists()

    def test_ts_field_added_automatically(self, tmp_path):
        trace_dir = str(tmp_path / "traces")
        append_trace_event(trace_dir, "ts.jsonl", {"x": 1})

        event = json.loads((tmp_path / "traces" / "ts.jsonl").read_text().strip())
        assert "ts" in event
        # ts should be first key (prepended)
        keys = list(event.keys())
        assert keys[0] == "ts"

    def test_survives_invalid_trace_dir(self):
        """append_trace_event must never raise -- tracing is best-effort."""
        # Use a path that can't be created (null byte in name)
        append_trace_event("/dev/null/impossible\x00path", "f.jsonl", {"x": 1})
        # Should not raise

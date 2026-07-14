"""Tests for src/ksi/runtime/native_memory.py -- native memory and archive collection."""

from __future__ import annotations

import os

from ksi.runtime.native_memory import (
    _env_int,
    collect_native_session_memory,
)


# ---------------------------------------------------------------------------
# _env_int
# ---------------------------------------------------------------------------
class TestEnvInt:
    def test_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("TEST_VAR", raising=False)
        assert _env_int("TEST_VAR", 42) == 42

    def test_returns_parsed_value(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "100")
        assert _env_int("TEST_VAR", 42) == 100

    def test_returns_default_on_non_integer(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "not_a_number")
        assert _env_int("TEST_VAR", 42) == 42

    def test_returns_default_on_empty_string(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "")
        assert _env_int("TEST_VAR", 42) == 42

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "  7  ")
        # "  7  ".strip() == "7" which is parseable
        assert _env_int("TEST_VAR", 42) == 7

    def test_zero_value(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "0")
        assert _env_int("TEST_VAR", 42) == 0

    def test_negative_value(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "-5")
        assert _env_int("TEST_VAR", 42) == -5


# ---------------------------------------------------------------------------
# collect_native_session_memory
# ---------------------------------------------------------------------------
class TestCollectNativeSessionMemory:
    def test_empty_group_folder(self):
        assert collect_native_session_memory("") == ""

    def test_disabled_when_max_chars_zero(self):
        assert collect_native_session_memory("any", max_chars=0) == ""

    def test_disabled_when_max_chars_negative(self):
        assert collect_native_session_memory("any", max_chars=-1) == ""

    def test_returns_empty_when_sessions_root_missing(self, tmp_path, monkeypatch):
        """When no session directory exists, returns empty."""
        monkeypatch.chdir(tmp_path)
        result = collect_native_session_memory("nonexistent_group", max_chars=1000)
        assert result == ""

    def test_collects_files(self, tmp_path, monkeypatch):
        """Create a fake sessions directory structure and verify collection."""
        monkeypatch.chdir(tmp_path)
        # Create the directory structure that _sessions_root_candidates expects:
        # runtime_state/provider_sessions/tasks/<group>/.claude/<files>
        session_dir = tmp_path / "runtime_state" / "provider_sessions" / "tasks" / "task__group1" / ".claude"
        session_dir.mkdir(parents=True)

        (session_dir / "a.jsonl").write_text('{"message":"Hello from file A"}')
        (session_dir / "b.jsonl").write_text('{"message":"Hello from file B"}')

        result = collect_native_session_memory("task__group1", max_chars=10000)
        assert "Hello from file A" in result
        assert "Hello from file B" in result

    def test_respects_max_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        session_dir = tmp_path / "runtime_state" / "provider_sessions" / "tasks" / "task__grp" / ".claude"
        session_dir.mkdir(parents=True)

        for i in range(5):
            f = session_dir / f"file{i}.jsonl"
            f.write_text(f'{{"message":"content {i}"}}')
            # Stagger mtime so sort order is deterministic
            os.utime(f, (1000 + i, 1000 + i))

        result = collect_native_session_memory("task__grp", max_chars=100000, max_files=2)
        # Only the 2 most recent files (file4, file3) should be included
        assert "content 4" in result
        assert "content 3" in result
        # file0 should be excluded (oldest)
        assert "content 0" not in result

    def test_respects_max_chars_per_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        session_dir = tmp_path / "runtime_state" / "provider_sessions" / "tasks" / "task__grp" / ".claude"
        session_dir.mkdir(parents=True)

        (session_dir / "big.jsonl").write_text("X" * 200)

        result = collect_native_session_memory("task__grp", max_chars=100000, max_chars_per_file=50)
        # File content should be truncated (takes last N chars)
        assert len(result) < 200 + 50  # header + truncated content

    def test_respects_max_chars_total(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        session_dir = tmp_path / "runtime_state" / "provider_sessions" / "tasks" / "task__grp" / ".claude"
        session_dir.mkdir(parents=True)

        for i in range(10):
            (session_dir / f"file{i}.jsonl").write_text("A" * 500)

        result = collect_native_session_memory("task__grp", max_chars=1000)
        assert len(result) <= 1000

    def test_ignores_non_matching_extensions(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        session_dir = tmp_path / "runtime_state" / "provider_sessions" / "tasks" / "task__grp" / ".claude"
        session_dir.mkdir(parents=True)

        (session_dir / "data.json").write_text('{"key": "value"}')
        (session_dir / "note.md").write_text("A note")
        (session_dir / "note.jsonl").write_text('{"message":"A jsonl note"}')

        result = collect_native_session_memory("task__grp", max_chars=10000)
        # Only .jsonl transcript-like files are eligible.
        assert "key" not in result
        assert "A note" not in result
        assert "A jsonl note" in result

    def test_prefers_project_jsonl_transcripts_over_debug_noise(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        session_dir = tmp_path / "runtime_state" / "provider_sessions" / "tasks" / "task__grp" / ".claude"
        debug_dir = session_dir / "debug"
        project_dir = session_dir / "projects" / "-workspace-task"
        debug_dir.mkdir(parents=True)
        project_dir.mkdir(parents=True)

        (debug_dir / "trace.txt").write_text("DEBUG NOISE")
        (project_dir / "session.jsonl").write_text('{"type":"assistant","text":"useful transcript"}')

        result = collect_native_session_memory("task__grp", max_chars=10000)
        assert "useful transcript" in result
        assert "DEBUG NOISE" not in result

    def test_excludes_sidechain_agent_jsonl_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        session_dir = tmp_path / "runtime_state" / "provider_sessions" / "tasks" / "task__grp" / ".claude"
        project_dir = session_dir / "projects" / "-workspace-task"
        project_dir.mkdir(parents=True)

        (project_dir / "session.jsonl").write_text('{"isSidechain": false, "message": {"content": "main transcript"}}')
        (project_dir / "agent-a23feab.jsonl").write_text(
            '{"isSidechain": true, "message": {"content": "Warmup helper transcript"}}'
        )

        result = collect_native_session_memory("task__grp", max_chars=10000)
        assert "main transcript" in result
        assert "Warmup helper transcript" not in result
        assert "agent-a23feab.jsonl" not in result

    def test_strips_sidechain_entries_inside_jsonl_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        session_dir = tmp_path / "runtime_state" / "provider_sessions" / "tasks" / "task__grp" / ".claude"
        project_dir = session_dir / "projects" / "-workspace-task"
        project_dir.mkdir(parents=True)

        (project_dir / "session.jsonl").write_text(
            "\n".join(
                [
                    '{"isSidechain": true, "message": {"content": "Warmup"}}',
                    '{"isSidechain": false, "message": {"content": "Primary task transcript"}}',
                ]
            )
        )

        result = collect_native_session_memory("task__grp", max_chars=10000)
        assert "Primary task transcript" in result
        assert "Warmup" not in result

    def test_fallback_excludes_debug_and_todos(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        session_dir = tmp_path / "runtime_state" / "provider_sessions" / "tasks" / "task__grp" / ".claude"
        (session_dir / "debug").mkdir(parents=True)
        (session_dir / "todos").mkdir(parents=True)
        (session_dir / "notes").mkdir(parents=True)
        (session_dir / "skills" / "agent-browser").mkdir(parents=True)

        (session_dir / "debug" / "trace.jsonl").write_text('{"message":"debug text"}')
        (session_dir / "todos" / "todo.jsonl").write_text('{"message":"todo text"}')
        (session_dir / "notes" / "useful.jsonl").write_text('{"message":"useful note"}')
        (session_dir / "skills" / "agent-browser" / "SKILL.md").write_text("skill text")

        result = collect_native_session_memory("task__grp", max_chars=10000)
        assert "useful note" in result
        assert "debug text" not in result
        assert "todo text" not in result
        assert "skill text" not in result

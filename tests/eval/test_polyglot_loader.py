"""Tests for the polyglot task loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ksi.benchmarks.loaders import _load_polyglot_tasks
from ksi.tasks.loaders import SUPPORTED_TASK_SOURCES, load_tasks_for_source


def test_polyglot_in_supported_sources():
    assert "polyglot" in SUPPORTED_TASK_SOURCES


def _make_fixture(tmp_path: Path, entries: list[dict]) -> Path:
    """Write a JSON fixture file and return its path."""
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps(entries), encoding="utf-8")
    return p


class TestLoadPolyglotTasks:
    """Unit tests for _load_polyglot_tasks."""

    def test_loads_two_tasks(self, tmp_path: Path):
        entries = [
            {
                "instance_id": "python__hello-world",
                "language": "python",
                "exercise_name": "hello-world",
                "problem_statement": "Write a function that returns 'Hello, World!'",
                "starter_code": {"hello_world.py": "def hello(): ..."},
                "test_files": {"test_hello.py": "def test_hello(): assert hello() == 'Hello, World!'"},
                "reference_solution": "def hello():\n    return 'Hello, World!'",
                "build_files": {},
                "test_command": "pytest test_hello.py",
                "meta_config": {"difficulty": "easy"},
            },
            {
                "instance_id": "rust__two-fer",
                "language": "rust",
                "exercise_name": "two-fer",
                "problem_statement": "Create a sentence of the form 'One for X, one for me.'",
                "starter_code": {"src/lib.rs": "pub fn twofer(name: &str) -> String { todo!() }"},
                "test_files": {"tests/two_fer.rs": "// test code"},
                "reference_solution": 'pub fn twofer(name: &str) -> String { format!("One for {name}, one for me.") }',
                "build_files": {"Cargo.toml": '[package]\nname = "two-fer"'},
                "test_command": "cargo test",
                "meta_config": {"difficulty": "easy", "topics": ["strings"]},
            },
        ]
        path = _make_fixture(tmp_path, entries)
        tasks = _load_polyglot_tasks(path)

        assert len(tasks) == 2

        # First task: id comes from instance_id
        t0 = tasks[0]
        assert t0.id == "python__hello-world"
        assert t0.repo == ""
        assert t0.prompt == "Write a function that returns 'Hello, World!'"
        assert t0.metadata["task_source"] == "polyglot"
        assert t0.metadata["language"] == "python"
        assert t0.metadata["exercise_name"] == "hello-world"
        assert t0.metadata["starter_code"] == {"hello_world.py": "def hello(): ..."}
        assert "test_hello.py" in t0.metadata["test_files"]
        assert t0.metadata["reference_solution"].startswith("def hello()")
        assert t0.metadata["build_files"] == {}
        assert t0.metadata["test_command"] == "pytest test_hello.py"
        assert t0.metadata["meta_config"]["difficulty"] == "easy"

        # Second task: instance_id from JSON
        t1 = tasks[1]
        assert t1.id == "rust__two-fer"
        assert t1.metadata["language"] == "rust"
        assert t1.metadata["exercise_name"] == "two-fer"
        assert "Cargo.toml" in t1.metadata["build_files"]
        assert t1.metadata["test_command"] == "cargo test"
        assert t1.metadata["meta_config"]["topics"] == ["strings"]

    def test_empty_array_returns_empty_list(self, tmp_path: Path):
        path = _make_fixture(tmp_path, [])
        tasks = _load_polyglot_tasks(path)
        assert tasks == []

    def test_rejects_non_json_file(self, tmp_path: Path):
        p = tmp_path / "tasks.parquet"
        p.write_text("not parquet", encoding="utf-8")
        with pytest.raises(ValueError, match="expects a .json file"):
            _load_polyglot_tasks(p)

    def test_rejects_non_array_json(self, tmp_path: Path):
        p = tmp_path / "tasks.json"
        p.write_text(json.dumps({"not": "an array"}), encoding="utf-8")
        with pytest.raises(ValueError, match="must be an array"):
            _load_polyglot_tasks(p)

    def test_skips_entries_missing_language(self, tmp_path: Path):
        entries = [
            {"exercise_name": "hello-world"},  # missing language
            {"language": "python", "exercise_name": "hello-world"},
        ]
        path = _make_fixture(tmp_path, entries)
        tasks = _load_polyglot_tasks(path)
        assert len(tasks) == 1
        assert tasks[0].metadata["language"] == "python"

    def test_skips_entries_missing_exercise_name(self, tmp_path: Path):
        entries = [
            {"language": "python"},  # missing exercise_name
        ]
        path = _make_fixture(tmp_path, entries)
        tasks = _load_polyglot_tasks(path)
        assert tasks == []

    def test_defaults_for_optional_fields(self, tmp_path: Path):
        """Entries with only language and exercise_name should get sane defaults."""
        entries = [{"language": "go", "exercise_name": "leap"}]  # no instance_id → synthesized
        path = _make_fixture(tmp_path, entries)
        tasks = _load_polyglot_tasks(path)

        assert len(tasks) == 1
        assert tasks[0].id == "go__leap"  # synthesized from language__exercise_name
        assert "leap" in tasks[0].prompt  # fallback prompt mentions exercise
        m = tasks[0].metadata
        assert m["starter_code"] == {}
        assert m["test_files"] == {}
        assert m["reference_solution"] == ""
        assert m["build_files"] == {}
        assert m["test_command"] == ""
        assert m["meta_config"] == {}

    def test_test_feedback_metadata_defaults(self, tmp_path: Path):
        entries = [{"language": "python", "exercise_name": "bowling", "instance_id": "python__bowling"}]
        path = _make_fixture(tmp_path, entries)
        tasks = _load_polyglot_tasks(path)

        assert tasks[0].metadata["polyglot_test_feedback_tries"] == 2
        assert tasks[0].metadata["polyglot_test_feedback_max_lines"] == 50

    def test_test_feedback_metadata_honors_explicit_overrides(self, tmp_path: Path):
        entries = [{"language": "python", "exercise_name": "bowling", "instance_id": "python__bowling"}]
        path = _make_fixture(tmp_path, entries)
        tasks = _load_polyglot_tasks(
            path,
            polyglot_test_feedback_tries=1,
            polyglot_test_feedback_max_lines=20,
        )

        assert tasks[0].metadata["polyglot_test_feedback_tries"] == 1
        assert tasks[0].metadata["polyglot_test_feedback_max_lines"] == 20


class TestLoadTasksForSourcePolyglot:
    """Integration test: load_tasks_for_source routes to polyglot loader."""

    def test_routing(self, tmp_path: Path):
        entries = [{"language": "python", "exercise_name": "hello-world"}]
        path = _make_fixture(tmp_path, entries)
        tasks = load_tasks_for_source(task_source="polyglot", tasks_path=path)
        assert len(tasks) == 1
        assert tasks[0].metadata["task_source"] == "polyglot"


def test_load_tasks_for_source_rejects_parquet_source(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        '{"instance_id": "demo__task", "repo": "demo/repo", "problem_statement": "no-op"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported task_source='parquet'"):
        load_tasks_for_source(task_source="parquet", tasks_path=tasks_path)

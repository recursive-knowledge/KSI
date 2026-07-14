import json
from pathlib import Path

import pytest

from ksi.tasks.custom import load_custom_tasks, validate_custom_tasks_path
from ksi.tasks.loaders import load_tasks_for_source
from ksi.tasks.registry import resolve_source


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def test_custom_source_registered_with_command_default():
    spec = resolve_source("custom")
    assert spec is not None
    assert spec.default_evaluator == "command"
    assert spec.prompt_kind == "generic"
    assert spec.loader is not None
    assert spec.validate_tasks_path is not None


def test_load_jsonl_files_record(tmp_path):
    tasks_file = _write_jsonl(
        tmp_path / "tasks.jsonl",
        [
            {
                "task_id": "t1",
                "prompt": "Write solution.py so tests pass.",
                "files": {"tests.py": "import solution\n"},
                "eval": {"command": "python3 tests.py", "timeout_sec": 42},
            }
        ],
    )
    tasks = load_custom_tasks(tasks_file)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.id == "t1"
    assert "solution.py" in t.prompt
    md = t.metadata
    assert md["task_source"] == "custom"
    assert md["eval_command"] == "python3 tests.py"
    assert md["eval_timeout_sec"] == 42.0
    seed = Path(md["repo_path"])
    assert seed.is_dir()
    assert (seed / "tests.py").read_text(encoding="utf-8") == "import solution\n"


def test_load_json_array_workspace_dir(tmp_path):
    ws = tmp_path / "proj"
    ws.mkdir()
    (ws / "hello.txt").write_text("hi", encoding="utf-8")
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text(
        json.dumps([{"task_id": "t1", "prompt": "p", "workspace_dir": "proj"}]),
        encoding="utf-8",
    )
    tasks = load_custom_tasks(tasks_file)
    assert tasks[0].metadata["repo_path"] == str(ws.resolve())
    assert tasks[0].metadata["eval_command"] == ""


def test_neither_files_nor_workspace_gets_empty_seed(tmp_path):
    tasks_file = _write_jsonl(tmp_path / "t.jsonl", [{"task_id": "a", "prompt": "p"}])
    tasks = load_custom_tasks(tasks_file)
    seed = Path(tasks[0].metadata["repo_path"])
    assert seed.is_dir()
    assert list(seed.iterdir()) == []


@pytest.mark.parametrize(
    "record, match",
    [
        ({"prompt": "p"}, "task_id"),
        ({"task_id": "a"}, "prompt"),
        ({"task_id": "a", "prompt": "p", "files": {"../x": "y"}}, "files"),
        ({"task_id": "a", "prompt": "p", "files": {"/abs": "y"}}, "files"),
        ({"task_id": "a", "prompt": "p", "workspace_dir": "missing"}, "workspace_dir"),
        ({"task_id": "a", "prompt": "p", "files": {"x": "y"}, "workspace_dir": "."}, "mutually exclusive"),
        ({"task_id": "a", "prompt": "p", "eval": {"command": ""}}, "eval.command"),
        ({"task_id": "a", "prompt": "p", "eval": {"command": "x", "timeout_sec": -1}}, "timeout_sec"),
        ({"task_id": "a", "prompt": "p", "bogus": 1}, "unknown"),
    ],
)
def test_invalid_records_raise_with_context(tmp_path, record, match):
    tasks_file = _write_jsonl(tmp_path / "t.jsonl", [record])
    with pytest.raises(ValueError, match=match):
        load_custom_tasks(tasks_file)


def test_duplicate_task_ids_raise(tmp_path):
    tasks_file = _write_jsonl(
        tmp_path / "t.jsonl",
        [{"task_id": "a", "prompt": "p"}, {"task_id": "a", "prompt": "q"}],
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_custom_tasks(tasks_file)


def test_validate_tasks_path(tmp_path):
    assert validate_custom_tasks_path(tmp_path / "nope.jsonl") is not None
    bad = tmp_path / "bad.txt"
    bad.write_text("x", encoding="utf-8")
    assert ".json" in (validate_custom_tasks_path(bad) or "")
    good = _write_jsonl(tmp_path / "ok.jsonl", [{"task_id": "a", "prompt": "p"}])
    assert validate_custom_tasks_path(good) is None
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert validate_custom_tasks_path(empty) is not None


def test_load_via_registry_dispatch(tmp_path):
    tasks_file = _write_jsonl(tmp_path / "t.jsonl", [{"task_id": "a", "prompt": "p"}])
    tasks = load_tasks_for_source(task_source="custom", tasks_path=tasks_file)
    assert [t.id for t in tasks] == ["a"]


def test_prompt_carries_workspace_guidance(tmp_path):
    tasks_file = _write_jsonl(
        tmp_path / "t.jsonl",
        [{"task_id": "a", "prompt": "Do the thing.", "files": {"f.txt": "x"}}],
    )
    t = load_custom_tasks(tasks_file)[0]
    assert t.prompt.startswith("Do the thing.")
    assert "repo" in t.prompt  # workspace-location guidance appended


def test_absolute_workspace_dir(tmp_path):
    ws = tmp_path / "abs_proj"
    ws.mkdir()
    tasks_file = _write_jsonl(
        tmp_path / "t.jsonl",
        [{"task_id": "a", "prompt": "p", "workspace_dir": str(ws)}],
    )
    assert load_custom_tasks(tasks_file)[0].metadata["repo_path"] == str(ws.resolve())


def test_temp_seed_dirs_registered_for_cleanup(tmp_path):
    from ksi.tasks import custom as custom_mod

    tasks_file = _write_jsonl(
        tmp_path / "t.jsonl",
        [{"task_id": "a", "prompt": "p", "files": {"f.txt": "x"}}],
    )
    t = load_custom_tasks(tasks_file)[0]
    assert t.metadata["repo_path"] in custom_mod._TEMP_SEED_DIRS

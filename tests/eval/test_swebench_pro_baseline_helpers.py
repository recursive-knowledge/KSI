import importlib.util
import json
from pathlib import Path

import pytest
from conftest import REPO_ROOT

from ksi.models import TaskSpec
from ksi.tasks.repo_cache import prepare_swebench_repo_snapshots

WRAPPER = REPO_ROOT / "benchmarks" / "scripts" / "run_swebench_pro_eval.py"
PREP = REPO_ROOT / "benchmarks" / "scripts" / "dataprep" / "prepare_swebench_pro_repo_cache.py"
VALID_SHA = "a" * 40
VALID_SHA_2 = "b" * 40
VALID_SHA_3 = "c" * 40


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_swebench_pro_eval_wrapper_maps_baseline_cli_to_official_cli(tmp_path: Path) -> None:
    module = _load_script(WRAPPER, "run_swebench_pro_eval")
    eval_root = tmp_path / "SWE-bench_Pro-os"
    scripts_dir = eval_root / "run_scripts"
    scripts_dir.mkdir(parents=True)
    (eval_root / "swe_bench_pro_eval.py").write_text("# stub\n", encoding="utf-8")
    (eval_root / module.REVISION_MARKER).write_text(
        module.EXPECTED_EVAL_REVISION,
        encoding="utf-8",
    )
    raw_sample = tmp_path / "test.jsonl"
    raw_sample.write_text("{}", encoding="utf-8")
    patch_file = tmp_path / "patches.json"
    patch_file.write_text("[]", encoding="utf-8")

    args = module.parse_args(
        [
            "--patch-path",
            str(patch_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--source-dir",
            str(eval_root),
            "--raw-sample-path",
            str(raw_sample),
            "--scripts-dir",
            str(scripts_dir),
            "--dockerhub-username",
            "example",
            "--use-local-docker",
            "--num-workers",
            "7",
            "--redo",
            "--block-network",
            "--docker-platform",
            "linux/amd64",
        ]
    )

    cmd, cwd = module.build_eval_command(args)

    assert cwd == eval_root.resolve()
    assert cmd[1] == str((eval_root / "swe_bench_pro_eval.py").resolve())
    assert "--raw_sample_path" in cmd
    assert "--patch_path" in cmd
    assert "--output_dir" in cmd
    assert "--dockerhub_username" in cmd
    assert "--scripts_dir" in cmd
    assert "--use_local_docker" in cmd
    assert "--block_network" in cmd
    assert cmd[cmd.index("--num_workers") + 1] == "7"
    assert cmd[cmd.index("--docker_platform") + 1] == "linux/amd64"


def test_prepare_swebench_pro_repo_cache_filters_task_map(monkeypatch, tmp_path: Path) -> None:
    module = _load_script(PREP, "prepare_swebench_pro_repo_cache")
    task_map = tmp_path / "task_map.json"
    task_map.write_text(
        json.dumps({"tasks": [{"task_id": "task-b"}, {"task_id": "task-a"}]}),
        encoding="utf-8",
    )
    tasks = [
        TaskSpec(
            id="task-a",
            prompt="a",
            repo="owner/a",
            metadata={"task_source": "swebench_pro", "base_commit": VALID_SHA},
        ),
        TaskSpec(
            id="task-b",
            prompt="b",
            repo="owner/b",
            metadata={"task_source": "swebench_pro", "base_commit": VALID_SHA_2},
        ),
        TaskSpec(
            id="task-c",
            prompt="c",
            repo="owner/c",
            metadata={"task_source": "swebench_pro", "base_commit": VALID_SHA_3},
        ),
    ]
    captured: dict[str, object] = {}

    def fake_load_tasks_for_source(**kwargs):  # type: ignore[no-untyped-def]
        captured["load_kwargs"] = kwargs
        return tasks

    def fake_prepare_swebench_repo_snapshots(*, tasks, repos_cache_dir, seed_test_files=False):  # type: ignore[no-untyped-def]
        captured["task_ids"] = [task.id for task in tasks]
        captured["repos_cache_dir"] = repos_cache_dir

    monkeypatch.setattr(module, "load_tasks_for_source", fake_load_tasks_for_source)
    monkeypatch.setattr(module, "prepare_swebench_repo_snapshots", fake_prepare_swebench_repo_snapshots)

    count = module.prepare_repo_cache(
        tasks_path=tmp_path / "test.jsonl",
        task_map=task_map,
        repo_cache=tmp_path / "repo_cache",
    )

    assert count == 2
    assert captured["task_ids"] == ["task-b", "task-a"]
    assert captured["repos_cache_dir"] == tmp_path / "repo_cache"
    assert captured["load_kwargs"] == {
        "task_source": "swebench_pro",
        "tasks_path": tmp_path / "test.jsonl",
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"task_ids": ["task-a", 7]}, "'task_ids'"),
        ({"tasks": [{"task_id": "task-a"}, {"id": "task-b"}]}, "tasks\\[1\\]\\.task_id"),
        (["task-a", {"task_id": "task-b"}], "list entries"),
    ],
)
def test_prepare_swebench_pro_repo_cache_rejects_malformed_task_map_entries(
    payload: object,
    message: str,
    tmp_path: Path,
) -> None:
    module = _load_script(PREP, "prepare_swebench_pro_repo_cache")
    task_map = tmp_path / "task_map.json"
    task_map.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        module._load_task_ids(task_map)


def test_prepare_swebench_pro_repo_cache_rejects_duplicate_task_map_ids(tmp_path: Path) -> None:
    module = _load_script(PREP, "prepare_swebench_pro_repo_cache")
    task_map = tmp_path / "task_map.json"
    task_map.write_text(json.dumps({"task_ids": ["task-a", "task-a"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate task ID"):
        module._load_task_ids(task_map)


def test_prepare_swebench_pro_repo_cache_rejects_unsafe_task_map_ids(tmp_path: Path) -> None:
    module = _load_script(PREP, "prepare_swebench_pro_repo_cache")
    task_map = tmp_path / "task_map.json"
    task_map.write_text(json.dumps({"task_ids": ["../outside"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid task ID"):
        module._load_task_ids(task_map)


def test_prepare_swebench_pro_repo_cache_rejects_duplicate_loaded_task_ids(monkeypatch, tmp_path: Path) -> None:
    module = _load_script(PREP, "prepare_swebench_pro_repo_cache")
    task_map = tmp_path / "task_map.json"
    task_map.write_text(json.dumps({"task_ids": ["task-a"]}), encoding="utf-8")
    tasks = [
        TaskSpec(
            id="task-a",
            prompt="a",
            repo="owner/a",
            metadata={"task_source": "swebench_pro", "base_commit": VALID_SHA},
        ),
        TaskSpec(
            id="task-a",
            prompt="b",
            repo="owner/b",
            metadata={"task_source": "swebench_pro", "base_commit": VALID_SHA_2},
        ),
    ]

    monkeypatch.setattr(module, "load_tasks_for_source", lambda **_kwargs: tasks)

    with pytest.raises(ValueError, match="Duplicate task ID"):
        module.prepare_repo_cache(
            tasks_path=tmp_path / "test.jsonl",
            task_map=task_map,
            repo_cache=tmp_path / "repo_cache",
        )


def test_prepare_swebench_repo_snapshots_rejects_traversal_task_id(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_prepare_one_repo(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)

    monkeypatch.setattr("ksi.tasks.repo_cache._prepare_one_repo", fake_prepare_one_repo)
    task = TaskSpec(
        id="../outside",
        prompt="a",
        repo="owner/repo",
        metadata={"task_source": "swebench_pro", "base_commit": VALID_SHA},
    )

    with pytest.raises(ValueError, match="Invalid repo-cache task ID"):
        prepare_swebench_repo_snapshots(tasks=[task], repos_cache_dir=tmp_path / "repo_cache")

    assert calls == []
    assert "repo_path" not in task.metadata


def test_prepare_swebench_repo_snapshots_rejects_duplicate_task_ids(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_prepare_one_repo(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)

    monkeypatch.setattr("ksi.tasks.repo_cache._prepare_one_repo", fake_prepare_one_repo)
    tasks = [
        TaskSpec(
            id="task-a",
            prompt="a",
            repo="owner/a",
            metadata={"task_source": "swebench_pro", "base_commit": VALID_SHA},
        ),
        TaskSpec(
            id="task-a",
            prompt="b",
            repo="owner/b",
            metadata={"task_source": "swebench_pro", "base_commit": VALID_SHA_2},
        ),
    ]

    with pytest.raises(ValueError, match="Duplicate SWE-bench Pro task ID"):
        prepare_swebench_repo_snapshots(tasks=tasks, repos_cache_dir=tmp_path / "repo_cache")

    assert calls == []


def test_prepare_swebench_pro_repo_cache_accepts_empty_task_map(monkeypatch, tmp_path: Path) -> None:
    module = _load_script(PREP, "prepare_swebench_pro_repo_cache")
    captured: dict[str, object] = {}

    def fake_prepare_repo_cache(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return 3

    monkeypatch.setattr(module, "prepare_repo_cache", fake_prepare_repo_cache)
    monkeypatch.setattr(
        module,
        "DEFAULT_TASKS_PATH",
        tmp_path / "test.jsonl",
    )
    monkeypatch.setattr(
        module,
        "DEFAULT_REPO_CACHE",
        tmp_path / "repo_cache",
    )

    assert module.main(["--task-map", ""]) == 0

    assert captured["task_map"] is None

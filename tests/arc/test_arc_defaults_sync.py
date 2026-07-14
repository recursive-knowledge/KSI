import json

from conftest import REPO_ROOT

ARC_DEFAULTS = REPO_ROOT / "configs" / "benchmarks" / "arc_defaults.json"


def _load_defaults() -> dict:
    return json.loads(ARC_DEFAULTS.read_text(encoding="utf-8"))


def test_arc_defaults_point_to_train_50_task_maps():
    defaults = _load_defaults()
    for benchmark in ("arc1", "arc2"):
        config = defaults[benchmark]
        assert config["split"] == "training"
        assert "train_50" in config["selection_name"]
        assert "train_50" in config["task_map"]
        assert "source/data/training" in config["task_dir"]
        assert "train_50" in config["payload_manifest"]

        task_dir = REPO_ROOT / config["task_dir"]
        task_map = REPO_ROOT / config["task_map"]
        assert task_map.is_file()

        payload = json.loads(task_map.read_text(encoding="utf-8"))
        tasks = payload.get("tasks") or []
        assert len(tasks) == 50
        assert all("/training/" in item["source_file"] for item in tasks)

        if task_dir.exists():
            assert task_dir.is_dir()
            assert all((REPO_ROOT / item["source_file"]).is_file() for item in tasks)

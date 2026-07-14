"""Guards the bundled synthetic ARC demo and `scripts/quickstart.sh`'s bootstrap.

`scripts/quickstart.sh` itself now runs the custom-tasks demo (see
`tests/test_custom_tasks_example.py`); the ARC demo tested here stays bundled
as a `--task-source arc` example (see `benchmarks/README.md`). The demo tasks
must load through the real ARC loader and each must encode a self-consistent
transformation (output == rule(input)) so a correct setup can actually solve
them. The profile-synthesis test below exercises quickstart.sh's bootstrap
behavior directly.
"""

from __future__ import annotations

import subprocess

from kcsi.layout import PROJECT_ROOT
from kcsi.providers import load_provider_profile
from kcsi.tasks.loaders import load_tasks_for_source

DEMO_DIR = PROJECT_ROOT / "examples" / "quickstart" / "arc_demo"
QUICKSTART_SH = PROJECT_ROOT / "scripts" / "quickstart.sh"

RULES = {
    "demo_recolor": lambda g: [[8 if c == 2 else c for c in row] for row in g],
    "demo_mirror": lambda g: [list(reversed(row)) for row in g],
    "demo_transpose": lambda g: [list(col) for col in zip(*g)],
}


def test_demo_dir_is_present_and_tracked() -> None:
    assert DEMO_DIR.is_dir(), f"missing bundled demo dir: {DEMO_DIR}"
    files = sorted(p.stem for p in DEMO_DIR.glob("*.json"))
    assert files == ["demo_mirror", "demo_recolor", "demo_transpose"]


def test_demo_loads_through_arc_loader() -> None:
    tasks = load_tasks_for_source(task_source="arc", tasks_path=DEMO_DIR)
    assert len(tasks) == 3
    for task in tasks:
        meta = task.metadata
        assert meta["task_source"] == "arc"
        assert meta["arc_train_pairs"], f"{task.id} has no train pairs"
        assert meta["arc_test_inputs"], f"{task.id} has no test inputs"


def test_demo_transformations_are_self_consistent() -> None:
    import json

    for stem, rule in RULES.items():
        data = json.loads((DEMO_DIR / f"{stem}.json").read_text())
        for pair in data["train"] + data["test"]:
            assert rule(pair["input"]) == pair["output"], f"{stem}: rule mismatch on {pair['input']}"


def test_quickstart_synthesizes_loadable_profile_from_env_key(tmp_path) -> None:
    """`scripts/quickstart.sh` should turn an ambient API key into a profile the
    real provider loader accepts, with no manual `.env` editing."""
    out = tmp_path / ".env.quickstart"
    env = {
        "PATH": __import__("os").environ.get("PATH", ""),
        "ANTHROPIC_API_KEY": "sk-ant-dummy-not-real",
        "SKIP_DOCTOR": "1",
        "DRY_RUN": "true",
        "PROFILE": str(tmp_path / ".env.missing"),
        "QUICKSTART_PROFILE_OUT": str(out),
    }
    res = subprocess.run(
        ["bash", str(QUICKSTART_SH)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert res.returncode == 0, res.stderr
    assert out.is_file(), "expected a synthesized profile"
    assert f"--provider-profile {out}" in res.stdout
    loaded = load_provider_profile(str(out))
    assert loaded["MODEL_PROVIDER"] == "anthropic"

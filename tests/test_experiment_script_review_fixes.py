import os
import shutil
import subprocess

from conftest import REPO_ROOT

PROFILE = "configs/kcsi/.env.haiku.template"
SETUP_ALL = REPO_ROOT / "scripts" / "setup_all.sh"


def _run_script(rel_path: str, args: list[str], env_overrides: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(REPO_ROOT / rel_path), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def test_run_kcsi_task_source_override_supports_missing_dataset_dry_run():
    name = "pytest_run_kcsi_override"
    result = _run_script(
        "scripts/run_kcsi.sh",
        [
            "--data",
            "missing.json",
            "--model",
            "openai",
            "--name",
            name,
            "--task-source",
            "polyglot",
            "--evaluator",
            "polyglot_harness",
            "--preset",
            "smoke",
        ],
        {"DRY_RUN": "true", "OPENAI_PROFILE": PROFILE},
    )
    try:
        assert result.returncode == 0, result.stderr
        assert "unbound variable" not in result.stderr
        assert "--task-source    polyglot" in result.stdout
        assert "--evaluator    polyglot_harness" in result.stdout
        assert "--no-drop-solved" in result.stdout
    finally:
        shutil.rmtree(REPO_ROOT / "results" / name, ignore_errors=True)


def test_run_kcsi_missing_option_value_is_clean_error():
    result = _run_script("scripts/run_kcsi.sh", ["--model"], {"DRY_RUN": "true"})

    assert result.returncode == 2
    assert "Missing value for --model" in result.stderr
    assert "unbound variable" not in result.stderr


def test_seeded_campaign_artifacts_include_seed_suffix():
    # One LLM per invocation: model positional (haiku), swarm is the default
    # condition. SEED=123 is honored as a single-seed override (back-compat).
    env = {"DRY_RUN": "true", "HAIKU_PROFILE": PROFILE, "SEED": "123"}
    polyglot = _run_script("benchmarks/run_polyglot.sh", ["haiku"], env)
    swebench = _run_script("benchmarks/run_swebench_pro.sh", ["haiku"], env)

    assert polyglot.returncode == 0, polyglot.stderr
    assert "polyglot_haiku_swarm_seed123_knowledge.sqlite" in polyglot.stdout
    assert "polyglot_haiku_swarm_seed123.json" in polyglot.stdout

    assert swebench.returncode == 0, swebench.stderr
    assert "swebench_haiku_swarm_seed123_knowledge.sqlite" in swebench.stdout
    assert "swebench_haiku_swarm_seed123.json" in swebench.stdout


def test_setup_all_uses_locked_dependency_installs():
    text = SETUP_ALL.read_text(encoding="utf-8")

    assert "uv sync --locked --extra memory --extra swebench-pro" in text
    assert "package-lock.json" in text
    assert "npm-shrinkwrap.json" in text
    assert 'ci --silent && ok "$label npm ci"' in text


def test_setup_all_shell_syntax_is_valid():
    result = subprocess.run(["bash", "-n", str(SETUP_ALL)], text=True, capture_output=True)

    assert result.returncode == 0, result.stderr


def test_setup_all_smoke_pytest_fails_closed():
    text = SETUP_ALL.read_text(encoding="utf-8")

    failure_block = text.split("Some tests failed", maxsplit=1)[1].split("fi", maxsplit=1)[0]
    assert "exit 1" in failure_block

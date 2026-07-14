import os
import subprocess

from conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "benchmarks" / "run_terminal_bench_2.sh"
# Any existing file passes common.sh's validate_profiles file-existence check;
# real provider profiles (configs/kcsi/.env.*) are untracked, so point the
# profile env vars at a committed file for the DRY_RUN preview.
EXISTING_PROFILE = "configs/kcsi/.env.haiku.template"


def _run_script(args, env_overrides):
    env = os.environ.copy()
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def test_tb2_campaign_dry_run_composes_expected_cli():
    # One LLM per invocation: model is the first positional (haiku|openai);
    # the swarm condition is the default.
    result = _run_script(
        ["haiku"],
        {"DRY_RUN": "true", "HAIKU_PROFILE": EXISTING_PROFILE},
    )

    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "[DRY RUN] Terminal-Bench 2: Haiku 4.5 Knowledge-Sharing" in out
    # Pinned task source + evaluator + runtime.
    assert "--task-source terminal_bench_2" in out
    assert "--evaluator terminal_bench_2" in out
    assert "--runtime container" in out
    # Default task map.
    assert "benchmarks/terminal_bench_2/task_maps/terminal_bench_2_all.json" in out
    # Env-knob defaults: GENERATIONS=10 (paper main protocol), SEEDS=1,
    # MAX_CONCURRENT=25. The runtime timeout is FIXED at -1 (no hard cap; the
    # per-task task.toml [agent].timeout_sec binds) and is not configurable.
    assert "--generations 10" in out
    assert "--seed 1" in out
    assert "--runtime-timeout-sec -1" in out
    assert "--max-concurrent-tasks 25" in out
    # Fairness: swarm condition keeps knowledge sharing on (no forum-disable flags).
    assert "--per-task-forum-rounds 0" not in out
    # drop-solved is explicit by default to follow the paper's main
    # 10-generation, solved-removed protocol. Harbor-fairness submissions can
    # pass --no-drop-solved.
    assert "--drop-solved" in out
    assert "--no-drop-solved" not in out
    # Runtime audit DB is enabled for Harbor submission tooling; artifact names
    # now carry the seed suffix.
    assert "--runtime-db-path ./tb2_haiku_swarm_seed1_runtime.sqlite" in out


def test_tb2_campaign_sets_require_pull_and_never_sets_step_cap():
    # KCSI_TB2_REQUIRE_PULL is exported by the wrapper (fairness); it defaults
    # to 1 but stays overridable. KCSI_TB2_MAX_STEPS is deliberately left unset
    # (no kcsi-side step cap) — the wrapper must never assign it.
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'export KCSI_TB2_REQUIRE_PULL="${KCSI_TB2_REQUIRE_PULL:-1}"' in text
    assert "KCSI_TB2_IMAGE_DIGEST_MANIFEST" in text
    assert "unset KCSI_TB2_MAX_STEPS" in text
    assert "KCSI_TB2_MAX_STEPS=" not in text
    assert 'uv run python - "$TB2_TASK_MAP"' in text
    assert "json.load(open('$TB2_TASK_MAP'))" not in text


def test_tb2_campaign_env_overrides_flow_into_cli():
    # openai receiver + noforum condition; SEED=3 is honored as a single-seed
    # override (back-compat) when SEEDS is unset.
    result = _run_script(
        ["openai", "noforum"],
        {
            "DRY_RUN": "true",
            "OPENAI_PROFILE": EXISTING_PROFILE,
            "GENERATIONS": "7",
            "SEED": "3",
            "MAX_CONCURRENT": "10",
        },
    )

    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "[DRY RUN] Terminal-Bench 2: GPT-5.4-mini No-Discussion" in out
    assert "--generations 7" in out
    assert "--seed 3" in out
    # The runtime timeout is FIXED (no hard cap; per-task task.toml binds) and
    # not configurable via TIMEOUT.
    assert "--runtime-timeout-sec -1" in out
    assert "--max-concurrent-tasks 10" in out
    # no-discussion condition disables forums + distillation.
    assert "--per-task-forum-rounds 0 --cross-task-forum-rounds 0" in out
    assert "--distill-enabled false" in out


def test_tb2_campaign_rejects_user_set_timeout():
    """TB2's timeout is not configurable: the per-task task.toml [agent].timeout_sec
    is authoritative, so a user-set TIMEOUT aborts the run loudly."""
    result = _run_script(
        ["haiku"],
        {
            "DRY_RUN": "true",
            "HAIKU_PROFILE": EXISTING_PROFILE,
            "TIMEOUT": "1200",
        },
    )
    assert result.returncode == 1
    assert "TIMEOUT is not configurable" in result.stderr
    assert "task.toml" in result.stderr


def test_tb2_campaign_rejects_unknown_model():
    result = _run_script(
        ["not-a-model"],
        {"DRY_RUN": "true", "HAIKU_PROFILE": EXISTING_PROFILE},
    )
    assert result.returncode == 1
    assert "unknown model 'not-a-model'" in result.stderr


def test_tb2_campaign_rejects_unknown_condition():
    result = _run_script(
        ["haiku", "not-a-condition"],
        {"DRY_RUN": "true", "HAIKU_PROFILE": EXISTING_PROFILE},
    )
    assert result.returncode == 1
    assert "Unknown condition: not-a-condition" in result.stderr


def test_tb2_campaign_accepts_no_drop_solved_option():
    result = _run_script(
        ["--no-drop-solved", "haiku"],
        {"DRY_RUN": "true", "HAIKU_PROFILE": EXISTING_PROFILE},
    )

    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "--no-drop-solved" in out
    assert "--drop-solved" not in out

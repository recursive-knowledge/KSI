from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

import pytest

from ksi.benchmarks.tb2_submission import (
    KsiTb2Attempt,
    SubmissionMetadata,
    attempt_is_unscored,
    build_harbor_config,
    build_harbor_result,
    build_submission,
    load_tb2_attempts_from_db,
    validate_submission,
)
from ksi.benchmarks.terminal_bench_2 import (
    TB2_VERIFIER_FAIL_CLOSED_STATUS,
    TB2_VERIFIER_MISSING_STATUS,
)


def _metadata() -> SubmissionMetadata:
    return SubmissionMetadata(
        agent_url="https://github.com/example/ksi",
        agent_display_name="KSI",
        agent_org_display_name="KSI Lab",
        model_name="claude-haiku-4-5",
        model_provider="anthropic",
        model_display_name="Claude Haiku 4.5",
        model_org_display_name="Anthropic",
        task_corpus_git_commit="53ff2b87",
    )


def _attempt(
    task_id: str,
    generation: int,
    *,
    reward: float | None = 0.0,
    attempt_no: int = 1,
    trial_status: str | None = None,
) -> KsiTb2Attempt:
    meta: dict = {
        "task_source": "terminal_bench_2",
        "reward": reward,
        "tool_trace": [{"tool_name": "tb2_shell"}] * 12,
        "token_usage": {"input_tokens": 1234, "output_tokens": 567},
        "verifier_stdout_tail": "PASSED 5 tests\n",
    }
    if trial_status is not None:
        meta["trial_status"] = trial_status
    return KsiTb2Attempt(
        task_id=task_id,
        generation=generation,
        attempt_no=attempt_no,
        reward=reward,
        started_at="2026-05-12 10:00:00",
        ended_at="2026-05-12 10:05:00",
        runtime_meta=meta,
        error_text="",
    )


def test_build_harbor_config_has_no_overrides() -> None:
    cfg = build_harbor_config(
        _attempt("adaptive-rejection-sampler", 1),
        _metadata(),
        trial_name="adaptive-rejection-sampler__abc1234",
        trials_dir="/tmp/job-1",
        job_id="job-uuid-1",
    )

    assert cfg["timeout_multiplier"] == 1.0
    assert cfg["agent_timeout_multiplier"] is None
    assert cfg["verifier_timeout_multiplier"] is None
    assert cfg["agent"]["override_timeout_sec"] is None
    assert cfg["agent"]["max_timeout_sec"] is None
    assert cfg["environment"]["override_cpus"] is None
    assert cfg["environment"]["override_memory_mb"] is None
    assert cfg["environment"]["override_storage_mb"] is None
    assert cfg["verifier"]["override_timeout_sec"] is None
    assert cfg["verifier"]["max_timeout_sec"] is None
    assert cfg["task"]["git_commit_id"] == "53ff2b87"
    assert cfg["task"]["git_url"] == "https://github.com/harbor-framework/terminal-bench-2.git"


def test_build_harbor_result_carries_reward_and_timing() -> None:
    metadata = _metadata()
    a = _attempt("dna-assembly", 3, reward=1.0)
    cfg = build_harbor_config(a, metadata, trial_name="dna-assembly__x", trials_dir="/x", job_id="j")
    res = build_harbor_result(a, metadata, cfg, trial_name="dna-assembly__x", trial_uri="file:///x")

    assert res["verifier_result"]["rewards"]["reward"] == 1.0
    assert res["task_name"] == "dna-assembly"
    assert res["task_id"]["path"] == "dna-assembly"
    assert res["task_id"]["git_commit_id"] == "53ff2b87"
    assert res["agent_info"]["model_info"] == {"name": "claude-haiku-4-5", "provider": "anthropic"}
    assert res["agent_result"]["n_input_tokens"] == 1234
    assert res["agent_result"]["n_output_tokens"] == 567
    assert res["agent_result"]["metadata"]["iterations"] == 12
    assert res["agent_result"]["metadata"]["success"] is True
    # SQLite-style "YYYY-MM-DD HH:MM:SS" -> normalized to ISO Z form
    assert res["started_at"] == "2026-05-12T10:00:00Z"
    assert res["finished_at"] == "2026-05-12T10:05:00Z"
    assert res["agent_result"]["metadata"]["duration_ms"] == 5 * 60 * 1000


def test_build_submission_rejects_fewer_than_five_generations(tmp_path: Path) -> None:
    attempts = [_attempt("t1", gen) for gen in range(1, 4)]
    with pytest.raises(ValueError, match=">=5 trials per task"):
        build_submission(attempts, _metadata(), out_dir=tmp_path)


def test_build_submission_writes_compliant_tree(tmp_path: Path) -> None:
    task_ids = ["task-a", "task-b", "task-c"]
    attempts = [
        _attempt(task, gen, reward=(1.0 if (gen + hash(task)) % 2 == 0 else 0.0))
        for gen in range(1, 6)
        for task in task_ids
    ]
    root = build_submission(attempts, _metadata(), out_dir=tmp_path)

    assert (root / "metadata.yaml").is_file()
    metadata_text = (root / "metadata.yaml").read_text(encoding="utf-8")
    assert 'agent_display_name: "KSI"' in metadata_text
    assert "model_name: claude-haiku-4-5" in metadata_text

    # 5 job folders (gens 1-5), each with 3 task trials
    job_folders = sorted(p for p in root.iterdir() if p.is_dir())
    assert len(job_folders) == 5
    for job in job_folders:
        trial_dirs = sorted(p for p in job.iterdir() if p.is_dir())
        assert len(trial_dirs) == 3
        for trial_dir in trial_dirs:
            assert (trial_dir / "config.json").is_file()
            assert (trial_dir / "result.json").is_file()
            assert (trial_dir / "verifier" / "reward.txt").is_file()
            assert (trial_dir / "verifier" / "test-stdout.txt").is_file()

    issues = validate_submission(root)
    assert issues == [], f"local validation failed: {issues}"


def test_validate_submission_flags_missing_trials(tmp_path: Path) -> None:
    # Only 3 trials per task — should fail
    attempts = [_attempt(task, gen) for gen in range(1, 6) for task in ("t1", "t2")]
    # Drop two generations of one task to drop below the 5-trial floor
    attempts = [a for a in attempts if not (a.task_id == "t1" and a.generation in (4, 5))]
    metadata = _metadata()
    by_gen: dict[int, list[KsiTb2Attempt]] = {}
    for a in attempts:
        by_gen.setdefault(a.generation, []).append(a)
    root = tmp_path / "submissions" / "terminal-bench" / "2.0" / "KSI__Claude-Haiku-4.5"
    root.mkdir(parents=True)
    (root / "metadata.yaml").write_text("agent_url: x\n", encoding="utf-8")

    for gen in sorted(by_gen):
        job = root / f"gen-{gen}"
        job.mkdir()
        for a in by_gen[gen]:
            tn = f"{a.task_id}__{uuid.uuid4().hex[:7]}"
            td = job / tn
            td.mkdir()
            cfg = build_harbor_config(a, metadata, trial_name=tn, trials_dir=str(job), job_id="j")
            res = build_harbor_result(a, metadata, cfg, trial_name=tn, trial_uri="file:///x")
            (td / "config.json").write_text(json.dumps(cfg))
            (td / "result.json").write_text(json.dumps(res))
            v = td / "verifier"
            v.mkdir()
            (v / "reward.txt").write_text("0\n")

    issues = validate_submission(root)
    assert any("'t1'" in i and "3 trials" in i for i in issues), issues


def test_load_tb2_attempts_filters_other_task_sources(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE runs (id INTEGER PRIMARY KEY);
        CREATE TABLE generations (id INTEGER PRIMARY KEY, run_id INTEGER, generation INTEGER);
        CREATE TABLE agents (id INTEGER PRIMARY KEY, run_id INTEGER, agent_id TEXT);
        CREATE TABLE tasks (id INTEGER PRIMARY KEY, run_id INTEGER, task_id TEXT);
        CREATE TABLE assignments (
            id INTEGER PRIMARY KEY,
            generation_id INTEGER,
            agent_ref INTEGER,
            task_ref INTEGER,
            started_at TEXT,
            ended_at TEXT
        );
        CREATE TABLE attempts (
            id INTEGER PRIMARY KEY,
            assignment_id INTEGER,
            attempt_no INTEGER,
            runtime_meta_json TEXT,
            error_text TEXT
        );
        INSERT INTO runs (id) VALUES (1);
        INSERT INTO generations (id, run_id, generation) VALUES (10, 1, 1);
        INSERT INTO agents (id, run_id, agent_id) VALUES (100, 1, 'a');
        INSERT INTO tasks (id, run_id, task_id) VALUES
            (1000, 1, 'tb2-task-1'),
            (1001, 1, 'arc-task');
        INSERT INTO assignments (id, generation_id, agent_ref, task_ref, started_at, ended_at) VALUES
            (1, 10, 100, 1000, '2026-05-12 10:00:00', '2026-05-12 10:05:00'),
            (2, 10, 100, 1001, '2026-05-12 10:10:00', '2026-05-12 10:15:00');
        INSERT INTO attempts (id, assignment_id, attempt_no, runtime_meta_json, error_text) VALUES
            (1, 1, 1, '{"task_source":"terminal_bench_2","reward":1.0}', ''),
            (2, 2, 1, '{"task_source":"arc","reward":1.0}', '');
        """
    )
    conn.commit()
    conn.close()

    attempts = load_tb2_attempts_from_db(db_path)
    assert len(attempts) == 1
    assert attempts[0].task_id == "tb2-task-1"
    assert attempts[0].reward == 1.0
    assert attempts[0].started_at == "2026-05-12 10:00:00"


@pytest.mark.parametrize("stored", ["Infinity", "-Infinity", "NaN"])
def test_load_tb2_attempts_drops_non_finite_stored_reward(tmp_path: Path, stored: str) -> None:
    """A legacy (pre-#1267) DB row may carry a non-finite reward in
    ``runtime_meta_json`` (``json.loads`` accepts Infinity/NaN). ``float`` doesn't
    raise on it, so without the isfinite gate ``inf`` would export as a false
    ``success`` (inf >= 1.0). It must load as unscored (``reward=None``) instead."""
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        f"""
        CREATE TABLE runs (id INTEGER PRIMARY KEY);
        CREATE TABLE generations (id INTEGER PRIMARY KEY, run_id INTEGER, generation INTEGER);
        CREATE TABLE agents (id INTEGER PRIMARY KEY, run_id INTEGER, agent_id TEXT);
        CREATE TABLE tasks (id INTEGER PRIMARY KEY, run_id INTEGER, task_id TEXT);
        CREATE TABLE assignments (
            id INTEGER PRIMARY KEY, generation_id INTEGER, agent_ref INTEGER,
            task_ref INTEGER, started_at TEXT, ended_at TEXT
        );
        CREATE TABLE attempts (
            id INTEGER PRIMARY KEY, assignment_id INTEGER, attempt_no INTEGER,
            runtime_meta_json TEXT, error_text TEXT
        );
        INSERT INTO runs (id) VALUES (1);
        INSERT INTO generations (id, run_id, generation) VALUES (10, 1, 1);
        INSERT INTO agents (id, run_id, agent_id) VALUES (100, 1, 'a');
        INSERT INTO tasks (id, run_id, task_id) VALUES (1000, 1, 'tb2-task-1');
        INSERT INTO assignments (id, generation_id, agent_ref, task_ref, started_at, ended_at) VALUES
            (1, 10, 100, 1000, '2026-05-12 10:00:00', '2026-05-12 10:05:00');
        INSERT INTO attempts (id, assignment_id, attempt_no, runtime_meta_json, error_text) VALUES
            (1, 1, 1, '{{"task_source":"terminal_bench_2","reward":{stored}}}', '');
        """
    )
    conn.commit()
    conn.close()

    attempts = load_tb2_attempts_from_db(db_path)
    assert len(attempts) == 1
    assert attempts[0].reward is None


def test_load_tb2_attempts_rejects_knowledge_db(tmp_path: Path) -> None:
    # A knowledge DB lacks the 'assignments' table — the loader should refuse it
    # with a message pointing at the runtime audit DB.
    db_path = tmp_path / "experiment_knowledge.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("CREATE TABLE attempts (id INTEGER);")
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="runtime"):
        load_tb2_attempts_from_db(db_path)


def _write_submission_tree(tmp_path: Path, *, mutate_config=None) -> Path:
    """Write a compliant 5-gen submission tree, optionally mutating each config.

    `mutate_config(cfg)` is called on every trial's config dict before it is
    written, so tests can inject Harbor-policy deviations.
    """
    task_ids = ["task-a", "task-b"]
    metadata = _metadata()
    by_gen: dict[int, list[KsiTb2Attempt]] = {}
    for gen in range(1, 6):
        for task in task_ids:
            by_gen.setdefault(gen, []).append(_attempt(task, gen))

    root = tmp_path / "submissions" / "terminal-bench" / "2.0" / "KSI__Claude-Haiku-4.5"
    root.mkdir(parents=True)
    (root / "metadata.yaml").write_text("agent_url: x\n", encoding="utf-8")

    for gen in sorted(by_gen):
        job = root / f"gen-{gen}"
        job.mkdir()
        for a in by_gen[gen]:
            tn = f"{a.task_id}__{uuid.uuid4().hex[:7]}"
            td = job / tn
            td.mkdir()
            cfg = build_harbor_config(a, metadata, trial_name=tn, trials_dir=str(job), job_id="j")
            res = build_harbor_result(a, metadata, cfg, trial_name=tn, trial_uri="file:///x")
            if mutate_config is not None:
                mutate_config(cfg)
            (td / "config.json").write_text(json.dumps(cfg))
            (td / "result.json").write_text(json.dumps(res))
            v = td / "verifier"
            v.mkdir()
            (v / "reward.txt").write_text("0\n")
    return root


def test_validate_submission_compliant_tree_has_no_issues(tmp_path: Path) -> None:
    root = _write_submission_tree(tmp_path)
    assert validate_submission(root) == []


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda cfg: cfg.update(timeout_multiplier=2.0), "timeout_multiplier != 1.0"),
        (lambda cfg: cfg["agent"].update(override_timeout_sec=900), "agent.override_timeout_sec is set"),
        (lambda cfg: cfg["agent"].update(max_timeout_sec=900), "agent.max_timeout_sec is set"),
        (lambda cfg: cfg["environment"].update(override_cpus=4), "environment.override_cpus is set"),
        (lambda cfg: cfg["environment"].update(override_memory_mb=8192), "environment.override_memory_mb is set"),
        (lambda cfg: cfg["environment"].update(override_storage_mb=20480), "environment.override_storage_mb is set"),
        (lambda cfg: cfg["verifier"].update(override_timeout_sec=600), "verifier.override_timeout_sec is set"),
    ],
)
def test_validate_submission_flags_deviations(tmp_path: Path, mutate, needle: str) -> None:
    root = _write_submission_tree(tmp_path, mutate_config=mutate)
    issues = validate_submission(root)
    assert any(needle in i for i in issues), issues


# --- #1222: unscored attempts must never be published as a fabricated 0.0 ---


@pytest.mark.parametrize(
    ("reward", "trial_status", "expected"),
    [
        (0.0, None, False),  # genuine verifier-ran failure -> scored
        (1.0, None, False),  # genuine pass -> scored
        (None, None, True),  # verifier produced no reward -> unscored
        (None, TB2_VERIFIER_MISSING_STATUS, True),  # verifier missing -> unscored
        (None, TB2_VERIFIER_FAIL_CLOSED_STATUS, True),  # strict fail-closed -> unscored
        (0.0, TB2_VERIFIER_FAIL_CLOSED_STATUS, True),  # status wins over a present 0.0
    ],
)
def test_attempt_is_unscored(reward, trial_status, expected: bool) -> None:
    a = _attempt("t1", 1, reward=reward, trial_status=trial_status)
    assert attempt_is_unscored(a) is expected


@pytest.mark.parametrize("trial_status", [None, TB2_VERIFIER_MISSING_STATUS, TB2_VERIFIER_FAIL_CLOSED_STATUS])
def test_build_harbor_result_refuses_unscored(trial_status) -> None:
    metadata = _metadata()
    a = _attempt("t1", 1, reward=None, trial_status=trial_status)
    cfg = build_harbor_config(a, metadata, trial_name="t1__x", trials_dir="/x", job_id="j")
    with pytest.raises(ValueError, match="unscored"):
        build_harbor_result(a, metadata, cfg, trial_name="t1__x", trial_uri="file:///x")


def test_build_harbor_result_keeps_genuine_zero() -> None:
    metadata = _metadata()
    a = _attempt("t1", 1, reward=0.0)
    cfg = build_harbor_config(a, metadata, trial_name="t1__x", trials_dir="/x", job_id="j")
    res = build_harbor_result(a, metadata, cfg, trial_name="t1__x", trial_uri="file:///x")
    assert res["verifier_result"]["rewards"]["reward"] == 0.0
    assert res["agent_result"]["metadata"]["success"] is False


def _read_rewards(root: Path) -> dict[str, list[str]]:
    """Map task_name -> list of reward.txt contents across all trials."""
    out: dict[str, list[str]] = {}
    for job in root.iterdir():
        if not job.is_dir():
            continue
        for trial in job.iterdir():
            if not trial.is_dir():
                continue
            task = trial.name.rsplit("__", 1)[0]
            reward_txt = trial / "verifier" / "reward.txt"
            if reward_txt.is_file():
                out.setdefault(task, []).append(reward_txt.read_text(encoding="utf-8").strip())
    return out


def test_build_submission_excludes_unscored_attempts(tmp_path: Path) -> None:
    # task-a has 5 genuine scored attempts (reward 0.0) across gens 1-5, plus one
    # UNSCORED attempt (reward=None) in gen 1. The unscored attempt must NOT be
    # written as a trial, and the genuine 0.0 attempts must export as reward "0".
    attempts: list[KsiTb2Attempt] = []
    for gen in range(1, 6):
        attempts.append(_attempt("task-a", gen, reward=0.0, attempt_no=1))
    attempts.append(_attempt("task-a", 1, reward=None, attempt_no=2))

    root = build_submission(attempts, _metadata(), out_dir=tmp_path)

    rewards = _read_rewards(root)
    # Exactly 5 genuine trials for task-a; the unscored attempt was dropped.
    assert len(rewards["task-a"]) == 5
    # Every written reward is the genuine 0.0 -> "0"; none is a fabricated zero
    # standing in for the unscored (None) attempt.
    assert rewards["task-a"] == ["0"] * 5
    # No result.json anywhere claims a verifier ran for the unscored attempt.
    assert validate_submission(root) == []


def test_build_submission_rejects_task_with_no_scored_trials_after_filtering(tmp_path: Path) -> None:
    # A mixed export with one valid task and one all-unscored task must not
    # silently drop the all-unscored task and then pass local validation.
    attempts: list[KsiTb2Attempt] = []
    for gen in range(1, 6):
        attempts.append(_attempt("scored-task", gen, reward=0.0))
        attempts.append(_attempt("all-unscored-task", gen, reward=None))

    with pytest.raises(ValueError, match="all-unscored-task.*0 scored trial"):
        build_submission(attempts, _metadata(), out_dir=tmp_path)


def test_build_submission_excludes_fail_closed_status(tmp_path: Path) -> None:
    # Fail-closed / missing-verifier statuses are unscored even if a reward
    # number is present in runtime_meta — they must be excluded from the export.
    attempts: list[KsiTb2Attempt] = []
    for gen in range(1, 6):
        attempts.append(_attempt("task-a", gen, reward=1.0, attempt_no=1))
    # A gen-1 attempt whose verifier refused to run (strict fail-closed).
    attempts.append(_attempt("task-a", 1, reward=0.0, attempt_no=2, trial_status=TB2_VERIFIER_FAIL_CLOSED_STATUS))
    # A gen-2 attempt where the verifier never produced a reward.
    attempts.append(_attempt("task-a", 2, reward=None, attempt_no=2, trial_status=TB2_VERIFIER_MISSING_STATUS))

    root = build_submission(attempts, _metadata(), out_dir=tmp_path)

    rewards = _read_rewards(root)
    # Only the 5 genuine passes are exported; both unscored attempts dropped.
    assert rewards["task-a"] == ["1"] * 5

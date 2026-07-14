"""Tests for #990a: provenance stamp (code commit, resolved model, scoring
mode) surfaced in ``_ksi_code_commit`` and the ``--output-json`` payload.

The full ``main()`` path is exercised the same way
``tests/distillation/test_removed_channel_env_guard.py`` exercises the
``GenerationalOrchestrator`` boundary and ``api.run``'s
``GenerationalOrchestrator`` substitution: swap in a fake orchestrator class
(the one heavy/networked seam) so the run completes instantly, while the real
argument parsing, task loading, provider-profile loading, and
``--output-json`` writing code paths in ``ksi.cli.main`` all execute for
real.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from dataclasses import fields

import pytest

from ksi.cli import _ksi_code_commit
from ksi.layout import PROJECT_ROOT
from ksi.models import GenerationConfig, TaskTrace

DEMO_DIR = PROJECT_ROOT / "examples" / "quickstart" / "arc_demo"


def test_generation_config_config_json_does_not_shift_seed_phase_positional_abi() -> None:
    names = [field.name for field in fields(GenerationConfig)]

    assert names[names.index("scoring_mode") + 1] == "per_task_forum_rounds"
    assert names.index("config_json") > names.index("holdout_task_ids")


def test_ksi_code_commit_resolves_head_sha_in_this_repo() -> None:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert sha  # sanity: this repo is a git checkout
    # A dirty checkout (uncommitted tracked changes, e.g. mid-development) gets a
    # +dirty suffix; a clean checkout (CI) gets the bare SHA. Compute the same.
    dirty = (
        subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=str(PROJECT_ROOT), capture_output=True).returncode == 1
    )
    expected = f"{sha}+dirty" if dirty else sha
    assert _ksi_code_commit(PROJECT_ROOT) == expected


def test_ksi_code_commit_returns_unknown_outside_a_git_repo(tmp_path) -> None:
    assert _ksi_code_commit(tmp_path) == "unknown"


def test_ksi_code_commit_marks_dirty_tracked_changes(tmp_path) -> None:
    import subprocess as sp

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        sp.run(["git", *args], cwd=str(repo), check=True, capture_output=True)

    git("init")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (repo / "f.py").write_text("x = 1\n", encoding="utf-8")
    git("add", "f.py")
    git("commit", "-m", "init")
    head = sp.run(["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True).stdout.strip()

    # Clean tree → bare SHA.
    assert _ksi_code_commit(repo) == head
    # Untracked file → still clean (ignored, like git describe --dirty).
    (repo / "artifact.txt").write_text("stray\n", encoding="utf-8")
    assert _ksi_code_commit(repo) == head
    # Modified tracked file → +dirty.
    (repo / "f.py").write_text("x = 2\n", encoding="utf-8")
    assert _ksi_code_commit(repo) == f"{head}+dirty"


class _FakeOrchestrator:
    """Stand-in for GenerationalOrchestrator that skips real task execution.

    Only the surface `ksi.cli.main` actually calls is implemented.
    """

    def __init__(self, **kwargs):
        self.config = kwargs["config"]
        self.accumulator = _FakeAccumulator()

    def set_improvement_strategy(self, _strategy) -> None:
        pass

    def run(self, tasks):
        return []

    def get_claim_debug_history(self):
        return []

    def holdout_solve_rate_by_generation(self):
        return {}

    def knowledge_phase_health_by_generation(self):
        return {}

    def knowledge_phase_health_measured(self):
        return False


class _FakeAccumulator:
    def to_dict(self):
        return {}


def test_output_json_payload_has_provenance_fields(tmp_path, monkeypatch) -> None:
    import ksi.cli as cli_module

    monkeypatch.setattr(cli_module, "GenerationalOrchestrator", _FakeOrchestrator)

    profile_path = tmp_path / ".env.test_profile"
    profile_path.write_text(
        "MODEL_PROVIDER=anthropic\nMODEL=claude-haiku-4-5-20251001\n"
        "MODEL_AUTH_MODE=api\nANTHROPIC_API_KEY=sk-ant-test-not-real\n"
    )
    output_path = tmp_path / "results.json"
    knowledge_db_path = tmp_path / "knowledge.sqlite"

    rc = cli_module.main(
        [
            "--task-source",
            "arc",
            "--tasks-path",
            str(DEMO_DIR),
            "--evaluator",
            "none",
            "--knowledge-db-path",
            str(knowledge_db_path),
            "--provider-profile",
            str(profile_path),
            "--output-json",
            str(output_path),
            "--generations",
            "1",
        ]
    )
    assert rc == 0

    payload = json.loads(output_path.read_text())
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = (
        subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=str(PROJECT_ROOT), capture_output=True).returncode == 1
    )
    assert payload["code_commit"] == (f"{sha}+dirty" if dirty else sha)
    assert payload["resolved_model"] == "anthropic/claude-haiku-4-5-20251001"
    assert "/" in payload["resolved_model"]
    assert payload["scoring_mode"] == "none"
    # Task 1 (#991) added arc_split alongside this; assert it's still present,
    # not removed/duplicated by this change.
    assert "arc_split" in payload
    # The final, post-run() write is the authoritative one; distinguishes it
    # from a mid-run incremental snapshot (see run_complete=False below).
    assert payload["run_complete"] is True


def test_output_json_resume_uses_write_once_db_provenance(tmp_path, monkeypatch) -> None:
    import ksi.cli as cli_module

    monkeypatch.setattr(cli_module, "GenerationalOrchestrator", _FakeOrchestrator)

    runtime_db_path = tmp_path / "runtime.sqlite"
    conn = sqlite3.connect(runtime_db_path)
    conn.execute(
        "CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, experiment TEXT NOT NULL UNIQUE, "
        "created_at TEXT DEFAULT (datetime('now')), code_commit TEXT, resolved_model TEXT, scoring_mode TEXT)"
    )
    conn.execute(
        "INSERT INTO runs (experiment, code_commit, resolved_model, scoring_mode) VALUES (?, ?, ?, ?)",
        ("resume-exp", "original-commit", "anthropic/original-model", "arc_session"),
    )
    conn.commit()
    conn.close()

    profile_path = tmp_path / ".env.test_profile"
    profile_path.write_text(
        "MODEL_PROVIDER=anthropic\nMODEL=claude-haiku-4-5-20251001\n"
        "MODEL_AUTH_MODE=api\nANTHROPIC_API_KEY=sk-ant-test-not-real\n"
    )
    output_path = tmp_path / "results.json"
    knowledge_db_path = tmp_path / "knowledge.sqlite"

    rc = cli_module.main(
        [
            "--task-source",
            "arc",
            "--tasks-path",
            str(DEMO_DIR),
            "--evaluator",
            "none",
            "--experiment-name",
            "resume-exp",
            "--resume",
            "--knowledge-db-path",
            str(knowledge_db_path),
            "--runtime-db-path",
            str(runtime_db_path),
            "--provider-profile",
            str(profile_path),
            "--output-json",
            str(output_path),
            "--generations",
            "1",
        ]
    )
    assert rc == 0

    payload = json.loads(output_path.read_text())
    assert payload["code_commit"] == "original-commit"
    assert payload["resolved_model"] == "anthropic/original-model"
    assert payload["scoring_mode"] == "arc_session"


def test_output_json_payload_records_task_map_identity(tmp_path, monkeypatch) -> None:
    import ksi.cli as cli_module

    monkeypatch.setattr(cli_module, "GenerationalOrchestrator", _FakeOrchestrator)

    profile_path = tmp_path / ".env.test_profile"
    profile_path.write_text(
        "MODEL_PROVIDER=anthropic\nMODEL=claude-haiku-4-5-20251001\n"
        "MODEL_AUTH_MODE=api\nANTHROPIC_API_KEY=sk-ant-test-not-real\n"
    )
    task_map = tmp_path / "arc1_eval_1_seed1_kt.json"
    task_map.write_text(
        json.dumps(
            {
                "benchmark": "arc1",
                "split": "evaluation",
                "seed": 1,
                "count": 1,
                "selection_name": "arc1_eval_1_seed1_kt",
                "source_repo": "fchollet/ARC-AGI",
                "source_branch": "master",
                "source_commit": "399030444e0ab0cc8b4e199870fb20b863846f34",
                "source_path": "data/evaluation",
                "selection_algorithm": "fixture",
                "tasks": [{"index": 1, "task_id": "demo_mirror"}],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "results.json"
    knowledge_db_path = tmp_path / "knowledge.sqlite"

    rc = cli_module.main(
        [
            "--task-source",
            "arc",
            "--tasks-path",
            str(DEMO_DIR),
            "--evaluator",
            "none",
            "--knowledge-db-path",
            str(knowledge_db_path),
            "--provider-profile",
            str(profile_path),
            "--task-ids-file",
            str(task_map),
            "--task-map-path",
            str(task_map),
            "--output-json",
            str(output_path),
            "--generations",
            "1",
        ]
    )
    assert rc == 0

    payload = json.loads(output_path.read_text())
    assert payload["args"]["task_map_path"] == str(task_map)
    assert payload["task_map"]["path"] == str(task_map)
    assert payload["task_map"]["selection_name"] == "arc1_eval_1_seed1_kt"
    assert payload["task_map"]["source_commit"] == "399030444e0ab0cc8b4e199870fb20b863846f34"
    assert payload["task_map"]["sha256"]
    assert payload["task_map"]["task_ids_count"] == 1
    assert payload["task_map"]["task_ids_sha256"]


def test_task_map_path_alone_selects_run_tasks(tmp_path, monkeypatch) -> None:
    import ksi.cli as cli_module

    captured: dict[str, list[str]] = {}

    class _CapturingFake(_FakeOrchestrator):
        def run(self, tasks):
            captured["task_ids"] = [task.id for task in tasks]
            return []

    monkeypatch.setattr(cli_module, "GenerationalOrchestrator", _CapturingFake)

    profile_path = tmp_path / ".env.test_profile"
    profile_path.write_text(
        "MODEL_PROVIDER=anthropic\nMODEL=claude-haiku-4-5-20251001\n"
        "MODEL_AUTH_MODE=api\nANTHROPIC_API_KEY=sk-ant-test-not-real\n"
    )
    task_map = tmp_path / "arc1_eval_1_seed1_kt.json"
    task_map.write_text(
        json.dumps(
            {
                "benchmark": "arc1",
                "split": "evaluation",
                "seed": 1,
                "count": 1,
                "selection_name": "arc1_eval_1_seed1_kt",
                "tasks": [{"index": 1, "task_id": "demo_mirror"}],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "results.json"

    rc = cli_module.main(
        [
            "--task-source",
            "arc",
            "--tasks-path",
            str(DEMO_DIR),
            "--evaluator",
            "none",
            "--knowledge-db-path",
            str(tmp_path / "knowledge.sqlite"),
            "--provider-profile",
            str(profile_path),
            "--task-map-path",
            str(task_map),
            "--output-json",
            str(output_path),
            "--generations",
            "1",
        ]
    )

    assert rc == 0
    assert captured["task_ids"] == ["demo_mirror"]
    payload = json.loads(output_path.read_text())
    assert payload["args"]["task_ids_file"] == str(task_map)
    assert payload["task_map"]["path"] == str(task_map)


def test_main_stamps_full_config_json_into_generation_config(tmp_path, monkeypatch) -> None:
    # The DB-bound GenerationConfig must carry the full effective launch config
    # as JSON, so the knowledge DB is self-describing (recoverable without the
    # gitignored --output-json sidecar).
    import ksi.cli as cli_module

    captured: dict = {}

    class _CapturingFake(_FakeOrchestrator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured["config"] = kwargs["config"]

    monkeypatch.setattr(cli_module, "GenerationalOrchestrator", _CapturingFake)

    profile_path = tmp_path / ".env.test_profile"
    profile_path.write_text(
        "MODEL_PROVIDER=anthropic\nMODEL=claude-haiku-4-5-20251001\n"
        "MODEL_AUTH_MODE=api\nANTHROPIC_API_KEY=sk-ant-test-not-real\n"
    )
    rc = cli_module.main(
        [
            "--task-source",
            "arc",
            "--tasks-path",
            str(DEMO_DIR),
            "--evaluator",
            "none",
            "--knowledge-db-path",
            str(tmp_path / "knowledge.sqlite"),
            "--provider-profile",
            str(profile_path),
            "--seed",
            "7",
            "--generations",
            "1",
        ]
    )
    assert rc == 0
    parsed = json.loads(captured["config"].config_json)
    assert parsed["task_source"] == "arc"
    assert parsed["seed"] == 7
    assert parsed["evaluator"] == "none"
    # No secret should ride along (API keys live in env / the profile, not args).
    assert "ANTHROPIC_API_KEY" not in parsed


class _FakeOrchestratorMidRunCrash:
    """Stand-in orchestrator that emits generation-1 persistence events (so
    the incremental --output-json snapshot has something to capture) and then
    raises, simulating a mid-run crash after generation 1 completes but
    before ``run()`` returns."""

    def __init__(self, **kwargs):
        self.config = kwargs["config"]
        self.persistence = kwargs["persistence"]
        self.accumulator = _FakeAccumulator()

    def set_improvement_strategy(self, _strategy) -> None:
        pass

    def run(self, tasks):
        trace = TaskTrace(generation=1, agent_id="agent-0", task_id=tasks[0].id if tasks else "task-0")
        self.persistence.on_task_trace(trace)
        self.persistence.on_generation_end(generation=1, agents=[])
        raise RuntimeError("simulated crash after generation 1")

    def get_claim_debug_history(self):
        return []

    def holdout_solve_rate_by_generation(self):
        return {}

    def knowledge_phase_health_by_generation(self):
        return {}

    def knowledge_phase_health_measured(self):
        return False


def test_output_json_snapshot_written_after_each_generation(tmp_path, monkeypatch) -> None:
    """Issue #990b: the incremental snapshot must persist generation-1
    traces even when the run itself crashes before reaching the final,
    post-``run()`` write."""
    import ksi.cli as cli_module

    monkeypatch.setattr(cli_module, "GenerationalOrchestrator", _FakeOrchestratorMidRunCrash)

    profile_path = tmp_path / ".env.test_profile"
    profile_path.write_text(
        "MODEL_PROVIDER=anthropic\nMODEL=claude-haiku-4-5-20251001\n"
        "MODEL_AUTH_MODE=api\nANTHROPIC_API_KEY=sk-ant-test-not-real\n"
    )
    output_path = tmp_path / "results.json"
    knowledge_db_path = tmp_path / "knowledge.sqlite"

    with pytest.raises(RuntimeError, match="simulated crash after generation 1"):
        cli_module.main(
            [
                "--task-source",
                "arc",
                "--tasks-path",
                str(DEMO_DIR),
                "--evaluator",
                "none",
                "--knowledge-db-path",
                str(knowledge_db_path),
                "--provider-profile",
                str(profile_path),
                "--output-json",
                str(output_path),
                "--generations",
                "1",
            ]
        )

    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert payload["num_traces"] >= 1
    # A count-only assertion would pass even if the snapshot silently wrote
    # an empty traces list while still bumping a stale num_traces; assert the
    # actual generation-1 trace content survived the crash.
    assert [t["task_id"] for t in payload["traces"]] == ["demo_mirror"]
    assert payload["traces"][0]["agent_id"] == "agent-0"
    assert payload["traces"][0]["generation"] == 1
    # This is a mid-run incremental snapshot, not the final authoritative
    # write — the payload must say so explicitly.
    assert payload["run_complete"] is False


class _FakeOrchestratorMultiGenCrash:
    """Stand-in orchestrator that completes two generations (each firing its
    own incremental snapshot) before crashing, so the snapshot's
    trace-accumulation across generations can be verified directly rather
    than assumed from a single-generation test."""

    def __init__(self, **kwargs):
        self.config = kwargs["config"]
        self.persistence = kwargs["persistence"]
        self.accumulator = _FakeAccumulator()

    def set_improvement_strategy(self, _strategy) -> None:
        pass

    def run(self, tasks):
        task_id = tasks[0].id if tasks else "task-0"
        trace_gen1 = TaskTrace(generation=1, agent_id="agent-0", task_id=task_id)
        self.persistence.on_task_trace(trace_gen1)
        self.persistence.on_generation_end(generation=1, agents=[])

        trace_gen2 = TaskTrace(generation=2, agent_id="agent-0", task_id=task_id)
        self.persistence.on_task_trace(trace_gen2)
        self.persistence.on_generation_end(generation=2, agents=[])

        raise RuntimeError("simulated crash after generation 2")

    def get_claim_debug_history(self):
        return []

    def holdout_solve_rate_by_generation(self):
        return {}

    def knowledge_phase_health_by_generation(self):
        return {}

    def knowledge_phase_health_measured(self):
        return False


def test_output_json_snapshot_accumulates_traces_across_generations(tmp_path, monkeypatch) -> None:
    """Issue #990b: the incremental snapshot fired after generation 2 must
    include generation 1's traces too, not just the latest generation's —
    ``CollectingPersistence.traces`` accumulates across the whole run, and
    each ``on_generation_snapshot`` call is passed the full list so far."""
    import ksi.cli as cli_module

    monkeypatch.setattr(cli_module, "GenerationalOrchestrator", _FakeOrchestratorMultiGenCrash)

    profile_path = tmp_path / ".env.test_profile"
    profile_path.write_text(
        "MODEL_PROVIDER=anthropic\nMODEL=claude-haiku-4-5-20251001\n"
        "MODEL_AUTH_MODE=api\nANTHROPIC_API_KEY=sk-ant-test-not-real\n"
    )
    output_path = tmp_path / "results.json"
    knowledge_db_path = tmp_path / "knowledge.sqlite"

    with pytest.raises(RuntimeError, match="simulated crash after generation 2"):
        cli_module.main(
            [
                "--task-source",
                "arc",
                "--tasks-path",
                str(DEMO_DIR),
                "--evaluator",
                "none",
                "--knowledge-db-path",
                str(knowledge_db_path),
                "--provider-profile",
                str(profile_path),
                "--output-json",
                str(output_path),
                "--generations",
                "2",
            ]
        )

    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert payload["num_traces"] == 2
    assert [t["generation"] for t in payload["traces"]] == [1, 2]
    assert payload["run_complete"] is False

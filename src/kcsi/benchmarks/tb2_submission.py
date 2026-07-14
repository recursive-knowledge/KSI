"""Build a Harbor leaderboard submission from a kcsi TB2 experiment.

Reads attempts from a kcsi runtime audit SQLite database, groups them
by generation (each generation becomes one trial-pass over all tasks), and
emits a directory tree matching the format expected by the Terminal-Bench 2
leaderboard at https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard.

This tool requires the runtime audit DB (the `<experiment>_runtime.sqlite`
produced via `--runtime-db-path` at experiment time) because per-attempt
`runtime_meta_json` lives only there. The knowledge DB is not supported.

Run:

    uv run python -m kcsi.benchmarks.tb2_submission \\
      --db runs/<experiment>/<experiment>_runtime.sqlite \\
      --out-dir submissions/ \\
      --agent-url https://github.com/recursive-knowledge/KCSI \\
      --agent-display-name "KCSI (knowledge bundle)" \\
      --agent-org "KCSI" \\
      --model-name claude-haiku-4-5 \\
      --model-provider anthropic \\
      --model-display-name "Claude Haiku 4.5" \\
      --model-org "Anthropic" \\
      --task-corpus-git-commit 53ff2b87

Validates locally against Harbor's documented submission rules (no timeout or
resource overrides, valid result.json per trial, >=5 trials per task).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..tasks.registry import resolve_source
from .terminal_bench_2 import TB2_VERIFIER_UNSCORED_STATUSES

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubmissionMetadata:
    agent_url: str
    agent_display_name: str
    agent_org_display_name: str
    model_name: str
    model_provider: str
    model_display_name: str
    model_org_display_name: str
    task_corpus_git_commit: str
    task_corpus_git_url: str = "https://github.com/harbor-framework/terminal-bench-2.git"
    agent_import_path: str = "kcsi.runtime.terminal_bench_2:TerminalBench2Executor"
    agent_name_short: str = "kcsi"
    agent_version: str = "unknown"


@dataclass(frozen=True)
class KcsiTb2Attempt:
    task_id: str
    generation: int
    attempt_no: int
    reward: float | None
    started_at: str | None
    ended_at: str | None
    runtime_meta: dict[str, Any]
    error_text: str


def load_tb2_attempts_from_db(db_path: Path) -> list[KcsiTb2Attempt]:
    """Read terminal_bench_2 attempts from a kcsi runtime audit DB.

    Requires the runtime DB schema (assignments/generations/tasks tables and
    the per-attempt runtime_meta_json column). The knowledge DB is not
    supported; passing one raises ValueError.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        has_assignments = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='assignments'"
        ).fetchone()
        if has_assignments is None:
            raise ValueError(
                f"DB at {db_path} has no 'assignments' table — this looks like a "
                f"knowledge DB; pass the <experiment>_runtime.sqlite (runtime audit "
                f"DB) instead."
            )
        rows = conn.execute(
            """
            SELECT
                a.attempt_no,
                a.runtime_meta_json,
                a.error_text,
                asg.started_at AS asg_started_at,
                asg.ended_at AS asg_ended_at,
                g.generation,
                t.task_id
            FROM attempts a
            JOIN assignments asg ON a.assignment_id = asg.id
            JOIN generations g ON asg.generation_id = g.id
            JOIN tasks t ON asg.task_ref = t.id
            ORDER BY g.generation, t.task_id, a.attempt_no
            """
        ).fetchall()
    finally:
        conn.close()

    attempts: list[KcsiTb2Attempt] = []
    for r in rows:
        try:
            meta = json.loads(r["runtime_meta_json"] or "{}")
        except json.JSONDecodeError:
            meta = {}
        _spec = resolve_source(meta.get("task_source"))
        if _spec is None or not _spec.delegates_runtime:
            continue
        reward = meta.get("reward")
        try:
            reward_f = float(reward) if reward is not None else None
        except (TypeError, ValueError):
            reward_f = None
        # Defense-in-depth: runtime_meta_json is read from the DB and may carry a
        # legacy non-finite reward. `float("inf")` doesn't raise, so an
        # ungated inf would export as a false `success` (inf >= 1.0). Treat a
        # non-finite stored reward as unscored (None), same as the source-side gate.
        if reward_f is not None and not math.isfinite(reward_f):
            reward_f = None
        attempts.append(
            KcsiTb2Attempt(
                task_id=r["task_id"],
                generation=int(r["generation"]),
                attempt_no=int(r["attempt_no"]),
                reward=reward_f,
                started_at=r["asg_started_at"],
                ended_at=r["asg_ended_at"],
                runtime_meta=meta,
                error_text=r["error_text"] or "",
            )
        )
    return attempts


def attempt_is_unscored(attempt: KcsiTb2Attempt) -> bool:
    """Return True when the attempt has no trustworthy verifier reward.

    A ``reward`` of ``None`` (the verifier never produced one) or a
    ``trial_status`` in :data:`TB2_VERIFIER_UNSCORED_STATUSES` (verifier missing
    / strict-mode fail-closed refusal) means the verifier did not
    run to completion, so there is *no* genuine score. These attempts must not
    be published as a fabricated ``0.0`` verifier-ran failure.
    """
    if attempt.reward is None:
        return True
    status = str(attempt.runtime_meta.get("trial_status") or "").strip()
    return status in TB2_VERIFIER_UNSCORED_STATUSES


def build_harbor_config(
    attempt: KcsiTb2Attempt,
    metadata: SubmissionMetadata,
    *,
    trial_name: str,
    trials_dir: str,
    job_id: str,
) -> dict[str, Any]:
    """Build the Harbor trial ``config.json`` payload for one attempt.

    Emits a compliant config with no timeout or resource overrides, matching
    the schema the Terminal-Bench 2 leaderboard validator expects.
    """
    return {
        "task": {
            "path": attempt.task_id,
            "git_url": metadata.task_corpus_git_url,
            "git_commit_id": metadata.task_corpus_git_commit,
            "overwrite": False,
            "download_dir": None,
            "source": "terminal-bench",
        },
        "trial_name": trial_name,
        "trials_dir": trials_dir,
        "timeout_multiplier": 1.0,
        "agent_timeout_multiplier": None,
        "verifier_timeout_multiplier": None,
        "agent_setup_timeout_multiplier": None,
        "environment_build_timeout_multiplier": None,
        "agent": {
            "name": None,
            "import_path": metadata.agent_import_path,
            "model_name": f"{metadata.model_provider}/{metadata.model_name}",
            "override_timeout_sec": None,
            "override_setup_timeout_sec": None,
            "max_timeout_sec": None,
            "kwargs": {},
            "env": {},
        },
        "environment": {
            "type": "docker",
            "import_path": None,
            "force_build": False,
            "delete": True,
            "override_cpus": None,
            "override_memory_mb": None,
            "override_storage_mb": None,
            "override_gpus": None,
            "suppress_override_warnings": False,
            "kwargs": {},
        },
        "verifier": {
            "override_timeout_sec": None,
            "max_timeout_sec": None,
            "disable": False,
        },
        "artifacts": [],
        "job_id": job_id,
    }


def _iso(ts: str | None) -> str:
    """Return an ISO-8601 UTC timestamp, falling back to epoch zero."""
    if not ts:
        return "1970-01-01T00:00:00Z"
    # SQLite datetime('now') format: "2026-05-12 11:22:33"
    if "T" not in ts and " " in ts:
        ts = ts.replace(" ", "T")
    if not ts.endswith("Z") and "+" not in ts:
        ts = ts + "Z"
    return ts


def _duration_ms(started_at_iso: str, finished_at_iso: str) -> int | None:
    try:
        s = datetime.fromisoformat(started_at_iso.replace("Z", "+00:00"))
        e = datetime.fromisoformat(finished_at_iso.replace("Z", "+00:00"))
        return max(0, int((e - s).total_seconds() * 1000))
    except (ValueError, AttributeError):
        return None


def build_harbor_result(
    attempt: KcsiTb2Attempt,
    metadata: SubmissionMetadata,
    config: dict[str, Any],
    *,
    trial_name: str,
    trial_uri: str,
) -> dict[str, Any]:
    """Build the Harbor trial ``result.json`` payload for one attempt.

    Carries the verifier reward, token usage, and timing derived from the
    attempt's runtime_meta into the leaderboard's result schema.

    Refuses to build a result for an unscored attempt (``reward is None`` or a
    fail-closed / missing-verifier ``trial_status``): coercing that to ``0.0``
    would publish an infra failure as a genuine verifier-ran zero.
    Filter such attempts out (``build_submission`` does) before calling this.
    """
    if attempt_is_unscored(attempt):
        raise ValueError(
            f"cannot build a Harbor result for unscored attempt "
            f"task={attempt.task_id!r} gen={attempt.generation} "
            f"attempt={attempt.attempt_no} (reward={attempt.reward!r}, "
            f"trial_status={attempt.runtime_meta.get('trial_status')!r}): the "
            f"verifier produced no trustworthy reward. Filter unscored attempts "
            f"before export rather than fabricating a 0.0."
        )
    reward = attempt.reward
    # `attempt_is_unscored` above returns True when reward is None and we raised,
    # so reward is a genuine float here; narrow for the type checker.
    assert reward is not None
    started_at = _iso(attempt.started_at)
    finished_at = _iso(attempt.ended_at) if attempt.ended_at else started_at

    token_usage = attempt.runtime_meta.get("token_usage") or {}
    n_input = token_usage.get("input_tokens")
    n_output = token_usage.get("output_tokens")
    n_cache = token_usage.get("cache_read_tokens") or token_usage.get("cache_tokens")
    iterations = len(attempt.runtime_meta.get("tool_trace") or [])

    return {
        "id": str(uuid.uuid4()),
        "task_name": attempt.task_id,
        "trial_name": trial_name,
        "trial_uri": trial_uri,
        "task_id": {
            "git_url": metadata.task_corpus_git_url,
            "git_commit_id": metadata.task_corpus_git_commit,
            "path": attempt.task_id,
        },
        "source": "terminal-bench",
        "task_checksum": "",
        "config": config,
        "agent_info": {
            "name": metadata.agent_name_short,
            "version": metadata.agent_version,
            "model_info": {
                "name": metadata.model_name,
                "provider": metadata.model_provider,
            },
        },
        "agent_result": {
            "n_input_tokens": n_input,
            "n_cache_tokens": n_cache,
            "n_output_tokens": n_output,
            "cost_usd": None,
            "rollout_details": None,
            "metadata": {
                "model": metadata.model_name,
                "started_at": started_at,
                "duration_ms": _duration_ms(started_at, finished_at),
                "iterations": iterations,
                "totalTokens": (n_input or 0) + (n_output or 0),
                "success": reward >= 1.0,
            },
        },
        "verifier_result": {
            "rewards": {"reward": float(reward)},
        },
        "exception_info": attempt.error_text or None,
        "started_at": started_at,
        "finished_at": finished_at,
        "environment_setup": {
            "started_at": started_at,
            "finished_at": started_at,
        },
        "agent_setup": {
            "started_at": started_at,
            "finished_at": started_at,
        },
        "agent_execution": {
            "started_at": started_at,
            "finished_at": finished_at,
        },
        "verifier": {
            "started_at": finished_at,
            "finished_at": finished_at,
        },
    }


def _sanitize_slug(s: str) -> str:
    return s.replace(" ", "-").replace("/", "-")


def _write_metadata_yaml(path: Path, metadata: SubmissionMetadata) -> None:
    lines = [
        f"agent_url: {metadata.agent_url}",
        f'agent_display_name: "{metadata.agent_display_name}"',
        f'agent_org_display_name: "{metadata.agent_org_display_name}"',
        "",
        "models:",
        f"  - model_name: {metadata.model_name}",
        f"    model_provider: {metadata.model_provider}",
        f'    model_display_name: "{metadata.model_display_name}"',
        f'    model_org_display_name: "{metadata.model_org_display_name}"',
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_submission(
    attempts: list[KcsiTb2Attempt],
    metadata: SubmissionMetadata,
    *,
    out_dir: Path,
    generations: list[int] | None = None,
) -> Path:
    """Write a Harbor-compliant submission tree under out_dir.

    Each generation in `generations` becomes one job-folder containing one
    trial per task. Harbor's validator requires >=5 trials per task, so
    >=5 generations must be provided. If `generations` is None, all
    generations present in `attempts` are used.

    Returns the path to the submission root (the per-agent__model directory).

    Unscored attempts (verifier never ran / strict-mode fail-closed refusal —
    see :func:`attempt_is_unscored`) are skipped with a warning: they have no
    genuine verifier reward and must not be published as a fabricated ``0.0``.
    If excluding them drops any selected task below Harbor's 5-trial
    floor, the export fails instead of silently omitting that task.
    """
    if generations is None:
        generations = sorted({a.generation for a in attempts})
    if not generations:
        raise ValueError("no generations selected for submission")
    if len(generations) < 5:
        raise ValueError(
            f"Harbor validator requires >=5 trials per task; "
            f"only {len(generations)} generations supplied: {generations}."
        )
    selected_generations = set(generations)

    by_gen: dict[int, list[KcsiTb2Attempt]] = {}
    selected_tasks: set[str] = set()
    scored_trials_per_task: dict[str, int] = {}
    skipped_unscored = 0
    for a in attempts:
        if a.generation not in selected_generations:
            continue
        selected_tasks.add(a.task_id)
        if attempt_is_unscored(a):
            skipped_unscored += 1
            log.warning(
                "tb2 submission: skipping unscored attempt task=%s gen=%s attempt=%s "
                "(reward=%r, trial_status=%r) — verifier produced no trustworthy "
                "reward; not exporting a fabricated 0.0",
                a.task_id,
                a.generation,
                a.attempt_no,
                a.reward,
                a.runtime_meta.get("trial_status"),
            )
            continue
        by_gen.setdefault(a.generation, []).append(a)
        scored_trials_per_task[a.task_id] = scored_trials_per_task.get(a.task_id, 0) + 1
    if skipped_unscored:
        log.warning(
            "tb2 submission: excluded %d unscored attempt(s) from the Harbor export "
            "(no genuine verifier reward); refusing export if any task falls below "
            "the 5-trial floor",
            skipped_unscored,
        )

    undersupplied_tasks = [
        (task_id, scored_trials_per_task.get(task_id, 0))
        for task_id in sorted(selected_tasks)
        if scored_trials_per_task.get(task_id, 0) < 5
    ]
    if undersupplied_tasks:
        detail = ", ".join(f"{task_id!r}: {count} scored trial(s)" for task_id, count in undersupplied_tasks)
        raise ValueError(
            "Harbor validator requires >=5 trials per task after excluding unscored attempts; "
            f"undersupplied task(s): {detail}."
        )

    missing = [g for g in generations if g not in by_gen]
    if missing:
        raise ValueError(f"no attempts found for generations: {missing}")

    slug = f"{_sanitize_slug(metadata.agent_display_name)}__{_sanitize_slug(metadata.model_display_name)}"
    root = out_dir / "submissions" / "terminal-bench" / "2.0" / slug
    root.mkdir(parents=True, exist_ok=True)
    _write_metadata_yaml(root / "metadata.yaml", metadata)

    for gen in generations:
        job_id = str(uuid.uuid4())
        job_folder = root / f"kcsi-tb2-gen-{gen}-{job_id[:8]}"
        job_folder.mkdir()
        trials_dir = str(job_folder.resolve())

        for a in by_gen[gen]:
            trial_hash = uuid.uuid4().hex[:7]
            trial_name = f"{a.task_id}__{trial_hash}"
            trial_dir = job_folder / trial_name
            trial_dir.mkdir()
            trial_uri = f"file://{trial_dir.resolve()}"

            config = build_harbor_config(a, metadata, trial_name=trial_name, trials_dir=trials_dir, job_id=job_id)
            result = build_harbor_result(a, metadata, config, trial_name=trial_name, trial_uri=trial_uri)

            (trial_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            (trial_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

            verifier_dir = trial_dir / "verifier"
            verifier_dir.mkdir()
            # Unscored attempts were filtered out above, so a.reward is a genuine
            # verifier-produced float here — never a fabricated 0.0.
            reward_value = a.reward
            reward_str = f"{int(reward_value)}" if reward_value in (0.0, 1.0) else f"{reward_value}"
            (verifier_dir / "reward.txt").write_text(reward_str + "\n", encoding="utf-8")
            verifier_stdout = str(a.runtime_meta.get("verifier_stdout_tail") or "")
            (verifier_dir / "test-stdout.txt").write_text(verifier_stdout, encoding="utf-8")

    return root


def validate_submission(root: Path) -> list[str]:
    """Locally check Harbor's documented submission rules.

    Returns a list of human-readable issues; empty means the submission
    looks valid per the rules documented at
    https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard.
    """
    issues: list[str] = []

    if not (root / "metadata.yaml").is_file():
        issues.append("metadata.yaml missing at submission root")

    job_folders = sorted(p for p in root.iterdir() if p.is_dir())
    if not job_folders:
        issues.append("no job folders found under submission root")
        return issues

    trials_per_task: dict[str, int] = {}
    for job in job_folders:
        for trial_dir in sorted(job.iterdir()):
            if not trial_dir.is_dir():
                continue
            task_name = trial_dir.name.rsplit("__", 1)[0]
            trials_per_task[task_name] = trials_per_task.get(task_name, 0) + 1

            for needed in ("config.json", "result.json", "verifier/reward.txt"):
                if not (trial_dir / needed).exists():
                    issues.append(f"{trial_dir.relative_to(root)}: missing {needed}")

            cfg_path = trial_dir / "config.json"
            if cfg_path.is_file():
                try:
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    issues.append(f"{cfg_path.relative_to(root)}: invalid JSON ({exc})")
                    continue
                if cfg.get("timeout_multiplier") != 1.0:
                    issues.append(f"{trial_dir.relative_to(root)}: timeout_multiplier != 1.0")
                for ag_field in ("override_timeout_sec", "max_timeout_sec"):
                    if cfg.get("agent", {}).get(ag_field) is not None:
                        issues.append(f"{trial_dir.relative_to(root)}: agent.{ag_field} is set")
                for env_field in ("override_cpus", "override_memory_mb", "override_storage_mb"):
                    if cfg.get("environment", {}).get(env_field) is not None:
                        issues.append(f"{trial_dir.relative_to(root)}: environment.{env_field} is set")
                if cfg.get("verifier", {}).get("override_timeout_sec") is not None:
                    issues.append(f"{trial_dir.relative_to(root)}: verifier.override_timeout_sec is set")

            res_path = trial_dir / "result.json"
            if res_path.is_file():
                try:
                    res = json.loads(res_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    issues.append(f"{res_path.relative_to(root)}: invalid JSON ({exc})")
                    continue
                if not isinstance(res.get("verifier_result", {}).get("rewards", {}).get("reward"), (int, float)):
                    issues.append(f"{res_path.relative_to(root)}: verifier_result.rewards.reward not numeric")

    for task, count in sorted(trials_per_task.items()):
        if count < 5:
            issues.append(f"task {task!r}: only {count} trials present (need >=5)")

    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--db",
        required=True,
        type=Path,
        help=(
            "Path to the kcsi runtime audit DB (<experiment>_runtime.sqlite). "
            "The knowledge DB is not supported (it lacks runtime_meta)."
        ),
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output root (will create submissions/ subtree)")
    parser.add_argument(
        "--generations", help="Comma-separated generation numbers, e.g. '6,7,8,9,10'. Default: all available."
    )
    parser.add_argument("--agent-url", required=True)
    parser.add_argument("--agent-display-name", required=True)
    parser.add_argument("--agent-org", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-provider", required=True)
    parser.add_argument("--model-display-name", required=True)
    parser.add_argument("--model-org", required=True)
    parser.add_argument("--task-corpus-git-url", default="https://github.com/harbor-framework/terminal-bench-2.git")
    parser.add_argument("--task-corpus-git-commit", required=True)
    parser.add_argument("--agent-import-path", default="kcsi.runtime.terminal_bench_2:TerminalBench2Executor")
    parser.add_argument("--agent-name-short", default="kcsi")
    parser.add_argument("--agent-version", default="unknown")
    args = parser.parse_args(argv)

    metadata = SubmissionMetadata(
        agent_url=args.agent_url,
        agent_display_name=args.agent_display_name,
        agent_org_display_name=args.agent_org,
        model_name=args.model_name,
        model_provider=args.model_provider,
        model_display_name=args.model_display_name,
        model_org_display_name=args.model_org,
        task_corpus_git_url=args.task_corpus_git_url,
        task_corpus_git_commit=args.task_corpus_git_commit,
        agent_import_path=args.agent_import_path,
        agent_name_short=args.agent_name_short,
        agent_version=args.agent_version,
    )

    attempts = load_tb2_attempts_from_db(args.db)
    if not attempts:
        print(f"no terminal_bench_2 attempts found in {args.db}", file=sys.stderr)
        return 2

    generations: list[int] | None = None
    if args.generations:
        generations = [int(g) for g in args.generations.split(",")]

    root = build_submission(attempts, metadata, out_dir=args.out_dir, generations=generations)

    issues = validate_submission(root)
    print(f"wrote submission to {root}", file=sys.stderr)
    if issues:
        print("local validation issues:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        return 1
    print("local validation passed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Compatibility wrapper for SWE-bench Pro evaluation.

Baseline adapters call this top-level script with dashed argument names. The
external evaluator checkout expects underscore argument names, so this wrapper
normalizes the CLI and delegates to the official runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from ksi.benchmarks.swebench_pro_external import (
    DEFAULT_EVALUATOR_RELATIVE,
    EVALUATOR_REVISION,
    REVISION_MARKER,
    SETUP_COMMAND,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_ROOT = REPO_ROOT / DEFAULT_EVALUATOR_RELATIVE
EXPECTED_EVAL_REVISION = EVALUATOR_REVISION


def _git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(path),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _revision_marker(path: Path) -> str | None:
    marker = path / REVISION_MARKER
    if not marker.is_file():
        return None
    value = marker.read_text(encoding="utf-8").strip()
    return value or None


def _verify_eval_root_revision(eval_root: Path) -> None:
    current = _git_head(eval_root)
    if current is not None:
        if current != EXPECTED_EVAL_REVISION:
            raise RuntimeError(
                f"SWE-bench Pro evaluator at {eval_root} is pinned to {current}, "
                f"expected {EXPECTED_EVAL_REVISION}. Reinstall it with: {SETUP_COMMAND}"
            )
        return

    marker = _revision_marker(eval_root)
    if marker != EXPECTED_EVAL_REVISION:
        marker_note = f" found marker {marker}" if marker else " no revision marker found"
        raise RuntimeError(
            f"Cannot verify SWE-bench Pro evaluator revision at {eval_root}:"
            f"{marker_note}; expected {EXPECTED_EVAL_REVISION}. Reinstall it with: {SETUP_COMMAND}"
        )


def _resolve_eval_root(source_dir: Path) -> Path:
    source_dir = source_dir.expanduser().resolve()
    candidates = [source_dir, DEFAULT_EVAL_ROOT.resolve()]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "swe_bench_pro_eval.py").is_file():
            _verify_eval_root_revision(candidate)
            return candidate
    searched = ", ".join(str(path) for path in seen)
    raise FileNotFoundError(
        "Could not find SWE-bench Pro evaluator script swe_bench_pro_eval.py. "
        f"Searched: {searched}. Install it with: {SETUP_COMMAND}"
    )


def _resolve_scripts_dir(scripts_dir: Path | None, eval_root: Path) -> Path:
    primary = (scripts_dir or eval_root / "run_scripts").expanduser().resolve()
    if primary.is_dir():
        return primary
    fallback = (eval_root / "run_scripts").resolve()
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError(f"Could not find SWE-bench Pro run_scripts at {primary} or {fallback}")


def _load_raw_sample_ids(raw_sample_path: Path) -> set[str]:
    if raw_sample_path.suffix.lower() == ".jsonl":
        ids = set()
        with raw_sample_path.open(encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                row = json.loads(payload)
                if isinstance(row, dict):
                    instance_id = str(row.get("instance_id") or "").strip()
                    if instance_id:
                        ids.add(instance_id)
        return ids

    with raw_sample_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "instance_id" not in reader.fieldnames:
            raise ValueError(f"raw sample file must include an instance_id column: {raw_sample_path}")
        return {
            str(row.get("instance_id") or "").strip() for row in reader if str(row.get("instance_id") or "").strip()
        }


def _load_patch_instance_ids(patch_path: Path) -> set[str]:
    with patch_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"patch file must contain a JSON list of patch entries: {patch_path}")
    return {
        str(entry.get("instance_id") or "").strip()
        for entry in payload
        if isinstance(entry, dict) and str(entry.get("instance_id") or "").strip()
    }


def _ensure_patch_bundle_matches_raw_sample(raw_sample_path: Path, patch_path: Path) -> None:
    raw_sample_ids = _load_raw_sample_ids(raw_sample_path)
    patch_ids = _load_patch_instance_ids(patch_path)
    if not raw_sample_ids:
        raise ValueError(f"raw sample file has no instance_id entries: {raw_sample_path}")
    if not patch_ids:
        raise ValueError(f"patch file has no entries with instance_id: {patch_path}")
    if raw_sample_ids.isdisjoint(patch_ids):
        raw_preview = ", ".join(sorted(raw_sample_ids)[:5])
        patch_preview = ", ".join(sorted(patch_ids)[:5])
        raise ValueError(
            "patch file has no entries matching raw sample instance_id values: "
            f"patch_path={patch_path}, raw_sample_path={raw_sample_path}, "
            f"patch_ids=[{patch_preview}], raw_sample_ids=[{raw_preview}]"
        )


def build_eval_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    eval_root = _resolve_eval_root(args.source_dir)
    scripts_dir = _resolve_scripts_dir(args.scripts_dir, eval_root)
    raw_sample_path = args.raw_sample_path.expanduser().resolve()
    patch_path = args.patch_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not raw_sample_path.is_file():
        raise FileNotFoundError(f"raw sample file not found: {raw_sample_path}")
    if not patch_path.is_file():
        raise FileNotFoundError(f"patch file not found: {patch_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(eval_root / "swe_bench_pro_eval.py"),
        "--raw_sample_path",
        str(raw_sample_path),
        "--patch_path",
        str(patch_path),
        "--output_dir",
        str(output_dir),
        "--dockerhub_username",
        args.dockerhub_username,
        "--scripts_dir",
        str(scripts_dir),
        "--num_workers",
        str(args.num_workers),
    ]
    if args.use_local_docker:
        cmd.append("--use_local_docker")
    if args.redo:
        cmd.append("--redo")
    if args.block_network:
        cmd.append("--block_network")
    if args.docker_platform:
        cmd.extend(["--docker_platform", args.docker_platform])
    return cmd, eval_root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-path", "--patch_path", dest="patch_path", type=Path, required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, required=True)
    parser.add_argument("--source-dir", "--source_dir", dest="source_dir", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--raw-sample-path", "--raw_sample_path", dest="raw_sample_path", type=Path, required=True)
    parser.add_argument(
        "--scripts-dir",
        "--scripts_dir",
        dest="scripts_dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--dockerhub-username", "--dockerhub_username", dest="dockerhub_username", default="jefzda")
    parser.add_argument("--use-local-docker", "--use_local_docker", dest="use_local_docker", action="store_true")
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=50)
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--block-network", "--block_network", dest="block_network", action="store_true")
    parser.add_argument("--docker-platform", "--docker_platform", dest="docker_platform", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cmd, eval_root = build_eval_command(args)
    _ensure_patch_bundle_matches_raw_sample(
        args.raw_sample_path.expanduser().resolve(),
        args.patch_path.expanduser().resolve(),
    )
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{eval_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(eval_root)
    subprocess.run(cmd, cwd=eval_root, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

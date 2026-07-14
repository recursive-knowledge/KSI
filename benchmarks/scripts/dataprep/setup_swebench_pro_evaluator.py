#!/usr/bin/env python3
"""Install and verify the external SWE-bench Pro evaluator checkout.

The ksi repository keeps SWE-bench Pro dataset/task-map artifacts in tree,
but the official evaluator is an external benchmark oracle. This script pins
that checkout, verifies the per-instance assets needed by the local dataset,
and creates compatibility links for older baseline adapters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ksi.benchmarks.swebench_pro_external import (
    DEFAULT_DATASET_RELATIVE,
    DEFAULT_EVALUATOR_RELATIVE,
    EVALUATOR_PATCHES_RELATIVE,
    EVALUATOR_REPO_URL,
    EVALUATOR_REVISION,
    PATCH_STATE_MARKER,
    REVISION_MARKER,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPO_URL = EVALUATOR_REPO_URL
DEFAULT_REVISION = EVALUATOR_REVISION
DEFAULT_EVALUATOR_DIR = REPO_ROOT / DEFAULT_EVALUATOR_RELATIVE
DEFAULT_DATASET_PATH = REPO_ROOT / DEFAULT_DATASET_RELATIVE
KSI_EVALUATOR_PATCHES_DIR = REPO_ROOT / EVALUATOR_PATCHES_RELATIVE

COMPATIBILITY_LINKS = (REPO_ROOT / "benchmarks" / "swebench_pro" / "source",)


@dataclass(frozen=True)
class VerificationSummary:
    instance_count: int
    run_script_count: int
    extra_run_script_count: int


def _git(args: list[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        return _git(["rev-parse", "HEAD"], cwd=path)
    except subprocess.CalledProcessError:
        return None


def _git_is_dirty(path: Path) -> bool:
    if not (path / ".git").exists():
        return False
    return bool(_git(["status", "--porcelain"], cwd=path))


def _revision_marker_path(path: Path) -> Path:
    return path / REVISION_MARKER


def _read_revision_marker(path: Path) -> str | None:
    marker = _revision_marker_path(path)
    if not marker.is_file():
        return None
    value = marker.read_text(encoding="utf-8").strip()
    return value or None


def _write_revision_marker(path: Path, revision: str) -> None:
    _revision_marker_path(path).write_text(revision + "\n", encoding="utf-8")


def verify_evaluator_revision(*, evaluator_dir: Path, revision: str) -> None:
    current = _git_head(evaluator_dir)
    if current is not None:
        if current != revision:
            raise RuntimeError(
                f"SWE-bench Pro evaluator at {evaluator_dir} is pinned to {current}, expected {revision}"
            )
        return

    marker = _read_revision_marker(evaluator_dir)
    if marker != revision:
        marker_note = f" found marker {marker}" if marker else " no revision marker found"
        raise RuntimeError(
            f"Cannot verify SWE-bench Pro evaluator revision at {evaluator_dir}:"
            f"{marker_note}; expected {revision}. Reinstall it with this setup script."
        )


def ensure_evaluator_checkout(*, evaluator_dir: Path, repo_url: str, revision: str) -> None:
    if evaluator_dir.exists():
        if not (evaluator_dir / ".git").exists():
            if (evaluator_dir / "swe_bench_pro_eval.py").is_file():
                verify_evaluator_revision(evaluator_dir=evaluator_dir, revision=revision)
                print(f"Using existing non-git evaluator checkout: {evaluator_dir}")
                return
            raise FileExistsError(f"{evaluator_dir} exists but is not a SWE-bench Pro evaluator checkout")

        current = _git_head(evaluator_dir)
        if current == revision:
            _write_revision_marker(evaluator_dir, revision)
            print(f"SWE-bench Pro evaluator already pinned at {revision}")
            return
        if _git_is_dirty(evaluator_dir):
            raise RuntimeError(f"{evaluator_dir} has local changes; clean it before checking out {revision}")
        _git(["fetch", "--depth", "1", "origin", revision], cwd=evaluator_dir)
        _git(["checkout", "--detach", revision], cwd=evaluator_dir)
        _write_revision_marker(evaluator_dir, revision)
        print(f"Updated SWE-bench Pro evaluator to {revision}")
        return

    evaluator_dir.parent.mkdir(parents=True, exist_ok=True)
    _git(["clone", "--filter=blob:none", repo_url, str(evaluator_dir)])
    _git(["checkout", "--detach", revision], cwd=evaluator_dir)
    _write_revision_marker(evaluator_dir, revision)
    print(f"Cloned SWE-bench Pro evaluator at {revision} into {evaluator_dir}")


def _patch_state_path(evaluator_dir: Path) -> Path:
    return evaluator_dir / PATCH_STATE_MARKER


def _patch_digest(patch_file: Path) -> str:
    return hashlib.sha256(patch_file.read_bytes()).hexdigest()


def _read_patch_state(evaluator_dir: Path) -> dict[str, dict[str, str]]:
    state_path = _patch_state_path(evaluator_dir)
    if not state_path.is_file():
        return {}
    try:
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(
            f"  WARN: {PATCH_STATE_MARKER} is not valid JSON; ignoring patch-state (drift detection disabled this run)"
        )
        return {}
    patches = loaded.get("patches") if isinstance(loaded, dict) else None
    if not isinstance(patches, list):
        print(
            f"  WARN: {PATCH_STATE_MARKER} has no 'patches' list; ignoring patch-state (drift detection disabled this run)"
        )
        return {}
    state: dict[str, dict[str, str]] = {}
    for item in patches:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        sha256 = str(item.get("sha256") or "")
        target = str(item.get("target") or "")
        if name and sha256 and target:
            state[name] = {"sha256": sha256, "target": target}
    return state


def _rollback_applied(evaluator_dir: Path, applied: list[tuple[Path, str]]) -> None:
    """Reverse-apply patches in LIFO order so an aborted run leaves the evaluator
    checkout unpatched rather than half-patched.

    Best-effort: a revert that itself fails is surfaced as a warning so we revert
    as many as we can before the caller re-raises the original error.
    """
    if not applied:
        return
    print(
        f"  ROLLING BACK {len(applied)} previously-applied patch(es) to leave evaluator checkout in a consistent state"
    )
    for prev_patch, prev_target in reversed(applied):
        try:
            _git(
                ["apply", "--reverse", "--whitespace=nowarn", str(prev_patch)],
                cwd=evaluator_dir,
            )
            print(f"    REVERTED {prev_patch.name} <- {prev_target}")
        except subprocess.CalledProcessError as revert_exc:
            revert_err = (revert_exc.stderr or "").strip()
            print(f"    WARN: could not revert {prev_patch.name}: {revert_err or revert_exc}")


def _write_patch_state(evaluator_dir: Path, records: list[dict[str, str]]) -> None:
    payload = {
        "schema_version": 1,
        "patches": sorted(records, key=lambda item: item["name"]),
    }
    _patch_state_path(evaluator_dir).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def apply_ksi_evaluator_patches(
    *,
    evaluator_dir: Path,
    patches_dir: Path = KSI_EVALUATOR_PATCHES_DIR,
) -> None:
    """Apply ksi-side patches to the SWE-bench Pro evaluator checkout.

    The official evaluator at scaleapi/SWE-bench_Pro-os has a few rough edges
    that block long-running DGM baseline experiments (notably:
    container.wait() with no timeout, leaving zombie containers when tests
    hang; and container.run() with no memory cap, letting an unbounded
    container get OOM-killed by the host mid-test-run). We carry minimal
    local patches in benchmarks/swebench_pro/evaluator_patches/ and apply
    them idempotently after each checkout.

    NOTE: each patch's idempotency marker check (below) keys on the *target
    file*, not on individual patch content — a second patch touching a
    target file another patch already marked is silently skipped instead of
    applied. So multiple related changes to the same target file (e.g. the
    timeout and memory-limit changes above, both in swe_bench_pro_eval.py)
    must ship as hunks within ONE patch file, not as separate patch files.

    Idempotency: each patch's target file should contain a `KSI PATCH:`
    (or pre-rename `SWARMS PATCH:`)
    marker comment after it lands. We check for that marker before applying;
    if present, the patch is considered applied and we skip it. We also write
    a small patch-state marker in the evaluator checkout so the exact patch
    filenames and hashes are visible after setup.

    Rollback on abort: if patch N applies cleanly but patch N+1 fails — or a
    hash-drift guard trips on an already-marked patch — we roll back patches
    0..N before raising. Otherwise the evaluator checkout is left in a
    half-patched state where some patches are applied and others aren't —
    `verify_evaluator_revision()` would then reject the checkout as dirty on
    next setup, but with only a confusing partial diff.
    """
    if not patches_dir.exists():
        return
    patch_files = sorted(p for p in patches_dir.glob("*.patch") if p.is_file())
    if not patch_files:
        return
    print(f"Applying {len(patch_files)} ksi patch(es) to {evaluator_dir}")
    applied: list[tuple[Path, str]] = []  # (patch_file, target_rel) for rollback
    patch_state = _read_patch_state(evaluator_dir)
    records: list[dict[str, str]] = []
    for patch_file in patch_files:
        patch_sha256 = _patch_digest(patch_file)
        # Find the target file path from the patch's `+++ b/<path>` line
        target_rel: str | None = None
        for line in patch_file.read_text().splitlines():
            if line.startswith("+++ b/"):
                target_rel = line[len("+++ b/") :]
                break
        if target_rel is None:
            print(f"  WARN: could not determine target for {patch_file.name}; skipping")
            continue
        target_path = evaluator_dir / target_rel
        # "SWARMS PATCH:" is the pre-rename marker (issue #758); checkouts
        # patched before the ksi rename still carry it.
        target_text = target_path.read_text(errors="ignore") if target_path.is_file() else ""
        if "KSI PATCH:" in target_text or "SWARMS PATCH:" in target_text:
            existing = patch_state.get(patch_file.name)
            if existing and existing.get("sha256") != patch_sha256:
                # The patch content changed since this checkout was patched.
                # Roll back anything we applied earlier this run before bailing,
                # so the operator reinstalls from a consistent (unpatched) state.
                _rollback_applied(evaluator_dir, applied)
                raise RuntimeError(
                    f"{target_rel} already has a KSI/SWARMS patch marker, but "
                    f"{PATCH_STATE_MARKER} records a different hash for {patch_file.name}; "
                    "reinstall the evaluator checkout before applying changed patches."
                )
            # NOTE: a checkout patched by the pre-#907 script has no
            # PATCH_STATE_MARKER, so `existing` is None and a changed patch can
            # escape this guard on the first re-run after upgrading. The state
            # file we write below closes the gap for every subsequent run.
            records.append({"name": patch_file.name, "sha256": patch_sha256, "target": target_rel})
            print(f"  SKIP {patch_file.name}: patch marker already present in {target_rel}")
            continue
        try:
            _git(["apply", "--whitespace=nowarn", str(patch_file)], cwd=evaluator_dir)
            applied.append((patch_file, target_rel))
            records.append({"name": patch_file.name, "sha256": patch_sha256, "target": target_rel})
            print(f"  OK   {patch_file.name} -> {target_rel}")
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            print(f"  FAIL {patch_file.name}: {stderr or exc}")
            _rollback_applied(evaluator_dir, applied)
            raise
    _write_patch_state(evaluator_dir, records)


def _load_dataset_ids(dataset_path: Path) -> list[str]:
    suffix = dataset_path.suffix.lower()
    ids: list[str] = []
    if suffix == ".jsonl":
        with dataset_path.open(encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                row = json.loads(payload)
                if isinstance(row, dict):
                    instance_id = str(row.get("instance_id") or "").strip()
                    if instance_id:
                        ids.append(instance_id)
    elif suffix == ".csv":
        with dataset_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                instance_id = str(row.get("instance_id") or "").strip()
                if instance_id:
                    ids.append(instance_id)
    else:
        raise ValueError(f"unsupported SWE-bench Pro dataset file type: {dataset_path}")

    duplicates = sorted({instance_id for instance_id in ids if ids.count(instance_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate instance_id values in {dataset_path}: {duplicates[:5]}")
    if not ids:
        raise ValueError(f"no instance_id values found in {dataset_path}")
    return ids


def _first_missing(paths: list[Path]) -> str:
    preview = ", ".join(str(path) for path in paths[:5])
    if len(paths) > 5:
        preview += f", ... ({len(paths)} total)"
    return preview


def verify_evaluator_assets(
    *, evaluator_dir: Path, dataset_path: Path, skip_dataset_verification: bool = False
) -> VerificationSummary:
    required = [
        evaluator_dir / "swe_bench_pro_eval.py",
        evaluator_dir / "helper_code" / "image_uri.py",
        evaluator_dir / "run_scripts",
        evaluator_dir / "dockerfiles" / "base_dockerfile",
        evaluator_dir / "dockerfiles" / "instance_dockerfile",
    ]
    missing_static = [path for path in required if not path.exists()]
    if missing_static:
        raise FileNotFoundError(
            "SWE-bench Pro evaluator checkout is missing required files: " + _first_missing(missing_static)
        )

    if skip_dataset_verification:
        # CI-only path (grader invocation-contract smoke, #1138 part 1): the
        # per-instance dataset asset check below requires the gitignored dataset
        # (benchmarks/swebench_pro/dataset/test.jsonl), which is absent in CI. We
        # still verify the dataset-independent evaluator files above; we just
        # skip the per-instance loop that would need the dataset.
        run_script_ids = {path.name for path in (evaluator_dir / "run_scripts").iterdir() if path.is_dir()}
        return VerificationSummary(
            instance_count=0,
            run_script_count=len(run_script_ids),
            extra_run_script_count=0,
        )

    dataset_ids = _load_dataset_ids(dataset_path)
    missing_run_scripts: list[Path] = []
    missing_parsers: list[Path] = []
    missing_base_dockerfiles: list[Path] = []
    missing_instance_dockerfiles: list[Path] = []

    for instance_id in dataset_ids:
        run_dir = evaluator_dir / "run_scripts" / instance_id
        if not (run_dir / "run_script.sh").is_file():
            missing_run_scripts.append(run_dir / "run_script.sh")
        if not (run_dir / "parser.py").is_file():
            missing_parsers.append(run_dir / "parser.py")
        base = evaluator_dir / "dockerfiles" / "base_dockerfile" / instance_id / "Dockerfile"
        if not base.is_file():
            missing_base_dockerfiles.append(base)
        instance = evaluator_dir / "dockerfiles" / "instance_dockerfile" / instance_id / "Dockerfile"
        if not instance.is_file():
            missing_instance_dockerfiles.append(instance)

    problems = []
    if missing_run_scripts:
        problems.append("run scripts: " + _first_missing(missing_run_scripts))
    if missing_parsers:
        problems.append("parsers: " + _first_missing(missing_parsers))
    if missing_base_dockerfiles:
        problems.append("base Dockerfiles: " + _first_missing(missing_base_dockerfiles))
    if missing_instance_dockerfiles:
        problems.append("instance Dockerfiles: " + _first_missing(missing_instance_dockerfiles))
    if problems:
        raise FileNotFoundError("SWE-bench Pro evaluator assets do not cover the dataset: " + "; ".join(problems))

    run_script_ids = {path.name for path in (evaluator_dir / "run_scripts").iterdir() if path.is_dir()}
    return VerificationSummary(
        instance_count=len(dataset_ids),
        run_script_count=len(run_script_ids),
        extra_run_script_count=len(run_script_ids - set(dataset_ids)),
    )


def _relative_target(*, link_path: Path, target: Path) -> Path:
    return Path(os.path.relpath(target, start=link_path.parent))


def _is_stale_generated_compat_path(path: Path) -> bool:
    if path.is_symlink() or not path.is_dir():
        return False
    if (path / ".git").exists() or (path / "swe_bench_pro_eval.py").exists():
        return False
    files = [item for item in path.rglob("*") if item.is_file()]
    return all("__pycache__" in item.parts or item.suffix == ".pyc" for item in files)


def ensure_compatibility_links(*, evaluator_dir: Path, links: tuple[Path, ...] = COMPATIBILITY_LINKS) -> None:
    target = evaluator_dir.resolve()
    for link in links:
        if link.resolve() == target:
            continue
        if link.exists() or link.is_symlink():
            if link.is_symlink() and link.resolve() == target:
                print(f"Compatibility link already exists: {link}")
                continue
            if _is_stale_generated_compat_path(link):
                shutil.rmtree(link)
            else:
                raise FileExistsError(f"compatibility path already exists and is not a link to {target}: {link}")
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(_relative_target(link_path=link, target=target), target_is_directory=True)
        print(f"Created compatibility link: {link} -> {target}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-dir", type=Path, default=DEFAULT_EVALUATOR_DIR)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--check-only", action="store_true", help="Verify an existing checkout without cloning.")
    parser.add_argument(
        "--no-compat-links",
        action="store_true",
        help="Do not create compatibility links for older baseline defaults.",
    )
    parser.add_argument(
        "--skip-dataset-verification",
        action="store_true",
        help=(
            "Verify only dataset-independent evaluator files; skip the per-instance "
            "asset checks that require the (gitignored) dataset. Used by the CI "
            "grader invocation-contract smoke (#1138)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evaluator_dir = args.evaluator_dir.expanduser().resolve()
    dataset_path = args.dataset_path.expanduser().resolve()

    if args.check_only:
        if not evaluator_dir.exists():
            raise FileNotFoundError(
                f"SWE-bench Pro evaluator is not installed at {evaluator_dir}; "
                "run this script without --check-only to install it"
            )
        verify_evaluator_revision(evaluator_dir=evaluator_dir, revision=args.revision)
    else:
        ensure_evaluator_checkout(
            evaluator_dir=evaluator_dir,
            repo_url=args.repo_url,
            revision=args.revision,
        )
        apply_ksi_evaluator_patches(evaluator_dir=evaluator_dir)

    summary = verify_evaluator_assets(
        evaluator_dir=evaluator_dir,
        dataset_path=dataset_path,
        skip_dataset_verification=args.skip_dataset_verification,
    )
    if args.skip_dataset_verification:
        print(
            "Verified SWE-bench Pro evaluator (dataset verification skipped): "
            f"{summary.run_script_count} run-script directories"
        )
    else:
        print(
            "Verified SWE-bench Pro evaluator: "
            f"{summary.instance_count} dataset instances, "
            f"{summary.run_script_count} run-script directories, "
            f"{summary.extra_run_script_count} extra run-script directories"
        )

    if not args.no_compat_links:
        ensure_compatibility_links(evaluator_dir=evaluator_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

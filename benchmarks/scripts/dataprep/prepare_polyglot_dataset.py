#!/usr/bin/env python3
"""Prepare polyglot benchmark dataset from Aider-AI/polyglot-benchmark.

Clones the benchmark repo, extracts per-exercise metadata (problem description,
starter code, test files, reference solution, build files), and writes a JSON
metadata file filtered to a given subset.

Usage:
    # Default subset is the repo-committed 50-task map
    # (benchmarks/polyglot/task_maps/polyglot_medium_50_seed0_ids.json):
    python benchmarks/scripts/dataprep/prepare_polyglot_dataset.py \
        --output data/polyglot_medium.json

    # Materialise a different committed subset (or a remote URL):
    python benchmarks/scripts/dataprep/prepare_polyglot_dataset.py \
        --subset-url benchmarks/polyglot/task_maps/polyglot_rest_ids.json \
        --output data/polyglot_rest.json

    # Reuse a previously cloned repo:
    python benchmarks/scripts/dataprep/prepare_polyglot_dataset.py \
        --repo-cache /tmp/polyglot-benchmark \
        --output data/polyglot_medium.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

# Load .env at module import so downstream users of a distributable package can
# override URLs/paths without editing source. load_dotenv() is idempotent and a
# no-op if cli.py has already called it.
load_dotenv()

log = logging.getLogger(__name__)

BENCHMARK_REPO = os.environ.get(
    "POLYGLOT_BENCHMARK_REPO",
    "https://github.com/Aider-AI/polyglot-benchmark.git",
)
SOURCE_COMMIT_ENV = "POLYGLOT_SOURCE_COMMIT"

# Repo root = three levels up from benchmarks/scripts/dataprep/.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Canonical polyglot subset is the repo-committed 50-task map (medium, seed 0),
# so the materialised dataset is reproducible without a network fetch. Override
# with POLYGLOT_SUBSET_URL to fetch a live/remote subset instead. _fetch_subset
# handles both local paths and http(s) URLs.
DEFAULT_SUBSET_URL = os.environ.get(
    "POLYGLOT_SUBSET_URL",
    str(_REPO_ROOT / "benchmarks" / "polyglot" / "task_maps" / "polyglot_medium_50_seed0_ids.json"),
)

# Per-language configuration for locating exercise files.
#
# Keys:
#   exercises_dir  - path under repo root to the exercises/practice directory
#   test_glob      - glob pattern for test files (relative to exercise dir)
#   solution_glob  - glob pattern for starter/stub files (tests excluded later)
#   example_dir    - subdirectory containing the reference solution
#   example_glob   - glob pattern for the reference solution within example_dir
#   test_command   - shell command to run the exercise's tests
LANG_CONFIG: dict[str, dict[str, str]] = {
    "python": {
        "exercises_dir": "python/exercises/practice",
        "test_glob": "*_test.py",
        "solution_glob": "*.py",
        "example_dir": ".meta",
        "example_glob": "example.py",
        "test_command": "python -m pytest -rA --tb=long",
    },
    "rust": {
        "exercises_dir": "rust/exercises/practice",
        "test_glob": "tests/*.rs",
        "solution_glob": "src/lib.rs",
        "example_dir": ".meta",
        "example_glob": "example.rs",
        "test_command": "cargo test -- --include-ignored",
    },
    "go": {
        "exercises_dir": "go/exercises/practice",
        "test_glob": "*_test.go",
        "solution_glob": "*.go",
        "example_dir": ".meta",
        "example_glob": "example.go",
        "test_command": "go test ./...",
    },
    "javascript": {
        "exercises_dir": "javascript/exercises/practice",
        "test_glob": "*.spec.js",
        "solution_glob": "*.js",
        "example_dir": ".meta",
        "example_glob": "proof.ci.js",
        "test_command": "npm test",
    },
    "java": {
        "exercises_dir": "java/exercises/practice",
        "test_glob": "src/test/java/*.java",
        "solution_glob": "src/main/java/*.java",
        "example_dir": ".meta/src/reference/java",
        "example_glob": "*.java",
        "test_command": "gradle test",
    },
    "cpp": {
        "exercises_dir": "cpp/exercises/practice",
        "test_glob": "*_test.cpp",
        "solution_glob": "*.cpp",
        "solution_glob_extra": "*.h",
        "example_dir": ".meta",
        "example_glob": "example.cpp",
        "example_glob_extra": "example.h",
        "test_command": ("cmake -B build && cmake --build build && cd build && ctest --output-on-failure"),
    },
}

# Build/config files to capture from each exercise directory.
_BUILD_FILE_NAMES = (
    "Cargo.toml",
    "go.mod",
    "package.json",
    "build.gradle",
    "CMakeLists.txt",
    ".exercism/config.json",
)


def _is_remote_subset_url(url_or_path: str) -> bool:
    return url_or_path.startswith(("http://", "https://"))


def _subset_meta_path(url_or_path: str) -> Path | None:
    """Return the sibling manifest metadata path for a local subset, if any."""
    if _is_remote_subset_url(url_or_path):
        return None
    path = Path(url_or_path)
    return path.with_name(f"{path.stem}.meta.json")


def _source_commit_for_subset(url_or_path: str) -> str | None:
    """Read a local subset manifest's pinned upstream commit, when available."""
    meta_path = _subset_meta_path(url_or_path)
    if meta_path is None or not meta_path.exists():
        return None
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    commit = data.get("source_commit")
    return commit if isinstance(commit, str) and commit.strip() else None


def _resolve_source_commit(subset_url: str, source_commit: str | None) -> str | None:
    requested = (source_commit or os.environ.get(SOURCE_COMMIT_ENV, "")).strip()
    if requested:
        return requested
    return _source_commit_for_subset(subset_url)


def _clone_repo(dest: Path, *, source_commit: str | None = None) -> None:
    """Clone the polyglot-benchmark repo into *dest*.

    When a source commit is known, fetch only that commit instead of cloning the
    moving branch tip. This keeps generated benchmark data tied to the committed
    task-map provenance.
    """
    if source_commit:
        log.info("Fetching %s@%s -> %s", BENCHMARK_REPO, source_commit, dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(dest)], check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "-C", str(dest), "remote", "add", "origin", BENCHMARK_REPO],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(dest), "fetch", "--depth=1", "origin", source_commit],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(dest), "checkout", "--detach", source_commit],
            check=True,
            capture_output=True,
            text=True,
        )
        return

    log.info("Cloning %s -> %s", BENCHMARK_REPO, dest)
    subprocess.run(
        ["git", "clone", "--depth=1", BENCHMARK_REPO, str(dest)],
        check=True,
        capture_output=True,
        text=True,
    )


def _repo_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _ensure_repo_at_commit(repo_root: Path, source_commit: str | None) -> None:
    if not source_commit:
        return
    head = _repo_head(repo_root)
    if head != source_commit:
        raise RuntimeError(
            f"Polyglot repo cache {repo_root} is at {head}, expected {source_commit}. "
            "Remove the cache or pass --source-commit matching its checkout."
        )


def _fetch_subset(url_or_path: str) -> list[str]:
    """Load the subset JSON (a list of task-ID strings) from a URL or local file."""
    if url_or_path.startswith(("http://", "https://")):
        log.info("Fetching subset from %s", url_or_path)
        with urllib.request.urlopen(url_or_path) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
    else:
        path = Path(url_or_path)
        if not path.exists():
            raise FileNotFoundError(f"Subset file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of task IDs, got {type(data).__name__}")
    return data


def _read_glob(directory: Path, pattern: str) -> dict[str, str]:
    """Read all files matching *pattern* under *directory*.

    Returns ``{relative_path: content}`` where *relative_path* is the path
    relative to *directory*.  Falls back to ``rglob`` for nested patterns
    (e.g. ``src/test/java/*.java``).
    """
    results: dict[str, str] = {}

    # Direct glob first
    for p in directory.glob(pattern):
        if p.is_file():
            try:
                rel = str(p.relative_to(directory))
                results[rel] = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                log.debug("Skipping non-UTF-8 file: %s", p)

    # If direct glob found nothing, try rglob on the filename portion.
    # This handles patterns like "tests/*.rs" when the glob doesn't match
    # at the top level.
    if not results:
        leaf = pattern.split("/")[-1]
        for p in directory.rglob(leaf):
            if p.is_file():
                try:
                    rel = str(p.relative_to(directory))
                    results[rel] = p.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    log.debug("Skipping non-UTF-8 file: %s", p)

    return results


def _read_first_match(directory: Path, pattern: str) -> str:
    """Read the first file matching *pattern*, return content or ``""``."""
    for p in directory.glob(pattern):
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return ""
    return ""


def _is_test_file(filename: str, test_names: set[str]) -> bool:
    """Return True if *filename* looks like a test or spec file."""
    if filename in test_names:
        return True
    lower = filename.lower()
    return "_test." in lower or ".spec." in lower or lower.startswith("test_")


def _extract_exercise(
    repo_root: Path,
    language: str,
    exercise_name: str,
) -> dict | None:
    """Extract metadata for a single exercise.

    Returns a dict suitable for the output JSON, or ``None`` if the exercise
    directory is missing or the language is unknown.
    """
    cfg = LANG_CONFIG.get(language)
    if not cfg:
        log.warning("Unknown language %r for exercise %s", language, exercise_name)
        return None

    exercise_dir = repo_root / cfg["exercises_dir"] / exercise_name
    if not exercise_dir.is_dir():
        log.warning("Exercise dir not found: %s", exercise_dir)
        return None

    # -- Problem description ------------------------------------------------
    instructions_path = exercise_dir / ".docs" / "instructions.md"
    problem_statement = ""
    if instructions_path.exists():
        try:
            problem_statement = instructions_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log.warning("Could not read instructions for %s__%s", language, exercise_name)

    # -- Test files ---------------------------------------------------------
    test_files = _read_glob(exercise_dir, cfg["test_glob"])

    # -- Starter / solution stub files --------------------------------------
    solution_files = _read_glob(exercise_dir, cfg["solution_glob"])
    # Some languages have extra file patterns (e.g. C++ headers)
    extra_glob = cfg.get("solution_glob_extra")
    if extra_glob:
        solution_files.update(_read_glob(exercise_dir, extra_glob))
    # Exclude test files from solution stubs
    test_names = set(test_files.keys())
    solution_files = {k: v for k, v in solution_files.items() if not _is_test_file(k, test_names)}

    # -- Reference solution -------------------------------------------------
    example_dir = exercise_dir / cfg["example_dir"]
    reference_solution = ""
    reference_solution_files: dict[str, str] = {}
    if example_dir.is_dir():
        reference_solution = _read_first_match(example_dir, cfg["example_glob"])
        # Capture all reference files (e.g. C++ has both example.cpp and example.h)
        for glob_key in ("example_glob", "example_glob_extra"):
            pattern = cfg.get(glob_key)
            if pattern:
                reference_solution_files.update(_read_glob(example_dir, pattern))

    # -- Build files --------------------------------------------------------
    build_files: dict[str, str] = {}
    for build_name in _BUILD_FILE_NAMES:
        bp = exercise_dir / build_name
        if bp.exists() and bp.is_file():
            try:
                build_files[build_name] = bp.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                log.debug("Skipping non-UTF-8 build file: %s", bp)

    # -- .meta/config.json --------------------------------------------------
    config_path = exercise_dir / ".meta" / "config.json"
    meta_config: dict = {}
    if config_path.exists():
        try:
            meta_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            log.warning("Could not parse .meta/config.json for %s__%s", language, exercise_name)

    instance_id = f"{language}__{exercise_name}"

    return {
        "instance_id": instance_id,
        "language": language,
        "exercise_name": exercise_name,
        "problem_statement": problem_statement,
        "starter_code": solution_files,
        "test_files": test_files,
        "reference_solution": reference_solution,
        "reference_solution_files": reference_solution_files,
        "build_files": build_files,
        "test_command": cfg["test_command"],
        "meta_config": meta_config,
    }


def prepare_dataset(
    *,
    subset_url: str,
    output: Path,
    repo_cache: str,
    source_commit: str | None = None,
    allow_partial: bool = False,
    allow_mutable_source: bool = False,
) -> list[dict]:
    """Core pipeline: fetch subset, clone repo, extract exercises, write JSON.

    Returns the list of extracted task dicts.

    Fails closed (``SystemExit``) if any requested task could not be extracted,
    so a partial materialization can't silently produce a short dataset that a
    downstream sweep then reports under the full "medium-50" label. Pass
    ``allow_partial=True`` (CLI ``--allow-partial``) to permit a short build.
    """
    # -- Fetch subset -------------------------------------------------------
    subset_ids = _fetch_subset(subset_url)
    log.info("Subset contains %d task IDs", len(subset_ids))
    resolved_source_commit = _resolve_source_commit(subset_url, source_commit)
    if _is_remote_subset_url(subset_url) and not resolved_source_commit and not allow_mutable_source:
        raise SystemExit(
            "ERROR: remote polyglot subset URLs must be paired with --source-commit "
            f"or {SOURCE_COMMIT_ENV}. Otherwise the subset and cloned "
            "polyglot-benchmark source can drift independently. Pass "
            "--allow-mutable-source only for deliberate exploratory live-tip builds."
        )
    if resolved_source_commit:
        log.info("Using polyglot-benchmark source commit %s", resolved_source_commit)

    # -- Clone or reuse repo ------------------------------------------------
    cleanup_dir: str | None = None
    if repo_cache and Path(repo_cache).is_dir():
        repo_root = Path(repo_cache)
        log.info("Reusing cached repo at %s", repo_root)
        _ensure_repo_at_commit(repo_root, resolved_source_commit)
    else:
        tmpdir = tempfile.mkdtemp(prefix="polyglot-bench-")
        if repo_cache:
            # Clone into the user-specified path for future reuse
            repo_root = Path(repo_cache)
            repo_root.parent.mkdir(parents=True, exist_ok=True)
            # Clean up the tmpdir we won't use
            shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            repo_root = Path(tmpdir) / "polyglot-benchmark"
            cleanup_dir = tmpdir
        _clone_repo(repo_root, source_commit=resolved_source_commit)

    try:
        requested: list[tuple[str, str, str]] = []
        missing: list[str] = []
        for task_id in subset_ids:
            parts = task_id.split("__", 1)
            if len(parts) != 2:
                log.warning("Invalid task ID format: %r (expected lang__exercise)", task_id)
                missing.append(task_id)
                continue
            language, exercise_name = parts
            cfg = LANG_CONFIG.get(language)
            if not cfg:
                log.warning("Unknown language %r for exercise %s", language, exercise_name)
                missing.append(task_id)
                continue
            requested.append((task_id, language, exercise_name))

        if not allow_partial:
            for task_id, language, exercise_name in requested:
                cfg = LANG_CONFIG[language]
                exercise_dir = repo_root / cfg["exercises_dir"] / exercise_name
                if not exercise_dir.is_dir():
                    log.warning("Exercise dir not found: %s", exercise_dir)
                    missing.append(task_id)
            if missing:
                log.warning(
                    "Could not extract %d/%d exercises: %s",
                    len(missing),
                    len(subset_ids),
                    missing,
                )
                raise SystemExit(
                    f"ERROR: extracted only 0/{len(subset_ids)} requested "
                    f"polyglot exercises ({len(missing)} missing: {missing}). Refusing to "
                    f"write a short dataset that would be run and reported under the full "
                    f"subset label. Re-run after fixing the source/cache, or pass "
                    f"--allow-partial to build the short set deliberately."
                )

        # -- Extract exercises ----------------------------------------------
        tasks: list[dict] = []
        extraction_missing: list[str] = []
        for task_id, language, exercise_name in requested:
            result = _extract_exercise(repo_root, language, exercise_name)
            if result:
                tasks.append(result)
            else:
                extraction_missing.append(task_id)
        missing.extend(extraction_missing)

        if missing:
            log.warning(
                "Could not extract %d/%d exercises: %s",
                len(missing),
                len(subset_ids),
                missing,
            )
            if not allow_partial:
                raise SystemExit(
                    f"ERROR: extracted only {len(tasks)}/{len(subset_ids)} requested "
                    f"polyglot exercises ({len(missing)} missing: {missing}). Refusing to "
                    f"write a short dataset that would be run and reported under the full "
                    f"subset label. Re-run after fixing the source/cache, or pass "
                    f"--allow-partial to build the short set deliberately."
                )

        # -- Write output ---------------------------------------------------
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(tasks, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        log.info(
            "Wrote %d tasks to %s (%.1f KB)",
            len(tasks),
            output,
            output.stat().st_size / 1024,
        )

    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    return tasks


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Prepare polyglot benchmark dataset from Aider-AI/polyglot-benchmark.",
    )
    parser.add_argument(
        "--subset-url",
        default=DEFAULT_SUBSET_URL,
        help=(
            "URL or local path to subset JSON (list of task IDs like "
            '"python__poker"). Default: the repo-committed 50-task map '
            "benchmarks/polyglot/task_maps/polyglot_medium_50_seed0_ids.json "
            "(override with the POLYGLOT_SUBSET_URL env var, e.g. the old "
            "HyperAgents remote medium.json, or pass this flag explicitly)."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/polyglot_medium.json",
        help="Output path for the metadata JSON file.",
    )
    parser.add_argument(
        "--repo-cache",
        default="",
        help=(
            "Path to cache the cloned benchmark repo. If the directory already "
            "exists, reuse it instead of cloning. If empty, uses a temp dir "
            "that is deleted after extraction."
        ),
    )
    parser.add_argument(
        "--source-commit",
        default=None,
        help=(
            "polyglot-benchmark commit to materialise. Default: POLYGLOT_SOURCE_COMMIT "
            "if set, otherwise the sibling .meta.json source_commit for local subset maps."
        ),
    )
    parser.add_argument(
        "--allow-mutable-source",
        action="store_true",
        help=(
            "Allow a remote subset URL without a pinned polyglot-benchmark source commit. "
            "Use only for exploratory live-tip builds; reproducible datasets should pass "
            "--source-commit or POLYGLOT_SOURCE_COMMIT."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Permit writing a short dataset when some requested exercises can't be "
            "extracted. Default: fail closed so a partial build never runs under the "
            "full subset label."
        ),
    )
    args = parser.parse_args(argv)

    tasks = prepare_dataset(
        subset_url=args.subset_url,
        output=Path(args.output),
        repo_cache=args.repo_cache,
        source_commit=args.source_commit,
        allow_partial=args.allow_partial,
        allow_mutable_source=args.allow_mutable_source,
    )

    # Summary
    by_lang: dict[str, int] = {}
    for t in tasks:
        lang = t.get("language", "unknown")
        by_lang[lang] = by_lang.get(lang, 0) + 1

    print(
        f"Prepared {len(tasks)} tasks -> {args.output}"
        + (f"  ({', '.join(f'{l}={n}' for l, n in sorted(by_lang.items()))})" if by_lang else "")
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env bash
# clean_run_artifacts.sh - remove local root-level run sidecars and old strays.
#
# Benchmark wrappers and older runs may leave knowledge/runtime SQLite DBs at
# the repo root; direct CLI defaults use runtime_state/knowledge/. Top-level
# results/ campaign directories can also accumulate. These paths are gitignored.
#
# Usage:
#   bash scripts/dev/clean_run_artifacts.sh                 # dry run (default): list, delete nothing
#   bash scripts/dev/clean_run_artifacts.sh --yes           # delete root DBs + memory snapshots
#   bash scripts/dev/clean_run_artifacts.sh --yes --results # also delete results/<run>/ trace dirs
#
# Deletion is irreversible: these paths are gitignored and therefore unrecoverable.
# Anything git tracks is skipped, never deleted.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

APPLY=false
INCLUDE_RESULTS=false
for arg in "$@"; do
    case "$arg" in
        --yes) APPLY=true ;;
        --results) INCLUDE_RESULTS=true ;;
        -h | --help)
            sed -n '2,14p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg (valid: --yes --results)" >&2
            exit 1
            ;;
    esac
done

# Current schema is <stem>_{knowledge,runtime}.sqlite; _docs/_memory/_forum are the
# retired pre-rename stems, still present in older checkouts.
targets=()
while IFS= read -r path; do
    [[ -n "$path" ]] && targets+=("$path")
done < <(
    find . -maxdepth 1 \( \
        -name '*_knowledge.sqlite*' -o \
        -name '*_runtime.sqlite*' -o \
        -name '*_docs.sqlite*' -o \
        -name '*_memory.sqlite*' -o \
        -name '*_forum.sqlite*' -o \
        -name 'memory_snapshot_*.json' \
        \) -printf '%P\n' | sort
)

if $INCLUDE_RESULTS && [[ -d results ]]; then
    while IFS= read -r path; do
        [[ -n "$path" ]] && targets+=("$path")
    done < <(find results -mindepth 1 -maxdepth 1 -type d -printf 'results/%f\n' | sort)
fi

if [[ ${#targets[@]} -eq 0 ]]; then
    echo "Nothing to clean."
    exit 0
fi

# Never delete anything git tracks, whatever the glob matched.
kept=()
for path in "${targets[@]}"; do
    if git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
        echo "SKIP (git-tracked): $path" >&2
    else
        kept+=("$path")
    fi
done

if [[ ${#kept[@]} -eq 0 ]]; then
    echo "Nothing to clean (all matches are git-tracked)."
    exit 0
fi

total="$(du -ch "${kept[@]}" 2>/dev/null | tail -1 | cut -f1)"
printf '%s\n' "${kept[@]}"
echo "---"
echo "${#kept[@]} path(s), $total"

if ! $APPLY; then
    echo
    echo "Dry run — nothing deleted. Re-run with --yes to delete."
    $INCLUDE_RESULTS || echo "results/ trace dirs not considered; add --results to include them."
    exit 0
fi

rm -rf -- "${kept[@]}"
echo "Deleted ${#kept[@]} path(s), $total freed."

#!/bin/bash
# Rebuild kcsi-agent container images when the baked TypeScript is out of
# sync with the current source tree. Intended to run from a git post-merge /
# post-checkout hook so contributors don't silently run stale bench images
# after pulling a PR that changed `runtime_runner/agent-runner/src/`.
#
# Usage:
#   bash scripts/rebuild_container_if_needed.sh             # auto-detect, rebuild :bench if needed
#   bash scripts/rebuild_container_if_needed.sh --bench     # only :bench
#   bash scripts/rebuild_container_if_needed.sh --all       # :bench AND :latest
#   bash scripts/rebuild_container_if_needed.sh --force     # rebuild regardless of hash check
#   bash scripts/rebuild_container_if_needed.sh --dry-run   # report what would happen, don't run docker
#
# Exit codes:
#   0  rebuilt successfully OR no rebuild needed
#   1  rebuild attempted but docker build failed
#   2  docker not available (soft no-op with warning)
#
# Environment:
#   SKIP_CONTAINER_REBUILD=1  skip entirely (for CI / offline / low-trust envs)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

FORCE=0
DRY_RUN=0
DO_BENCH=1
DO_LATEST=0

for arg in "$@"; do
  case "$arg" in
    --bench)   DO_BENCH=1; DO_LATEST=0 ;;
    --latest)  DO_BENCH=0; DO_LATEST=1 ;;
    --all)     DO_BENCH=1; DO_LATEST=1 ;;
    --force)   FORCE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [[ "${SKIP_CONTAINER_REBUILD:-0}" == "1" ]]; then
  echo "[rebuild-if-needed] SKIP_CONTAINER_REBUILD=1 — skipping."
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[rebuild-if-needed] docker not available — skipping (non-fatal)." >&2
  exit 2
fi

# Compute hash of live TS source the same way the container entrypoint does.
# The Dockerfile/entrypoint `cd`s into /app (where src/ sits as an immediate
# subdir) and then `find src -type f ...`. To match byte-for-byte on the
# host, we subshell-`cd` into runtime_runner/agent-runner so the md5sum
# output carries identical relative paths (src/index.ts, not
# runtime_runner/agent-runner/src/index.ts).
live_hash() {
  ( cd runtime_runner/agent-runner && \
    find src -type f -name '*.ts' -exec md5sum {} + 2>/dev/null \
    | sort | md5sum | cut -d' ' -f1 )
}

# Extract the hash baked into an image's /tmp/dist/.src_hash.
# Returns empty string if the image is missing or the file is absent.
baked_hash() {
  local tag="$1"
  if ! docker image inspect "$tag" >/dev/null 2>&1; then
    return 0
  fi
  docker run --rm --entrypoint=/bin/sh "$tag" \
    -c 'cat /tmp/dist/.src_hash 2>/dev/null' 2>/dev/null | tr -d '[:space:]'
}

rebuild_one() {
  local tag="$1" dockerfile="$2"
  local cur_live baked
  cur_live="$(live_hash)"
  baked="$(baked_hash "$tag")"
  echo "[rebuild-if-needed] $tag  live=${cur_live:0:12}  baked=${baked:0:12}"

  if [[ "$FORCE" -ne 1 ]] && [[ -n "$baked" ]] && [[ "$cur_live" == "$baked" ]]; then
    echo "[rebuild-if-needed] $tag is up to date — skipping."
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[rebuild-if-needed] DRY-RUN: would run: docker build --no-cache -f $dockerfile -t $tag ."
    return 0
  fi

  echo "[rebuild-if-needed] rebuilding $tag (docker build --no-cache -f $dockerfile -t $tag .) ..."
  if docker build --no-cache -f "$dockerfile" -t "$tag" . 2>&1 | tail -8; then
    echo "[rebuild-if-needed] $tag rebuilt."
  else
    echo "[rebuild-if-needed] ERROR: $tag build failed." >&2
    return 1
  fi
}

rc=0
if [[ "$DO_BENCH" -eq 1 ]]; then
  rebuild_one "kcsi-agent:bench" "container/Dockerfile.bench" || rc=1
fi
if [[ "$DO_LATEST" -eq 1 ]]; then
  rebuild_one "kcsi-agent:latest" "container/Dockerfile" || rc=1
fi
exit "$rc"

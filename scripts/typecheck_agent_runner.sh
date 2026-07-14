#!/bin/bash
# Type-check the in-container agent-runner TypeScript with its own tsconfig.
#
# The `runtime_runner/tsconfig.json` typecheck covers `runtime_runner/src/`
# (host-side launcher) only — NOT `runtime_runner/agent-runner/src/`
# (in-container adapters including index.ts and openai.ts). The agent-runner
# has a separate tsconfig (`runtime_runner/agent-runner/tsconfig.json`) and
# different dependencies (@openai/agents, @anthropic-ai/claude-agent-sdk,
# @modelcontextprotocol/sdk) that are only installed inside the Docker image
# at build time. If you don't run this script, errors in agent-runner code
# surface only during `bash container/build.sh --bench`, costing a full
# rebuild round-trip instead of failing fast at commit / review time.
#
# This script provisions a host-side install of the container's deps in a
# cache directory (default: $HOME/.cache/kcsi-agent-runner-typecheck) and
# runs `tsc --noEmit` against the mounted source. Deps persist between runs
# so subsequent checks take ~2s.
#
# Exit codes:
#   0  typecheck passed (or, in LOCAL mode only, npm/tsc unavailable — soft skip)
#   1  typecheck failed, OR (in STRICT/CI mode) tooling was unavailable so the
#      typecheck could not run — fail hard instead of failing open (issue #1226)
#   2  script invocation error
#
# Strict (fail-hard) mode:
#   When `CI` is set to a truthy value (GitHub Actions sets `CI=true`) or
#   `KCSI_TYPECHECK_STRICT=1`, a missing npm/node, a failed `npm ci`, or a
#   missing `tsc` is a HARD FAILURE (exit 1) rather than a soft skip. This
#   prevents CI from going green without ever typechecking agent-runner/src.
#   Outside strict mode the local soft-skip convenience is preserved.
#
# Env:
#   KCSI_TYPECHECK_CACHE    override the cache dir
#   KCSI_TYPECHECK_SKIP=1   soft skip (return 0) — honored even in strict mode
#   KCSI_TYPECHECK_STRICT=1 force fail-hard mode (implied by CI=true)

set -uo pipefail

if [[ "${KCSI_TYPECHECK_SKIP:-}" == "1" ]]; then
  echo "[typecheck_agent_runner] KCSI_TYPECHECK_SKIP=1 — skipping."
  exit 0
fi

# Strict mode: fail hard when tooling is unavailable instead of skipping.
# Honor the standard CI env var (truthy) or an explicit opt-in.
STRICT=0
case "${CI:-}" in
  1 | true | TRUE | True | yes | on) STRICT=1 ;;
esac
if [[ "${KCSI_TYPECHECK_STRICT:-}" == "1" ]]; then
  STRICT=1
fi

# Skip in local mode, but fail hard in strict/CI mode.
# Usage: skip_or_fail "<reason message>"
skip_or_fail() {
  local reason="$1"
  if [[ "$STRICT" == "1" ]]; then
    echo "[typecheck_agent_runner] $reason — failing (strict/CI mode; set KCSI_TYPECHECK_SKIP=1 to bypass)." >&2
    exit 1
  fi
  echo "[typecheck_agent_runner] $reason — skipping typecheck." >&2
  exit 0
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$REPO_ROOT/runtime_runner/agent-runner/src"
MANIFEST_DIR="$REPO_ROOT/runtime_runner/agent-runner"
CACHE_DIR="${KCSI_TYPECHECK_CACHE:-$HOME/.cache/kcsi-agent-runner-typecheck}"

if [[ ! -d "$SRC_DIR" ]]; then
  skip_or_fail "$SRC_DIR not found"
fi
if [[ ! -f "$MANIFEST_DIR/tsconfig.json" || ! -f "$MANIFEST_DIR/package.json" ]]; then
  echo "[typecheck_agent_runner] missing tsconfig/package in $MANIFEST_DIR." >&2
  exit 2
fi

# npm/node not available: soft-skip locally, fail hard in strict/CI mode.
if ! command -v npm >/dev/null 2>&1 || ! command -v node >/dev/null 2>&1; then
  skip_or_fail "npm/node not on PATH"
fi

mkdir -p "$CACHE_DIR"
cp -f "$MANIFEST_DIR/package.json" "$CACHE_DIR/package.json"
cp -f "$MANIFEST_DIR/package-lock.json" "$CACHE_DIR/package-lock.json"
cp -f "$MANIFEST_DIR/tsconfig.json" "$CACHE_DIR/tsconfig.json"
# Mirror src as a symlink so edits in the real tree are picked up.
ln -sfn "$SRC_DIR" "$CACHE_DIR/src"

# Install / refresh deps only when package.json or package-lock.json changed
# (fast path ~0s). npm ci (not npm install) so the typecheck dependency set
# matches what the Docker build actually installs (container/Dockerfile.bench
# also runs npm ci against this same lockfile).
PKG_HASH_FILE="$CACHE_DIR/.package-json.md5"
CURRENT_HASH="$(cat "$CACHE_DIR/package.json" "$CACHE_DIR/package-lock.json" | md5sum | awk '{print $1}')"
STORED_HASH="$(cat "$PKG_HASH_FILE" 2>/dev/null || echo '')"
if [[ "$CURRENT_HASH" != "$STORED_HASH" || ! -x "$CACHE_DIR/node_modules/.bin/tsc" ]]; then
  echo "[typecheck_agent_runner] installing agent-runner deps (one-time, ~5s)..." >&2
  (cd "$CACHE_DIR" && npm ci --legacy-peer-deps --no-audit --no-fund --silent) || \
    skip_or_fail "npm ci failed"
  echo "$CURRENT_HASH" > "$PKG_HASH_FILE"
fi

if [[ ! -x "$CACHE_DIR/node_modules/.bin/tsc" ]]; then
  skip_or_fail "tsc missing after install"
fi

"$CACHE_DIR/node_modules/.bin/tsc" --noEmit -p "$CACHE_DIR" "$@"

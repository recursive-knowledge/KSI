#!/bin/bash
# KCSI agent container entrypoint.
#
# Responsibilities:
#   1. Skip-or-rebuild the agent-runner TypeScript dist based on the
#      source-hash comparison baked during `docker build` (CLAUDE.md
#      documents this ~500ms vs 5-10s optimization).
#   2. On tsc failure (exit 2), emit a framed diagnostic block to STDERR
#      so the kcsi host can surface the underlying compile error
#      instead of the opaque "Container exited with code 2: " message.
#   3. Read the runtime payload from stdin and execute the compiled runner.
#
# Stream discipline:
#   - STDOUT is reserved for the runtime output envelope (see PR #365).
#   - STDERR carries diagnostics, including the TSC_COMPILE_FAILED frame.
#   - The original `npx tsc --outDir /tmp/dist 2>&1 >&2` merge semantics
#     are preserved; tee'ing to a log file does not change stream routing.
set -e
RUNNER_ROOT="${KCSI_RUNNER_ROOT:-/app}"
cd "$RUNNER_ROOT"
mkdir -p /tmp/dist
chmod -R a+rwX /tmp/dist 2>/dev/null || true
SRC_HASH=$(find src -type f -name "*.ts" -exec md5sum {} + 2>/dev/null | sort | md5sum | cut -d" " -f1)
CACHE_HASH=""
if [ -f /tmp/dist/.src_hash ]; then
  CACHE_HASH=$(cat /tmp/dist/.src_hash)
fi
if [ "$SRC_HASH" != "$CACHE_HASH" ] || [ ! -f /tmp/dist/index.js ]; then
  # Capture merged stdout+stderr of tsc to a log file while preserving the
  # original merge-to-stderr behaviour. We temporarily disable `set -e` around
  # the tsc call so we can inspect its exit code and emit the framed
  # diagnostic before propagating the failure.
  TSC_LOG=/tmp/tsc-compile.log
  : > "$TSC_LOG"
  set +e
  # Route tsc stderr+stdout (merged) both to STDERR (original behaviour from
  # the baked inline entrypoint: `2>&1 >&2`) AND to $TSC_LOG via `tee`.
  # Using process substitution avoids the pipeline-exit-code pitfalls while
  # keeping the stream routing identical for live operators.
  npx tsc --outDir /tmp/dist > >(tee -a "$TSC_LOG" >&2) 2>&1
  tsc_status=$?
  # Wait for the process-substitution tee to drain before reading the log.
  wait 2>/dev/null || true
  set -e
  if [ "$tsc_status" != "0" ]; then
    {
      echo "====TSC_COMPILE_FAILED===="
      head -n 40 "$TSC_LOG" 2>/dev/null || true
      echo "====END_TSC_COMPILE_FAILED===="
    } >&2
    exit "$tsc_status"
  fi
  echo "$SRC_HASH" > /tmp/dist/.src_hash
  ln -sfn "$RUNNER_ROOT/node_modules" /tmp/dist/node_modules
fi
cat > /tmp/input.json
node /tmp/dist/index.js < /tmp/input.json

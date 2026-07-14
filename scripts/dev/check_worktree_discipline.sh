#!/usr/bin/env bash
# check_worktree_discipline.sh
#
# Self-test invoked at the start of a parallel subagent run to verify the
# agent is operating inside its own git worktree, not the primary repo tree
# (the "cd hazard": a stray `cd` into the primary repo root moves subsequent
# Edit/Write/git calls onto the shared checkout instead of the isolated one).
#
# Usage:
#   bash scripts/dev/check_worktree_discipline.sh <expected-worktree-root>
#
# Behaviour:
#   - Resolves the expected root to an absolute real path.
#   - Runs `git -C <expected> rev-parse --show-toplevel` and compares.
#   - Compares the *caller's* cwd top-level to the expected root.
#   - Exit 0 on match, exit 1 with a loud error message on mismatch.
#
# Intentionally has no dependencies beyond POSIX bash + git.

set -u

fail() {
    printf 'check_worktree_discipline: FAIL: %s\n' "$*" >&2
    exit 1
}

if [ "$#" -lt 1 ]; then
    fail "missing argument: expected worktree root (absolute path)"
fi

expected_raw="$1"

# Resolve to an absolute real path. `readlink -f` on Linux; fall back to
# a cd/pwd trick if readlink -f is unavailable.
if command -v readlink >/dev/null 2>&1 && readlink -f / >/dev/null 2>&1; then
    expected=$(readlink -f -- "$expected_raw")
else
    if [ -d "$expected_raw" ]; then
        expected=$(cd -- "$expected_raw" && pwd -P)
    else
        fail "expected worktree root does not exist or is not a directory: $expected_raw"
    fi
fi

if [ -z "$expected" ] || [ ! -d "$expected" ]; then
    fail "could not resolve expected worktree root: $expected_raw"
fi

if ! command -v git >/dev/null 2>&1; then
    fail "git not found on PATH"
fi

# git -C <expected> top-level
gitdir_top=$(git -C "$expected" rev-parse --show-toplevel 2>/dev/null || true)
if [ -z "$gitdir_top" ]; then
    fail "path is not inside a git worktree: $expected"
fi

# Resolve gitdir_top through readlink if possible to compare apples-to-apples.
if command -v readlink >/dev/null 2>&1 && readlink -f / >/dev/null 2>&1; then
    gitdir_top_real=$(readlink -f -- "$gitdir_top")
else
    gitdir_top_real=$(cd -- "$gitdir_top" && pwd -P)
fi

if [ "$gitdir_top_real" != "$expected" ]; then
    fail "expected worktree root $expected but git rev-parse returned $gitdir_top_real — you are likely in the primary repo tree; do NOT cd into it"
fi

# Check caller's cwd too — if the caller's shell is already inside another
# worktree, warn about it (but don't fail; the caller may have launched us
# from outside deliberately).
cwd_top=$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || true)
if [ -n "$cwd_top" ]; then
    if command -v readlink >/dev/null 2>&1 && readlink -f / >/dev/null 2>&1; then
        cwd_top_real=$(readlink -f -- "$cwd_top")
    else
        cwd_top_real=$(cd -- "$cwd_top" && pwd -P)
    fi
    if [ "$cwd_top_real" != "$expected" ]; then
        printf 'check_worktree_discipline: WARN: caller cwd top-level is %s, expected %s\n' "$cwd_top_real" "$expected" >&2
    fi
fi

printf 'check_worktree_discipline: OK: %s\n' "$expected"
exit 0

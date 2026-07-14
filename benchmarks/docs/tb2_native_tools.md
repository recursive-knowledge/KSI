# TB2 native file tools — kcsi-specific deviation from canonical Harbor

This document records that kcsi' TB2 bridge exposes five SDK-style JSON
actions (`read`, `write`, `edit`, `glob`, `grep`) **in addition to** the
canonical TB2 shell action. This is a deliberate deviation from the
Meta-Harness reference TB2 setup, which exposes only `execute_commands`
(tmux keystrokes), `task_complete`, and `image_read` (see
`stanford-iris-lab/meta-harness/reference_examples/terminal_bench_2/agents/baseline_kira.py`).

These native tools are unrelated to the agent-facing `WebSearch`/`WebFetch`
tools governed by [web_tools_policy.md](./web_tools_policy.md) — TB2
native tools are host-container file operations (read/write/edit/glob/grep
against the task's own container), not internet access.

## What the native tools do

| Action | Purpose | Container mechanism |
|--------|---------|---------------------|
| `read` | Line-range file read with `head -c` byte cap | `docker exec ... awk ... \| head -c` |
| `write` | Exact-content file write (mkdir parent first) | `docker cp host_tmp container:path` |
| `edit` | Unique-string substitution with `replace_all` option | `docker cp` in, modify, `docker cp` back |
| `glob` | `find -name <pattern>` (basename-only) | `docker exec ... find` |
| `grep` | `grep -rE` with `files_with_matches`/`content`/`count` modes | `docker exec ... grep` |

All five mutate state only via `docker exec` / `docker cp` against the same
container the canonical `shell` action targets. No new file-system surface
is exposed beyond what `shell` already has.

## Why this is a deviation worth labeling

The Meta-Harness reference contract gives the agent shell-only access. With
shell-only access, the agent must construct file ops via `awk` / `find` /
`grep` flag-soup, which costs tokens and burns turns on syntax mistakes.
The five native tools give the agent higher-fidelity, lower-friction
file operations — analogous to what Claude Code's built-in tools provide
in interactive use.

Effect on TB2 scores: plausibly inflates them relative to shell-only
baselines. The deviation is in our favor.

## Why we keep the deviation

The Meta-Harness reference contract is the contract as of 2026. Meta-Harness
itself will evolve and is expected to add similar SDK-style tools in future
revisions, in which case kcsi' current native tools become the canonical
shape. Locking the deviation behind a flag now would force a re-run when
Meta-Harness catches up; keeping it on with disclosure is the cheaper
forward path.

## How TB2 results must be labeled

Any TB2 number produced under this bridge must carry one of:

- **TB2 (kcsi + native tools)** — preferred, explicit
- **TB2 (kcsi-adapted)** — acceptable umbrella if a result artifact
  predates this disclosure

DO NOT aggregate TB2 numbers under "TB2" without one of these qualifiers.
The Harbor leaderboard's published numbers use the canonical shell-only
contract; comparing un-qualified kcsi TB2 numbers against the leaderboard
mixes contracts.

## Code locations

All symbols below live in `src/kcsi/runtime/terminal_bench_2_trial.py` and are
greppable by name (line numbers are intentionally omitted because this file
shifts often). The former path `kcsi.benchmarks.terminal_bench_2_runtime` survives as
a thin back-compat alias module that re-exports this one, so existing imports
and monkeypatch targets keep resolving:

- Action dispatch: the `handler = { "read": _handle_tb2_read, ... }` dict
  inside the bridge loop, which routes each native tool call's `kind` to its
  handler.
- Handlers: `_handle_tb2_read`, `_handle_tb2_write`, `_handle_tb2_edit`,
  `_handle_tb2_glob`, `_handle_tb2_grep`.
- System-prompt action listing: the `_TB2_VALID_ACTIONS` set.
- Step history serialization: `_tb2_bridge_stable_header` renders the
  per-step "Recent shell history" the agent sees each turn (with
  `_tb2_format_history_step` formatting each entry), and
  `_build_tb2_bridge_transcript` renders the final saved transcript; both
  format one history entry per native tool call.

## What's NOT a deviation

- Step cap: unlimited by default (`KCSI_TB2_MAX_STEPS` opt-in for CI
  smoke tests). This matches canonical Terminus 2's `max_episodes=1_000_000`.
- Agent/verifier wall-clock: KCSI does not apply the generic container-runner
  30-minute default to TB2 tasks. It rereads each task's `task.toml` at runtime,
  uses `[agent].timeout_sec` for the agent phase, uses
  `[verifier].timeout_sec` for the verifier phase, and records
  `timeout_source = "task.toml"` in metadata. Task-map timeout fields are
  provenance only.
- Verifier phase: KCSI still runs the task's `/tests/test.sh`, but hardens the
  invocation by extracting trusted `/bin/bash` from the pristine task image and
  injecting it into the verifier phase. If that trusted toolchain cannot be
  established, KCSI fails closed as unscored by default
  (`trial_status=verifier_fail_closed_untrusted_toolchain`). Set
  `KCSI_TB2_REQUIRE_TRUSTED_VERIFIER=0` only for legacy comparisons.

## Separate image-acquisition caveat

KCSI does use an 1800s floor for local Docker image builds when pulling the
upstream TB2 image fails. Harbor task configs commonly declare
`[environment].build_timeout_sec = 600.0`, so this is not identical to a
strict local Harbor build. It does not change the task's agent or verifier
wall-clock budget. For leaderboard-comparable runs, use pulled upstream images
with `KCSI_TB2_REQUIRE_PULL=1` and enforce a digest manifest via
`KCSI_TB2_IMAGE_DIGEST_MANIFEST=/path/to/image_digests.json`. Manifest entries
can be keyed by task id (`{"tasks":{"task-id":"repo/image@sha256:..."}}`) or by
image tag (`{"images":{"repo/image:tag":"repo/image@sha256:..."}}`). When the
variable is set, missing entries, missing Docker registry digests, and digest
mismatches fail closed before the task container starts.

## See also

- [`src/kcsi/memory/parity.py`](https://github.com/recursive-knowledge/KCSI/blob/main/src/kcsi/memory/parity.py)
  for the general feedback-channel/leakage rules this TB2-specific disclosure
  is an instance of.
- [web_tools_policy.md](./web_tools_policy.md) for the *different*,
  unrelated agent-facing web-search/fetch policy.

# Terminal-Bench 2 Integration Architecture

This document describes the shipped KSI integration boundary for
Terminal-Bench 2 (TB2). For run examples and task-map management see
[README.md](./README.md).

## Status

TB2 is wired into `ksi.cli` end-to-end. The runtime, loader, evaluator,
and CLI surface are all in `main`. Run a smoke with:

```bash
uv run python -m ksi.cli \
  --task-source terminal_bench_2 \
  --tasks-path benchmarks/terminal_bench_2/task_maps/terminal_bench_2_smoke_5_seed0.json \
  --evaluator terminal_bench_2 --runtime container \
  --provider-profile configs/ksi/.env.haiku \
  --generations 2 \
  --max-concurrent-tasks 5 --seed 0 --drop-solved \
  --experiment-name tb2_smoke_haiku
```

`--runtime-timeout-sec` is deliberately omitted: for TB2 the per-task
`task.toml [agent].timeout_sec` is the authoritative wall-time bound (Harbor
parity), so the timeout is not configurable — omission means no KSI hard
container cap, and passing a non-negative value is rejected.

## Benchmark Contract

Each TB2 task is a Harbor-native package with this layout:

- `instruction.md`
- `task.toml`
- `environment/Dockerfile`
- `solution/solve.sh`
- `tests/test.sh`

`task.toml` carries metadata, resource caps, and timeouts. Critical fields:

- `[agent].timeout_sec`
- `[verifier].timeout_sec`
- `[environment].docker_image` — the upstream-published image; KSI pulls
  this preferentially and only falls back to building from `environment/Dockerfile`
  if the pull fails (override with `KSI_TB2_DISABLE_PULL=1`)
- `[environment].build_timeout_sec` — task-side build cap; KSI takes the
  max of this, `KSI_TB2_BUILD_TIMEOUT_SEC`, and a 1800s floor

The Harbor verifier contract:

- the agent runs inside the task-specific environment
- the verifier entrypoint is `tests/test.sh`
- the verifier writes a numeric score to `/logs/verifier/reward.txt`
- artifacts may include `ctrf.json`, stdout, and stderr logs

## Runtime Architecture

The TB2 path is host-driven and keeps the upstream task container as the
environment under test:

1. Acquire the task image — `docker pull` of `[environment].docker_image`
   first, fall back to `docker build -f environment/Dockerfile environment/`.
   The image tag is content-addressed (`ksi-tb2-<task>:<sha256[:12]>`)
   so parallel runs share the same Docker daemon work.
2. Mount a minimal native-first workspace overlay at
   `/workspace/task/workspace/tb2/` containing the upstream `instruction.md`
   and `task.toml`. This overlay is the only KSI-side surface inside the
   container; the task's own filesystem stays authoritative.
3. Run a provider-backed KSI loop on the host. The loop issues shell
   commands into the live TB2 container via `docker exec` and records each
   action as a normal KSI `tool_trace`. Wall-clock deadline is
   `[agent].timeout_sec`. The bridge applies no step cap by default
   (`KSI_TB2_MAX_STEPS` unset → unlimited, matching the canonical Harbor
   harness which actively warns against capping `max_turns`). Set
   `KSI_TB2_MAX_STEPS=<N>` to opt into a cap for CI smoke tests.
4. Run `bash /tests/test.sh` after the agent finishes.
5. Collect `/logs/verifier/reward.txt` and `ctrf.json`. Reward becomes
   `native_score`; `resolved` is `True` for `reward >= 1.0`.

This avoids any assumption that a TB2 image already contains Node, `tsx`, or
the KSI agent-runner dependencies.

## KSI Task Metadata Contract

The TB2 loader emits `TaskSpec.metadata` with at least:

- `task_source = "terminal_bench_2"`
- `task_root` — absolute path to the task directory
- `source_path` — pinned TB2 source root
- `docker_image` — image name from `task.toml`
- `agent_timeout_sec`
- `verifier_timeout_sec`
- `category`
- `difficulty`

The contract is validated against the on-disk task package at load time.

## Native-First Seed Policy

TB2 does not rewrite the benchmark into a synthetic task spec. The seed policy:

- mount upstream `instruction.md` as a native workspace file
- mount upstream `task.toml` as a native workspace file
- keep KSI `MEMORY.md` as the primary iterative knowledge asset
- mount a `TOOLS.md` note when the runtime needs to surface tool constraints

Native files live under `tb2/` inside the workspace so they don't collide with
other control files on case-insensitive filesystems.

## Runtime Tunables

| Env var | Default | Purpose |
|---|---|---|
| `KSI_TB2_MAX_STEPS` | unlimited | Optional bridge loop step ceiling. Unset (or `0` / negative) means no ksi-side cap; the per-task `[agent].timeout_sec` is the sole wall-time bound, matching the canonical Harbor contract. Set a positive integer to opt into a cap for CI smoke tests. |
| `KSI_TB2_BUILD_TIMEOUT_SEC` | 1800 (floor) | Per-image build timeout when pull falls back to build. Use higher values on slow infrastructure. |
| `KSI_TB2_DISABLE_PULL` | unset | Set to `1` to skip pull-first and force a local build (for testing local Dockerfile changes). Mutually exclusive with `KSI_TB2_REQUIRE_PULL`. |
| `KSI_TB2_REQUIRE_PULL` | unset | Set to `1` to abort the trial when the canonical image cannot be pulled. Use for fairness-mode runs where local Dockerfile drift would invalidate comparison to upstream numbers. |
| `KSI_TB2_KEEP_IMAGES` | `1` | Retain content-addressed images across trials so the daemon's layer cache deduplicates work. Set to `0` for ephemeral cleanup. |

## Harbor Leaderboard Compliance

To stay consistent with the [Harbor leaderboard validation rules][hf-rules]
(HF dataset card, `harborframework/terminal-bench-2-leaderboard`):

- **No timeout overrides.** `[agent].timeout_sec` and `[verifier].timeout_sec`
  from `task.toml` are used as declared. The ksi bridge applies only a 30 s
  floor on `agent_timeout_sec` (irrelevant for any TB2 task — the smallest in
  the corpus at SHA `53ff2b87` is 600 s).
- **No resource overrides.** `cpus` and `memory` are passed unchanged to
  `docker run`. `storage` is parsed but cannot be enforced under `overlay2`
  (see Known Limitations).
- **No step cap by default.** `KSI_TB2_MAX_STEPS` is unset by default. The
  canonical Terminus 2 agent declares `max_episodes = 1_000_000` and the
  harness warns when callers limit it.
- **5 trials minimum per task.** The Harbor validator requires `--n-attempts 5`
  (or higher). On the ksi side this maps to running `--generations 5` (or
  more) over the same task map, since each generation produces one attempt per
  task. Document the trial count in any published score.
- **Fairness-mode recommended invocation:**

  ```bash
  KSI_TB2_REQUIRE_PULL=1 \
    uv run python -m ksi.cli --task-source terminal_bench_2 \
      --tasks-path benchmarks/terminal_bench_2/task_maps/terminal_bench_2_all.json \
      --evaluator terminal_bench_2 --runtime container \
      --provider-profile configs/ksi/.env.haiku \
      --generations 5 --max-concurrent-tasks 25 \
      --seed 0 --no-drop-solved --experiment-name tb2_fairness_haiku
  ```

  `KSI_TB2_REQUIRE_PULL=1` rejects any task whose canonical image fails to
  pull (no silent local-Dockerfile fallback). Step cap is unset → wall-time
  binds.

### Submitting to the Harbor leaderboard

After the fairness run completes, build a submission tree from the ksi
runtime audit database with:

```bash
uv run python -m ksi.eval.tb2_submission \
  --db runs/tb2_fairness_haiku/tb2_fairness_haiku_runtime.sqlite \
  --out-dir submissions/ \
  --agent-url https://github.com/recursive-knowledge/KSI \
  --agent-display-name "KSI" \
  --agent-org "KSI" \
  --model-name claude-haiku-4-5 \
  --model-provider anthropic \
  --model-display-name "Claude Haiku 4.5" \
  --model-org "Anthropic" \
  --task-corpus-git-commit 53ff2b87
```

The tool requires the runtime audit DB (`<experiment>_runtime.sqlite`), which
must have been enabled at experiment time via `--runtime-db-path`; the
knowledge DB is not supported for submissions because per-attempt
`runtime_meta_json` and the `assignments` table it joins live only in the
runtime audit DB.

This writes `submissions/terminal-bench/2.0/<agent>__<model>/` with one job
folder per generation, each containing one `config.json` / `result.json` /
`verifier/reward.txt` per task. The generated `result.json` does not yet
include an image digest field (see
[`UPSTREAM_DIGEST_PINNING.md`](./UPSTREAM_DIGEST_PINNING.md) for the proposal
to carry one). The tool validates locally against Harbor's
rules (no timeout/resource overrides, ≥5 trials per task, valid result.json
per trial) and exits non-zero on any issue. To submit, fork
`harborframework/terminal-bench-2-leaderboard` on Hugging Face, copy the
generated tree in, and open a PR — the leaderboard bot re-validates.

**Multi-generation note.** ksi' knowledge-bundle outer loop evolves a memory
bundle across generations. The 5-trial-minimum rule applies to *the evaluation
phase*, not the discovery phase. Two valid submission strategies:

1. Run N evolution generations to evolve the bundle, then run a fresh N≥5
   evaluation generations with the final bundle frozen, and submit only the
   evaluation generations via `--generations 6,7,8,9,10` (or whichever range
   matches the frozen-bundle phase).
2. Run ≥5 generations total and submit them all, accepting that the score
   reflects an evolving rather than frozen bundle. Document this clearly in
   the submission's `agent_display_name`.

The tool defaults to "all available generations." Use `--generations` to pick
a specific range.

[hf-rules]: https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard

### Image integrity (digest recording)

Every TB2 trial records the image bytes it actually ran against in
`runtime_meta`:

- `image_acquired_via`: `"pull"` or `"build"`
- `image_acquired_digest`: registry digest (`<repo>@sha256:...`) when pull
  succeeded; empty on build path
- `image_acquired_id`: local Docker image ID (`sha256:...`), set when
  `docker image inspect` succeeds after acquisition; empty string if inspect
  fails (timeout, malformed JSON, or missing image)

These let post-hoc analysis detect within-experiment drift (an image's ID
differing across trials of the same task) or compare against an externally
recorded canonical digest. `KSI_TB2_REQUIRE_PULL=1` only guarantees that
the pull succeeded; the digest record proves *which bytes* were used.

The upstream TB2 contract does not yet declare canonical digests in
`task.toml` — see [`UPSTREAM_DIGEST_PINNING.md`](./UPSTREAM_DIGEST_PINNING.md)
for the proposal (filed as upstream issue
[harbor-framework/terminal-bench-2#66](https://github.com/harbor-framework/terminal-bench-2/issues/66))
to add an optional `docker_image_digest` field upstream. Until that lands,
ksi records digests downstream; comparison against a canonical reference
requires that reference to exist somewhere (externally maintained manifest,
upstream task.toml, etc.).

## Comparability With Upstream Harbor

The KSI-mode bridge agent uses a JSON-action protocol over `docker exec`,
not Harbor's installed terminus-2 / claude-code agents. Reward semantics match
(binary 0/1 from `reward.txt`), but published KSI-on-TB2 numbers measure
the KSI mechanism on the TB2 corpus, not Harbor leaderboard parity.

## Known Limitations

- `[environment].storage` from `task.toml` is parsed and recorded in
  `runtime_meta` but not enforced via `docker run --storage-opt`. Most hosts
  use `overlay2` which doesn't honor that flag.
- One task in the upstream corpus has no `ctrf.json` (87/89 coverage as of the
  pinned commit).

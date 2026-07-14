# Terminal-Bench 2

This directory holds the upstream Terminal-Bench 2 task corpus tracked as a
git submodule, plus committed task-subset manifests.

- `source/`: upstream checkout at a pinned commit
- `task_maps/`: committed KSI-side task subsets and manifests

For the runtime architecture and full integration boundary see
[INTEGRATION.md](./INTEGRATION.md).

## Why This Is A Submodule

Terminal-Bench 2 is an upstream-maintained task corpus with Harbor-native
execution semantics. Each task directory is a small benchmark package with:

- `instruction.md`
- `task.toml`
- `environment/Dockerfile`
- `solution/solve.sh`
- `tests/test.sh`

The pinned upstream checkout under `source/` contains 89 task directories.

Initialize after cloning KSI:

```bash
git submodule update --init --recursive benchmarks/terminal_bench_2/source
```

## Quick Start

Run a 5-task × 2-generation smoke with the curated subset:

```bash
uv run python -m ksi.cli \
  --task-source terminal_bench_2 \
  --tasks-path benchmarks/terminal_bench_2/task_maps/terminal_bench_2_local_smoke_5_curated.json \
  --evaluator terminal_bench_2 --runtime container \
  --provider-profile configs/ksi/.env.haiku \
  --generations 2 \
  --max-concurrent-tasks 5 --seed 0 --drop-solved \
  --experiment-name tb2_smoke_haiku
```

`--runtime-timeout-sec` is omitted on purpose: for TB2 the per-task
`task.toml [agent].timeout_sec` is the authoritative wall-time bound (Harbor
parity), so the timeout is not configurable — omission means no KSI hard
container cap, and a non-negative value is rejected.

Per-task images are pulled from Docker Hub (`alexgshaw/<task>:<date>` per
each task's `task.toml`); local builds happen only on pull failure or when
`KSI_TB2_DISABLE_PULL=1`.

## Task Maps

| Map | Tasks | Selection | Use for |
|---|---|---|---|
| `terminal_bench_2_all.json` | 89 | full corpus | full-corpus campaigns |
| `terminal_bench_2_smoke_5_seed0.json` | 5 | random seed=0 | smoke with random selection |
| `terminal_bench_2_seed1_5.json` | 5 | random seed=1 | second-seed smoke (different category mix) |
| `terminal_bench_2_local_smoke_5_curated.json` | 5 | hand-picked | fastest smoke (all 900s timeouts, locally-buildable; recommended for first-time validation) |

Generate a fresh subset manifest with:

```bash
uv run python benchmarks/scripts/dataprep/generate_terminal_bench_2_task_map.py \
  --source benchmarks/terminal_bench_2/source \
  --count 5 \
  --seed 0 \
  --output benchmarks/terminal_bench_2/task_maps/terminal_bench_2_smoke_5_seed0.json
```

The generator validates the expected TB2 task layout before writing the map.

## Single-Task Smoke / Oracle Trial

`scripts/debug/run_terminal_bench_2_trial.py` is a debug-style driver
that runs ONE task in oracle / noop / command / ksi mode without going
through the full KSI engine. Useful for verifying that the Harbor task
contract works end-to-end against a specific task:

```bash
PYTHONPATH=. uv run python scripts/debug/run_terminal_bench_2_trial.py \
  --task-root benchmarks/terminal_bench_2/source/git-multibranch \
  --agent-mode oracle \
  --output-dir /tmp/tb2-git-multibranch-oracle
```

This script shares the trial driver (`run_terminal_bench_2_trial`) with the
production evaluator path; it just doesn't record into the knowledge DB.

For full campaign runs use `ksi.cli` (see Quick Start above).

# Benchmarks

This directory holds the **committed reproducibility artifacts** for each
benchmark — primarily *task maps* (explicit task-ID subsets) plus any
benchmark-specific scaffolding — along with the data-prep scripts and run
presets used to fetch datasets and launch benchmark runs. The actual task
**data** is not committed; it is cloned/downloaded into per-benchmark
`source/` directories (gitignored). See
[BENCHMARK_PREPARE.md](./docs/BENCHMARK_PREPARE.md) for how to fetch each
dataset.

## Layout

| Folder / file | What it is | Notable contents |
|--------|-----------|------------------|
| [`arc/`](./arc/) | **Shared, dataset-agnostic ARC infrastructure** (common to ARC-AGI-1 *and* ARC-AGI-2) | `workspace_ui/` (browser workspace), `native_mode/` (text-first prompts), `task_maps/` (task-map *template* + the policy [README](./arc/task_maps/README.md)) |
| [`arc1/`](./arc1/) | **ARC-AGI-1 dataset** artifacts | `task_maps/` (per-subset maps); `source/` is cloned here (gitignored) |
| [`arc2/`](./arc2/) | **ARC-AGI-2 dataset** artifacts | `task_maps/` (per-subset maps incl. hold-outs); `source/` is cloned here (gitignored) |
| [`polyglot/`](./polyglot/) | Polyglot benchmark | `task_maps/` (`*_ids.json`, some with `*.meta.json` sidecars) |
| [`swebench_pro/`](./swebench_pro/) | SWE-bench Pro benchmark | `task_maps/`, `evaluator_patches/` |
| [`terminal_bench_2/`](./terminal_bench_2/) | Terminal-Bench 2 benchmark | `task_maps/`, integration docs (`README.md`, `INTEGRATION.md`, `UPSTREAM_DIGEST_PINNING.md`); `source/` (git submodule) |
| [`scripts/`](./scripts/) | Data-prep entry points (`dataprep/`, `arc_prep/`) and compatibility wrappers | See [BENCHMARK_PREPARE.md](./docs/BENCHMARK_PREPARE.md) |
| [`docs/`](./docs/) | Benchmark-specific policy docs | `BENCHMARK_PREPARE.md`, `web_tools_policy.md`, `tb2_native_tools.md` |
| `run_arc.sh`, `run_polyglot.sh`, `run_swebench_pro.sh`, `run_terminal_bench_2.sh` | Run presets — one LLM per invocation over the CLI | See **Run presets** below |
| `common.sh` | Shared helpers sourced by the run presets (and `scripts/run_ksi.sh`) | Not run directly |
| `egress_smoke.sh`, `polyglot_test_feedback_smoke.sh` | Live e2e smoke scripts (require Docker + provider creds) | Not part of a benchmark run |

## Data preparation

Each benchmark needs its dataset fetched/generated and a task map validated
before a run. See [BENCHMARK_PREPARE.md](./docs/BENCHMARK_PREPARE.md) for the
full walkthrough (source trees, task-map generation, ARC workspace payloads,
Polyglot dataset export, SWE-bench Pro dataset + evaluator setup, and the
Terminal-Bench 2 submodule checkout). The underlying scripts live under
[`scripts/dataprep/`](./scripts/dataprep/) and [`scripts/arc_prep/`](./scripts/arc_prep/).

## Run presets

Four bash wrappers assemble `ksi.cli` flag lists for a maintained,
one-LLM-per-invocation run of each benchmark:

```bash
bash benchmarks/run_arc.sh <1|2> <haiku|openai> [swarm|noforum ...]
bash benchmarks/run_polyglot.sh <haiku|openai> [swarm|noforum ...]
bash benchmarks/run_swebench_pro.sh <haiku|openai> [swarm|noforum ...]
bash benchmarks/run_terminal_bench_2.sh <haiku|openai> [swarm|noforum ...] [--no-drop-solved]
```

Set `DRY_RUN=true` to print the composed CLI command without executing it.
Each wrapper's own header comment documents its arguments and environment
variables. For ad-hoc runs (any dataset, auto-detected task source), use
[`scripts/run_ksi.sh`](../scripts/run_ksi.sh) instead.

## Quickstart demo (synthetic, no dataset required)

The bundled `examples/quickstart/arc_demo/` tasks are synthetic and are not
part of any official benchmark — no data preparation needed:

```bash
uv run python -m ksi.cli --task-source arc --tasks-path examples/quickstart/arc_demo ...
```

The primary fresh-clone quickstart uses the custom-task demo documented in
[../docs/getting-started.md](../docs/getting-started.md); this ARC-format demo
is for exercising the ARC loader/evaluator without a dataset download.

## Where results land

Direct CLI runs default the knowledge DB to
`runtime_state/knowledge/<experiment-name>/<experiment-name>_knowledge.sqlite`
and derive the runtime DB as a sibling unless explicit paths are passed. The
run presets pass `--output-json results/<artifact-name>.json` and explicit
repo-root SQLite sidecars such as `./<artifact-name>_knowledge.sqlite`; the
Terminal-Bench preset also writes `./<artifact-name>_runtime.sqlite`.
`scripts/run_ksi.sh` instead scopes everything under `results/<name>/`.
Execution traces land under `analysis/traces/<experiment-name>/`. See
[../docs/artifacts.md](../docs/artifacts.md) for the full report layout and
cleanup workflow.

## Licensing & attribution

The task-map manifests committed under `benchmarks/*/task_maps/*.json` are
KSI-authored (explicit task-ID subsets, seeds, and provenance metadata) —
they are not redistributed upstream benchmark data. Benchmark **datasets**
themselves are not committed; they are fetched at prep time from their own
upstream sources (ARC-AGI, ARC-AGI-2, the Polyglot benchmark, SWE-bench Pro,
Terminal-Bench 2 — see [BENCHMARK_PREPARE.md](./docs/BENCHMARK_PREPARE.md) for
exact sources and pinned revisions) and remain subject to those sources' own
licenses.

## Why ARC is three folders

Every other benchmark is a single folder. ARC is split because ARC-AGI-1 and
ARC-AGI-2 are **two distinct datasets** that nonetheless share the same UI,
native-mode prompt builder, and task-map policy:

- `arc/` — the **shared** scaffolding and conventions (no dataset-specific maps).
- `arc1/` and `arc2/` — the **per-dataset** task maps and (cloned) `source/`
  data.

Keeping the shared assets in `arc/` avoids duplicating the workspace UI and
native-mode docs across both datasets.

## Task maps

A task map pins the exact task IDs (for a given dataset + split + seed) used by a
run, so subset experiments are reproducible and provider comparisons stay fair.
The conventions and required fields are documented in
[`arc/task_maps/README.md`](./arc/task_maps/README.md); the same shape is used by
the other benchmarks' `task_maps/` directories.

# Running larger runs

The [quickstart](getting-started.md) and [your own tasks](your_own_tasks.md)
walkthroughs run one generation over a handful of tiny tasks. This page
covers the flags that matter once you scale up — more tasks, more
generations, or a maintained reference benchmark instead of your own tasks.

There is **one canonical launch surface**: the `ksi.cli` argument parser.
Everything else (bash presets, `ksi.run(...)`) is a layer over the same
`GenerationConfig`.

```bash
uv run python -m ksi.cli --help
uv run ksi --help          # console-script alias for the same entry point
```

## Flags that matter at scale

| Flag | Effect |
|------|--------|
| `--max-concurrent-tasks N` | How many task attempts run in parallel during execution. Population size (agent count) derives from the filtered task pool; this caps concurrency, not pool size. |
| `--max-concurrent-forum-tasks N` | Concurrency cap for the discussion phases, independent of `--max-concurrent-tasks`. |
| `--generations N` | Number of knowledge-refinement generations to run. |
| `--per-task-forum-rounds N` / `--cross-task-forum-rounds N` | Discussion rounds per generation; `0` skips the corresponding phase. |
| `--distill-enabled {true,false}` | Whether raw evidence is distilled into reusable guidance after discussion. |
| `--knowledge-db-path PATH` | The authoritative knowledge substrate (attempts, best scores, forum posts, distilled guidance). Defaults to `runtime_state/knowledge/<experiment>/<experiment>_knowledge.sqlite`. |
| `--runtime-db-path PATH` / `--no-runtime-db` | Optional audit sidecar (raw transcripts, token phases); disable it if you don't need it. |
| `--experiment-name NAME` / `--resume` | Names the run's artifacts and DB rows; `--resume` continues a prior run under the same name. |
| `--output-json PATH` | Writes a machine-readable result report, refreshed after every completed generation. |
| `--max-tasks N` | Caps the task pool after filtering — useful for a cheap slice of a large task set before committing to a full run. |

See [glossary.md](glossary.md) for term definitions and
[artifacts.md](artifacts.md) for where results, traces, and DBs land.

## Presets over the CLI

Two convenience layers assemble CLI flag lists for you; neither is a separate
system:

- [`scripts/run_ksi.sh`](https://github.com/recursive-knowledge/KSI/blob/main/scripts/run_ksi.sh) —
  a single-entry launcher that auto-detects the right task source/evaluator
  from a dataset file and resolves `--model` to a provider profile. Use it
  for ad-hoc runs against any dataset.
- [`benchmarks/run_*.sh`](https://github.com/recursive-knowledge/KSI/blob/main/benchmarks/README.md) —
  maintained, one-LLM-per-invocation presets for each reference benchmark
  (ARC, Polyglot, SWE-bench Pro, Terminal-Bench 2). See
  [benchmarks/README.md](https://github.com/recursive-knowledge/KSI/blob/main/benchmarks/README.md)
  for their arguments and environment variables.

Both print the exact composed CLI command without running anything when
`DRY_RUN=true` is set:

```bash
DRY_RUN=true bash benchmarks/run_arc.sh 2 haiku
```

Task selection for a reference benchmark is usually pinned with
`--task-map-path` (a named, reproducible task-ID subset) rather than a bare
`--tasks-path` directory — see
[BENCHMARK_PREPARE.md](https://github.com/recursive-knowledge/KSI/blob/main/benchmarks/docs/BENCHMARK_PREPARE.md#task-maps)
for how task maps are built and validated.

## Observability

The CLI exposes an **Observability** flag group controlling logging verbosity:

| Flag / env var | Effect |
|----------------|--------|
| `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}` | Logging verbosity. Overrides the `KSI_LOG_LEVEL` env var. |
| `-v` / `--verbose` | Shortcut for `--log-level DEBUG`. |
| `KSI_LOG_LEVEL` (env var) | Fallback verbosity when `--log-level` is not passed. |

Precedence is `--log-level` → `KSI_LOG_LEVEL` → `INFO` (the default when
neither is set). At `INFO` the engine emits per-task progress logs as
attempts start and complete; `DEBUG` adds runtime/subprocess detail.

## Cross-task distillation: target conditioning

`--cross-task-distill-target-conditioning` (bool, **default true**) conditions
cross-task distillation on the downstream task: each unsolved (or held-out)
task gets its own cross-task bundle, distilled with that task's full prompt
as the relevance signal and delivered only to the agent attempting it. Pass
`--cross-task-distill-target-conditioning false` to use a single broadcast
bundle shared across all agents instead.

## See also

- [benchmarks/docs/BENCHMARK_PREPARE.md](https://github.com/recursive-knowledge/KSI/blob/main/benchmarks/docs/BENCHMARK_PREPARE.md) —
  dataset preparation and task-map details for the reference benchmarks.
- [artifacts.md](artifacts.md) — result report layout and cleanup workflow.

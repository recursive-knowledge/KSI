# Scripts

This directory keeps thin operational entry points for setting up and
maintaining a local checkout. Benchmark-specific data-prep helpers and run
presets live under [`benchmarks/`](../benchmarks/README.md) instead.

## Layout

| Path | Purpose |
|------|---------|
| `run_ksi.sh` | Single-entry launcher that auto-detects the right harness from a dataset path. |
| `setup_all.sh` | End-to-end local setup (dependencies, container images, benchmark source trees). |
| `quickstart.sh` | Minimal quickstart smoke. |
| `rebuild_container_if_needed.sh` | Rebuilds the agent container image when its inputs changed. |
| `typecheck_agent_runner.sh` | Type-checks the TypeScript agent runner. |
| `dev/` | Developer guardrails such as worktree discipline checks. |

Benchmark dataset preparation (`dataprep/`, `arc_prep/`) and the run presets
(`run_arc.sh`, `run_polyglot.sh`, `run_swebench_pro.sh`,
`run_terminal_bench_2.sh`) live under
[`benchmarks/`](../benchmarks/README.md); see
[`benchmarks/docs/BENCHMARK_PREPARE.md`](../benchmarks/docs/BENCHMARK_PREPARE.md)
for the full data-prep walkthrough.

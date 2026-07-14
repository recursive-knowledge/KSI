# Artifact Conventions

This document records where run artifacts belong, which paths are default CLI
behavior, and which repo-root sidecars are maintained wrapper output.

## Canonical locations

| Artifact | Default / preferred location |
|----------|--------------------|
| Result JSON reports and campaign bundles | `results/` (prefer a per-campaign subdirectory) |
| Knowledge / runtime DBs (`*_knowledge.sqlite`, `*_runtime.sqlite`) | `runtime_state/knowledge/<experiment>/` (per-experiment subdir) |
| Memory / board snapshots (`memory_snapshot_*.json`) | `runtime_state/` |
| Runtime event trace | `analysis/traces/<experiment>/runtime_events.jsonl` (or `$KSI_TRACE_DIR`) |

`runtime_state/`, `results/`, and `analysis/` are gitignored. Direct CLI runs
use these roots by default. Some benchmark wrappers intentionally pass
repo-root SQLite paths for per-run sidecars; see
[Benchmark wrapper sidecars](#benchmark-wrapper-sidecars).

## Result reports

Pass `--output-json results/<name>.json` to write a machine-readable run
report. Along with traces, assignments, per-generation summaries, and token
usage, the report stamps the code commit, resolved model, scoring mode, and
ARC split. When `--task-map-path` is used, it also stores map identity and
provenance under `task_map`.

KSI refreshes this report after each completed generation. Check
`run_complete` before treating it as final: `false` is a best-effort progress
snapshot preserved after a mid-run failure; `true` is the final report.

Results land in `results/<experiment>.json` with solve rates, token usage, and
reproducibility metadata (code commit, provider, ARC split).

## Benchmark wrapper sidecars

The benchmark run presets currently pass explicit `--knowledge-db-path`
arguments such as `./<artifact-name>_knowledge.sqlite`; Terminal-Bench also
passes `--runtime-db-path ./<artifact-name>_runtime.sqlite`. These root-level
DBs are current wrapper sidecars, not direct CLI defaults.

## Cleaning up root-level artifacts

Older runs may also leave retired DB stems or memory snapshots in the repo
root. After archiving anything you need, these generated artifacts are safe to
move under `runtime_state/` or delete:

- `*_knowledge.sqlite*` / `*_runtime.sqlite*` at the repo root
- retired `*_docs.sqlite*` / `*_memory.sqlite*` / `*_forum.sqlite*` at the repo root
- `memory_snapshot_*.json` at the repo root

These are gitignored (both by anchored `/.../...` rules and the broader
`*.sqlite*` / `*.json` globs) so they cannot be committed accidentally.

List root-level generated artifacts without deleting them:

```bash
bash scripts/dev/clean_run_artifacts.sh
```

After archiving anything you need, delete the listed root-level DBs and memory
snapshots with `--yes`. Add `--results` only when you also intend to delete
top-level campaign directories under `results/`; the script never deletes
git-tracked paths.

```bash
bash scripts/dev/clean_run_artifacts.sh --yes
bash scripts/dev/clean_run_artifacts.sh --yes --results
```

## See also

- [faq.md](./faq.md#where-do-my-results-go) — artifact table and JSON schema.
- [experiments.md](./experiments.md#presets-over-the-cli)
  — task-map selection and run presets.

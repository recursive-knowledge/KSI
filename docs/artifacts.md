# Artifact Conventions

This document records where run artifacts belong, so that experiment output
never lands at the repo root and stray files don't get committed by accident.

## Canonical locations

| Artifact | Canonical location |
|----------|--------------------|
| Result JSON reports and campaign bundles | `results/` (prefer a per-campaign subdirectory) |
| Knowledge / runtime DBs (`*_knowledge.sqlite`, `*_runtime.sqlite`) | `runtime_state/knowledge/<experiment>/` (per-experiment subdir) |
| Memory / board snapshots (`memory_snapshot_*.json`) | `runtime_state/` |
| Runtime event trace | `analysis/traces/<experiment>/runtime_events.jsonl` (or `$KCSI_TRACE_DIR`) |

`runtime_state/`, `results/`, and `analysis/` are gitignored. Keep generated
artifacts under these roots; do not write them to the repo root.

## Result reports

Pass `--output-json results/<name>.json` to write a machine-readable run
report. Along with traces, assignments, per-generation summaries, and token
usage, the report stamps the code commit, resolved model, scoring mode, and
ARC split. When `--task-map-path` is used, it also stores map identity and
provenance under `task_map`.

KCSI refreshes this report after each completed generation. Check
`run_complete` before treating it as final: `false` is a best-effort progress
snapshot preserved after a mid-run failure; `true` is the final report.

Results land in `results/<experiment>.json` with solve rates, token usage, and reproducibility metadata (code commit, provider, ARC split).

## Repo-root strays (legacy)

Older runs wrote some artifacts to the repo root. These are **legacy strays**
and are safe to move into `runtime_state/` or delete after archiving:

- `*.sqlite` / `*.sqlite-shm` / `*.sqlite-wal` / `*.sqlite.lock` at the repo root
- `memory_snapshot_*.json` at the repo root

These are gitignored (both by anchored `/.../...` rules and the broader
`*.sqlite*` / `*.json` globs) so they cannot be committed accidentally.

List legacy strays without deleting them:

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

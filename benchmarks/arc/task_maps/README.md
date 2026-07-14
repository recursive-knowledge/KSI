# ARC Task Maps

Task maps are the reproducibility artifact for ARC subset experiments.

If an experiment uses a 50-task subset, the exact task IDs must be committed in
JSON and referenced by the run command. Do not rely on directory ordering or
`--max-tasks 50` alone.

## Required Fields

Each task-map JSON should record:

- `selection_name`
- `benchmark`
- `source_repo`
- `source_branch`
- `source_commit`
- `source_path`
- `split`
- `seed`
- `count`
- `selection_algorithm`
- `tasks`
- `selection_notes`

## Run Pattern

Use the map with:

```bash
uv run python -m ksi.cli \
  --task-source arc \
  --tasks-path benchmarks/arc2/source/data/training \
  --task-ids-file benchmarks/arc2/task_maps/<selection>.json \
  --task-map-path benchmarks/arc2/task_maps/<selection>.json \
  --runtime container \
  --evaluator arc_session \
  --provider-profile configs/ksi/.env.openai
```

Validate committed ARC maps before campaign runs:

```bash
uv run python benchmarks/scripts/dataprep/validate_task_map.py \
  --task-map benchmarks/arc2/task_maps/<selection>.json \
  --task-source arc \
  --tasks-path benchmarks/arc2/source/data/training \
  --require-provenance
```

`--task-ids-file` controls the selected task IDs. `--task-map-path` is the
run-artifact identity/provenance pointer; `--output-json` records its compact
metadata and file/ID hashes under `task_map`.

Per-dataset maps live under `benchmarks/arc1/task_maps/` and
`benchmarks/arc2/task_maps/`. This directory holds templates and shared
references.

`--task-ids-file` accepts either:

- a raw JSON array of strings
- a JSON object with a `task_ids` field
- a JSON object with a `tasks` field where each item has `task_id`

## Recommended ARC 50 Policy

For reproducible ARC 50 subsets, use:

- an explicit split in the map name and metadata
- an explicit `seed`
- one object per selected task
- a committed map for every campaign subset

Current baseline campaign maps use the training split. Knowledge-transfer
recipient maps may use disjoint evaluation or held-out subsets. Do not compare
numbers across different splits without naming the split.

Example per-task fields:

- `task_id`
- `source_file`
- `benchmark`
- `split`
- `index`
- optional `notes`

## Why This Exists

ARC training and evaluation directories contain many tasks. A committed task map:

- makes the subset explicit
- makes provider comparisons fair
- makes replication possible for other users
- avoids accidental drift when file ordering changes

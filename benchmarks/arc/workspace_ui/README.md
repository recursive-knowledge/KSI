# ARC Workspace UI

This is a benchmark-safe shared workspace for both `arc1` and `arc2`.

It is intentionally different from the original ARC1 testing interface:

- it loads a redacted payload, not the raw task JSON
- it never receives `test.output`
- it exports only a predicted output grid

The key design is:

- one canonical ARC pair payload is shared by both native mode and UI mode
- UI mode renders that payload directly
- native mode derives a text prompt from that same payload

Actual prompt files:

- [SYSTEM_PROMPT.txt](SYSTEM_PROMPT.txt)
- [USER_PROMPT.txt](USER_PROMPT.txt)

### Serving the UI

`workspace.js` is loaded as an ES module so it can share grid helpers
with the node-based unit tests in `tests/js/grid.test.mjs`. Chromium
browsers block ES-module imports under the `file://` scheme, so the UI
must be served over HTTP:

```bash
uv run python -m http.server --directory benchmarks/arc/workspace_ui 8000
# then open http://localhost:8000/
```

Firefox and Safari permit `file://` module imports, so opening
`index.html` directly works in those browsers. In all browsers, the
"Load Path" input requires HTTP — `fetch()` is blocked under `file://`.

### Running the JS tests

Grid helpers live in `js/grid.js` (pure ES module, no DOM) and are
unit-tested under node's built-in test runner:

```bash
node --test tests/js/grid.test.mjs
```

## Supported workflow

1. Generate redacted per-pair payloads from a task map.
2. Open `index.html`.
3. Load a payload JSON file or fetch a relative payload path.
4. Edit the output grid.
5. Select the attempt index (1 or 2) in the Prediction Export panel.
6. Export the prediction JSON once per attempt. Downloaded files are named
   `<task_id>_pair<N>_attempt<K>_prediction.json`.
7. Convert prediction JSONs into ARC benchmarking submissions.
8. Score them with `arc-agi-benchmarking`.

ARC scoring requires two attempts per pair. Switch the Attempt radio to `2`,
edit the grid for a second guess, and download again to produce the
`attempt2` file alongside the `attempt1` file.

## Redacted payload schema

```json
{
  "benchmark": "arc1",
  "task_id": "845d6e51",
  "pair_index": 0,
  "train": [
    {
      "input": [[0, 1], [1, 0]],
      "output": [[1, 0], [0, 1]]
    }
  ],
  "test_input": [[0, 1], [1, 0]]
}
```

## Prediction export schema

```json
{
  "benchmark": "arc1",
  "task_id": "845d6e51",
  "pair_index": 0,
  "attempt_index": 1,
  "prediction": [[1, 0], [0, 1]]
}
```

## Preparing canonical payloads

Use:

```bash
uv run python benchmarks/scripts/arc_prep/prepare_arc_workspace_payloads.py \
  --task-map benchmarks/arc1/task_maps/arc1_train_50_seed0.json
```

The script writes one payload per `task_id + pair_index` and a `manifest.json`.

## Converting predictions into ARC submissions

Use:

```bash
uv run python benchmarks/scripts/arc_prep/convert_arc_workspace_predictions.py \
  --predictions-dir /path/to/prediction_jsons \
  --output-submission-dir /path/to/submissions \
  --manifest benchmarks/arc/workspace_payloads/arc1/arc1_train_50_seed0/manifest.json
```

This creates one submission JSON per `task_id` in the format expected by `arc-agi-benchmarking`.

## Scoring with ARC benchmarking

Use:

```bash
PYTHONPATH=benchmarks/arc/benchmarking/src \
uv run python benchmarks/arc/benchmarking/src/arc_agi_benchmarking/scoring/scoring.py \
  --task_dir benchmarks/arc1/source/data/training \
  --submission_dir /path/to/submissions \
  --results_dir /path/to/results
```

Swap the `task_dir` to the corresponding `arc2` split when scoring ARC2.

# Benchmark Preparation

This document covers preparation of static benchmark artifacts under
`benchmarks/` plus generated benchmark inputs under `data/`. ARC has extra
payload, prompt, workspace UI, and scorer-prep steps; Polyglot and SWE-bench
Pro use dataset and task-map preparation. Runtime execution and evaluation for
all benchmarks go through `ksi.cli` and `ksi.eval.*`; see
[README.md](https://github.com/recursive-knowledge/KSI/blob/main/README.md) and [benchmarks/README.md](https://github.com/recursive-knowledge/KSI/blob/main/benchmarks/README.md)
for those flows. Those documents describe the current knowledge-centric
protocol; this page is limited to benchmark artifact preparation.

## Benchmark overview

| Benchmark     | Prep docs                                     | Runtime workflow                                |
|---------------|-----------------------------------------------|-------------------------------------------------|
| ARC-AGI-1 / ARC-AGI-2 | This document (task maps, payloads, prompts) | `README.md` direct CLI example and experiment scripts |
| Polyglot      | This document (generated dataset + task maps) | `README.md` and `benchmarks/README.md` |
| SWE-bench Pro | This document (dataset export + task map)     | `README.md` and `benchmarks/README.md` |
| Terminal-Bench 2 | This document (submodule checkout + task maps) | Harbor-native task corpus with KSI TB2 runtime/evaluator integration |

## Directory layout

```
benchmarks/
  arc/                      assets shared across ARC1 and ARC2
    native_mode/            native ARC eval docs (prompt pipeline notes)
    workspace_ui/           browser UI for manual / human-in-the-loop solves
    benchmarking/           arc-agi-benchmarking scorer checkout (gitignored)
    native_prompts/         rendered prompts (gitignored)
    workspace_payloads/     redacted per-pair payloads (gitignored)
    task_maps/              templates; per-dataset maps live under arc1/arc2/
  arc1/
    source/                 cloned fchollet/ARC-AGI tree (gitignored)
    task_maps/              arc1_{train,eval}_50_seed0.json
  arc2/
    source/                 cloned arcprize/ARC-AGI-2 tree (gitignored)
    task_maps/              arc2_{train,eval}_50_seed0.json
  polyglot/task_maps/       polyglot_medium_50_seed0_ids.json (baseline 50)
                           polyglot_eval_50_seed1_kt_ids.json (disjoint KT eval 50)
  swebench_pro/
    dataset/test.jsonl      raw SWE-bench Pro test split (gitignored; regenerate, see below)
    task_maps/              swebench_pro_test_50_seed0_v1.json
    repo_cache/             pre-checked-out repos (gitignored)
  terminal_bench_2/
    source/                 git submodule checkout of harbor-framework/terminal-bench-2
    task_maps/              KSI-side TB2 subset manifests
```

Per-dataset inputs (task maps, cloned source trees) live under
`benchmarks/arc1/` and `benchmarks/arc2/`. Assets that are identical across
datasets — prompt templates, the workspace UI, the scorer checkout, and
generated artifacts keyed by `<benchmark>/<task_map_stem>/` — live once
under `benchmarks/arc/` so they aren't duplicated per dataset.

Task map JSON files are committed. Generated source checkouts, repo caches,
Polyglot datasets under `data/`, rendered prompts, and workspace payloads are
gitignored under their respective roots.

## Terminal-Bench 2 submodule checkout

Terminal-Bench 2 is different from the other benchmarks in this repo: the
upstream repository is already a Harbor-native task corpus with one directory
per task. We therefore track it as a pinned submodule instead of exporting it
into a single dataset file.

Initialise or refresh the checkout with:

```bash
git submodule update --init --recursive benchmarks/terminal_bench_2/source
```

Each upstream task directory is expected to contain:

- `instruction.md`
- `task.toml`
- `environment/Dockerfile`
- `solution/solve.sh`
- `tests/test.sh`

Generate a stable subset manifest from the validated checkout with:

```bash
uv run python benchmarks/scripts/dataprep/generate_terminal_bench_2_task_map.py \
  --source benchmarks/terminal_bench_2/source \
  --count 5 \
  --seed 0 \
  --output benchmarks/terminal_bench_2/task_maps/terminal_bench_2_smoke_5_seed0.json
```

Current status:

- the upstream TB2 corpus is pinned locally under `benchmarks/terminal_bench_2/source`
- task-map generation is supported in this repo
- faithful execution is wired into `ksi.cli` through the TB2-specific
  loader, executor, and evaluator path
- official timeout contract: KSI parses each task's `task.toml` at runtime
  and uses `[agent].timeout_sec` for the agent phase and
  `[verifier].timeout_sec` for the verifier phase. This matches Harbor's
  leaderboard-safe shape: upstream Harbor trial configs default to
  `timeout_multiplier = 1.0` with no agent/verifier override timeout fields
  (KSI leaves the multiplier unset rather than emitting a literal `1.0`, so a
  KSI submission does not carry that field). The timeout values copied into
  KSI task maps are
  provenance metadata only; runtime rereads the authoritative `task.toml` and
  records `timeout_source = "task.toml"` in preflight and trial metadata.
- runtime tunables (env vars):
  - `KSI_TB2_MAX_STEPS` (default: unlimited, matching canonical Terminus 2's
    `max_episodes=1_000_000`) — bridge loop step ceiling. Set to a positive
    integer to opt into a step cap for CI smoke tests; `0` is the sentinel
    for unlimited; negative or non-numeric values fall through to unlimited.
  - `KSI_TB2_NATIVE_TIMEOUT_SCALE` (default `1.0`, minimum `1.0`) —
    multiplier on the per-action timeout ceiling for the 5 native actions
    (`read`/`write`/`edit`/`glob`/`grep`). Base ceiling is 120s; set
    `=2.0` to allow 240s per native action when grepping large trees
    or copying large files via `edit`.
  - `KSI_TB2_BUILD_TIMEOUT_SEC` (default 1800s floor) — per-image build cap
    when pull falls back to build; raises only, never lowers. This is an
    image-acquisition tolerance, not an agent/verifier solve-time change. For
    leaderboard-comparable or publication runs, prefer pulled upstream images
    (`KSI_TB2_REQUIRE_PULL=1`) and retain the recorded image digest.
  - `KSI_TB2_DISABLE_PULL` (unset) — set to `1` to skip pull-first and
    force a local build (for testing local Dockerfile changes)
  - `KSI_TB2_REQUIRE_PULL` (unset) — set to `1` to fail the trial rather than
    fall back to a local image build when an upstream pull fails. Required-pull
    attempts use the outer `--max-task-retries` budget for transient registry
    failures; deterministic failures do not retry. Healthy sibling trials
    continue, but a generation where every dispatched trial fails during
    registry acquisition aborts after its failed traces are persisted.
  - `KSI_TB2_IMAGE_DIGEST_MANIFEST` (unset) — path to a JSON manifest that
    pins expected upstream image digests. When set, every TB2 task must have a
    matching digest entry and the runtime aborts before the container starts if
    Docker reports a different registry digest. Supported shapes:
    `{"tasks":{"task-id":"repo/image@sha256:..."}}` or
    `{"images":{"repo/image:tag":"repo/image@sha256:..."}}`.
  - `KSI_TB2_KEEP_IMAGES` (default `1`) — retain content-addressed images
    across trials so Docker's layer cache deduplicates work; set to `0` for
    ephemeral cleanup
  - `KSI_TB2_REQUIRE_TRUSTED_VERIFIER` (default `1`) — **fail closed**
    on the verifier's trusted-toolchain hardening: if the trusted
    image-extracted `bash` cannot be injected (which a root agent can force by
    failing the in-container injection setup), the trial is left **unscored**
    (`reward=None`, `trial_status=verifier_fail_closed_untrusted_toolchain`)
    instead of falling back to the legacy PATH-resolved invocation. Set to `0`
    only for legacy comparison runs that intentionally preserve the
    never-worse-than-main fallback. Detect via `runtime_meta.verifier_fail_closed`
    / `verifier_trusted_bash_detail`.

## Populate ARC source trees

ARC task maps reference concrete files under `benchmarks/arc1/source/`
and `benchmarks/arc2/source/`. These trees are gitignored; clone them
before running any downstream prep step.

Default split is "training" (see README's benchmark comparability caveat) —
override the task map to run on "evaluation".

The simplest path is `bash scripts/setup_all.sh`, which clones both trees and
checks out the exact pinned commits below so `task_maps/*.json` stay
reproducible. To clone manually, pin to the SAME commits `scripts/setup_all.sh`
uses — do NOT clone the branch tip, which drifts and breaks reproducibility.

ARC-AGI-1 (task maps use `benchmarks/arc1/source/data/{training,evaluation}/*.json`):

```bash
git clone https://github.com/fchollet/ARC-AGI.git benchmarks/arc1/source
git -C benchmarks/arc1/source checkout 399030444e0ab0cc8b4e199870fb20b863846f34
```

ARC-AGI-2 (task maps use `benchmarks/arc2/source/data/{training,evaluation}/*.json`):

```bash
git clone https://github.com/arcprize/ARC-AGI-2.git benchmarks/arc2/source
git -C benchmarks/arc2/source checkout f3283f727488ad98fe575ea6a5ac981e4a188e49
```

## Task maps

Regenerate with the scripts in `benchmarks/scripts/dataprep/`:

| Benchmark     | Generator                                            |
|---------------|------------------------------------------------------|
| ARC-AGI-1/2   | `benchmarks/scripts/dataprep/generate_arc_task_map.py`          |
| SWE-bench Pro | `benchmarks/scripts/dataprep/generate_swebench_pro_task_map.py` |
| Polyglot      | `benchmarks/scripts/dataprep/prepare_polyglot_dataset.py`       |

Validate with `benchmarks/scripts/dataprep/validate_task_map.py`:

```bash
uv run python benchmarks/scripts/dataprep/validate_task_map.py \
  --task-map benchmarks/arc1/task_maps/arc1_train_50_seed0.json \
  --task-source arc \
  --tasks-path benchmarks/arc1/source \
  --require-provenance
```

Pass `--check-sources` (ARC only) to also verify each task's
`source_file` points to an existing, JSON-parseable file with `train`
and `test` keys. The script exits non-zero if any source is missing or
malformed, which is useful after moving ARC fixture files around.

## ARC workspace payloads

`benchmarks/scripts/arc_prep/prepare_arc_workspace_payloads.py` strips hidden
`test.output` grids from a task map and writes one benchmark-safe payload JSON
per `(task_id, pair_index)` along with a `manifest.json`.

```bash
uv run python benchmarks/scripts/arc_prep/prepare_arc_workspace_payloads.py \
  --task-map benchmarks/arc1/task_maps/arc1_train_50_seed0.json
```

Optional: `--output-dir <path>` to override the default location. Default
output is `benchmarks/arc/workspace_payloads/<benchmark>/<task_map_stem>/`.
Output directory is gitignored.

## ARC native-mode prompts

Native-mode prompts are built by the upstream benchmarking repo's prompt
builder (`arc_agi_benchmarking.prompts.prompt_manager.convert_task_pairs_to_prompt`);
see `benchmarks/arc/native_mode/README.md` for the design notes. Render
per-task prompts from the payload manifest produced above:

```bash
uv run python benchmarks/scripts/arc_prep/prepare_arc_native_prompts.py \
  --payload-manifest benchmarks/arc/workspace_payloads/arc1/arc1_train_50_seed0/manifest.json
```

Optional: `--output-dir <path>`. Default output is
`benchmarks/arc/native_prompts/<benchmark>/<task_map_stem>/`.

Requires `arc_agi_benchmarking` installed (not a runtime dep of `ksi`);
the script will hint at cloning it under `benchmarks/arc/benchmarking/`
if it cannot import it.

**Pin the upstream benchmarking commit** to ensure the prompt builder and scoring schemas don't drift:

```bash
git clone https://github.com/arcprize/arc-agi-benchmarking.git benchmarks/arc/benchmarking
cd benchmarks/arc/benchmarking
git checkout 7a2efa0f65a55a57bd8da08ef02d826e882cfec8
cd ../../..
```

## ARC workspace UI

Static UI under `benchmarks/arc/workspace_ui/` (HTML + vanilla JS). See
[`benchmarks/arc/workspace_ui/README.md`](https://github.com/recursive-knowledge/KSI/blob/main/benchmarks/arc/workspace_ui/README.md)
for the full solver workflow. Quick summary:

1. **Generate payloads** — see `benchmarks/scripts/arc_prep/prepare_arc_workspace_payloads.py`
   invocation above.

2. **Solve** — open `benchmarks/arc/workspace_ui/index.html` in a browser,
   load a payload file, draw the output grid, and export the per-pair
   prediction JSON.

3. **Convert** — merge exported predictions into an arc-agi-benchmarking
   submission:

   ```bash
   uv run python benchmarks/scripts/arc_prep/convert_arc_workspace_predictions.py \
     --predictions-dir /path/to/exported/predictions \
     --output-submission-dir benchmarks/arc/submissions/run-1 \
     --manifest benchmarks/arc/workspace_payloads/arc1/arc1_train_50_seed0/manifest.json
   ```

   Per-task submission shape matches the public ARC-AGI format:
   `[{"attempt_1": ..., "attempt_2": ...}, ...]`. Prediction grids are
   validated (rectangular, integers in `[0,9]`) at conversion time.
   `--manifest` is optional; when supplied it enforces payload coverage.

## End-to-end ARC flow

```
task_map (committed)
   → benchmarks/scripts/arc_prep/prepare_arc_workspace_payloads.py → payloads/ + manifest.json
       → benchmarks/scripts/arc_prep/prepare_arc_native_prompts.py --payload-manifest manifest.json
           → native_prompts/
       → (optional) workspace_ui browser solve
           → prediction JSONs
               → benchmarks/scripts/arc_prep/convert_arc_workspace_predictions.py → ARC submissions
                   → arc_agi_benchmarking scoring/scoring.py → results
```

### Score submissions

After converting predictions into an ARC submissions directory, score them
against the source task grids with the pinned `arc_agi_benchmarking` scorer:

```bash
PYTHONPATH=benchmarks/arc/benchmarking/src \
uv run python benchmarks/arc/benchmarking/src/arc_agi_benchmarking/scoring/scoring.py \
  --task_dir benchmarks/arc1/source/data/evaluation \
  --submission_dir /path/to/submissions \
  --results_dir /path/to/results
```

Swap `--task_dir` to the corresponding `arc2` split when scoring ARC2.

## Polyglot dataset prep

Generate the ignored runtime dataset used by Polyglot campaign wrappers:

```bash
uv run python benchmarks/scripts/dataprep/prepare_polyglot_dataset.py \
  --output data/polyglot_medium.json
```

With no `--subset-url`, the subset defaults to the repo-committed 50-task map
`benchmarks/polyglot/task_maps/polyglot_medium_50_seed0_ids.json`, and the
benchmark source checkout is pinned by the sibling
`polyglot_medium_50_seed0_ids.meta.json` `source_commit`. That keeps both the
task-ID selection and exercise contents reproducible. Set the
`POLYGLOT_SUBSET_URL` env var (or pass `--subset-url`) to override the task
selection. Set `POLYGLOT_SOURCE_COMMIT` (or pass `--source-commit`) only when
you intentionally want a different upstream `polyglot-benchmark` checkout.

Committed task maps under `benchmarks/polyglot/task_maps/` define stable task
ID selections used by baseline adapters and transfer sweeps. The KSI
Polyglot runner consumes the generated `data/polyglot_medium.json` dataset.

Build the KSI evaluator image before running Polyglot experiments:

```bash
uv run python -c "from ksi.benchmarks.polyglot_docker import build_image; build_image()"
```

`ksi-polyglot-eval:latest` uses the exact HyperAgents `pb.base` Dockerfile
recipe as its base and adds only the direct-harness tools KSI needs because
it mounts generated task files into one shared image. Polyglot result artifacts
include `polyglot_environment` labels for the image recipe and source.

## SWE-bench Pro dataset export and prep

The raw test split (`benchmarks/swebench_pro/dataset/test.jsonl`) is **not
committed**: it is a third-party dataset (ScaleAI SWE-bench Pro) that embeds
upstream repository code, so it is gitignored rather than redistributed in-tree. Regenerate it locally with the export step below; it pulls from the
upstream HuggingFace dataset and requires the `swebench-pro` extra and HF
access.

Pin the upstream dataset revision with `--revision` (a HuggingFace git
tag/branch/commit SHA) so the export is reproducible, and thread the same value
through `--source-revision` so the task map records which snapshot it was built
from. The KSI default pin is
`7ab5114912baf22bb098818e604c02fe7ad2c11f`; pass an empty string only for an
explicit unpinned local experiment. A map generated without `--source-revision`
still records `source_sha256` (used by the run-time integrity tripwire below)
but lacks the human-readable revision pin.

```bash
uv run python benchmarks/scripts/dataprep/export_swebench_pro_dataset.py \
    --split test --format jsonl \
    --revision 7ab5114912baf22bb098818e604c02fe7ad2c11f \
    --output benchmarks/swebench_pro/dataset/test.jsonl

uv run python benchmarks/scripts/dataprep/generate_swebench_pro_task_map.py \
    --dataset-path benchmarks/swebench_pro/dataset/test.jsonl \
    --source-revision 7ab5114912baf22bb098818e604c02fe7ad2c11f \
    --selection-name swebench_pro_test_50_seed0_v1 \
    --seed 0 --count 50 \
    --output benchmarks/swebench_pro/task_maps/swebench_pro_test_50_seed0_v1.json
# Produces: benchmarks/swebench_pro/task_maps/swebench_pro_test_50_seed0_v1.json
# The map records source_sha256 of the dataset file. At run time, passing the
# map as --task-ids-file with --task-source swebench_pro re-hashes --tasks-path.
# Maps that record source_revision fail closed on drift by default. Legacy maps
# without source_revision warn by default; pass --strict-swebench-dataset-integrity,
# or set KSI_STRICT_SWEBENCH_DATASET_INTEGRITY=1, to fail closed for those too.
# The standalone benchmarks/scripts/dataprep/validate_task_map.py check remains fail-closed.

uv sync --extra swebench-pro
uv run python benchmarks/scripts/dataprep/setup_swebench_pro_evaluator.py
# Installs the pinned official evaluator under benchmarks/swebench_pro/evaluator
# and creates compatibility links for older baseline adapters.
```

> **Note — the committed `swebench_pro_test_50_seed0_v1.json` map records no
> `source_revision`, but its `source_sha256` IS reproducible.** The map was
> generated WITHOUT a pinned revision, from a dataset export that is now
> gitignored and not in-tree. Exporting the test split at the KSI default pin
> (`--revision 7ab5114912baf22bb098818e604c02fe7ad2c11f`, as shown above)
> reproduces the recorded digest byte-for-byte
> (`59cc275b33ee3477810bffe6d457e187c120c2075ed1ddbae026d9ef32619474`), verified
> across two independent exports.
>
> Because the map lacks `source_revision`, the run-time integrity tripwire still
> **WARNS by default** for it — the fail-closed default applies only to maps that
> record a revision (`effective_strict = strict or source_revision is not None`).
> Since the digest does reproduce, strict enforcement works today: pass
> `--strict-swebench-dataset-integrity` (or set
> `KSI_STRICT_SWEBENCH_DATASET_INTEGRITY=1`) and the check verifies the hash and
> passes. Strict mode only refuses maps that record no `source_sha256` at all,
> which is not the case here. To get a map that fails closed without a flag,
> regenerate it with `--revision` / `--source-revision` pinned as shown above.

For HyperAgents and DGM runs, also materialize the per-instance repo cache
used by the SWE-bench Pro baseline adapters:

```bash
uv run python benchmarks/scripts/dataprep/prepare_swebench_pro_repo_cache.py
```

The cache refresh runs `git fetch --all --tags --prune --force`; any
upstream-moved tags (e.g. rolling `production` deploy tags that SWE-bench Pro
repos occasionally reuse) are overwritten rather than aborting the prep step
with a "would clobber existing tag" error.

## Polyglot and SWE-bench Pro execution

Use `ksi.cli` with the corresponding `--task-source` / `--evaluator`
flags. Use `uv run python -m ksi.cli --help` for the current flag surface
and `benchmarks/README.md` for maintained wrapper commands.

The `src/ksi/eval/swebench_pro` adapter delegates to the pinned official
SWE-bench Pro evaluator checkout prepared by
`benchmarks/scripts/dataprep/setup_swebench_pro_evaluator.py`. The top-level
`benchmarks/scripts/run_swebench_pro_eval.py` wrapper remains as the compatibility entry
point for baseline adapters that use dashed CLI flags.

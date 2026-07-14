# ARC-format demo tasks

Three tiny **synthetic** ARC-format tasks — a bundled, no-download example of
the `--task-source arc` task format (see
[`../../../benchmarks/README.md`](../../../benchmarks/README.md)). The
primary quickstart (`scripts/quickstart.sh`) runs the
[`examples/custom_tasks/`](../../custom_tasks/) demo instead; these ARC tasks
are for exercising the ARC loader/evaluator without a dataset download.

These are *not* the ARC-AGI benchmark. They are deliberately trivial (a single,
obvious transformation each) so a run against them finishes fast and a
correctly configured setup produces solved tasks:

| File | Transformation rule |
|------|---------------------|
| `demo_recolor.json` | replace every `2` with `8` |
| `demo_mirror.json`  | mirror each row left-to-right |
| `demo_transpose.json` | transpose the grid |

Each file follows the ARC task schema consumed by `src/ksi/tasks/loaders.py`:

```json
{
  "train": [{"input": [[...]], "output": [[...]]}, ...],
  "test":  [{"input": [[...]], "output": [[...]]}]
}
```

For real ARC-AGI runs, clone the benchmark sources (see the repo README's
"Preparing Benchmarks" section) and point `--tasks-path` at them instead.

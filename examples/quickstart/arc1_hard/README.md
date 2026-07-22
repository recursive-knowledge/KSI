# Quickstart hard ARC-AGI-1 tasks

Three ARC-AGI-1 tasks used by `scripts/quickstart.sh` to demonstrate the full
knowledge-refinement loop (execute → forum → distill → seed) across generations.

They are chosen to be **hard for current models** — none is reliably one-shot,
so with `--drop-solved` (on by default) the unsolved tasks carry forward and the
forum/distill/seed phases actually run across all three generations, instead of
the pool emptying after generation 1.

| task | ARC-1 split | observed pass rate (single attempt) |
|------|-------------|-------------------------------------|
| `97239e3d` | evaluation | GPT ~0/22, Haiku ~4/14 |
| `d22278a0` | training   | GPT ~0/4, Haiku ~1/6 |
| `776ffc46` | training   | GPT ~0/4, Haiku ~1/6 |

Pass rates are approximate, from internal runs; they show these tasks are
solvable *sometimes* (so the knowledge loop has something to learn and transfer)
but rarely on the first try.

## Provenance

The task JSONs are copied verbatim from the vendored ARC-AGI-1 corpus under
`benchmarks/arc1/source/data/{evaluation,training}/`, originally from
[`fchollet/ARC-AGI`](https://github.com/fchollet/ARC-AGI) (Apache-2.0). They are
duplicated here only so the quickstart can point at a single `--tasks-path`
directory holding all three (they span two splits), keeping the demo
self-contained with no dataset download.

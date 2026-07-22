# Getting started

Go from a fresh clone to the full knowledge loop running in one command, then learn what just happened.

## What you'll do

Run three generations of agents against five bundled **ARC-AGI-1** tasks — hard enough for any
current model that they don't all solve on the first try — and watch the full knowledge loop —
execute, discuss, distill, seed — fire end to end. No dataset download, no manual setup (the ARC-1
tasks are vendored under `benchmarks/arc1/`). ARC attempts are slow, so a full three-generation
run takes on the order of 15–20 minutes; it leaves you with a working environment ready to run your
own tasks or a reference benchmark.

## Prerequisites

- **Docker** — sandboxes each agent in an isolated container.
- **Node.js 22.16.0** — the runtime host (`runtime_runner`) is TypeScript; this is the repository pin in `.nvmrc`.
- **`uv`** — used to invoke the CLI, doctor, and all Python tooling.
- **An API key** — `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`; the quickstart synthesizes a provider profile from whichever you export.

## Run it

```bash
export ANTHROPIC_API_KEY=sk-ant-...    # or: export OPENAI_API_KEY=sk-...
bash scripts/quickstart.sh
```

The script self-bootstraps everything it needs: it synthesizes a provider profile from your key,
builds the `ksi-agent:bench` image on first run (this takes a few minutes), installs the host
Node dependencies, then runs three generations over three hard **ARC-AGI-1** tasks — bundled under
[`examples/quickstart/arc1_hard/`](https://github.com/recursive-knowledge/KSI/tree/main/examples/quickstart/arc1_hard)
(copied from the ARC-1 corpus vendored under `benchmarks/arc1/`, no download) — with the per-task
and cross-task forums on so every phase of the loop fires. Each agent studies an ARC task's
input→output training examples and writes its predicted grids, scored by the `arc_session`
evaluator (exact-match, up to two attempts per test). These three tasks are chosen for being hard
for current models (see the [directory README](https://github.com/recursive-knowledge/KSI/blob/main/examples/quickstart/arc1_hard/README.md)
for their observed pass rates), so they don't all solve on the first generation.

For the complete benchmark environment (including benchmark preparation and
smoke tests), run `bash scripts/setup_all.sh`. Use `--no-test` when you need
the setup without its smoke-test phase.

If it fails partway, see the FAQ's
[troubleshooting entry](faq.md#my-run-failed-or-produced-an-empty-knowledge-db-where-do-i-start) —
`uv run ksi-doctor` covers the most common causes (Docker not running, the
image not built, a missing/invalid API key).

## Check readiness (optional)

```bash
uv run ksi-doctor
```

Prints a ✓/✗ checklist covering Docker availability, the `ksi-agent:bench` image, host Node
dependencies, and a provider profile with a real key — plus the exact command to fix anything
missing.

## What you'll see

The run logs each attempt and its score as it progresses. When it finishes, results land at:

| Artifact | Path |
|----------|------|
| Run log | `/tmp/ksi-experiments/<experiment>.log` |
| Knowledge DB | `runtime_state/knowledge/<experiment>/<experiment>_knowledge.sqlite` |
| Runtime audit DB (optional) | sibling `<experiment>_runtime.sqlite` |
| Score summary (optional — only when `--output-json` is set) | `results/<experiment>.json` |
| Execution traces | `analysis/traces/<experiment>/` |

For the quickstart, `<experiment>` defaults to `quickstart_demo`. The run logs
each task's score (ARC scoring is exact-match: `1.0` solved, `0.0` not) as it
goes, and ends with a single `completed traces=… tasks=… solved=N/M` summary
line. These ARC tasks are hard for current models, so expect at least one to
remain unsolved after generation 1 (a strong model may solve the rest); the
unsolved tasks carry forward and get re-attempted each generation, now seeded
with what the population distilled. **The signal that your environment is set up
correctly is that attempts run and get scored at all** — not that everything
solves. Whether an unsolved task flips to solved by generation 3 depends on the
model. Elapsed times, token counts, and solve counts vary by model and run.

??? note "A closer look — sample output, optional artifacts, and the knowledge DB"

    The quickstart doesn't pass `--output-json`, so it writes no
    `results/quickstart_demo.json`; pass that flag yourself (or use a `benchmarks/`
    run preset, which sets it for you) for a score summary on disk. Traces default
    to `analysis/traces/<experiment>/` — set `KSI_TRACE_DIR` to change the root.

    An excerpt from a real run (`gpt-5.4-mini`, timestamps trimmed). ARC scores
    are binary (exact-match); a strong model solves some tasks on generation 1
    while the hardest carry forward:

    ```text
    INFO ksi.orchestrator.execution_phase: [gen 1] task=776ffc46 agent=agent-0 done elapsed=281.6s score=1.0000
    INFO ksi.orchestrator.execution_phase: [gen 1] task=97239e3d agent=agent-1 done elapsed=281.6s score=1.0000
    INFO ksi.orchestrator.execution_phase: [gen 1] task=d22278a0 agent=agent-2 done elapsed=287.4s score=0.0000
    INFO ksi.orchestrator.distillation_phase: [ENGINE] distill gen=1: 1 per-task bundle(s), cross_task=0
    INFO ksi.orchestrator.persistence: [gen 2] start agents=1
    ```

    Here two tasks solved and dropped out (`--drop-solved`), while `d22278a0`
    stayed unsolved and carried through generations 2 and 3 — each time
    re-attempted with freshly distilled guidance seeded in. The run ended with
    `completed traces=5 tasks=3 solved=2/3 (66.7%)`: five attempts across three
    generations over three unique tasks.

    **Knowledge DB check** — because the demo runs the full loop, the knowledge
    DB carries rows from every phase, not just execution. Group by `entry_type`
    and `source_phase` to see them (real counts from the run above):

    ```console
    $ sqlite3 runtime_state/knowledge/quickstart_demo/quickstart_demo_knowledge.sqlite \
        "select entry_type, source_phase, count(*) from knowledge group by entry_type, source_phase order by entry_type, source_phase;"
    attempt|execution|5
    distillation|cross_task_distill|1
    distillation|per_task_distill|3
    insight|execution|5
    post|cross_task_forum|1
    ```

    The `distillation` rows (one per-task bundle per generation, plus a cross-task
    bundle) confirm the distill phase ran each generation, and

    ```console
    $ sqlite3 runtime_state/knowledge/quickstart_demo/quickstart_demo_knowledge.sqlite \
        "select count(*) from seed_snapshots;"
    2
    ```

    the two `seed_snapshots` confirm seeding fired between each pair of
    generations. Exact counts vary with the model and how much each agent posts.

## What just happened?

KSI runs a knowledge-refinement loop across generations:

1. A population of agents each attempt the tasks in isolated containers.
2. They record every attempt in the knowledge database.
3. They [*discuss*](glossary.md#forum) what worked in two [forums](glossary.md#forum): a **per-task forum**, where the agents that attempted the same task compare their approaches, and a **cross-task forum**, where lessons that generalize beyond a single task are shared across the whole population.
4. The system [*distills*](glossary.md#distillation) those discussions into reusable guidance.
5. The next generation is [*seeded*](glossary.md#seeding) with that guidance.

!!! note "Why ARC tasks — and why hard ones"
    The demo uses ARC-AGI-1 tasks because they're hard for every current model:
    if the tasks were easy, every agent would solve them on the first attempt and
    — with `--drop-solved` (on by default) — the task pool would empty after
    generation 1, so the run would **stop** before the forum, distill, and seed
    phases could show their value. The three bundled tasks
    (`97239e3d`, `d22278a0`, `776ffc46`) are chosen for low observed pass rates
    (see [`examples/quickstart/arc1_hard/`](https://github.com/recursive-knowledge/KSI/tree/main/examples/quickstart/arc1_hard)),
    so unsolved tasks carry forward under the default `--drop-solved` and the full
    loop runs across all three generations. On your own *easy* tasks, expect the
    run to stop early once everything is solved; that's the intended behavior. See
    [experiments.md](experiments.md).

## Next steps

- **Bring your own tasks** — the record schema, the `command` evaluator's
  scoring contract, and a minimal CLI run reusing the profile the quickstart
  above just generated for you:

  ```bash
  uv run python -m ksi.cli \
    --task-source custom \
    --tasks-path examples/custom_tasks/tasks.jsonl \
    --evaluator command \
    --provider-profile configs/ksi/.env.quickstart
  ```

  For a durable profile instead of the quickstart's throwaway one, copy
  `configs/ksi/.env.haiku.template` to `configs/ksi/.env.haiku` and add
  your key. Full contract: [your_own_tasks.md](your_own_tasks.md).

- **Run a reference benchmark instead** — ARC-AGI, SWE-bench Pro, Polyglot,
  Terminal-Bench 2: [benchmarks/README.md](https://github.com/recursive-knowledge/KSI/blob/main/benchmarks/README.md)
- **Scale up** — flags that matter for more tasks and more generations: [experiments.md](experiments.md)
- **Drive it from Python** — `ksi.run(...)`, no CLI: [programmatic_api.md](programmatic_api.md)
- **Understand the design** — runtime, DB ownership, execution path: [architecture.md](architecture.md)
- **Common questions** — [faq.md](faq.md)
- **Term definitions** — forum, distillation, seeding, and more: [glossary.md](glossary.md)

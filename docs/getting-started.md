# Getting started

Go from a fresh clone to a solved demo task in one command, then learn what just happened.

## What you'll do

Run three generations of agents against four bundled, self-contained tasks — pitched hard enough
that they don't all solve on the first try — and watch the full knowledge loop — execute, discuss,
distill, seed — fire end to end. No dataset download, no manual setup. The demo takes several
minutes and leaves you with a working environment ready to run your own tasks or a reference
benchmark.

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
Node dependencies, then runs three generations over the four bundled tasks under
[`examples/custom_tasks/`](https://github.com/recursive-knowledge/KSI/tree/main/examples/custom_tasks)
(`calc-eval`, `range-queries`, `precise-sum`, `tsp-heuristic`) — each graded on the host by the
task's own eval command, with the per-task and cross-task forums on so every phase of the loop
fires.

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
each task's score as it goes, and each generation ends with a
`completed … solved=N/M` line — three in all, one per generation. These tasks
are meant to be hard, so expect only some to solve on generation 1, with the
solve count and the `tsp-heuristic` score generally improving across the three
generations as distilled knowledge accumulates (the hardest tasks may stay
unsolved — that's fine). **The signal that your environment is set up correctly
is that attempts run and get scored at all** — not that everything solves.
Elapsed times, token counts, and exact solve counts vary by model and run.

??? note "A closer look — sample output, optional artifacts, and the knowledge DB"

    The quickstart doesn't pass `--output-json`, so it writes no
    `results/quickstart_demo.json`; pass that flag yourself (or use a `benchmarks/`
    run preset, which sets it for you) for a score summary on disk. Traces default
    to `analysis/traces/<experiment>/` — set `KSI_TRACE_DIR` to change the root.

    An illustrative excerpt from a generation's execution phase against
    `claude-haiku-4-5-20251001` (the default `configs/ksi/.env.haiku` profile),
    timestamps trimmed — each generation logs a block like this, followed by the
    forum and distillation phases. Scores are mixed on generation 1 by design
    (`tsp-heuristic` is a continuous score, never a clean `1.0`):

    ```text
    INFO ksi.orchestrator.execution_phase: [gen 1] task=range-queries agent=agent-1 done elapsed=31.2s score=1.0000
    INFO ksi.orchestrator.execution_phase: [gen 1] task=tsp-heuristic agent=agent-3 done elapsed=44.7s score=0.7800
    INFO ksi.orchestrator.execution_phase: [gen 1] task=calc-eval agent=agent-0 done elapsed=28.1s score=0.0000
    INFO ksi.orchestrator.execution_phase: [gen 1] task=precise-sum agent=agent-2 done elapsed=26.5s score=0.0000
    INFO ksi.orchestrator.engine: completed traces=4 tasks=4 solved=1/4 (25.0%)
    ```

    **Knowledge DB check** — because the demo now runs the full loop, the
    knowledge DB carries rows from every phase, not just execution. Group by
    `entry_type` and `source_phase` to see them:

    ```console
    $ sqlite3 runtime_state/knowledge/quickstart_demo/quickstart_demo_knowledge.sqlite \
        "select entry_type, source_phase, count(*) from knowledge group by entry_type, source_phase order by entry_type, source_phase;"
    ```

    You'll see `attempt` and `insight` rows from `execution`, `post` rows from
    `per_task_forum` and `cross_task_forum` (the two discussion phases), and
    `distillation` rows from `per_task_distill` and `cross_task_distill`. Exact
    counts vary with the model and how much each agent posts, and they grow with
    each of the three generations.

## What just happened?

KSI runs a knowledge-refinement loop across generations:

1. A population of agents each attempt the tasks in isolated containers.
2. They record every attempt in the knowledge database.
3. They [*discuss*](glossary.md#forum) what worked in two [forums](glossary.md#forum): a **per-task forum**, where the agents that attempted the same task compare their approaches, and a **cross-task forum**, where lessons that generalize beyond a single task are shared across the whole population.
4. The system [*distills*](glossary.md#distillation) those discussions into reusable guidance.
5. The next generation is [*seeded*](glossary.md#seeding) with that guidance.

!!! note "Why these tasks are hard on purpose"
    Earlier versions of this demo used trivial tasks that every agent solved on
    the first attempt — so with `--drop-solved` (on by default) the task pool
    emptied after generation 1 and the run **stopped** before the forum, distill,
    and seed phases could show their value. These four tasks are pitched beyond a
    reliable one-shot solve — a truncation-toward-zero parser trap, a range-query
    task that needs a Fenwick tree, numerically-stable summation, and a
    continuous-score TSP heuristic that is effectively never "perfect" — so
    unsolved tasks carry forward under the default `--drop-solved` and the full
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

# Getting started

Go from a fresh clone to a solved demo task in one command, then learn what just happened.

## What you'll do

Run three generations of agents against three bundled, self-contained tasks and watch the full
knowledge loop — execute, discuss, distill, seed — fire end to end. No dataset download, no manual
setup. The demo takes several minutes and leaves you with a working environment ready to run your
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
Node dependencies, then runs three generations over the three bundled tasks under
[`examples/custom_tasks/`](https://github.com/recursive-knowledge/KSI/tree/main/examples/custom_tasks)
(`fizzbuzz`, `reverse-words`, `anagram-groups`) — each graded by running `python3 tests.py`
against the agent's attempt, with the per-task and cross-task forums on so every phase of the
loop fires.

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

For the quickstart, `<experiment>` defaults to `quickstart_demo`. The run prints
each task's score as it goes, and each generation ends with a
`completed … solved=3/3 (100.0%)` line — three in all, one per generation.
Seeing `solved=3/3` is the signal your environment is set up correctly. Elapsed
times and token counts vary by model and run; the task names and `solved=3/3`
don't.

??? note "A closer look — sample output, optional artifacts, and the knowledge DB"

    The quickstart doesn't pass `--output-json`, so it writes no
    `results/quickstart_demo.json`; pass that flag yourself (or use a `benchmarks/`
    run preset, which sets it for you) for a score summary on disk. Traces default
    to `analysis/traces/<experiment>/` — set `KSI_TRACE_DIR` to change the root.

    An illustrative excerpt from a generation's execution phase against
    `claude-haiku-4-5-20251001` (the default `configs/ksi/.env.haiku` profile),
    timestamps trimmed — each generation logs a block like this, followed by the
    forum and distillation phases:

    ```text
    INFO ksi.orchestrator.execution_phase: [gen 1] task=reverse-words agent=agent-1 done elapsed=27.4s score=1.0000
    INFO ksi.orchestrator.execution_phase: [gen 1] task=fizzbuzz agent=agent-0 done elapsed=28.1s score=1.0000
    INFO ksi.orchestrator.execution_phase: [gen 1] task=anagram-groups agent=agent-2 done elapsed=33.0s score=1.0000
    INFO ksi.orchestrator.engine: completed traces=3 tasks=3 solved=3/3 (100.0%)
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

!!! note "Why the demo keeps solved tasks (`--no-drop-solved`)"
    The quickstart runs three generations with both forums on, so all five steps
    fire. There's one wrinkle: every demo task solves on the first attempt, and a
    solved task is normally dropped from later generations (`--drop-solved`, on by
    default), which would empty the task pool and **stop the run** before seeding
    ever fires. So the quickstart passes `--no-drop-solved` to retain the solved
    tasks and carry the full loop across all three generations. On your own tasks,
    leaving `--drop-solved` on (the default) is usually what you want — the loop
    then concentrates each generation on what's still unsolved. See
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

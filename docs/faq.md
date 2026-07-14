# Frequently asked questions

Common questions about KSI — from first-time setup through research use.

## What is KSI in one sentence?

KSI (Knowledge-centric Self-Improvement) is a benchmark framework that treats
agents as disposable workers and keeps improvement in a persistent shared
knowledge substrate rather than in any individual agent's memory or identity.
Transient agents run in sandboxed Docker containers, write attempts and forum
posts to SQLite-backed knowledge stores, distill reusable guidance, and seed
later generations from those distilled bundles.

## What problem does it solve — when should I use it?

KSI addresses the question of where improvement should live in an agentic
system. Its core thesis is that improvement should reside in durable shared
knowledge rather than in any individual agent; that an agent's workstream
should be an on-demand capability, not a persistent identity; and that the
resulting knowledge should be model-agnostic and transferable across model
families and new tasks. Use it when you want to study or deploy self-improving
agents on structured benchmarks (coding, reasoning, dialogue) and need a
principled, reproducible substrate for that improvement.

## When should I NOT use this?

KSI's overhead — Docker sandboxing, a SQLite knowledge substrate, multi-phase
forum/distill/seed generations — pays off when you're studying or running
*multiple generations of self-improvement across a task population*. It's
probably the wrong tool if any of these apply: you need a single agent to
solve one task right now with no multi-run learning loop (a plain agent call
is simpler and faster); you don't have Docker available or can't run
containers in your environment; you need sub-second iteration on prompt/tool
changes (each attempt is a full containerized run); or your benchmark doesn't
fit the task/evaluator model (a `TaskSpec` in, an `EvalResult` out — see
[programmatic_api.md](programmatic_api.md)) and adapting it isn't worth the
investment. If you're unsure, the Quickstart in the README costs one Docker
image build and a few cents of API usage to try.

## How is this different from just running an agent in a loop?

A naive agent loop accumulates knowledge only inside the agent's context window,
which is discarded when the session ends. KSI keeps improvement in a durable,
queryable SQLite knowledge substrate that persists across generations and is
shared by all agents; agents are disposable workers that are replaced and
re-seeded from distilled guidance at each generation boundary. This means
progress survives model swaps, restarts, and scale-out without relying on any
single agent's private memory. See [improvement_strategies.md](improvement_strategies.md)
for a full description of the per-generation loop.

## Do I really need Docker?

Yes. Every agent attempt runs in a sandboxed `ksi-agent:bench` container for
isolation and reproducibility — the container controls what tools the agent can
call, what files it can read, and what outputs it can produce. Build the image
once with `bash container/build.sh --bench` (or let `bash scripts/quickstart.sh`
do it on first run); after that, container startup is the only Docker cost per
task. If Docker is not running or the image is missing, `uv run ksi-doctor`
will tell you exactly what to fix.

## What models and providers can I use? Do I need an API key?

You supply your own API key. KSI supports Anthropic models (Haiku, Sonnet,
Opus) and OpenAI models through provider profiles stored under
`configs/ksi/`. The bundled profile templates are:

| Template | Model |
|----------|-------|
| `.env.haiku.template` | `claude-haiku-4-5-20251001` |
| `.env.sonnet.template` | `claude-sonnet-4-6` |
| `.env.opus.template` | `claude-opus-4-6` |
| `.env.sonnet35.template` | `claude-3-5-sonnet-20241022` |
| `.env.openai.template` | `gpt-5.4-mini` (with `REASONING_EFFORT=medium`) |

Copy the template for your provider, fill in your key, and pass the path with
`--provider-profile configs/ksi/.env.haiku` (or equivalent).

## What does a run cost?

Every run makes real LLM API calls billed to the key in your provider profile;
there is no built-in spending cap. Cost scales with the number of tasks,
generations, and the model you choose. To get a feel before committing,
start with the bundled synthetic demo: `bash scripts/quickstart.sh` runs
three tasks, one generation, with Haiku — the fastest and cheapest option.
Use `DRY_RUN=true` on any experiment wrapper script to print the full CLI
command and DB paths without launching anything.

## Where do my results go?

A run writes to several locations by default:

- **Log:** `/tmp/ksi-experiments/<experiment>.log`
- **Knowledge DB:** `runtime_state/knowledge/<experiment>/<experiment>_knowledge.sqlite`
- **Runtime DB (optional):** matching `<stem>_runtime.sqlite` audit sidecar
- **Traces:** `analysis/traces/<experiment>/`
- **Result summaries:** `results/<experiment>.json` or campaign-specific files
  under `results/`

Wrapper scripts under `benchmarks/` (and `scripts/run_ksi.sh`) write to
experiment-specific paths; run any wrapper with `DRY_RUN=true` to see the
exact paths before launching. See [artifacts.md](artifacts.md) for analysis
helper commands.

## What is the difference between the knowledge DB and the runtime DB?

`<stem>_knowledge.sqlite` is the authoritative knowledge substrate, owned by
`KnowledgeStore`. It stores run and generation state, task attempts, best
scores, forum posts, distilled guidance, seed snapshots, and full-text search
(FTS5) and optional vector indexes. This is what carries improvement across
generations and is the file you back up or share between experiments.

`<stem>_runtime.sqlite` is an optional audit sidecar, owned by `MemoryStore`.
It stores raw transcripts, artifacts, token-phase breakdowns, and
runtime/debug metadata. Disable it with `--no-runtime-db` if you do not need
it. See [architecture.md](architecture.md) for the full database ownership model.

## What is a "forum"? What do "distillation" and "seeding" mean?

- **Forum:** a structured multi-agent discussion phase where agents share what
  they learned from task attempts — either per-task (Phase 2) or cross-task
  (Phase 3).
- **Distillation:** compression of forum outputs into compact, reusable guidance
  bundles that are written to the knowledge DB (Phase 4).
- **Seeding:** injection of distilled bundles into the next generation's agents
  before they start executing tasks (Phase 5).

See the [Glossary](glossary.md) for precise definitions of all phases.

## What benchmarks are supported out of the box?

The following task sources and their paired evaluators ship with KSI:

| Benchmark | `--task-source` | `--evaluator` |
|-----------|-----------------|----------------|
| ARC-AGI-1 / ARC-AGI-2 | `arc` | `arc_session` |
| SWE-bench Pro | `swebench_pro` | `swebench_pro` |
| Polyglot | `polyglot` | `polyglot_harness` |
| Terminal-Bench 2 | `terminal_bench_2` | `terminal_bench_2` |

The bundled `examples/quickstart/arc_demo/` tasks are synthetic and are not
part of any official benchmark. See [BENCHMARK_PREPARE.md](https://github.com/recursive-knowledge/KSI/blob/main/benchmarks/docs/BENCHMARK_PREPARE.md)
for dataset download instructions.

## Can I add my own benchmark or evaluator?

Yes. Task sources and evaluators are registry-backed: add a `register_task_source`
or `register_evaluator` call and KSI picks it up with no edits to core dispatch
code. See [extending.md](extending.md) for the extension overview,
[adding_a_benchmark.md](adding_a_benchmark.md) for the task-source seam, and
[adding_an_evaluator.md](adding_an_evaluator.md) for the evaluator seam.

## Can I run KSI from Python instead of the CLI?

Yes. `ksi.run(...)` is the programmatic entry point and accepts the same
configuration as the CLI flags through a `GenerationConfig` dataclass (single
source of truth). A complete runnable example is at
`examples/programmatic/run_arc_demo.py`. See [programmatic_api.md](programmatic_api.md)
for the full API reference and migration notes if you are coming from an older
programmatic interface.

## What does `--improvement-strategy knowledge` vs `raw_attempts` do?

`knowledge` (the default) runs the full self-improvement loop each generation:
per-task forum, cross-task forum, distillation, and seeding. It is
behavior-preserving relative to the engine's historical defaults.

`raw_attempts` is the true knowledge-off ablation baseline: forums,
distillation, knowledge-guided seeding, and same-task enrichment (prior-attempt
history, best-score, memory-snapshot injection) are all skipped, regardless of
`--no-memory`; agents receive only raw-attempts seeding. Use it to measure how
much of the performance gain comes from the knowledge loop versus simply
accumulating more attempts. See
[improvement_strategies.md](improvement_strategies.md) for details and the
`register_strategy` seam to add custom strategies.

## My run failed or produced an empty knowledge DB — where do I start?

Run `uv run ksi-doctor` first; it prints a checklist of everything a run
needs and the exact command to fix each gap. The most common causes are: Docker
is not running, the `ksi-agent:bench` image has not been built (`bash
container/build.sh --bench`), or the API key in your provider profile is
missing or invalid. If the image builds but containers are slow to start, see
[runtime-startup-performance.md](runtime-startup-performance.md). If attempts
are completing but the knowledge DB stays empty, check the run log at
`/tmp/ksi-experiments/<experiment>.log` for `KnowledgeStore initialization`
errors.

## Quick troubleshooting reference

For anything not covered by `uv run ksi-doctor` or the entry above, check the
symptom against its first diagnostic step:

| Symptom | First check |
|---------|--------------|
| Image not found | `docker images ksi-agent:bench` |
| Slow container startup | [runtime-startup-performance.md](runtime-startup-performance.md) |
| Semantic vector search not active | Expected by default (FTS5 is the default). Only if you passed `--require-vector`: `uv sync --extra memory` and set `HF_TOKEN` |
| Empty knowledge DB | Confirm attempts are completing and `KnowledgeStore` initialization succeeded (check the run log for errors) |
| Forum hangs | Check `--forum-timeout-sec`, `--cross-task-forum-timeout-sec`, and `--forum-early-exit` |
| Stale run command | Run with `DRY_RUN=true bash benchmarks/<script>.sh ...` to print the exact CLI invocation without launching anything |


# Glossary

Core terms used throughout the KSI documentation; other pages link here instead of re-explaining each term.

### agent

A disposable worker that attempts a single task inside a sandboxed container. Agents are not persistent identities — each [generation](#generation) spawns fresh agents from the current [knowledge substrate](#knowledge-substrate). The number of agents in a generation equals the number of tasks in the filtered task pool.

### generation

One full iteration of the knowledge-refinement loop: all [agents](#agent) in the [population](#population) execute their tasks, then the [forum](#forum) runs, [distillation](#distillation) condenses the evidence into [bundles](#bundle), and [seeding](#seeding) prepares those bundles for the next generation.

### population

The set of [agents](#agent) working in a given [generation](#generation). Population size is derived automatically from the filtered task pool — one agent per task.

### task source

A registry-backed plugin that supplies tasks for agents to attempt. Maintained task sources: `arc`, `swebench_pro`, `polyglot`, `terminal_bench_2`. Register a new one via `src/ksi/tasks/registry.py`. See [Adding a benchmark](adding_a_benchmark.md) for details.

### evaluator

A registry-backed plugin that scores an [agent's](#agent) [attempt](#attempt) against a task. Maintained evaluators: `none`, `arc_session`, `swebench_pro`, `polyglot_harness`, `terminal_bench_2`. Register a new one via `src/ksi/eval/registry.py`.

### runtime

The registry-backed execution backend that runs an [agent](#agent) in a container and collects its outputs. The only maintained runtime is `container`. Backed by `src/ksi/runtime/registry.py`; selected with `--runtime container`.

### forum

The structured discussion phases that follow task execution within a [generation](#generation). There are two: a per-task forum (agents share what worked on a specific task, controlled by `--per-task-forum-rounds`) and a cross-task forum (agents share general observations across all tasks, controlled by `--cross-task-forum-rounds`). Both phases write into the [knowledge substrate](#knowledge-substrate).

### distillation

The phase after the [forum](#forum) in which in-process distillers compress raw evidence — [attempts](#attempt), scores, and discussion posts — into reusable per-task and cross-task [bundles](#bundle). Disabled with `--distill-enabled false`.

### seeding

The final phase of a [generation](#generation): distilled [bundles](#bundle) are injected into the [knowledge substrate](#knowledge-substrate) as the starting knowledge for the next generation's [agents](#agent). Seeding is part of the `knowledge` [improvement strategy](#improvement-strategy) and is bypassed when `--no-memory` is set.

### knowledge substrate

The authoritative `<stem>_knowledge.sqlite` database owned by `KnowledgeStore`. It stores run/generation/task state, [attempts](#attempt), best scores, [forum](#forum) posts, distilled guidance, seed snapshots, FTS5 indexes, and optional sqlite-vec indexes. Set its path with `--knowledge-db-path`. Also referred to as the "knowledge DB."

### runtime DB

The optional `<stem>_runtime.sqlite` audit sidecar owned by `MemoryStore`. It stores raw transcripts, artifacts, token phases, and runtime/debug metadata. Enabled by default; disabled with `--no-runtime-db`; set its path with `--runtime-db-path`. Distinct from the [knowledge substrate](#knowledge-substrate), which is the authoritative store.

### improvement strategy

Determines how the system improves between [generations](#generation). `knowledge` (the default) runs the full forum → distill → seed loop. `raw_attempts` skips knowledge transfer and serves as a clean baseline. Selectable with `--improvement-strategy {knowledge,raw_attempts}`; backed by `src/ksi/orchestrator/strategy.py`.

### provider profile

A local `.env` file under `configs/ksi/` that selects the model and API provider for a run (e.g. `MODEL`, `MODEL_PROVIDER`, API keys). Copy from the committed templates and keep real keys untracked. Pass a profile with `--provider-profile`.

### attempt

One agent's try at a task in a given [generation](#generation). Every attempt is recorded in the [knowledge substrate](#knowledge-substrate) regardless of score, enabling resume, best-score tracking, and [forum](#forum) discussion.

### bundle

A packaged set of distilled knowledge that can be exported from a [knowledge substrate](#knowledge-substrate) as a JSON artifact (KT export) and injected into a new run with `--seed-bundle-path` to transfer cross-task guidance across experiments.

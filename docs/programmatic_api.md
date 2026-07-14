# Programmatic API

The CLI (`uv run python -m kcsi.cli`) is one front door; `kcsi.run(...)` is
the other. Use it to drive a generational run from Python or a notebook without
argparse.

```python
import os
from pathlib import Path

import kcsi
from kcsi.benchmarks import ArcSessionEvaluator
from kcsi.runtime import KcsiContainerExecutor
from kcsi.runtime.llm import build_llm_caller
from kcsi.tasks.loaders import load_tasks_for_source

tasks = load_tasks_for_source(task_source="arc", tasks_path=my_tasks_path)

knowledge_db_path = str(Path("runtime_state") / "programmatic_api_knowledge.sqlite")
config = kcsi.GenerationConfig(num_generations=1, num_agents=1, knowledge_db_path=knowledge_db_path)
llm = build_llm_caller(provider="anthropic", model="claude-sonnet-4-6")
evaluator = ArcSessionEvaluator()

# Provider env is REQUIRED — the container host validates provider auth and
# raises before running if it is missing. Unlike the CLI, `kcsi.run` does NOT
# load a provider profile for you, so pass the env explicitly. This minimal
# dict is what the anthropic/api path needs; supply your own key via the
# environment (e.g. `export ANTHROPIC_API_KEY=...`).
provider_env = {
    "MODEL_PROVIDER": "anthropic",
    "MODEL": "claude-sonnet-4-6",
    "MODEL_AUTH_MODE": "api",
    "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
}

runtime = KcsiContainerExecutor(
    command=["npx", "--yes", "--prefix", "runtime_runner", "tsx", "runtime_runner/src/main.ts"],
    working_dir=".",
    knowledge_db_path=config.knowledge_db_path,
    env=provider_env,
)

traces = kcsi.run(config, tasks, runtime=runtime, evaluator=evaluator, llm=llm)
```

For OpenAI, use `{"MODEL_PROVIDER": "openai", "MODEL": "gpt-5.4-mini",
"MODEL_AUTH_MODE": "api", "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]}`
(anthropic subscription auth uses `MODEL_AUTH_MODE="subscription"` with
`CLAUDE_CODE_OAUTH_TOKEN` instead of `ANTHROPIC_API_KEY`).

Rather than hand-building this dict, you can reuse the exact helper the CLI
uses to load a [provider profile](./glossary.md#provider-profile) — the local
`.env.*` files under `configs/kcsi/` (copy a committed `*.template`, add your
real key, keep it untracked; same files `--provider-profile` reads):

```python
from kcsi.providers import load_provider_profile

provider_env = load_provider_profile("configs/kcsi/.env.sonnet")
runtime = KcsiContainerExecutor(..., knowledge_db_path=config.knowledge_db_path, env=provider_env)
```

A complete runnable script is in
[`examples/programmatic/run_arc_demo.py`](https://github.com/recursive-knowledge/KCSI/blob/main/examples/programmatic/run_arc_demo.py).

`GenerationConfig` requires `num_generations` and `num_agents`; pass those
explicitly, so the minimal valid form is
`GenerationConfig(num_generations=1, num_agents=1)` (add `knowledge_db_path=...`
to enable the knowledge substrate). The remaining fields carry the same
defaults as the CLI flags (single source of truth) — notably
`cross_task_forum_rounds=2`, so a minimal knowledge-enabled config such as
`GenerationConfig(num_generations=1, num_agents=1, knowledge_db_path=...)`
runs two cross-task forum rounds per generation (the engine's old
programmatic fallback was 1). Set fields explicitly to trim phase-3
container/LLM cost, e.g.
`GenerationConfig(num_generations=1, num_agents=1, knowledge_db_path=..., cross_task_forum_rounds=1)`.
To disable forums programmatically, zero both `per_task_forum_rounds` and
`cross_task_forum_rounds`. (The legacy `forum_rounds` field was removed along
with the `--forum-rounds` CLI flag, which now hard-errors.)
One migration note: `experiment_name` now defaults to `"kcsi"` everywhere
(previously the programmatic default was `"default"` and the CLI default was
`"swarms_v2"`). Resuming an experiment created under either old default
requires passing the old name explicitly (`experiment_name="default"` or
`--experiment-name swarms_v2`) — otherwise `resume=True` looks up the new
name, finds nothing, and starts fresh.

## The contract

`kcsi.run(config, tasks, *, runtime, evaluator, llm, persistence=None,
working_dir=".") -> list[TaskTrace]` is a thin, behavior-preserving wrapper
around `GenerationalOrchestrator(...).run(tasks=...)` — the orchestrator
construction and run match the CLI path. CLI-only conveniences are not applied:
provider profile / environment loading and signal handling are your
responsibility.
You construct `runtime` / `evaluator` / `llm` yourself, which keeps every run
explicit about what it depends on.

## Building blocks

| Piece | Where | Notes |
|-------|-------|-------|
| `run` | `kcsi.run` | the entry point |
| `GenerationConfig` | `kcsi.GenerationConfig` | run parameters |
| `GenerationalOrchestrator` | `kcsi.GenerationalOrchestrator` | the engine (use `run` instead of constructing directly unless you need to) |
| LLM caller | `kcsi.runtime.llm.build_llm_caller` | provider/model → `LLMCaller` |
| Evaluators | `kcsi.eval` (`NoopEvaluator`) / `kcsi.benchmarks` (e.g. `ArcSessionEvaluator`) | or build via the [evaluator registry](./adding_an_evaluator.md) |
| Runtimes | `kcsi.runtime.KcsiContainerExecutor` | or build via the runtime registry |
| Tasks | `kcsi.tasks.loaders.load_tasks_for_source` | or construct `TaskSpec(...)` directly |
| Persistence/callbacks | `kcsi.protocols.PersistenceObserver` | optional; the CLI's `SqlitePersistence` is one implementation |

## Custom tasks

`load_tasks_for_source` and `CommandEvaluator` work the same way for the
built-in `custom` task source as for any benchmark — swap `task_source="arc"`
+ `ArcSessionEvaluator` for `task_source="custom"` + `CommandEvaluator`:

```python
from kcsi.eval.command import CommandEvaluator
from kcsi.tasks.loaders import load_tasks_for_source

tasks = load_tasks_for_source(task_source="custom", tasks_path="tasks.jsonl")
evaluator = CommandEvaluator()
```

`tasks.jsonl` is the same JSON/JSONL record file the CLI's
`--task-source custom --tasks-path` expects; you can also skip the file and
construct `TaskSpec` objects directly. See
[Your own tasks](your_own_tasks.md) for the full record schema, the
`command` evaluator's scoring contract, and both forms side by side.

## Typed package

`kcsi` ships a PEP 561 `py.typed` marker, so type checkers (mypy/pyright) see
its annotations when you build on the public API.

## Prerequisites

`kcsi.run` executes agents in containers, so the same prerequisites as the CLI
apply: Docker running, the `kcsi-agent:bench` image built, runtime_runner Node
deps installed, and a provider API key in the environment. See the
[README](https://github.com/recursive-knowledge/KCSI/blob/main/README.md) and [docs/architecture.md](./architecture.md).

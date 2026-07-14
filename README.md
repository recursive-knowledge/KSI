# KSI — Knowledge-centric Self-Improvement

KSI runs a population of disposable agents on **your own tasks**, each
attempting independently inside a sandboxed container. They share what worked
in a structured forum, and the system distills that discussion into reusable
guidance that seeds the next generation. Improvement lives in a shared
knowledge store — not in any single agent — so it survives across runs.

**[Quickstart](#quickstart)** · **[Your own tasks](#your-own-tasks)** ·
**[Docs site](https://recursive-knowledge.github.io/KSI/)** ·
**[Reference benchmarks](#reference-benchmarks)**

## Quickstart

From a fresh clone to a solved task in **one command** — no dataset download,
no prior setup step. With Docker and Node.js 22.16.0 installed (and either
`uv`, or a local editable install via `pip install -e .`), just provide an
API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...    # or: export OPENAI_API_KEY=sk-...
bash scripts/quickstart.sh
```

The script self-bootstraps everything it needs: it synthesizes a provider
profile from your key, builds the `ksi-agent:bench` image on first run,
installs the host Node dependencies, then runs one generation over the
bundled [`examples/custom_tasks/`](./examples/custom_tasks/) demo — three
small, self-contained Python tasks (`fizzbuzz`, `reverse-words`,
`anagram-groups`), each graded by running `python3 tests.py` against the
agent's attempt.

Already ran the full setup? `uv run ksi-doctor` prints a ✓/✗ checklist of
everything a run needs — Docker, the agent image, host Node deps, and a
provider profile with a real key — plus the exact command to fix anything
missing.

→ Full walkthrough: [Getting started](docs/getting-started.md).

## Your own tasks

Point KSI at any JSON/JSONL file of task records — no benchmark dataset and
no custom loader code required. Each record:

```jsonc
{
  "task_id": "my-task-1",                     // required, unique
  "prompt": "…instruction for the agent…",    // required
  "workspace_dir": "path/to/starting/files",  // optional, dir; relative to the tasks file
  "files": {"relative/path.py": "content"},   // optional, inline alternative to workspace_dir
  "eval": {"command": "python3 tests.py",     // optional; graded by the `command` evaluator
           "timeout_sec": 300}
}
```

Save that (or the bundled `examples/custom_tasks/tasks.jsonl`) as
`tasks.jsonl` and run it:

```bash
uv run python -m ksi.cli \
  --task-source custom \
  --tasks-path tasks.jsonl \
  --evaluator command \
  --provider-profile configs/ksi/.env.haiku
```

Or drive the same run from Python with `ksi.run(...)` — no argparse, no CLI:

```python
import ksi
from ksi.eval.command import CommandEvaluator
from ksi.providers import load_provider_profile
from ksi.runtime import KsiContainerExecutor
from ksi.runtime.llm import build_llm_caller
from ksi.tasks.loaders import load_tasks_for_source

tasks = load_tasks_for_source(task_source="custom", tasks_path="tasks.jsonl")

config = ksi.GenerationConfig(
    num_generations=1,
    num_agents=1,
    experiment_name="my_custom_run",
    knowledge_db_path="runtime_state/my_custom_run_knowledge.sqlite",
)

provider_env = load_provider_profile("configs/ksi/.env.haiku")
llm = build_llm_caller(provider=provider_env["MODEL_PROVIDER"], model=provider_env["MODEL"])
runtime = KsiContainerExecutor(
    command=["npx", "--yes", "--prefix", "runtime_runner", "tsx", "runtime_runner/src/main.ts"],
    working_dir=".",
    knowledge_db_path=config.knowledge_db_path,
    env=provider_env,
)

traces = ksi.run(config, tasks, runtime=runtime, evaluator=CommandEvaluator(), llm=llm)
```

Full contract — eval-command semantics, `score.json` partial credit, the
workspace/`repo/` layout, and a security note on running untrusted tasks
files — is in [docs/your_own_tasks.md](docs/your_own_tasks.md).

## How it works

1. A population of agents each attempt your tasks in isolated containers.
2. They discuss what worked in a structured [forum](docs/glossary.md#forum).
3. The system [distills](docs/glossary.md#distillation) the discussion into
   reusable guidance.
4. The next generation is [seeded](docs/glossary.md#seeding) with that
   guidance.

See [docs/architecture.md](docs/architecture.md) for the full runtime design.

## Reference benchmarks

KSI also ships evaluators and task-map infrastructure for ARC-AGI-1/2,
SWE-bench Pro, Polyglot, and Terminal-Bench 2, so the same CLI can run
against those instead of your own tasks. Dataset preparation, task maps, and
run presets for each live under [`benchmarks/`](benchmarks/README.md) — start
there for setup and for the licensing/attribution details of those
third-party task corpora.

## Requirements & setup

- **Docker** — sandboxes each agent in an isolated container.
- **Node.js 22.16.0** — the runtime host (`runtime_runner`) is TypeScript;
  see `.nvmrc`.
- **Python 3.12+ with `uv`** — used to invoke the CLI, doctor, and all Python
  tooling.
- **An API key** — `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.

`bash scripts/setup_all.sh` installs Python dependencies, builds the agent
container image, and generates provider profile templates; pass `--no-test`
to skip the pytest check. `uv run ksi-doctor` prints a ✓/✗ readiness
checklist (Docker, image, Node deps, provider profile) with the exact command
to fix anything missing.

## Licensing

ksi's own code is licensed under [Apache-2.0](./LICENSE). Task-map manifests
committed under `benchmarks/*/task_maps/*.json` are KSI-authored under the
same license; the reference-benchmark **datasets** themselves are third-party
and remain under their own upstream licenses — see
[benchmarks/README.md](./benchmarks/README.md#licensing--attribution) for
sources and attribution.

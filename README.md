# KSI — Knowledge-centric Self-Improvement

[![Documentation](https://img.shields.io/badge/docs-recursive--knowledge.github.io%2FKSI-3f51b5?logo=materialformkdocs&logoColor=white)](https://recursive-knowledge.github.io/KSI/)

KSI runs a population of disposable agents on **your own tasks**, each
attempting independently inside a sandboxed container. They share what worked
in a structured forum, and the system distills that discussion into reusable
guidance that seeds the next generation. Improvement lives in a shared
knowledge store — not in any single agent — so it survives across runs.

Point it at any JSON/JSONL file of task records — no benchmark dataset and no
loader code required — or at the bundled reference benchmarks (ARC-AGI-1/2,
SWE-bench Pro, Polyglot, Terminal-Bench 2).

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
bundled [`examples/custom_tasks/`](./examples/custom_tasks/) demo. If
anything is missing, `uv run ksi-doctor` prints a ✓/✗ readiness checklist
with the exact command to fix it.

## Documentation

The **[docs site](https://recursive-knowledge.github.io/KSI/)** is the
canonical reference:

- **[Getting started](https://recursive-knowledge.github.io/KSI/getting-started/)** —
  full walkthrough: requirements, `setup_all.sh`, provider profiles, first run
- **[Your own tasks](https://recursive-knowledge.github.io/KSI/your_own_tasks/)** —
  the task-record schema, the `command` evaluator's scoring contract, and the
  workspace/`repo/` layout
- **[Programmatic API](https://recursive-knowledge.github.io/KSI/programmatic_api/)** —
  drive the same runs from Python with `ksi.run(...)`, no CLI
- **[Architecture](https://recursive-knowledge.github.io/KSI/architecture/)** —
  how a generation works: attempts → forum → distillation → seeding
- **[Extending KSI](https://recursive-knowledge.github.io/KSI/extending/)** —
  add a task source, evaluator, runtime, or improvement strategy
- **[Benchmarks](https://recursive-knowledge.github.io/KSI/benchmarks/)** —
  dataset preparation and run presets for the reference benchmarks
- **[FAQ](https://recursive-knowledge.github.io/KSI/faq/)**

The same pages are browsable as Markdown under [`docs/`](./docs/) in this
repo, and benchmark-specific setup lives in
[`benchmarks/`](./benchmarks/README.md).

For the research behind KSI — the method narrative, results, and interactive
knowledge dashboards — see the
**[paper page](https://recursive-knowledge.github.io/knowledge-centric-self-improvement/)**.

## Licensing

ksi's own code is licensed under [Apache-2.0](./LICENSE). Task-map manifests
committed under `benchmarks/*/task_maps/*.json` are KSI-authored under the
same license; the reference-benchmark **datasets** themselves are third-party
and remain under their own upstream licenses — see
[benchmarks/README.md](./benchmarks/README.md#licensing--attribution) for
sources and attribution.

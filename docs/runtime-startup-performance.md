# Runtime Startup Performance

Current notes for the path from `uv run python -m ksi.cli ...` to the first
agent API call.

## Current Hot Path

```text
ksi.cli
  -> GenerationalOrchestrator
  -> KsiContainerExecutor
  -> runtime_runner/src/main.ts
  -> runtime_runner/src/container_runner.ts
  -> Docker container
  -> container/entrypoint.sh
  -> runtime_runner/agent-runner/src/index.ts
  -> provider SDK + MCP tools
```

## Shipped Improvements

- TypeScript is precompiled into the Docker image. Container startup performs a
  source-hash check and skips `tsc` when the mounted source matches the baked
  `dist/`.
- Lexical FTS5 retrieval is the default and backs the agent-facing MCP `query`
  tool with no embedding model to load. Semantic vector search is opt-in via
  `--require-vector`; on that path the embedding model loads synchronously and
  the run fails fast if it (or sqlite-vec) can't initialize. The `query`
  response carries a `retrieval_mode` field (`semantic` / `fts` / `none`) so the
  mode in effect is visible per call.
- The old three-file memory layout has been replaced by a runtime DB plus a
  sibling knowledge DB, so the orchestrator no longer opens three legacy memory
  stores on startup.
- `sentence-transformers==3.2.1` and `transformers<5` are pinned to avoid a
  known embedder-load break.

## Remaining Bottlenecks

### MCP Process Startup

Each container can start multiple MCP processes:

- KSI MCP server
- memory MCP server when knowledge tools are enabled
- ARC MCP tools for ARC tasks

Python process startup plus imports are paid per container. Combining memory
and ARC tools into one Python MCP process would reduce repeated startup cost.

### Workspace Rebuilds

`--wipe-workspace-per-task true` is the default. The runtime removes and
rebuilds the task workspace before each attempt, then copies or mounts task
repo content. This is clean and deterministic, but expensive for large
SWE-bench Pro repositories.

Potential fix: cache workspace materialization by repo/source hash and reuse it
when the next task has identical source content.

### Agent-Runner Source And Skill Sync

The runtime copies agent-runner source and container skills into the session
area. These copies are safe but mostly mechanical.

Potential fix: hash or timestamp the source directory and skip copies when the
same session group already has the current content.

### MCP Polling Tail Latency

The agent-runner still has polling paths for IPC-style communication. A polling
interval adds tail latency and unnecessary filesystem reads.

Potential fix: use shorter polling for scheduled benchmark tasks or switch to an
event-driven file notification path where available.

### SWE-bench Pro Repository Size

SWE-bench Pro tasks can involve large checked-out repositories and Docker-based
evaluation. Repo cache misses, repeated copies, and evaluator container setup
dominate startup more than Python orchestration overhead.

Potential fix: pre-check repo snapshots, prefer local cache hits, and keep
evaluation Docker images warm before a campaign.

## Rough Timing

Warm-image, local-cache expectation:

| Phase | Typical Cost |
|-------|--------------|
| CLI parse, config, task loading | ~1s |
| runtime + DB initialization | ~1s |
| Docker container creation | ~1s |
| TypeScript hash check | ~0.5s |
| workspace and file copies | ~1-5s |
| MCP process startup | ~1-6s |
| provider SDK query setup | ~1-3s |

Cold starts can be much slower when Docker images, HuggingFace models,
Node dependencies, SWE-bench Pro repos, or Polyglot images are missing.

## Practical Preflight

```bash
docker images ksi-agent:bench
npm --prefix runtime_runner ci
uv sync --extra memory
```

For Polyglot:

```bash
docker images ksi-polyglot-eval:latest
```

For SWE-bench Pro, verify the dataset, task map, and repo cache paths before
launching a high-concurrency run.

## See also

- [architecture.md §1 System Map](./architecture.md#1-system-map) — the
  full component chain this page's hot path traces through.
- [faq.md](./faq.md#my-run-failed-or-produced-an-empty-knowledge-db-where-do-i-start) —
  the general troubleshooting entry point; this page is specifically about
  *slow*, not *failing*, runs.

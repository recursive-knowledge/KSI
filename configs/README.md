# `configs/`

Configuration is grouped **by who consumes it**:

| Directory | Consumer | What it holds |
|-----------|----------|---------------|
| `ksi/` | The KSI runtime (`--provider-profile`) | Provider *profiles* — one `.env` per model backend (`MODEL_PROVIDER`, `MODEL_AUTH_MODE`, `MODEL`, `REASONING_EFFORT`, plus the matching API key). |
| `benchmarks/` | Benchmark task selection | Static JSON defaults (e.g. `arc_defaults.json`). |

## `ksi/` — runtime provider profiles

Copy a committed `*.template` to a real (untracked) profile and fill in the key:

```bash
cp -n configs/ksi/.env.haiku.template configs/ksi/.env.haiku
vim configs/ksi/.env.haiku      # set ANTHROPIC_API_KEY=sk-ant-...
```

Then launch with `--provider-profile configs/ksi/.env.haiku`. Available
templates: `haiku`, `sonnet`, `sonnet35`, `opus`, `openai`. `.env.shared.example`
documents the shared provider-credential block.

`scripts/setup_all.sh` generates the common profiles automatically.

## Tracking

Real profiles/configs are gitignored; only `*.example` / `*.template` files are
committed (see the `configs/` rules in `.gitignore`). Keep real API keys
untracked.

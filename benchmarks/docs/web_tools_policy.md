# Web-Tool Policy for Benchmark Tasks (`KCSI_ALLOW_WEB_TOOLS`)

> Integrity fix.
> Default-OFF web tools is the new baseline behavior and creates a **code-era
> boundary**:
> results recorded *before* this fix had web tools available on non-ARC Claude
> runs and are not directly comparable to post-fix results.

## Policy

Native web tools (`WebSearch`, `WebFetch`) are **OFF by default for every
benchmark task**. They are offered to the Claude agent only when the operator
explicitly opts in by setting the environment variable `KCSI_ALLOW_WEB_TOOLS`
to a truthy value (anything other than empty / `0` / `false` / `no` / `off`):

```bash
KCSI_ALLOW_WEB_TOOLS=1 uv run python -m kcsi.cli ...
```

ARC stays **strictly offline regardless of the flag** — the existing
`isOffline` (ARC) guard always wins, so an ARC run never has web tools even
with `KCSI_ALLOW_WEB_TOOLS=1`.

| Task source         | Default (flag unset) | `KCSI_ALLOW_WEB_TOOLS=1` |
|---------------------|----------------------|----------------------------|
| `arc`               | denied               | denied (ARC always offline)|
| `swebench_pro`      | denied               | enabled                    |
| `polyglot`          | denied               | enabled                    |
| `terminal_bench_2`  | denied               | enabled                    |

> **Cross-provider warning.** `KCSI_ALLOW_WEB_TOOLS=1` re-enables web tools
> for the **Claude loop only** — the OpenAI agent loop has no web tools and no
> equivalent flag. Enabling it therefore reintroduces the Claude-vs-GPT scaffold
> asymmetry. Do not
> set it for cross-provider comparison runs; reserve it for explicitly
> web-allowed, single-provider experiments.

## Rationale

Web tools are a **benchmark-solution leak vector**:

- **SWE-bench Pro**: an agent can fetch/clone the task's own upstream repo,
  which contains the future fix commit — i.e. the answer. Paper-era traces show
  successful `git clone` of `django` / `sphinx` *mid-attempt*, plus
  hundreds of `WebFetch`/`WebSearch` calls per cell.
- **Polyglot**: Exercism exercises have published solutions that a
  `WebSearch`/`WebFetch` turn can retrieve directly.
- **Provider asymmetry**: only Claude ever received these tools. The OpenAI
  agent loop has no web tools, so the exposure was Claude-side only — a
  scaffold asymmetry.

There is no prompt-level prohibition against looking up solutions on non-ARC
tasks, so removing the tools is the reliable mitigation.

The bullets above describe the *affordance*. The per-trace audit
quantified what was **realized** in the paper-era cells: solution-seeking web
use appeared **only in the Haiku Polyglot cell** (6/40 = 15% of *solved* tasks
fetched Exercism tests / canonical-data before grading). The SWE-bench Pro,
ARC, terminal_bench_2, and all OpenAI cells were benign (no solution lookup) —
the SWE `git clone` traces were dependency fetches, not the graded fix. So the
integrity exposure was broad in *capability* but narrow in *realization*.

## Baseline fairness

The provider asymmetry above is *internal* to kcsi (Claude loop vs OpenAI
loop). There is also an **external** one: the published baselines never had web
tools at all. The DGM and HyperAgents agents are given only a `bash` + `editor`
tool surface — no `WebSearch`/`WebFetch`, and no equivalent. So every pre-fix
**non-ARC kcsi-Claude** cell (SWE-bench Pro, Polyglot, terminal_bench_2) was
obtained with a capability the baselines lacked, biasing the kcsi-vs-baseline
head-to-head in kcsi' favor — and per the web-tool audit that bias was realized
in the Haiku Polyglot cell (6/40 solved tasks). Pre-fix kcsi-vs-baseline
non-ARC comparisons should therefore be **re-run post-fix before being
published as head-to-head**; ARC and OpenAI cells are unaffected (no web tools
either way).

## Enforcement mechanism (the critical correctness point)

The main agent query runs with `systemPrompt: { preset: 'claude_code' }`. The
`claude_code` preset loads the **full default Claude Code tool surface**
(WebSearch/WebFetch included) into the model's context. Consequently:

- `allowedTools` is an **allowlist that gates invocation**, but *omitting* a
  tool from it does **not** remove a preset-loaded tool from the model's
  context — the model can still attempt it.
- `disallowedTools` is the only field that **removes a tool outright**
  ("These tools will be removed from the model's context and cannot be used,
  even if they would otherwise be allowed" — the SDK's `disallowedTools` JSDoc).
  `runtime_runner/agent-runner/package.json` pins `@anthropic-ai/claude-agent-sdk`
  at the exact version `0.1.77`; there the field is documented in
  `entrypoints/sdk/runtimeTypes.d.ts` (line ~281 in `0.1.77`). The exact
  file/line varies by SDK version (newer majors move it to `sdk.d.ts`), so
  treat the JSDoc, not the path, as canonical. `sdk.mjs` forwards it to the
  Claude Code CLI as `--disallowedTools`.

Therefore the denial is enforced by **`disallowedTools`**, not by mere
omission. `runtime_runner/agent-runner/src/query_config.ts` (`buildToolPolicy`)
adds `['WebSearch', 'WebFetch']` to `disallowedToolsList` whenever web tools are
not enabled (ARC always, and every benchmark unless `KCSI_ALLOW_WEB_TOOLS=1`).
The reflection/follow-up turn already runs with `allowedTools: []` and lists the
web tools in its `disallowedTools`, so it was never a leak path.

The agent logs one self-documenting line at start, e.g.:

```
Web tools (WebSearch/WebFetch): DISABLED [default-off (set KCSI_ALLOW_WEB_TOOLS=1 to enable)]
```

## Flag threading

The truthiness parse matches the Python `_is_enabled_env` helper
(`src/kcsi/runtime/container_host.py`); the env-var path is:

```
src/kcsi/runtime/container_host.py (_build_runner_env, setdefault from os.environ)
  -> runtime_runner/src/container_runner.ts  (env allowlist forwarded with -e)
  -> container process.env
  -> runtime_runner/agent-runner/src/query_config.ts (buildToolPolicy ->
     buildWebToolGating -> isWebToolsAllowed, in ./web_tools)
```

A provider-profile value in `base_env` takes precedence over the host
`os.environ` value (`setdefault` is a no-op when the key is already present).
The host side mirrors the same resolution when building `TOOLS.md`, so the
agent's tool list reflects the effective web-tool state.

## Residual exposure: shell-level egress

This fix removes the **agent-facing web tools**. The shell-level egress channel
(`curl`, `wget`, `git clone`) that this section originally flagged as future
work has since been closed by default: isolated agent containers now run on a
Docker `--internal` network behind an allowlisting CONNECT proxy sidecar,
with external DNS blackholed via `--dns 0.0.0.0`, and SWE-bench `.git`
history is sanitized so the fix commit is unreadable offline. The proxy
only permits the provider API hosts derived at launch (extendable via
`KCSI_EGRESS_ALLOW`). The legacy direct-bridge behavior is restored only by the
explicit `KCSI_EGRESS=open` escape hatch — debugging only, never production. See
[docs/architecture.md §10](../../docs/architecture.md#10-egress-isolation) for the full topology.

The `Task` sub-agent tool is in the same residual class: on benchmark runs it
is omitted from `allowedTools` but, like any preset tool, is not removed unless
disallowed, and a spawned sub-agent inherits the parent's `bypassPermissions`
mode and `Bash`. It reaches only the **already-described shell egress** (it does
not reopen `WebSearch`/`WebFetch`, which the inherited deny rules still block),
so it adds no new channel beyond `curl`/`wget`/`git clone`.

## Deploying this change

The web-tool gate lives in `runtime_runner/agent-runner/src` (`query_config.ts`
(`buildToolPolicy`) + `web_tools.ts`), which the container host **mounts from disk and recompiles at
container startup** when the source hash changes (`entrypoint.sh`). Editing
those files is therefore picked up on the next run **without** an image rebuild;
a `docker build` is only needed if you change baked image layers (deps,
Dockerfile). The gating decision is also unit-tested without the container:
`node --test tests/js/web_tools_gating.test.mjs` (behavioral, via `tsx`).

## Era implication

| Era | Non-ARC Claude web tools |
|-----|--------------------------|
| Before this fix | `WebSearch` + `WebFetch` **available** on every non-ARC task |
| After this fix  | **Denied by default**; opt-in per run via `KCSI_ALLOW_WEB_TOOLS=1` |

When comparing or pooling results across this boundary, treat pre-fix non-ARC
Claude numbers as confounded by potential solution lookup. See the code-era
boundary reference for the canonical comparability table.

## See also

- [tb2_native_tools.md](./tb2_native_tools.md) for the *different*,
  unrelated TB2 native file-operation tools (read/write/edit/glob/grep) —
  not agent-facing web access.
- [`src/kcsi/memory/parity.py`](https://github.com/recursive-knowledge/KCSI/blob/main/src/kcsi/memory/parity.py)
  for the general feedback-channel/leakage rules this web-tool gate is one
  instance of.

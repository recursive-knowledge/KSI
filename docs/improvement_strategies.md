# Improvement Strategies (the engine seam)

The generational orchestrator runs a self-improvement loop each generation:

```
execute attempts         (Phase 1)
  -> per-task forum      (Phase 2)
  -> cross-task forum    (Phase 3)
  -> distill             (Phase 4)
  -> seed next generation (Phase 5)
```

Historically these phases were welded into `GenerationalOrchestrator.run()` as
inline private-method calls, gated by a maze of flags
(`per_task_forum_rounds`, `cross_task_forum_rounds`, `distill_enabled`,
`no_memory`, seed flags). The project's research goal is to *compare*
improvement mechanisms (raw-attempts, no-forum, generic-preamble, prompt
evolution), so the mechanism needs to be swappable rather than flag-encoded.

`src/ksi/orchestrator/strategy.py` introduces the `ImprovementStrategy` seam to
make that swap possible.

## The seam

`ImprovementStrategy` is a `Protocol` with one hook per phase boundary:

| Hook | Generation phase | Default delegates to |
|------|------------------|----------------------|
| `per_task_forum(ctx)` | Phase 2 | `ctx.phases.per_task_forum(...)` |
| `cross_task_forum(ctx)` | Phase 3 | `ctx.phases.cross_task_forum(...)` |
| `distill(ctx)` | Phase 4 | `ctx.phases.distill(...)` |
| `plan_seed_next_generation(ctx, remaining_tasks=...)` | Phase 5 schedule | `seed`, `skip`, or `stop` plus the next task set |
| `seed_next_generation(ctx)` | Phase 5 | `ctx.phases.seed_next_generation(...)` |

`GenerationContext` (a dataclass) is the thin per-generation capability view
passed to each hook. It exposes exactly what the inline calls consumed:

- `phases` — explicit phase capabilities for per-task forum, cross-task forum,
  distillation, and seeding. The engine-backed implementation is
  `EngineImprovementPhaseServices`; forum delegates to
  `EngineForumPhaseService`, distillation delegates to
  `EngineDistillationPhaseService`, and seeding delegates to
  `EngineSeedingPhaseService`.
- `generation` — 1-based generation index.
- `fresh_traces` — traces from this generation, excluding carried-forward
  (already-solved) ones.
- `next_task_pool_size` — count of tasks carrying into the next generation.
- `next_remaining_tasks(remaining_tasks)` — callback for applying the engine's
  configured task-retention policy (for example `--drop-solved`) without a
  strategy reaching into engine internals.
- `config` / `knowledge` — read-only run handles for strategy inspection.
- `distill_task_ids()` — the de-duplicated, sorted, non-blank `task_id` list
  consumed by the distillation phase service.

The engine constructs `DefaultKnowledgeStrategy()` in `__init__` and invokes the
hooks in `run()`. Seed scheduling is strategy-owned via
`plan_seed_next_generation(...)`: the default strategy returns `skip` after the
final generation or under `--no-memory`, `stop` when the configured next-task
set is empty, and `seed` with the selected next task set otherwise. The engine
**keeps** the surrounding try/except + `AuthenticationFailure` re-raise policy,
the per-generation token flush, and the `persistence` callbacks — those are not
improvement-mechanism concerns.

Swap the strategy programmatically:

```python
orch = GenerationalOrchestrator(...)
orch.set_improvement_strategy(RawAttemptsStrategy())
```

## Bundled strategies

- **`DefaultKnowledgeStrategy`** — current behaviour, extracted verbatim via
  phase-service delegation. Adds no new branching; every gating flag still
  works as before.
- **`RawAttemptsStrategy`** — the true knowledge-off ablation: no forums, no
  distillation, and no same-task enrichment (prior-attempt history,
  best-score, memory-snapshot injection are all skipped via
  `should_enrich() -> False`); seeding still runs. It is the seam-level
  equivalent of the flag combination
  `--per-task-forum-rounds 0 --cross-task-forum-rounds 0 --distill-enabled false`
  (+ raw-attempts seeding), narrowed further by disabling enrichment.
  `tests/test_improvement_strategy.py` asserts it produces the same
  forum/distill engine calls as that combination;
  `tests/test_enrich_seed_packages.py` asserts the enrichment-skip behavior.

## Selecting a strategy

Strategies are registry-backed (mirroring the task-source / evaluator / runtime
registries). Two are built in: `knowledge` (default) and `raw_attempts`
(alias `raw`).

- **CLI:** `--improvement-strategy {knowledge,raw_attempts}`. Default
  `knowledge` rebuilds the engine's own default, so it is behavior-preserving;
  the strategy is authoritative over phase execution (e.g. `raw_attempts` skips
  forums/distillation regardless of the forum-round flags). `raw_attempts`
  additionally disables per-task seed-package enrichment regardless of
  `--no-memory`.
- **Programmatically:** `orch.set_improvement_strategy(...)`, or resolve by name
  with `get_strategy_spec(name).factory()`.

## Adding a variant

1. Subclass `DefaultKnowledgeStrategy` (to inherit seed scheduling and seeding)
   or implement the
   `ImprovementStrategy` protocol directly.
2. Override only the hooks your variant changes; return early to skip a phase,
   or override `plan_seed_next_generation(...)` to choose `seed`, `skip`, or
   `stop` and the next task set.
3. Read run context from `ctx.config` / `ctx.knowledge` and invoke phase work
   through `ctx.phases`.
4. Register it so the CLI and API can select it by name:

   ```python
   from ksi.orchestrator.strategy import StrategySpec, register_strategy

   register_strategy(StrategySpec(name="my_strategy", factory=MyStrategy))
   ```

   (Or wire it ad-hoc via `orch.set_improvement_strategy(MyStrategy())`.)
5. Add an equivalence/behaviour test alongside
   `tests/test_improvement_strategy.py`.

## What remains engine-owned

This refactor is deliberately phase-service-first:

- The execution, forum, distillation, and seeding bodies now live behind
  engine-backed phase services. The old private engine phase wrappers have been
  removed; direct callers should use the explicit services.
- `GenerationContext` no longer carries the whole engine. Strategies invoke
  only explicit phase capabilities instead of engine-private methods.
- The try/except policy, token flush, and persistence callbacks remain in the
  engine loop.

(The `--improvement-strategy` CLI flag now exists — see "Selecting a strategy"
above.)

## Distillation path: window-bundle (the single path)

Distillation uses a **window-bundle** approach for both per-task and
cross-task phases:

- **Per-task**: recomputes the advice-prose Insight bundle
  (transferable_insights / confirmed_constraints / rejected_hypotheses /
  pitfalls / checks / next_steps) from the full attempt/post history each
  generation. Stateless; re-processes full history every generation.
- **Cross-task**: recomputes bundles over the last
  `KSI_CROSS_TASK_DISTILL_GEN_WINDOW` (default 6) generations of cross-task
  posts. Context overflow cannot occur on the cross-task **forum prompt**: forum
  agents see only the **current generation's** posts (prior-generation history
  is not loaded), which is bounded by `(rounds-1) * num_agents`. Cross-generation
  knowledge still flows forward through distillation → seeding, not through raw
  forum history. The **distillation input** is bounded by the distiller's own
  budget selector (`distillation/cross_task.py::_select_cross_posts_for_budget`),
  which keeps every generation represented and, under target-conditioning, ranks
  posts by **lexical relevance to the target task** (`target_relevance` in
  `ksi/memory/cross_task_context.py`) rather than recency — so large low-solve
  benchmarks (e.g. Terminal-Bench 2 at 89 tasks) do NOT need
  `KSI_CROSS_TASK_DISTILL_GEN_WINDOW` lowered to stay under the 200K limit.
  **Memory horizon (trade-off):** because the forum no longer carries
  prior-generation history, the cross-generation knowledge path is
  forum → distill → seed → next-gen bundle, and bundles are **not** re-distilled
  into later generations. The distill step is windowed to the last
  `KSI_CROSS_TASK_DISTILL_GEN_WINDOW` (default 6) generations, so an insight
  older than the window decays unless an agent re-surfaces it in a recent forum
  post. Raise the window (or set it to `0` to disable windowing) for campaigns
  that need a longer effective memory, at the cost of a larger distill prompt.
  Under the default (`--cross-task-distill-target-conditioning`, default
  **true**), each downstream seed target gets its own per-task-conditioned
  cross-task bundle — keyed by that target `task_id`, distilled with the task's
  full prompt as the relevance signal, and injected only into that task's
  MEMORY.md. With default `--drop-solved`, targets are unsolved training tasks
  plus hold-out probes; with `--no-drop-solved`, retained solved tasks are
  included. Pass `--cross-task-distill-target-conditioning false` for the
  legacy ablation: a single broadcast bundle per generation (stored under
  `CROSS_TASK_SENTINEL`) injected into every task's MEMORY.md.

There is no alternative distillation strategy or channel format. The
environment variables `KSI_DISTILL_STRATEGY`, `KSI_PER_TASK_CHANNEL`, and
`KSI_CROSS_TASK_CHANNEL` have been **removed**; setting any non-default value
now raises a `RuntimeError` at orchestrator run startup (see
`src/ksi/distillation/_removed_env.py`), including `--no-memory` and `raw_attempts`
runs. The only selectable axis is `--improvement-strategy {knowledge,raw_attempts}`
(see "Selecting a strategy" above).

## Bundle render caps: external vs. internal (a comparison confound)

Seeded per-task knowledge items are rendered into MEMORY.md with a per-item
character cap, and the two channels use **different** caps
(`src/ksi/runtime/seeding.py`):

- **External KT bundles** — both channels: per-task
  (`--seed-per-task-bundles-path`) and cross-task (`--seed-bundle-path`),
  each stamped `_external_seed_source` at inject time — render at a generous
  **4000-char per-item** cap plus a **16000-char whole-section** budget
  (`_EXTERNAL_BUNDLE_ITEM_MAX_CHARS` / `_EXTERNAL_BUNDLE_TOTAL_MAX_CHARS`).
- **Internal per-task and cross-task items** stay byte-identical at the
  historical **480-char per-item** cap with no section budget.

The rationale is that external bundles carry the transfer *treatment itself* —
they are deliberately injected experiment payloads (a large item can be
silently cut at the old 480-char cap so the agent never sees the payload),
whereas internal items are authored short by the distiller.

**Caveat for KT experiments.** This asymmetry means any A/B that compares
**internal-channel KT vs. external-bundle KT carries an ~8x per-item
delivery-capacity confound** — a difference in how much text reaches the agent,
not (only) a channel effect. Such comparisons should either equalize the caps
across both channels or explicitly account for the capacity gap when
attributing any delta. Known limitation: the per-item evidence sub-fields
(`applies_when` / `does_not_apply_when` at 200 chars, evidence `quote` at 140
chars) stay capped even on the external path, so external items are not
uniformly uncapped.

### Resilience to transient host→provider failures

The distillation LLM calls run on the **host** (not inside the task
containers), so a host-side network/DNS blip to the provider API can fail a
whole generation's distillation even while container-side attempts keep
succeeding. Two guards bound that:

- **Retry** — each distill LLM call retries transient failures (connection
  errors, provider 5xx/rate-limit, DNS `EAI_AGAIN`) with jittered exponential
  backoff before giving up. Auth and deterministic failures are never retried.
  Retries default to **6** (deliberately more generous than `--max-task-retries`,
  since a zeroed distill generation wastes all of that generation's attempt
  compute); tune with `KSI_DISTILL_MAX_RETRIES` (0 disables retry).
- **Escalation** — when a generation is attempted but its distillation is
  fully zeroed by failures, the engine logs at `ERROR` and counts consecutive
  zeroed generations (a healthy generation resets the count). This is a
  sustained-outage signal retry cannot fix; by default the run continues.
  `--abort-on-distill-stall N` (0 = disabled, default) makes the run abort
  once the streak reaches `N`, instead of spending the rest of the campaign's
  attempt compute for no learning.

## See also

- [extending.md](./extending.md) — the index of every extension seam,
  including this one.
- [adding_a_benchmark.md](./adding_a_benchmark.md) and
  [adding_an_evaluator.md](./adding_an_evaluator.md) — the other three
  seams that follow the same register-a-spec pattern.
- [architecture.md §4 Generation Knowledge Loop](./architecture.md#4-generation-knowledge-loop) —
  the full per-generation control-flow diagram this seam's hooks plug into.

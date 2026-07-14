# Adding a Benchmark (Task Source)

Task sources (benchmarks) are described by a single `TaskSourceSpec` in the
central registry: [`src/kcsi/tasks/registry.py`](https://github.com/recursive-knowledge/KCSI/blob/main/src/kcsi/tasks/registry.py).

Before this registry, adding a benchmark meant shotgun-editing ~15 files because
`task_source == "<name>"` string-equality dispatch was scattered across loaders,
prompts, distillation, runtime, layout and CLI eval-selection. Now there is
**one** authoritative place that lists valid sources and their per-source
capabilities and values; every dispatch site consults the spec
(`spec.supports_mcp_arc`, `spec.default_evaluator`, `spec.prompt_kind`, ...)
rather than comparing name strings.

## The spec

`TaskSourceSpec` fields map 1:1 to the variation points that used to be
`if task_source == ...` branches:

| field | drives |
|-------|--------|
| `name` / `aliases` | canonical id + accepted synonyms (e.g. `arc1`, `arc2`, `arc_agi_*`, `swebench`) |
| `default_evaluator` | `cli._normalize_evaluator_for_task_source` + M14 evaluator/source warn-map |
| `prompt_kind` | string key into the hardcoded `build_execution_prompt` / `build_task_markdown` **fallback** chain (`src/kcsi/prompts/__init__.py`), used when no `execution_prompt_builder` / `task_markdown_builder` is attached |
| `execution_prompt_builder` | optional per-source builder; called as `execution_prompt_builder(task, *, has_memory=..., generation=...)` → prompt string. Consulted before the `prompt_kind` fallback chain |
| `task_markdown_builder` | optional per-source builder; called as `task_markdown_builder(task)` → `TASK.md` string. Consulted before the `prompt_kind` fallback chain (the `task_md_override` metadata hook still wins over both) |
| `distill_domain_hint` | optional per-source distillation domain hint (`src/kcsi/distillation/prompts.py::_domain_hint`): the hint **string**, or a zero-arg **callable** returning it. Opt-in — when unset, **no** domain-hint paragraph is injected (the generic hint is reserved for the unresolvable/cross-task case) |
| `loader` | task-loader callable; `load_tasks_for_source` calls `spec.loader(tasks_path, *, task_source=..., evals_path=..., arc_max_trials=...)` (built-in loaders are attached by `src/kcsi/tasks/loaders.py` at import time) |
| `supports_mcp_arc` | marks the source as ARC-native: the container materializes `payload.json` + attempt files and the agent uses native file tools (name retained for back-compat) |
| `is_offline` | sealed/offline benchmark; provider-native tools disabled |
| `uses_repo_snapshots` | needs SWE-bench-style repo cloning/snapshots |
| `supports_classification` | `--classify` / categories-json path |
| `needs_eval_records` | eval-records sidecar loading |
| `delegates_runtime` | uses a dedicated runtime executor (e.g. TB2) |
| `arc_task_reference` | engine/snapshots build a hidden ARC reference payload |
| `upstream_strict` | maintained upstream-strict published benchmark; enables the disclosure warning when `--no-drop-solved` can carry solved-task answers forward |
| `validate_tasks_path` | optional callable validating `--tasks-path`; called as `validate_tasks_path(tasks_path, *, evals_path=...)` and returns the `parser.error` message string or `None` when acceptable. A source whose hook is `None` is rejected as unsupported. Built-in validators are attached by `src/kcsi/tasks/path_validation.py` at import time |

### A real example: how `arc` is registered

The shortest of the four built-in registrations is `arc`
(`src/kcsi/tasks/registry.py`):

```python
register_task_source(
    TaskSourceSpec(
        name="arc",
        aliases=("arc1", "arc2", "arc_agi", "arc_agi_1", "arc_agi_2"),
        default_evaluator="arc_session",
        prompt_kind="arc",
        distill_domain_hint=_DISTILL_HINT_ARC,
        supports_mcp_arc=True,
        is_offline=True,
        arc_task_reference=True,
        upstream_strict=True,
    )
)
```

Reading it against the field table above: `arc` accepts five aliases so
`--task-source arc1` and `--task-source arc_agi_2` both resolve to the same
spec; it defaults to the `arc_session` evaluator so `--evaluator` can be
omitted for ARC runs; `prompt_kind="arc"` selects ARC's hardcoded prompt
branch (rather than attaching an `execution_prompt_builder` callable — ARC
predates that hook, which is why it still uses the fallback-chain path), while
`distill_domain_hint=_DISTILL_HINT_ARC` attaches ARC's distillation hint
directly on the spec;
`supports_mcp_arc=True` marks the source as ARC-native (the container
materializes `payload.json` + attempt files; name retained for back-compat);
`is_offline=True` disables provider-native web tools regardless of
`KCSI_ALLOW_WEB_TOOLS` (see [web_tools_policy.md](https://github.com/recursive-knowledge/KCSI/blob/main/benchmarks/docs/web_tools_policy.md));
`arc_task_reference=True` tells the engine to build ARC's hidden reference
payload during enrichment; `upstream_strict=True` marks ARC as a maintained
published benchmark where retaining solved tasks needs disclosure. Every field
it *doesn't* set (`loader`,
`validate_tasks_path`, `uses_repo_snapshots`, `supports_classification`,
`needs_eval_records`, `delegates_runtime`, ...) keeps its conservative
default — `loader` and `validate_tasks_path` are populated separately by
`src/kcsi/tasks/loaders.py` and `src/kcsi/tasks/path_validation.py` at import
time rather than inline in the registration call, which is why a real source
can look shorter than the full field table suggests.

Compare this against `swebench_pro`'s registration in the same file — it sets
`uses_repo_snapshots=True`, `supports_classification=True`, and
`needs_eval_records=True` instead, showing a source with a completely
different capability profile than ARC's.

## Steps

1. **Register the spec.** Add a `register_task_source(TaskSourceSpec(...))` call
   in `src/kcsi/tasks/registry.py` (or at runtime via `register_task_source` for a
   plugin). Set only the flags your benchmark needs; defaults are the
   conservative generic behavior.

   ```python
   from kcsi.tasks.registry import TaskSourceSpec, register_task_source

   def _load_my_bench(tasks_path, **kwargs):  # accept **kwargs for forward compat
       ...

   def _validate_my_bench(tasks_path, *, evals_path):
       # Return the exact ``parser.error`` message, or None when the path is OK.
       if not tasks_path.exists():
           return f"--tasks-path for --task-source my_bench must exist: {tasks_path}"
       return None

   register_task_source(
       TaskSourceSpec(
           name="my_bench",
           aliases=("mybench",),
           default_evaluator="my_bench",
           prompt_kind="my_bench",
           distill_domain_hint="DOMAIN HINT (my_bench): primitives to anchor on ...",
           loader=_load_my_bench,
           validate_tasks_path=_validate_my_bench,
       )
   )
   ```

   Registering automatically adds `my_bench` to `SUPPORTED_TASK_SOURCES` (the
   CLI `--task-source` choices) and makes `get_spec("my_bench")` resolve. An
   unknown source now fails early at CLI validation with a message listing the
   valid sources.

2. **Supply prompts and the domain hint via the spec (preferred) — or fall
   back to the hardcoded chains:**
   - Task loading needs no dispatch edit: `load_tasks_for_source` calls
     `spec.loader` directly. It is invoked as
     `loader(tasks_path, *, task_source=..., evals_path=..., arc_max_trials=...)`
     and must return a list of `TaskSpec`; a registered source without a
     loader fails with a clear error.
   - Prompts (no dispatch edit, preferred): attach `execution_prompt_builder`
     and `task_markdown_builder` to the spec. `build_execution_prompt` /
     `build_task_markdown` consult these first and only fall back to the
     hardcoded `prompt_kind == ...` chain when they are `None`. Example:

     ```python
     def _my_exec_prompt(task, *, has_memory, generation):
         return f"You are solving a my_bench task. This is generation {generation}.\n..."

     def _my_task_md(task):
         return "# TASK\n\n- task_source: my_bench\n..."

     register_task_source(
         TaskSourceSpec(
             name="my_bench",
             loader=_load_my_bench,
             validate_tasks_path=_validate_my_bench,
             execution_prompt_builder=_my_exec_prompt,
             task_markdown_builder=_my_task_md,
             distill_domain_hint="DOMAIN HINT (my_bench): primitives to anchor on ...",
         )
     )
     ```

     (Built-in sources still use the hardcoded `prompt_kind` chains in
     `src/kcsi/prompts/__init__.py`; you may add a `prompt_kind == "my_bench"`
     branch there instead of the callables if you prefer, but the spec-attached
     callables keep the addition to the single registry entry.)
   - Distillation domain hint (optional, no dispatch edit): set
     `distill_domain_hint` (a string or a zero-arg callable) on the spec.
     It is opt-in — leave it unset and no domain-hint paragraph is injected
     for the source (the generic `_GENERIC_DOMAIN_HINT` is reserved for the
     unresolvable/cross-task case, where there is no single benchmark to key
     on). The four built-in sources set it on their own specs in
     `src/kcsi/tasks/registry.py`.
   - Evaluator: register the evaluator with `register_evaluator` (see [adding_an_evaluator.md](./adding_an_evaluator.md)) if new.
   - CLI `--tasks-path` validation needs no dispatch edit: set
     `validate_tasks_path` on the spec (as above). A source without it is
     rejected as unsupported.

3. **Add an evaluator** under `src/kcsi/eval/` if the benchmark needs one, and
   register its name via `register_evaluator` (see [adding_an_evaluator.md](./adding_an_evaluator.md)).

4. **Tests:** extend `tests/test_task_registry.py` to pin your new source's
   capability flags, and add loader/prompt tests as appropriate.

The single registry entry keeps the valid-source list and the capability table
in one place; the per-benchmark code lives behind capability checks rather than
duplicated name-string comparisons.

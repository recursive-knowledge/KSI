# Extending ksi

ksi is built around small, typed **seams** so researchers and developers can
implement variants without editing core engine code. Every seam is the same
shape: a `Protocol` that defines the contract + a **registry** you register into.
No `if name == ...` dispatch edits anywhere.

## The seams

| I want to add… | Protocol | Register with | Guide |
|----------------|----------|---------------|-------|
| A **benchmark / task source** | — (`TaskSourceSpec`) | `register_task_source` (`src/ksi/tasks/registry.py`) | [adding_a_benchmark.md](./adding_a_benchmark.md) |
| An **evaluator** (how attempts are scored) | `Evaluator` (`src/ksi/protocols.py`) | `register_evaluator` (`src/ksi/eval/registry.py`) | [adding_an_evaluator.md](./adding_an_evaluator.md) |
| A **runtime** (how agents execute) | `RuntimeExecutor` (`src/ksi/protocols.py`) | `register_runtime` (`src/ksi/runtime/registry.py`) | [adding_an_evaluator.md](./adding_an_evaluator.md#adding-a-runtime) |
| An **improvement strategy** (forum/distill/seed mechanism) | `ImprovementStrategy` (`src/ksi/orchestrator/strategy.py`) | `register_strategy` (`src/ksi/orchestrator/strategy.py`) | [improvement_strategies.md](./improvement_strategies.md) |
| A **lifecycle observer** (transcripts, tokens, callbacks) | `PersistenceObserver` (`src/ksi/protocols.py`) | passed to the run (`CompositePersistence` fans out) | [architecture.md](./architecture.md) |

## The pattern

Each registry exposes the same four functions (names vary by seam):

```python
register_<seam>(spec, *, replace=False)   # add a spec (+ aliases); raises on dup unless replace=True
resolve_<seam>(name)                      # -> spec | None  (case-insensitive, whitespace-stripped)
get_<seam>_spec(name)                     # -> spec; raises a helpful error listing valid names
supported_<seam>s(*, include_aliases=False)  # -> tuple of names
```

A `*Spec` is a frozen dataclass carrying the seam's name, aliases, and a
`factory` (or capability flags, for task sources). Registration happens at import
time in the package `__init__`, so importing the package populates the registry —
and a plugin can register its own spec the same way.

Example (evaluator):

```python
from ksi.eval.registry import EvaluatorSpec, register_evaluator

def _build_my_eval(args):
    return MyEvaluator(threshold=getattr(args, "my_threshold", 0.5))

register_evaluator(EvaluatorSpec(name="my_eval", factory=_build_my_eval, description="..."))
```

The CLI `--evaluator` / `--runtime` / `--improvement-strategy` choices are all
derived from these registries, so a registered seam is immediately selectable.

The registration functions are re-exported at the top level for convenience:
`ksi.register_evaluator`, `ksi.register_runtime`, `ksi.register_task_source`,
and `ksi.register_strategy`, each alongside its spec dataclass
(`ksi.EvaluatorSpec`, `ksi.RuntimeSpec`, `ksi.StrategySpec`,
`ksi.TaskSourceSpec`).

Register through `register_<seam>` — never by mutating a registry's `REGISTRY`
dict directly. Direct mutation bypasses the duplicate-name detection, so the
`REGISTRY` dicts are intentionally **not** part of the public surface (not in any
registry module's `__all__`). Read the registered set with `resolve_<seam>` /
`supported_<seam>s`.

Every exception ksi raises subclasses `ksi.KsiError`, so a programmatic
caller can `except ksi.KsiError` to catch any ksi-originated failure (the
concrete types keep their historical `RuntimeError` / `ValueError` base too).

### Constructing a registered component programmatically

`EvaluatorSpec.factory` is called as `factory(args)` and `RuntimeSpec.factory`
as `factory(args, provider_env)`, where `args` is a config object exposing the
attributes the factory reads (the CLI passes its `argparse.Namespace`). To build
a registered component **without** a namespace, use `ksi.build_evaluator` /
`ksi.build_runtime`, which start from the full set of CLI defaults and apply
keyword overrides:

```python
import ksi

evaluator = ksi.build_evaluator(
    "swebench_pro",
    swebench_pro_raw_sample_path="/data/samples.jsonl",
    swebench_timeout_sec=1800,
)
runtime = ksi.build_runtime("container", knowledge_db_path="/tmp/run_knowledge.sqlite")
```

Override keys are the CLI argument *dest* names (underscored; run `ksi --help`
for the full set). They are matched by attribute name only — a mistyped or
unknown key raises `TypeError` rather than being silently ignored, so a typo
fails loudly instead of falling back to the default. `build_runtime` returns
the base runtime; the Terminal-Bench-2 delegation wrapper is a CLI-only concern.

> Strategy components need no construction helper — `StrategySpec.factory` takes
> no arguments, so `ksi.register_strategy`'d strategies are already buildable
> via `get_strategy_spec(name).factory()`.

## Driving ksi programmatically

Once your seam is registered (or constructed directly), drive a run from Python
with `ksi.run(...)` — see [programmatic_api.md](./programmatic_api.md).

## Contributing

Setup, test/lint commands, and PR conventions are in
[CONTRIBUTING.md](./CONTRIBUTING.md).

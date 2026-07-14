# Adding an Evaluator (and Runtime)

Evaluators and runtimes are registry-backed, mirroring the task-source registry
([docs/adding_a_benchmark.md](./adding_a_benchmark.md)).

## Adding an evaluator

An evaluator implements the `Evaluator` protocol (`src/ksi/protocols.py`):

```python
class Evaluator(Protocol):
    def evaluate(self, *, task: TaskSpec, model_output: str, **kwargs: Any) -> EvalResult: ...
```

`EvalResult` is a `TypedDict` (`src/ksi/models.py`, exported from `ksi`), so
returning a plain `dict` remains valid; the type just documents the expected
keys. Register it once — no CLI edits required:

```python
from ksi.eval.registry import EvaluatorSpec, register_evaluator

def _build_my_eval(args):
    return MyEvaluator(threshold=getattr(args, "my_threshold", 0.5))

register_evaluator(EvaluatorSpec(name="my_eval", factory=_build_my_eval, description="..."))
```

Built-ins register in `src/ksi/eval/registry.py` (imported via
`src/ksi/eval/__init__.py`). The `--evaluator` CLI choices and the
`SUPPORTED_EVALUATORS` tuple are derived from the registry automatically. The
factory receives the CLI argparse `Namespace`; read whatever flags you need with
`getattr(args, ...)`.

If your evaluator's `EvalResult` captures any hidden test-runner output (a
`*_stdout_tail`/`*_stderr_tail` key in `eval_results`) or hidden verifier field
(in `attempt_meta`), you must register it for redaction in
[`src/ksi/memory/parity.py`](https://github.com/recursive-knowledge/KSI/blob/main/src/ksi/memory/parity.py)
(`HIDDEN_TEST_RUNNER_TAIL_KEYS` / `HIDDEN_ATTEMPT_META_KEYS`) — otherwise it
leaks to agent-facing surfaces (MEMORY.md, forum tools, distillation prompts).

### Scoring convention: `None` means unscored, not a failure

A task source's `TaskSourceSpec.score_from_eval` hook (see
`src/ksi/orchestrator/scoring.py`) converts an evaluator's raw `EvalResult`
into the float fed into `_best_scores`, distillation, and forum. Returning
`None` from that hook means "no trustworthy verdict" — e.g. a Docker/harness
timeout, no tool trace captured at all, or broken reference data for the
task — and is excluded from `_best_scores` updates, distillation, and forum
context, so an infra crash never contaminates the multi-generation learning
signal as if the agent had genuinely failed. A real `0.0` is reserved for
"the agent produced something and it was wrong." `terminal_bench_2`,
`swebench_pro`, `polyglot`, and `arc` all follow this convention — gate your dedicated scorer on the evaluator's true infra-failure
statuses *before* falling through to the generic `score_from_eval_results`
fallback, which does not itself distinguish the two cases.

## Adding a runtime

A runtime implements the `RuntimeExecutor` protocol (`src/ksi/protocols.py`) and
registers via `src/ksi/runtime/registry.py`:

```python
from ksi.runtime.registry import RuntimeSpec, register_runtime

register_runtime(RuntimeSpec(name="my_runtime", factory=_build_my_runtime))
```

The factory is called as `factory(args, provider_env)`. Note: per-task-source
runtime *delegation* (e.g. Terminal-Bench 2) is handled separately in
`cli._choose_runtime` via `TaskSourceSpec.delegates_runtime`, not by the runtime
registry.

**The Python registration above is only half the seam.** The only maintained
runtime today, `container`, is a Python-side `RuntimeExecutor`
(`KsiContainerExecutor`) that shells out to a TypeScript host launcher and a
Docker-executed agent runner — see
[architecture.md's System Map](./architecture.md#1-system-map) for the full
chain (`ksi.runtime.container_host` → `runtime_runner/src/main.ts` →
`runtime_runner/src/container_runner.ts` → the `ksi-agent:bench` Docker image
→ `runtime_runner/agent-runner/src/index.ts`). A genuinely new execution
backend needs the equivalent host-launcher and in-container wiring on the
TypeScript/Docker side, not just the Python `RuntimeSpec` registration above —
registering the Python spec alone gets you a selectable `--runtime` name with
nothing behind it. This is the same class of gap this repo's own gotchas file
documents for feature flags: a new host↔container feature needs wiring at
*all three* of the Python host, `runtime_runner/src/main.ts`'s payload
translation, and the in-container consumer — skipping the middle one compiles
and unit-tests clean on both ends while being silently unreachable in
production.

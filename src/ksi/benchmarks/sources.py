"""Built-in benchmark ``TaskSourceSpec`` registrations.

Registers the four reference benchmarks (swebench_pro, arc, polyglot,
terminal_bench_2) into the central ``ksi.tasks.registry`` REGISTRY. Imported
by ``ksi.benchmarks.register_all()`` — see that function's docstring for the
core->benchmarks wiring contract and registration-order guarantee.
"""

from __future__ import annotations

from ..tasks.registry import TaskSourceSpec, register_task_source

# Distillation domain hints for the built-in sources. These steer the distiller
# toward benchmark-specific concrete primitives instead of generic process
# meta-advice; see ``src/ksi/distillation/prompts.py::_domain_hint`` for how
# they are consumed. The hint is opt-in: a source that leaves
# ``distill_domain_hint`` unset injects no domain-hint paragraph, and
# ``_GENERIC_DOMAIN_HINT`` is used only for the unresolvable/cross-task case.
_DISTILL_HINT_SWEBENCH = (
    "DOMAIN HINT (SWE-bench Pro): primitives to anchor on include "
    "specific API calls, function/class/method names, import paths, "
    "file paths, decorator usage, signature changes, test fixtures, "
    "error types and their stack-frame origins, and concrete "
    'code-refactor patterns (e.g. "wrap X in try/except Y", '
    '"rename kwarg A to B in call sites matching pattern C"). '
    "Bias toward these."
)
_DISTILL_HINT_ARC = (
    "DOMAIN HINT (ARC-AGI): primitives to anchor on include concrete "
    "grid operations (flood-fill, connected components, reflection, "
    "rotation, tiling, scaling, color replacement, symmetry axes), "
    "explicit color indices, shape signatures (e.g. 3x3 cross, L-piece, "
    "hollow rectangle), and transformation rules written as pixel "
    'mappings (e.g. "isolated 0s -> 3x3 cross of 7s", "point '
    'reflection (2mr-r, 2mc-c)"). Bias toward these.'
)
_DISTILL_HINT_POLYGLOT = (
    "DOMAIN HINT (polyglot / Exercism): primitives to anchor on "
    "include specific language features, stdlib modules or idioms, "
    "test-runner invocations and flags (pytest, cargo test, go test, "
    "vitest, junit, catch2), build-tool commands, and common "
    "language-specific pitfalls (e.g. Rust lifetimes, Go nil "
    "interfaces, C++ header includes). Bias toward these."
)
_DISTILL_HINT_TB2 = (
    "DOMAIN HINT (Terminal-Bench 2): primitives to anchor on include "
    "service/process lifecycle commands, package-manager installs, build "
    "commands, file paths under /etc or /app, ports, SSH/nginx/systemd "
    "configuration, artifact existence checks, and the attempt's own "
    "outcome/reward signal. Prefer bullets that name the concrete command, path, service, "
    "artifact, or runtime dependency that should be checked or changed. "
    "For TB2, strong next_steps usually include an exact shell command or "
    "path plus the first verifier-aligned check to run immediately after. "
    "If a prior attempt only worked from one shell, cwd, or ad hoc env var, "
    "say how to make that behavior persist for a fresh shell and the verifier."
)


# Canonical specs. Registration order defines the canonical-name ordering used
# by ``supported_task_sources`` and mirrors the legacy
# ``SUPPORTED_TASK_SOURCES`` tuple. The ``custom`` source registers separately,
# AFTER these four (see ``ksi.tasks.loaders`` / ``ksi.tasks.custom.register_custom_source``),
# so that the canonical order stays swebench_pro, arc, polyglot,
# terminal_bench_2, custom.
register_task_source(
    TaskSourceSpec(
        name="swebench_pro",
        aliases=("swebench",),
        default_evaluator="swebench_pro",
        prompt_kind="swebench_pro",
        distill_domain_hint=_DISTILL_HINT_SWEBENCH,
        uses_repo_snapshots=True,
        supports_classification=True,
        needs_eval_records=True,
        upstream_strict=True,
    )
)
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
register_task_source(
    TaskSourceSpec(
        name="polyglot",
        default_evaluator="polyglot_harness",
        prompt_kind="polyglot",
        distill_domain_hint=_DISTILL_HINT_POLYGLOT,
        upstream_strict=True,
    )
)
register_task_source(
    TaskSourceSpec(
        name="terminal_bench_2",
        default_evaluator="terminal_bench_2",
        prompt_kind="terminal_bench_2",
        distill_domain_hint=_DISTILL_HINT_TB2,
        delegates_runtime=True,
        upstream_strict=True,
    )
)

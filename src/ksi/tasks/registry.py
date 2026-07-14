"""Central registry of supported task sources (benchmarks).

Historically, adding a benchmark required shotgun edits across ~15 files because
``task_source == "<name>"`` string-equality dispatch was scattered through
loaders, prompts, distillation, runtime, layout and CLI eval-selection code.
Failures surfaced late, as raises deep in the call stack.

This module makes ONE place authoritative for:

* the canonical set of valid task sources and their aliases,
* the per-source *capability flags* and *values* that the rest of the codebase
  branches on (default evaluator, prompt builder kind, MCP/offline behavior,
  repo-snapshot needs, classification support, runtime delegation),
* a single ``get_spec`` entry point that raises an early, helpful error for an
  unknown source instead of a late ``ValueError`` somewhere downstream.

Call sites consult *capabilities and values* (``spec.supports_mcp_arc``,
``spec.default_evaluator``, ...), not name strings. This is a PURE refactor:
the flags below were derived from the pre-existing dispatch sites and must
reproduce their behavior exactly.

Adding a new benchmark
----------------------
Instead of editing every dispatch site, register one spec::

    from ksi.tasks.registry import TaskSourceSpec, register_task_source

    def _load_my_bench(tasks_path, **kwargs):
        ...

    def _validate_my_bench(tasks_path, *, evals_path=None):
        ...  # return an error string if tasks_path is unusable, else None

    register_task_source(
        TaskSourceSpec(
            name="my_bench",
            aliases=("mybench",),
            default_evaluator="my_bench",
            prompt_kind="my_bench",     # build_execution_prompt / build_task_markdown switch on this
            distill_domain_hint="DOMAIN HINT (my_bench): primitives to anchor on ...",
            loader=_load_my_bench,
            # REQUIRED: without validate_tasks_path the CLI rejects the source
            # at --tasks-path validation ("registered but has no validate_tasks_path").
            validate_tasks_path=_validate_my_bench,
        )
    )

Then implement the matching branches keyed on the spec fields (loaders consult
``spec.loader`` — ``load_tasks_for_source`` calls it directly, no dispatch edit
needed; prompts switch on ``spec.prompt_kind``; the CLI maps
``spec.default_evaluator`` / validation kind). The single registry entry keeps
the valid-source list and capability table in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# ``REGISTRY`` is intentionally NOT part of the public surface:
# direct mutation (``REGISTRY[name] = spec``) bypasses ``register_task_source``'s
# duplicate-name detection. Register via ``register_task_source`` and read via
# ``resolve_source`` / ``supported_task_sources``.
__all__ = [
    "TaskSourceSpec",
    "get_spec",
    "resolve_source",
    "register_task_source",
    "supported_task_sources",
]


@dataclass(frozen=True)
class TaskSourceSpec:
    """Per-source variation points that describe how a benchmark plugs into KSI.

    Fields fall into three groups:

    * ``name`` / ``aliases`` — the canonical id and accepted synonyms a caller
      may pass as ``--task-source``.
    * ``default_evaluator`` / ``prompt_kind`` — string keys that select this
      source's default evaluator and its fallback prompt behavior when no
      callable override is attached.
    * Optional callables (``execution_prompt_builder``, ``task_markdown_builder``,
      ``distill_domain_hint``, ``loader``, ``approach_diagnosis``,
      ``score_from_eval``, ``trace_condensed``, ``attempt_meta_builder``,
      ``validate_tasks_path``) — per-source hooks a plugin attaches to
      customize behavior without editing shared dispatch code. Each is
      documented at its attribute definition below.

    Boolean/flag fields (``supports_mcp_arc``, ``is_offline``,
    ``uses_repo_snapshots``, ``supports_classification``,
    ``needs_eval_records``, ``delegates_runtime``, ``arc_task_reference``)
    declare capabilities the source has or needs; the default for every flag
    is the conservative "off" behavior, so a new source only sets the flags
    it actually requires.

    See [adding_a_benchmark.md](https://recursive-knowledge.github.io/KSI/adding_a_benchmark/)
    for a worked example of registering a new source, and the attribute-level
    docstrings below for what each optional callable receives and returns.
    """

    name: str
    aliases: tuple[str, ...] = ()
    # Used by cli._normalize_evaluator_for_task_source and the M14 evaluator/source warn-map.
    default_evaluator: str = "none"
    # Fallback key into build_execution_prompt / build_task_markdown (src/ksi/prompts/__init__.py)
    # when no execution_prompt_builder / task_markdown_builder is attached.
    prompt_kind: str = "generic"
    # Consulted before the prompt_kind fallback chain in build_execution_prompt.
    execution_prompt_builder: Optional[Callable[..., str]] = field(default=None, compare=False)
    # Consulted before the prompt_kind fallback chain in build_task_markdown; task_md_override
    # metadata still wins over both.
    task_markdown_builder: Optional[Callable[..., str]] = field(default=None, compare=False)
    # The source's distillation domain hint (src/ksi/distillation/prompts.py::_domain_hint):
    # the hint string, or a zero-arg callable returning it. Opt-in — when unset, no
    # domain-hint paragraph is injected for the source (the generic hint is reserved for
    # the unresolvable/cross-task case).
    distill_domain_hint: "Optional[str | Callable[[], str]]" = field(default=None, compare=False)
    # load_tasks_for_source calls this directly; built-in loaders are attached by
    # src/ksi/tasks/loaders.py at import time.
    loader: Optional[Callable[..., object]] = field(default=None, compare=False)
    # ARC-source marker. container_host.py keys the native ARC path
    # (attempt-file synth) on this: arc_no_mcp_active = supports_mcp_arc.
    # (Historical name — the legacy ARC MCP toolset it once gated is removed.)
    supports_mcp_arc: bool = False
    # container_host._native_tools ARC branch.
    is_offline: bool = False
    # cli prepare_swebench_repo_snapshots / repo_cache.
    uses_repo_snapshots: bool = False
    # cli: swebench_pro only (--classify / categories-json path).
    supports_classification: bool = False
    # loaders.load_eval_records_for_source: swebench_pro only.
    needs_eval_records: bool = False
    # cli/runtime: terminal_bench_2 only.
    delegates_runtime: bool = False
    # EngineEnrichmentPhaseService.enrich builds a hidden ARC reference payload when set.
    arc_task_reference: bool = False
    # cli._validate_and_normalize_args: True for maintained upstream-strict
    # published benchmarks that give no correctness feedback, so --no-drop-solved
    # (retaining solved tasks) risks per-task answer carry-forward. Drives the
    # disclosure warning; derive the set via upstream_strict_task_sources().
    upstream_strict: bool = False
    # engine._build_approach_diagnosis; populated for built-ins by
    # ksi.orchestrator.approach_diagnosis at import time.
    approach_diagnosis: Optional[Callable[..., list[str]]] = field(default=None, compare=False)
    # engine._score_from_eval; populated for swebench by ksi.orchestrator.scoring at import time.
    score_from_eval: Optional[Callable[..., "float | None"]] = field(default=None, compare=False)
    # engine.GenerationalOrchestrator._knowledge_trace_condensed; populated for terminal_bench_2
    # by ksi.orchestrator.attempt_events at import time.
    trace_condensed: Optional[Callable[..., str]] = field(default=None, compare=False)
    # cli rejects the source as unsupported when this hook is None. Populated for built-ins by
    # ksi.tasks.path_validation at import time.
    validate_tasks_path: Optional[Callable[..., "str | None"]] = field(default=None, compare=False)
    # Merged via _merge_optional_meta; populated for terminal_bench_2 by
    # ksi.orchestrator.attempt_events at import time, mirroring trace_condensed.
    attempt_meta_builder: Optional[Callable[..., "dict[str, Any] | None"]] = field(default=None, compare=False)

    def all_names(self) -> tuple[str, ...]:
        """Canonical name plus every alias (deduped, canonical first)."""
        seen: list[str] = [self.name]
        for alias in self.aliases:
            if alias not in seen:
                seen.append(alias)
        return tuple(seen)


# ── Registry population ──────────────────────────────────────────────────────
#
# These specs encode the EXISTING behavior of the scattered ``task_source ==``
# branches. Do not change the flags here without changing the corresponding
# dispatch site in lockstep — they are a behavioral contract.

REGISTRY: dict[str, TaskSourceSpec] = {}


def register_task_source(spec: TaskSourceSpec, *, replace: bool = False) -> TaskSourceSpec:
    """Register ``spec`` (and its aliases) into the global ``REGISTRY``.

    Raises ``ValueError`` on a duplicate name/alias unless ``replace=True``.
    Returns the spec for convenience.
    """
    if not replace:
        for key in spec.all_names():
            existing = REGISTRY.get(key)
            if existing is not None:
                raise ValueError(
                    f"task source name/alias {key!r} already registered to {existing.name!r}; "
                    f"pass replace=True to override"
                )
    for key in spec.all_names():
        REGISTRY[key] = spec
    return spec


def _normalize(task_source: object) -> str:
    return str(task_source or "").strip().lower()


def resolve_source(task_source: object) -> Optional[TaskSourceSpec]:
    """Return the spec for ``task_source`` (canonical or alias), or ``None``."""
    return REGISTRY.get(_normalize(task_source))


def get_spec(task_source: object) -> TaskSourceSpec:
    """Resolve ``task_source`` to its spec, raising a clear early error.

    Accepts the canonical name or any registered alias (case-insensitive).
    """
    spec = resolve_source(task_source)
    if spec is None:
        valid = supported_task_sources(include_aliases=True)
        raise ValueError(f"unsupported task_source={task_source!r}; valid sources (incl. aliases): {', '.join(valid)}")
    return spec


def supported_task_sources(*, include_aliases: bool = False) -> tuple[str, ...]:
    """Canonical task-source names (registration order).

    With ``include_aliases=True``, appends aliases after the canonical names.
    """
    canonical: list[str] = []
    for spec in REGISTRY.values():
        if spec.name not in canonical:
            canonical.append(spec.name)
    if not include_aliases:
        return tuple(canonical)
    out = list(canonical)
    for spec in REGISTRY.values():
        for alias in spec.aliases:
            if alias not in out:
                out.append(alias)
    return tuple(out)


def upstream_strict_task_sources() -> tuple[str, ...]:
    """Canonical names of maintained upstream-strict published benchmarks.

    These give no correctness feedback, so retaining solved tasks
    (``--no-drop-solved``) risks per-task answer carry-forward. Derived from
    the registry so a newly-registered published source is covered without a
    parallel hardcoded list.
    """
    return tuple(spec.name for spec in REGISTRY.values() if spec.upstream_strict)


# No specs are registered here at import time. This module stays import-light
# (a pure dataclass + REGISTRY dict + lookup functions) so it can be imported
# from anywhere without pulling in loader/benchmark deps. The four built-in
# benchmark specs live in ``ksi.benchmarks.sources`` (registered via
# ``ksi.benchmarks.register_all()``), and the ``custom`` spec is registered by
# ``ksi.tasks.custom.register_custom_source()``. Both are wired from
# ``ksi.tasks.loaders`` at import time, in that order, so
# ``supported_task_sources()`` stays ``(swebench_pro, arc, polyglot,
# terminal_bench_2, custom)``.

"""Improvement-strategy seam for the generational orchestrator.

This module factors the *self-improvement mechanism* (per-task forum ->
cross-task forum -> distillation -> seeding) out of
:class:`kcsi.orchestrator.engine.GenerationalOrchestrator` as a thin,
swappable seam.  It is a **pure refactor / extraction**: the
:class:`DefaultKnowledgeStrategy` delegates each hook to explicit phase
services, so behaviour (and the flags that gate each phase) is unchanged while
strategies no longer depend on private orchestrator methods.

The seam exists so that the project's research variants (raw-attempts,
no-forum, generic-preamble, prompt-evolution) can eventually be expressed as
distinct ``ImprovementStrategy`` implementations instead of a flag maze inside
the engine. The execution, forum, distillation, and seeding phases now have
engine-backed service boundaries, and strategies use those explicit
capabilities instead of private orchestrator methods.

What remains engine-owned (intentionally — see
``docs/improvement_strategies.md``):

* The try/except + ``AuthenticationFailure`` re-raise policy around forum and
  distill phases stays in the engine ``run()`` loop, so error handling is
  identical to before.
* Per-generation token flush and ``persistence`` callbacks stay in the engine
  loop — they are not improvement-mechanism concerns.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, cast, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.knowledge_store import KnowledgeStore
    from ..models import GenerationConfig, TaskSpec, TaskTrace

SeedScheduleAction = Literal["seed", "skip", "stop"]
NextRemainingTasks = Callable[[list["TaskSpec"]], list["TaskSpec"]]


def _identity_next_remaining_tasks(remaining_tasks: list["TaskSpec"]) -> list["TaskSpec"]:
    return list(remaining_tasks)


@runtime_checkable
class ImprovementPhaseServices(Protocol):
    """Capabilities strategies can invoke for the improvement phases.

    The default engine-backed implementation lives in
    :mod:`kcsi.orchestrator.phase_services`.  Keeping this protocol here lets
    strategies depend on explicit capabilities instead of the full
    orchestrator object.
    """

    def per_task_forum(
        self,
        generation: int,
        traces: "list[TaskTrace]",
        *,
        next_task_pool_size: int | None = None,
    ) -> None: ...

    def cross_task_forum(self, *, generation: int, traces: "list[TaskTrace]") -> None: ...

    def distill(self, *, generation: int, task_ids: list[str]) -> None: ...

    def seed_next_generation(self, generation: int, *, next_task_pool_size: int | None = None) -> None: ...


@dataclass
class GenerationContext:
    """Per-generation state handed to each improvement-strategy hook.

    This is a thin capability view.  It exposes exactly the values the current
    phase calls consume:

    * ``phases`` — explicit phase capabilities for the current orchestrator.
    * ``generation`` — the 1-based generation index.
    * ``fresh_traces`` — traces produced this generation, excluding
      carried-forward (already-solved) traces.  This is what the forum and
      distill phases iterate.
    * ``next_task_pool_size`` — count of tasks that will carry into the next
      generation (drives forum workstream sizing and the seed phase).
    * ``next_remaining_tasks`` — strategy-visible planner for applying the
      engine's configured task-retention policy without reaching into engine
      internals.
    * ``config`` / ``knowledge`` — read-only handles exposed for custom
      strategies that need to inspect the active run context.
    """

    generation: int
    fresh_traces: "list[TaskTrace]"
    phases: ImprovementPhaseServices
    config: "GenerationConfig"
    next_task_pool_size: int | None = None
    knowledge: "KnowledgeStore | None" = None
    next_remaining_tasks: NextRemainingTasks = _identity_next_remaining_tasks

    def distill_task_ids(self) -> list[str]:
        """Task ids that had a fresh attempt this generation.

        Mirrors the inline computation in ``run()`` that builds the
        distillation phase input: sorted, de-duplicated, non-empty stripped
        ``task_id`` strings from ``fresh_traces``.
        """
        return sorted({str(t.task_id).strip() for t in self.fresh_traces if str(t.task_id).strip()})


@dataclass(frozen=True)
class SeedSchedulePlan:
    """Strategy-owned decision for the next-generation seed boundary."""

    action: SeedScheduleAction
    next_tasks: tuple["TaskSpec", ...] = ()
    reason: str = ""


@runtime_checkable
class ImprovementStrategy(Protocol):
    """The self-improvement mechanism, as a set of hooks on phase boundaries.

    Each hook mirrors one boundary in the engine's generation loop.  Hooks are
    invoked in order, once per generation, after task attempts complete:

    1. :meth:`per_task_forum` — Phase 2 (per-task discussion).
    2. :meth:`cross_task_forum` — Phase 3 (shared cross-task discussion).
    3. :meth:`distill` — Phase 4 (per-task + cross-task distillation).
    4. :meth:`plan_seed_next_generation` — decide seed / skip / stop and the
       next task set.
    5. :meth:`seed_next_generation` — Phase 5 (prepare the next population).

    The engine retains responsibility for the surrounding try/except policy,
    token flushing and persistence callbacks; strategies only own the
    improvement-mechanism phases themselves.
    """

    def per_task_forum(self, ctx: GenerationContext) -> None: ...

    def cross_task_forum(self, ctx: GenerationContext) -> None: ...

    def distill(self, ctx: GenerationContext) -> None: ...

    def plan_seed_next_generation(
        self,
        ctx: GenerationContext,
        *,
        remaining_tasks: list["TaskSpec"],
    ) -> SeedSchedulePlan: ...

    def seed_next_generation(self, ctx: GenerationContext) -> None: ...

    def should_enrich(self) -> bool: ...


class _SeedPlanner(Protocol):
    def __call__(
        self,
        ctx: GenerationContext,
        *,
        remaining_tasks: list["TaskSpec"],
    ) -> SeedSchedulePlan: ...


class DefaultKnowledgeStrategy:
    """The current behaviour, extracted verbatim via phase-service delegation.

    Every hook calls the corresponding phase capability with the same arguments
    the inline ``run()`` loop used.  All gating flags
    (``per_task_forum_rounds``, ``cross_task_forum_rounds``,
    ``distill_enabled``, ``no_memory``, seed flags) continue to be read inside
    those methods exactly as before — this class adds no new branching.
    """

    def per_task_forum(self, ctx: GenerationContext) -> None:
        ctx.phases.per_task_forum(
            ctx.generation,
            ctx.fresh_traces,
            next_task_pool_size=ctx.next_task_pool_size,
        )

    def cross_task_forum(self, ctx: GenerationContext) -> None:
        ctx.phases.cross_task_forum(generation=ctx.generation, traces=ctx.fresh_traces)

    def distill(self, ctx: GenerationContext) -> None:
        ctx.phases.distill(generation=ctx.generation, task_ids=ctx.distill_task_ids())

    def plan_seed_next_generation(
        self,
        ctx: GenerationContext,
        *,
        remaining_tasks: list["TaskSpec"],
    ) -> SeedSchedulePlan:
        if ctx.generation >= ctx.config.num_generations:
            return SeedSchedulePlan(action="skip", reason="final generation")
        if ctx.config.no_memory:
            return SeedSchedulePlan(action="skip", reason="no_memory")

        next_tasks = tuple(ctx.next_remaining_tasks(remaining_tasks))
        if not next_tasks:
            return SeedSchedulePlan(action="stop", reason="all tasks solved")
        return SeedSchedulePlan(action="seed", next_tasks=next_tasks)

    def seed_next_generation(self, ctx: GenerationContext) -> None:
        ctx.phases.seed_next_generation(ctx.generation, next_task_pool_size=ctx.next_task_pool_size)

    def should_enrich(self) -> bool:
        return True


def plan_seed_next_generation(
    strategy: object,
    ctx: GenerationContext,
    *,
    remaining_tasks: list["TaskSpec"],
) -> SeedSchedulePlan:
    """Plan the seed boundary while preserving legacy strategy compatibility."""
    planner = getattr(strategy, "plan_seed_next_generation", None)
    if callable(planner):
        return cast(_SeedPlanner, planner)(ctx, remaining_tasks=remaining_tasks)
    return DefaultKnowledgeStrategy().plan_seed_next_generation(ctx, remaining_tasks=remaining_tasks)


class RawAttemptsStrategy(DefaultKnowledgeStrategy):
    """True knowledge-off ablation: no forums, no distillation, and no
    same-task attempt-history/best-score/memory-snapshot enrichment.
    Seeding still runs.

    This is the strategy-seam expression of the existing flag combination::

        --per-task-forum-rounds 0 --cross-task-forum-rounds 0 \\
        --distill-enabled false   (+ raw-attempts seeding)

    Rather than re-implement the phases, this skips the forum and distill
    hooks entirely (exactly what the gating flags above do inside the default
    methods) and delegates seeding to the engine.  Seeding still runs so the
    next generation's population is sized and task labels are assigned; the
    *content* of seed packages is governed by ``memory_seed_raw_attempts``
    (consumed downstream in the runtime), which this strategy leaves to the
    existing config exactly as the flag combination would.

    The equivalence claim — that this produces the same engine calls as the
    flag combination above — is asserted in
    ``tests/test_improvement_strategy.py``. Unlike that flag combination,
    ``should_enrich()`` returns ``False`` here, so this strategy is strictly
    narrower: it also disables same-task seed-package enrichment.
    """

    def per_task_forum(self, ctx: GenerationContext) -> None:
        # No-op: equivalent to --per-task-forum-rounds 0.
        return

    def cross_task_forum(self, ctx: GenerationContext) -> None:
        # No-op: equivalent to --cross-task-forum-rounds 0.
        return

    def distill(self, ctx: GenerationContext) -> None:
        # No-op: equivalent to --distill-enabled false.
        return

    # seed_next_generation inherited from DefaultKnowledgeStrategy.

    def should_enrich(self) -> bool:
        # True knowledge-off ablation: also skip same-task
        # attempt-history / best-score / memory-snapshot enrichment.
        return False


# --------------------------------------------------------------------------- #
# Strategy registry (mirrors src/kcsi/tasks/registry.py and the eval/runtime
# registries). ONE place lists selectable improvement strategies so the CLI and
# the public API can pick one by name without editing the engine.
#
# Adding a variant:
#     from kcsi.orchestrator.strategy import StrategySpec, register_strategy
#     register_strategy(StrategySpec(name="my_strategy", factory=MyStrategy))
# --------------------------------------------------------------------------- #
STRATEGY_REGISTRY: "dict[str, StrategySpec]" = {}


@dataclass(frozen=True)
class StrategySpec:
    """How to construct one improvement strategy. ``factory`` is called with no
    arguments and returns a fresh ``ImprovementStrategy``."""

    name: str
    factory: Callable[[], ImprovementStrategy]
    aliases: tuple[str, ...] = ()
    description: str = ""

    def all_names(self) -> tuple[str, ...]:
        seen: list[str] = [self.name]
        for alias in self.aliases:
            if alias not in seen:
                seen.append(alias)
        return tuple(seen)


def register_strategy(spec: StrategySpec, *, replace: bool = False) -> StrategySpec:
    """Register ``spec`` (and aliases). Raises ``ValueError`` on duplicate unless
    ``replace=True``. Returns the spec for convenience."""
    if not replace:
        for key in spec.all_names():
            existing = STRATEGY_REGISTRY.get(key)
            if existing is not None:
                raise ValueError(
                    f"improvement strategy name/alias {key!r} already registered to {existing.name!r}; "
                    f"pass replace=True to override"
                )
    for key in spec.all_names():
        STRATEGY_REGISTRY[key] = spec
    return spec


def _normalize_strategy(name: object) -> str:
    return str(name or "").strip().lower()


def resolve_strategy(name: object) -> "StrategySpec | None":
    return STRATEGY_REGISTRY.get(_normalize_strategy(name))


def get_strategy_spec(name: object) -> StrategySpec:
    spec = resolve_strategy(name)
    if spec is None:
        valid = supported_strategies(include_aliases=True)
        raise ValueError(
            f"unsupported improvement strategy={name!r}; valid strategies (incl. aliases): {', '.join(valid)}"
        )
    return spec


def supported_strategies(*, include_aliases: bool = False) -> tuple[str, ...]:
    canonical: list[str] = []
    for spec in STRATEGY_REGISTRY.values():
        if spec.name not in canonical:
            canonical.append(spec.name)
    if not include_aliases:
        return tuple(canonical)
    out = list(canonical)
    for spec in STRATEGY_REGISTRY.values():
        for alias in spec.aliases:
            if alias not in out:
                out.append(alias)
    return tuple(out)


# Built-ins. ``knowledge`` is the default and reproduces the engine's inline
# behaviour; ``raw_attempts`` is the no-forum/no-distill ablation.
register_strategy(
    StrategySpec(
        name="knowledge",
        factory=DefaultKnowledgeStrategy,
        description="Default: per-task + cross-task forum, distillation, seeding (gated by their flags).",
    )
)
register_strategy(
    StrategySpec(
        name="raw_attempts",
        aliases=("raw",),
        factory=RawAttemptsStrategy,
        description="True knowledge-off ablation: skip forums + distillation + same-task enrichment; seeding still runs.",
    )
)

"""Tests for the ImprovementStrategy seam (refactor move 3, #648/#651).

These tests are the equivalence harness for the pure-refactor extraction of
the self-improvement mechanism out of ``GenerationalOrchestrator``.  They
assert two things:

1. The default strategy (:class:`DefaultKnowledgeStrategy`) drives the same
   phase capabilities, with the same arguments, that the old inline ``run()``
   loop called.

2. :class:`RawAttemptsStrategy` produces the *same phase-service calls* as the
   existing flag combination it stands in for
   (``--per-task-forum-rounds 0 --cross-task-forum-rounds 0
   --distill-enabled false``): no forum / no distill, seeding still runs.
"""

from unittest.mock import MagicMock, call

from conftest import _build_mock_evaluator, _build_mock_llm, _build_mock_runtime  # noqa: F401

from kcsi.models import GenerationConfig, TaskSpec, TaskTrace
from kcsi.orchestrator.distillation_phase import DistillationPhaseInput, DistillationPhaseResult
from kcsi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from kcsi.orchestrator.forum_phase import CrossTaskForumPhaseInput, PerTaskForumPhaseInput
from kcsi.orchestrator.phase_services import EngineImprovementPhaseServices
from kcsi.orchestrator.seeding_phase import SeedingPhaseInput, SeedingPhaseResult
from kcsi.orchestrator.strategy import (
    DefaultKnowledgeStrategy,
    GenerationContext,
    ImprovementPhaseServices,
    ImprovementStrategy,
    RawAttemptsStrategy,
    SeedSchedulePlan,
)


def _make_orch(config):
    return GenerationalOrchestrator(
        config=config,
        runtime=_build_mock_runtime(),
        evaluator=_build_mock_evaluator(),
        llm=_build_mock_llm(),
        persistence=NoopPersistence(),
    )


def _make_ctx(
    *,
    generation: int = 1,
    traces: list[TaskTrace] | None = None,
    phases=None,
    next_task_pool_size: int | None = None,
    config=None,
    knowledge=None,
    next_remaining_tasks=None,
) -> GenerationContext:
    return GenerationContext(
        generation=generation,
        fresh_traces=list(traces or []),
        phases=phases or MagicMock(),
        next_task_pool_size=next_task_pool_size,
        config=config,
        knowledge=knowledge,
        next_remaining_tasks=next_remaining_tasks or (lambda tasks: list(tasks)),
    )


# --------------------------------------------------------------------------- #
# Seam wiring
# --------------------------------------------------------------------------- #
def test_default_strategy_is_wired_by_default():
    orch = _make_orch(GenerationConfig(num_generations=1, num_agents=1))
    assert isinstance(orch._improvement_strategy, DefaultKnowledgeStrategy)
    assert isinstance(orch._improvement_strategy, ImprovementStrategy)


def test_set_improvement_strategy_swaps_the_seam():
    orch = _make_orch(GenerationConfig(num_generations=1, num_agents=1))
    raw = RawAttemptsStrategy()
    orch.set_improvement_strategy(raw)
    assert orch._improvement_strategy is raw


def test_raw_attempts_satisfies_protocol():
    assert isinstance(RawAttemptsStrategy(), ImprovementStrategy)


def test_default_strategy_should_enrich_true():
    assert DefaultKnowledgeStrategy().should_enrich() is True


def test_raw_attempts_should_enrich_false():
    assert RawAttemptsStrategy().should_enrich() is False


# --------------------------------------------------------------------------- #
# GenerationContext mirrors the inline computations
# --------------------------------------------------------------------------- #
def test_context_distill_task_ids_matches_inline_computation():
    traces = [
        TaskTrace(agent_id="a", task_id=" t2 ", generation=1),
        TaskTrace(agent_id="a", task_id="t1", generation=1),
        TaskTrace(agent_id="b", task_id="t1", generation=1),  # duplicate
        TaskTrace(agent_id="b", task_id="  ", generation=1),  # blank -> dropped
    ]
    ctx = _make_ctx(generation=1, traces=traces)
    # Same expression the old inline _distill_task_ids used:
    expected = sorted({str(t.task_id).strip() for t in traces if str(t.task_id).strip()})
    assert ctx.distill_task_ids() == expected == ["t1", "t2"]


def test_context_passthroughs():
    config = GenerationConfig(num_generations=1, num_agents=1)
    orch = _make_orch(config)
    ctx = _make_ctx(generation=3, config=orch.config, knowledge=orch._knowledge)
    assert ctx.config is orch.config
    assert ctx.knowledge is orch._knowledge
    assert not hasattr(ctx, "engine")


# --------------------------------------------------------------------------- #
# Seed scheduling is strategy-owned, not an engine-side flag gate.
# --------------------------------------------------------------------------- #
def test_default_strategy_plans_seed_next_generation_with_strategy_task_set():
    tasks = [TaskSpec(id="t1", prompt="one"), TaskSpec(id="t2", prompt="two")]
    ctx = _make_ctx(
        generation=1,
        config=GenerationConfig(num_generations=2, num_agents=1),
        next_remaining_tasks=lambda remaining: [remaining[1]],
    )

    plan = DefaultKnowledgeStrategy().plan_seed_next_generation(ctx, remaining_tasks=tasks)

    assert plan == SeedSchedulePlan(action="seed", next_tasks=(tasks[1],))


def test_default_strategy_skips_seed_after_final_generation_without_planning_tasks():
    def fail_if_called(_remaining):
        raise AssertionError("final-generation seed planning should skip before task retention")

    ctx = _make_ctx(
        generation=2,
        config=GenerationConfig(num_generations=2, num_agents=1),
        next_remaining_tasks=fail_if_called,
    )

    plan = DefaultKnowledgeStrategy().plan_seed_next_generation(ctx, remaining_tasks=[TaskSpec(id="t1")])

    assert plan == SeedSchedulePlan(action="skip", reason="final generation")


def test_default_strategy_skips_seed_for_no_memory_without_planning_tasks():
    def fail_if_called(_remaining):
        raise AssertionError("no_memory seed planning should skip before task retention")

    ctx = _make_ctx(
        generation=1,
        config=GenerationConfig(num_generations=2, num_agents=1, no_memory=True),
        next_remaining_tasks=fail_if_called,
    )

    plan = DefaultKnowledgeStrategy().plan_seed_next_generation(ctx, remaining_tasks=[TaskSpec(id="t1")])

    assert plan == SeedSchedulePlan(action="skip", reason="no_memory")


def test_default_strategy_stops_when_strategy_task_set_is_empty():
    ctx = _make_ctx(
        generation=1,
        config=GenerationConfig(num_generations=2, num_agents=1),
        next_remaining_tasks=lambda _remaining: [],
    )

    plan = DefaultKnowledgeStrategy().plan_seed_next_generation(ctx, remaining_tasks=[TaskSpec(id="t1")])

    assert plan == SeedSchedulePlan(action="stop", reason="all tasks solved")


def test_engine_asks_strategy_for_seed_schedule_even_when_no_memory():
    class SpyStrategy(DefaultKnowledgeStrategy):
        def __init__(self):
            self.plan_generations: list[int] = []
            self.seed_calls = 0

        def plan_seed_next_generation(self, ctx, *, remaining_tasks):
            self.plan_generations.append(ctx.generation)
            return SeedSchedulePlan(action="skip", reason="test")

        def seed_next_generation(self, ctx):
            self.seed_calls += 1

    config = GenerationConfig(num_generations=2, num_agents=1, no_memory=True)
    orch = GenerationalOrchestrator(
        config=config,
        runtime=_build_mock_runtime(),
        evaluator=_build_mock_evaluator(resolved=False, native_score=0.0),
        llm=_build_mock_llm(),
        persistence=NoopPersistence(),
    )
    strategy = SpyStrategy()
    orch.set_improvement_strategy(strategy)

    traces = orch.run([TaskSpec(id="t1", prompt="fix bug")])

    assert strategy.plan_generations == [1, 2]
    assert strategy.seed_calls == 0
    assert [trace.generation for trace in traces] == [1, 2]


def test_engine_falls_back_for_legacy_strategy_without_seed_planner():
    class LegacyStrategy:
        def per_task_forum(self, ctx):
            return None

        def cross_task_forum(self, ctx):
            return None

        def distill(self, ctx):
            return None

        def seed_next_generation(self, ctx):
            return None

        def should_enrich(self):
            return True

    config = GenerationConfig(num_generations=2, num_agents=1, no_memory=True)
    orch = GenerationalOrchestrator(
        config=config,
        runtime=_build_mock_runtime(),
        evaluator=_build_mock_evaluator(resolved=False, native_score=0.0),
        llm=_build_mock_llm(),
        persistence=NoopPersistence(),
    )
    orch.set_improvement_strategy(LegacyStrategy())

    traces = orch.run([TaskSpec(id="t1", prompt="fix bug")])

    assert [trace.generation for trace in traces] == [1, 2]


def test_engine_uses_strategy_next_tasks_for_next_generation_dispatch():
    class OneTaskNextGenStrategy(DefaultKnowledgeStrategy):
        def plan_seed_next_generation(self, ctx, *, remaining_tasks):
            if ctx.generation == 1:
                return SeedSchedulePlan(action="seed", next_tasks=(remaining_tasks[1],))
            return SeedSchedulePlan(action="skip", reason="done")

    config = GenerationConfig(num_generations=2, num_agents=2, no_memory=True, drop_solved=False)
    orch = GenerationalOrchestrator(
        config=config,
        runtime=_build_mock_runtime(),
        evaluator=_build_mock_evaluator(resolved=False, native_score=0.0),
        llm=_build_mock_llm(),
        persistence=NoopPersistence(),
    )
    orch.set_improvement_strategy(OneTaskNextGenStrategy())

    traces = orch.run([TaskSpec(id="t1", prompt="one"), TaskSpec(id="t2", prompt="two")])

    by_generation = {
        gen: [trace.task_id for trace in traces if trace.generation == gen]
        for gen in sorted({trace.generation for trace in traces})
    }
    assert sorted(by_generation[1]) == ["t1", "t2"]
    assert by_generation[2] == ["t2"]


# --------------------------------------------------------------------------- #
# DefaultKnowledgeStrategy delegates to explicit phase services,
# with the SAME arguments the inline loop used.
# --------------------------------------------------------------------------- #
def test_default_strategy_delegates_to_phase_services():
    phases = MagicMock()
    traces = [TaskTrace(agent_id="a", task_id="t1", generation=1)]
    ctx = _make_ctx(generation=1, traces=traces, phases=phases, next_task_pool_size=4)

    strat = DefaultKnowledgeStrategy()
    strat.per_task_forum(ctx)
    strat.cross_task_forum(ctx)
    strat.distill(ctx)
    strat.seed_next_generation(ctx)

    phases.per_task_forum.assert_called_once_with(1, traces, next_task_pool_size=4)
    phases.cross_task_forum.assert_called_once_with(generation=1, traces=traces)
    phases.distill.assert_called_once_with(generation=1, task_ids=["t1"])
    phases.seed_next_generation.assert_called_once_with(1, next_task_pool_size=4)


def test_engine_phase_services_delegate_to_phase_services():
    """The aggregate service routes hooks to phase-specific services."""
    orch = _make_orch(GenerationConfig(num_generations=2, num_agents=1))
    forum_phase = MagicMock()
    distillation_phase = MagicMock()
    distillation_phase.run.return_value = DistillationPhaseResult()
    seeding_phase = MagicMock()
    seeding_phase.run.return_value = SeedingPhaseResult()
    phases = EngineImprovementPhaseServices(
        orch,
        forum_phase=forum_phase,
        distillation_phase=distillation_phase,
        seeding_phase=seeding_phase,
    )
    traces = [TaskTrace(agent_id="a", task_id="t1", generation=1)]

    assert isinstance(phases, ImprovementPhaseServices)
    phases.per_task_forum(1, traces, next_task_pool_size=4)
    phases.cross_task_forum(generation=1, traces=traces)
    phases.distill(generation=1, task_ids=["t1"])
    phases.seed_next_generation(1, next_task_pool_size=4)

    forum_phase.per_task_forum.assert_called_once_with(
        PerTaskForumPhaseInput(generation=1, traces=traces, next_task_pool_size=4)
    )
    forum_phase.cross_task_forum.assert_called_once_with(CrossTaskForumPhaseInput(generation=1, traces=traces))
    distillation_phase.run.assert_called_once_with(DistillationPhaseInput(generation=1, task_ids=["t1"]))
    seeding_phase.run.assert_called_once_with(SeedingPhaseInput(generation=1, next_task_pool_size=4))


# --------------------------------------------------------------------------- #
# Equivalence: RawAttemptsStrategy == the flag combination.
#
# We drive the FULL run() loop and capture which phase-service methods are
# invoked, comparing:
#   (A) DefaultKnowledgeStrategy + flags that disable forum/distill, vs
#   (B) RawAttemptsStrategy + default flags.
# Both must invoke exactly the same phase capabilities with the same args.
# --------------------------------------------------------------------------- #
def _spy_phase_calls(orch):
    """Wrap each phase-service method to record calls without changing behavior."""
    calls = []
    for name in (
        "per_task_forum",
        "cross_task_forum",
        "distill",
        "seed_next_generation",
    ):
        original = getattr(orch._improvement_phases, name)

        def _wrap(orig, nm):
            def _inner(*args, **kwargs):
                calls.append((nm, args, kwargs))
                return orig(*args, **kwargs)

            return _inner

        setattr(orch._improvement_phases, name, _wrap(original, name))
    return calls


def _run_capturing(config, strategy=None):
    orch = _make_orch(config)
    if strategy is not None:
        orch.set_improvement_strategy(strategy)
    calls = _spy_phase_calls(orch)
    orch.run([])  # empty task list: phases gated; no container needed
    return calls


def test_raw_attempts_matches_flag_combination_calls():
    # This asserts forum/distill call-log parity only. `raw_attempts` is
    # strictly narrower than the flag combination because it also disables
    # enrichment (#987) — see
    # `test_raw_attempts_strategy_skips_enrichment_entirely` in
    # `tests/test_enrich_seed_packages.py` for that behavior.
    #
    # (A) Default strategy with the flag combination that disables the
    #     improvement phases.
    flag_config = GenerationConfig(num_generations=1, num_agents=1)
    flag_config.per_task_forum_rounds = 0
    flag_config.cross_task_forum_rounds = 0
    flag_config.distill_enabled = False
    calls_flags = _run_capturing(flag_config)

    # (B) RawAttemptsStrategy with default (forum-enabled) flags.
    raw_config = GenerationConfig(num_generations=1, num_agents=1)
    raw_config.per_task_forum_rounds = 1
    raw_config.cross_task_forum_rounds = 1
    raw_config.distill_enabled = True
    calls_raw = _run_capturing(raw_config, strategy=RawAttemptsStrategy())

    # With an empty task list, num_generations=1, the inline loop never reaches
    # any phase (it breaks on "no remaining tasks"), so this empty-task-list
    # run never exercises enrichment either way. The meaningful equivalence
    # is that neither variant invokes the forum/distill phase BODIES, which the
    # direct-hook test below covers. Here we assert both produced identical call
    # logs (both empty) as a regression guard on the run() wiring.
    assert calls_flags == calls_raw == []


def test_raw_attempts_skips_forum_and_distill_but_seeds():
    """Directly exercise the RawAttemptsStrategy hooks and assert it never
    touches forum/distill phase services, but does delegate seeding."""
    orch = _make_orch(GenerationConfig(num_generations=2, num_agents=1))
    forum_phase = MagicMock()
    distillation_phase = MagicMock()
    distillation_phase.run.return_value = DistillationPhaseResult()
    seeding_phase = MagicMock()
    seeding_phase.run.return_value = SeedingPhaseResult()
    phases = EngineImprovementPhaseServices(
        orch,
        forum_phase=forum_phase,
        distillation_phase=distillation_phase,
        seeding_phase=seeding_phase,
    )

    ctx = _make_ctx(
        generation=1,
        traces=[TaskTrace(agent_id="a", task_id="t1", generation=1)],
        phases=phases,
        next_task_pool_size=2,
    )

    strat = RawAttemptsStrategy()
    strat.per_task_forum(ctx)
    strat.cross_task_forum(ctx)
    strat.distill(ctx)
    strat.seed_next_generation(ctx)

    forum_phase.per_task_forum.assert_not_called()
    forum_phase.cross_task_forum.assert_not_called()
    distillation_phase.run.assert_not_called()
    seeding_phase.run.assert_called_once_with(SeedingPhaseInput(generation=1, next_task_pool_size=2))


def test_raw_attempts_seed_equals_default_seed_call():
    """The raw strategy's seed hook must issue the identical engine call the
    default strategy issues (seeding is shared, not disabled)."""
    orch = _make_orch(GenerationConfig(num_generations=2, num_agents=1))
    seeding_phase = MagicMock()
    seeding_phase.run.return_value = SeedingPhaseResult()
    ctx = _make_ctx(
        generation=2,
        phases=EngineImprovementPhaseServices(orch, seeding_phase=seeding_phase),
        next_task_pool_size=5,
    )

    DefaultKnowledgeStrategy().seed_next_generation(ctx)
    RawAttemptsStrategy().seed_next_generation(ctx)

    assert seeding_phase.run.call_args_list == [
        call(SeedingPhaseInput(generation=2, next_task_pool_size=5)),
        call(SeedingPhaseInput(generation=2, next_task_pool_size=5)),
    ]


def test_raw_attempts_equivalent_to_flag_combination_at_knowledge_boundary():
    """Behavioural equivalence of RawAttemptsStrategy and the flag combination
    it stands in for, observed at the KnowledgeStore boundary.

    This asserts forum/distill call-log parity only. `raw_attempts` is
    strictly narrower than the flag combination because it also disables
    enrichment (#987) — see
    `test_raw_attempts_strategy_skips_enrichment_entirely` in
    `tests/test_enrich_seed_packages.py` for that behavior.

    The flag gating (``per_task_forum_rounds==0`` etc.) lives *inside* the real
    phase bodies, so RawAttemptsStrategy (which skips the calls) and
    DefaultKnowledgeStrategy-with-zeroed-flags (which calls bodies that then
    early-return) are not call-count identical — but they must be identical in
    their effect on the knowledge substrate: neither writes a forum post or a
    distillation bundle, and both delegate seeding identically.

    We drive the REAL forum/distill phase bodies (not mocks) with a spy
    KnowledgeStore so the gating is actually exercised, then compare the
    knowledge-write footprint of the two paths.
    """

    def _knowledge_footprint(strategy, *, zero_flags):
        config = GenerationConfig(num_generations=2, num_agents=1)
        # Default strategy stands in for "flag combination"; raw strategy uses
        # the forum-enabled flags to prove the *strategy* (not the flags) does
        # the disabling.
        config.per_task_forum_rounds = 0 if zero_flags else 1
        config.cross_task_forum_rounds = 0 if zero_flags else 1
        config.distill_enabled = not zero_flags
        orch = _make_orch(config)
        # Real bodies run; a spy knowledge store records any write attempts.
        knowledge = MagicMock()
        orch._knowledge = knowledge
        seeding_phase = MagicMock()
        seeding_phase.run.return_value = SeedingPhaseResult()
        ctx = _make_ctx(
            generation=2,
            traces=[TaskTrace(agent_id="a", task_id="t1", generation=1)],
            phases=EngineImprovementPhaseServices(orch, seeding_phase=seeding_phase),
            next_task_pool_size=3,
        )
        strategy.per_task_forum(ctx)
        strategy.cross_task_forum(ctx)
        strategy.distill(ctx)
        strategy.seed_next_generation(ctx)
        return {
            "record_post": knowledge.record_post.call_count,
            "record_distillation": knowledge.record_distillation.call_count,
            "seed": seeding_phase.run.call_args_list,
        }

    flags = _knowledge_footprint(DefaultKnowledgeStrategy(), zero_flags=True)
    raw = _knowledge_footprint(RawAttemptsStrategy(), zero_flags=False)

    # Neither path writes forum posts or distillation bundles.
    assert flags["record_post"] == raw["record_post"] == 0
    assert flags["record_distillation"] == raw["record_distillation"] == 0
    # Seeding is shared and identical between the two paths.
    assert flags["seed"] == raw["seed"] == [call(SeedingPhaseInput(generation=2, next_task_pool_size=3))]


# --------------------------------------------------------------------------- #
# Strategy registry + CLI selection (PR3): make the seam selectable by name.
# --------------------------------------------------------------------------- #
def test_strategy_registry_resolves_builtins():
    from kcsi.orchestrator.strategy import get_strategy_spec, supported_strategies

    assert supported_strategies() == ("knowledge", "raw_attempts")
    assert supported_strategies(include_aliases=True) == ("knowledge", "raw_attempts", "raw")
    assert isinstance(get_strategy_spec("knowledge").factory(), DefaultKnowledgeStrategy)
    assert isinstance(get_strategy_spec("raw_attempts").factory(), RawAttemptsStrategy)
    # Alias + case/whitespace normalization.
    assert isinstance(get_strategy_spec("  RAW ").factory(), RawAttemptsStrategy)


def test_strategy_registry_unknown_raises():
    import pytest

    from kcsi.orchestrator.strategy import get_strategy_spec

    with pytest.raises(ValueError) as exc:
        get_strategy_spec("does_not_exist")
    assert "does_not_exist" in str(exc.value)


def test_strategy_registry_register_duplicate_raises_then_replace():
    import pytest

    from kcsi.orchestrator.strategy import (
        STRATEGY_REGISTRY,
        StrategySpec,
        register_strategy,
    )

    spec = StrategySpec(name="dummy_strat_pr3", factory=DefaultKnowledgeStrategy)
    register_strategy(spec)
    try:
        with pytest.raises(ValueError):
            register_strategy(StrategySpec(name="dummy_strat_pr3", factory=DefaultKnowledgeStrategy))
        register_strategy(StrategySpec(name="dummy_strat_pr3", factory=RawAttemptsStrategy), replace=True)
        assert STRATEGY_REGISTRY["dummy_strat_pr3"].factory is RawAttemptsStrategy
    finally:
        STRATEGY_REGISTRY.pop("dummy_strat_pr3", None)


def test_cli_parser_accepts_improvement_strategy_flag():
    from kcsi.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "arc",
            "--tasks-path",
            "/tmp/tasks.json",
            "--knowledge-db-path",
            "/tmp/knowledge.sqlite",
            "--improvement-strategy",
            "raw_attempts",
        ]
    )
    assert args.improvement_strategy == "raw_attempts"


def test_improvement_strategy_defaults_to_knowledge():
    from kcsi.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "arc",
            "--tasks-path",
            "/tmp/tasks.json",
            "--knowledge-db-path",
            "/tmp/knowledge.sqlite",
        ]
    )
    assert args.improvement_strategy == "knowledge"


def test_cli_wiring_contract_sets_strategy_on_orchestrator():
    """Exercise the exact contract cli.main() uses to apply the flag:
    get_strategy_spec(args.improvement_strategy).factory() -> set on engine."""
    from kcsi.orchestrator.strategy import get_strategy_spec

    orch = _make_orch(GenerationConfig(num_generations=1, num_agents=1))
    orch.set_improvement_strategy(get_strategy_spec("raw_attempts").factory())
    assert isinstance(orch._improvement_strategy, RawAttemptsStrategy)

    # Default selection rebuilds the engine's own default (behavior-preserving).
    orch.set_improvement_strategy(get_strategy_spec("knowledge").factory())
    assert isinstance(orch._improvement_strategy, DefaultKnowledgeStrategy)


def test_default_strategy_matches_engine_autoinstalled_default():
    """The default 'knowledge' selection must rebuild the engine's OWN
    auto-installed default — that equivalence is what makes the default
    --improvement-strategy behavior-preserving. Pin the type so a future
    divergence (engine default changes, or 'knowledge' maps elsewhere) fails."""
    from kcsi.orchestrator.strategy import get_strategy_spec

    orch = _make_orch(GenerationConfig(num_generations=1, num_agents=1))
    engine_autoinstalled = type(orch._improvement_strategy)
    orch.set_improvement_strategy(get_strategy_spec("knowledge").factory())
    assert type(orch._improvement_strategy) is engine_autoinstalled is DefaultKnowledgeStrategy

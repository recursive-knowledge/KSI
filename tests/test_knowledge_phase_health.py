"""knowledge_phase_health: per-generation degradation visibility (issue #740).

Covers the two layers of the feature:
1. Engine accumulator: ``_record_knowledge_phase_failure`` +
   ``knowledge_phase_health_by_generation`` (counter logic + stable schema).
2. Distill threading: the distillation phase service rolls the distiller's ``DistillOutput.failures``
   (and a wholesale distill exception) into the per-generation health block.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ksi.distillation import DistillOutput, PerTaskBundle
from ksi.memory.forum_bus import ForumBus
from ksi.models import GenerationConfig
from ksi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence, _drain_forum_bus
from ksi.tokens import LLMResponse, TokenUsage
from tests.orchestrator_phase_helpers import load_cross_task_seed_bundle, run_distill


def _make_orch(tmp_path) -> GenerationalOrchestrator:
    db_path = str(tmp_path / "knowledge.sqlite")
    llm = MagicMock()
    llm.call.return_value = LLMResponse(text="{}", usage=TokenUsage(input_tokens=1, output_tokens=1))
    config = GenerationConfig(num_generations=1, num_agents=1, knowledge_db_path=db_path)
    return GenerationalOrchestrator(
        config=config,
        runtime=MagicMock(),
        evaluator=MagicMock(),
        llm=llm,
        persistence=NoopPersistence(),
    )


# ── Layer 1: engine accumulator ───────────────────────────────────────────────


def test_accessor_empty_when_healthy(tmp_path):
    orch = _make_orch(tmp_path)
    assert orch.knowledge_phase_health_by_generation() == {}


def test_record_and_accessor_emits_full_schema(tmp_path):
    orch = _make_orch(tmp_path)
    orch._record_knowledge_phase_failure(2, "drain_failures")
    orch._record_knowledge_phase_failure(2, "drain_failures")
    orch._record_knowledge_phase_failure(2, "forum_agent_failures")
    orch._record_knowledge_phase_failure(3, "distill_failures", 5)

    out = orch.knowledge_phase_health_by_generation()
    # Every recorded generation carries the complete kind schema (zeros filled).
    assert out == {
        2: {"drain_failures": 2, "forum_agent_failures": 1, "distill_failures": 0, "seed_failures": 0},
        3: {"drain_failures": 0, "forum_agent_failures": 0, "distill_failures": 5, "seed_failures": 0},
    }


def test_record_nonpositive_is_noop(tmp_path):
    orch = _make_orch(tmp_path)
    orch._record_knowledge_phase_failure(1, "distill_failures", 0)
    orch._record_knowledge_phase_failure(1, "distill_failures", -3)
    assert orch.knowledge_phase_health_by_generation() == {}


def test_measured_flag_true_with_memory_false_without(tmp_path):
    # {} is ambiguous: clean vs not-measured. The measured flag disambiguates
    # (#827, M7). Memory-enabled (default) -> measured; --no-memory -> not.
    orch = _make_orch(tmp_path)
    assert orch.knowledge_phase_health_measured() is True

    orch.config.no_memory = True
    assert orch.knowledge_phase_health_measured() is False


def test_seed_load_failure_records_seed_failure(tmp_path, monkeypatch):
    # A cross-task seed-bundle load failure silently degrades the NEXT
    # generation's seed; it must surface as a seed_failures event (#827, M5).
    orch = _make_orch(tmp_path)

    def boom(*_args, **_kwargs):
        raise RuntimeError("load_distillation exploded")

    monkeypatch.setattr(orch._knowledge, "load_distillation", boom)
    result = load_cross_task_seed_bundle(orch, generation=7)

    assert result is None
    assert orch.knowledge_phase_health_by_generation()[7]["seed_failures"] == 1


def test_drain_drop_callback_records_exact_drop_count(tmp_path, monkeypatch):
    orch = _make_orch(tmp_path)
    bus = ForumBus(db_path=str(tmp_path / "forum.sqlite"), experiment="default", generation=1)
    for idx in range(2):
        bus.append(
            round_num=0,
            agent_id=f"agent-{idx}",
            message_type="post",
            content={"task_id": f"task-{idx}", "text": "dropped"},
        )

    def fail_record_post(*_args, **_kwargs):
        raise RuntimeError("knowledge write failed")

    # The drain batches knowledge-row writes into one transaction; each event
    # runs through ``_record_post_locked`` inside its own SAVEPOINT, so inject
    # the failure there to exercise the per-event drop accounting.
    monkeypatch.setattr(orch._knowledge, "_record_post_locked", fail_record_post)
    drained = _drain_forum_bus(
        forum_bus=bus,
        knowledge=orch._knowledge,
        generation=1,
        experiment=orch.config.experiment_name,
        on_drop=lambda n: orch._record_knowledge_phase_failure(1, "drain_failures", n),
    )

    assert drained == 0
    assert orch.knowledge_phase_health_by_generation()[1]["drain_failures"] == 2


# ── Layer 2: distill phase threads DistillOutput.failures ──────────────────────


def test_distill_phase_records_distiller_failures(tmp_path, monkeypatch):
    orch = _make_orch(tmp_path)
    calls: list[int] = []

    def fake_distill(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
        calls.append(1)
        return DistillOutput(per_task={}, cross_task=None, failures=3)

    import ksi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", fake_distill)
    run_distill(orch, generation=4, task_ids=["t1"])

    # Guard against a silent early-exit (no_memory / no knowledge / no tasks /
    # distill disabled) that would skip distill() and make the assertion below
    # vacuous: confirm the stub actually ran.
    assert calls, "distillation phase early-exited before calling distill()"
    assert orch.knowledge_phase_health_by_generation()[4]["distill_failures"] == 3


def test_distill_phase_wholesale_exception_records_one_failure(tmp_path, monkeypatch):
    orch = _make_orch(tmp_path)
    calls: list[int] = []

    def boom(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
        calls.append(1)
        raise RuntimeError("distill exploded")

    import ksi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", boom)
    run_distill(orch, generation=2, task_ids=["t1"])

    assert calls, "distillation phase early-exited before calling distill()"
    assert orch.knowledge_phase_health_by_generation()[2]["distill_failures"] == 1


def test_distill_phase_records_persistence_failures(tmp_path, monkeypatch):
    orch = _make_orch(tmp_path)
    calls: list[int] = []

    def fake_distill(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
        calls.append(1)
        return DistillOutput(
            per_task={
                "t1": PerTaskBundle(
                    task_id="t1",
                    transferable_insights=["i"],
                    pitfalls=[],
                    checks=[],
                    evidence_post_ids=[],
                )
            },
            cross_task=None,
            failures=0,
        )

    def fail_record_distillation(*args, **kwargs):
        raise RuntimeError("record failed")

    import ksi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", fake_distill)
    monkeypatch.setattr(orch._knowledge, "record_distillation", fail_record_distillation)
    run_distill(orch, generation=5, task_ids=["t1"])

    assert calls, "distillation phase early-exited before calling distill()"
    assert orch.knowledge_phase_health_by_generation()[5]["distill_failures"] == 1


def test_distill_output_failures_defaults_zero():
    assert DistillOutput(per_task={}, cross_task=None).failures == 0

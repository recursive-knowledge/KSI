"""Tests for the distillation phase service (Plan Task 14).

The distill phase calls :func:`kcsi.distillation.distill` with a
``DistillInput`` containing:
- the current generation
- the set of task ids that had attempts this generation
- the KnowledgeStore instance
- an ``LLMCallable`` adapter around the orchestrator's LLM

Returned bundles are persisted via ``KnowledgeStore.record_distillation``:
- per-task bundles with ``scope="per_task"`` and the matching ``task_id``
- the cross-task bundle (if any) with ``scope="cross_task"`` and
  ``task_id=CROSS_TASK_SENTINEL``.

If :func:`distill` raises, the phase logs a warning and returns cleanly
— no cascading retry, no crash.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from kcsi.memory.knowledge_store import CROSS_TASK_SENTINEL
from kcsi.models import GenerationConfig
from kcsi.orchestrator.distillation_phase import (
    DistillationPhaseInput,
    DistillationPhaseResult,
    EngineDistillationPhaseService,
)
from kcsi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from kcsi.tokens import LLMResponse, TokenUsage
from tests.orchestrator_phase_decoupling_guard import functions_referencing_engine
from tests.orchestrator_phase_helpers import run_distill


def _make_orch(tmp_path) -> GenerationalOrchestrator:
    db_path = str(tmp_path / "knowledge.sqlite")
    runtime = MagicMock()
    evaluator = MagicMock()
    llm = MagicMock()
    llm.call.return_value = LLMResponse(
        text=json.dumps(
            {
                "transferable_insights": [],
                "pitfalls": [],
                "checks": [],
                "evidence_post_ids": [],
            }
        ),
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        knowledge_db_path=db_path,
    )
    return GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )


def test_engine_distillation_phase_service_exposes_run(tmp_path):
    orch = _make_orch(tmp_path)
    assert callable(EngineDistillationPhaseService(orch).run)


def test_distill_phase_test_helper_delegates_to_service(tmp_path):
    orch = _make_orch(tmp_path)
    fake_service = MagicMock()
    fake_service.run.return_value = DistillationPhaseResult()
    orch._distillation_phase = fake_service

    run_distill(orch, generation=3, task_ids=["alpha", "beta"])

    fake_service.run.assert_called_once_with(DistillationPhaseInput(generation=3, task_ids=["alpha", "beta"]))


def test_distill_called_with_generation_and_task_ids(tmp_path, monkeypatch, caplog):
    """``distill`` receives a DistillInput with the right gen + task ids."""
    from kcsi.distillation import DistillOutput

    orch = _make_orch(tmp_path)

    captured: dict = {}

    def fake_distill(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
        captured["generation"] = inp.generation
        captured["task_ids"] = list(inp.task_ids)
        captured["knowledge_store"] = inp.knowledge_store
        captured["llm"] = inp.llm
        return DistillOutput(per_task={}, cross_task=None)

    # Patch at the engine-module import path — the engine imports distill
    # from kcsi.distillation at call time, so we patch both locations.
    import kcsi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", fake_distill)

    with caplog.at_level("WARNING"):
        run_distill(orch, generation=3, task_ids=["alpha", "beta"])

    assert captured["generation"] == 3
    assert captured["task_ids"] == ["alpha", "beta"]
    assert not any("distill phase failed" in rec.message for rec in caplog.records)
    # The knowledge store passed through is the engine's store.
    assert captured["knowledge_store"] is orch._knowledge
    # llm must be callable (str, str) -> str
    out = captured["llm"]("sys", "user")
    assert isinstance(out, str)


def test_persists_per_task_and_cross_task_bundles(tmp_path, monkeypatch):
    """Output bundles land in KnowledgeStore with the right scope/task_id."""
    from kcsi.distillation import DistillOutput
    from kcsi.distillation.types import CrossTaskBundle, PerTaskBundle

    orch = _make_orch(tmp_path)

    per_task_bundle = PerTaskBundle(
        task_id="t1",
        transferable_insights=["insight-1"],
        pitfalls=["pitfall-1"],
        checks=["check-1"],
        evidence_post_ids=[42],
    )
    cross_task_bundle = CrossTaskBundle(
        transferable_insights=["cross-1"],
        pitfalls=[],
        checks=[],
        evidence_post_ids=[],
    )

    def fake_distill(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
        return DistillOutput(
            per_task={"t1": per_task_bundle},
            cross_task=cross_task_bundle,
        )

    import kcsi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", fake_distill)

    result = EngineDistillationPhaseService(orch).run(DistillationPhaseInput(generation=1, task_ids=["t1"]))

    per = orch._knowledge.load_distillation(
        generation=1,
        task_id="t1",
        scope="per_task",
    )
    cross = orch._knowledge.load_distillation(
        generation=1,
        task_id=CROSS_TASK_SENTINEL,
        scope="cross_task",
    )
    assert per is not None
    assert per.get("transferable_insights") == ["insight-1"]
    assert per.get("pitfalls") == ["pitfall-1"]
    assert cross is not None
    assert cross.get("transferable_insights") == ["cross-1"]
    assert result == DistillationPhaseResult(
        attempted_task_ids=("t1",),
        persisted_per_task=1,
        persisted_cross_task=True,
    )


def test_per_task_embeddings_computed_in_one_batch_and_aligned(tmp_path, monkeypatch):
    """Per-task bundles embed in a single batch call, aligned to each bundle."""
    from kcsi.distillation import DistillOutput
    from kcsi.distillation.types import PerTaskBundle

    orch = _make_orch(tmp_path)

    bundles = {
        tid: PerTaskBundle(
            task_id=tid,
            transferable_insights=[f"insight-{tid}"],
            pitfalls=[],
            checks=[],
            evidence_post_ids=[],
        )
        for tid in ("t1", "t2", "t3")
    }

    def fake_distill(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
        return DistillOutput(per_task=bundles, cross_task=None)

    import kcsi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", fake_distill)

    # One batch call returns a distinct sentinel vector per input text.
    batch_calls: list[list[str]] = []

    def fake_batch(texts):
        batch_calls.append(list(texts))
        return [[float(i)] for i in range(len(texts))]

    monkeypatch.setattr(orch, "_maybe_embed_batch", fake_batch)

    recorded: list[tuple[str, object]] = []
    real_record = orch._knowledge.record_distillation

    def spy_record(*, task_id, embedding=None, **kw):
        recorded.append((task_id, embedding))
        return real_record(task_id=task_id, embedding=embedding, **kw)

    monkeypatch.setattr(orch._knowledge, "record_distillation", spy_record)

    EngineDistillationPhaseService(orch).run(DistillationPhaseInput(generation=1, task_ids=["t1", "t2", "t3"]))

    # Exactly one batched embed call, covering all three bundles.
    assert len(batch_calls) == 1
    assert len(batch_calls[0]) == 3
    # Each per-task record got the embedding aligned to its position.
    per_task_records = [r for r in recorded if r[0] in ("t1", "t2", "t3")]
    assert [emb for _, emb in per_task_records] == [[0.0], [1.0], [2.0]]


def test_distill_failure_logs_and_returns(tmp_path, monkeypatch, caplog):
    """If distill raises, the phase logs and returns cleanly (no raise)."""
    orch = _make_orch(tmp_path)

    def fake_distill(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
        raise RuntimeError("LLM exploded")

    import kcsi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", fake_distill)

    # Should not raise.
    with caplog.at_level("WARNING"):
        run_distill(orch, generation=1, task_ids=["t1"])

    # And nothing should have been persisted under gen=1.
    per = orch._knowledge.load_distillation(
        generation=1,
        task_id="t1",
        scope="per_task",
    )
    assert per is None
    cross = orch._knowledge.load_distillation(
        generation=1,
        task_id=CROSS_TASK_SENTINEL,
        scope="cross_task",
    )
    assert cross is None
    # A warning about the failure should be present.
    assert any("distill" in rec.message.lower() for rec in caplog.records)


def test_empty_task_ids_is_noop(tmp_path, monkeypatch):
    """If no tasks had attempts this generation, distill is skipped."""
    from kcsi.distillation import DistillOutput

    orch = _make_orch(tmp_path)
    called = {"n": 0}

    def fake_distill(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
        called["n"] += 1
        return DistillOutput(per_task={}, cross_task=None)

    import kcsi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", fake_distill)

    run_distill(orch, generation=1, task_ids=[])
    # With no task_ids there's nothing to distill.
    assert called["n"] == 0


def test_distill_model_overrides_are_propagated(tmp_path, monkeypatch):
    """When ``distill_per_task_model`` / ``distill_cross_task_model`` are
    configured, the engine must build per-phase LLM callables that invoke
    ``self.llm.call`` with the corresponding ``model=`` kwarg. The distiller
    then picks those callables over the default ``inp.llm``.
    """
    from kcsi.distillation import DistillOutput

    orch = _make_orch(tmp_path)
    # Configure per-phase overrides on the config (engine reads via getattr).
    orch.config.distill_per_task_model = "claude-per-task-model"
    orch.config.distill_cross_task_model = "claude-cross-task-model"

    captured = {}

    def fake_distill(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
        captured["llm_per_task"] = inp.llm_per_task
        captured["llm_cross_task"] = inp.llm_cross_task
        # Actually invoke the per-phase callables so we can verify they
        # forward the ``model=`` kwarg to ``self.llm.call``.
        if inp.llm_per_task is not None:
            inp.llm_per_task("sys", "u")
        if inp.llm_cross_task is not None:
            inp.llm_cross_task("sys", "u")
        return DistillOutput(per_task={}, cross_task=None)

    import kcsi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", fake_distill)

    run_distill(orch, generation=1, task_ids=["t1"])

    # Both per-phase callables should be present.
    assert captured["llm_per_task"] is not None
    assert captured["llm_cross_task"] is not None

    # ``self.llm.call`` should have been called at least twice with the
    # model overrides (once per phase). Find the model kwargs used.
    models_seen = [call.kwargs.get("model") for call in orch.llm.call.call_args_list if "model" in call.kwargs]
    assert "claude-per-task-model" in models_seen
    assert "claude-cross-task-model" in models_seen


def test_distill_no_overrides_still_builds_labeled_phase_llms(tmp_path, monkeypatch):
    """Even without model overrides, both per-phase LLMs should be populated
    so distill-time tokens get attributed to the correct phase label in the
    lifecycle accumulator."""
    from kcsi.distillation import DistillOutput

    orch = _make_orch(tmp_path)
    # Do NOT set distill_per_task_model / distill_cross_task_model.

    captured = {}

    def fake_distill(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
        captured["llm_per_task"] = inp.llm_per_task
        captured["llm_cross_task"] = inp.llm_cross_task
        captured["llm_default"] = inp.llm
        return DistillOutput(per_task={}, cross_task=None)

    import kcsi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", fake_distill)

    run_distill(orch, generation=1, task_ids=["t1"])

    # Both phase callables are always built so tokens are attributed to
    # ``distill_per_task`` / ``distill_cross_task`` in ``record_lifecycle``.
    assert callable(captured["llm_per_task"])
    assert callable(captured["llm_cross_task"])
    # The default ``inp.llm`` now points at the per-task callable (any of
    # the two labeled callables would be fine as fallback, but this is the
    # engine's current choice).
    assert captured["llm_default"] is captured["llm_per_task"]


def test_distill_adapter_threads_json_schema_to_structured_caller(tmp_path):
    """The distill ``LLMCallable`` adapter passes ``json_schema`` through to a
    structured-capable LLM caller and surfaces the provider's parsed dict."""
    from kcsi.distillation.types import DistillLLMResult

    orch = _make_orch(tmp_path)
    # The mock LLM advertises schema support and returns an LLMResponse with a
    # provider-validated ``parsed`` dict.
    orch.llm.supports_json_schema = True
    parsed_payload = {"transferable_insights": ["x"], "pitfalls": []}
    orch.llm.call.return_value = LLMResponse(
        text=json.dumps(parsed_payload),
        usage=TokenUsage(input_tokens=1, output_tokens=1),
        parsed=parsed_payload,
    )

    adapter = orch._make_distill_llm(generation=0, phase="distill_per_task")
    schema = {"name": "distill_bundle", "schema": {"type": "object"}}
    result = adapter("sys", "user", json_schema=schema)

    assert isinstance(result, DistillLLMResult)
    assert result.parsed == parsed_payload
    # The schema reached self.llm.call.
    assert orch.llm.call.call_args.kwargs["json_schema"] is schema


def test_distill_adapter_skips_schema_for_unsupported_caller(tmp_path):
    """If the LLM caller does not advertise ``supports_json_schema``, the
    adapter must NOT pass ``json_schema`` and returns plain text."""
    from kcsi.distillation.types import DistillLLMResult

    orch = _make_orch(tmp_path)
    # Default MagicMock attribute access would be truthy; force it False.
    orch.llm.supports_json_schema = False

    adapter = orch._make_distill_llm(generation=0, phase="distill_per_task")
    schema = {"name": "distill_bundle", "schema": {"type": "object"}}
    result = adapter("sys", "user", json_schema=schema)

    # Without provider support the adapter still asks for structured output
    # from the distill layer's perspective (returns a carrier), but it must
    # not have forwarded json_schema to the caller.
    assert isinstance(result, DistillLLMResult)
    assert "json_schema" not in orch.llm.call.call_args.kwargs


def test_distill_adapter_reraises_internal_typeerror(tmp_path):
    """A real TypeError from inside a schema-capable llm.call must surface —
    not be misread as 'schema rejected' and silently retried on a worse path
    (#982 #4). The static supports_json_schema guard replaces the runtime probe.
    """
    import pytest as _pytest

    orch = _make_orch(tmp_path)
    orch.llm.supports_json_schema = True
    orch.llm.call.side_effect = TypeError("internal bug: NoneType is not subscriptable")

    adapter = orch._make_distill_llm(generation=0, phase="distill_per_task")
    schema = {"name": "distill_bundle", "schema": {"type": "object"}}
    with _pytest.raises(TypeError, match="internal bug"):
        adapter("sys", "user", json_schema=schema)
    # Exactly one call: no silent schema-less fallback retry that masks the bug.
    assert orch.llm.call.call_count == 1


def test_distillation_body_has_no_engine_access():
    from kcsi.orchestrator import distillation_phase

    offenders = functions_referencing_engine(distillation_phase.__file__)
    assert offenders <= {"_collaborators"}, offenders


def test_distillation_collaborators_is_frozen():
    from dataclasses import FrozenInstanceError

    from kcsi.orchestrator.distillation_phase import DistillationCollaborators

    c = DistillationCollaborators(
        config=object(),
        knowledge=None,
        tasks_by_id={},
        record_phase_failure=lambda *a, **k: None,
        record_distill_result=lambda *a, **k: None,
        maybe_embed=lambda t: None,
        maybe_embed_batch=lambda ts: [None] * len(ts),
        make_distill_llm=lambda **k: None,
        is_holdout=lambda t: False,
    )
    try:
        c.knowledge = object()  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("DistillationCollaborators must be frozen")


def test_persists_per_task_cross_task_when_conditioning_on(tmp_path, monkeypatch):
    """Under target-conditioning, each attempted task's cross-task bundle is
    persisted under its own task_id (scope='cross_task'), not the sentinel."""
    from kcsi.distillation import DistillOutput
    from kcsi.distillation.types import CrossTaskBundle

    orch = _make_orch(tmp_path)
    # Default config has cross_task_distill_target_conditioning = True.
    assert orch.config.cross_task_distill_target_conditioning is True

    b1 = CrossTaskBundle(transferable_insights=["ct-t1"], pitfalls=[], checks=[], evidence_post_ids=[])
    b2 = CrossTaskBundle(transferable_insights=["ct-t2"], pitfalls=[], checks=[], evidence_post_ids=[])

    def fake_distill(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
        assert inp.cross_task_target_conditioning is True
        return DistillOutput(per_task={}, cross_task=None, cross_task_by_task={"t1": b1, "t2": b2})

    import kcsi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", fake_distill)

    result = EngineDistillationPhaseService(orch).run(DistillationPhaseInput(generation=1, task_ids=["t1", "t2"]))

    for tid, marker in (("t1", "ct-t1"), ("t2", "ct-t2")):
        got = orch._knowledge.load_distillation(generation=1, task_id=tid, scope="cross_task")
        assert got is not None and got.get("transferable_insights") == [marker]
    # No sentinel-keyed cross-task bundle when conditioning is on.
    assert orch._knowledge.load_distillation(generation=1, task_id=CROSS_TASK_SENTINEL, scope="cross_task") is None
    assert result.persisted_cross_task is True


def test_per_target_selection_config_threads_to_distiller(tmp_path, monkeypatch):
    """The GenerationConfig flag ``cross_task_distill_per_target_selection``
    reaches ``DistillInput.cross_task_per_target_selection``; default False."""
    import kcsi.distillation as dist_pkg
    from kcsi.distillation import DistillOutput

    captured: dict = {}

    def fake_distill(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
        captured["per_target"] = inp.cross_task_per_target_selection
        return DistillOutput(per_task={}, cross_task=None)

    monkeypatch.setattr(dist_pkg, "distill", fake_distill)

    # Default (False).
    orch = _make_orch(tmp_path)
    assert orch.config.cross_task_distill_per_target_selection is False
    run_distill(orch, generation=1, task_ids=["t1"])
    assert captured["per_target"] is False

    # Opt-in (True).
    orch.config.cross_task_distill_per_target_selection = True
    run_distill(orch, generation=1, task_ids=["t1"])
    assert captured["per_target"] is True


def test_holdout_not_leaked_into_learning_set_on_solved_set_exception(tmp_path, monkeypatch):
    """When ``solved_task_ids()`` raises (so ``unsolved_task_ids`` falls back to
    None), the ``source_ids`` fallback must COPY the hold-out-filtered task
    list, not alias it — otherwise the subsequent ``.extend()`` mutates
    ``task_ids`` in place, reintroducing hold-out ids into the learning set
    (``DistillInput.task_ids``) and violating the hold-out exclusion invariant.
    """
    from kcsi.distillation import DistillOutput

    orch = _make_orch(tmp_path)
    # Conditioning + drop_solved on (defaults) so the mutation-prone fallback
    # branch actually runs.
    assert orch.config.cross_task_distill_target_conditioning is True
    orch._is_holdout = lambda tid: tid == "hold"
    orch._tasks_by_id = {
        "t1": type("Task", (), {"prompt": "p1", "metadata": {}})(),
        "hold": type("Task", (), {"prompt": "ph", "metadata": {}})(),
    }

    # Force the solved-set query to raise -> unsolved_task_ids becomes None.
    def boom(*a, **k):
        raise RuntimeError("solved-set query failed")

    monkeypatch.setattr(orch._knowledge, "solved_task_ids", boom)

    captured: dict = {}

    def fake_distill(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
        captured["task_ids"] = list(inp.task_ids)
        captured["unsolved"] = unsolved_task_ids
        captured["cross_task_target_ids"] = list(inp.cross_task_target_ids or [])
        return DistillOutput(per_task={}, cross_task=None)

    import kcsi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", fake_distill)

    run_distill(orch, generation=1, task_ids=["t1", "hold"])

    # The exception path really was taken.
    assert captured["unsolved"] is None
    # Hold-out id must NOT have leaked into the learning set...
    assert "hold" not in captured["task_ids"]
    assert captured["task_ids"] == ["t1"]
    # ...even though it legitimately remains a cross-task conditioning target.
    assert "hold" in captured["cross_task_target_ids"]

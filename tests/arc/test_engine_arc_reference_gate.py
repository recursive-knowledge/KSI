"""The engine's ARC reference-payload gate consults TaskSourceSpec.arc_task_reference.

Pins the #715 registry wiring (issue #766 item 2): the hidden ARC reference
payload built in ``_enrich_seed_packages`` is gated on the registry capability
flag, not the raw ``task_source == "arc"`` string. Unit-level — no containers.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kcsi.models import TaskSpec
from kcsi.orchestrator.engine import GenerationalOrchestrator
from kcsi.orchestrator.enrichment_phase import ArcAnswerSanitizationError, EngineEnrichmentPhaseService
from kcsi.orchestrator.strategy import DefaultKnowledgeStrategy
from kcsi.tasks.registry import REGISTRY, TaskSourceSpec, register_task_source

TRAIN_PAIRS = [{"input": [[0]], "output": [[1]]}]
TEST_PAIRS = [{"input": [[2]], "output": [[3]]}]


def _run_enrich(task_source: str) -> tuple[SimpleNamespace, MagicMock]:
    """Call _enrich_seed_packages on a minimal stub engine; return (agent, memory_store)."""
    memory_store = MagicMock()
    memory_store.list_task_summaries.return_value = []
    memory_store.query_task_memory.return_value = []
    agent = SimpleNamespace(id="a1", seed_package={})
    stub = SimpleNamespace(
        config=SimpleNamespace(no_memory=False, experiment_name="exp"),
        _memory_store=memory_store,
        _knowledge=None,
        _best_scores={},
        _holdout_ids=frozenset(),
        agents=[agent],
        _improvement_strategy=DefaultKnowledgeStrategy(),
    )
    stub._is_holdout = GenerationalOrchestrator._is_holdout.__get__(stub)
    task = TaskSpec(
        id="t1",
        repo="",
        prompt="p",
        metadata={
            "task_source": task_source,
            "arc_train_pairs": TRAIN_PAIRS,
            "arc_eval_test_pairs": TEST_PAIRS,
            "arc_max_trials": 2,
        },
    )
    EngineEnrichmentPhaseService(stub).enrich(
        generation=1,
        assigned_map={"a1": ["t1"]},
        tasks=[task],
    )
    return agent, memory_store


def test_arc_source_still_triggers_reference_payload():
    """Behavior preservation + #923 H1: the canonical 'arc' source builds the
    hidden payload, but TEST OUTPUTS are stripped from everything mounted into
    the solver (snapshot + arc_task_refs). Train pairs retain outputs."""
    agent, memory_store = _run_enrich("arc")
    payloads = agent.seed_package["memory_snapshot"]["arc_payload_by_task"]
    assert payloads["t1"]["train"] == TRAIN_PAIRS
    # Test pairs are reduced to inputs only — no "output" key reaches the mount.
    assert payloads["t1"]["test"] == [{"input": [[2]]}]
    assert all("output" not in pair for pair in payloads["t1"]["test"])
    memory_store.upsert_arc_task_reference.assert_called_once()
    # The DB row (also mounted at /app/memory-db) is built from the same
    # hidden_payload, so it must be stripped too.
    upserted = memory_store.upsert_arc_task_reference.call_args.kwargs["payload"]
    assert upserted["test"] == [{"input": [[2]]}]
    assert upserted["train"] == TRAIN_PAIRS


def test_unflagged_source_does_not_trigger_gate():
    """A source without arc_task_reference (polyglot) skips the payload entirely."""
    agent, memory_store = _run_enrich("polyglot")
    assert agent.seed_package["memory_snapshot"]["arc_payload_by_task"] == {}
    memory_store.upsert_arc_task_reference.assert_not_called()


def test_synthetic_flagged_source_triggers_gate_via_registry():
    """A registered spec with arc_task_reference=True triggers the engine gate
    with no engine.py edit — proving the gate reads the registry flag."""
    register_task_source(TaskSourceSpec(name="synthetic_ref_bench", arc_task_reference=True))
    try:
        agent, memory_store = _run_enrich("synthetic_ref_bench")
    finally:
        REGISTRY.pop("synthetic_ref_bench", None)
    payloads = agent.seed_package["memory_snapshot"]["arc_payload_by_task"]
    assert payloads["t1"]["train"] == TRAIN_PAIRS
    memory_store.upsert_arc_task_reference.assert_called_once()


def test_resume_fallback_strips_test_outputs_from_db_row():
    """#923 H1: when the task lacks inline pairs, the engine falls back to
    get_arc_task_reference().  A pre-fix DB row may carry unstripped test
    outputs; the engine must sanitize them on load so the mounted snapshot
    never exposes the answer.
    """
    stale_ref = {
        "task_id": "t1",
        "train": [{"input": [[0]], "output": [[1]]}],
        "test": [{"input": [[2]], "output": [[3]]}],  # pre-fix: output present
        "max_trials": 2,
    }
    memory_store = MagicMock()
    memory_store.list_task_summaries.return_value = []
    memory_store.query_task_memory.return_value = []
    memory_store.get_arc_task_reference.return_value = stale_ref

    agent = SimpleNamespace(id="a1", seed_package={})
    stub = SimpleNamespace(
        config=SimpleNamespace(no_memory=False, experiment_name="exp"),
        _memory_store=memory_store,
        _knowledge=None,
        _best_scores={},
        _holdout_ids=frozenset(),
        agents=[agent],
        _improvement_strategy=DefaultKnowledgeStrategy(),
    )
    stub._is_holdout = GenerationalOrchestrator._is_holdout.__get__(stub)

    # Task has no inline pairs — forces the elif/fallback branch.
    task = TaskSpec(
        id="t1",
        repo="",
        prompt="p",
        metadata={"task_source": "arc"},
    )
    EngineEnrichmentPhaseService(stub).enrich(
        generation=1,
        assigned_map={"a1": ["t1"]},
        tasks=[task],
    )

    payloads = agent.seed_package["memory_snapshot"]["arc_payload_by_task"]
    assert "t1" in payloads, "fallback ref was not added to arc_payload_by_task"
    # Test outputs must be stripped on load.
    assert payloads["t1"]["test"] == [{"input": [[2]]}], (
        f"test output leaked from stale DB row: {payloads['t1']['test']!r}"
    )
    # Train outputs must be kept.
    assert payloads["t1"]["train"] == [{"input": [[0]], "output": [[1]]}]

    # #923 H1: the runtime sqlite is bind-mounted RO into the container, so
    # sanitizing only the in-memory snapshot is not enough — the on-disk
    # arc_task_refs row (read via `sqlite3 ... SELECT payload_json`) must be
    # overwritten with the stripped payload too. Assert the engine wrote the
    # answer-free row back to the DB.
    memory_store.upsert_arc_task_reference.assert_called_once()
    written = memory_store.upsert_arc_task_reference.call_args.kwargs["payload"]
    assert written["test"] == [{"input": [[2]]}], (
        f"on-disk DB row not sanitized — answer still on disk: {written['test']!r}"
    )
    assert all("output" not in pair for pair in written["test"])
    assert written["train"] == [{"input": [[0]], "output": [[1]]}]


def _run_resume_fallback(stale_ref, *, upsert_side_effect=None):
    """Drive the elif/get_arc_task_reference fallback in _enrich_seed_packages
    with a mocked store, optionally making the sanitizing upsert fail."""
    memory_store = MagicMock()
    memory_store.list_task_summaries.return_value = []
    memory_store.query_task_memory.return_value = []
    memory_store.get_arc_task_reference.return_value = stale_ref
    if upsert_side_effect is not None:
        memory_store.upsert_arc_task_reference.side_effect = upsert_side_effect

    agent = SimpleNamespace(id="a1", seed_package={})
    stub = SimpleNamespace(
        config=SimpleNamespace(no_memory=False, experiment_name="exp"),
        _memory_store=memory_store,
        _knowledge=None,
        _best_scores={},
        _holdout_ids=frozenset(),
        agents=[agent],
        _improvement_strategy=DefaultKnowledgeStrategy(),
    )
    stub._is_holdout = GenerationalOrchestrator._is_holdout.__get__(stub)
    task = TaskSpec(id="t1", repo="", prompt="p", metadata={"task_source": "arc"})
    EngineEnrichmentPhaseService(stub).enrich(generation=1, assigned_map={"a1": ["t1"]}, tasks=[task])
    return memory_store, agent


def test_resume_fallback_fails_closed_when_answer_bearing_strip_fails():
    """#966: if the on-disk row carries an answer and the sanitizing upsert
    fails, the engine must FAIL CLOSED (raise) rather than mount the leaking
    runtime DB — the DB is bind-mounted RO whole, so one unstripped row leaks."""
    answer_bearing = {
        "task_id": "t1",
        "train": [{"input": [[0]], "output": [[1]]}],
        "test": [{"input": [[2]], "output": [[3]]}],  # answer present on disk
        "max_trials": 2,
    }
    with pytest.raises(ArcAnswerSanitizationError, match="refusing to mount"):
        _run_resume_fallback(answer_bearing, upsert_side_effect=OSError("db locked"))


def test_resume_fallback_already_clean_row_only_warns_on_upsert_failure():
    """A row that no longer carries an answer (output already stripped) is not a
    leak risk, so a failed idempotent rewrite only warns — no fail-closed."""
    already_clean = {
        "task_id": "t1",
        "train": [{"input": [[0]], "output": [[1]]}],
        "test": [{"input": [[2]]}],  # no output: already sanitized
        "max_trials": 2,
    }
    # Does not raise even though the upsert fails.
    _, agent = _run_resume_fallback(already_clean, upsert_side_effect=OSError("db locked"))
    payloads = agent.seed_package["memory_snapshot"]["arc_payload_by_task"]
    assert payloads["t1"]["test"] == [{"input": [[2]]}]

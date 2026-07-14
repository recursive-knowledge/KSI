"""Characterization tests for the extracted seed-package enrichment phase.

The enrichment body moved out of ``engine.py`` into
``EngineEnrichmentPhaseService.enrich`` (behavior-preserving). These tests pin
the live engine behavior against a real KnowledgeStore plus the
``*Collaborators`` decoupling invariant.
"""

from ksi.models import AgentState, GenerationConfig, TaskSpec
from ksi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from ksi.orchestrator.enrichment_phase import (
    EnrichmentCollaborators,
    EnrichmentPhaseService,
)
from ksi.orchestrator.strategy import DefaultKnowledgeStrategy
from tests.orchestrator_phase_decoupling_guard import functions_referencing_engine


def _make_orch(tmp_path, mock_runtime, mock_evaluator, mock_llm) -> GenerationalOrchestrator:
    """Real orchestrator with an authoritative KnowledgeStore (memory enabled)."""
    db_path = str(tmp_path / "enrich_knowledge.sqlite")
    config = GenerationConfig(
        num_generations=2,
        num_agents=1,
        per_task_forum_rounds=0,
        knowledge_db_path=db_path,
        experiment_name="enrich_exp",
    )
    return GenerationalOrchestrator(
        config=config,
        runtime=mock_runtime(),
        evaluator=mock_evaluator(),
        llm=mock_llm(),
        persistence=NoopPersistence(),
    )


def test_engine_enrichment_phase_service_satisfies_protocol(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    orch = _make_orch(tmp_path, mock_runtime, mock_evaluator, mock_llm)
    try:
        assert isinstance(orch._enrichment_phase, EnrichmentPhaseService)
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_enrich_body_has_no_engine_access():
    from ksi.orchestrator import enrichment_phase

    offenders = functions_referencing_engine(enrichment_phase.__file__)
    assert offenders <= {"_collaborators"}, offenders


def test_enrichment_collaborators_is_frozen():
    from dataclasses import FrozenInstanceError

    import pytest

    c = EnrichmentCollaborators(
        config=None,
        knowledge=None,
        memory_store=None,
        agents=[],
        best_scores={},
        holdout_ids=frozenset(),
        is_holdout=lambda _t: False,
        external_per_task_bundles={},
        improvement_strategy=DefaultKnowledgeStrategy(),
    )
    with pytest.raises(FrozenInstanceError):
        c.agents = []  # type: ignore[misc]


def test_enrich_attaches_prior_attempts_and_best_score(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """A prior KnowledgeStore attempt surfaces as the agent's seed_package
    ``prior_attempts``/``best_score`` for the assigned task (live engine path)."""
    orch = _make_orch(tmp_path, mock_runtime, mock_evaluator, mock_llm)
    task_id = "django__django-12345"
    try:
        assert orch._knowledge is not None
        orch._knowledge.record_attempt(
            task_id=task_id,
            agent_id="agent-prev",
            generation=1,
            eval_results={"native_score": 0.5, "resolved": False},
            native_score=0.5,
            experiment="enrich_exp",
            model_output="prior attempt output",
            trace_condensed="prior attempt trace",
        )
        orch._best_scores[task_id] = 0.5

        agent = AgentState(id="agent-0", generation=2, seed_package={})
        orch.agents = [agent]
        task = TaskSpec(
            id=task_id,
            repo="django/django",
            prompt="fix it",
            metadata={"task_source": "swebench_pro", "repo": "django/django"},
        )

        orch._enrichment_phase.enrich(
            generation=2,
            assigned_map={"agent-0": [task_id]},
            tasks=[task],
        )

        pkg = agent.seed_package
        assert pkg["assigned_task_id"] == task_id
        assert pkg["best_score"] == 0.5
        prior = pkg["prior_attempts"]
        assert len(prior) == 1
        assert prior[0]["gen"] == 1
        assert prior[0]["agent_id"] == "agent-prev"
        assert prior[0]["eval_results"]["native_score"] == 0.5
        assert prior[0]["full_memory_trace_condensed"] == "prior attempt trace"
        snapshot = pkg["memory_snapshot"]
        assert snapshot["version"] == 2
        assert snapshot["task_id"] == task_id
        assert snapshot["generation"] == 2
        assert snapshot["query_records_by_task"][task_id] == prior
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_enrich_warns_when_external_bundles_loaded_but_none_attach(
    tmp_path, mock_runtime, mock_evaluator, mock_llm, caplog
):
    """External per-task bundles loaded but keyed to a task no agent is
    assigned to → 0 attach. The attach-count warning must fire so a mis-keyed
    donor/recipient id map is not silently a baseline-with-no-KT run (Bug 3)."""
    import logging

    orch = _make_orch(tmp_path, mock_runtime, mock_evaluator, mock_llm)
    task_id = "django__django-12345"
    try:
        # Donor bundle keyed to a DIFFERENT task id than the agent's assignment.
        orch._external_per_task_bundles = {"some__other-task": {"confirmed_constraints": ["x"]}}

        agent = AgentState(id="agent-0", generation=2, seed_package={})
        orch.agents = [agent]
        task = TaskSpec(
            id=task_id,
            repo="django/django",
            prompt="fix it",
            metadata={"task_source": "swebench_pro", "repo": "django/django"},
        )

        with caplog.at_level(logging.WARNING, logger="ksi.orchestrator.enrichment_phase"):
            orch._enrichment_phase.enrich(
                generation=2,
                assigned_map={"agent-0": [task_id]},
                tasks=[task],
            )

        assert "per_task_bundle" not in agent.seed_package
        assert any("NONE" in rec.getMessage() and rec.levelno == logging.WARNING for rec in caplog.records)
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_enrich_attaches_external_bundle_on_matching_task_id(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """When the donor key matches the assigned task id, the bundle attaches
    (the positive companion to the mis-keyed warning case)."""
    orch = _make_orch(tmp_path, mock_runtime, mock_evaluator, mock_llm)
    task_id = "django__django-12345"
    try:
        orch._external_per_task_bundles = {task_id: {"confirmed_constraints": ["x"]}}
        agent = AgentState(id="agent-0", generation=2, seed_package={})
        orch.agents = [agent]
        task = TaskSpec(
            id=task_id,
            repo="django/django",
            prompt="fix it",
            metadata={"task_source": "swebench_pro", "repo": "django/django"},
        )

        orch._enrichment_phase.enrich(
            generation=2,
            assigned_map={"agent-0": [task_id]},
            tasks=[task],
        )

        assert agent.seed_package.get("per_task_bundle") == {"confirmed_constraints": ["x"]}
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_related_summaries_prefers_statement_relevance_over_prefix(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """H5b: a content-unrelated same-language (same task_id prefix) polyglot
    sibling is NOT injected when a statement-relevant candidate exists, and the
    statement-relevant candidate (a *different* prefix, which prefix-routing
    would never surface) ranks first."""
    orch = _make_orch(tmp_path, mock_runtime, mock_evaluator, mock_llm)
    try:
        assert orch._knowledge is not None
        # Same prefix (cpp), content-unrelated: dungeons/dragons character gen.
        orch._knowledge.record_attempt(
            task_id="cpp__dnd-character",
            agent_id="agent-a",
            generation=1,
            eval_results={"native_score": 0.9, "resolved": False},
            native_score=0.9,
            experiment="enrich_exp",
            trace_condensed="roll dice for dungeons and dragons character ability scores strength",
        )
        # Different prefix (python), statement-relevant: complex-number arithmetic.
        orch._knowledge.record_attempt(
            task_id="python__complex-numbers",
            agent_id="agent-b",
            generation=1,
            eval_results={"native_score": 0.1, "resolved": False},
            native_score=0.1,
            experiment="enrich_exp",
            trace_condensed="complex number arithmetic multiply add conjugate real imaginary parts",
        )

        agent = AgentState(id="agent-0", generation=2, seed_package={})
        orch.agents = [agent]
        task = TaskSpec(
            id="cpp__complex-numbers",
            prompt="Implement complex number arithmetic: multiply add conjugate real imaginary parts",
            metadata={"task_source": "polyglot"},
        )

        orch._enrichment_phase.enrich(
            generation=2,
            assigned_map={"agent-0": ["cpp__complex-numbers"]},
            tasks=[task],
        )

        related = agent.seed_package["related_summaries"]
        related_ids = [r["task_id"] for r in related]
        assert "python__complex-numbers" in related_ids
        assert "cpp__dnd-character" not in related_ids
        assert related_ids[0] == "python__complex-numbers"
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_related_summaries_falls_back_to_prefix_when_no_relevant_candidate(
    tmp_path, mock_runtime, mock_evaluator, mock_llm
):
    """H5b: when NO candidate clears the similarity floor, the historical
    prefix/repo candidate set still fires (no regression to an empty list)."""
    orch = _make_orch(tmp_path, mock_runtime, mock_evaluator, mock_llm)
    try:
        assert orch._knowledge is not None
        # Same prefix (cpp) sibling; its content shares no topical tokens with
        # the current task statement, so statement-relevance drops it.
        orch._knowledge.record_attempt(
            task_id="cpp__dnd-character",
            agent_id="agent-a",
            generation=1,
            eval_results={"native_score": 0.7, "resolved": False},
            native_score=0.7,
            experiment="enrich_exp",
            trace_condensed="roll dice for dungeons dragons character ability scores strength",
        )

        agent = AgentState(id="agent-0", generation=2, seed_package={})
        orch.agents = [agent]
        task = TaskSpec(
            id="cpp__zebra-puzzle",
            prompt="Solve the zebra constraint logic puzzle houses nationalities pets beverages",
            metadata={"task_source": "polyglot"},
        )

        orch._enrichment_phase.enrich(
            generation=2,
            assigned_map={"agent-0": ["cpp__zebra-puzzle"]},
            tasks=[task],
        )

        related = agent.seed_package["related_summaries"]
        related_ids = [r["task_id"] for r in related]
        # Fallback to prefix routing keeps the same-language sibling.
        assert related_ids == ["cpp__dnd-character"]
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_related_summaries_arc_channel_no_longer_dead(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """H5b: an ARC-style task (no ``__`` prefix, no repo) now gets a non-empty
    ``related_summaries`` when a statement-relevant candidate exists — the
    channel was previously dead (empty candidate set) for ARC."""
    orch = _make_orch(tmp_path, mock_runtime, mock_evaluator, mock_llm)
    try:
        assert orch._knowledge is not None
        orch._knowledge.record_attempt(
            task_id="arc-related-abc",
            agent_id="agent-a",
            generation=1,
            eval_results={"native_score": 0.4, "resolved": False},
            native_score=0.4,
            experiment="enrich_exp",
            trace_condensed="mirror symmetry grid reflect pixels rows columns diagonal pattern",
        )

        agent = AgentState(id="agent-0", generation=2, seed_package={})
        orch.agents = [agent]
        task = TaskSpec(
            id="arc-task-xyz",
            prompt="Reflect the grid pattern across the diagonal symmetry mirror pixels rows columns",
            metadata={"task_source": "arc"},
        )

        orch._enrichment_phase.enrich(
            generation=2,
            assigned_map={"agent-0": ["arc-task-xyz"]},
            tasks=[task],
        )

        related = agent.seed_package["related_summaries"]
        related_ids = [r["task_id"] for r in related]
        assert related_ids == ["arc-related-abc"]
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_enrich_redacts_hidden_output_in_snapshot_summaries(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """A sibling task summary whose ``trace_condensed``/insights carry stale
    hidden verifier fragments must arrive redacted in the mounted snapshot the
    task agent receives (``search_rows`` / ``related_summaries``) — #1111.

    Without the fix, ``list_task_summaries`` copies raw ``trace_condensed`` into
    ``approach`` and dumps raw insights into ``lessons``, both of which reach the
    solver container unredacted.
    """
    import json

    orch = _make_orch(tmp_path, mock_runtime, mock_evaluator, mock_llm)
    assigned_id = "django__django-12345"
    sibling_id = "django__django-99999"  # same prefix -> related_summaries + search_rows
    try:
        assert orch._knowledge is not None
        # Sibling attempt with poisoned derived text (approach + lessons).
        orch._knowledge.record_attempt(
            task_id=sibling_id,
            agent_id="agent-sibling",
            generation=1,
            eval_results={"native_score": 0.0, "resolved": False},
            native_score=0.0,
            experiment="enrich_exp",
            model_output="sibling output",
            trace_condensed=("reward=0.0 agent_exit=0; failure_signature=secretcanarytoken_leak; tool_count=3"),
            insights=[
                "clean lesson about the parser",
                "verifier_stdout_tail=Expected foo; got secretcanarytoken_insight; tool_count=1",
            ],
            repo="django/django",
        )
        # Assigned task's own prior attempt (clean).
        orch._knowledge.record_attempt(
            task_id=assigned_id,
            agent_id="agent-prev",
            generation=1,
            eval_results={"native_score": 0.5, "resolved": False},
            native_score=0.5,
            experiment="enrich_exp",
            model_output="prior attempt output",
            trace_condensed="prior attempt trace",
            repo="django/django",
        )

        agent = AgentState(id="agent-0", generation=2, seed_package={})
        orch.agents = [agent]
        task = TaskSpec(
            id=assigned_id,
            repo="django/django",
            prompt="fix it",
            metadata={"task_source": "swebench_pro"},
        )

        orch._enrichment_phase.enrich(
            generation=2,
            assigned_map={"agent-0": [assigned_id]},
            tasks=[task],
        )

        snapshot = agent.seed_package["memory_snapshot"]
        search_rows = snapshot["search_rows"]
        sib_rows = [r for r in search_rows if r.get("task_id") == sibling_id]
        assert sib_rows, "sibling summary should appear in search_rows"
        sib = sib_rows[0]
        # approach (from trace_condensed) is scrubbed but the safe signals survive.
        assert "secretcanarytoken" not in sib["approach"]
        assert "failure_signature" not in sib["approach"]
        assert "reward=0.0" in sib["approach"]
        assert "tool_count=3" in sib["approach"]
        # lessons stays valid JSON; the clean insight survives, the poisoned one is scrubbed.
        lessons = json.loads(sib["lessons"])
        assert "clean lesson about the parser" in lessons
        assert "secretcanarytoken" not in sib["lessons"]
        assert "verifier_stdout_tail" not in sib["lessons"]

        # The same redacted row object flows into related_summaries (prefix match).
        related = agent.seed_package["related_summaries"]
        rel_rows = [r for r in related if r.get("task_id") == sibling_id]
        assert rel_rows, "sibling summary should be prefix-matched into related_summaries"
        assert "secretcanarytoken" not in rel_rows[0]["approach"]
        assert "secretcanarytoken" not in rel_rows[0]["lessons"]
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_enrich_raises_on_multiple_tasks_per_agent(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """Risk 1: ``enrich`` reads ``task_ids[0]`` per agent. If the claim phase
    ever assigns >1 task to one agent, silently enriching only the first would
    hide the bug — the guard must fail loudly instead."""
    import pytest

    orch = _make_orch(tmp_path, mock_runtime, mock_evaluator, mock_llm)
    task_a = "django__django-1"
    task_b = "django__django-2"
    try:
        agent = AgentState(id="agent-0", generation=2, seed_package={})
        orch.agents = [agent]
        tasks = [
            TaskSpec(id=task_a, repo="django/django", prompt="a", metadata={"task_source": "swebench_pro"}),
            TaskSpec(id=task_b, repo="django/django", prompt="b", metadata={"task_source": "swebench_pro"}),
        ]
        with pytest.raises(RuntimeError, match="1-task-per-agent"):
            orch._enrichment_phase.enrich(
                generation=2,
                assigned_map={"agent-0": [task_a, task_b]},
                tasks=tasks,
            )
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()


def test_enrich_collaborators_read_live_engine_state(tmp_path, mock_runtime, mock_evaluator, mock_llm):
    """The per-call factory reads live engine state (agents reassigned per gen)."""
    orch = _make_orch(tmp_path, mock_runtime, mock_evaluator, mock_llm)
    try:
        new_agents = [AgentState(id="agent-live", generation=2, seed_package={})]
        orch.agents = new_agents
        collab = orch._enrichment_phase._collaborators()
        assert collab.agents is new_agents
        assert collab.knowledge is orch._knowledge
        assert collab.best_scores is orch._best_scores
    finally:
        if orch._knowledge is not None:
            orch._knowledge.close()
        if orch._memory_store is not None:
            orch._memory_store.close()

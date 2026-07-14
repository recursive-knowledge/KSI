"""Tests for GenerationalOrchestrator._enrich_seed_packages()."""

from __future__ import annotations

from unittest.mock import MagicMock

from kcsi.models import AgentState, TaskSpec
from kcsi.orchestrator.engine import GenerationalOrchestrator
from kcsi.orchestrator.enrichment_phase import EngineEnrichmentPhaseService, _knowledge_attempts_to_seed_records
from kcsi.orchestrator.strategy import DefaultKnowledgeStrategy, RawAttemptsStrategy
from kcsi.runtime.seeding import format_query_records_md, seed_package_to_memory_md

# ---------------------------------------------------------------------------
# Helpers — build a lightweight engine stub that has the attributes
# _enrich_seed_packages reads, without spinning up a real orchestrator.
# ---------------------------------------------------------------------------


def _make_engine_stub(
    *,
    agents: list[AgentState] | None = None,
    memory_store: MagicMock | None = MagicMock(),
    docs_store: MagicMock | None = None,  # deprecated, ignored
    knowledge: MagicMock | None = None,
    best_scores: dict[str, float] | None = None,
    experiment_name: str = "test_exp",
    improvement_strategy: object | None = None,
) -> MagicMock:
    """Return a MagicMock that looks like a GenerationalOrchestrator instance."""
    stub = MagicMock(spec=[])  # empty spec so we can set arbitrary attrs
    stub.agents = agents or []
    stub._memory_store = memory_store
    stub._knowledge = knowledge
    stub._best_scores = best_scores or {}
    # Hold-out transfer probe state (feature off): bind the real helper so
    # the stub behaves like an orchestrator with no hold-out tasks.
    stub._holdout_ids = frozenset()
    stub._is_holdout = GenerationalOrchestrator._is_holdout.__get__(stub)
    stub.config = MagicMock()
    stub.config.experiment_name = experiment_name
    stub.config.no_memory = False
    # Default matches today's behavior (should_enrich() -> True) so every
    # existing call site keeps working unchanged (#987).
    stub._improvement_strategy = improvement_strategy or DefaultKnowledgeStrategy()
    return stub


def _make_task(task_id: str, *, task_source: str = "swebench_pro", repo: str = "django/django") -> TaskSpec:
    return TaskSpec(
        id=task_id,
        repo=repo,
        prompt=f"Fix {task_id}",
        metadata={"task_source": task_source, "repo": repo},
    )


def _make_agent(agent_id: str, **kwargs) -> AgentState:
    return AgentState(id=agent_id, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEnrichSeedPackages:
    """Unit tests for _enrich_seed_packages."""

    def test_attaches_prior_attempts(self):
        """Memory store records are attached as prior_attempts in seed_package."""
        agent = _make_agent("agent-0")
        task = _make_task("django__django-12345")
        records = [
            {"generation": 1, "agent_id": "agent-0", "task_id": "django__django-12345", "eval_results_json": "{}"},
            {"generation": 2, "agent_id": "agent-1", "task_id": "django__django-12345", "eval_results_json": "{}"},
        ]

        mem_store = MagicMock()
        mem_store.query_task_memory.return_value = records
        mem_store.list_task_summaries.return_value = []

        stub = _make_engine_stub(
            agents=[agent],
            memory_store=mem_store,
        )

        EngineEnrichmentPhaseService(stub).enrich(
            generation=2,
            assigned_map={"agent-0": ["django__django-12345"]},
            tasks=[task],
        )

        assert agent.seed_package["prior_attempts"] == records
        mem_store.query_task_memory.assert_called_once_with(
            task_id="django__django-12345",
            experiment="test_exp",
            limit=8,
        )

    def test_attaches_related_summaries(self):
        """Prefix-based filtering picks related summaries from the same repo prefix."""
        agent = _make_agent("agent-0")
        task = _make_task("django__django-12345", repo="django/django")
        summaries = [
            {"task_id": "django__django-99999", "repo": "django/django"},
            {"task_id": "django__django-11111", "repo": "django/django"},
            {"task_id": "sphinx__sphinx-00001", "repo": "sphinx-doc/sphinx"},
            {"task_id": "django__django-12345", "repo": "django/django"},  # self — should be excluded
        ]

        mem_store = MagicMock()
        mem_store.list_task_summaries.return_value = summaries

        stub = _make_engine_stub(
            agents=[agent],
            memory_store=mem_store,
        )

        EngineEnrichmentPhaseService(stub).enrich(
            generation=2,
            assigned_map={"agent-0": ["django__django-12345"]},
            tasks=[task],
        )

        related = agent.seed_package["related_summaries"]
        assert len(related) == 2
        # Self should be excluded
        assert all(s["task_id"] != "django__django-12345" for s in related)
        # Sphinx should be excluded (different prefix)
        assert all(s["task_id"] != "sphinx__sphinx-00001" for s in related)

    def test_related_summaries_rank_solved_sibling_over_recency(self):
        """#1040: a solved sibling with a good native_score must not be pushed
        out of the top-5 candidate window by unsolved siblings that merely
        have a higher (more recent) row id.

        ``list_task_summaries`` returns rows ordered by recency (latest
        attempt id DESC). Under ``--drop-solved`` a solved task's id freezes
        at the generation it was solved, while an unsolved task's id keeps
        climbing every generation — so unsolved siblings systematically
        outrank a solved sibling in a pure recency ordering, even though the
        solved sibling's approach is the useful one to surface.
        """
        agent = _make_agent("agent-0")
        task = _make_task("django__django-12345", repo="django/django")
        # Recency order (as returned by the SQL ORDER BY id DESC): five
        # unsolved siblings (no score) followed by one SOLVED sibling with a
        # perfect native_score that simply attempted earlier and stopped.
        summaries = [
            {"task_id": f"django__django-9000{i}", "repo": "django/django", "score": None, "outcome": "unresolved"}
            for i in range(1, 6)
        ] + [
            {"task_id": "django__django-90006", "repo": "django/django", "score": 1.0, "outcome": "resolved"},
        ]

        mem_store = MagicMock()
        mem_store.list_task_summaries.return_value = summaries

        stub = _make_engine_stub(
            agents=[agent],
            memory_store=mem_store,
        )

        EngineEnrichmentPhaseService(stub).enrich(
            generation=2,
            assigned_map={"agent-0": ["django__django-12345"]},
            tasks=[task],
        )

        related = agent.seed_package["related_summaries"]
        assert len(related) == 5
        related_ids = {s["task_id"] for s in related}
        assert "django__django-90006" in related_ids, (
            "the solved sibling (native_score=1.0) was dropped from the top-5 "
            f"candidate window in favor of purely-more-recent unsolved siblings: {related_ids}"
        )

    def test_related_summaries_single_token_hit_does_not_block_prefix_fallback(self):
        """A weak one-word statement overlap should not count as relevance.

        Before the overlap floor, an unrelated row sharing only "bowling" cleared
        the low Jaccard floor and suppressed the historical prefix/repo fallback,
        hiding the solved same-prefix sibling.
        """
        agent = _make_agent("agent-0")
        task = TaskSpec(
            id="python__target",
            repo="python/exercises",
            prompt="Repair bowling frame scoring with strikes and spares.",
            metadata={"task_source": "polyglot"},
        )
        summaries = [
            {
                "task_id": "ruby__weak",
                "repo": "ruby/exercises",
                "approach": "Queue parsing bug in bowling parser",
                "lessons": "",
                "score": 1.0,
            },
            {
                "task_id": "python__fallback",
                "repo": "python/exercises",
                "approach": "",
                "lessons": "",
                "score": 1.0,
            },
        ]

        mem_store = MagicMock()
        mem_store.list_task_summaries.return_value = summaries

        stub = _make_engine_stub(
            agents=[agent],
            memory_store=mem_store,
        )

        EngineEnrichmentPhaseService(stub).enrich(
            generation=2,
            assigned_map={"agent-0": ["python__target"]},
            tasks=[task],
        )

        related_ids = [s["task_id"] for s in agent.seed_package["related_summaries"]]
        assert related_ids == ["python__fallback"]

    def test_related_summaries_use_taskspec_repo_not_metadata(self):
        """Repo-channel filtering must read ``task.repo`` (the ``TaskSpec``
        field), not ``task.metadata["repo"]``.

        Regression test for issue #1039 (knowledge-retrieval.md #1): no real
        task loader (ARC, SWE-bench Pro, polyglot, TB2) ever puts a "repo"
        key into ``TaskSpec.metadata`` — only ``TaskSpec.repo`` carries it.
        Task ids are chosen with no shared ``__`` prefix so only the
        repo channel (not the prefix channel) can produce a match, isolating
        the behavior under test.
        """
        agent = _make_agent("agent-0")
        task = TaskSpec(
            id="task-alpha-001",
            repo="astropy/astropy",
            prompt="Fix task-alpha-001",
            metadata={"task_source": "swebench_pro"},  # no "repo" key, matches real loaders
        )
        summaries = [
            {"task_id": "task-beta-002", "repo": "astropy/astropy"},  # same repo, unrelated prefix
            {"task_id": "task-gamma-003", "repo": "psf/requests"},  # different repo
        ]

        mem_store = MagicMock()
        mem_store.list_task_summaries.return_value = summaries

        stub = _make_engine_stub(
            agents=[agent],
            memory_store=mem_store,
        )

        EngineEnrichmentPhaseService(stub).enrich(
            generation=2,
            assigned_map={"agent-0": ["task-alpha-001"]},
            tasks=[task],
        )

        related_ids = {s["task_id"] for s in agent.seed_package["related_summaries"]}
        assert "task-beta-002" in related_ids
        assert "task-gamma-003" not in related_ids

    def test_attaches_best_score(self):
        """best_score is taken from _best_scores dict."""
        agent = _make_agent("agent-0")
        task = _make_task("django__django-12345")

        mem_store = MagicMock()
        mem_store.list_task_summaries.return_value = []

        stub = _make_engine_stub(
            agents=[agent],
            memory_store=mem_store,
            best_scores={"django__django-12345": 0.75},
        )

        EngineEnrichmentPhaseService(stub).enrich(
            generation=3,
            assigned_map={"agent-0": ["django__django-12345"]},
            tasks=[task],
        )

        assert agent.seed_package["best_score"] == 0.75

    def test_no_stores_returns_immediately(self):
        """Both stores None => seed_package unchanged."""
        agent = _make_agent("agent-0")
        agent.seed_package = {"existing": True}
        task = _make_task("task-0")

        stub = _make_engine_stub(
            agents=[agent],
            memory_store=None,
            knowledge=None,
        )

        EngineEnrichmentPhaseService(stub).enrich(
            generation=1,
            assigned_map={"agent-0": ["task-0"]},
            tasks=[task],
        )

        assert agent.seed_package == {"existing": True}

    def test_memory_store_only(self):
        """Only memory store present — partial enrichment, no crash."""
        agent = _make_agent("agent-0")
        task = _make_task("django__django-12345")
        records = [{"generation": 1, "task_id": "django__django-12345"}]

        mem_store = MagicMock()
        mem_store.query_task_memory.return_value = records

        stub = _make_engine_stub(
            agents=[agent],
            memory_store=mem_store,
        )

        EngineEnrichmentPhaseService(stub).enrich(
            generation=2,
            assigned_map={"agent-0": ["django__django-12345"]},
            tasks=[task],
        )

        assert agent.seed_package["prior_attempts"] == records
        # related_summaries depends on list_task_summaries which may not be configured
        assert agent.seed_package.get("related_summaries") is not None

    def test_knowledge_store_only(self):
        """Only _knowledge set: prior attempts come from authoritative KnowledgeStore."""
        agent = _make_agent("agent-0")
        task = _make_task("django__django-12345")
        knowledge = MagicMock()
        page = {
            "task_id": "django__django-12345",
            "attempts": [
                {
                    "gen": 1,
                    "agent_id": "agent-0",
                    "score": 0.5,
                    "content": {
                        "eval_results": {"native_score": 0.5, "resolved": False},
                        "trace_condensed": "first attempt",
                        "insights": ["check parser branch"],
                    },
                }
            ],
        }
        # The engine batches the per-task knowledge read via query_tasks.
        knowledge.query_tasks.return_value = {"django__django-12345": page}

        stub = _make_engine_stub(
            agents=[agent],
            memory_store=None,
            knowledge=knowledge,
        )

        EngineEnrichmentPhaseService(stub).enrich(
            generation=2,
            assigned_map={"agent-0": ["django__django-12345"]},
            tasks=[task],
        )

        assert agent.seed_package["prior_attempts"][0]["gen"] == 1
        assert agent.seed_package["prior_attempts"][0]["eval_results"]["native_score"] == 0.5
        assert agent.seed_package["prior_attempts"][0]["full_memory_trace_condensed"] == "first attempt"
        assert agent.seed_package["related_summaries"] == []
        knowledge.query_tasks.assert_called_once_with(
            ["django__django-12345"],
            entry_types=["attempt", "insight"],
            experiment="test_exp",
            limit=8,
        )
        knowledge.query_task.assert_not_called()

    def test_knowledge_store_insight_rows_seed_memory_md(self):
        """Standalone KnowledgeStore R0 insight rows must reach seed memory."""
        agent = _make_agent("agent-0")
        task = _make_task("django__django-12345")
        insight_text = "VERBATIM_INSIGHT_TAIL " + ("context " * 80)
        knowledge = MagicMock()
        knowledge.list_task_summaries.return_value = []
        page = {
            "task_id": "django__django-12345",
            "attempts": [
                {
                    "gen": 1,
                    "agent_id": "agent-0",
                    "score": 0.0,
                    "content": {
                        "eval_results": {"native_score": 0.0, "resolved": False},
                        "trace_condensed": "first attempt",
                        "insights": [],
                    },
                }
            ],
            "insights": [
                {
                    "gen": 1,
                    "agent_id": "agent-0",
                    "text": insight_text,
                    "scope": "task",
                }
            ],
        }
        knowledge.query_tasks.return_value = {"django__django-12345": page}

        stub = _make_engine_stub(
            agents=[agent],
            memory_store=None,
            knowledge=knowledge,
        )

        EngineEnrichmentPhaseService(stub).enrich(
            generation=2,
            assigned_map={"agent-0": ["django__django-12345"]},
            tasks=[task],
        )

        prior_attempts = agent.seed_package["prior_attempts"]
        assert prior_attempts[0]["task_specific_insights"] == [insight_text.strip()]
        md = seed_package_to_memory_md(agent.seed_package, current_task_id="django__django-12345")
        assert "VERBATIM_INSIGHT_TAIL" in md
        assert insight_text[-80:].strip() in md

    def test_knowledge_store_seed_records_do_not_mutate_query_page(self):
        """Merging standalone insight rows must not mutate KnowledgeStore query results."""
        page = {
            "task_id": "django__django-12345",
            "attempts": [
                {
                    "gen": 1,
                    "agent_id": "agent-0",
                    "score": 0.0,
                    "content": {
                        "eval_results": {"native_score": 0.0},
                        "trace_condensed": "first attempt",
                        "insights": [],
                    },
                }
            ],
            "insights": [{"gen": 1, "agent_id": "agent-0", "text": "standalone insight"}],
        }

        records = _knowledge_attempts_to_seed_records(page)

        assert records[0]["task_specific_insights"] == ["standalone insight"]
        assert page["attempts"][0]["content"]["insights"] == []

    def test_knowledge_store_seed_records_keep_newest_first_with_standalone_insights(self):
        """Standalone insight-only records should preserve newest-first seed ordering."""
        page = {
            "task_id": "django__django-12345",
            "attempts": [
                {
                    "gen": 1,
                    "agent_id": "agent-0",
                    "score": 0.0,
                    "content": {
                        "eval_results": {"native_score": 0.0},
                        "trace_condensed": "first attempt",
                        "insights": [],
                    },
                }
            ],
            "insights": [
                {"gen": 2, "agent_id": "agent-1", "text": "newer standalone insight"},
                {"gen": 3, "agent_id": "agent-2", "text": "newest standalone insight"},
            ],
        }

        records = _knowledge_attempts_to_seed_records(page)

        assert [(record["gen"], record["agent_id"]) for record in records] == [
            (3, "agent-2"),
            (2, "agent-1"),
            (1, "agent-0"),
        ]

    def test_new_standalone_insight_survives_render_cap(self):
        """Newest standalone insights should not be dropped behind older attempt insights."""
        page = {
            "task_id": "django__django-12345",
            "attempts": [
                {
                    "gen": gen,
                    "agent_id": f"agent-{gen}",
                    "score": 0.0,
                    "content": {
                        "eval_results": {"native_score": 0.0},
                        "trace_condensed": f"attempt {gen}",
                        "insights": [f"old insight {gen}a", f"old insight {gen}b"],
                    },
                }
                for gen in range(1, 9)
            ],
            "insights": [
                {
                    "gen": 9,
                    "agent_id": "agent-9",
                    "text": "NEW_STANDALONE insight should be first",
                }
            ],
        }

        records = _knowledge_attempts_to_seed_records(page)
        md = format_query_records_md(records, task_id="django__django-12345")

        assert "NEW_STANDALONE insight should be first" in md

    def test_fetches_summaries_once(self):
        """memory_store.list_task_summaries called exactly once even with 3 agents."""
        agents = [_make_agent(f"agent-{i}") for i in range(3)]
        tasks = [_make_task(f"task-{i}") for i in range(3)]

        mem_store = MagicMock()
        mem_store.list_task_summaries.return_value = []

        stub = _make_engine_stub(
            agents=agents,
            memory_store=mem_store,
        )

        assigned_map = {f"agent-{i}": [f"task-{i}"] for i in range(3)}
        EngineEnrichmentPhaseService(stub).enrich(
            generation=1,
            assigned_map=assigned_map,
            tasks=tasks,
        )

        mem_store.list_task_summaries.assert_called_once()

    def test_skips_agents_not_in_assigned_map(self):
        """Unassigned agent's seed_package remains unchanged."""
        assigned_agent = _make_agent("agent-0")
        unassigned_agent = _make_agent("agent-1")
        unassigned_agent.seed_package = {"untouched": True}
        task = _make_task("task-0")

        mem_store = MagicMock()
        mem_store.list_task_summaries.return_value = []

        stub = _make_engine_stub(
            agents=[assigned_agent, unassigned_agent],
            memory_store=mem_store,
        )

        EngineEnrichmentPhaseService(stub).enrich(
            generation=1,
            assigned_map={"agent-0": ["task-0"]},
            tasks=[task],
        )

        assert unassigned_agent.seed_package == {"untouched": True}
        assert "assigned_task_id" in assigned_agent.seed_package

    def test_builds_memory_snapshot(self):
        """Verify snapshot dict has the correct structure and fields."""
        agent = _make_agent("agent-0")
        task = _make_task("django__django-12345", task_source="swebench_pro", repo="django/django")
        records = [{"generation": 1, "task_id": "django__django-12345"}]
        summaries = [
            {"task_id": "django__django-99999", "repo": "django/django"},
        ]

        mem_store = MagicMock()
        mem_store.query_task_memory.return_value = records
        mem_store.list_task_summaries.return_value = summaries

        stub = _make_engine_stub(
            agents=[agent],
            memory_store=mem_store,
            experiment_name="exp1",
        )

        EngineEnrichmentPhaseService(stub).enrich(
            generation=3,
            assigned_map={"agent-0": ["django__django-12345"]},
            tasks=[task],
        )

        snapshot = agent.seed_package["memory_snapshot"]
        assert snapshot["version"] == 2
        assert snapshot["experiment"] == "exp1"
        assert snapshot["generation"] == 3
        assert snapshot["task_source"] == "swebench_pro"
        assert snapshot["task_id"] == "django__django-12345"
        assert snapshot["relevant_task_ids"] == ["django__django-12345"]
        assert "django__django-12345" in snapshot["query_records_by_task"]
        assert isinstance(snapshot["search_rows"], list)
        assert isinstance(snapshot["related_summaries"], list)
        assert isinstance(snapshot["arc_payload_by_task"], dict)

    def test_arc_upsert_and_get(self):
        """ARC task snapshots come from task metadata and audit into runtime DB."""
        agent = _make_agent("agent-0")
        task = _make_task("arc-task-001", task_source="arc", repo="")
        task.metadata["arc_train_pairs"] = [{"input": [[1]], "output": [[2]]}]
        task.metadata["arc_eval_test_pairs"] = [{"input": [[3]], "output": [[4]]}]
        task.metadata["arc_max_trials"] = 2

        mem_store = MagicMock()
        mem_store.list_task_summaries.return_value = []

        stub = _make_engine_stub(
            agents=[agent],
            memory_store=mem_store,
        )

        EngineEnrichmentPhaseService(stub).enrich(
            generation=1,
            assigned_map={"agent-0": ["arc-task-001"]},
            tasks=[task],
        )

        expected_payload = {
            "task_id": "arc-task-001",
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}],
            "max_trials": 2,
        }
        mem_store.upsert_arc_task_reference.assert_called_once_with(
            task_id="arc-task-001",
            payload=expected_payload,
            experiment="test_exp",
        )
        mem_store.get_arc_task_reference.assert_not_called()

        snapshot = agent.seed_package["memory_snapshot"]
        assert snapshot["arc_payload_by_task"]["arc-task-001"] == expected_payload

    def test_arc_snapshot_without_runtime_db(self):
        """ARC remains runnable when the optional runtime DB sidecar is disabled."""
        agent = _make_agent("agent-0")
        task = _make_task("arc-task-no-runtime", task_source="arc", repo="")
        task.metadata["arc_train_pairs"] = [{"input": [[1]], "output": [[2]]}]
        task.metadata["arc_eval_test_pairs"] = [{"input": [[3]], "output": [[4]]}]
        task.metadata["arc_max_trials"] = 2
        knowledge = MagicMock()
        knowledge.query_tasks.return_value = {task.id: {"task_id": task.id, "attempts": []}}

        stub = _make_engine_stub(
            agents=[agent],
            memory_store=None,
            knowledge=knowledge,
        )

        EngineEnrichmentPhaseService(stub).enrich(
            generation=1,
            assigned_map={"agent-0": [task.id]},
            tasks=[task],
        )

        snapshot = agent.seed_package["memory_snapshot"]
        assert snapshot["arc_payload_by_task"][task.id] == {
            "task_id": task.id,
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}],
            "max_trials": 2,
        }

    def test_raw_attempts_strategy_skips_enrichment_entirely(self):
        """RawAttemptsStrategy is a true knowledge-off ablation (#987): even
        with memory_store and knowledge present, no enrichment is applied."""
        stub = _make_engine_stub(
            memory_store=MagicMock(),
            knowledge=MagicMock(),
            improvement_strategy=RawAttemptsStrategy(),
        )
        agent = _make_agent("a1", seed_package={})
        stub.agents = [agent]

        EngineEnrichmentPhaseService(stub).enrich(
            generation=2,
            assigned_map={"a1": ["t1"]},
            tasks=[_make_task("t1")],
        )

        assert "prior_attempts" not in agent.seed_package
        assert "best_score" not in agent.seed_package

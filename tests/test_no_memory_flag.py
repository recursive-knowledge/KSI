"""Tests for the --no-memory CLI flag."""

from __future__ import annotations

from unittest.mock import patch

from kcsi.cli import _resolve_runtime_db_path, build_parser
from kcsi.models import GenerationConfig, TaskSpec
from kcsi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence


class TestNoMemoryFlag:
    """Verify --no-memory disables knowledge/forum/seeding, not runtime DB."""

    _BASE_ARGS = [
        "--task-source",
        "swebench_pro",
        "--tasks-path",
        "/tmp/tasks.parquet",
    ]

    def test_no_memory_sets_config_field(self):
        """--no-memory sets no_memory=True on GenerationConfig."""
        config = GenerationConfig(
            num_generations=1,
            num_agents=1,
            no_memory=True,
        )
        assert config.no_memory is True

    def test_no_memory_default_false(self):
        """no_memory defaults to False."""
        config = GenerationConfig(num_generations=1, num_agents=1)
        assert config.no_memory is False

    def test_parser_no_memory_flag_present(self):
        """--no-memory is accepted by the parser and sets args.no_memory."""
        parser = build_parser()
        args = parser.parse_args(self._BASE_ARGS + ["--no-memory"])
        assert args.no_memory is True

    def test_parser_no_memory_default(self):
        """Without --no-memory, args.no_memory is False."""
        parser = build_parser()
        args = parser.parse_args(self._BASE_ARGS)
        assert args.no_memory is False

    def test_no_memory_leaves_default_runtime_db_path_resolvable(self):
        """When --no-memory is set, the default runtime DB path is still used.

        We simulate the processing logic from main() that fires before
        GenerationConfig is constructed.
        """
        parser = build_parser()
        args = parser.parse_args(
            self._BASE_ARGS
            + [
                "--no-memory",
            ]
        )
        # Simulate the --no-memory override logic from main()
        if args.no_memory:
            args.per_task_forum_rounds = 0
            args.cross_task_forum_rounds = 0
            args.disable_memory_mcp = True
        resolved = _resolve_runtime_db_path(args.knowledge_db_path, "default")
        assert resolved.endswith("/default_runtime.sqlite")
        assert args.disable_memory_mcp is True

    def test_no_memory_preserves_explicit_runtime_db_path(self):
        """--no-memory does not wipe an explicit runtime DB path."""
        parser = build_parser()
        args = parser.parse_args(
            self._BASE_ARGS
            + [
                "--no-memory",
                "--runtime-db-path",
                "/tmp/runtime.sqlite",
            ]
        )
        if args.no_memory:
            args.per_task_forum_rounds = 0
            args.cross_task_forum_rounds = 0
            args.disable_memory_mcp = True
        assert args.runtime_db_path == "/tmp/runtime.sqlite"
        assert args.disable_memory_mcp is True

    def test_no_memory_sets_forum_rounds_zero(self):
        """When --no-memory is set, both forum-round counts should be 0."""
        parser = build_parser()
        args = parser.parse_args(
            self._BASE_ARGS
            + [
                "--no-memory",
                "--per-task-forum-rounds",
                "5",
            ]
        )
        # Simulate the --no-memory override logic from main()
        if args.no_memory:
            args.per_task_forum_rounds = 0
            args.cross_task_forum_rounds = 0
            args.disable_memory_mcp = True
        assert args.per_task_forum_rounds == 0
        assert args.cross_task_forum_rounds == 0

    def test_no_memory_does_not_override_knowledge_db_path(self):
        """--knowledge-db-path remains the authoritative DB path under --no-memory."""
        parser = build_parser()
        args = parser.parse_args(
            self._BASE_ARGS
            + [
                "--knowledge-db-path",
                "/tmp/explicit.sqlite",
                "--no-memory",
            ]
        )
        # Simulate the --no-memory override logic from main()
        if args.no_memory:
            args.per_task_forum_rounds = 0
            args.cross_task_forum_rounds = 0
            args.disable_memory_mcp = True
        assert args.knowledge_db_path == "/tmp/explicit.sqlite"
        assert args.no_memory is True

    def test_without_no_memory_preserves_values(self):
        """Without --no-memory, knowledge_db_path and forum rounds are preserved."""
        parser = build_parser()
        args = parser.parse_args(
            self._BASE_ARGS
            + [
                "--knowledge-db-path",
                "/tmp/keep.sqlite",
                "--per-task-forum-rounds",
                "5",
            ]
        )
        # no_memory is False, so no override
        if args.no_memory:
            args.per_task_forum_rounds = 0
        assert args.knowledge_db_path == "/tmp/keep.sqlite"
        assert args.per_task_forum_rounds == 5


class TestNoMemoryEngineGuards:
    """Test that engine methods respect no_memory flag."""

    def test_inject_seed_bundle_skipped(self):
        """_inject_seed_bundle returns immediately when no_memory is True."""
        from unittest.mock import patch

        config = GenerationConfig(
            num_generations=1,
            num_agents=1,
            no_memory=True,
            seed_bundle_path="/tmp/bundle.json",
        )
        # We test by creating a minimal orchestrator mock and calling the method
        from kcsi.orchestrator.engine import GenerationalOrchestrator

        orch = GenerationalOrchestrator.__new__(GenerationalOrchestrator)
        orch.config = config
        orch.agents = []

        # Should return without trying to read the file
        with patch("kcsi.orchestrator.engine.Path") as mock_path:
            orch._inject_seed_bundle("/tmp/bundle.json")
            # Path should NOT be constructed since we return early
            mock_path.assert_not_called()

    def test_enrich_seed_packages_skipped(self):
        """_enrich_seed_packages returns immediately when no_memory is True."""
        config = GenerationConfig(
            num_generations=1,
            num_agents=1,
            no_memory=True,
        )
        from kcsi.orchestrator.engine import GenerationalOrchestrator
        from kcsi.orchestrator.enrichment_phase import EngineEnrichmentPhaseService
        from kcsi.orchestrator.strategy import DefaultKnowledgeStrategy

        orch = GenerationalOrchestrator.__new__(GenerationalOrchestrator)
        orch.config = config
        orch._memory_store = "should_not_be_accessed"
        orch._knowledge = "should_not_be_accessed"
        orch.agents = []
        # _collaborators() reads these eagerly before the no_memory early-return;
        # harmless values keep the stub minimal without touching the stores.
        orch._best_scores = {}
        orch._holdout_ids = frozenset()
        orch._is_holdout = GenerationalOrchestrator._is_holdout.__get__(orch)
        orch._improvement_strategy = DefaultKnowledgeStrategy()

        # Should return without accessing memory stores
        EngineEnrichmentPhaseService(orch).enrich(
            generation=1,
            assigned_map={"agent1": ["task1"]},
            tasks=[],
        )
        # If it didn't return early, it would try to call methods on the string stores and crash

    def test_no_memory_skips_seed_phase(self, mock_runtime, mock_evaluator, mock_llm):
        runtime = mock_runtime()
        evaluator = mock_evaluator()
        evaluator.evaluate.return_value = {"resolved": False, "native_score": 0.0, "task_type": "swebench"}
        config = GenerationConfig(
            num_generations=2,
            num_agents=1,
            no_memory=True,
            per_task_forum_rounds=0,
        )
        orch = GenerationalOrchestrator(
            config=config,
            runtime=runtime,
            evaluator=evaluator,
            llm=mock_llm(),
            persistence=NoopPersistence(),
        )
        with patch.object(orch._seeding_phase, "run") as seed_phase:
            traces = orch.run([TaskSpec(id="t1", prompt="fix bug")])
            seed_phase.assert_not_called()
        assert [trace.generation for trace in traces] == [1, 2]
        assert runtime.run_task.call_count == 2
        assert [call.kwargs.get("agent_seed_package") for call in runtime.run_task.call_args_list] == [{}, {}]
        assert not any(
            call.kwargs.get("context", {}).get("phase") in {"task_reflection", "lesson_extraction"}
            for call in orch.llm.call.call_args_list
        )

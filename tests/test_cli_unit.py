"""Unit tests for kcsi.cli -- parser defaults and task filtering."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pytest

from kcsi import cli as cli_module
from kcsi.benchmarks.polyglot_harness import DEFAULT_POLYGLOT_TIMEOUT_SEC, PolyglotHarnessEvaluator
from kcsi.cli import (
    _choose_evaluator,
    _ensure_trace_dir,
    _filter_tasks,
    _normalize_evaluator_for_task_source,
    _resolve_runtime_timeout_default,
    _runtime_container_name_prefix,
    _set_container_name_prefix,
    _temporary_env_override,
    _validate_and_normalize_args,
    build_parser,
)
from kcsi.models import TaskSpec
from kcsi.tasks.registry import REGISTRY, TaskSourceSpec, register_task_source


class TestBuildParser:
    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "swebench_pro",
                "--tasks-path",
                "/tmp/tasks.parquet",
                "--knowledge-db-path",
                "/tmp/memory.sqlite",
                "--max-concurrent-tasks",
                "8",
                "--max-concurrent-forum-tasks",
                "4",
            ]
        )
        assert args.runtime == "container"
        assert args.generations == 10
        # --evaluator omitted parses to the None sentinel; the per-task-source
        # default is applied later in main() (issue #1225).
        assert args.evaluator is None
        assert _normalize_evaluator_for_task_source(args.evaluator, task_source="swebench_pro") == "swebench_pro"
        assert args.polyglot_timeout_sec == DEFAULT_POLYGLOT_TIMEOUT_SEC
        assert PolyglotHarnessEvaluator().timeout_sec == DEFAULT_POLYGLOT_TIMEOUT_SEC
        assert args.session_scope == "task"
        assert args.wipe_workspace_per_task == "true"
        assert args.max_task_retries == 3

    def test_task_source_alias_parses_and_canonicalizes(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "arc1",
                "--tasks-path",
                "/tmp/arc-tasks",
                "--knowledge-db-path",
                "/tmp/memory.sqlite",
            ]
        )

        assert args.task_source == "arc1"
        _validate_and_normalize_args(args, parser)
        assert args.task_source == "arc"

    def test_task_source_choices_do_not_block_dynamic_registry(self):
        parser = build_parser()
        spec = TaskSourceSpec(
            name="tmp_dynamic_cli_source_review",
            aliases=("tmp_dynamic_cli_source_review_alias",),
        )
        register_task_source(
            spec,
            replace=True,
        )
        try:
            args = parser.parse_args(
                [
                    "--task-source",
                    "tmp_dynamic_cli_source_review_alias",
                    "--tasks-path",
                    "/tmp/tasks.json",
                    "--knowledge-db-path",
                    "/tmp/memory.sqlite",
                ]
            )

            assert args.task_source == "tmp_dynamic_cli_source_review_alias"
        finally:
            for key in spec.all_names():
                REGISTRY.pop(key, None)

    def test_task_source_required(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_task_map_path_becomes_task_ids_file_when_only_selector(self, tmp_path):
        task_map = tmp_path / "task_map.json"
        task_map.write_text(json.dumps({"tasks": [{"task_id": "t2"}, {"task_id": "t1"}]}), encoding="utf-8")
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "arc",
                "--tasks-path",
                "/tmp/arc-tasks",
                "--knowledge-db-path",
                "/tmp/memory.sqlite",
                "--task-map-path",
                str(task_map),
            ]
        )

        _validate_and_normalize_args(args, parser)

        assert args.task_map_path == str(task_map)
        assert args.task_ids_file == str(task_map)

    def test_task_map_path_rejects_task_ids_mismatch(self, tmp_path, capsys):
        task_map = tmp_path / "task_map.json"
        task_map.write_text(json.dumps({"task_ids": ["t1", "t2"]}), encoding="utf-8")
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "arc",
                "--tasks-path",
                "/tmp/arc-tasks",
                "--knowledge-db-path",
                "/tmp/memory.sqlite",
                "--task-map-path",
                str(task_map),
                "--task-ids",
                "t2,t1",
            ]
        )

        with pytest.raises(SystemExit):
            _validate_and_normalize_args(args, parser)

        assert "task IDs differ" in capsys.readouterr().err

    def test_task_map_path_rejects_task_ids_file_mismatch(self, tmp_path, capsys):
        task_map = tmp_path / "task_map.json"
        task_ids_file = tmp_path / "ids.json"
        task_map.write_text(json.dumps({"task_ids": ["t1", "t2"]}), encoding="utf-8")
        task_ids_file.write_text(json.dumps(["t1", "t3"]), encoding="utf-8")
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "arc",
                "--tasks-path",
                "/tmp/arc-tasks",
                "--knowledge-db-path",
                "/tmp/memory.sqlite",
                "--task-map-path",
                str(task_map),
                "--task-ids-file",
                str(task_ids_file),
            ]
        )

        with pytest.raises(SystemExit):
            _validate_and_normalize_args(args, parser)

        assert "task IDs differ" in capsys.readouterr().err

    def test_three_phase_argparse_defaults_match_dataclass(self):
        """The CLI argparse defaults for the three-phase fields must equal the
        GenerationConfig dataclass defaults (single source of truth, issue #702).
        Guards against the cross_task_forum_rounds 1-vs-2 divergence recurring."""
        from kcsi.models import GenerationConfig

        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "swebench_pro",
                "--tasks-path",
                "/tmp/tasks.parquet",
                "--knowledge-db-path",
                "/tmp/memory.sqlite",
            ]
        )
        cfg = GenerationConfig(num_generations=1, num_agents=1)
        for fld in (
            "per_task_forum_rounds",
            "cross_task_forum_rounds",
            "cross_task_forum_timeout_sec",
            "cross_task_shared_container",
            "distill_enabled",
            "distill_per_task_model",
            "distill_cross_task_model",
            "forum_early_exit",
            "forum_early_exit_poll_sec",
            "forum_early_exit_quorum_pct",
            "forum_early_exit_quorum_grace_sec",
            "require_vector",
        ):
            assert getattr(args, fld) == getattr(cfg, fld), (
                f"CLI default for {fld}={getattr(args, fld)!r} diverges from "
                f"GenerationConfig default {getattr(cfg, fld)!r}"
            )

    def test_arc_native_prompt_writes_advertised_attempt_files(self):
        """ARC is always native: the execution prompt instructs the agent to
        write attempt_1.txt / attempt_2.txt (or per-test attempt files),
        never ``prediction.json``. (The container-side file *writing* itself
        lives in TypeScript under ``runtime_runner/agent-runner/`` and is out
        of scope for this Python test; the filename contract is what binds the
        prompt to behavior.)
        """
        from kcsi.prompts import _build_arc_no_mcp_execution_prompt

        single = _build_arc_no_mcp_execution_prompt(has_memory=False, generation=1, test_count=1)
        assert "attempt_1.txt" in single
        assert "attempt_2.txt" in single
        assert "prediction.json" not in single

        multi = _build_arc_no_mcp_execution_prompt(has_memory=False, generation=1, test_count=3)
        # Multi-test tasks are instructed to use the per-test attempt files the
        # help text advertises.
        assert "attempt_i_1.txt" in multi
        assert "attempt_i_2.txt" in multi
        assert "prediction.json" not in multi

    def test_custom_values(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "swebench_pro",
                "--tasks-path",
                "/tmp/tasks.parquet",
                "--knowledge-db-path",
                "/tmp/memory.sqlite",
                "--max-concurrent-tasks",
                "8",
                "--max-concurrent-forum-tasks",
                "4",
                "--generations",
                "5",
                "--evaluator",
                "none",
                "--seed",
                "42",
            ]
        )
        assert args.generations == 5
        assert args.evaluator == "none"
        assert args.seed == 42

    def test_swebench_timeout_default(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "swebench_pro",
                "--tasks-path",
                "/tmp/tasks.parquet",
                "--knowledge-db-path",
                "/tmp/memory.sqlite",
                "--max-concurrent-tasks",
                "8",
                "--max-concurrent-forum-tasks",
                "4",
            ]
        )
        assert args.swebench_timeout_sec == 3600
        assert args.swebench_harness_grace_sec == 0
        assert args.swebench_docker_network_mode == "host"
        assert args.arc_max_trials == 2

    def test_polyglot_directory_rejected_by_cli_validation(self, tmp_path: Path, capsys, monkeypatch):
        tasks_dir = tmp_path / "polyglot_tasks"
        tasks_dir.mkdir()
        monkeypatch.setattr(cli_module, "_ensure_trace_dir", lambda _experiment_name: None)

        with pytest.raises(SystemExit) as excinfo:
            cli_module.main(
                [
                    "--task-source",
                    "polyglot",
                    "--tasks-path",
                    str(tasks_dir),
                    "--evaluator",
                    "polyglot_harness",
                    "--knowledge-db-path",
                    str(tmp_path / "knowledge.sqlite"),
                    "--runtime-db-path",
                    str(tmp_path / "runtime.sqlite"),
                ]
            )

        assert excinfo.value.code == 2
        captured = capsys.readouterr()
        assert "--task-source polyglot must be an existing .json file" in captured.err

    def test_temporary_env_override_restores_existing(self, monkeypatch):
        monkeypatch.setenv("KCSI_TMP_ENV_OVERRIDE_TEST", "old")
        with _temporary_env_override("KCSI_TMP_ENV_OVERRIDE_TEST", "new"):
            assert cli_module.os.environ["KCSI_TMP_ENV_OVERRIDE_TEST"] == "new"
        assert cli_module.os.environ["KCSI_TMP_ENV_OVERRIDE_TEST"] == "old"

    def test_temporary_env_override_removes_new_value(self, monkeypatch):
        monkeypatch.delenv("KCSI_TMP_ENV_OVERRIDE_TEST", raising=False)
        with _temporary_env_override("KCSI_TMP_ENV_OVERRIDE_TEST", "new"):
            assert cli_module.os.environ["KCSI_TMP_ENV_OVERRIDE_TEST"] == "new"
        assert "KCSI_TMP_ENV_OVERRIDE_TEST" not in cli_module.os.environ

    def test_swebench_network_mode_none_maps_to_block_network(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "swebench_pro",
                "--tasks-path",
                "/tmp/tasks.parquet",
                "--knowledge-db-path",
                "/tmp/memory.sqlite",
                "--swebench-docker-network-mode",
                "none",
                "--evaluator",
                "swebench_pro",
            ]
        )
        evaluator = _choose_evaluator(args)
        assert evaluator.block_network is True

    def test_swebench_network_mode_none_can_be_overridden_by_canonical_no(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "swebench_pro",
                "--tasks-path",
                "/tmp/tasks.parquet",
                "--knowledge-db-path",
                "/tmp/memory.sqlite",
                "--swebench-docker-network-mode",
                "none",
                "--no-swebench-pro-block-network",
                "--evaluator",
                "swebench_pro",
            ]
        )
        evaluator = _choose_evaluator(args)
        assert evaluator.block_network is False

    def test_swebench_custom_legacy_network_mode_is_accepted_as_noop(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "swebench_pro",
                "--tasks-path",
                "/tmp/tasks.parquet",
                "--knowledge-db-path",
                "/tmp/memory.sqlite",
                "--swebench-docker-network-mode",
                "custom_network",
                "--evaluator",
                "swebench_pro",
            ]
        )
        evaluator = _choose_evaluator(args)
        assert evaluator.block_network is False

    def test_swebench_harness_grace_passes_through_choose_evaluator(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "swebench_pro",
                "--tasks-path",
                "/tmp/tasks.parquet",
                "--knowledge-db-path",
                "/tmp/memory.sqlite",
                "--swebench-harness-grace-sec",
                "7",
                "--evaluator",
                "swebench_pro",
            ]
        )
        evaluator = _choose_evaluator(args)
        assert evaluator.harness_grace_sec == 7

    def test_choose_evaluator_tolerates_legacy_namespace_without_new_attrs(self):
        args = argparse.Namespace(
            evaluator="swebench_pro",
            swebench_timeout_sec=30,
        )
        evaluator = _choose_evaluator(args)
        assert evaluator.timeout_sec == 30
        assert evaluator.harness_grace_sec == 0
        assert evaluator.use_local_docker is True
        assert evaluator.block_network is False

    def test_choose_evaluator_returns_terminal_bench_2_evaluator(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "terminal_bench_2",
                "--tasks-path",
                "/tmp/tb2_tasks.json",
                "--knowledge-db-path",
                "/tmp/memory.sqlite",
                "--evaluator",
                "terminal_bench_2",
            ]
        )
        evaluator = _choose_evaluator(args)
        from kcsi.benchmarks import TerminalBench2Evaluator

        assert isinstance(evaluator, TerminalBench2Evaluator)

    def test_normalize_evaluator_omitted_uses_task_source_default(self):
        # Omitted --evaluator (None sentinel) resolves to the task source's
        # registered default evaluator.
        assert _normalize_evaluator_for_task_source(None, task_source="terminal_bench_2") == "terminal_bench_2"
        assert _normalize_evaluator_for_task_source(None, task_source="arc") == "arc_session"
        assert _normalize_evaluator_for_task_source(None, task_source="swebench_pro") == "swebench_pro"

    def test_normalize_evaluator_omitted_unknown_source_falls_back(self):
        # Unknown/unset task source falls back to the historical default.
        assert _normalize_evaluator_for_task_source(None, task_source="") == "swebench_pro"

    def test_normalize_evaluator_preserves_explicit_swebench_pro_for_arc(self):
        # Issue #1225: an explicit --evaluator swebench_pro must NOT be rewritten
        # to arc_session just because it equals the historical parser default.
        assert _normalize_evaluator_for_task_source("swebench_pro", task_source="arc") == "swebench_pro"

    def test_normalize_evaluator_preserves_other_explicit_values(self):
        assert _normalize_evaluator_for_task_source("none", task_source="arc") == "none"
        assert (
            _normalize_evaluator_for_task_source("polyglot_harness", task_source="swebench_pro") == "polyglot_harness"
        )

    def _capture_main_evaluator(self, monkeypatch, tmp_path, extra_args):
        """Drive main() far enough to resolve args.evaluator, then short-circuit.

        Evaluator resolution runs before _validate_and_normalize_args (which we
        intercept), so args.evaluator is already final when captured.
        """
        captured = {}

        class _Stop(Exception):
            pass

        def _capture(args, parser):  # noqa: ANN001
            captured["evaluator"] = args.evaluator
            raise _Stop

        monkeypatch.setattr(cli_module, "_ensure_trace_dir", lambda _experiment_name: None)
        monkeypatch.setattr(cli_module, "_validate_and_normalize_args", _capture)
        tasks = tmp_path / "tasks.json"
        tasks.write_text("[]")
        argv = [
            "--task-source",
            "arc",
            "--tasks-path",
            str(tasks),
            "--knowledge-db-path",
            str(tmp_path / "memory.sqlite"),
            *extra_args,
        ]
        with pytest.raises(_Stop):
            cli_module.main(argv)
        return captured["evaluator"]

    def test_main_preserves_explicit_evaluator_swebench_pro_for_arc(self, monkeypatch, tmp_path):
        # End-to-end through main(): explicit --evaluator swebench_pro with
        # --task-source arc reaches args.evaluator unchanged (issue #1225).
        assert self._capture_main_evaluator(monkeypatch, tmp_path, ["--evaluator", "swebench_pro"]) == "swebench_pro"

    def test_main_omitted_evaluator_resolves_arc_session(self, monkeypatch, tmp_path):
        assert self._capture_main_evaluator(monkeypatch, tmp_path, []) == "arc_session"

    @pytest.mark.parametrize(
        "flag,value",
        [
            ("--swebench-timeout-sec", "0"),
            ("--swebench-timeout-sec", "-1"),
            ("--swebench-harness-grace-sec", "-1"),
            ("--polyglot-test-feedback-max-lines", "-1"),
        ],
    )
    def test_invalid_swebench_timeout_contract_exits(self, flag, value, monkeypatch):
        monkeypatch.setattr(cli_module, "_ensure_trace_dir", lambda _experiment_name: None)
        with pytest.raises(SystemExit):
            cli_module.main(
                [
                    "--task-source",
                    "swebench_pro",
                    "--tasks-path",
                    "/tmp/tasks.parquet",
                    "--knowledge-db-path",
                    "/tmp/memory.sqlite",
                    flag,
                    value,
                ]
            )

    def test_drop_solved_default_true(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "swebench_pro",
                "--tasks-path",
                "/tmp/tasks.parquet",
                "--knowledge-db-path",
                "/tmp/memory.sqlite",
                "--max-concurrent-tasks",
                "8",
                "--max-concurrent-forum-tasks",
                "4",
            ]
        )
        assert args.drop_solved is True

    def test_no_drop_solved_flag(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "swebench_pro",
                "--tasks-path",
                "/tmp/tasks.parquet",
                "--knowledge-db-path",
                "/tmp/memory.sqlite",
                "--max-concurrent-tasks",
                "8",
                "--max-concurrent-forum-tasks",
                "4",
                "--no-drop-solved",
            ]
        )
        assert args.drop_solved is False

    def test_no_drop_solved_warns_for_benchmark_runs(self, caplog):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "arc",
                "--tasks-path",
                "/tmp/tasks.json",
                "--no-drop-solved",
            ]
        )

        with caplog.at_level(logging.WARNING, logger="kcsi.cli"):
            cli_module._validate_and_normalize_args(args, parser)

        assert any(
            "--no-drop-solved is enabled" in rec.message and "per-task answers can carry forward" in rec.message
            for rec in caplog.records
        )

    def test_drop_solved_default_does_not_warn(self, caplog):
        # Negative case: the safe default (drop_solved=True) must stay silent so
        # a regression dropping the `is False` guard is caught.
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "arc",
                "--tasks-path",
                "/tmp/tasks.json",
            ]
        )
        assert args.drop_solved is True

        with caplog.at_level(logging.WARNING, logger="kcsi.cli"):
            cli_module._validate_and_normalize_args(args, parser)

        assert not any("--no-drop-solved is enabled" in rec.message for rec in caplog.records)

    def test_published_benchmark_set_is_registry_derived(self):
        # The warned-source set must be derived from the registry's
        # upstream_strict flag, not a parallel hardcoded list — otherwise a
        # future published source registered but not added here would silently
        # skip the disclosure warning (issue #1143 follow-up).
        from kcsi.tasks import upstream_strict_task_sources

        assert cli_module._PUBLISHED_BENCHMARK_TASK_SOURCES == frozenset(upstream_strict_task_sources())

    def test_resolve_runtime_timeout_default_no_env(self, monkeypatch):
        """Without CROSS_RUNNER_AGENT_TIMEOUT_SEC, default is 1800."""
        monkeypatch.delenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", raising=False)
        assert _resolve_runtime_timeout_default() == 1800

    def test_resolve_runtime_timeout_default_with_env(self, monkeypatch):
        """CROSS_RUNNER_AGENT_TIMEOUT_SEC overrides the 1800 default."""
        monkeypatch.setenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "3600")
        assert _resolve_runtime_timeout_default() == 3600

    def test_resolve_runtime_timeout_default_invalid_env(self, monkeypatch):
        """Non-integer CROSS_RUNNER_AGENT_TIMEOUT_SEC falls back to 1800."""
        monkeypatch.setenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "notanint")
        assert _resolve_runtime_timeout_default() == 1800

    def test_resolve_runtime_timeout_default_zero_env(self, monkeypatch):
        """Zero CROSS_RUNNER_AGENT_TIMEOUT_SEC falls back to 1800."""
        monkeypatch.setenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "0")
        assert _resolve_runtime_timeout_default() == 1800

    def _parse_with_timeout(self, task_source: str, timeout: int):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                task_source,
                "--tasks-path",
                "/tmp/tasks.parquet",
                "--knowledge-db-path",
                "/tmp/memory.sqlite",
                "--runtime-timeout-sec",
                str(timeout),
            ]
        )
        return parser, args

    def _parse_without_timeout(self, task_source: str):
        """Parse with --runtime-timeout-sec OMITTED (exercises the default)."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                task_source,
                "--tasks-path",
                "/tmp/tasks.parquet",
                "--knowledge-db-path",
                "/tmp/memory.sqlite",
            ]
        )
        return parser, args

    def test_negative_runtime_timeout_rejected_for_non_tb2(self, capsys):
        """A negative --runtime-timeout-sec (no hard cap) must hard-error for a
        non-TB2 source: those have no host-side wall-clock backstop."""
        parser, args = self._parse_with_timeout("swebench_pro", -1)
        with pytest.raises(SystemExit):
            _validate_and_normalize_args(args, parser)
        err = capsys.readouterr().err
        assert "terminal_bench_2" in err
        assert "runtime-timeout-sec" in err

    def test_negative_runtime_timeout_allowed_for_tb2(self):
        """TB2 opts into no hard cap via a negative value (task.toml binds)."""
        parser, args = self._parse_with_timeout("terminal_bench_2", -1)
        # Must not raise: TB2's native trial loop enforces its own deadline.
        _validate_and_normalize_args(args, parser)
        assert args.runtime_timeout_sec == -1

    def test_zero_runtime_timeout_allowed_for_non_tb2(self):
        """0 (keep the 1800s cap) is always valid, including non-TB2 sources."""
        parser, args = self._parse_with_timeout("swebench_pro", 0)
        _validate_and_normalize_args(args, parser)
        assert args.runtime_timeout_sec == 0

    @pytest.mark.parametrize("timeout", [0, 600, 1800])
    def test_nonnegative_runtime_timeout_rejected_for_tb2(self, timeout, capsys):
        """TB2's timeout is NOT user-configurable: the per-task task.toml
        [agent].timeout_sec is authoritative, so any KCSI-side hard cap (a
        non-negative value) is rejected — it could only truncate a task below
        its official budget."""
        parser, args = self._parse_with_timeout("terminal_bench_2", timeout)
        with pytest.raises(SystemExit):
            _validate_and_normalize_args(args, parser)
        err = capsys.readouterr().err
        assert "terminal_bench_2" in err
        assert "task.toml" in err

    def test_omitted_runtime_timeout_defaults_to_no_cap_for_tb2(self):
        """A bare TB2 run (no --runtime-timeout-sec) must default to the no-cap
        sentinel so the per-task task.toml timeout binds — NOT silently fall
        into the 1800s cap that would kill long tasks (e.g. build-pov-ray)."""
        parser, args = self._parse_without_timeout("terminal_bench_2")
        _validate_and_normalize_args(args, parser)
        assert args.runtime_timeout_sec < 0

    def test_omitted_runtime_timeout_defaults_to_resolved_for_non_tb2(self, monkeypatch):
        """A bare non-TB2 run resolves to _resolve_runtime_timeout_default()
        (1800s, or CROSS_RUNNER_AGENT_TIMEOUT_SEC when set)."""
        monkeypatch.delenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", raising=False)
        parser, args = self._parse_without_timeout("swebench_pro")
        _validate_and_normalize_args(args, parser)
        assert args.runtime_timeout_sec == 1800

        monkeypatch.setenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "3600")
        parser, args = self._parse_without_timeout("swebench_pro")
        _validate_and_normalize_args(args, parser)
        assert args.runtime_timeout_sec == 3600


class TestFilterTasks:
    def _make_tasks(self, ids: list[str]) -> list[TaskSpec]:
        return [TaskSpec(id=tid, prompt=f"prompt-{tid}") for tid in ids]

    def test_no_filter(self):
        tasks = self._make_tasks(["t1", "t2", "t3"])
        result = _filter_tasks(tasks, None, 0)
        assert len(result) == 3

    def test_filter_by_ids(self):
        tasks = self._make_tasks(["t1", "t2", "t3"])
        result = _filter_tasks(tasks, "t1,t3", 0)
        assert [t.id for t in result] == ["t1", "t3"]

    def test_max_tasks(self):
        tasks = self._make_tasks(["t1", "t2", "t3"])
        result = _filter_tasks(tasks, None, 2)
        assert len(result) == 2

    def test_filter_and_max(self):
        tasks = self._make_tasks(["t1", "t2", "t3"])
        result = _filter_tasks(tasks, "t1,t2,t3", 1)
        assert len(result) == 1

    def test_missing_ids_ignored(self):
        tasks = self._make_tasks(["t1"])
        result = _filter_tasks(tasks, "t1,t99", 0)
        assert [t.id for t in result] == ["t1"]

    def test_empty_task_list(self):
        result = _filter_tasks([], "t1", 0)
        assert result == []

    def test_max_tasks_zero_means_all(self):
        tasks = self._make_tasks(["t1", "t2", "t3"])
        result = _filter_tasks(tasks, None, 0)
        assert len(result) == 3

    def test_task_ids_file_plain_list(self, tmp_path):
        """--task-ids-file with a plain JSON array works."""
        import json

        f = tmp_path / "ids.json"
        f.write_text(json.dumps(["t1", "t3"]))
        tasks = self._make_tasks(["t1", "t2", "t3"])
        result = _filter_tasks(tasks, None, 0, task_ids_file=str(f))
        assert [t.id for t in result] == ["t1", "t3"]

    def test_task_ids_file_dict_format(self, tmp_path):
        """--task-ids-file with a dict containing 'task_ids' key works."""
        import json

        f = tmp_path / "subset.json"
        f.write_text(json.dumps({"task_source": "swebench_pro", "task_ids": ["t2", "t3"]}))
        tasks = self._make_tasks(["t1", "t2", "t3"])
        result = _filter_tasks(tasks, None, 0, task_ids_file=str(f))
        assert [t.id for t in result] == ["t2", "t3"]

    def test_missing_ids_strict(self):
        tasks = self._make_tasks(["t1"])
        with pytest.raises(ValueError, match="requested task IDs not found"):
            _filter_tasks(tasks, "t99", 0, strict=True)

    def test_missing_ids_non_strict(self):
        tasks = self._make_tasks(["t1"])
        result = _filter_tasks(tasks, "t99", 0)
        assert [t.id for t in result] == []

    def test_duplicate_ids_strict(self):
        tasks = self._make_tasks(["t1", "t2"])
        with pytest.raises(ValueError, match="duplicate task id requested"):
            _filter_tasks(tasks, "t1,t1,t2", 0, strict=True)

    def test_duplicate_ids_preserved_order_strict(self):
        tasks = self._make_tasks(["t1", "t2", "t3"])
        result = _filter_tasks(tasks, "t3,t1,t2", 0, strict=True)
        assert [t.id for t in result] == ["t3", "t1", "t2"]

    def test_duplicate_loaded_requested_ids_strict(self):
        tasks = [
            TaskSpec(id="t1", prompt="first"),
            TaskSpec(id="t1", prompt="second"),
            TaskSpec(id="t2", prompt="third"),
        ]
        with pytest.raises(ValueError, match="duplicate loaded task id"):
            _filter_tasks(tasks, "t1", 0, strict=True)

    def test_duplicate_loaded_unrequested_ids_strict_do_not_fail(self):
        tasks = [
            TaskSpec(id="t1", prompt="first"),
            TaskSpec(id="t1", prompt="second"),
            TaskSpec(id="t2", prompt="third"),
        ]
        result = _filter_tasks(tasks, "t2", 0, strict=True)
        assert [t.id for t in result] == ["t2"]

    def test_duplicate_ids_deduped_non_strict(self):
        tasks = self._make_tasks(["t1", "t2"])
        result = _filter_tasks(tasks, "t2,t1,t2", 0)
        assert [t.id for t in result] == ["t2", "t1"]

    def test_task_ids_file_invalid_format(self, tmp_path):
        """--task-ids-file with wrong format raises ValueError."""
        import json

        f = tmp_path / "bad.json"
        f.write_text(json.dumps({"not_task_ids": [1, 2]}))
        tasks = self._make_tasks(["t1"])
        with pytest.raises(ValueError, match="JSON array of strings"):
            _filter_tasks(tasks, None, 0, task_ids_file=str(f))


class TestTraceDirDefaults:
    def test_default_trace_dir_when_env_unset(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.delenv("KCSI_TRACE_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        out = _ensure_trace_dir("arc 5a/5g:10t")
        p = Path(out)
        assert p.exists()
        assert p.is_dir()
        assert p == (tmp_path / "analysis" / "traces" / "arc_5a_5g_10t").resolve()

    def test_preserve_existing_trace_dir_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        custom = tmp_path / "custom_traces" / "expA"
        monkeypatch.setenv("KCSI_TRACE_DIR", str(custom))
        out = _ensure_trace_dir("ignored_experiment_name")
        p = Path(out)
        assert p.exists()
        assert p.is_dir()
        assert p == custom.resolve()


class TestContainerNamePrefix:
    """Prefix must match container_runner.ts naming so `docker ps --filter` hits
    only containers belonging to this experiment. Regression guard: prior cleanup
    filtered on `kcsi-runtime-` alone and cross-killed sibling experiments."""

    def test_basic_experiment_name(self):
        # layout.sanitize_key keeps '_', then JS-side replace converts '_' to '-'
        assert _runtime_container_name_prefix("arc2_audit_v3") == "kcsi-runtime-task--arc2-audit-v3--"

    def test_differing_experiments_produce_distinct_prefixes(self):
        a = _runtime_container_name_prefix("arc2_audit")
        b = _runtime_container_name_prefix("polyglot_audit")
        assert a != b
        assert not a.startswith(b) and not b.startswith(a)

    def test_empty_name_falls_back_to_default(self):
        # Matches layout.sanitize_key fallback behavior.
        assert _runtime_container_name_prefix("") == "kcsi-runtime-task--default--"

    def test_name_longer_than_24_chars_is_truncated(self):
        # sanitize_key caps experiment_part at 24 chars so the container-name
        # prefix is stable and doesn't blow past runtime validator caps.
        name = "very_long_experiment_name_that_exceeds_twenty_four_chars"
        prefix = _runtime_container_name_prefix(name)
        # Strip literal wrapper to recover the experiment-part segment.
        segment = prefix[len("kcsi-runtime-task--") : -len("--")]
        assert len(segment) <= 24

    def test_special_characters_replaced(self):
        # sanitize_key converts non-[A-Za-z0-9._-] to '_', then JS replace
        # converts '_' and '.' to '-'. Slashes and spaces all collapse to '-'.
        assert _runtime_container_name_prefix("arc 5a/5g:10t") == "kcsi-runtime-task--arc-5a-5g-10t--"


class TestContainerPrefixAutoSuffixDesync:
    """Regression guard for the auto-suffix desync bug.

    main() captures `_container_name_prefix` from args.experiment_name before
    the orchestrator is constructed. The engine's __init__ may auto-suffix
    config.experiment_name on DB name collision (e.g., "arc_audit" -> "arc_audit_2").
    If the CLI doesn't re-capture after engine init, atexit/signal cleanup
    targets the original (stale) prefix and either misses this run's
    containers or cross-kills a sibling experiment's. See PR #329 regression.
    """

    @pytest.fixture(autouse=True)
    def _reset_prefix(self):
        # Preserve / restore the module-level prefix around each test so we
        # don't leak state into other tests that also touch cli_module.
        saved = cli_module._container_name_prefix
        cli_module._container_name_prefix = None
        yield
        cli_module._container_name_prefix = saved

    def test_helper_updates_module_level_prefix(self):
        _set_container_name_prefix("arc_audit")
        assert cli_module._container_name_prefix == _runtime_container_name_prefix("arc_audit")
        assert cli_module._container_name_prefix == "kcsi-runtime-task--arc-audit--"

    def test_recapture_after_engine_auto_suffix_hits_suffixed_prefix(self, tmp_path: Path):
        """Simulate the real bug: initial name "arc_audit" exists in the DB,
        engine auto-suffixes config.experiment_name to "arc_audit_2". Without
        the re-capture, _container_name_prefix points at the unsuffixed prefix.
        After the re-capture, it must track the suffixed final name.
        """
        # Phase 1: mimic main() -- initial capture from args.experiment_name.
        original_name = "arc_audit"
        _set_container_name_prefix(original_name)
        initial_prefix = cli_module._container_name_prefix
        assert initial_prefix == "kcsi-runtime-task--arc-audit--"

        # Phase 2: simulate engine.__init__ auto-suffix on DB collision. We
        # don't spin up the full orchestrator, but we do exercise the real
        # MemoryStore probe to prove has_experiment / next_experiment_name
        # produce the suffix the engine would apply.
        from kcsi.memory.store import MemoryStore

        db_path = str(tmp_path / "memory.sqlite")
        seed = MemoryStore(db_path, default_experiment=original_name)
        try:
            # Create the run row so has_experiment() returns True for original_name.
            seed._ensure_run(original_name)
        finally:
            seed.close()

        probe = MemoryStore(db_path, default_experiment=original_name)
        try:
            assert probe.has_experiment(original_name), "seed step failed to register the experiment name in DB"
            suffixed = probe.next_experiment_name(original_name)
        finally:
            probe.close()
        assert suffixed == "arc_audit_2", f"expected engine-style suffix arc_audit_2, got {suffixed!r}"

        # Phase 3: the fix -- cli.py re-captures after orchestrator init using
        # config.experiment_name (which the engine rewrote to `suffixed`).
        _set_container_name_prefix(suffixed)

        final_prefix = cli_module._container_name_prefix
        assert final_prefix == "kcsi-runtime-task--arc-audit-2--"
        assert final_prefix != initial_prefix, (
            "prefix must change after auto-suffix; else atexit cleanup targets "
            "the stale unsuffixed prefix and misses this run's containers"
        )

    def test_no_auto_suffix_leaves_prefix_stable(self, tmp_path: Path):
        """'Starts clean' path -- no collision, no suffix. Re-capturing with
        the unchanged experiment name must be a no-op (same prefix value)."""
        from kcsi.memory.store import MemoryStore

        name = "fresh_experiment"
        _set_container_name_prefix(name)
        before = cli_module._container_name_prefix

        # Engine's probe runs against an empty DB; no collision, no rewrite.
        db_path = str(tmp_path / "memory.sqlite")
        probe = MemoryStore(db_path, default_experiment=name)
        try:
            assert not probe.has_experiment(name)
        finally:
            probe.close()

        # Simulate the re-capture with the unchanged name.
        _set_container_name_prefix(name)
        assert cli_module._container_name_prefix == before


class TestDbPathValidation:
    def test_identical_runtime_and_knowledge_db_path_raises(self, tmp_path):
        """F5: two stores on same file recreates AB-BA hazard (PR #368)."""
        from kcsi.cli import _validate_db_paths

        shared = str(tmp_path / "shared.sqlite")
        with pytest.raises(ValueError, match="must differ"):
            _validate_db_paths(runtime_db_path=shared, knowledge_db_path=shared)

    def test_identical_via_symlink_raises(self, tmp_path):
        real = tmp_path / "real.sqlite"
        real.touch()
        link = tmp_path / "link.sqlite"
        link.symlink_to(real)
        from kcsi.cli import _validate_db_paths

        with pytest.raises(ValueError, match="must differ"):
            _validate_db_paths(runtime_db_path=str(real), knowledge_db_path=str(link))

    def test_different_paths_accepted(self, tmp_path):
        from kcsi.cli import _validate_db_paths

        _validate_db_paths(
            runtime_db_path=str(tmp_path / "runtime.sqlite"),
            knowledge_db_path=str(tmp_path / "knowledge.sqlite"),
        )

    def test_same_basename_different_directories_raises(self, tmp_path):
        """Both DBs are bind-mounted into /app/memory-db by basename
        (runtime_runner/src/container_mounts.ts::addSqliteFileMounts) — a
        basename collision across different directories makes Docker refuse
        to start the container with "Duplicate mount point"."""
        from kcsi.cli import _validate_db_paths

        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        with pytest.raises(ValueError, match="must not share a filename"):
            _validate_db_paths(
                runtime_db_path=str(tmp_path / "a" / "x.sqlite"),
                knowledge_db_path=str(tmp_path / "b" / "x.sqlite"),
            )

    def test_empty_runtime_path_accepted(self, tmp_path):
        from kcsi.cli import _validate_db_paths

        _validate_db_paths(
            runtime_db_path="",
            knowledge_db_path=str(tmp_path / "knowledge.sqlite"),
        )


class TestSeedTestsFlag:
    """``--swebench-pro-seed-tests`` seeds grader test files into the agent's
    repo (DGM-equivalent harness). Default false = upstream-strict."""

    def test_default_argparse_value_is_false(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "swebench_pro",
                "--tasks-path",
                "/tmp/tasks.parquet",
                "--knowledge-db-path",
                "/tmp/m.sqlite",
                "--max-concurrent-tasks",
                "2",
                "--max-concurrent-forum-tasks",
                "1",
            ]
        )
        # PR #585 default — upstream-strict.
        assert args.swebench_pro_seed_tests is False

    def test_explicit_flag_sets_true(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "swebench_pro",
                "--tasks-path",
                "/tmp/tasks.parquet",
                "--knowledge-db-path",
                "/tmp/m.sqlite",
                "--max-concurrent-tasks",
                "2",
                "--max-concurrent-forum-tasks",
                "1",
                "--swebench-pro-seed-tests",
            ]
        )
        assert args.swebench_pro_seed_tests is True


def test_cleanup_containers_scoped_to_experiment_prefix(monkeypatch):
    """Cleanup must filter docker ps by THIS experiment's container-name
    prefix only (never cross-kill a sibling experiment's workers)."""
    import subprocess as sp_mod

    import kcsi.cli as cli_module

    seen_filters = []

    class _Result:
        stdout = ""

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["docker", "ps"]:
            seen_filters.append(cmd[-1])
        return _Result()

    monkeypatch.setattr(cli_module, "_container_name_prefix", "kcsi-runtime-task--myexp--")
    monkeypatch.setattr(sp_mod, "run", fake_run)
    cli_module._cleanup_containers()
    assert seen_filters == ["name=kcsi-runtime-task--myexp--"]

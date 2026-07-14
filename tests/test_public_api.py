"""Tests for the public programmatic API (kcsi.run + package surface)."""

from __future__ import annotations

import warnings
from pathlib import Path

import kcsi


def test_public_surface_is_importable():
    for name in ("run", "GenerationalOrchestrator", "GenerationConfig", "TaskSpec", "TaskTrace"):
        assert name in kcsi.__all__, f"{name} missing from kcsi.__all__"
        assert hasattr(kcsi, name)
    # Extension-registration entry points are exported for variant authors.
    # All four seams re-export both their register_* function and their *Spec
    # dataclass at the top level (issue #739).
    for name in (
        "register_evaluator",
        "register_runtime",
        "register_task_source",
        "register_strategy",
        "EvaluatorSpec",
        "RuntimeSpec",
        "StrategySpec",
        "TaskSourceSpec",
    ):
        assert name in kcsi.__all__, f"{name} missing from kcsi.__all__"
        assert hasattr(kcsi, name)


def test_kcsi_error_is_exported_base_for_concrete_exceptions():
    # ``KcsiError`` is the single base API callers can catch (issue #739).
    assert "KcsiError" in kcsi.__all__
    assert hasattr(kcsi, "KcsiError")

    from kcsi.benchmarks.terminal_bench_2 import TerminalBench2ContractError
    from kcsi.errors import AuthenticationFailure, ContainerRegistryError, WriteIndeterminateError
    from kcsi.orchestrator.engine import ForumValidationError
    from kcsi.providers import ProviderConfigError
    from kcsi.runtime.normalize import SilentAgentRuntimeError

    for exc in (
        AuthenticationFailure,
        ContainerRegistryError,
        WriteIndeterminateError,
        ProviderConfigError,
        SilentAgentRuntimeError,
        TerminalBench2ContractError,
        ForumValidationError,
    ):
        assert issubclass(exc, kcsi.KcsiError), f"{exc.__name__} must subclass KcsiError"

    # Historical second bases are preserved so existing handlers keep working.
    assert issubclass(AuthenticationFailure, RuntimeError)
    assert issubclass(ContainerRegistryError, RuntimeError)
    assert issubclass(TerminalBench2ContractError, ValueError)


def test_registry_dict_is_not_public_surface():
    # Mutating the bare REGISTRY dict bypasses register_*'s duplicate detection,
    # so it is not advertised as public API (issue #739, item 3).
    from kcsi.eval import registry as eval_registry
    from kcsi.runtime import registry as runtime_registry
    from kcsi.tasks import registry as tasks_registry

    for mod in (eval_registry, runtime_registry, tasks_registry):
        assert "REGISTRY" not in mod.__all__, f"REGISTRY should not be in {mod.__name__}.__all__"

    import kcsi.tasks as tasks_pkg

    assert "REGISTRY" not in tasks_pkg.__all__


def test_py_typed_marker_present():
    marker = Path(kcsi.__file__).resolve().parent / "py.typed"
    assert marker.exists(), "PEP 561 py.typed marker missing"


def test_run_wires_orchestrator_and_returns_traces(monkeypatch):
    captured: dict[str, object] = {}
    sentinel_traces = ["trace-1", "trace-2"]

    class _FakeOrchestrator:
        def __init__(self, *, config, runtime, evaluator, llm, persistence, working_dir):
            captured["config"] = config
            captured["runtime"] = runtime
            captured["evaluator"] = evaluator
            captured["llm"] = llm
            captured["persistence"] = persistence
            captured["working_dir"] = working_dir

        def run(self, *, tasks):
            captured["tasks"] = tasks
            return sentinel_traces

    monkeypatch.setattr("kcsi.api.GenerationalOrchestrator", _FakeOrchestrator)

    config = object()
    tasks = ["t1", "t2"]
    runtime = object()
    evaluator = object()
    llm = object()
    persistence = object()

    # config is a bare object() with no knowledge_db_path, so run() correctly
    # emits the empty-path UserWarning; this test only checks wiring, so ignore it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = kcsi.run(
            config,
            tasks,
            runtime=runtime,
            evaluator=evaluator,
            llm=llm,
            persistence=persistence,
            working_dir="/tmp/wd",
        )

    assert result is sentinel_traces
    assert captured["config"] is config
    assert captured["tasks"] is tasks
    assert captured["runtime"] is runtime
    assert captured["evaluator"] is evaluator
    assert captured["llm"] is llm
    assert captured["persistence"] is persistence
    assert captured["working_dir"] == "/tmp/wd"


def test_run_defaults_persistence_none_and_cwd(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeOrchestrator:
        def __init__(self, *, config, runtime, evaluator, llm, persistence, working_dir):
            captured["persistence"] = persistence
            captured["working_dir"] = working_dir

        def run(self, *, tasks):
            return []

    monkeypatch.setattr("kcsi.api.GenerationalOrchestrator", _FakeOrchestrator)

    # Bare object() config → run() emits the empty-knowledge_db_path warning;
    # this test only checks persistence/cwd defaults, so ignore it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        kcsi.run(object(), [], runtime=object(), evaluator=object(), llm=object())

    assert captured["persistence"] is None
    assert captured["working_dir"] == "."


def test_build_helpers_are_exported():
    # Programmatic construction path for registered components (issue #739 item 1).
    for name in ("build_evaluator", "build_runtime"):
        assert name in kcsi.__all__, f"{name} missing from kcsi.__all__"
        assert hasattr(kcsi, name)


def test_default_arg_namespace_has_defaults_without_required_flags():
    # The required CLI flags (--task-source/--tasks-path) are absent as values,
    # and the runtime default is present — so factories never need argv.
    from kcsi.cli import default_arg_namespace

    ns = default_arg_namespace()
    assert ns.task_source is None  # required flag, default unset, not read by factories
    # --evaluator now defaults to the None "omitted" sentinel (issue #1225); the
    # per-task-source default is resolved in main(), and build_evaluator() takes
    # the evaluator name explicitly, so the namespace value is never read.
    assert ns.evaluator is None
    assert ns.runtime == "container"


def test_arc_is_native_by_default():
    # The legacy --arc-mcp / --arc-no-mcp flags have been removed: ARC is always
    # native (attempt-file) now. The flags must no longer parse, and the
    # container executor must default to the native path.
    import pytest

    from kcsi.cli import build_parser
    from kcsi.runtime.container_host import KcsiContainerExecutor

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--task-source", "arc", "--tasks-path", "x", "--arc-mcp"])
    # The executor defaults to the native ARC path.
    assert KcsiContainerExecutor.arc_no_mcp is True


def test_default_arg_namespace_matches_parse_args_for_all_shared_dests():
    # General invariant: when several actions share a dest, default_arg_namespace()
    # must pick the SAME default argparse's parse_args would — the first-registered
    # action's default. This locks the fix against a future flag pair whose
    # defaults differ landing in the wrong order.
    import argparse
    from collections import defaultdict

    from kcsi.cli import build_parser, default_arg_namespace

    parser = build_parser()
    by_dest: dict[str, list] = defaultdict(list)
    for action in parser._actions:
        if action.dest != argparse.SUPPRESS:
            by_dest[action.dest].append(action)
    shared = {dest for dest, acts in by_dest.items() if len(acts) > 1}
    assert "drop_solved" in shared  # sanity: --drop-solved / --no-drop-solved share a dest

    ns = default_arg_namespace()
    cli_ns = parser.parse_args(["--task-source", "arc", "--tasks-path", "x"])
    for dest in shared:
        assert getattr(ns, dest) == getattr(cli_ns, dest), (
            f"default_arg_namespace().{dest} diverges from parse_args default"
        )


def test_build_evaluator_constructs_without_namespace():
    from kcsi.eval import NoopEvaluator

    # No argparse Namespace, no overrides — defaults alone construct the component.
    assert isinstance(kcsi.build_evaluator("none"), NoopEvaluator)


def test_build_evaluator_applies_keyword_overrides():
    from kcsi.benchmarks import PolyglotHarnessEvaluator

    evaluator = kcsi.build_evaluator(
        "polyglot_harness",
        polyglot_docker_image="custom/image:tag",
        polyglot_timeout_sec=42,
    )
    assert isinstance(evaluator, PolyglotHarnessEvaluator)
    assert evaluator.docker_image == "custom/image:tag"
    assert evaluator.timeout_sec == 42


def test_build_runtime_applies_overrides_and_returns_base_runtime(monkeypatch):
    from kcsi.runtime import KcsiContainerExecutor

    monkeypatch.delenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", raising=False)
    # Passing container_command skips the npm bootstrap path.
    runtime = kcsi.build_runtime(
        "container",
        container_command="echo hi",
        knowledge_db_path="/tmp/run_knowledge.sqlite",
    )
    assert isinstance(runtime, KcsiContainerExecutor)
    assert runtime.command == ["echo", "hi"]
    # The programmatic path skips _validate_and_normalize_args, so the arg
    # default (None since the TB2 timeout fix) must be normalized to a concrete
    # int here — a leaked None would TypeError in _build_runner_env at run time.
    assert isinstance(runtime.timeout_sec, int)
    assert runtime.timeout_sec == 1800


def test_build_evaluator_unknown_name_raises():
    import pytest

    with pytest.raises(ValueError, match="unsupported evaluator"):
        kcsi.build_evaluator("does_not_exist")


def _fake_orchestrator_patch(monkeypatch):
    class _FakeOrchestrator:
        def __init__(self, *, config, runtime, evaluator, llm, persistence, working_dir):
            pass

        def run(self, *, tasks):
            return []

    monkeypatch.setattr("kcsi.api.GenerationalOrchestrator", _FakeOrchestrator)


def test_run_warns_when_knowledge_db_path_empty(monkeypatch):
    # The knowledge loop is silently disabled when knowledge_db_path is empty
    # (engine guards store init on `if knowledge_db_path:`). Unlike the CLI,
    # kcsi.run does not derive a default path, so it must make the degrade loud.
    import pytest

    _fake_orchestrator_patch(monkeypatch)
    config = kcsi.GenerationConfig(num_generations=1, num_agents=1)
    assert config.knowledge_db_path == ""

    with pytest.warns(UserWarning, match="knowledge loop is DISABLED"):
        kcsi.run(config, [], runtime=object(), evaluator=object(), llm=object())


def test_run_does_not_warn_when_knowledge_db_path_set(monkeypatch):
    # When a path IS set the loop is active, so no warning should fire.
    import warnings

    _fake_orchestrator_patch(monkeypatch)
    config = kcsi.GenerationConfig(
        num_generations=1,
        num_agents=1,
        knowledge_db_path="/tmp/kcsi_api_test_knowledge.sqlite",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # Would raise if any UserWarning were emitted.
        kcsi.run(config, [], runtime=object(), evaluator=object(), llm=object())

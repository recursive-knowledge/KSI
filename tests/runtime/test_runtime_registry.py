"""Tests for the central runtime registry (src/kcsi/runtime/registry.py).

Pins the PURE-REFACTOR contract: ``container`` resolves, unknown names raise
early, and the registry is runtime-extensible.
"""

from __future__ import annotations

import pytest

from kcsi.runtime.registry import (
    REGISTRY,
    RuntimeSpec,
    get_runtime_spec,
    register_runtime,
    resolve_runtime,
    supported_runtimes,
)


def test_builtin_container():
    import kcsi.runtime  # noqa: F401  (import populates REGISTRY)

    assert supported_runtimes() == ("container",)
    assert supported_runtimes(include_aliases=True) == ("container",)
    assert get_runtime_spec("  CONTAINER ").name == "container"


def test_unknown_runtime_raises():
    with pytest.raises(ValueError):
        get_runtime_spec("does_not_exist")


def test_register_and_resolve_roundtrip():
    spec = RuntimeSpec(name="dummy_rt_t4", factory=lambda args, env=None: object())
    register_runtime(spec)
    try:
        assert resolve_runtime("dummy_rt_t4") is spec
    finally:
        for key in spec.all_names():
            REGISTRY.pop(key, None)


import argparse

import kcsi.runtime as _rt  # noqa: E402
from kcsi.cli import _choose_runtime  # noqa: E402


def _rt_args(runtime="container", task_source=""):
    # container_command non-empty => skips npm bootstrap; benign command.
    return argparse.Namespace(
        runtime=runtime,
        task_source=task_source,
        container_command="echo test",
        runtime_timeout_sec=60,
        session_scope="task",
        wipe_workspace_per_task="false",
        knowledge_db_path="",
        runtime_db_path="",
        disable_memory_mcp=False,
        no_memory=False,
        forum_timeout_sec=60,
    )


def test_choose_runtime_returns_container():
    assert isinstance(_choose_runtime(_rt_args(), provider_env={}), _rt.KcsiContainerExecutor)


def test_choose_runtime_unknown_name_raises():
    import pytest

    with pytest.raises(ValueError):
        _choose_runtime(_rt_args(runtime="openai"), provider_env={})


def test_choose_runtime_tb2_delegates():
    result = _choose_runtime(_rt_args(task_source="terminal_bench_2"), provider_env={})
    assert isinstance(result, _rt.TerminalBench2Executor)


def test_choose_runtime_wires_container_kwargs():
    """Parity guard: the moved container factory must pass behavior-bearing
    kwargs through unchanged. isinstance checks alone miss argument drift."""
    args = _rt_args()
    args.runtime_timeout_sec = 123
    args.session_scope = "experiment"
    args.wipe_workspace_per_task = "true"
    args.no_memory = True  # must disable the memory MCP

    executor = _choose_runtime(args, provider_env={})
    assert isinstance(executor, _rt.KcsiContainerExecutor)
    assert executor.timeout_sec == 123
    assert executor.session_scope == "experiment"
    assert executor.wipe_workspace_per_task is True
    assert executor.disable_memory_mcp is True


def test_choose_runtime_honors_programmatic_bool_wipe_flag():
    """The isinstance(bool) branch added in registry.py: a programmatic
    build_runtime(..., wipe_workspace_per_task=True/False) must be honored
    as-is, NOT coerced via the str(...)=='true' path (which would make a real
    bool `False` -> 'false' -> False by luck but a bool `True` -> 'true' ==
    'true' only because str(True).lower() happens to be 'true'). Pin both
    polarities so a regression in the bool short-circuit is caught."""
    args_true = _rt_args()
    args_true.wipe_workspace_per_task = True
    assert _choose_runtime(args_true, provider_env={}).wipe_workspace_per_task is True

    args_false = _rt_args()
    args_false.wipe_workspace_per_task = False
    assert _choose_runtime(args_false, provider_env={}).wipe_workspace_per_task is False

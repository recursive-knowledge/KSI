"""Tests for the central task-source registry (src/kcsi/tasks/registry.py).

These pin the PURE-REFACTOR contract: every previously-supported source still
resolves, aliases collapse to canonical specs, an unknown source raises early
with a helpful message, and the capability flags reproduce the behavior the
scattered ``task_source ==`` dispatch sites encoded before the refactor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kcsi.models import TaskSpec
from kcsi.tasks import (
    SUPPORTED_TASK_SOURCES,
    get_spec,
    load_tasks_for_source,
    register_task_source,
    resolve_source,
    supported_task_sources,
)
from kcsi.tasks.registry import REGISTRY, TaskSourceSpec

CANONICAL = ("swebench_pro", "arc", "polyglot", "terminal_bench_2", "custom")


def test_canonical_sources_match_legacy_tuple():
    # The first four entries pin the historical benchmark tuple's value/order;
    # custom is the built-in bring-your-own-tasks source appended in registration order.
    assert SUPPORTED_TASK_SOURCES == CANONICAL
    assert supported_task_sources() == CANONICAL


@pytest.mark.parametrize("name", CANONICAL)
def test_every_canonical_source_resolves_to_itself(name):
    spec = get_spec(name)
    assert spec.name == name
    assert resolve_source(name) is spec


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("arc1", "arc"),
        ("arc2", "arc"),
        ("arc_agi", "arc"),
        ("arc_agi_1", "arc"),
        ("arc_agi_2", "arc"),
        ("swebench", "swebench_pro"),
    ],
)
def test_aliases_map_to_canonical(alias, canonical):
    assert get_spec(alias).name == canonical


def test_resolution_is_case_insensitive_and_strips_whitespace():
    assert get_spec("  ARC  ").name == "arc"
    assert get_spec("SweBench_Pro").name == "swebench_pro"


def test_unknown_source_raises_helpful_early_error():
    with pytest.raises(ValueError) as exc:
        get_spec("does_not_exist")
    msg = str(exc.value)
    assert "does_not_exist" in msg
    # The message lists valid sources to guide the caller.
    assert "swebench_pro" in msg
    assert "arc" in msg
    assert "polyglot" in msg
    assert "terminal_bench_2" in msg


def test_resolve_source_returns_none_for_unknown_and_empty():
    assert resolve_source("nope") is None
    assert resolve_source("") is None
    assert resolve_source(None) is None


# ── Capability flags pin current behavior ────────────────────────────────────


def test_arc_supports_mcp_and_is_offline():
    spec = get_spec("arc")
    assert spec.supports_mcp_arc is True
    assert spec.is_offline is True
    assert spec.arc_task_reference is True
    assert spec.prompt_kind == "arc"
    # ARC is the ONLY source that registers the MCP arc toolset / is offline.
    for other in ("swebench_pro", "polyglot", "terminal_bench_2"):
        assert get_spec(other).supports_mcp_arc is False
        assert get_spec(other).is_offline is False


def test_polyglot_evaluator_and_prompt():
    spec = get_spec("polyglot")
    assert spec.default_evaluator == "polyglot_harness"
    assert spec.prompt_kind == "polyglot"
    assert spec.distill_domain_hint.startswith("DOMAIN HINT (polyglot / Exercism)")


def test_swebench_pro_capabilities():
    spec = get_spec("swebench_pro")
    assert spec.default_evaluator == "swebench_pro"
    assert spec.uses_repo_snapshots is True
    assert spec.supports_classification is True
    assert spec.needs_eval_records is True
    # Only swebench_pro uses repo snapshots / classification / eval records.
    for other in ("arc", "polyglot", "terminal_bench_2"):
        s = get_spec(other)
        assert s.uses_repo_snapshots is False
        assert s.supports_classification is False
        assert s.needs_eval_records is False


def test_terminal_bench_2_delegates_runtime():
    spec = get_spec("terminal_bench_2")
    assert spec.delegates_runtime is True
    assert spec.default_evaluator == "terminal_bench_2"
    # Only TB2 uses the dedicated delegating runtime.
    for other in ("arc", "polyglot", "swebench_pro"):
        assert get_spec(other).delegates_runtime is False


def test_default_evaluators_match_legacy_normalization_map():
    # Mirrors the pre-refactor cli._normalize_evaluator_for_task_source map.
    expected = {
        "arc": "arc_session",
        "polyglot": "polyglot_harness",
        "swebench_pro": "swebench_pro",
        "terminal_bench_2": "terminal_bench_2",
    }
    for source, evaluator in expected.items():
        assert get_spec(source).default_evaluator == evaluator


def test_every_canonical_source_validates_tasks_path():
    # Every built-in source wires a ``--tasks-path`` validator on its spec (issue
    # #741, seam 4); the cli rejects any source whose hook is ``None`` as
    # unsupported. This replaced the former ``tasks_path_kind`` if/elif chain.
    for name in CANONICAL:
        assert get_spec(name).validate_tasks_path is not None


# ── Extensibility ────────────────────────────────────────────────────────────


def test_register_task_source_round_trip_and_cleanup():
    spec = TaskSourceSpec(
        name="unit_test_bench",
        aliases=("utb",),
        default_evaluator="none",
        prompt_kind="generic",
    )
    register_task_source(spec)
    try:
        assert get_spec("unit_test_bench") is spec
        assert get_spec("utb") is spec
        assert "unit_test_bench" in supported_task_sources()
    finally:
        # Keep the global registry clean for other tests.
        REGISTRY.pop("unit_test_bench", None)
        REGISTRY.pop("utb", None)


def test_register_duplicate_name_raises_without_replace():
    with pytest.raises(ValueError):
        register_task_source(TaskSourceSpec(name="arc"))


def test_register_replace_allows_override():
    original = get_spec("polyglot")
    replacement = TaskSourceSpec(
        name="polyglot",
        default_evaluator="polyglot_harness",
        prompt_kind="polyglot",
    )
    register_task_source(replacement, replace=True)
    try:
        assert get_spec("polyglot") is replacement
    finally:
        register_task_source(original, replace=True)
    assert get_spec("polyglot") is original


def test_supported_task_sources_includes_aliases_on_request():
    with_aliases = supported_task_sources(include_aliases=True)
    assert with_aliases[: len(CANONICAL)] == CANONICAL
    for alias in ("arc1", "arc2", "arc_agi", "arc_agi_1", "arc_agi_2", "swebench"):
        assert alias in with_aliases


# ── Loader wiring: load_tasks_for_source consults spec.loader ────────────────


def test_builtin_specs_have_loaders_attached():
    # Importing kcsi.tasks (above) pulls in loaders.py, which attaches the
    # built-in loader callables to the registered specs at import time.
    for name in CANONICAL:
        assert callable(get_spec(name).loader), f"{name} spec has no loader attached"


def test_load_tasks_for_source_uses_registered_loader(tmp_path):
    """A synthetic source with a custom loader loads with no loaders.py dispatch edit."""
    calls: list[dict] = []

    def _custom_loader(tasks_path, **kwargs):
        calls.append({"tasks_path": tasks_path, **kwargs})
        return [TaskSpec(id="synthetic-1", repo="", prompt="p", metadata={"task_source": "synthetic_bench"})]

    register_task_source(TaskSourceSpec(name="synthetic_bench", loader=_custom_loader))
    try:
        tasks_path = tmp_path / "tasks.json"
        tasks = load_tasks_for_source(task_source="synthetic_bench", tasks_path=tasks_path)
        assert [t.id for t in tasks] == ["synthetic-1"]
        assert len(calls) == 1
        assert calls[0]["tasks_path"] == tasks_path
        assert calls[0]["task_source"] == "synthetic_bench"
    finally:
        REGISTRY.pop("synthetic_bench", None)


def test_load_tasks_for_source_clear_error_without_loader():
    register_task_source(TaskSourceSpec(name="loaderless_bench"))
    try:
        with pytest.raises(ValueError, match="registered without a loader"):
            load_tasks_for_source(task_source="loaderless_bench", tasks_path=Path("unused.json"))
    finally:
        REGISTRY.pop("loaderless_bench", None)

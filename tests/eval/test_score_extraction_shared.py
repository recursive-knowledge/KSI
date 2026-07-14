"""Tests for the shared generic eval->score precedence helper (issue #741)."""

import subprocess
import sys
import textwrap
from pathlib import Path

import ksi.memory.knowledge_store as _knowledge_store_mod
import ksi.memory.store as _store_mod
from ksi.orchestrator.scoring import score_from_eval_results


def test_native_score_numeric_is_honored():
    assert score_from_eval_results({"native_score": 0.5}) == 0.5


def test_native_score_outranks_resolved():
    # numeric native_score takes precedence over a conflicting resolved flag
    assert score_from_eval_results({"native_score": 0.0, "resolved": True}) == 0.0


def test_resolved_true():
    assert score_from_eval_results({"resolved": True}) == 1.0


def test_resolved_false():
    assert score_from_eval_results({"resolved": False}) == 0.0


def test_instance_report_resolved_true():
    assert score_from_eval_results({"instance_report": {"resolved": True}}) == 1.0


def test_instance_report_resolved_false():
    assert score_from_eval_results({"instance_report": {"resolved": False}}) == 0.0


def test_pass_true():
    assert score_from_eval_results({"pass": True}) == 1.0


def test_pass_false():
    assert score_from_eval_results({"pass": False}) == 0.0


def test_empty_dict_returns_none():
    assert score_from_eval_results({}) is None


def test_non_numeric_native_score_falls_through_to_pass():
    # Strict isinstance guard: a non-numeric (non-None) native_score is NOT
    # honored; precedence falls through to ``pass`` (behavior-preserving
    # unification of the engine's loose guard with store's strict one).
    assert score_from_eval_results({"native_score": "x", "pass": True}) == 1.0


def test_non_bool_resolved_is_not_trusted():
    # #966: only a real ``bool`` is an authoritative verdict. The string
    # ``"false"`` is truthy, so plain ``bool(...)`` would have scored 1.0;
    # the isinstance guard makes it fall through to ``pass`` instead.
    assert score_from_eval_results({"resolved": "false", "pass": False}) == 0.0
    assert score_from_eval_results({"resolved": "false"}) is None
    # int 1 is not a bool either -> falls through to instance_report / pass.
    assert score_from_eval_results({"instance_report": {"resolved": 1}, "pass": False}) == 0.0


def test_store_loads_in_container_mcp_script_mode():
    """store.py must import as a top-level script (container MCP runs
    ``python3 /app/memory/mcp_server.py`` with only ``src/ksi/memory/`` mounted,
    so ``..orchestrator.scoring`` is NOT importable). The score-helper import
    must be guarded like store.py's other relative imports, and the inline
    fallback must reproduce the precedence. Regression for PR #815.
    """
    store_path = Path(_store_mod.__file__)
    script = textwrap.dedent(
        f"""
        import importlib.util
        spec = importlib.util.spec_from_file_location("store", r"{store_path}")
        mod = importlib.util.module_from_spec(spec)
        # __package__ == "" here -> the ``from ..orchestrator...`` import fails
        # exactly as it does in the container, exercising the fallback branch.
        spec.loader.exec_module(mod)
        assert mod.score_from_eval_results({{"resolved": True}}) == 1.0
        assert mod.score_from_eval_results({{"native_score": 0.25}}) == 0.25
        assert mod.score_from_eval_results({{}}) is None
        print("OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"script-mode load failed:\n{proc.stderr}"
    assert "OK" in proc.stdout


def test_knowledge_store_loads_in_container_mcp_script_mode():
    """knowledge_store.py must also import as a top-level script. The container
    MCP server loads it (``from knowledge_store import KnowledgeStore``) with only
    ``src/ksi/memory/`` on sys.path, so its ``from ._store_common import ...``
    relative import fails and must fall back to a sibling top-level import.
    Regression for #862 (the _store_common extraction).
    """
    ks_path = Path(_knowledge_store_mod.__file__)
    script = textwrap.dedent(
        f"""
        import importlib.util
        spec = importlib.util.spec_from_file_location("knowledge_store", r"{ks_path}")
        mod = importlib.util.module_from_spec(spec)
        # __package__ == "" -> the ``from ._store_common`` relative import fails,
        # exercising the script-mode fallback branch.
        spec.loader.exec_module(mod)
        assert hasattr(mod, "KnowledgeStore")
        print("OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"script-mode load failed:\n{proc.stderr}"
    assert "OK" in proc.stdout

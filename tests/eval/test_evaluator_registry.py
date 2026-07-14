"""Tests for the central evaluator registry (src/ksi/eval/registry.py).

Pins the PURE-REFACTOR contract: every previously-supported evaluator still
resolves, an unknown name raises early with a helpful message, and the registry
is runtime-extensible (mirrors src/ksi/tasks/registry.py).
"""

from __future__ import annotations

import pytest

from ksi.eval.registry import (
    REGISTRY,
    EvaluatorSpec,
    get_evaluator_spec,
    register_evaluator,
    resolve_evaluator,
    supported_evaluators,
)


def test_register_and_resolve_roundtrip():
    spec = EvaluatorSpec(name="dummy_eval_t1", factory=lambda args: object())
    register_evaluator(spec)
    try:
        assert resolve_evaluator("dummy_eval_t1") is spec
        assert get_evaluator_spec("DUMMY_EVAL_T1  ").name == "dummy_eval_t1"
    finally:
        for key in spec.all_names():
            REGISTRY.pop(key, None)


def test_duplicate_registration_raises_without_replace():
    spec = EvaluatorSpec(name="dup_eval_t1", factory=lambda args: object())
    register_evaluator(spec)
    try:
        with pytest.raises(ValueError):
            register_evaluator(EvaluatorSpec(name="dup_eval_t1", factory=lambda args: object()))
        register_evaluator(EvaluatorSpec(name="dup_eval_t1", factory=lambda args: object()), replace=True)
    finally:
        REGISTRY.pop("dup_eval_t1", None)


def test_unknown_name_raises_helpful_error():
    with pytest.raises(ValueError) as exc:
        get_evaluator_spec("does_not_exist")
    msg = str(exc.value)
    assert "does_not_exist" in msg


def test_resolve_returns_none_for_unknown_and_empty():
    assert resolve_evaluator("nope_xyz") is None
    assert resolve_evaluator("") is None
    assert resolve_evaluator(None) is None


def test_supported_evaluators_returns_tuple():
    result = supported_evaluators()
    assert isinstance(result, tuple)


CANONICAL_EVALUATORS = ("none", "command", "arc_session", "swebench_pro", "polyglot_harness", "terminal_bench_2")


def test_builtins_match_legacy_tuple():
    # Import triggers built-in registration.
    import ksi.eval  # noqa: F401
    from ksi.eval import SUPPORTED_EVALUATORS

    assert SUPPORTED_EVALUATORS == CANONICAL_EVALUATORS
    assert supported_evaluators() == CANONICAL_EVALUATORS


@pytest.mark.parametrize("name", CANONICAL_EVALUATORS)
def test_every_builtin_resolves(name):
    import ksi.eval  # noqa: F401

    assert get_evaluator_spec(name).name == name


import argparse  # noqa: E402

import ksi.benchmarks as _bench  # noqa: E402
import ksi.eval as _eval  # noqa: E402
from ksi.cli import _choose_evaluator  # noqa: E402


def _eval_args(name):
    # polyglot reads two attrs directly (not via getattr); supply benign values.
    return argparse.Namespace(evaluator=name, polyglot_docker_image="img:test", polyglot_timeout_sec=60)


def test_choose_evaluator_dispatch_parity():
    assert isinstance(_choose_evaluator(_eval_args("none")), _eval.NoopEvaluator)
    assert isinstance(_choose_evaluator(_eval_args("arc_session")), _bench.ArcSessionEvaluator)
    assert isinstance(_choose_evaluator(_eval_args("swebench_pro")), _bench.SwebenchProEvaluator)
    assert isinstance(_choose_evaluator(_eval_args("polyglot_harness")), _bench.PolyglotHarnessEvaluator)
    assert isinstance(_choose_evaluator(_eval_args("terminal_bench_2")), _bench.TerminalBench2Evaluator)


def test_polyglot_registry_uses_cli_timeout_value():
    ev = _choose_evaluator(_eval_args("polyglot_harness"))
    assert isinstance(ev, _bench.PolyglotHarnessEvaluator)
    assert ev.timeout_sec == 60


def test_choose_evaluator_unknown_raises():
    with pytest.raises(ValueError):
        _choose_evaluator(_eval_args("bogus_evaluator"))

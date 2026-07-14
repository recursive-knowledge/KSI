"""Existence/smoke check for the extracted eval stage.

This only asserts that ``_eval_one_attempt`` remains a method on
``EngineExecutionPhaseService``. Behavioral coverage of the eval stage lives in
``tests/orchestrator/test_execution_phase_service.py`` (see
``test_eval_one_attempt_produces_scored_trace_via_service``).
"""

from kcsi.orchestrator.execution_phase import EngineExecutionPhaseService


def test_eval_one_attempt_method_exists():
    assert callable(getattr(EngineExecutionPhaseService, "_eval_one_attempt", None))

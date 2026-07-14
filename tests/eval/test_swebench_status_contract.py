"""Contract test for the ``swebench_status`` producer/consumer vocabulary.

The evaluator (``src/kcsi/benchmarks/swebench_pro.py``, producer) tags eval-result
dicts with a ``swebench_status`` string; ``score_swebench_from_eval``
(``src/kcsi/orchestrator/scoring.py``, consumer) maps any FAILURE status to an
unscored ``None``. Both sides now reference the single source of truth
``SWEBENCH_FAILURE_STATUSES`` in ``src/kcsi/benchmarks/swebench_pro_external.py``.

These tests fail loudly if the producer ever emits a ``swebench_status`` value
that is not part of the centralized vocabulary, or if the consumer stops
treating a centralized failure status as unscored (``None``).
"""

from __future__ import annotations

import ast
from pathlib import Path

from kcsi.benchmarks.swebench_pro_external import SWEBENCH_FAILURE_STATUSES, SWEBENCH_STATUS_OK
from kcsi.orchestrator.scoring import score_swebench_from_eval

_PRODUCER_SOURCE = Path(__file__).resolve().parents[2] / "src" / "kcsi" / "benchmarks" / "swebench_pro.py"


def _emitted_swebench_statuses() -> set[str]:
    """Collect every string literal assigned to a ``"swebench_status"`` key in the producer."""
    tree = ast.parse(_PRODUCER_SOURCE.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "swebench_status"
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                found.add(value.value)
    return found


def test_producer_emits_only_centralized_statuses():
    allowed = set(SWEBENCH_FAILURE_STATUSES) | {SWEBENCH_STATUS_OK}
    emitted = _emitted_swebench_statuses()
    # Guard against an empty scan masking drift.
    assert emitted, "expected to find swebench_status literals in the producer"
    assert emitted <= allowed, f"producer emits uncentralized swebench_status values: {emitted - allowed}"


def test_producer_emits_every_failure_status():
    # Every centralized failure status is actually produced somewhere — keeps the
    # constant from accumulating dead vocabulary the producer never emits.
    assert set(SWEBENCH_FAILURE_STATUSES) <= _emitted_swebench_statuses()


def test_consumer_scores_unscored_for_every_failure_status():
    for status in SWEBENCH_FAILURE_STATUSES:
        assert score_swebench_from_eval({"swebench_status": status}, task=None) is None, status


def test_consumer_does_not_zero_ok_status():
    # ``ok`` is not a failure; it falls through to the final ``return 0.0`` only
    # because no resolved/unresolved evidence is supplied — sanity-check the
    # status itself is not in the failure set.
    assert SWEBENCH_STATUS_OK not in SWEBENCH_FAILURE_STATUSES

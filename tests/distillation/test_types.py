import pytest

from ksi.distillation.per_task import _as_int_list
from ksi.distillation.types import (
    CROSS_TASK_INSIGHT_FIELDS,
    CrossTaskBundle,
    PerTaskBundle,
    coerce_positive_int,
)


def test_per_task_bundle_fields():
    b = PerTaskBundle(
        task_id="t1",
        transferable_insights=["i"],
        confirmed_constraints=["cc"],
        rejected_hypotheses=["rh"],
        pitfalls=["p"],
        checks=["c"],
        next_steps=["n"],
        evidence_post_ids=[1, 2],
    )
    assert b.task_id == "t1"
    assert b.to_dict()["confirmed_constraints"] == ["cc"]
    assert b.to_dict()["next_steps"] == ["n"]


def test_cross_task_bundle_fields():
    b = CrossTaskBundle(
        transferable_insights=["i"],
        pitfalls=[],
        checks=[],
        evidence_post_ids=[],
        next_steps=["n"],
    )
    assert b.transferable_insights == ["i"]
    assert b.to_dict()["next_steps"] == ["n"]


def test_cross_task_insight_fields_single_source_of_truth():
    """Every consumer of the cross-task insight-field schema must reference the
    single canonical ``CROSS_TASK_INSIGHT_FIELDS`` tuple, not a re-declared
    literal. A future drift in field names/order then fails loudly here instead
    of silently desyncing distillation/embedding/dedup.
    """
    from ksi.distillation import per_task
    from ksi.orchestrator import distillation_phase, kt_adapter

    # per_task aliases the canonical tuple object directly.
    assert per_task._BUNDLE_ITEM_FIELDS is CROSS_TASK_INSIGHT_FIELDS

    # The orchestrator modules import the canonical name (rather than inlining a
    # literal), so their module namespace holds the same object.
    assert kt_adapter.CROSS_TASK_INSIGHT_FIELDS is CROSS_TASK_INSIGHT_FIELDS
    assert distillation_phase.CROSS_TASK_INSIGHT_FIELDS is CROSS_TASK_INSIGHT_FIELDS

    # Pin the expected schema so a deliberate change must be made here too.
    assert CROSS_TASK_INSIGHT_FIELDS == (
        "transferable_insights",
        "confirmed_constraints",
        "rejected_hypotheses",
        "pitfalls",
        "checks",
        "next_steps",
    )


@pytest.mark.parametrize(
    "bad",
    [float("inf"), float("-inf"), float("nan"), "Infinity", "-Infinity", "nan"],
)
def test_coerce_positive_int_drops_non_finite_without_raising(bad):
    """Non-finite / non-integer-coercible evidence ids must be DROPPED (return
    None), never raise out of the lenient parser (deep-review #1264).

    ``int(float("inf"))`` raises ``OverflowError`` — previously outside the
    caught ``(TypeError, ValueError)`` tuple, so it propagated and crashed
    ``_as_int_list``/evidence-id parsing.
    """
    assert coerce_positive_int(bad) is None


def test_as_int_list_drops_non_finite_and_continues():
    """A non-finite id in the middle of a list is silently dropped; valid ids
    on either side are still parsed."""
    assert _as_int_list([1, float("inf"), 2, float("nan"), "Infinity", 3]) == [1, 2, 3]

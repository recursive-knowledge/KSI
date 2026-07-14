from kcsi.cli import _resolve_arc_split
from kcsi.models import TaskSpec


def test_resolve_arc_split_training():
    tasks = [TaskSpec(id="t1", metadata={"arc_split": "training"})]
    assert _resolve_arc_split(tasks) == "training"


def test_resolve_arc_split_evaluation():
    tasks = [TaskSpec(id="t1", metadata={"arc_split": "evaluation"})]
    assert _resolve_arc_split(tasks) == "evaluation"


def test_resolve_arc_split_mixed_or_absent_is_none():
    mixed = [
        TaskSpec(id="a", metadata={"arc_split": "training"}),
        TaskSpec(id="b", metadata={"arc_split": "evaluation"}),
    ]
    assert _resolve_arc_split(mixed) is None
    assert _resolve_arc_split([TaskSpec(id="c", metadata={})]) is None

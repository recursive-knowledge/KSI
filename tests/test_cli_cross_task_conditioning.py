import pytest

from kcsi.cli import build_parser

_REQUIRED = ["--task-source", "arc", "--tasks-path", "/tmp/does-not-matter.json"]


def _parse(argv):
    return build_parser().parse_args([*_REQUIRED, *argv])


def test_cross_task_conditioning_defaults_true():
    args = _parse([])
    assert args.cross_task_distill_target_conditioning is True


def test_cross_task_conditioning_can_be_disabled():
    args = _parse(["--cross-task-distill-target-conditioning", "false"])
    assert args.cross_task_distill_target_conditioning is False


def test_cross_task_conditioning_bare_flag_is_true():
    args = _parse(["--cross-task-distill-target-conditioning"])
    assert args.cross_task_distill_target_conditioning is True


def test_cross_task_conditioning_rejects_invalid_boolean():
    with pytest.raises(SystemExit):
        _parse(["--cross-task-distill-target-conditioning", "flase"])


def test_per_target_selection_defaults_false():
    args = _parse([])
    assert args.cross_task_distill_per_target_selection is False


def test_per_target_selection_can_be_enabled():
    args = _parse(["--cross-task-distill-per-target-selection", "true"])
    assert args.cross_task_distill_per_target_selection is True


def test_per_target_selection_bare_flag_is_true():
    args = _parse(["--cross-task-distill-per-target-selection"])
    assert args.cross_task_distill_per_target_selection is True


def test_per_target_selection_rejects_invalid_boolean():
    with pytest.raises(SystemExit):
        _parse(["--cross-task-distill-per-target-selection", "flase"])


def test_per_target_selection_requires_target_conditioning():
    with pytest.raises(SystemExit):
        _parse(
            [
                "--cross-task-distill-target-conditioning",
                "false",
                "--cross-task-distill-per-target-selection",
                "true",
            ]
        )

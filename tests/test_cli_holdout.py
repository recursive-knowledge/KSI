"""Tests for the --holdout-task-ids / --holdout-task-ids-file CLI surface.

Hold-out tasks are attempted every generation with the current cross-task
knowledge injected but are excluded from learning, ``--drop-solved``,
early-stop, and headline metrics. The CLI helpers tested here resolve the
hold-out id set and enforce disjointness against the training task ids.
"""

from __future__ import annotations

import json

import pytest

from ksi.cli import _resolve_holdout_ids, _select_holdout_tasks, build_parser
from ksi.models import GenerationConfig, TaskSpec


def _tasks(*ids: str) -> list[TaskSpec]:
    return [TaskSpec(id=i, prompt=f"prompt-{i}") for i in ids]


class TestResolveHoldoutIds:
    def test_none_inputs_yield_empty_list(self):
        assert _resolve_holdout_ids(None, None, training_ids={"t1"}) == []

    def test_parse_csv(self):
        assert _resolve_holdout_ids("h1,h2", None, training_ids={"t1"}) == ["h1", "h2"]

    def test_csv_strips_whitespace_and_empties(self):
        assert _resolve_holdout_ids(" h1 , ,h2,", None, training_ids=set()) == ["h1", "h2"]

    def test_dedupes_preserving_order(self):
        assert _resolve_holdout_ids("h2,h1,h2", None, training_ids=set()) == ["h2", "h1"]

    def test_file_plain_list(self, tmp_path):
        f = tmp_path / "ids.json"
        f.write_text(json.dumps(["h1", "h2"]))
        assert _resolve_holdout_ids(None, str(f), training_ids={"t1"}) == ["h1", "h2"]

    def test_file_task_ids_dict(self, tmp_path):
        f = tmp_path / "ids.json"
        f.write_text(json.dumps({"task_ids": ["h1", "h2"]}))
        assert _resolve_holdout_ids(None, str(f), training_ids={"t1"}) == ["h1", "h2"]

    def test_file_task_map_shape(self, tmp_path):
        f = tmp_path / "map.json"
        f.write_text(json.dumps({"tasks": [{"task_id": "h1"}, {"task_id": "h2"}]}))
        assert _resolve_holdout_ids(None, str(f), training_ids={"t1"}) == ["h1", "h2"]

    def test_csv_and_file_merge_dedupe(self, tmp_path):
        f = tmp_path / "ids.json"
        f.write_text(json.dumps(["h2", "h3"]))
        assert _resolve_holdout_ids("h1,h2", str(f), training_ids=set()) == ["h1", "h2", "h3"]

    def test_file_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="holdout-task-ids-file"):
            _resolve_holdout_ids(None, str(tmp_path / "nope.json"), training_ids=set())

    def test_file_bad_shape_raises(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text(json.dumps({"wrong": 1}))
        with pytest.raises(ValueError, match="holdout-task-ids-file"):
            _resolve_holdout_ids(None, str(f), training_ids=set())

    def test_overlap_with_training_raises_disjoint(self):
        with pytest.raises(ValueError, match="disjoint") as exc_info:
            _resolve_holdout_ids("h1,t1", None, training_ids={"t1", "t2"})
        assert "t1" in str(exc_info.value)


class TestSelectHoldoutTasks:
    def test_selects_specs_in_holdout_order(self):
        all_tasks = _tasks("t1", "h2", "h1")
        selected = _select_holdout_tasks(all_tasks, ["h1", "h2"])
        assert [t.id for t in selected] == ["h1", "h2"]

    def test_missing_from_source_raises(self):
        all_tasks = _tasks("t1", "h1")
        with pytest.raises(ValueError, match="missing") as exc_info:
            _select_holdout_tasks(all_tasks, ["h1", "ghost"])
        assert "ghost" in str(exc_info.value)

    def test_empty_ids_yield_empty_list(self):
        assert _select_holdout_tasks(_tasks("t1"), []) == []


class TestHoldoutArgparse:
    def test_flags_default_none(self):
        parser = build_parser()
        args = parser.parse_args(["--task-source", "arc", "--tasks-path", "x.json"])
        assert args.holdout_task_ids is None
        assert args.holdout_task_ids_file is None

    def test_flags_accepted(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task-source",
                "arc",
                "--tasks-path",
                "x.json",
                "--holdout-task-ids",
                "h1,h2",
                "--holdout-task-ids-file",
                "ids.json",
            ]
        )
        assert args.holdout_task_ids == "h1,h2"
        assert args.holdout_task_ids_file == "ids.json"


class TestGenerationConfigHoldout:
    def test_default_empty_list(self):
        config = GenerationConfig(num_generations=1, num_agents=1)
        assert config.holdout_task_ids == []

    def test_default_factory_not_shared(self):
        a = GenerationConfig(num_generations=1, num_agents=1)
        b = GenerationConfig(num_generations=1, num_agents=1)
        a.holdout_task_ids.append("h1")
        assert b.holdout_task_ids == []

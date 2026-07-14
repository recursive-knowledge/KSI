"""Coverage for task-driven population sizing, claiming, and seeding."""

import pytest

from kcsi.discussion.prompts import extract_json
from kcsi.models import GenerationConfig
from kcsi.orchestrator.population import (
    TaskStrategy,
    make_strategy,
)
from kcsi.seeding.seeder import PopulationSeeder


def test_task_mode_cross_task_bundle_and_task_labels():
    """Task-mode seeding: every agent gets the shared cross-task bundle and
    a per-task label (which doubles as the workstream name)."""
    cross_task_bundle = {
        "transferable_insights": ["Use fixtures to isolate setup."],
        "pitfalls": ["Don't guess — check logs first."],
        "checks": ["Run the full test suite."],
        "evidence_post_ids": [1, 2],
    }
    labels = ["task-a", "task-b", "task-c"]

    seeder = PopulationSeeder()
    agents = seeder.seed(
        num_agents=3,
        generation=0,
        cross_task_bundle=cross_task_bundle,
        task_labels=labels,
    )

    assert [agent.workstream for agent in agents] == labels
    assert [agent.workstream_description for agent in agents] == labels
    assert agents[0].seed_package["workstream_name"] == "task-a"
    assert agents[1].seed_package["workstream_name"] == "task-b"
    assert agents[2].seed_package["workstream_name"] == "task-c"
    # Every agent carries the same cross-task bundle.
    for agent in agents:
        assert agent.seed_package["cross_task_bundle"] == cross_task_bundle


# ── Strategy arithmetic ──────────────────────────────────────────────────────


class TestTaskStrategy:
    def test_next_agent_count(self):
        s = TaskStrategy()
        assert s.next_agent_count(generation=1, remaining_tasks=7) == 7


class TestMakeStrategy:
    def test_task(self):
        cfg = GenerationConfig(num_generations=3, num_agents=10)
        assert isinstance(make_strategy(cfg), TaskStrategy)

    def test_default_is_task(self):
        cfg = GenerationConfig(num_generations=3, num_agents=10)
        assert isinstance(make_strategy(cfg), TaskStrategy)


class TestExtractJson:
    def test_direct_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert extract_json('text\n```json\n{"a": 1}\n```\nmore') == {"a": 1}

    def test_embedded_json(self):
        assert extract_json('prefix {"a": 1} suffix') == {"a": 1}

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON"):
            extract_json("no json here")

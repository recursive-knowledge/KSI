"""Tests for core data models."""

from ksi.models import (
    AgentState,
    Assignment,
    GenerationConfig,
    Insight,
    TaskSpec,
    TaskTrace,
)


def test_generation_config_new_fields():
    cfg = GenerationConfig(num_generations=3, num_agents=5)
    assert cfg.drop_solved is True
    assert cfg.solved_threshold == 1.0


def test_generation_config_three_phase_defaults():
    """A directly-constructed GenerationConfig carries the three-phase defaults
    that match the CLI argparse defaults (issue #702). The key invariant is
    cross_task_forum_rounds == 2 (previously the engine getattr default was 1,
    diverging from the CLI default of 2)."""
    cfg = GenerationConfig(num_generations=1, num_agents=1)
    assert cfg.per_task_forum_rounds == 1
    assert cfg.cross_task_forum_rounds == 2  # single source of truth, matches CLI default
    assert cfg.cross_task_forum_timeout_sec == 900
    assert cfg.cross_task_shared_container is False
    assert cfg.distill_enabled is True
    assert cfg.distill_per_task_model is None
    assert cfg.distill_cross_task_model is None
    assert cfg.forum_early_exit is False
    assert cfg.forum_early_exit_poll_sec == 3.0
    # Quorum-based early exit (#1045): default preserves the pre-#1045
    # all-required behavior exactly.
    assert cfg.forum_early_exit_quorum_pct == 100.0
    assert cfg.forum_early_exit_quorum_grace_sec == 0.0
    assert cfg.require_vector is False


def test_generation_config_three_phase_overridable():
    """The three-phase fields are real declared fields (not dynamic attrs), so
    they accept constructor overrides and appear in asdict()."""
    from dataclasses import asdict

    cfg = GenerationConfig(
        num_generations=1,
        num_agents=1,
        cross_task_forum_rounds=5,
        distill_enabled=False,
        distill_per_task_model="claude-haiku-4-5",
    )
    assert cfg.cross_task_forum_rounds == 5
    assert cfg.distill_enabled is False
    assert cfg.distill_per_task_model == "claude-haiku-4-5"
    d = asdict(cfg)
    # asdict() now includes these (previously dropped because they were
    # monkey-patched on after construction).
    assert d["cross_task_forum_rounds"] == 5
    assert d["forum_early_exit_poll_sec"] == 3.0
    assert "require_vector" in d


def test_generation_config_no_evolution_fields():
    cfg = GenerationConfig(num_generations=1, num_agents=1)
    assert not hasattr(cfg, "kill_fraction")
    assert not hasattr(cfg, "sexual_prob")
    assert not hasattr(cfg, "niche_threshold")
    assert not hasattr(cfg, "cost_weight")
    assert not hasattr(cfg, "no_evolution")
    assert not hasattr(cfg, "no_memory_inherit")
    assert not hasattr(cfg, "no_wipe")


def test_agent_state_workstream_field():
    agent = AgentState(id="a-0", generation=0, workstream="debugging")
    assert agent.workstream == "debugging"
    assert not hasattr(agent, "parent_ids")
    assert not hasattr(agent, "fitness")


def test_insight_dataclass():
    ins = Insight(
        id="ins-1",
        text="Use pytest fixtures for setup",
        author_agent_id="a-0",
        generation=1,
        workstream="testing",
        confidence="high",
    )
    assert ins.workstream == "testing"
    assert ins.source_task_id is None
    assert ins.evidence_refs == []


def test_task_spec_unchanged():
    ts = TaskSpec(id="t1", repo="django", prompt="fix bug")
    assert ts.id == "t1"


def test_assignment_unchanged():
    a = Assignment(generation=0, agent_id="a0", task_id="t1")
    assert a.agent_id == "a0"


def test_task_trace_unchanged():
    tt = TaskTrace(generation=0, agent_id="a0", task_id="t1", model_output="patch", eval_result={}, native_score=1.0)
    assert tt.native_score == 1.0

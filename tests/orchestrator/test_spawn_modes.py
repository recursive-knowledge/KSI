"""Tests for task-mode orchestration features (simplified from spawn modes).

The old shrink/chaos/workstream-claiming modes have been removed. Only
task-mode round-robin assignment remains.
"""

from unittest.mock import MagicMock

from kcsi.models import GenerationConfig, TaskSpec
from kcsi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from kcsi.tokens import LLMResponse, TokenUsage


class _DummyLLM:
    def __init__(self, raw: str):
        self._raw = raw

    def call(self, system: str, user: str, *, context=None):
        return LLMResponse(text=self._raw, usage=TokenUsage(input_tokens=1, output_tokens=1))


class _ByAgentLLM:
    def __init__(self, mapping: dict[str, str], default: str = '{"claimed_tasks": []}'):
        self._mapping = mapping
        self._default = default

    def call(self, system: str, user: str, *, context=None):
        agent_id = (context or {}).get("agent_id", "")
        return LLMResponse(
            text=self._mapping.get(agent_id, self._default),
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )


def _make_orchestrator(config: GenerationConfig, llm=None) -> GenerationalOrchestrator:
    runtime = MagicMock()
    evaluator = MagicMock()
    return GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm or _DummyLLM('{"buckets": []}'),
        persistence=NoopPersistence(),
    )


def test_round_robin_claim_phase_assigns_all_tasks():
    """Round-robin claim phase distributes tasks evenly across agents."""
    cfg = GenerationConfig(
        num_generations=1,
        num_agents=3,
    )
    orch = _make_orchestrator(cfg)
    tasks = [TaskSpec(id="t1"), TaskSpec(id="t2"), TaskSpec(id="t3")]
    assignments = orch._claim_phase.claim(1, tasks)

    assert len(assignments) == 3
    assigned_ids = {a.task_id for a in assignments}
    assert assigned_ids == {"t1", "t2", "t3"}
    # Round-robin: each agent gets one task
    agent_ids = {a.agent_id for a in assignments}
    assert len(agent_ids) == 3


def test_round_robin_more_tasks_than_agents():
    """With more tasks than agents, each agent claims exactly one task; the rest
    go unassigned (structural one-task-per-agent invariant)."""
    cfg = GenerationConfig(
        num_generations=1,
        num_agents=2,
    )
    orch = _make_orchestrator(cfg)
    tasks = [TaskSpec(id=f"t{i}") for i in range(5)]
    assignments = orch._claim_phase.claim(1, tasks)

    assert len(assignments) == 2
    # Each agent appears at most once
    by_agent: dict[str, list[str]] = {}
    for a in assignments:
        by_agent.setdefault(a.agent_id, []).append(a.task_id)
    assert all(len(tids) == 1 for tids in by_agent.values())
    assert set(by_agent.keys()) == {"agent-0", "agent-1"}
    # First-in-order tasks win
    assigned_tids = {a.task_id for a in assignments}
    assert assigned_tids == {"t0", "t1"}


def test_align_task_spawn_agents_matches_task_pool_size():
    cfg = GenerationConfig(
        num_generations=3,
        num_agents=5,
    )
    orch = _make_orchestrator(cfg)
    assert len(orch.agents) == 5

    orch._align_task_spawn_agents(2)
    assert [a.id for a in orch.agents] == ["agent-0", "agent-1"]

    orch._align_task_spawn_agents(4)
    assert [a.id for a in orch.agents] == ["agent-0", "agent-1", "agent-2", "agent-3"]


def test_spawn_task_claim_phase_uses_one_to_one_matching():
    cfg = GenerationConfig(
        num_generations=5,
        num_agents=3,
        drop_solved=True,
        solved_threshold=1.0,
    )
    llm = _ByAgentLLM(
        {
            "agent-0": '{"claimed_tasks": ["t1", "t2", "t3"]}',
            "agent-1": '{"claimed_tasks": ["t2", "t3", "t1"]}',
            "agent-2": '{"claimed_tasks": ["t3", "t1", "t2"]}',
        }
    )
    orch = _make_orchestrator(cfg, llm=llm)
    tasks = [TaskSpec(id="t1"), TaskSpec(id="t2"), TaskSpec(id="t3")]

    assignments = orch._claim_phase.claim(1, tasks)

    assert len(assignments) == 3
    assert {a.task_id for a in assignments} == {"t1", "t2", "t3"}


def test_spawn_task_claim_phase_fallback_covers_collisions():
    cfg = GenerationConfig(
        num_generations=1,
        num_agents=3,
    )
    llm = _ByAgentLLM(
        {
            "agent-0": '{"claimed_tasks": ["t1"]}',
            "agent-1": '{"claimed_tasks": ["t1"]}',
            "agent-2": '{"claimed_tasks": ["t1"]}',
        }
    )
    orch = _make_orchestrator(cfg, llm=llm)
    tasks = [TaskSpec(id="t1"), TaskSpec(id="t2"), TaskSpec(id="t3")]

    assignments = orch._claim_phase.claim(1, tasks)

    assert len(assignments) == 3
    assert {a.agent_id for a in assignments} == {"agent-0", "agent-1", "agent-2"}
    assert {a.task_id for a in assignments} == {"t1", "t2", "t3"}


# -- _llm_call TypeError fallback tests --


class _NoContextLLM:
    """LLM that does not accept a context kwarg."""

    def call(self, system: str, user: str):
        return LLMResponse(text="ok", usage=TokenUsage(input_tokens=1, output_tokens=1))


class _ContextLLM:
    """LLM that accepts a context kwarg."""

    def __init__(self):
        self.last_context = None

    def call(self, system: str, user: str, *, context=None):
        self.last_context = context
        return LLMResponse(text="ok", usage=TokenUsage(input_tokens=1, output_tokens=1))


class _BrokenLLM:
    """LLM whose call() raises a TypeError unrelated to context."""

    def call(self, system: str, user: str, *, context=None):
        raise TypeError("missing required argument: 'prompt'")


def test_llm_call_falls_back_without_context_kwarg():
    cfg = GenerationConfig(num_agents=1, num_generations=1)
    orch = _make_orchestrator(cfg, llm=_NoContextLLM())
    resp = orch._llm_call(system="sys", user="usr", context={"key": "val"})
    assert resp.text == "ok"


def test_llm_call_passes_context_when_supported():
    llm = _ContextLLM()
    cfg = GenerationConfig(num_agents=1, num_generations=1)
    orch = _make_orchestrator(cfg, llm=llm)
    orch._llm_call(system="sys", user="usr", context={"agent_id": "a1"})
    assert llm.last_context == {"agent_id": "a1"}


def test_llm_call_reraises_real_type_error():
    import pytest

    cfg = GenerationConfig(num_agents=1, num_generations=1)
    orch = _make_orchestrator(cfg, llm=_BrokenLLM())
    with pytest.raises(TypeError, match="missing required argument"):
        orch._llm_call(system="sys", user="usr")

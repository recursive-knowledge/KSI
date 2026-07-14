"""Tests for token tracking in the container runtime.

Verifies that KsiContainerExecutor correctly populates runtime_meta["tokens_used"]
from the runner JSON output, and that the engine accumulates tokens on agents.
"""

from __future__ import annotations

import json
import sys
import types as _types
from unittest.mock import MagicMock, patch

from conftest import REPO_ROOT

_ROOT = REPO_ROOT


from tests.helpers import _load_by_path

# ── Synthetic ksi_tt package hierarchy ─────────────────────────────────────

_PKG = "ksi_tt"

_root_pkg = _types.ModuleType(_PKG)
_root_pkg.__path__ = [str(_ROOT / "src" / "ksi")]  # type: ignore[attr-defined]
_root_pkg.__package__ = _PKG
sys.modules[_PKG] = _root_pkg

# models
_models_mod = _load_by_path(f"{_PKG}.models", "src/ksi/models.py", package=_PKG)
TaskSpec = _models_mod.TaskSpec
AgentState = _models_mod.AgentState
GenerationConfig = _models_mod.GenerationConfig

# runtime package
_runtime_pkg = _types.ModuleType(f"{_PKG}.runtime")
_runtime_pkg.__path__ = [str(_ROOT / "src" / "ksi" / "runtime")]  # type: ignore[attr-defined]
_runtime_pkg.__package__ = f"{_PKG}.runtime"
sys.modules[f"{_PKG}.runtime"] = _runtime_pkg
_root_pkg.runtime = _runtime_pkg  # type: ignore[attr-defined]

# runtime.types
_types_mod = _load_by_path(f"{_PKG}.runtime.types", "src/ksi/runtime/types.py", package=f"{_PKG}.runtime")
RuntimeResult = _types_mod.RuntimeResult
_runtime_pkg.RuntimeResult = RuntimeResult  # type: ignore[attr-defined]

# prompts stub
_prompts_stub = _types.ModuleType(f"{_PKG}.prompts")
_prompts_stub.__path__ = [str(_ROOT / "src" / "ksi" / "prompts")]  # type: ignore[attr-defined]
_prompts_stub.build_execution_prompt = lambda task, **kw: f"Execute: {task.prompt}"  # type: ignore[attr-defined]
_prompts_stub.build_task_markdown = lambda task: f"# Task {task.id}\n{task.prompt}"  # type: ignore[attr-defined]
sys.modules[f"{_PKG}.prompts"] = _prompts_stub
# engine imports `from ..prompts.kt_adapter import build_kt_adapter_prompts` (#861);
# load the real submodule (no relative imports of its own) under the synthetic pkg.
_kt_prompts_mod = _load_by_path(
    f"{_PKG}.prompts.kt_adapter", "src/ksi/prompts/kt_adapter.py", package=f"{_PKG}.prompts"
)
_prompts_stub.kt_adapter = _kt_prompts_mod  # type: ignore[attr-defined]

# tokens module (required by container_host)
_tokens_mod = _load_by_path(f"{_PKG}.tokens", "src/ksi/tokens.py", package=_PKG)
TokenUsage = _tokens_mod.TokenUsage
_root_pkg.TokenUsage = TokenUsage  # type: ignore[attr-defined]

# container_host
_exe_mod = _load_by_path(
    f"{_PKG}.runtime.container_host",
    "src/ksi/runtime/container_host.py",
    package=f"{_PKG}.runtime",
)
_runtime_pkg.container_host = _exe_mod  # type: ignore[attr-defined]

KsiContainerExecutor = _exe_mod.KsiContainerExecutor
_parse_runner_stdout = _exe_mod._parse_runner_stdout
RUNNER_PATCH = f"{_PKG}.runtime.container_host._run_command_with_backstop"

# discussion package (required by engine)
_disc_pkg = _types.ModuleType(f"{_PKG}.discussion")
_disc_pkg.__path__ = [str(_ROOT / "src" / "ksi" / "discussion")]  # type: ignore[attr-defined]
_disc_pkg.__package__ = f"{_PKG}.discussion"
sys.modules[f"{_PKG}.discussion"] = _disc_pkg

_disc_prompts_mod = _load_by_path(
    f"{_PKG}.discussion.prompts",
    "src/ksi/discussion/prompts.py",
    package=f"{_PKG}.discussion",
)
_disc_pkg.prompts = _disc_prompts_mod  # type: ignore[attr-defined]

# seeding package (required by engine)
_seed_pkg = _types.ModuleType(f"{_PKG}.seeding")
_seed_pkg.__path__ = [str(_ROOT / "src" / "ksi" / "seeding")]  # type: ignore[attr-defined]
_seed_pkg.__package__ = f"{_PKG}.seeding"
sys.modules[f"{_PKG}.seeding"] = _seed_pkg

_seeder_mod = _load_by_path(
    f"{_PKG}.seeding.seeder",
    "src/ksi/seeding/seeder.py",
    package=f"{_PKG}.seeding",
)
_seed_pkg.seeder = _seeder_mod  # type: ignore[attr-defined]

# db package (required by engine)
_db_pkg = _types.ModuleType(f"{_PKG}.db")
# orchestrator package
_orch_pkg_mod = _types.ModuleType(f"{_PKG}.orchestrator")
_orch_pkg_mod.__path__ = [str(_ROOT / "src" / "ksi" / "orchestrator")]  # type: ignore[attr-defined]
_orch_pkg_mod.__package__ = f"{_PKG}.orchestrator"
sys.modules[f"{_PKG}.orchestrator"] = _orch_pkg_mod

# engine
_engine_mod = _load_by_path(
    f"{_PKG}.orchestrator.engine",
    "src/ksi/orchestrator/engine.py",
    package=f"{_PKG}.orchestrator",
)
GenerationalOrchestrator = _engine_mod.GenerationalOrchestrator
NoopPersistence = _engine_mod.NoopPersistence


# ── _parse_runner_stdout token extraction ─────────────────────────────────────


class TestParseRunnerStdoutTokens:
    def test_meta_with_total_tokens_sets_tokens_used(self):
        payload = json.dumps(
            {
                "result": "42",
                "tool_trace": [],
                "meta": {"generation": 1, "total_tokens": 350},
            }
        )
        result = _parse_runner_stdout(payload, key="result")
        assert result["token_usage"].total == 350

    def test_meta_with_per_direction_tokens_sets_tokens_used(self):
        payload = json.dumps(
            {
                "result": "ok",
                "tool_trace": [],
                "meta": {"input_tokens": 120, "output_tokens": 80},
            }
        )
        result = _parse_runner_stdout(payload, key="result")
        assert result["token_usage"].total == 200

    def test_meta_with_nested_usage_sets_tokens_used(self):
        payload = json.dumps(
            {
                "result": "answer",
                "tool_trace": [],
                "meta": {
                    "agent_id": "peer-0",
                    "usage": {"input_tokens": 500, "output_tokens": 300},
                },
            }
        )
        result = _parse_runner_stdout(payload, key="result")
        assert result["token_usage"].total == 800

    def test_meta_without_tokens_does_not_set_tokens_used(self):
        """When the runner doesn't emit token info, tokens_used must be absent (defaults to 0 upstream)."""
        payload = json.dumps(
            {
                "result": "ok",
                "tool_trace": [],
                "meta": {"generation": 1, "agent_id": "peer-0", "status": "success"},
            }
        )
        result = _parse_runner_stdout(payload, key="result")
        # tokens_used should NOT be injected when no token info is present
        assert result["runtime_meta"].get("tokens_used", 0) == 0

    def test_existing_tokens_used_not_overwritten(self):
        """If meta already has tokens_used, don't modify it."""
        payload = json.dumps(
            {
                "result": "ok",
                "tool_trace": [],
                "meta": {"tokens_used": 777, "input_tokens": 100, "output_tokens": 100},
            }
        )
        result = _parse_runner_stdout(payload, key="result")
        # tokens_used was already present — keep it as-is
        assert result["runtime_meta"]["tokens_used"] == 777

    def test_zero_tokens_not_injected(self):
        """When extraction yields 0, don't inject tokens_used to avoid confusion."""
        payload = json.dumps(
            {
                "result": "ok",
                "tool_trace": [],
                "meta": {},
            }
        )
        result = _parse_runner_stdout(payload, key="result")
        assert "tokens_used" not in result["runtime_meta"]

    def test_empty_stdout_gracefully_returns_zero(self):
        result = _parse_runner_stdout("", key="result")
        assert result["runtime_meta"] == {}
        assert result["runtime_meta"].get("tokens_used", 0) == 0

    def test_tokens_source_preserved_in_runtime_meta(self):
        """The agent-runner emits `tokens_source` to flag provenance:
        result_event / per_turn_sum / unavailable. The host MUST preserve
        this field in runtime_meta so downstream consumers (analysis
        scripts, DB audits) can tell a genuine-zero attempt apart from a
        reporting gap.
        """
        payload = json.dumps(
            {
                "result": "ok",
                "tool_trace": [],
                "meta": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "tokens_source": "per_turn_sum",
                },
            }
        )
        result = _parse_runner_stdout(payload, key="result")
        assert result["runtime_meta"]["tokens_source"] == "per_turn_sum"
        assert result["token_usage"].total == 150

    def test_tokens_source_unavailable_when_both_sources_missed(self):
        """When the stream carried no usable usage info, the runner tags
        `tokens_source=unavailable`. Zeros in the counters are still
        zero — but the flag lets us distinguish a reporting gap from a
        genuinely zero attempt (e.g. cached-only recall with no I/O).
        """
        payload = json.dumps(
            {
                "result": "ok",
                "tool_trace": [{"type": "assistant"}] * 10,  # non-trivial trace
                "meta": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "tokens_source": "unavailable",
                    "status": "success",
                },
            }
        )
        result = _parse_runner_stdout(payload, key="result")
        assert result["runtime_meta"]["tokens_source"] == "unavailable"
        assert result["token_usage"].total == 0

    def test_per_turn_sum_regression_bug_shape(self):
        """Regression for the 2026-04 ARC2 baseline-sweep bug: attempts
        275/483 had status=ok + 100+ tool_trace entries but all four token
        fields == 0. The root cause was an agent-runner accumulator that
        read ``message.usage`` (undefined on assistant messages) instead of
        ``message.message.usage`` (where the SDK actually nests per-turn
        usage). Post-fix, the runner sums per-turn deltas and writes them
        into the same meta fields with ``tokens_source=per_turn_sum``.

        This test pins the happy-path shape: if the runner reports
        non-zero tokens with source=per_turn_sum, they flow through
        ``_parse_runner_stdout`` identically to the result_event path.
        """
        # Shape matches attempt 275's trace: 109 trace entries, 47 assistant
        # messages, each with per-turn usage ~ (4 input, 6 output, 12k cache_read).
        payload = json.dumps(
            {
                "result": "answer",
                "tool_trace": [{"type": "assistant"}] * 109,
                "meta": {
                    "input_tokens": 4 * 47,
                    "output_tokens": 6 * 47,
                    "cache_read_input_tokens": 12000 * 47,
                    "cache_creation_input_tokens": 0,
                    "tokens_source": "per_turn_sum",
                    "status": "success",
                },
            }
        )
        result = _parse_runner_stdout(payload, key="result")
        usage = result["token_usage"]
        # All four buckets round-trip intact.
        assert usage.input_tokens == 4 * 47
        assert usage.output_tokens == 6 * 47
        assert usage.cache_read_input_tokens == 12000 * 47
        # Cache-inclusive total should reflect real billed volume, not 0.
        assert usage.total == 4 * 47 + 6 * 47 + 12000 * 47
        assert result["runtime_meta"]["tokens_source"] == "per_turn_sum"


# ── KsiContainerExecutor integration: tokens in RuntimeResult ──────────────


class TestKsiContainerExecutorTokens:
    def _make_executor(self, tmp_path, **kw):
        defaults = dict(
            command=["echo", "{}"],
            working_dir=str(tmp_path),
            timeout_sec=30,
            instruction_path=str(tmp_path / "INSTRUCTION.md"),
            agent_workspace_root=str(tmp_path / "workspaces"),
            env={
                "MODEL_PROVIDER": "anthropic",
                "MODEL_AUTH_MODE": "api",
                "MODEL": "claude-sonnet-4-6",
                "ANTHROPIC_API_KEY": "sk-test",
            },
        )
        defaults.update(kw)
        return KsiContainerExecutor(**defaults)

    def test_tokens_used_populated_in_runtime_result(self, tmp_path):
        out = json.dumps(
            {
                "result": "42",
                "tool_trace": [],
                "meta": {
                    "generation": 1,
                    "agent_id": "peer-0",
                    "task_id": "t1",
                    "status": "success",
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
            }
        )
        with patch(RUNNER_PATCH) as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=out, stderr="")
            ex = self._make_executor(tmp_path)
            result = ex.run_task(
                generation=1,
                agent_id="peer-0",
                task=TaskSpec(id="t1", prompt="q"),
            )
        assert result.token_usage.total == 150

    def test_tokens_used_zero_when_absent(self, tmp_path):
        out = json.dumps(
            {
                "result": "ok",
                "tool_trace": [],
                "meta": {
                    "generation": 1,
                    "agent_id": "peer-0",
                    "task_id": "t1",
                    "status": "success",
                },
            }
        )
        with patch(RUNNER_PATCH) as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=out, stderr="")
            ex = self._make_executor(tmp_path)
            result = ex.run_task(
                generation=1,
                agent_id="peer-0",
                task=TaskSpec(id="t1", prompt="q"),
            )
        # No tokens emitted by runner — default is 0, field may be absent
        assert result.runtime_meta.get("tokens_used", 0) == 0

    def test_tokens_used_with_nested_usage(self, tmp_path):
        out = json.dumps(
            {
                "result": "ok",
                "tool_trace": [],
                "meta": {
                    "generation": 1,
                    "agent_id": "peer-0",
                    "task_id": "t1",
                    "status": "success",
                    "usage": {"input_tokens": 1000, "output_tokens": 500},
                },
            }
        )
        with patch(RUNNER_PATCH) as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=out, stderr="")
            ex = self._make_executor(tmp_path)
            result = ex.run_task(
                generation=1,
                agent_id="peer-0",
                task=TaskSpec(id="t1", prompt="q"),
            )
        assert result.token_usage.total == 1500

    def test_total_tokens_field_used(self, tmp_path):
        out = json.dumps(
            {
                "result": "ok",
                "tool_trace": [],
                "meta": {
                    "generation": 1,
                    "agent_id": "peer-0",
                    "task_id": "t1",
                    "status": "success",
                    "total_tokens": 250,
                },
            }
        )
        with patch(RUNNER_PATCH) as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=out, stderr="")
            ex = self._make_executor(tmp_path)
            result = ex.run_task(
                generation=1,
                agent_id="peer-0",
                task=TaskSpec(id="t1", prompt="q"),
            )
        assert result.token_usage.total == 250


# ── Engine token accumulation ─────────────────────────────────────────────────


class TestEngineTokenAccumulation:
    """Engine token tracking: structural field checks and accumulation stubs.

    Full end-to-end accumulation (tokens_used from runtime_meta summed onto
    AgentState.token_usage) is wired in Task 4. Tests marked xfail document
    the intended behavior without requiring it to pass yet.
    """

    def _make_config(self, **kw):
        defaults = dict(
            num_generations=1,
            num_agents=1,
        )
        defaults.update(kw)
        return GenerationConfig(**defaults)

    def test_tokens_accumulated_on_agent(self):
        """Verify token_usage from RuntimeResult accumulates onto AgentState."""
        rt = MagicMock()
        rt.run_task.return_value = RuntimeResult(
            output="answer",
            tool_trace=[],
            runtime_meta={"tokens_used": 300},
            token_usage=TokenUsage(output_tokens=300),
        )
        ev = MagicMock()
        ev.evaluate.return_value = {"native_score": 1.0}
        llm = MagicMock()

        # 2 tasks → 2 agents, each accumulates 300 tokens.
        orch = GenerationalOrchestrator(
            config=self._make_config(num_agents=2),
            runtime=rt,
            evaluator=ev,
            llm=llm,
        )
        tasks = [
            TaskSpec(id="task-0", prompt="p0"),
            TaskSpec(id="task-1", prompt="p1"),
        ]
        orch.run(tasks=tasks)
        assert hasattr(orch.agents[0], "token_usage")
        assert orch.agents[0].token_usage == 300
        assert orch.agents[1].token_usage == 300

    def test_tokens_zero_when_absent(self):
        rt = MagicMock()
        rt.run_task.return_value = RuntimeResult(
            output="answer",
            tool_trace=[],
            runtime_meta={},  # no token info
        )
        ev = MagicMock()
        ev.evaluate.return_value = {"native_score": 1.0}
        llm = MagicMock()

        orch = GenerationalOrchestrator(
            config=self._make_config(),
            runtime=rt,
            evaluator=ev,
            llm=llm,
        )
        orch.run(tasks=[TaskSpec(id="task-0", prompt="p")])
        assert orch.agents[0].token_usage == 0

    def test_on_generation_end_called_per_generation(self):
        """Persistence should receive on_generation_end per generation."""
        rt = MagicMock()
        rt.run_task.return_value = RuntimeResult(
            output="42",
            tool_trace=[],
            runtime_meta={"tokens_used": 100},
        )
        ev = MagicMock()
        ev.evaluate.return_value = {"native_score": 0.9}
        llm = MagicMock()
        persistence = MagicMock()

        orch = GenerationalOrchestrator(
            config=self._make_config(num_agents=2),
            runtime=rt,
            evaluator=ev,
            llm=llm,
            persistence=persistence,
        )
        orch.run(tasks=[TaskSpec(id="t0", prompt="p")])
        assert persistence.on_generation_end.call_count >= 1


# ── NoopPersistence basic functionality ──────────────────────────────────────


class TestNoopPersistenceBasic:
    def test_on_generation_end_returns_none(self):
        p = NoopPersistence()
        agent = AgentState(id="peer-0")
        result = p.on_generation_end(generation=1, agents=[agent])
        assert result is None

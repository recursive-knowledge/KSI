"""Tests for KsiContainerExecutor auth validation."""

from __future__ import annotations

import sys
import types as _types

import pytest
from conftest import REPO_ROOT

_ROOT = REPO_ROOT

from tests.helpers import _load_by_path

# ── Synthetic ksi_ncbc package hierarchy ──────────────────────────────────

_PKG = "ksi_ncbc"

_root_pkg = _types.ModuleType(_PKG)
_root_pkg.__path__ = [str(_ROOT / "src" / "ksi")]  # type: ignore[attr-defined]
_root_pkg.__package__ = _PKG
sys.modules[_PKG] = _root_pkg

# models
_models_mod = _load_by_path(f"{_PKG}.models", "src/ksi/models.py", package=_PKG)

# runtime package
_runtime_pkg = _types.ModuleType(f"{_PKG}.runtime")
_runtime_pkg.__path__ = [str(_ROOT / "src" / "ksi" / "runtime")]  # type: ignore[attr-defined]
_runtime_pkg.__package__ = f"{_PKG}.runtime"
sys.modules[f"{_PKG}.runtime"] = _runtime_pkg

# tokens module (required by runtime.types and container_host)
_tokens_mod = _load_by_path(f"{_PKG}.tokens", "src/ksi/tokens.py", package=_PKG)
_root_pkg.TokenUsage = _tokens_mod.TokenUsage  # type: ignore[attr-defined]

# runtime.types
_types_mod = _load_by_path(f"{_PKG}.runtime.types", "src/ksi/runtime/types.py", package=f"{_PKG}.runtime")
_runtime_pkg.RuntimeResult = _types_mod.RuntimeResult  # type: ignore[attr-defined]

# prompts stub
_prompts_stub = _types.ModuleType(f"{_PKG}.prompts")
_prompts_stub.build_execution_prompt = lambda task: f"Execute: {task.prompt}"  # type: ignore[attr-defined]
_prompts_stub.build_task_markdown = lambda task: f"# Task {task.id}\n{task.prompt}"  # type: ignore[attr-defined]
sys.modules[f"{_PKG}.prompts"] = _prompts_stub

# container_host
_exe_mod = _load_by_path(
    f"{_PKG}.runtime.container_host",
    "src/ksi/runtime/container_host.py",
    package=f"{_PKG}.runtime",
)

_validate_provider_auth = _exe_mod._validate_provider_auth
_build_runner_env = _exe_mod._build_runner_env
# The module is loaded under the synthetic ``ksi_ncbc`` package, so it raises
# ``ksi_ncbc.errors.AuthenticationFailure`` — a distinct class object from
# ``ksi.errors.AuthenticationFailure``. Reference the one bound in the loaded
# module's namespace so ``pytest.raises`` matches the exact class raised.
AuthenticationFailure = _exe_mod.AuthenticationFailure


class TestValidateProviderAuth:
    def test_anthropic_api_valid(self):
        env = {
            "MODEL_PROVIDER": "anthropic",
            "MODEL_AUTH_MODE": "api",
            "MODEL": "claude-sonnet-4-6",
            "ANTHROPIC_API_KEY": "sk-test",
        }
        _validate_provider_auth(env)  # Should not raise

    def test_missing_api_key(self):
        env = {"MODEL_PROVIDER": "anthropic", "MODEL_AUTH_MODE": "api", "MODEL": "claude-sonnet-4-6"}
        with pytest.raises(AuthenticationFailure, match="ANTHROPIC_API_KEY"):
            _validate_provider_auth(env)

    def test_openai_valid(self):
        env = {"MODEL_PROVIDER": "openai", "MODEL_AUTH_MODE": "api", "MODEL": "gpt-4", "OPENAI_API_KEY": "sk-test"}
        _validate_provider_auth(env)  # Should not raise

    def test_openai_missing_key(self):
        env = {"MODEL_PROVIDER": "openai", "MODEL_AUTH_MODE": "api", "MODEL": "gpt-4"}
        with pytest.raises(AuthenticationFailure, match="OPENAI_API_KEY"):
            _validate_provider_auth(env)

    def test_bad_provider(self):
        env = {"MODEL_PROVIDER": "azureopenai", "MODEL_AUTH_MODE": "api", "MODEL": "gpt-4"}
        with pytest.raises(AuthenticationFailure, match="Unsupported MODEL_PROVIDER"):
            _validate_provider_auth(env)

    def test_subscription_valid(self):
        env = {
            "MODEL_PROVIDER": "anthropic",
            "MODEL_AUTH_MODE": "subscription",
            "MODEL": "claude-sonnet-4-6",
            "CLAUDE_CODE_OAUTH_TOKEN": "tok",
        }
        _validate_provider_auth(env)  # Should not raise

    def test_subscription_missing_token(self):
        env = {"MODEL_PROVIDER": "anthropic", "MODEL_AUTH_MODE": "subscription", "MODEL": "claude-sonnet-4-6"}
        with pytest.raises(AuthenticationFailure, match="CLAUDE_CODE_OAUTH_TOKEN"):
            _validate_provider_auth(env)

    def test_bad_auth_mode(self):
        env = {"MODEL_PROVIDER": "anthropic", "MODEL_AUTH_MODE": "magic", "MODEL": "claude-sonnet-4-6"}
        with pytest.raises(AuthenticationFailure, match="api.*subscription"):
            _validate_provider_auth(env)

    def test_missing_model(self):
        env = {"MODEL_PROVIDER": "anthropic", "MODEL_AUTH_MODE": "api", "ANTHROPIC_API_KEY": "sk-test"}
        with pytest.raises(AuthenticationFailure, match="[Mm]odel"):
            _validate_provider_auth(env)


class TestBuildNodeEnv:
    def test_sets_defaults(self):
        result = _build_runner_env({}, 1800)
        assert result["LOG_LEVEL"] == "silent"
        assert result["IDLE_TIMEOUT"] == "60000"
        assert result["CONTAINER_TIMEOUT"] == str((1800 - 15) * 1000)

    def test_preserves_existing(self):
        result = _build_runner_env({"LOG_LEVEL": "debug"}, 1800)
        assert result["LOG_LEVEL"] == "debug"

    def test_zero_timeout_uses_default_1800(self):
        # 0 / absent keeps the historical 1800s hard container safety cap.
        result = _build_runner_env({}, 0)
        assert result["CONTAINER_TIMEOUT"] == str(1800 * 1000)

    def test_negative_timeout_disables_cap(self):
        # Negative = explicit opt-in to no hard cap (TB2/Harbor fairness).
        result = _build_runner_env({}, -1)
        assert result["CONTAINER_TIMEOUT"] == "0"

    def test_small_timeout_clamps(self):
        result = _build_runner_env({}, 20)
        assert result["CONTAINER_TIMEOUT"] == str(300 * 1000)


class TestAllowWebToolsEnvPropagation:
    """Issue #666: ``KSI_ALLOW_WEB_TOOLS`` must reach the runner env so
    ``container_runner.ts`` can forward it into the container, where the Claude
    agent-runner offers WebSearch/WebFetch on non-ARC benchmarks. Default OFF:
    absent on the host and absent from the provider profile => not set, so the
    agent-runner denies the web tools."""

    def test_default_off_not_present(self, monkeypatch):
        monkeypatch.delenv("KSI_ALLOW_WEB_TOOLS", raising=False)
        result = _build_runner_env({}, 1800)
        assert "KSI_ALLOW_WEB_TOOLS" not in result

    def test_from_host_environ(self, monkeypatch):
        monkeypatch.setenv("KSI_ALLOW_WEB_TOOLS", "1")
        result = _build_runner_env({}, 1800)
        assert result["KSI_ALLOW_WEB_TOOLS"] == "1"

    def test_provider_profile_value_preserved(self, monkeypatch):
        # Provider profile already merged into base_env; host env unset.
        monkeypatch.delenv("KSI_ALLOW_WEB_TOOLS", raising=False)
        result = _build_runner_env({"KSI_ALLOW_WEB_TOOLS": "1"}, 1800)
        assert result["KSI_ALLOW_WEB_TOOLS"] == "1"

    def test_provider_profile_wins_over_host(self, monkeypatch):
        # base_env (provider profile) takes precedence; setdefault is a no-op
        # when the key already exists.
        monkeypatch.setenv("KSI_ALLOW_WEB_TOOLS", "0")
        result = _build_runner_env({"KSI_ALLOW_WEB_TOOLS": "1"}, 1800)
        assert result["KSI_ALLOW_WEB_TOOLS"] == "1"


class TestOpenAIParityToolsEnvPropagation:
    """Issue #634: ``OPENAI_PARITY_TOOLS`` must reach the runner env so
    ``container_runner.ts`` can forward it into the container. Default OFF:
    absent on the host and absent from the provider profile => not set."""

    def test_default_off_not_present(self, monkeypatch):
        monkeypatch.delenv("OPENAI_PARITY_TOOLS", raising=False)
        result = _build_runner_env({}, 1800)
        assert "OPENAI_PARITY_TOOLS" not in result

    def test_from_host_environ(self, monkeypatch):
        monkeypatch.setenv("OPENAI_PARITY_TOOLS", "1")
        result = _build_runner_env({}, 1800)
        assert result["OPENAI_PARITY_TOOLS"] == "1"

    def test_provider_profile_value_preserved(self, monkeypatch):
        # Provider profile already merged into base_env; host env unset.
        monkeypatch.delenv("OPENAI_PARITY_TOOLS", raising=False)
        result = _build_runner_env({"OPENAI_PARITY_TOOLS": "1"}, 1800)
        assert result["OPENAI_PARITY_TOOLS"] == "1"

    def test_provider_profile_wins_over_host(self, monkeypatch):
        # base_env (provider profile) takes precedence; setdefault is a no-op
        # when the key already exists.
        monkeypatch.setenv("OPENAI_PARITY_TOOLS", "0")
        result = _build_runner_env({"OPENAI_PARITY_TOOLS": "1"}, 1800)
        assert result["OPENAI_PARITY_TOOLS"] == "1"


class TestDirectArcMaxTokensEnvPropagation:
    """Issue #682: ``KSI_DIRECT_ARC_MAX_TOKENS`` (shared per-turn output cap for
    both direct-ARC adapters) must reach the runner env so ``container_runner.ts``
    can forward it into the container. Default: absent on host + absent from
    provider profile => not set, and the adapters fall back to 4096."""

    def test_default_absent_not_present(self, monkeypatch):
        monkeypatch.delenv("KSI_DIRECT_ARC_MAX_TOKENS", raising=False)
        result = _build_runner_env({}, 1800)
        assert "KSI_DIRECT_ARC_MAX_TOKENS" not in result

    def test_from_host_environ(self, monkeypatch):
        monkeypatch.setenv("KSI_DIRECT_ARC_MAX_TOKENS", "8192")
        result = _build_runner_env({}, 1800)
        assert result["KSI_DIRECT_ARC_MAX_TOKENS"] == "8192"

    def test_provider_profile_value_preserved(self, monkeypatch):
        monkeypatch.delenv("KSI_DIRECT_ARC_MAX_TOKENS", raising=False)
        result = _build_runner_env({"KSI_DIRECT_ARC_MAX_TOKENS": "2048"}, 1800)
        assert result["KSI_DIRECT_ARC_MAX_TOKENS"] == "2048"

    def test_provider_profile_wins_over_host(self, monkeypatch):
        monkeypatch.setenv("KSI_DIRECT_ARC_MAX_TOKENS", "8192")
        result = _build_runner_env({"KSI_DIRECT_ARC_MAX_TOKENS": "2048"}, 1800)
        assert result["KSI_DIRECT_ARC_MAX_TOKENS"] == "2048"

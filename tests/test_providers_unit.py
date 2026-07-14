"""Unit tests for kcsi.providers -- provider profile loading and validation."""

from __future__ import annotations

import pytest

from kcsi.providers import ProviderConfigError, load_provider_profile


class TestLoadProviderProfile:
    def test_missing_file_raises(self):
        with pytest.raises(ProviderConfigError, match="not found"):
            load_provider_profile("/nonexistent/path.env")

    def test_valid_anthropic_api_profile(self, tmp_path):
        p = tmp_path / ".env.test"
        p.write_text(
            "MODEL_PROVIDER=anthropic\n"
            "MODEL=claude-sonnet-4-20250514\n"
            "MODEL_AUTH_MODE=api\n"
            "ANTHROPIC_API_KEY=sk-test-key-123\n"
        )
        cfg = load_provider_profile(str(p))
        assert cfg["MODEL_PROVIDER"] == "anthropic"
        assert cfg["MODEL"] == "claude-sonnet-4-20250514"
        assert cfg["ANTHROPIC_API_KEY"] == "sk-test-key-123"

    def test_valid_anthropic_oauth_profile(self, tmp_path):
        p = tmp_path / ".env.test"
        p.write_text(
            "MODEL_PROVIDER=anthropic\nMODEL=claude-sonnet-4-20250514\nCLAUDE_CODE_OAUTH_TOKEN=oauth-token-abc\n"
        )
        cfg = load_provider_profile(str(p))
        assert cfg["MODEL_PROVIDER"] == "anthropic"
        assert cfg["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-token-abc"

    def test_valid_openai_profile_with_reasoning_effort(self, tmp_path):
        p = tmp_path / ".env.test"
        p.write_text(
            "MODEL_PROVIDER=openai\n"
            "MODEL=gpt-5.4-mini\n"
            "MODEL_AUTH_MODE=api\n"
            "OPENAI_API_KEY=sk-test-key-123\n"
            "REASONING_EFFORT=none\n"
            "KCSI_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2\n"
        )
        cfg = load_provider_profile(str(p))
        assert cfg["MODEL_PROVIDER"] == "openai"
        assert cfg["MODEL"] == "gpt-5.4-mini"
        assert cfg["MODEL_AUTH_MODE"] == "api"
        assert cfg["OPENAI_API_KEY"] == "sk-test-key-123"
        assert cfg["REASONING_EFFORT"] == "none"
        assert cfg["KCSI_EMBEDDING_MODEL"] == "sentence-transformers/all-MiniLM-L6-v2"

    def test_missing_provider_raises(self, tmp_path):
        p = tmp_path / ".env.test"
        p.write_text("MODEL=claude-sonnet-4-20250514\nANTHROPIC_API_KEY=sk-test\n")
        with pytest.raises(ProviderConfigError, match="Missing MODEL_PROVIDER"):
            load_provider_profile(str(p))

    def test_unsupported_provider_raises(self, tmp_path):
        p = tmp_path / ".env.test"
        p.write_text("MODEL_PROVIDER=google\nMODEL=gemini-pro\nANTHROPIC_API_KEY=sk-test\n")
        with pytest.raises(ProviderConfigError, match="Unsupported MODEL_PROVIDER"):
            load_provider_profile(str(p))

    def test_missing_credentials_raises(self, tmp_path):
        p = tmp_path / ".env.test"
        p.write_text("MODEL_PROVIDER=anthropic\nMODEL=claude-sonnet-4-20250514\n")
        with pytest.raises(ProviderConfigError, match="No credentials found"):
            load_provider_profile(str(p))

    def test_missing_model_raises(self, tmp_path):
        p = tmp_path / ".env.test"
        p.write_text("MODEL_PROVIDER=anthropic\nANTHROPIC_API_KEY=sk-test-key\n")
        with pytest.raises(ProviderConfigError, match="Missing MODEL"):
            load_provider_profile(str(p))

    def test_placeholder_api_key_rejected(self, tmp_path):
        p = tmp_path / ".env.test"
        p.write_text("MODEL_PROVIDER=anthropic\nMODEL=claude-sonnet-4-20250514\nANTHROPIC_API_KEY=<your-api-key>\n")
        with pytest.raises(ProviderConfigError, match="No credentials found"):
            load_provider_profile(str(p))

    def test_optional_passthrough_keys(self, tmp_path):
        p = tmp_path / ".env.test"
        p.write_text(
            "MODEL_PROVIDER=anthropic\n"
            "MODEL=claude-sonnet-4-20250514\n"
            "MODEL_AUTH_MODE=api\n"
            "ANTHROPIC_API_KEY=sk-test-key\n"
            "KCSI_DISABLE_VECTOR=1\n"
            "MEMORY_ENABLE_SEMANTIC_SEARCH=0\n"
            "TOKENIZERS_PARALLELISM=false\n"
            "KCSI_OPENAI_MAX_TURNS=40\n"
            "OPENAI_AGENTS_DISABLE_TRACING=1\n"
        )
        cfg = load_provider_profile(str(p))
        assert cfg["KCSI_DISABLE_VECTOR"] == "1"
        assert cfg["MEMORY_ENABLE_SEMANTIC_SEARCH"] == "0"
        assert cfg["TOKENIZERS_PARALLELISM"] == "false"
        assert cfg["KCSI_OPENAI_MAX_TURNS"] == "40"
        assert cfg["OPENAI_AGENTS_DISABLE_TRACING"] == "1"

    def test_optional_passthrough_keys_from_host_env(self, tmp_path, monkeypatch):
        p = tmp_path / ".env.test"
        p.write_text("MODEL_PROVIDER=openai\nMODEL=gpt-5.4-mini\nMODEL_AUTH_MODE=api\nOPENAI_API_KEY=sk-test-key-123\n")
        monkeypatch.setenv("KCSI_DISABLE_VECTOR", "1")
        monkeypatch.setenv("MEMORY_ENABLE_SEMANTIC_SEARCH", "0")
        cfg = load_provider_profile(str(p))
        assert cfg["KCSI_DISABLE_VECTOR"] == "1"
        assert cfg["MEMORY_ENABLE_SEMANTIC_SEARCH"] == "0"

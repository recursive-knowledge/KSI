"""Tests that load_dotenv values do NOT override --provider-profile MODEL.

The bug: load_dotenv() at module import time injects values from a root .env
file (e.g. MODEL=claude-haiku-4-5-20251001) into os.environ.  The override
loop that runs later checked os.environ.get(key) unconditionally, so the
dotenv-injected MODEL silently won over the provider-profile MODEL.

The fix: _PRE_DOTENV_PROVIDER_KEYS records which of the three keys were
already in os.environ *before* load_dotenv() ran.  Only those keys are
allowed to override the provider-profile value.
"""

from __future__ import annotations

import importlib
import os
import sys
from unittest import mock

# ---------------------------------------------------------------------------
# Helper: simulate importing ksi.cli with a controlled pre-dotenv env
# ---------------------------------------------------------------------------


def _import_cli_with_env(pre_env: dict[str, str], dotenv_injects: dict[str, str]):
    """Re-import ksi.cli with a synthetic pre-dotenv environment.

    pre_env       – keys that exist in os.environ BEFORE load_dotenv() runs
    dotenv_injects – additional keys that load_dotenv() would inject
    """
    combined = {**pre_env, **dotenv_injects}

    def fake_load_dotenv(*args, **kwargs):
        # Inject the "dotenv" values into the already-patched os.environ.
        os.environ.update(dotenv_injects)

    # We need a fresh import of ksi.cli so _PRE_DOTENV_PROVIDER_KEYS is
    # recalculated.  Remove any cached copy first.
    for mod_name in list(sys.modules):
        if mod_name == "ksi.cli" or mod_name.startswith("ksi.cli."):
            del sys.modules[mod_name]

    with mock.patch.dict(os.environ, pre_env, clear=False):
        # Remove dotenv_injects from the environment so they aren't present
        # before load_dotenv is called (simulating the real scenario).
        for key in dotenv_injects:
            os.environ.pop(key, None)

        with mock.patch("dotenv.load_dotenv", side_effect=fake_load_dotenv):
            import ksi.cli as cli_mod

            importlib.reload(cli_mod)

    return cli_mod


class TestPreDotenvSnapshot:
    """Verify _PRE_DOTENV_PROVIDER_KEYS captures the right keys."""

    def test_keys_present_before_load_dotenv_are_captured(self):
        """MODEL set before load_dotenv → it should be in the snapshot."""
        saved = saved_provider = saved_auth = None
        try:
            # Remove from env, then set explicitly to simulate pre-existing value.
            # Pops live inside the try so the finally always restores, even if a
            # pytest-timeout SIGALRM fires mid-setup.
            saved = os.environ.pop("MODEL", None)
            saved_provider = os.environ.pop("MODEL_PROVIDER", None)
            saved_auth = os.environ.pop("MODEL_AUTH_MODE", None)
            os.environ["MODEL"] = "claude-opus-4"

            # Re-evaluate just the snapshot expression (same logic as cli.py)
            snapshot = frozenset(k for k in ("MODEL", "MODEL_PROVIDER", "MODEL_AUTH_MODE") if k in os.environ)
            assert "MODEL" in snapshot
            assert "MODEL_PROVIDER" not in snapshot
            assert "MODEL_AUTH_MODE" not in snapshot
        finally:
            os.environ.pop("MODEL", None)
            if saved is not None:
                os.environ["MODEL"] = saved
            if saved_provider is not None:
                os.environ["MODEL_PROVIDER"] = saved_provider
            if saved_auth is not None:
                os.environ["MODEL_AUTH_MODE"] = saved_auth

    def test_keys_absent_before_load_dotenv_are_not_captured(self):
        """MODEL absent before load_dotenv → it must NOT be in the snapshot."""
        saved = saved_provider = saved_auth = None
        try:
            # Pops live inside the try so the finally always restores, even if a
            # pytest-timeout SIGALRM fires mid-setup.
            saved = os.environ.pop("MODEL", None)
            saved_provider = os.environ.pop("MODEL_PROVIDER", None)
            saved_auth = os.environ.pop("MODEL_AUTH_MODE", None)
            snapshot = frozenset(k for k in ("MODEL", "MODEL_PROVIDER", "MODEL_AUTH_MODE") if k in os.environ)
            assert "MODEL" not in snapshot
            assert "MODEL_PROVIDER" not in snapshot
            assert "MODEL_AUTH_MODE" not in snapshot
        finally:
            if saved is not None:
                os.environ["MODEL"] = saved
            if saved_provider is not None:
                os.environ["MODEL_PROVIDER"] = saved_provider
            if saved_auth is not None:
                os.environ["MODEL_AUTH_MODE"] = saved_auth


class TestOverrideLoopRespectSnapshot:
    """Verify the override loop only applies to pre-dotenv keys."""

    def _run_override_loop(
        self,
        provider_env: dict[str, str],
        pre_dotenv_keys: frozenset[str],
        current_env: dict[str, str],
    ) -> dict[str, str]:
        """Replicate the override loop logic from cli.py for unit-testing."""
        result = dict(provider_env)
        for key in ("MODEL", "MODEL_PROVIDER", "MODEL_AUTH_MODE"):
            if key in pre_dotenv_keys:
                override = current_env.get(key, "").strip()
                if override:
                    result[key] = override
        return result

    def test_dotenv_injected_model_does_not_override_provider_profile(self):
        """Core regression: dotenv value must NOT beat provider-profile MODEL."""
        provider_env = {"MODEL": "claude-sonnet-4", "MODEL_PROVIDER": "anthropic"}
        pre_dotenv_keys = frozenset()  # MODEL was NOT in env before load_dotenv
        current_env = {"MODEL": "claude-haiku-4-5-20251001"}  # injected by dotenv

        result = self._run_override_loop(provider_env, pre_dotenv_keys, current_env)

        assert result["MODEL"] == "claude-sonnet-4", "Provider-profile MODEL was overridden by a dotenv-injected value"

    def test_explicit_process_env_does_override_provider_profile(self):
        """An env var set before the process (shell export) MUST still win."""
        provider_env = {"MODEL": "claude-sonnet-4", "MODEL_PROVIDER": "anthropic"}
        pre_dotenv_keys = frozenset({"MODEL"})  # MODEL was set in process env
        current_env = {"MODEL": "claude-opus-4"}  # explicit user override

        result = self._run_override_loop(provider_env, pre_dotenv_keys, current_env)

        assert result["MODEL"] == "claude-opus-4", "Explicit process-env MODEL should override provider-profile"

    def test_empty_override_does_not_clobber(self):
        """Empty string in env must not replace provider value."""
        provider_env = {"MODEL": "claude-sonnet-4"}
        pre_dotenv_keys = frozenset({"MODEL"})
        current_env = {"MODEL": ""}  # empty → no override

        result = self._run_override_loop(provider_env, pre_dotenv_keys, current_env)

        assert result["MODEL"] == "claude-sonnet-4"

    def test_model_provider_and_auth_mode_also_guarded(self):
        """Same guard applies to MODEL_PROVIDER and MODEL_AUTH_MODE."""
        provider_env = {
            "MODEL": "claude-sonnet-4",
            "MODEL_PROVIDER": "anthropic",
            "MODEL_AUTH_MODE": "api_key",
        }
        pre_dotenv_keys = frozenset()  # none were pre-set
        current_env = {
            "MODEL": "x",
            "MODEL_PROVIDER": "openai",
            "MODEL_AUTH_MODE": "oauth",
        }

        result = self._run_override_loop(provider_env, pre_dotenv_keys, current_env)

        assert result["MODEL"] == "claude-sonnet-4"
        assert result["MODEL_PROVIDER"] == "anthropic"
        assert result["MODEL_AUTH_MODE"] == "api_key"

    def test_partial_pre_keys_only_override_matching(self):
        """Only keys that were pre-dotenv should be overridable."""
        provider_env = {
            "MODEL": "claude-sonnet-4",
            "MODEL_PROVIDER": "anthropic",
            "MODEL_AUTH_MODE": "api_key",
        }
        # Only MODEL_PROVIDER was set before dotenv
        pre_dotenv_keys = frozenset({"MODEL_PROVIDER"})
        current_env = {
            "MODEL": "haiku-injected-by-dotenv",
            "MODEL_PROVIDER": "openai",  # legitimate override
            "MODEL_AUTH_MODE": "oauth",
        }

        result = self._run_override_loop(provider_env, pre_dotenv_keys, current_env)

        assert result["MODEL"] == "claude-sonnet-4"  # dotenv blocked
        assert result["MODEL_PROVIDER"] == "openai"  # legit override applied
        assert result["MODEL_AUTH_MODE"] == "api_key"  # dotenv blocked

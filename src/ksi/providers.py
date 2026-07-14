from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

from .errors import KsiError


class ProviderConfigError(KsiError, RuntimeError):
    pass


_PROFILE_MOVES = (("configs/providers", "configs/ksi"),)


def _profile_migration_hint(requested: str, resolved: Path) -> str:
    """Hint when a missing profile path is an artifact of the configs reorg.

    Provider profiles moved ``configs/providers`` → ``configs/ksi``. Covers
    both a stale old-location path AND the case where the user's (untracked)
    real-key profile is still at the old location while they pass the new path.
    """
    for old, new in _PROFILE_MOVES:
        if old in requested:
            return f" — provider profiles moved {old} → {new}; try {requested.replace(old, new)}"
    resolved_str = str(resolved)
    for old, new in _PROFILE_MOVES:
        if f"/{new}/" in resolved_str:
            legacy = Path(resolved_str.replace(f"/{new}/", f"/{old}/"))
            if legacy.exists():
                return (
                    f" — found a profile at the old location {legacy}; provider profiles "
                    f"moved to {new}/ (move or copy your .env.* there)"
                )
    return ""


def load_provider_profile(profile_path: str) -> dict[str, str]:
    """Load provider config from a dotenv file and normalize env keys.

    Required for the container runtime to avoid accidental fallback to ambient .env.
    """
    p = Path(profile_path).expanduser().resolve()
    if not p.exists():
        raise ProviderConfigError(f"Provider profile not found: {p}{_profile_migration_hint(profile_path, p)}")

    raw = dotenv_values(p)
    cfg = {k: str(v) for k, v in raw.items() if isinstance(v, (str, int, float)) and str(v).strip()}

    provider_raw = cfg.get("MODEL_PROVIDER", "").strip() or cfg.get("LLM_PROVIDER", "").strip()
    if not provider_raw:
        raise ProviderConfigError("Missing MODEL_PROVIDER in provider profile. (Legacy LLM_PROVIDER is also accepted.)")
    provider = provider_raw.lower()
    model = cfg.get("MODEL", "").strip() or cfg.get("LLM_MODEL", "").strip()
    auth_mode = cfg.get("MODEL_AUTH_MODE", "").strip().lower()

    if provider not in {"anthropic", "openai"}:
        raise ProviderConfigError(f"Unsupported MODEL_PROVIDER={provider!r}. Supported providers: anthropic, openai.")

    # Resolve credentials based on provider.
    api_key = ""
    oauth_token = ""
    if provider == "openai":
        api_key = cfg.get("OPENAI_API_KEY", "").strip()
    else:
        api_key = cfg.get("ANTHROPIC_API_KEY", "").strip() or cfg.get("LLM_API_KEY", "").strip()
        oauth_token = cfg.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    # Reject obvious placeholder values left by setup_all.sh templates.
    if api_key and ("<" in api_key or api_key == "your-api-key"):
        api_key = ""
    if not api_key and not oauth_token:
        key_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
        raise ProviderConfigError(f"No credentials found. Set {key_name} in your provider profile.")

    if not model:
        raise ProviderConfigError("Missing MODEL in provider profile.")

    out: dict[str, str] = {"MODEL_PROVIDER": provider, "MODEL": model, "MODEL_AUTH_MODE": auth_mode}

    if provider == "anthropic":
        if auth_mode == "api":
            api_key = cfg.get("ANTHROPIC_API_KEY", "").strip()
            if not api_key:
                raise ProviderConfigError("anthropic/api mode requires ANTHROPIC_API_KEY.")
            out["ANTHROPIC_API_KEY"] = api_key
        else:
            token = cfg.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
            if not token:
                raise ProviderConfigError("anthropic/subscription mode requires CLAUDE_CODE_OAUTH_TOKEN.")
            out["CLAUDE_CODE_OAUTH_TOKEN"] = token
            # Pass through ANTHROPIC_API_KEY if present — orchestrator-side
            # Messages API calls (forum, claiming, sweep) require an API key.
            api_key = cfg.get("ANTHROPIC_API_KEY", "").strip()
            if api_key and "<" not in api_key and api_key != "your-api-key":
                out["ANTHROPIC_API_KEY"] = api_key
    elif provider == "openai":
        if auth_mode != "api":
            raise ProviderConfigError("openai supports MODEL_AUTH_MODE=api only.")
        key = cfg.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise ProviderConfigError("openai/api mode requires OPENAI_API_KEY.")
        out["OPENAI_API_KEY"] = key

    # Optional pass-throughs.
    for key in (
        "REASONING_EFFORT",
        "KSI_DISABLE_VECTOR",
        "MEMORY_ENABLE_SEMANTIC_SEARCH",
        "KSI_EMBEDDING_MODEL",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "USE_TF",
        "TOKENIZERS_PARALLELISM",
        "KSI_OPENAI_MAX_TURNS",
        "OPENAI_AGENTS_DISABLE_TRACING",
    ):
        value = cfg.get(key, "").strip() or os.environ.get(key, "").strip()
        if value:
            out[key] = value

    return out


def apply_provider_env(env_map: dict[str, str]) -> None:
    """Apply normalized provider env values to current process env."""
    for k, v in env_map.items():
        os.environ[k] = v

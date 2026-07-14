from __future__ import annotations

from pathlib import Path

import pytest

from ksi.models import GenerationConfig


def test_programmatic_api_generation_config_requires_core_counts():
    guide = Path("docs/programmatic_api.md").read_text()

    assert "A bare `GenerationConfig()`" not in guide
    assert "`GenerationConfig` requires `num_generations` and `num_agents`" in guide
    assert "`GenerationConfig(num_generations=1, num_agents=1)`" in guide


def test_generation_config_bare_construction_raises():
    """Behavioral backing for the doc claim: a bare ``GenerationConfig()``
    genuinely fails because ``num_generations``/``num_agents`` are required.

    Without this, the doc-string pin above would stay green even if either
    field grew a default -- the exact doc/code drift PR #1077 set out to fix.
    """
    with pytest.raises(TypeError):
        GenerationConfig()


def test_generation_config_with_core_counts_succeeds():
    """The doc's minimal example must actually construct."""
    cfg = GenerationConfig(num_generations=1, num_agents=1)
    assert cfg.num_generations == 1
    assert cfg.num_agents == 1

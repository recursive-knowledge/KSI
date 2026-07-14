"""Opt-in LIVE round-trip smoke for the bumped ``claude-agent-sdk`` (issue #947).

Skipped by default. This makes a REAL model call, so it requires both an
explicit opt-in and credentials:

  - ``KSI_LIVE_SDK_TEST=1`` to opt in, and
  - ``CLAUDE_CODE_OAUTH_TOKEN`` (subscription) or ``ANTHROPIC_API_KEY`` (API).

Run it after bumping the SDK to confirm the new version can still drive a model
end to end::

    KSI_LIVE_SDK_TEST=1 uv run pytest tests/runtime/test_sdk_live_roundtrip.py -v

What it exercises: the Python host's ``query()`` path
(``ksi.tasks.loaders._run_agent_sdk_query``), which spawns the SDK's bundled
Claude Code CLI and round-trips a single turn through the model. There is no
SDK-level wire protocol *between* the Python host and the TS agent-runner — each
side drives its own bundled CLI — so this smoke covers the Python side, while the
TS agent-runner (pinned to the same version, see
``test_sdk_version_lockstep.py``) is exercised by the container integration path.
"""

from __future__ import annotations

import os

import pytest

_OPT_IN = os.environ.get("KSI_LIVE_SDK_TEST", "").strip().lower() in {"1", "true", "yes", "on"}
_HAS_CREDS = bool(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY"))

pytestmark = [
    pytest.mark.skipif(not _OPT_IN, reason="set KSI_LIVE_SDK_TEST=1 to run the live SDK round-trip"),
    pytest.mark.skipif(
        not _HAS_CREDS,
        reason="needs CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY for the live SDK round-trip",
    ),
]


def test_live_sdk_query_roundtrip() -> None:
    from ksi.tasks.loaders import _run_agent_sdk_query

    model = os.environ.get("MODEL") or None
    response = _run_agent_sdk_query(
        "Reply with exactly the word: pong",
        model,
        system_prompt="You are a terse echo service. Reply with a single word.",
    )

    assert isinstance(response, str), f"expected a string response, got {type(response)!r}"
    assert response.strip(), "live SDK round-trip returned an empty response"

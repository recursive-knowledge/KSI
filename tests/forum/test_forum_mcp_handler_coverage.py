"""Handler-coverage guard for the forum MCP tool surface (issue #693).

Sibling of ``test_arc_mcp_handler_coverage.py``. The JS drift guard in
``tests/js/anthropic_scheduled_stream.test.mjs`` only pins the
``DIRECT_FORUM_TOOL_ALLOWLIST`` Set in
``runtime_runner/agent-runner/src/anthropic_direct_forum.ts`` against a hardcoded
literal; it does not reach the boundary that caused #693: a forum tool advertised
to the agent (in the Set) with no dispatch handler / definition in
``src/kcsi/memory/mcp_server.py`` would fall through to ``Unknown tool`` at runtime.
The request-body assertion that would otherwise catch this lives in a
``tsx``-gated test that *skips* when ``tsx`` is absent (e.g. minimal CI), so this
module asserts the coverage directly and dependency-free: every allowlisted forum
tool must have both a Python handler and a definition.
"""

from __future__ import annotations

import re

from conftest import REPO_ROOT

FORUM_TS = REPO_ROOT / "runtime_runner" / "agent-runner" / "src" / "anthropic_direct_forum.ts"
MCP_SERVER = REPO_ROOT / "src" / "kcsi" / "memory" / "mcp_server.py"


def _strip_line_comments(text: str, marker: str) -> str:
    """Drop ``marker``-to-EOL comments so a commented-out entry can't be parsed as
    live source: ``//`` for the TS Set, ``#`` for the Python server."""
    return re.sub(rf"{re.escape(marker)}[^\n]*", "", text)


def _forum_allowlist() -> set[str]:
    """Forum tool names the agent is allowed to call (the authoritative Set)."""
    src = FORUM_TS.read_text(encoding="utf-8")
    m = re.search(r"const DIRECT_FORUM_TOOL_ALLOWLIST\s*=\s*new Set\(\[([\s\S]*?)\]\)", src)
    assert m, "anthropic_direct_forum.ts must declare DIRECT_FORUM_TOOL_ALLOWLIST"
    names = re.findall(r"'([^']+)'", _strip_line_comments(m.group(1), "//"))
    assert names, "DIRECT_FORUM_TOOL_ALLOWLIST parsed empty"
    return set(names)


def _mcp_server_handlers() -> set[str]:
    """Tool names with a ``tool_name == "..."`` dispatch branch."""
    src = _strip_line_comments(MCP_SERVER.read_text(encoding="utf-8"), "#")
    return set(re.findall(r'tool_name\s*==\s*"([a-z][a-z0-9_]*)"', src))


def _mcp_server_definitions() -> set[str]:
    """Tool names advertised in the mcp_server.py ``"name": "..."`` lists."""
    src = _strip_line_comments(MCP_SERVER.read_text(encoding="utf-8"), "#")
    return set(re.findall(r'"name":\s*"([a-z][a-z0-9_]*)"', src))


def test_every_forum_tool_has_a_python_handler() -> None:
    allow = _forum_allowlist()
    handlers = _mcp_server_handlers()
    missing = allow - handlers
    assert not missing, (
        f"forum tools in DIRECT_FORUM_TOOL_ALLOWLIST with no dispatch handler in mcp_server.py: {sorted(missing)}"
    )


def test_every_forum_tool_is_advertised_in_mcp_server() -> None:
    allow = _forum_allowlist()
    defined = _mcp_server_definitions()
    missing = allow - defined
    assert not missing, (
        f"forum tools in DIRECT_FORUM_TOOL_ALLOWLIST not defined in the mcp_server.py tool list: {sorted(missing)}"
    )

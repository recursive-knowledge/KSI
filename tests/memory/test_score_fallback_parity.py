"""Parity guard for the inline ``score_from_eval_results`` fallback in store.py.

``src/ksi/memory/store.py`` imports the canonical
``ksi.orchestrator.scoring.score_from_eval_results`` but defines an INLINE
fallback copy under ``except Exception`` (with the same precedence chain), used
in container script-mode where the parent ``..orchestrator`` package is not
importable. Because CI always takes the real-import path, that fallback is
``# pragma: no cover`` and can silently DRIFT from the canonical scorer —
producing different scores from the in-container MCP server.

Approach (no source change): the fallback is defined inside a ``try/except`` at
import time, so when the real import succeeds the fallback object is never bound
at module scope. We therefore parse ``store.py``'s source with ``ast``, extract
just the nested ``def score_from_eval_results`` block, ``exec`` it in an isolated
namespace, and assert it returns IDENTICAL results to the canonical across a
representative table of eval dicts. Any future edit to one scorer without the
other fails this test.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Callable

import pytest

import ksi.memory.store as store_mod
from ksi.orchestrator.scoring import score_from_eval_results as canonical


def _extract_fallback_scorer() -> Callable[[dict[str, Any]], "float | None"]:
    """Return the inline fallback ``score_from_eval_results`` from store.py.

    Locate the ``def score_from_eval_results`` node in store.py's AST (it lives
    inside the import-fallback ``except`` handler and is never bound at module
    scope when the real import succeeds), extract its source segment, and exec it
    in a fresh namespace.
    """
    source = Path(store_mod.__file__).read_text()
    tree = ast.parse(source)
    func_node = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "score_from_eval_results"),
        None,
    )
    assert func_node is not None, "inline fallback score_from_eval_results not found in store.py"
    func_src = ast.get_source_segment(source, func_node)
    assert func_src is not None
    # Prepend the future-annotations import so the ``dict[str, Any]`` / ``float |
    # None`` annotations are not evaluated at def time in the isolated namespace.
    ns: dict[str, Any] = {}
    exec("from __future__ import annotations\n" + func_src, ns)  # noqa: S102
    return ns["score_from_eval_results"]


_FALLBACK = _extract_fallback_scorer()


# (eval_results dict, expected score) covering every branch of the canonical
# precedence chain plus the precedence ordering between branches.
_CASES: list[tuple[dict[str, Any], "float | None"]] = [
    # 1. native_score (numeric only)
    ({"native_score": 0.5}, 0.5),
    ({"native_score": 1}, 1.0),
    ({"native_score": 0}, 0.0),
    ({"native_score": -0.25}, -0.25),
    ({"native_score": "0.5"}, None),  # non-numeric falls through, nothing else -> None
    # 2. resolved (bool only)
    ({"resolved": True}, 1.0),
    ({"resolved": False}, 0.0),
    ({"resolved": None}, None),  # non-bool falls through
    ({"resolved": "false"}, None),  # truthy string must NOT be trusted
    # 3. instance_report.resolved (bool only)
    ({"instance_report": {"resolved": True}}, 1.0),
    ({"instance_report": {"resolved": False}}, 0.0),
    ({"instance_report": {"resolved": "nope"}}, None),
    ({"instance_report": {}}, None),
    ({"instance_report": "notadict"}, None),
    # 4. pass (bool-ish)
    ({"pass": True}, 1.0),
    ({"pass": False}, 0.0),
    ({"pass": 1}, 1.0),
    ({"pass": 0}, 0.0),
    ({"pass": None}, None),  # explicit None ignored
    # empty / fallthrough
    ({}, None),
    # precedence ordering
    ({"native_score": 0.3, "resolved": False}, 0.3),  # native_score wins
    ({"resolved": True, "instance_report": {"resolved": False}, "pass": False}, 1.0),  # resolved wins
    ({"instance_report": {"resolved": True}, "pass": False}, 1.0),  # instance_report wins over pass
]


@pytest.mark.parametrize(("eval_r", "expected"), _CASES)
def test_canonical_score_matches_expected(eval_r: dict[str, Any], expected: "float | None") -> None:
    """Pin the canonical scorer's behavior across the representative table."""
    assert canonical(eval_r) == expected


@pytest.mark.parametrize(("eval_r", "expected"), _CASES)
def test_fallback_matches_canonical(eval_r: dict[str, Any], expected: "float | None") -> None:
    """The store.py inline fallback must match the canonical scorer exactly.

    This is the drift guard: if either scorer's precedence logic is edited
    without the other, this assertion fails in CI.
    """
    fallback_result = _FALLBACK(eval_r)
    assert fallback_result == canonical(eval_r)
    assert fallback_result == expected

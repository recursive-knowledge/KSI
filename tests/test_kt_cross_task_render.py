"""End-to-end render test for KT cross-task bundle injection.

Regression test for the rendering-bug fix: a structured Insight dict in
the on-disk bundle JSON must reach ``_render_bundle_item`` as a dict (not
flattened to ``str(dict)``) so the rendered prompt block has the
``(confidence)``/``Applies when:``/``NOT when:``/``Evidence:`` markdown
shape rather than truncated Python repr.

Covers the integration point that ``test_extract_knowledge.py``
(extractor unit) and ``test_seed_package_memory_md.py`` (renderer unit
with bare-string items only) leave untested.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from ksi.runtime.seeding import _render_bundle_item


def test_render_bundle_item_structured_dict_produces_clean_markdown(structured_insight):
    """Structured Insight dict renders with stable line shape:
    one ``- (high) <text>`` line + indented child lines for boundaries
    and evidence. Asserts on line position, not just substring presence,
    so a future change that leaks ``Applies when:`` onto the wrong line
    fails this test.
    """
    item = structured_insight
    lines = _render_bundle_item(item, max_chars=300)

    assert lines[0].startswith("- (high) When a full-width zero band splits the grid")
    assert "{'text':" not in lines[0]

    # Boundary child lines are indented two spaces and exactly labelled.
    boundary_lines = {line.split(":", 1)[0].strip(): line for line in lines if ":" in line}
    assert "Applies when" in boundary_lines
    assert boundary_lines["Applies when"].startswith("  Applies when: ")
    assert "NOT when" in boundary_lines
    assert boundary_lines["NOT when"].startswith("  NOT when: ")
    assert "Evidence" in boundary_lines
    evidence_line = boundary_lines["Evidence"]
    assert evidence_line.startswith("  Evidence: ")
    assert "post#42" in evidence_line
    assert "post#91" in evidence_line
    assert "abc123" in evidence_line
    assert "def456" in evidence_line


def test_render_bundle_item_truncates_long_text_at_max_chars(structured_insight):
    """Paired truncation-positive test: with a tight ``max_chars`` we
    verify ``(truncated)`` actually fires when the text exceeds the cap.
    Pairs with the negative assertion in the structured-dict test
    (which uses ``max_chars=300`` against an ~80-char text).
    """
    long_item = dict(structured_insight)
    long_item["text"] = "X" * 500
    lines = _render_bundle_item(long_item, max_chars=40)
    rendered = "\n".join(lines)
    assert "(truncated)" in rendered


def test_render_bundle_item_legacy_stringified_dict_renders_as_string(structured_insight):
    """A legacy bundle that pre-flattened insights to ``str(dict)`` still
    renders -- the string-fallback path of ``_render_bundle_item`` produces a
    single ``- <text>`` line capped at ``max_chars``. The rendering will
    look ugly (Python repr fragment) but the runtime must not crash.
    """
    legacy = str(structured_insight)
    lines = _render_bundle_item(legacy, max_chars=300)
    rendered = "\n".join(lines)
    assert rendered.startswith("- ")
    # No structured fields parsed back out of the string repr.
    assert "Applies when:" not in rendered
    assert "Evidence:" not in rendered


def test_render_bundle_item_bare_string_back_compat():
    """Bare-string items (older donor bundles, hand-built fixtures) still
    render unchanged: a single ``- <text>`` line."""
    lines = _render_bundle_item("Use BFS flood-fill for connected-component extraction.", max_chars=300)
    assert lines == ["- Use BFS flood-fill for connected-component extraction."]


def test_render_bundle_item_dict_missing_text_drops_item():
    """An Insight dict without a ``text`` field returns no lines (caller
    skips). Defensive against half-populated donor outputs."""
    item = {"applies_when": "anywhere", "evidence": [{"post_id": 1}]}
    assert _render_bundle_item(item, max_chars=300) == []


def test_render_bundle_item_unknown_confidence_falls_through(structured_insight):
    """Confidence other than high/medium/low: prefix is dropped (no
    ``(garbage) `` leaks into the rendered output)."""
    item = dict(structured_insight)
    item["confidence"] = "speculative"
    lines = _render_bundle_item(item, max_chars=300)
    assert lines[0].startswith("- When a full-width zero band splits the grid")
    assert "(speculative)" not in lines[0]


def _build_orchestrator_with_two_agents(no_memory: bool = False):
    """Construct a real ``GenerationalOrchestrator`` with two stub agents.

    Mirrors the pattern in ``tests/test_seed_bundle_injection.py``:
    real ``__init__`` with mocked runtime/evaluator/llm so we exercise
    the actual factory wiring rather than ``__new__`` + duck-typing
    (which silently breaks if ``__init__`` adds new required state).

    Adaptation: the canonical config class is ``GenerationConfig`` from
    ``ksi.models`` (not ``OrchestratorConfig`` -- the latter does not
    exist in this repo). We construct a 2-agent config and then replace
    ``orch.agents`` with stubs whose ``seed_package`` defaults to
    ``None`` so we can assert "untouched" cleanly.
    """
    from ksi.models import GenerationConfig
    from ksi.orchestrator.engine import GenerationalOrchestrator

    config = GenerationConfig(
        num_generations=1,
        num_agents=2,
        no_memory=no_memory,
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=MagicMock(),
        evaluator=MagicMock(),
        llm=MagicMock(),
    )

    class _StubAgent:
        seed_package: dict | None = None
        workstream = ""
        workstream_description = ""

    orch.agents = [_StubAgent(), _StubAgent()]
    return orch


def test_inject_seed_bundle_structured_round_trip(tmp_path: Path, structured_insight):
    """End-to-end: bundle JSON on disk -> ``_inject_seed_bundle`` -> every
    agent's ``seed_package["cross_task_bundle"]`` carries the original
    structured Insight dicts (no string-flattening) and downstream rendering
    produces the clean markdown shape.
    """
    bundle_path = tmp_path / "donor.json"
    bundle = {
        "meta": {
            "source_experiment": "donor-test",
            "generation_extracted": 3,
        },
        "cross_task": {
            "transferable_insights": [structured_insight],
            "confirmed_constraints": [],
            "rejected_hypotheses": [],
            "pitfalls": [],
            "checks": [],
            "next_steps": [],
            "evidence_post_ids": [42, 91],
        },
    }
    bundle_path.write_text(json.dumps(bundle))

    orch = _build_orchestrator_with_two_agents()
    orch._inject_seed_bundle(str(bundle_path))

    for agent in orch.agents:
        seed_pkg = agent.seed_package
        ct = seed_pkg["cross_task_bundle"]
        # Structured dict survived the inject path (the bug-fix invariant).
        assert isinstance(ct["transferable_insights"][0], dict)
        assert ct["transferable_insights"][0]["text"].startswith("When a full-width zero band splits the grid")
        assert ct["evidence_post_ids"] == [42, 91]

    # Render path produces clean markdown, not Python repr.
    lines = _render_bundle_item(
        orch.agents[0].seed_package["cross_task_bundle"]["transferable_insights"][0],
        max_chars=300,
    )
    rendered = "\n".join(lines)
    assert "(high)" in rendered
    assert "Applies when:" in rendered
    assert "Evidence:" in rendered
    assert "{'text':" not in rendered


def test_inject_seed_bundle_no_memory_skips_injection(tmp_path: Path, structured_insight):
    """When ``config.no_memory`` is true, ``_inject_seed_bundle`` early-
    returns and agents have no ``seed_package`` written (no engine state
    pollution from a config flag the user explicitly used to opt out)."""
    bundle_path = tmp_path / "donor.json"
    bundle_path.write_text(json.dumps({"meta": {}, "cross_task": {"transferable_insights": [structured_insight]}}))

    orch = _build_orchestrator_with_two_agents(no_memory=True)
    orch._inject_seed_bundle(str(bundle_path))

    for agent in orch.agents:
        assert agent.seed_package is None


def test_inject_seed_bundle_empty_cross_task_falls_through_to_legacy(tmp_path: Path):
    """A ``cross_task`` dict that's empty (no insight-bearing fields
    populated) must fall through to the legacy ``assets`` path when the
    bundle carries legacy assets. (A bundle with BOTH empty — no usable
    content — now fails loud instead; that contract is pinned by
    ``test_seed_bundle_injection.py::test_inject_seed_bundle_raises_on_empty_bundle``.)"""
    bundle_path = tmp_path / "donor.json"
    bundle_path.write_text(
        json.dumps(
            {
                "meta": {},
                "cross_task": {},
                "assets": [{"asset_id": "a1", "text": "Use BFS flood-fill for component extraction."}],
            }
        )
    )

    orch = _build_orchestrator_with_two_agents()
    orch._inject_seed_bundle(str(bundle_path))

    for agent in orch.agents:
        assert agent.seed_package is not None
        assert "cross_task_bundle" not in agent.seed_package


def test_inject_seed_bundle_mixed_dict_and_string_list(tmp_path: Path, structured_insight):
    """A donor bundle whose ``transferable_insights`` mixes structured
    dicts with bare strings (e.g. partially-stringified legacy data)
    renders both shapes correctly without crashing the engine. Each item
    is dispatched independently by ``_render_bundle_item``."""
    bundle_path = tmp_path / "donor.json"
    bundle_path.write_text(
        json.dumps(
            {
                "meta": {},
                "cross_task": {
                    "transferable_insights": [
                        structured_insight,
                        "Use BFS flood-fill for connected-component extraction.",
                    ],
                    "confirmed_constraints": [],
                    "rejected_hypotheses": [],
                    "pitfalls": [],
                    "checks": [],
                    "next_steps": [],
                    "evidence_post_ids": [42],
                },
            }
        )
    )

    orch = _build_orchestrator_with_two_agents()
    orch._inject_seed_bundle(str(bundle_path))

    items = orch.agents[0].seed_package["cross_task_bundle"]["transferable_insights"]
    assert isinstance(items[0], dict)
    assert isinstance(items[1], str)
    # Render each item; the output for the structured one should still
    # have the labelled child lines, the bare-string one should not.
    structured_lines = _render_bundle_item(items[0], max_chars=300)
    string_lines = _render_bundle_item(items[1], max_chars=300)
    assert any("Applies when:" in line for line in structured_lines)
    assert string_lines == ["- Use BFS flood-fill for connected-component extraction."]


def test_inject_seed_bundle_does_not_leak_source_experiment_into_description(tmp_path: Path, structured_insight):
    """Regression: `meta.source_experiment` MUST NOT be interpolated into
    `workstream_description`, which is rendered into the recipient agent's
    MEMORY.md. Donor identity (model/config encoded in the experiment name)
    must not bleed across the transfer boundary.

    `source_experiment` is preserved in `meta` for server-side audit logs only.
    """
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "meta": {
                    # Identifying donor experiment name — must NOT reach the
                    # workstream_description string.
                    "source_experiment": "haiku_swebench_pro_baseline_20260504",
                    "source_db": "/path/to/donor_knowledge.sqlite",
                    "source_run_id": 1,
                    # `bundle_summary` / `bundle_title` intentionally absent so
                    # the fallback branch fires.
                },
                "cross_task": {
                    "transferable_insights": ["Use BFS for connected components."],
                    "confirmed_constraints": [],
                    "rejected_hypotheses": [],
                    "pitfalls": [],
                    "checks": [],
                    "next_steps": [],
                    "evidence_post_ids": [],
                },
            }
        )
    )

    orch = _build_orchestrator_with_two_agents()
    orch._inject_seed_bundle(str(bundle_path))

    desc = orch.agents[0].seed_package.get("workstream_description") or ""
    assert "haiku_swebench_pro_baseline_20260504" not in desc, (
        "source_experiment leaked into workstream_description; "
        "this regresses the donor-anonymization fix and would expose donor "
        "identity to recipient agents via MEMORY.md."
    )
    # Sanity: the fallback IS firing (no bundle_summary/title supplied).
    assert "anonymized donor experiment" in desc

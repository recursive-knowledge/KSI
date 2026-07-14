"""Spec-attached prompt/domain-hint callables (issue #860).

A benchmark supplies its execution prompt, TASK.md, and distillation domain
hint via ``TaskSourceSpec`` fields alone — no edits to the hardcoded
``if kind == ...`` chains in ``src/ksi/prompts/__init__.py`` or any table in
``src/ksi/distillation/prompts.py``. These tests pin that (a) the
spec-attached callable/string is used when present, (b) the built-in sources
carry their domain hint on the spec, and (c) a source with no hint falls back
to the generic hint.
"""

from __future__ import annotations

import pytest

from ksi.benchmarks.sources import _DISTILL_HINT_ARC
from ksi.distillation.prompts import (
    _GENERIC_DOMAIN_HINT,
    _domain_hint,
    build_cross_task_distill_prompt,
)
from ksi.models import TaskSpec
from ksi.prompts import build_execution_prompt, build_task_markdown
from ksi.tasks.registry import REGISTRY, TaskSourceSpec, register_task_source


def _task(task_source: str) -> TaskSpec:
    return TaskSpec(id="t1", metadata={"task_source": task_source})


def test_spec_supplied_builders_and_hint_are_used():
    spec = TaskSourceSpec(
        name="spec_callable_bench",
        aliases=("scb",),
        prompt_kind="generic",  # would route to the generic fallback if consulted
        execution_prompt_builder=lambda task, *, has_memory, generation: (
            f"EXEC[{task.id}] mem={has_memory} gen={generation}"
        ),
        task_markdown_builder=lambda task: f"MD[{task.id}]\n",
        distill_domain_hint="DOMAIN HINT (spec_callable_bench): custom primitive.",
    )
    register_task_source(spec)
    try:
        task = _task("spec_callable_bench")
        assert build_execution_prompt(task, has_memory=True, generation=3) == "EXEC[t1] mem=True gen=3"
        assert build_task_markdown(task) == "MD[t1]\n"
        assert _domain_hint("spec_callable_bench") == "DOMAIN HINT (spec_callable_bench): custom primitive."
        # Alias resolves to the same spec, so the same callables/hint apply.
        assert build_execution_prompt(_task("scb"), has_memory=False, generation=1).startswith("EXEC[t1]")
    finally:
        REGISTRY.pop("spec_callable_bench", None)
        REGISTRY.pop("scb", None)


def test_distill_domain_hint_callable_is_invoked():
    calls: list[int] = []

    def _hint() -> str:
        calls.append(1)
        return "DOMAIN HINT (callable bench): dynamic."

    spec = TaskSourceSpec(name="hint_callable_bench", distill_domain_hint=_hint)
    register_task_source(spec)
    try:
        assert _domain_hint("hint_callable_bench") == "DOMAIN HINT (callable bench): dynamic."
        assert calls, "callable distill_domain_hint should be invoked"
    finally:
        REGISTRY.pop("hint_callable_bench", None)


def test_task_md_override_still_wins_over_spec_builder():
    spec = TaskSourceSpec(
        name="override_bench",
        task_markdown_builder=lambda task: "SHOULD_NOT_APPEAR\n",
    )
    register_task_source(spec)
    try:
        task = TaskSpec(
            id="t1",
            metadata={"task_source": "override_bench", "task_md_override": "EXPLICIT OVERRIDE"},
        )
        assert build_task_markdown(task) == "EXPLICIT OVERRIDE\n"
    finally:
        REGISTRY.pop("override_bench", None)


def test_builtin_source_uses_spec_attached_hint():
    # ARC ships no spec-attached prompt/markdown builders → the hardcoded
    # prompt_kind=="arc" branch is still used for those. Its domain hint,
    # however, now lives on the spec.
    arc_spec = REGISTRY["arc"]
    assert arc_spec.execution_prompt_builder is None
    assert arc_spec.task_markdown_builder is None
    assert arc_spec.distill_domain_hint == _DISTILL_HINT_ARC

    exec_prompt = build_execution_prompt(_task("arc"), has_memory=False, generation=1)
    assert "You are solving one ARC visual reasoning task." in exec_prompt
    assert "- task_source: arc" in build_task_markdown(_task("arc"))
    assert _domain_hint("arc") == _DISTILL_HINT_ARC
    # An unknown/unresolvable source resolves to the generic hint.
    assert _domain_hint("totally_unknown_source") == _GENERIC_DOMAIN_HINT


def test_source_without_hint_injects_no_domain_hint():
    # The domain hint is opt-in: a resolved source that leaves
    # distill_domain_hint unset gets no hint at all (not the generic one), and
    # the built distill prompt omits the domain-hint paragraph entirely.
    spec = TaskSourceSpec(name="no_hint_bench")
    register_task_source(spec)
    try:
        assert spec.distill_domain_hint is None
        assert _domain_hint("no_hint_bench") == ""
        system, user = build_cross_task_distill_prompt(cross_posts=[], task_source="no_hint_bench")
        assert "DOMAIN HINT" not in system + user
    finally:
        REGISTRY.pop("no_hint_bench", None)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

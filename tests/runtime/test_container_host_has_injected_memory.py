"""Regression tests for ``_seed_package_has_real_memory``.

Background: ``has_injected_memory`` used to be ``bool(memory_md.strip())``. The
seed-package renderer emits a non-empty skeleton (workstream name + Task ID
Reference) even at gen 1 with no prior attempts or distilled bundles, so the
flag was True even when no real content existed. That falsely activated the
"Prior attempt summaries are already in MEMORY.md" branch of
``_memory_block``, which Haiku-class models then narrated bullet-by-bullet
against an empty MEMORY.md, wasting tool budget on TodoWrite scaffolding.

These tests pin the new contract: the flag tracks actual content, not the
rendered string.
"""

from __future__ import annotations

from kcsi.runtime.container_host import _seed_package_has_real_memory


def test_none_is_not_real_memory() -> None:
    assert _seed_package_has_real_memory(None) is False


def test_non_dict_is_not_real_memory() -> None:
    assert _seed_package_has_real_memory("a string") is False
    assert _seed_package_has_real_memory(["list"]) is False


def test_empty_dict_is_not_real_memory() -> None:
    assert _seed_package_has_real_memory({}) is False


def test_skeleton_only_seed_is_not_real_memory() -> None:
    """Gen-1 shape: workstream + Task ID Reference + empty content fields.

    This is the exact seed_package that ``_enrich_seed_packages`` builds for
    the very first generation of a fresh run. The renderer turns it into an
    ~86-byte MEMORY.md skeleton, but no real content is present.
    """
    skeleton = {
        "workstream_name": "task-1",
        "workstream_description": "describe the task",
        "assigned_task_id": "task-1",
        "prior_attempts": [],
        "related_summaries": [],
        "best_score": 0.0,
        "memory_snapshot": {},
    }
    assert _seed_package_has_real_memory(skeleton) is False


def test_real_prior_attempts_is_real_memory() -> None:
    seed = {
        "workstream_name": "task-1",
        "prior_attempts": [{"gen": 1, "score": 0.5, "approach": "tried X"}],
    }
    assert _seed_package_has_real_memory(seed) is True


def test_real_per_task_bundle_is_real_memory() -> None:
    seed = {
        "per_task_bundle": {
            "transferable_insights": [{"text": "approach idea"}],
        },
    }
    assert _seed_package_has_real_memory(seed) is True


def test_real_cross_task_bundle_is_real_memory() -> None:
    seed = {
        "cross_task_bundle": {
            "transferable_insights": [{"text": "shape rule"}],
        },
    }
    assert _seed_package_has_real_memory(seed) is True


def test_metadata_only_does_not_flip_flag() -> None:
    """Workstream name + description + Task IDs are pure metadata. They make
    ``seed_package_to_memory_md`` emit a non-empty skeleton string but do
    NOT count as real memory content.
    """
    seed = {
        "workstream_name": "task-1",
        "workstream_description": "a description",
        "assigned_task_id": "task-1",
    }
    assert _seed_package_has_real_memory(seed) is False


def test_empty_content_lists_still_not_real() -> None:
    """An empty list / dict for the content keys must not be truthy."""
    seed = {
        "prior_attempts": [],
        "per_task_bundle": {},
        "cross_task_bundle": {},
    }
    assert _seed_package_has_real_memory(seed) is False

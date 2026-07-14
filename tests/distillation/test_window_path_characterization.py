"""Characterization: the default window-bundle path output must not change
when the fold/motifs alternative channels are removed."""

import json

from ksi.distillation.cross_task import distill_cross_task
from ksi.distillation.per_task import distill_one_task
from ksi.distillation.types import CrossTaskBundle, PerTaskBundle


def _make_stub_llm(payload: dict):
    """Return a legacy (system, user) -> str callable that echoes a fixed JSON payload.

    Mirrors the ``fake_llm`` pattern in tests/distillation/test_per_task.py:
    a plain function that ignores prompts and returns serialised JSON.
    """

    def fake_llm(sys_prompt: str, user_prompt: str) -> str:
        return json.dumps(payload)

    return fake_llm


def test_window_per_task_bundle_shape_is_stable():
    # Field names from _load_attempts / existing test_per_task.py fixture.
    attempts = [
        {"agent_id": "a1", "native_score": 0.0, "model_output": "attempt one"},
        {"agent_id": "a2", "native_score": 0.0, "model_output": "attempt two"},
    ]
    posts = [{"id": 1, "agent_id": "a1", "text": "X fails"}]

    stub_llm = _make_stub_llm(
        {
            "transferable_insights": [{"text": "lesson from post 1", "applies_when": "always"}],
            "confirmed_constraints": [],
            "rejected_hypotheses": [],
            "pitfalls": [],
            "checks": [],
            "next_steps": [],
            "evidence_post_ids": [1],
        }
    )

    bundle = distill_one_task(
        task_id="t1",
        attempts=attempts,
        posts=posts,
        llm=stub_llm,
        task_source="polyglot",
        bundle_schema=None,
    )

    assert bundle is not None
    assert isinstance(bundle, PerTaskBundle)
    assert bundle.task_id == "t1"
    # transferable_insights items are structured dicts (V2 format); text is preserved.
    assert [i.get("text") for i in bundle.transferable_insights] == ["lesson from post 1"]
    # The bundle channel does NOT set raw; raw is None (not a dict with "format").
    assert "format" not in (bundle.raw or {})
    # Evidence ids are filtered to supplied post ids.
    assert bundle.evidence_post_ids == [1]


def test_window_per_task_strips_cited_id_not_in_posts():
    # Pins the exact semantic the channel-removal touched: with fold gone,
    # valid_post_ids is _post_id_set(posts) only (no prior_bundle/retrieved
    # widening). A cited post_id absent from `posts` MUST be stripped, so a
    # carried-over citation can never re-enter via the window path.
    attempts = [{"agent_id": "a1", "native_score": 0.0, "model_output": "attempt"}]
    posts = [{"id": 1, "agent_id": "a1", "text": "X fails"}]

    stub_llm = _make_stub_llm(
        {
            "transferable_insights": [{"text": "lesson", "applies_when": "always"}],
            "confirmed_constraints": [],
            "rejected_hypotheses": [],
            "pitfalls": [],
            "checks": [],
            "next_steps": [],
            "evidence_post_ids": [1, 999],  # 999 is not among `posts`
        }
    )

    bundle = distill_one_task(
        task_id="t1",
        attempts=attempts,
        posts=posts,
        llm=stub_llm,
        task_source="polyglot",
        bundle_schema=None,
    )

    assert bundle is not None
    assert bundle.evidence_post_ids == [1]  # 999 stripped, not admitted


def test_window_cross_task_bundle_shape_is_stable():
    posts = [{"id": 1, "text": "cross insight", "task_id": "t1"}]

    stub_llm = _make_stub_llm(
        {
            "transferable_insights": [{"text": "shared across post 1", "applies_when": "code tasks"}],
            "confirmed_constraints": [],
            "rejected_hypotheses": [],
            "pitfalls": [],
            "checks": [],
            "next_steps": [],
            "evidence_post_ids": [1],
        }
    )

    bundle = distill_cross_task(
        cross_posts=posts,
        llm=stub_llm,
        task_source="polyglot",
        bundle_schema=None,
    )

    assert bundle is not None
    assert isinstance(bundle, CrossTaskBundle)
    # transferable_insights items are structured dicts (V2 format); text is preserved.
    assert [i.get("text") for i in bundle.transferable_insights] == ["shared across post 1"]
    # Evidence ids are filtered to supplied post ids.
    assert bundle.evidence_post_ids == [1]
    # The cross-task bundle channel also does not set raw.
    assert "format" not in (bundle.raw or {})


def test_motifs_module_is_removed():
    import importlib

    import pytest

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("ksi.distillation.motifs")


def test_ledger_module_is_removed():
    import importlib

    import pytest

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("ksi.distillation.ledger")


def test_distill_one_task_has_no_channel_param():
    import inspect

    from ksi.distillation.per_task import distill_one_task

    assert "channel" not in inspect.signature(distill_one_task).parameters


def test_fold_and_retrieval_modules_removed():
    import importlib

    import pytest

    for mod in ("ksi.distillation.fold", "ksi.distillation.retrieval"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)


def test_distillers_have_no_fold_params():
    import inspect

    from ksi.distillation.cross_task import distill_cross_task
    from ksi.distillation.per_task import distill_one_task

    pt = inspect.signature(distill_one_task).parameters
    ct = inspect.signature(distill_cross_task).parameters
    assert "prior_bundle" not in pt and "retrieved_posts" not in pt
    assert "prior_bundle" not in ct and "retrieved_posts" not in ct

"""Guard against drift between the three representations of the distill bundle:
the machine ``DISTILL_BUNDLE_JSON_SCHEMA`` (sent to providers), the bundle
dataclasses, and the ``_as_insight_list`` coercion that consumes provider output.

These three are edited by hand in separate files; without this test, adding a
field to one and forgetting another silently no-ops (``additionalProperties:
true`` swallows extras in both directions).
"""

from __future__ import annotations

import dataclasses

from kcsi.distillation.per_task import _as_insight_list
from kcsi.distillation.prompts import DISTILL_BUNDLE_JSON_SCHEMA
from kcsi.distillation.types import CrossTaskBundle, PerTaskBundle


def _schema_top_level_keys() -> set[str]:
    return set(DISTILL_BUNDLE_JSON_SCHEMA["schema"]["properties"].keys())


def _bundle_fields(cls) -> set[str]:
    # ``raw`` is carrier metadata (alternate-format payload), never emitted
    # by the LLM — excluded like task_id.
    return {f.name for f in dataclasses.fields(cls)} - {"raw"}


def test_schema_keys_match_cross_task_bundle_fields():
    """Every machine-schema property is a CrossTaskBundle field and vice versa.
    CrossTaskBundle has no ``task_id`` (per-task only), so the sets are equal."""
    assert _schema_top_level_keys() == _bundle_fields(CrossTaskBundle), (
        "DISTILL_BUNDLE_JSON_SCHEMA top-level keys drifted from CrossTaskBundle "
        "fields — update both (and the prose _OUTPUT_SCHEMA) together."
    )


def test_schema_keys_match_per_task_bundle_fields_minus_task_id():
    """PerTaskBundle is the cross-task shape plus a ``task_id`` the LLM never emits."""
    assert _schema_top_level_keys() == (_bundle_fields(PerTaskBundle) - {"task_id"}), (
        "DISTILL_BUNDLE_JSON_SCHEMA top-level keys drifted from PerTaskBundle fields (excluding task_id)."
    )


def test_every_schema_insight_field_round_trips_through_coercion():
    """Each key the schema declares on an Insight item must be one the coercion
    layer actually reads — otherwise the provider is told to emit a field that is
    silently dropped on ingest."""
    item_props = DISTILL_BUNDLE_JSON_SCHEMA["schema"]["properties"]["transferable_insights"]["items"]["properties"]
    # Build a fully-populated Insight using exactly the schema-declared keys.
    insight = {
        "text": "when test_index equals 3, do Y",
        "applies_when": "condition X holds",
        "does_not_apply_when": "boundary Z",
        "confidence": "high",
        "evidence": [{"task_id": "t1", "post_id": 1, "quote": "verbatim"}],
    }
    # Sanity: the dict we built uses only keys the schema declares.
    assert set(insight) <= set(item_props), (
        f"test fixture uses keys absent from schema: {set(insight) - set(item_props)}"
    )

    out = _as_insight_list([insight], allowed_post_ids={1})
    assert len(out) == 1
    cleaned = out[0]
    assert isinstance(cleaned, dict)
    # Every scalar schema field survives coercion.
    assert cleaned["text"] == "when test_index equals 3, do Y"
    assert cleaned["applies_when"] == "condition X holds"
    assert cleaned["does_not_apply_when"] == "boundary Z"
    assert cleaned["confidence"] == "high"
    # Evidence sub-fields survive too.
    assert cleaned["evidence"] == [{"post_id": 1, "task_id": "t1", "quote": "verbatim"}]


def test_evidence_post_ids_is_integer_array_and_coercion_drops_nonpositive():
    """The schema types evidence_post_ids as integers; the coercion enforces the
    tighter positive-int rule. This pins the documented (minor) range mismatch so
    a future tightening of either side is a conscious change."""
    from kcsi.distillation.per_task import _as_int_list

    assert DISTILL_BUNDLE_JSON_SCHEMA["schema"]["properties"]["evidence_post_ids"] == {
        "type": "array",
        "items": {"type": "integer", "minimum": 1},
    }
    # Coercion keeps positive ints in the allowed set, drops 0 / negatives.
    assert _as_int_list([1, 0, -2, 3], allowed_values={1, 3}) == [1, 3]

"""Regression guards for ARC task-map provenance and run-artifact identity."""

from __future__ import annotations

import json

from conftest import REPO_ROOT

ARC_RUN_SCRIPT = REPO_ROOT / "benchmarks" / "run_arc.sh"


def test_arc_campaign_validates_and_persists_task_map_identity():
    text = ARC_RUN_SCRIPT.read_text(encoding="utf-8")
    assert 'validate_arc_task_map "$TASK_MAP" "$DATA_DIR"' in text
    assert '--task-ids-file "$TASK_MAP"' in text
    assert '--task-map-path "$TASK_MAP"' in text


def test_arc_kt_maps_record_source_commit():
    for rel in (
        "benchmarks/arc1/task_maps/arc1_eval_50_seed1_kt.json",
        "benchmarks/arc2/task_maps/arc2_eval_50_seed1_kt.json",
    ):
        payload = json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))
        source_commit = payload.get("source_commit")
        assert isinstance(source_commit, str), rel
        assert len(source_commit) == 40, rel

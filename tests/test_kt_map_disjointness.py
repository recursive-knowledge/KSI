"""Enforce the KT recipient/donor disjointness invariant for every task map
that self-declares it via ``disjoint_from``.

Knowledge-transfer results are only valid if the recipient (eval) subset shares
no task with the donor (train/baseline) pool the transferred knowledge came from
-- otherwise a measured transfer signal is indistinguishable from leakage.

The arc1/arc2/swebench_pro/polyglot KT
maps each declare ``disjoint_from`` in their JSON and instruct "Do not modify
membership after publishing KT results", but nothing enforced it -- a
regenerated or hand-edited map could silently introduce overlap with zero CI
signal. This test discovers every ``disjoint_from``-declaring map under
``benchmarks/`` and asserts the invariant holds, so new KT maps are covered
automatically.
"""

import json
from pathlib import Path

import pytest
from conftest import REPO_ROOT

BENCHMARKS = REPO_ROOT / "benchmarks"


def _task_ids(path: Path) -> set[str]:
    """Return the set of ``task_id`` values from a task-map file.

    Handles the shapes used across the repo: a dict with an inline ``tasks``
    list, a dict with an ``ids_file`` pointer to a sibling id list (polyglot
    meta maps), and a bare list of task entries. Entries may be dicts (with a
    ``task_id`` key) or plain id strings.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "tasks" in data:
            entries = data["tasks"]
        elif "ids_file" in data:
            return _task_ids(path.parent / data["ids_file"])
        else:
            raise KeyError(f"{path.name} has neither 'tasks' nor 'ids_file'")
    else:
        entries = data
    ids: set[str] = set()
    for entry in entries:
        if isinstance(entry, dict):
            ids.add(str(entry["task_id"]))
        else:
            ids.add(str(entry))
    return ids


def _maps_declaring_disjoint_from() -> list[Path]:
    # Scope to committed task-map dirs; never walk repo_cache/ (large checkouts
    # with transient files) or source/ dataset dirs.
    found: list[Path] = []
    for path in sorted(BENCHMARKS.glob("*/task_maps/*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and data.get("disjoint_from"):
            found.append(path)
    return found


def test_at_least_the_known_kt_maps_are_discovered():
    """Guard against the discovery glob silently matching nothing (which would
    make every parametrized test vacuously pass)."""
    names = {p.name for p in _maps_declaring_disjoint_from()}
    for expected in (
        "arc1_eval_50_seed1_kt.json",
        "arc2_eval_50_seed1_kt.json",
        "swebench_pro_test_50_seed1_kt.json",
    ):
        assert expected in names, f"expected KT map {expected} not discovered"


@pytest.mark.parametrize(
    "kt_map",
    _maps_declaring_disjoint_from(),
    ids=lambda p: p.name,
)
def test_kt_map_disjoint_from_declared_donor(kt_map: Path):
    data = json.loads(kt_map.read_text(encoding="utf-8"))
    donor_name = data["disjoint_from"]
    donor = kt_map.parent / donor_name
    assert donor.exists(), f"{kt_map.name} declares disjoint_from={donor_name!r} but it is missing"

    recipient_ids = _task_ids(kt_map)
    donor_ids = _task_ids(donor)
    assert recipient_ids, f"{kt_map.name} has no task ids"
    assert donor_ids, f"{donor_name} has no task ids"

    overlap = recipient_ids & donor_ids
    assert not overlap, f"{kt_map.name} overlaps its declared donor {donor_name}: {sorted(overlap)}"

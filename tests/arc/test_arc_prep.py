"""Tests for the ARC prep scripts.

These tests target the *public* behaviour of scripts under
``benchmarks/scripts/``:

- ``prepare_arc_workspace_payloads.py``
- ``convert_arc_workspace_predictions.py``
- ``prepare_arc_native_prompts.py``

``benchmarks/scripts/`` is not a Python package, so modules are loaded via
``importlib.util``. Tests avoid depending on repo-wide fixtures (the ARC
submodule, cloned source datasets) by constructing tiny fake ARC task
JSONs inside ``tmp_path``.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import FIXTURES_DIR, REPO_ROOT

SCRIPTS_DIR = REPO_ROOT / "benchmarks" / "scripts"


def _load_script(name: str):
    path = SCRIPTS_DIR / name
    if not path.exists():
        pytest.skip(f"{path} not available")
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_source_arc_task(path: Path, *, train=None, test=None) -> None:
    payload = {
        "train": train if train is not None else [{"input": [[0]], "output": [[1]]}],
        "test": test if test is not None else [{"input": [[2]], "output": [[3]]}],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_task_map(path: Path, *, benchmark: str, task_id: str, source_file: Path) -> None:
    payload = {
        "benchmark": benchmark,
        "split": "evaluation",
        "seed": 0,
        "count": 1,
        "selection_name": f"{benchmark}_test_1_seed0",
        "tasks": [
            {
                "index": 1,
                "task_id": task_id,
                # Absolute path so resolve_project_path keeps it as-is.
                "source_file": str(source_file),
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_script(script_name: str, args: list[str]) -> subprocess.CompletedProcess:
    """Run a script as a subprocess and return the completed process."""
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name), *args]
    return subprocess.run(cmd, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# prepare_arc_workspace_payloads
# ---------------------------------------------------------------------------


def test_grid_validation_rejects_malformed():
    mod = _load_script("prepare_arc_workspace_payloads.py")
    validate = getattr(mod, "_validate_grid", None)
    if validate is None:
        pytest.skip("_validate_grid not available in prepare_arc_workspace_payloads.py")

    # Empty
    with pytest.raises(ValueError):
        validate([], "test.origin")
    # Ragged rows
    with pytest.raises(ValueError):
        validate([[1, 2], [3]], "test.origin")
    # Non-int cell
    with pytest.raises(ValueError):
        validate([[1, 2, "x"]], "test.origin")
    # Out of [0, 9]
    with pytest.raises(ValueError):
        validate([[1, 2, 10]], "test.origin")
    # Negative
    with pytest.raises(ValueError):
        validate([[1, 2, -1]], "test.origin")
    # Bools are rejected even though isinstance(True, int) is True.
    with pytest.raises(ValueError):
        validate([[True, False]], "test.origin")
    # Empty row
    with pytest.raises(ValueError):
        validate([[]], "test.origin")
    # Not a list
    with pytest.raises(ValueError):
        validate("not-a-grid", "test.origin")

    # OK case: should not raise.
    validate([[0, 5, 9]], "test.origin")


def test_generate_payloads_strips_test_output(tmp_path):
    mod = _load_script("prepare_arc_workspace_payloads.py")

    source_file = tmp_path / "source" / "abc123.json"
    _write_source_arc_task(
        source_file,
        train=[{"input": [[0]], "output": [[1]]}],
        test=[{"input": [[2]], "output": [[3]]}],
    )
    task_map = tmp_path / "task_map.json"
    _write_task_map(task_map, benchmark="arc1", task_id="abc123", source_file=source_file)
    out_dir = tmp_path / "out"

    generate = mod.generate_payloads
    result_dir = generate(task_map, out_dir)
    assert Path(result_dir) == out_dir

    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["benchmark"] == "arc1"
    assert manifest["count"] == 1
    assert len(manifest["payloads"]) == 1

    payload_path = out_dir / "abc123_pair0.json"
    assert payload_path.exists()

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    # Required keys on the payload object.
    assert set(payload.keys()) >= {
        "benchmark",
        "task_id",
        "pair_index",
        "train",
        "test_input",
    }
    assert payload["benchmark"] == "arc1"
    assert payload["task_id"] == "abc123"
    assert payload["pair_index"] == 0
    assert payload["test_input"] == [[2]]

    # test_input is a bare grid, not an object with "output" — make sure
    # nothing in the payload leaks a test "output" grid.
    payload_text = payload_path.read_text(encoding="utf-8")
    # The source test output was [[3]]; verify it is absent from the payload.
    assert "[[3]]" not in payload_text.replace(" ", "")
    # Also the key "output" should only come from train pairs.
    assert isinstance(payload["test_input"], list)
    assert not any(isinstance(row, dict) and "output" in row for row in payload["test_input"])


def test_generate_payloads_rejects_missing_test(tmp_path):
    mod = _load_script("prepare_arc_workspace_payloads.py")

    source_file = tmp_path / "source" / "abc123.json"
    _write_source_arc_task(
        source_file,
        train=[{"input": [[0]], "output": [[1]]}],
        test=[],
    )
    task_map = tmp_path / "task_map.json"
    _write_task_map(task_map, benchmark="arc1", task_id="abc123", source_file=source_file)
    with pytest.raises(ValueError):
        mod.generate_payloads(task_map, tmp_path / "out")


# ---------------------------------------------------------------------------
# convert_arc_workspace_predictions
# ---------------------------------------------------------------------------


def _write_prediction(
    path: Path,
    *,
    task_id: str,
    pair_index: int,
    attempt_index: int,
    grid,
    benchmark: str = "arc1",
) -> None:
    payload = {
        "benchmark": benchmark,
        "task_id": task_id,
        "pair_index": pair_index,
        "attempt_index": attempt_index,
        "prediction": grid,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_convert_predictions_merges_attempt_1_and_2(tmp_path):
    mod = _load_script("convert_arc_workspace_predictions.py")

    preds_dir = tmp_path / "preds"
    out_dir = tmp_path / "out"

    _write_prediction(
        preds_dir / "abc123_pair0_attempt1_prediction.json",
        task_id="abc123",
        pair_index=0,
        attempt_index=1,
        grid=[[1, 2], [3, 4]],
    )
    _write_prediction(
        preds_dir / "abc123_pair0_attempt2_prediction.json",
        task_id="abc123",
        pair_index=0,
        attempt_index=2,
        grid=[[5, 6], [7, 8]],
    )

    predictions = mod.collect_predictions(preds_dir)
    mod.convert_predictions(predictions, out_dir, None)

    submission_file = out_dir / "abc123.json"
    assert submission_file.exists()
    submission = json.loads(submission_file.read_text(encoding="utf-8"))

    # The submission is a list of per-pair attempt dicts.
    assert isinstance(submission, list)
    assert len(submission) == 1
    pair0 = submission[0]
    assert "attempt_1" in pair0
    assert "attempt_2" in pair0
    assert pair0["attempt_1"]["answer"] == [[1, 2], [3, 4]]
    assert pair0["attempt_2"]["answer"] == [[5, 6], [7, 8]]


def test_convert_predictions_accepts_attempts_wrapper(tmp_path):
    mod = _load_script("convert_arc_workspace_predictions.py")

    preds_dir = tmp_path / "preds"
    preds_dir.mkdir()
    (preds_dir / "abc123_pair0.json").write_text(
        json.dumps(
            {
                "benchmark": "arc1",
                "task_id": "abc123",
                "pair_index": 0,
                "attempts": [
                    {"attempt_index": 1, "prediction": [[1, 2], [3, 4]]},
                    {"attempt_index": 2, "prediction": [[5, 6], [7, 8]]},
                ],
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    predictions = mod.collect_predictions(preds_dir)
    mod.convert_predictions(predictions, out_dir, None)

    submission = json.loads((out_dir / "abc123.json").read_text(encoding="utf-8"))
    pair0 = submission[0]
    assert pair0["attempt_1"]["answer"] == [[1, 2], [3, 4]]
    assert pair0["attempt_2"]["answer"] == [[5, 6], [7, 8]]


def test_convert_predictions_preserves_missing_placeholder_as_no_attempts(tmp_path):
    mod = _load_script("convert_arc_workspace_predictions.py")

    preds_dir = tmp_path / "preds"
    preds_dir.mkdir()
    (preds_dir / "zero_task_pair0.json").write_text(
        json.dumps(
            {
                "benchmark": "arc1",
                "task_id": "zero_task",
                "pair_index": 0,
                "missing_prediction": True,
                "attempts": [],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "payloads",
                "benchmark": "arc1",
                "payloads": [{"task_id": "zero_task", "pair_index": 0}],
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    predictions = mod.collect_predictions(preds_dir)
    mod.convert_predictions(predictions, out_dir, manifest_path)

    assert json.loads((out_dir / "zero_task.json").read_text(encoding="utf-8")) == [{}]


def test_convert_predictions_empty_dir_raises(tmp_path, monkeypatch):
    """main() must refuse an empty predictions directory.

    Exercises the guard end-to-end by driving main() with argparse — this is
    the user-facing entry point that actually raises ValueError, and catching
    that guard is the whole point of the test.
    """
    mod = _load_script("convert_arc_workspace_predictions.py")

    preds_dir = tmp_path / "preds"
    preds_dir.mkdir()
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_arc_workspace_predictions.py",
            "--predictions-dir",
            str(preds_dir),
            "--output-submission-dir",
            str(out_dir),
        ],
    )

    with pytest.raises(ValueError, match="No prediction JSON files"):
        mod.main()

    # And no submission files should have been written.
    assert not out_dir.exists() or list(out_dir.iterdir()) == []


def test_workspace_ui_export_schema_matches_converter(tmp_path):
    """Pin the UI export → converter contract.

    Fixtures under tests/fixtures/arc_workspace_ui/ are verbatim copies of
    the JSONs produced by benchmarks/arc/workspace_ui/js/workspace.js. If
    the UI adds/removes/renames fields, or the converter's expected schema
    drifts, this test fails — catching silent UI/converter divergence.
    """
    mod = _load_script("convert_arc_workspace_predictions.py")

    ui_fixtures = FIXTURES_DIR / "arc_workspace_ui"
    preds_dir = tmp_path / "preds"
    preds_dir.mkdir()
    for src in sorted(ui_fixtures.glob("*_prediction.json")):
        shutil.copy(src, preds_dir / src.name)

    out_dir = tmp_path / "out"
    predictions = mod.collect_predictions(preds_dir)
    assert len(predictions) == 2, "expected attempt_1 + attempt_2 fixtures"

    # Required keys the converter pulls off the UI export.
    for row in predictions:
        assert {"benchmark", "task_id", "pair_index", "attempt_index", "prediction"} <= set(row)

    mod.convert_predictions(predictions, out_dir, None)
    submission = json.loads((out_dir / "abc123.json").read_text(encoding="utf-8"))
    assert len(submission) == 1
    pair0 = submission[0]
    assert pair0["attempt_1"]["answer"] == [[1, 2], [3, 4]]
    assert pair0["attempt_2"]["answer"] == [[5, 6], [7, 8]]


def test_convert_predictions_multi_pair_single_task(tmp_path):
    mod = _load_script("convert_arc_workspace_predictions.py")
    preds_dir = tmp_path / "preds"
    out_dir = tmp_path / "out"

    for pair_index, grid in [(0, [[1]]), (1, [[2]]), (2, [[3]])]:
        _write_prediction(
            preds_dir / f"abc123_pair{pair_index}_attempt1.json",
            task_id="abc123",
            pair_index=pair_index,
            attempt_index=1,
            grid=grid,
        )

    predictions = mod.collect_predictions(preds_dir)
    mod.convert_predictions(predictions, out_dir, None)

    submission = json.loads((out_dir / "abc123.json").read_text(encoding="utf-8"))
    assert [pair["attempt_1"]["answer"] for pair in submission] == [
        [[1]],
        [[2]],
        [[3]],
    ]


def test_convert_predictions_multi_task(tmp_path):
    mod = _load_script("convert_arc_workspace_predictions.py")
    preds_dir = tmp_path / "preds"
    out_dir = tmp_path / "out"

    _write_prediction(
        preds_dir / "task_a_pair0_attempt1.json",
        task_id="task_a",
        pair_index=0,
        attempt_index=1,
        grid=[[1]],
    )
    _write_prediction(
        preds_dir / "task_b_pair0_attempt1.json",
        task_id="task_b",
        pair_index=0,
        attempt_index=1,
        grid=[[2]],
    )

    predictions = mod.collect_predictions(preds_dir)
    mod.convert_predictions(predictions, out_dir, None)

    assert (out_dir / "task_a.json").exists()
    assert (out_dir / "task_b.json").exists()
    assert json.loads((out_dir / "task_a.json").read_text())[0]["attempt_1"]["answer"] == [[1]]
    assert json.loads((out_dir / "task_b.json").read_text())[0]["attempt_1"]["answer"] == [[2]]


def test_convert_predictions_non_contiguous_pair_indexes(tmp_path):
    """Gaps in pair_index must not crash — the submission list is sorted, not indexed."""
    mod = _load_script("convert_arc_workspace_predictions.py")
    preds_dir = tmp_path / "preds"
    out_dir = tmp_path / "out"

    for pair_index, grid in [(0, [[1]]), (3, [[9]])]:  # gap at 1, 2
        _write_prediction(
            preds_dir / f"abc_pair{pair_index}_attempt1.json",
            task_id="abc",
            pair_index=pair_index,
            attempt_index=1,
            grid=grid,
        )

    predictions = mod.collect_predictions(preds_dir)
    mod.convert_predictions(predictions, out_dir, None)

    submission = json.loads((out_dir / "abc.json").read_text(encoding="utf-8"))
    # Two entries total, sorted by pair_index ascending; pair_index metadata preserved.
    assert len(submission) == 2
    assert submission[0]["attempt_1"]["metadata"]["pair_index"] == 0
    assert submission[1]["attempt_1"]["metadata"]["pair_index"] == 3


def test_convert_predictions_attempt_1_and_2_different_shapes(tmp_path):
    """attempt_1 and attempt_2 answers are independent — shapes need not match."""
    mod = _load_script("convert_arc_workspace_predictions.py")
    preds_dir = tmp_path / "preds"
    out_dir = tmp_path / "out"

    _write_prediction(
        preds_dir / "abc_pair0_attempt1.json",
        task_id="abc",
        pair_index=0,
        attempt_index=1,
        grid=[[1, 2], [3, 4]],  # 2x2
    )
    _write_prediction(
        preds_dir / "abc_pair0_attempt2.json",
        task_id="abc",
        pair_index=0,
        attempt_index=2,
        grid=[[5, 6, 7]],  # 1x3 — intentionally different shape
    )

    predictions = mod.collect_predictions(preds_dir)
    mod.convert_predictions(predictions, out_dir, None)

    pair0 = json.loads((out_dir / "abc.json").read_text(encoding="utf-8"))[0]
    assert pair0["attempt_1"]["answer"] == [[1, 2], [3, 4]]
    assert pair0["attempt_2"]["answer"] == [[5, 6, 7]]


@pytest.mark.parametrize("missing_field", ["benchmark", "task_id", "pair_index", "prediction"])
def test_collect_predictions_rejects_missing_fields(tmp_path, missing_field):
    mod = _load_script("convert_arc_workspace_predictions.py")
    preds_dir = tmp_path / "preds"
    preds_dir.mkdir()

    payload = {
        "benchmark": "arc1",
        "task_id": "abc",
        "pair_index": 0,
        "attempt_index": 1,
        "prediction": [[1]],
    }
    payload.pop(missing_field)
    (preds_dir / "abc_pair0_attempt1.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=missing_field):
        mod.collect_predictions(preds_dir)


def test_collect_predictions_rejects_non_object_json(tmp_path):
    mod = _load_script("convert_arc_workspace_predictions.py")
    preds_dir = tmp_path / "preds"
    preds_dir.mkdir()

    (preds_dir / "oops.json").write_text('["not", "an", "object"]', encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        mod.collect_predictions(preds_dir)


@pytest.mark.parametrize(
    "bad_grid",
    [
        [],  # empty
        [[1, 2], [3]],  # ragged
        [[1, 2, "x"]],  # non-int cell
        [[1, 2, 10]],  # out of [0,9]
        [[-1]],  # negative
        [[True, False]],  # bools rejected
        [[]],  # empty row
        "not-a-grid",  # not a list
    ],
)
def test_collect_predictions_rejects_malformed_grid(tmp_path, bad_grid):
    mod = _load_script("convert_arc_workspace_predictions.py")
    preds_dir = tmp_path / "preds"
    preds_dir.mkdir()

    payload = {
        "benchmark": "arc1",
        "task_id": "abc",
        "pair_index": 0,
        "attempt_index": 1,
        "prediction": bad_grid,
    }
    (preds_dir / "abc_pair0_attempt1.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        mod.collect_predictions(preds_dir)


def test_convert_predictions_rejects_non_int_pair_index(tmp_path):
    mod = _load_script("convert_arc_workspace_predictions.py")
    preds_dir = tmp_path / "preds"
    preds_dir.mkdir()

    payload = {
        "benchmark": "arc1",
        "task_id": "abc",
        "pair_index": "zero",  # not an int
        "attempt_index": 1,
        "prediction": [[1]],
    }
    (preds_dir / "abc_pair0_attempt1.json").write_text(json.dumps(payload), encoding="utf-8")

    predictions = mod.collect_predictions(preds_dir)
    with pytest.raises(ValueError):
        mod.convert_predictions(predictions, tmp_path / "out", None)


def test_convert_predictions_manifest_benchmark_mismatch(tmp_path):
    mod = _load_script("convert_arc_workspace_predictions.py")
    preds_dir = tmp_path / "preds"
    out_dir = tmp_path / "out"

    _write_prediction(
        preds_dir / "abc_pair0_attempt1.json",
        task_id="abc",
        pair_index=0,
        attempt_index=1,
        grid=[[1]],
        benchmark="arc2",  # prediction says arc2
    )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "payloads",
                "benchmark": "arc1",  # manifest says arc1 — must raise
                "payloads": [{"task_id": "abc", "pair_index": 0}],
            }
        ),
        encoding="utf-8",
    )

    predictions = mod.collect_predictions(preds_dir)
    with pytest.raises(ValueError, match="benchmark mismatch"):
        mod.convert_predictions(predictions, out_dir, manifest_path)


def test_convert_predictions_manifest_missing_prediction(tmp_path):
    mod = _load_script("convert_arc_workspace_predictions.py")
    preds_dir = tmp_path / "preds"
    out_dir = tmp_path / "out"

    _write_prediction(
        preds_dir / "abc_pair0_attempt1.json",
        task_id="abc",
        pair_index=0,
        attempt_index=1,
        grid=[[1]],
    )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "payloads",
                "benchmark": "arc1",
                "payloads": [
                    {"task_id": "abc", "pair_index": 0},
                    {"task_id": "abc", "pair_index": 1},  # no prediction for this
                ],
            }
        ),
        encoding="utf-8",
    )

    predictions = mod.collect_predictions(preds_dir)
    with pytest.raises(ValueError, match="Missing predictions"):
        mod.convert_predictions(predictions, out_dir, manifest_path)


def test_convert_predictions_rejects_wrong_manifest_kind(tmp_path):
    mod = _load_script("convert_arc_workspace_predictions.py")
    preds_dir = tmp_path / "preds"
    out_dir = tmp_path / "out"

    _write_prediction(
        preds_dir / "abc_pair0_attempt1.json",
        task_id="abc",
        pair_index=0,
        attempt_index=1,
        grid=[[1]],
    )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "prompts",  # wrong kind — must raise
                "benchmark": "arc1",
                "payloads": [{"task_id": "abc", "pair_index": 0}],
            }
        ),
        encoding="utf-8",
    )

    predictions = mod.collect_predictions(preds_dir)
    with pytest.raises(ValueError, match="payload manifest"):
        mod.convert_predictions(predictions, out_dir, manifest_path)


def test_duplicate_prediction_detection(tmp_path):
    mod = _load_script("convert_arc_workspace_predictions.py")

    preds_dir = tmp_path / "preds"
    out_dir = tmp_path / "out"

    _write_prediction(
        preds_dir / "abc123_pair0_attempt1_first.json",
        task_id="abc123",
        pair_index=0,
        attempt_index=1,
        grid=[[1, 2], [3, 4]],
    )
    _write_prediction(
        preds_dir / "abc123_pair0_attempt1_second.json",
        task_id="abc123",
        pair_index=0,
        attempt_index=1,
        grid=[[9, 9], [9, 9]],
    )

    predictions = mod.collect_predictions(preds_dir)
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        mod.convert_predictions(predictions, out_dir, None)


# ---------------------------------------------------------------------------
# Task-id sanitisation (shared helper if present)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "script_name",
    ["prepare_arc_workspace_payloads.py", "convert_arc_workspace_predictions.py"],
)
def test_task_id_sanitization(script_name):
    mod = _load_script(script_name)
    sanitize = getattr(mod, "_sanitize_task_id", None)
    if sanitize is None:
        pytest.skip(f"_sanitize_task_id not yet available in {script_name}")

    # Rejected inputs
    with pytest.raises(ValueError):
        sanitize("../evil")
    with pytest.raises(ValueError):
        sanitize("a/b")
    with pytest.raises(ValueError):
        sanitize("task 123")
    with pytest.raises(ValueError):
        sanitize("")
    # Accepted inputs
    assert sanitize("valid_task_123") == "valid_task_123"
    assert sanitize("abc-def") == "abc-def"
    assert sanitize("abc123") == "abc123"


# ---------------------------------------------------------------------------
# prepare_arc_native_prompts (light-touch; submodule required for e2e)
# ---------------------------------------------------------------------------


def test_prepare_arc_native_prompts_help_exits_zero():
    """--help must work without the arc-agi-benchmarking submodule present."""
    script = SCRIPTS_DIR / "prepare_arc_native_prompts.py"
    if not script.exists():
        pytest.skip("prepare_arc_native_prompts.py not available")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr={result.stderr!r} stdout={result.stdout!r}"
    assert "--payload-manifest" in result.stdout


def test_prepare_arc_native_prompts_argparse_contract():
    """The script's build_parser should expose --payload-manifest and --output-dir."""
    mod = _load_script("prepare_arc_native_prompts.py")
    build_parser = getattr(mod, "build_parser", None)
    if build_parser is None:
        pytest.skip("build_parser not available")
    parser = build_parser()
    dest_names = {a.dest for a in parser._actions}
    assert "payload_manifest" in dest_names
    assert "output_dir" in dest_names

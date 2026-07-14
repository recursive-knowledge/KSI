#!/usr/bin/env python3
"""Convert ARC workspace prediction JSONs into arc-agi-benchmarking submissions."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Allow `from arc_prep._common import ...` when this script is run directly.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from arc_prep._common import (
    coerce_int,
    load_json,
    require_field,
    save_json,
)
from arc_prep._common import (
    sanitize_task_id as _sanitize_task_id,
)
from arc_prep._common import (
    validate_grid as _validate_grid,
)

# Deterministic timestamp used for workspace-exported predictions that don't
# carry real timestamps. Keeping this constant makes converter output
# reproducible (important for diffing submission JSONs).
_WORKSPACE_UI_TIMESTAMP = "1970-01-01T00:00:00+00:00"


def make_attempt(prediction: dict, attempt_name: str) -> dict:
    # Prefer timestamps carried by the prediction (real runs); fall back to a
    # deterministic constant so workspace-UI exports produce stable output.
    fallback_ts = _WORKSPACE_UI_TIMESTAMP
    required = {"prediction", "task_id", "pair_index"}
    missing = required - set(prediction)
    if missing:
        raise ValueError(f"prediction payload missing required fields {sorted(missing)} for attempt {attempt_name!r}")
    grid = prediction["prediction"]
    assistant_content = json.dumps(grid)
    model_name = prediction.get("model", "workspace-ui")
    provider_name = prediction.get("provider", "workspace")
    test_id = prediction.get("test_id", "workspace")
    return {
        "answer": grid,
        "metadata": {
            "model": model_name,
            "provider": provider_name,
            "start_timestamp": prediction.get("start_timestamp", fallback_ts),
            "end_timestamp": prediction.get("end_timestamp", fallback_ts),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "user",
                        "content": f"ARC workspace prediction export {attempt_name}",
                    },
                },
                {
                    "index": 1,
                    "message": {
                        "role": "assistant",
                        "content": assistant_content,
                    },
                },
            ],
            "kwargs": prediction.get("kwargs", {}),
            "usage": prediction.get(
                "usage",
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "completion_tokens_details": {
                        "reasoning_tokens": 0,
                        "accepted_prediction_tokens": 0,
                        "rejected_prediction_tokens": 0,
                    },
                },
            ),
            "cost": prediction.get(
                "cost",
                {
                    "prompt_cost": 0.0,
                    "completion_cost": 0.0,
                    "total_cost": 0.0,
                },
            ),
            "task_id": prediction["task_id"],
            "pair_index": prediction["pair_index"],
            "test_id": test_id,
        },
        "correct": None,
    }


def collect_predictions(predictions_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(predictions_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        payload = load_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Prediction file must be a JSON object: {path}")
        if "attempts" in payload:
            required = {"benchmark", "task_id", "pair_index", "attempts"}
            if not required <= set(payload):
                raise ValueError(f"Prediction file missing fields {required - set(payload)}: {path}")
            if not isinstance(payload["attempts"], list):
                raise ValueError(f"{path}: attempts must be a list")
            if payload.get("missing_prediction") is True and not payload["attempts"]:
                row = {key: value for key, value in payload.items() if key != "attempts"}
                row["_missing_prediction"] = True
                row["_path"] = str(path.resolve())
                rows.append(row)
                continue
            for attempt in payload["attempts"]:
                if not isinstance(attempt, dict):
                    raise ValueError(f"{path}: each attempt must be an object")
                if "prediction" not in attempt:
                    raise ValueError(f"{path}: each attempt must contain prediction")
                row = {key: value for key, value in payload.items() if key != "attempts"}
                row.update(attempt)
                _validate_grid(row["prediction"], f"{path}: attempt {row.get('attempt_index', '?')} prediction")
                row["_path"] = str(path.resolve())
                rows.append(row)
            continue
        required = {"benchmark", "task_id", "pair_index", "prediction"}
        if not required <= set(payload):
            raise ValueError(f"Prediction file missing fields {required - set(payload)}: {path}")
        _validate_grid(payload["prediction"], f"{path}: prediction")
        payload["_path"] = str(path.resolve())
        rows.append(payload)
    return rows


def convert_predictions(predictions: list[dict], output_dir: Path, manifest_path: Path | None) -> None:
    grouped: dict[str, dict[int, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in predictions:
        origin = row.get("_path", "?")
        task_id = require_field(row, "task_id", origin=origin)
        pair_index = coerce_int(
            require_field(row, "pair_index", origin=origin),
            origin=f"{origin}: pair_index",
        )
        attempt_index = coerce_int(
            row.get("attempt_index", 1),
            origin=f"{origin}: attempt_index",
        )
        if row.get("_missing_prediction"):
            attempt_index = 0
        existing = grouped[task_id][pair_index].get(attempt_index)
        if existing is not None:
            raise ValueError(
                f"Duplicate prediction for task={task_id!r}, pair={pair_index}, "
                f"attempt={attempt_index}: {existing.get('_path', '?')} and {origin}"
            )
        grouped[task_id][pair_index][attempt_index] = row

    expected_pairs = None
    manifest = None
    if manifest_path is not None:
        manifest = load_json(manifest_path)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("payloads"), list):
            raise ValueError("Manifest must contain a payloads list.")
        kind = manifest.get("kind")
        if kind is not None and kind != "payloads":
            raise ValueError(
                f"--manifest expects a payload manifest (kind='payloads'); got kind={kind!r}. "
                f"Pass the manifest from benchmarks/scripts/arc_prep/prepare_arc_workspace_payloads.py."
            )
        expected_pairs = set()
        for idx, item in enumerate(manifest["payloads"]):
            item_origin = f"{manifest_path.name} payloads[{idx}]"
            task_id = require_field(item, "task_id", origin=item_origin)
            pair_index = coerce_int(
                require_field(item, "pair_index", origin=item_origin),
                origin=f"{item_origin}.pair_index",
            )
            expected_pairs.add((task_id, pair_index))

    if manifest is not None:
        expected_benchmark = manifest.get("benchmark")
        for row in predictions:
            if row.get("benchmark") != expected_benchmark:
                src = row.get("_path", "?")
                raise ValueError(
                    f"prediction benchmark mismatch: got {row.get('benchmark')!r}, "
                    f"expected {expected_benchmark!r} (file: {src})"
                )

    seen_pairs = {(task_id, pair_index) for task_id, pair_map in grouped.items() for pair_index in pair_map}
    if expected_pairs is not None:
        missing = sorted(expected_pairs - seen_pairs)
        if missing:
            raise ValueError(f"Missing predictions for {len(missing)} task/pair items, e.g. {missing[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for task_id, pair_map in grouped.items():
        submission: list[dict] = []
        for pair_index in sorted(pair_map):
            attempts = pair_map[pair_index]
            pair_payload = {}
            for attempt_index in sorted(attempts):
                if attempts[attempt_index].get("_missing_prediction"):
                    continue
                attempt_name = f"attempt_{attempt_index}"
                pair_payload[attempt_name] = make_attempt(attempts[attempt_index], attempt_name)
            submission.append(pair_payload)
        save_json(output_dir / f"{_sanitize_task_id(task_id)}.json", submission)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert ARC workspace predictions to arc-agi-benchmarking submissions."
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory of per-pair prediction JSONs.",
    )
    parser.add_argument(
        "--output-submission-dir",
        type=Path,
        required=True,
        help="Output directory for per-task submission JSONs.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional payload manifest for completeness validation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictions_dir = args.predictions_dir.resolve()
    predictions = collect_predictions(predictions_dir)
    if not predictions:
        raise ValueError(
            f"No prediction JSON files found under {predictions_dir}. "
            f"Run the workspace UI and export predictions first."
        )
    convert_predictions(
        predictions,
        args.output_submission_dir.resolve(),
        args.manifest.resolve() if args.manifest else None,
    )
    print(f"Saved ARC submissions to {args.output_submission_dir.resolve()}")


if __name__ == "__main__":
    main()

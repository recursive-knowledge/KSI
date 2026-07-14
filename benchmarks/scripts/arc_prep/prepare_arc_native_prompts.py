#!/usr/bin/env python3
"""Render ARC native-mode prompt files from canonical ARC pair payloads."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import sys
from pathlib import Path

# Allow `from arc_prep._common import ...` when this script is run directly.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from arc_prep._common import (
    load_json,
    require_field,
    save_json,
)
from arc_prep._common import (
    relative_to_manifest as _relative_to_manifest,
)
from arc_prep._common import (
    sanitize_task_id as _sanitize_task_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
ARC_BENCHMARKING_SRC = BENCHMARKS_DIR / "arc" / "benchmarking" / "src"


_ARC_BENCHMARKING_HINT = (
    "The arc-agi-benchmarking source tree is required to build ARC native prompts, "
    f"but it was not found at {ARC_BENCHMARKING_SRC}.\n"
    "To install:\n"
    "  git clone https://github.com/arcprize/arc-agi-benchmarking.git "
    "benchmarks/arc/benchmarking\n"
    "  # Pin to the SHA recorded in BENCHMARK_PREPARE.md:\n"
    "  (cd benchmarks/arc/benchmarking && git checkout <PINNED_SHA>)\n"
    "See BENCHMARK_PREPARE.md for the pinned SHA."
)


def _import_arc_benchmarking():
    """Import arc_agi_benchmarking helpers lazily.

    Keeps module import side-effect free so `--help` works without the
    arc-agi-benchmarking source checkout present.
    """
    if not ARC_BENCHMARKING_SRC.is_dir():
        raise RuntimeError(_ARC_BENCHMARKING_HINT)
    src_str = str(ARC_BENCHMARKING_SRC.resolve())
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    try:
        from arc_agi_benchmarking.prompts.prompt_manager import (  # noqa: E402
            convert_task_pairs_to_prompt,
        )
        from arc_agi_benchmarking.schemas import ARCPair  # noqa: E402
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import arc_agi_benchmarking from {ARC_BENCHMARKING_SRC}: {exc}.\n{_ARC_BENCHMARKING_HINT}"
        ) from exc
    return convert_task_pairs_to_prompt, ARCPair


def default_output_dir(payload_manifest: Path, benchmark: str) -> Path:
    return BENCHMARKS_DIR / "arc" / "native_prompts" / benchmark / payload_manifest.parent.name


def build_prompt_payload(pair_payload: dict, payload_path: Path, convert_fn, arc_pair_cls) -> dict:
    origin = str(payload_path)
    train = require_field(pair_payload, "train", origin=origin)
    test_input_value = require_field(pair_payload, "test_input", origin=origin)
    task_id = require_field(pair_payload, "task_id", origin=origin)
    pair_index = require_field(pair_payload, "pair_index", origin=origin)
    benchmark = require_field(pair_payload, "benchmark", origin=origin)
    train_pairs = [arc_pair_cls(**pair) for pair in train]
    test_input = arc_pair_cls(input=test_input_value, output=None)
    prompt = convert_fn(train_pairs, test_input)
    return {
        "benchmark": benchmark,
        "task_id": task_id,
        "pair_index": pair_index,
        "prompt": prompt,
    }


def render_prompts(payload_manifest: Path, output_dir: Path | None) -> Path:
    convert_fn, arc_pair_cls = _import_arc_benchmarking()

    manifest = load_json(payload_manifest)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("payloads"), list):
        raise ValueError("Payload manifest must contain a payloads list.")
    kind = manifest.get("kind")
    if kind is not None and kind != "payloads":
        raise ValueError(
            f"Expected a payload manifest (kind='payloads'); got kind={kind!r}. "
            f"Pass the manifest from benchmarks/scripts/arc_prep/prepare_arc_workspace_payloads.py, not from "
            f"benchmarks/scripts/arc_prep/prepare_arc_native_prompts.py."
        )

    benchmark = manifest.get("benchmark")
    if benchmark not in {"arc1", "arc2"}:
        raise ValueError("Payload manifest benchmark must be arc1 or arc2.")

    input_manifest_dir = payload_manifest.parent
    out_dir = output_dir or default_output_dir(payload_manifest, benchmark)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_dir = out_dir

    prompt_manifest = {
        "kind": "prompts",
        "benchmark": benchmark,
        "payload_manifest": _relative_to_manifest(payload_manifest, output_manifest_dir),
        "count": 0,
        "prompts": [],
    }

    for idx, item in enumerate(manifest["payloads"]):
        item_origin = f"{payload_manifest.name} payloads[{idx}]"
        payload_file_value = require_field(item, "payload_file", origin=item_origin)
        payload_file_entry = Path(payload_file_value)
        if payload_file_entry.is_absolute():
            payload_file = payload_file_entry
        else:
            payload_file = (input_manifest_dir / payload_file_entry).resolve()
        pair_payload = load_json(payload_file)
        prompt_payload = build_prompt_payload(pair_payload, payload_file, convert_fn, arc_pair_cls)
        prompt_file = (
            out_dir / f"{_sanitize_task_id(prompt_payload['task_id'])}_pair{prompt_payload['pair_index']}.json"
        )
        save_json(prompt_file, prompt_payload)
        prompt_manifest["prompts"].append(
            {
                "task_id": prompt_payload["task_id"],
                "pair_index": prompt_payload["pair_index"],
                "payload_file": _relative_to_manifest(payload_file, output_manifest_dir),
                "prompt_file": _relative_to_manifest(prompt_file, output_manifest_dir),
            }
        )

    prompt_manifest["count"] = len(prompt_manifest["prompts"])
    save_json(out_dir / "manifest.json", prompt_manifest)
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render ARC native-mode prompts from canonical ARC pair payloads.")
    parser.add_argument(
        "--payload-manifest",
        type=Path,
        required=True,
        help="Manifest generated by benchmarks/scripts/arc_prep/prepare_arc_workspace_payloads.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Override prompt output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = render_prompts(args.payload_manifest.resolve(), args.output_dir.resolve() if args.output_dir else None)
    print(f"Saved ARC native prompts to {out_dir}")


if __name__ == "__main__":
    main()

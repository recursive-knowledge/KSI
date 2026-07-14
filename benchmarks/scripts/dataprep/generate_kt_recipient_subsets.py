#!/usr/bin/env python3
"""Generate knowledge-transfer recipient subsets disjoint from baseline subsets.

For each benchmark the baseline sweep uses a specific 50-task subset (seed=0).
The KT recipient subset is drawn from the same pool but **excludes** the
baseline tasks so donor training data never contaminates recipient evaluation.

Rules per benchmark (frozen):
- ARC1: baseline = train_50_seed0 (50 from training split, 400 total);
    recipient = random.Random(1).sample(sorted(eval_pool), 50). Cross-split
    disjointness is automatic (baseline ⊂ training, recipient ⊂ evaluation).
- ARC2: baseline = train_50_seed0 (50 from training split, ~1000 total);
    recipient = random.Random(1).sample(sorted(eval_pool), 50). Same
    cross-split design as ARC1.
- SWE-bench Pro test: baseline = seed=0 v1 (50/731); recipient =
    random.Random(1).sample(sorted(pool - baseline), 50). Same-split, so
    explicit set-difference still applies.
- Polyglot: baseline = medium subset (50/225); recipient = ALL remaining
    (175 tasks, no sampling). Outputs a task-IDs list file to feed into
    prepare_polyglot_dataset.py.

Outputs:
- benchmarks/arc1/task_maps/arc1_eval_50_seed1_kt.json
- benchmarks/arc2/task_maps/arc2_eval_50_seed1_kt.json
- benchmarks/swebench_pro/task_maps/swebench_pro_test_50_seed1_kt.json
- benchmarks/polyglot/task_maps/polyglot_rest_ids.json (175 instance_ids)
- benchmarks/polyglot/task_maps/polyglot_rest_ids.meta.json
- benchmarks/polyglot/task_maps/polyglot_eval_50_seed1_kt_ids.json
- benchmarks/polyglot/task_maps/polyglot_eval_50_seed1_kt_ids.meta.json

Idempotent: re-running produces the same output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_task_ids_from_map(path: Path, key_candidates=("task_id",)) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    ids = set()
    for t in tasks:
        for k in key_candidates:
            v = t.get(k)
            if v:
                ids.add(str(v))
                break
    return ids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"could not resolve git HEAD for polyglot source repo {repo_root}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _load_meta(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object metadata in {path}")
    return payload


def build_arc_recipient(
    *,
    benchmark: str,
    source_dir: Path,
    baseline_map: Path,
    output: Path,
    seed: int = 1,
    count: int = 50,
    source_repo: str,
    source_branch: str,
) -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"ARC source dir not found: {source_dir}")
    all_ids = sorted(p.stem for p in source_dir.glob("*.json"))
    baseline_ids = _load_task_ids_from_map(baseline_map)
    # Cross-split is the intended config: baseline lives on the training split
    # while the recipient is drawn from evaluation. In that case
    # ``baseline_ids - set(all_ids)`` equals ``baseline_ids`` (no overlap by
    # construction) and the set difference below is a no-op — the recipient
    # is simply ``count`` tasks sampled deterministically from the eval pool.
    remaining = sorted(set(all_ids) - baseline_ids)
    if len(remaining) < count:
        raise ValueError(f"insufficient disjoint tasks: {len(remaining)} available, need {count}")
    selected = random.Random(seed).sample(remaining, count)

    source_path = f"benchmarks/{benchmark}/source/data/evaluation"
    task_entries = [
        {"index": i, "task_id": tid, "source_file": f"{source_path}/{tid}.json"}
        for i, tid in enumerate(selected, start=1)
    ]

    task_map = {
        "benchmark": benchmark,
        "split": "evaluation",
        "seed": seed,
        "count": count,
        "selection_name": output.stem,
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_path": source_path,
        "selection_algorithm": ("random.Random(seed).sample(sorted(pool - baseline), count)"),
        "selection_notes": [
            f"Knowledge-transfer recipient subset disjoint from {baseline_map.name}.",
            f"Pool size={len(all_ids)}, baseline={len(baseline_ids)}, remaining={len(remaining)}, selected={count}.",
            "Do not modify membership after publishing KT results.",
        ],
        "disjoint_from": baseline_map.name,
        "tasks": task_entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(task_map, indent=2, ensure_ascii=True) + "\n")
    print(f"wrote {output} | {benchmark} recipient seed={seed} count={count}")


def build_swebench_pro_recipient(
    *, dataset_jsonl: Path, baseline_map: Path, output: Path, seed: int = 1, count: int = 50
) -> None:
    if not dataset_jsonl.exists():
        raise FileNotFoundError(f"SWE-bench Pro dataset not found: {dataset_jsonl}")
    instance_ids = []
    with dataset_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            iid = row.get("instance_id")
            if iid:
                instance_ids.append(str(iid))
    instance_ids.sort()
    baseline_payload = json.loads(baseline_map.read_text(encoding="utf-8"))
    baseline_ids = _load_task_ids_from_map(baseline_map)
    remaining = sorted(set(instance_ids) - baseline_ids)
    if len(remaining) < count:
        raise ValueError(f"insufficient disjoint SWE-bench Pro tasks: {len(remaining)} available, need {count}")
    selected = random.Random(seed).sample(remaining, count)

    task_entries = [{"index": i, "task_id": tid} for i, tid in enumerate(selected, start=1)]
    task_map = {
        "selection_name": output.stem,
        "benchmark": "swebench_pro",
        "dataset_name": "swebench_pro",
        "source_path": "benchmarks/swebench_pro/dataset/test.jsonl",
        "source_sha256": _sha256(dataset_jsonl),
        "source_revision": baseline_payload.get("source_revision") if isinstance(baseline_payload, dict) else None,
        "split": "test",
        "selection_seed": seed,
        "task_count": count,
        "selection_notes": [
            f"KT recipient subset disjoint from {baseline_map.name}.",
            f"Pool size={len(instance_ids)}, baseline={len(baseline_ids)}, "
            f"remaining={len(remaining)}, selected={count}.",
            "Selection algorithm: random.Random(seed).sample(sorted(pool - baseline), count).",
            "Do not modify membership after publishing KT results.",
        ],
        "disjoint_from": baseline_map.name,
        "tasks": task_entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(task_map, indent=2, ensure_ascii=True) + "\n")
    print(f"wrote {output} | SWE-bench Pro recipient seed={seed} count={count}")


def build_polyglot_rest_ids(*, repo_root: Path, baseline_ids_file: Path, output: Path) -> None:
    """Write the list of polyglot task_ids NOT in the baseline medium subset.

    This is the full Aider-AI polyglot pool minus the 50 in medium. Output is
    a flat JSON list of "{lang}__{exercise}" strings, consumable by
    prepare_polyglot_dataset.py --subset-url <this-file>.
    """
    if not repo_root.is_dir():
        raise FileNotFoundError(
            f"polyglot-benchmark repo not found at {repo_root}; clone from "
            "https://github.com/Aider-AI/polyglot-benchmark.git"
        )
    languages = ("python", "rust", "go", "javascript", "java", "cpp")
    all_ids: list[str] = []
    for lang in languages:
        ex_root = repo_root / lang / "exercises" / "practice"
        if not ex_root.is_dir():
            continue
        for entry in sorted(ex_root.iterdir()):
            if entry.is_dir():
                all_ids.append(f"{lang}__{entry.name}")
    baseline = json.loads(baseline_ids_file.read_text(encoding="utf-8"))
    if not isinstance(baseline, list):
        raise ValueError(f"expected a JSON list of task IDs in {baseline_ids_file}")
    baseline_set = set(str(x) for x in baseline)
    rest = [tid for tid in all_ids if tid not in baseline_set]
    source_commit = _git_head(repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rest, indent=2, ensure_ascii=True) + "\n")
    meta = {
        "benchmark": "polyglot",
        "selection_name": output.stem,
        "source_repo": "Aider-AI/polyglot-benchmark",
        "source_branch": "main",
        "source_commit": source_commit,
        "count": len(rest),
        "ids_file": output.name,
        "disjoint_from": baseline_ids_file.name,
        "selection_algorithm": "sorted(all_polyglot_ids - baseline_medium_ids)",
        "usage_note": (
            "Pass this file to benchmarks/scripts/dataprep/prepare_polyglot_dataset.py "
            "--subset-url to materialise the full disjoint KT rest pool."
        ),
    }
    meta_path = output.with_name(f"{output.stem}.meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n")
    print(
        f"wrote {output} and {meta_path} | polyglot rest: "
        f"pool={len(all_ids)}, baseline={len(baseline_set)}, rest={len(rest)}"
    )


def build_polyglot_recipient_ids(
    *, rest_ids_file: Path, output: Path, meta_output: Path, seed: int = 1, count: int = 50
) -> None:
    """Write a deterministic 50-task Polyglot KT recipient split.

    The source pool is the committed rest-IDs file, which is already disjoint
    from the baseline medium-50 subset. Sampling from that file keeps the
    recipient set frozen and reproducible while remaining easy to materialize
    with prepare_polyglot_dataset.py --subset-url <this-file>.
    """
    rest = json.loads(rest_ids_file.read_text(encoding="utf-8"))
    if not isinstance(rest, list):
        raise ValueError(f"expected a JSON list of task IDs in {rest_ids_file}")
    rest_ids = sorted(str(x) for x in rest)
    if len(rest_ids) < count:
        raise ValueError(f"insufficient disjoint polyglot tasks: {len(rest_ids)} available, need {count}")
    selected = random.Random(seed).sample(rest_ids, count)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, indent=2, ensure_ascii=True) + "\n")

    parent_meta = _load_meta(rest_ids_file.with_name(f"{rest_ids_file.stem}.meta.json"))
    source_commit = str(parent_meta.get("source_commit") or "").strip()
    if not source_commit:
        raise ValueError(
            f"{rest_ids_file} is missing sibling metadata with source_commit; "
            "regenerate it with build_polyglot_rest_ids first"
        )
    meta = {
        "benchmark": "polyglot",
        "selection_name": output.stem,
        "source_repo": str(parent_meta.get("source_repo") or "Aider-AI/polyglot-benchmark"),
        "source_branch": str(parent_meta.get("source_branch") or "main"),
        "source_commit": source_commit,
        "selection_seed": seed,
        "count": count,
        "ids_file": output.name,
        "parent_pool_file": rest_ids_file.name,
        "disjoint_from": "polyglot_medium_50_seed0_ids.json",
        "selection_algorithm": ("random.Random(seed).sample(sorted(polyglot_rest_ids), count)"),
        "usage_note": (
            "Pass this file to benchmarks/scripts/dataprep/prepare_polyglot_dataset.py "
            "--subset-url to materialise the KT recipient dataset."
        ),
    }
    meta_output.write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n")
    print(
        f"wrote {output} and {meta_output} | polyglot recipient seed={seed} count={count} parent_pool={len(rest_ids)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate KT recipient subsets disjoint from baseline subsets.")
    parser.add_argument(
        "--polyglot-repo",
        default=os.environ.get("POLYGLOT_REPO", "/tmp/polyglot-benchmark"),
        help="Path to cloned Aider-AI/polyglot-benchmark",
    )
    parser.add_argument("--seed", type=int, default=1, help="Recipient selection seed")
    parser.add_argument("--count", type=int, default=50, help="Tasks per non-polyglot recipient")
    args = parser.parse_args()

    # ARC1: baseline on training split, KT recipient on evaluation split
    # (cross-split disjointness is automatic).
    build_arc_recipient(
        benchmark="arc1",
        source_dir=REPO_ROOT / "benchmarks/arc1/source/data/evaluation",
        baseline_map=REPO_ROOT / "benchmarks/arc1/task_maps/arc1_train_50_seed0.json",
        output=REPO_ROOT / "benchmarks/arc1/task_maps/arc1_eval_50_seed1_kt.json",
        seed=args.seed,
        count=args.count,
        source_repo="fchollet/ARC-AGI",
        source_branch="master",
    )

    # ARC2
    build_arc_recipient(
        benchmark="arc2",
        source_dir=REPO_ROOT / "benchmarks/arc2/source/data/evaluation",
        baseline_map=REPO_ROOT / "benchmarks/arc2/task_maps/arc2_train_50_seed0.json",
        output=REPO_ROOT / "benchmarks/arc2/task_maps/arc2_eval_50_seed1_kt.json",
        seed=args.seed,
        count=args.count,
        source_repo="arcprize/ARC-AGI-2",
        source_branch="main",
    )

    # SWE-bench Pro
    build_swebench_pro_recipient(
        dataset_jsonl=REPO_ROOT / "benchmarks/swebench_pro/dataset/test.jsonl",
        baseline_map=REPO_ROOT / "benchmarks/swebench_pro/task_maps/swebench_pro_test_50_seed0_v1.json",
        output=REPO_ROOT / "benchmarks/swebench_pro/task_maps/swebench_pro_test_50_seed1_kt.json",
        seed=args.seed,
        count=args.count,
    )

    # Polyglot: write rest-ids file and a stable 50-task KT recipient split;
    # follow-up step runs prepare_polyglot_dataset.py.
    build_polyglot_rest_ids(
        repo_root=Path(args.polyglot_repo),
        baseline_ids_file=REPO_ROOT / "benchmarks/polyglot/task_maps/polyglot_medium_50_seed0_ids.json",
        output=REPO_ROOT / "benchmarks/polyglot/task_maps/polyglot_rest_ids.json",
    )
    build_polyglot_recipient_ids(
        rest_ids_file=REPO_ROOT / "benchmarks/polyglot/task_maps/polyglot_rest_ids.json",
        output=REPO_ROOT / "benchmarks/polyglot/task_maps/polyglot_eval_50_seed1_kt_ids.json",
        meta_output=REPO_ROOT / "benchmarks/polyglot/task_maps/polyglot_eval_50_seed1_kt_ids.meta.json",
        seed=args.seed,
        count=args.count,
    )

    print()
    print("Next step for polyglot: materialise the KT recipient dataset via")
    print(
        "  uv run python benchmarks/scripts/dataprep/prepare_polyglot_dataset.py \\\n"
        "    --subset-url benchmarks/polyglot/task_maps/polyglot_eval_50_seed1_kt_ids.json \\\n"
        f"    --repo-cache {args.polyglot_repo} \\\n"
        "    --output data/polyglot_eval_50_seed1_kt.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

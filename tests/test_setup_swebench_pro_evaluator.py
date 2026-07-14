from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "benchmarks" / "scripts" / "dataprep" / "setup_swebench_pro_evaluator.py"
REAL_PATCHES_DIR = REPO_ROOT / "benchmarks" / "swebench_pro" / "evaluator_patches"
RESOURCE_LIMITS_PATCH = REAL_PATCHES_DIR / "per_instance_resource_limits.patch"


def _preimage_from_patch(patch_text: str) -> tuple[str, str]:
    """Reconstruct the (target_path, pre-image-file-content) a unified-diff patch
    expects, from the patch hunks alone — so a test can build a hermetic
    fixture without cloning the real upstream evaluator file.

    git apply locates a hunk by its context/deleted-line CONTENT (with line-number
    offset tolerance), so a file containing exactly the hunk's pre-image lines
    applies cleanly. We read the `+++ b/<path>` target and replay each hunk's
    context (' ') and deleted ('-') lines in order to materialize that pre-image.
    """
    target: str | None = None
    preimage: list[str] = []
    in_hunk = False
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            target = line[len("+++ b/") :]
            continue
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("diff --git") or line.startswith("--- "):
            in_hunk = False
            continue
        if line.startswith("+"):
            continue  # added line: not in pre-image
        # context (' ') or deleted ('-') line belongs to the pre-image
        preimage.append(line[1:] if line else "")
    assert target is not None, "patch had no +++ b/ target line"
    return target, "\n".join(preimage) + "\n"


def _load_script():
    spec = importlib.util.spec_from_file_location("setup_swebench_pro_evaluator_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_dataset(path: Path, instance_ids: list[str]) -> None:
    path.write_text(
        "".join(json.dumps({"instance_id": instance_id}) + "\n" for instance_id in instance_ids),
        encoding="utf-8",
    )


def _write_eval_asset_tree(root: Path, instance_ids: list[str]) -> None:
    (root / "helper_code").mkdir(parents=True)
    (root / "helper_code" / "image_uri.py").write_text("# stub\n", encoding="utf-8")
    (root / "swe_bench_pro_eval.py").write_text("# stub\n", encoding="utf-8")
    for instance_id in instance_ids:
        run_dir = root / "run_scripts" / instance_id
        run_dir.mkdir(parents=True)
        (run_dir / "run_script.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        (run_dir / "parser.py").write_text("# stub\n", encoding="utf-8")
        base_dir = root / "dockerfiles" / "base_dockerfile" / instance_id
        base_dir.mkdir(parents=True)
        (base_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
        instance_dir = root / "dockerfiles" / "instance_dockerfile" / instance_id
        instance_dir.mkdir(parents=True)
        (instance_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")


def test_verify_evaluator_assets_matches_dataset_instances(tmp_path: Path) -> None:
    module = _load_script()
    evaluator = tmp_path / "evaluator"
    dataset = tmp_path / "test.jsonl"
    _write_dataset(dataset, ["instance-a", "instance-b"])
    _write_eval_asset_tree(evaluator, ["instance-a", "instance-b", "extra-instance"])

    summary = module.verify_evaluator_assets(evaluator_dir=evaluator, dataset_path=dataset)

    assert summary.instance_count == 2
    assert summary.run_script_count == 3
    assert summary.extra_run_script_count == 1


def test_verify_evaluator_assets_rejects_missing_instance_files(tmp_path: Path) -> None:
    module = _load_script()
    evaluator = tmp_path / "evaluator"
    dataset = tmp_path / "test.jsonl"
    _write_dataset(dataset, ["instance-a", "instance-b"])
    _write_eval_asset_tree(evaluator, ["instance-a"])

    with pytest.raises(FileNotFoundError, match="instance-b"):
        module.verify_evaluator_assets(evaluator_dir=evaluator, dataset_path=dataset)


def test_compatibility_links_point_to_evaluator(tmp_path: Path) -> None:
    module = _load_script()
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    link = tmp_path / "legacy" / "SWE-bench_Pro-os"

    module.ensure_compatibility_links(evaluator_dir=evaluator, links=(link,))

    assert link.is_symlink()
    assert link.resolve() == evaluator.resolve()


def test_compatibility_links_replace_stale_generated_cache_dir(tmp_path: Path) -> None:
    module = _load_script()
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    link = tmp_path / "legacy" / "SWE-bench_Pro-os"
    stale_cache = link / "helper_code" / "__pycache__"
    stale_cache.mkdir(parents=True)
    (stale_cache / "image_uri.cpython-312.pyc").write_bytes(b"cache")

    module.ensure_compatibility_links(evaluator_dir=evaluator, links=(link,))

    assert link.is_symlink()
    assert link.resolve() == evaluator.resolve()


def test_compatibility_links_replace_empty_generated_dir(tmp_path: Path) -> None:
    module = _load_script()
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    link = tmp_path / "legacy" / "SWE-bench_Pro-os"
    link.mkdir(parents=True)

    module.ensure_compatibility_links(evaluator_dir=evaluator, links=(link,))

    assert link.is_symlink()
    assert link.resolve() == evaluator.resolve()


def test_compatibility_links_refuse_existing_real_directory(tmp_path: Path) -> None:
    module = _load_script()
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    link = tmp_path / "legacy" / "SWE-bench_Pro-os"
    link.mkdir(parents=True)
    (link / "README.md").write_text("do not replace\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        module.ensure_compatibility_links(evaluator_dir=evaluator, links=(link,))


def test_verify_evaluator_revision_accepts_non_git_marker(tmp_path: Path) -> None:
    module = _load_script()
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    (evaluator / "swe_bench_pro_eval.py").write_text("# stub\n", encoding="utf-8")
    (evaluator / module.REVISION_MARKER).write_text(module.DEFAULT_REVISION + "\n", encoding="utf-8")

    module.verify_evaluator_revision(evaluator_dir=evaluator, revision=module.DEFAULT_REVISION)


def test_verify_evaluator_revision_rejects_unmarked_non_git_checkout(tmp_path: Path) -> None:
    module = _load_script()
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    (evaluator / "swe_bench_pro_eval.py").write_text("# stub\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no revision marker"):
        module.verify_evaluator_revision(evaluator_dir=evaluator, revision=module.DEFAULT_REVISION)


def test_apply_kcsi_evaluator_patches_rolls_back_on_mid_list_failure(tmp_path: Path) -> None:
    """When patch N applies but patch N+1 fails, the previously-applied
    patches must be reverted so the evaluator checkout isn't left
    half-patched (which would confuse `verify_evaluator_revision` on the
    next setup run)."""
    import subprocess

    module = _load_script()

    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=evaluator, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=evaluator, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=evaluator, check=True)

    target_a = evaluator / "a.py"
    target_a.write_text("ORIGINAL_A\n", encoding="utf-8")
    target_b = evaluator / "b.py"
    target_b.write_text("ORIGINAL_B\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=evaluator, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=evaluator, check=True)

    patches = tmp_path / "patches"
    patches.mkdir()
    # patch_01 cleanly modifies a.py (it can apply against ORIGINAL_A)
    (patches / "01_clean.patch").write_text(
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1 +1 @@\n"
        "-ORIGINAL_A\n"
        "+PATCHED_A  # SWARMS PATCH: clean change\n",
        encoding="utf-8",
    )
    # patch_02 references context that doesn't exist in b.py -> cannot apply
    (patches / "02_broken.patch").write_text(
        "diff --git a/b.py b/b.py\n"
        "--- a/b.py\n"
        "+++ b/b.py\n"
        "@@ -1 +1 @@\n"
        "-NEVER_PRESENT\n"
        "+REPLACED  # SWARMS PATCH: broken change\n",
        encoding="utf-8",
    )

    with pytest.raises(subprocess.CalledProcessError):
        module.apply_kcsi_evaluator_patches(evaluator_dir=evaluator, patches_dir=patches)

    # After the broken patch fails, a.py must be back to its original content
    # (i.e., patch_01 was reverted on failure).
    assert target_a.read_text(encoding="utf-8") == "ORIGINAL_A\n", (
        "apply_kcsi_evaluator_patches left the evaluator checkout half-patched after "
        "a mid-list patch failure; this defeats the idempotency guard and "
        "produces confusing partial diffs on re-setup"
    )
    assert target_b.read_text(encoding="utf-8") == "ORIGINAL_B\n"
    # A rollback must not leave a patch-state marker claiming patches are applied.
    assert not (evaluator / module.PATCH_STATE_MARKER).is_file(), (
        "apply_kcsi_evaluator_patches wrote a patch-state file after rolling back; "
        "the state would falsely claim patches are present on the next run"
    )


def test_apply_kcsi_evaluator_patches_idempotent_via_marker(tmp_path: Path) -> None:
    """If a target file already contains 'SWARMS PATCH:', the patch is
    skipped — re-running setup is a no-op."""
    import subprocess

    module = _load_script()

    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=evaluator, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=evaluator, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=evaluator, check=True)

    # Target file ALREADY has the marker — patch should be skipped.
    target = evaluator / "a.py"
    target.write_text("PATCHED_A  # SWARMS PATCH: clean change\n", encoding="utf-8")

    patches = tmp_path / "patches"
    patches.mkdir()
    (patches / "01_would_break.patch").write_text(
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-ORIGINAL_A\n+PATCHED_A\n",
        encoding="utf-8",
    )

    # Should not raise: marker present → skip.
    module.apply_kcsi_evaluator_patches(evaluator_dir=evaluator, patches_dir=patches)
    assert target.read_text(encoding="utf-8") == "PATCHED_A  # SWARMS PATCH: clean change\n"
    # The skip path still records the patch in the state file, so the drift
    # guard has a hash to compare against on subsequent runs.
    patch_state = json.loads((evaluator / module.PATCH_STATE_MARKER).read_text(encoding="utf-8"))
    assert patch_state["patches"] == [
        {
            "name": "01_would_break.patch",
            "sha256": module._patch_digest(patches / "01_would_break.patch"),
            "target": "a.py",
        }
    ]


def test_real_per_instance_resource_limits_patch_applies_and_lands_marker(tmp_path: Path) -> None:
    """The shipped evaluator_patches/per_instance_resource_limits.patch must
    apply cleanly via apply_kcsi_evaluator_patches() and leave the `KCSI PATCH:`
    marker for both the eval timeout and the container memory limit it adds.

    Both changes live in ONE patch file (not two) because
    apply_kcsi_evaluator_patches()'s idempotency check keys on the *target
    file* containing a `KCSI PATCH:` marker, not on individual patch content:
    a second patch touching an already-marked target file is silently
    skipped rather than applied (see issue #1010). Since both changes target
    swe_bench_pro_eval.py, they must ship as hunks of a single patch file.

    Hermetic: we do NOT clone upstream scaleapi/SWE-bench_Pro-os. Instead we
    reconstruct the exact pre-image the patch's hunks expect from the patch file
    itself (git apply locates hunks by content, not absolute line number), commit
    it into a throwaway git checkout, and run the real apply function against it.
    This catches the patch silently rotting against its own declared context.
    """
    import subprocess

    module = _load_script()

    assert RESOURCE_LIMITS_PATCH.is_file(), f"missing shipped patch: {RESOURCE_LIMITS_PATCH}"
    patch_text = RESOURCE_LIMITS_PATCH.read_text(encoding="utf-8")
    target_rel, preimage = _preimage_from_patch(patch_text)
    assert target_rel == "swe_bench_pro_eval.py"
    assert "KCSI PATCH:" not in preimage, "pre-image already contains the marker"

    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=evaluator, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=evaluator, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=evaluator, check=True)

    target_path = evaluator / target_rel
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(preimage, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=evaluator, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "preimage"], cwd=evaluator, check=True)

    # Apply ONLY this patch (point patches_dir at a temp dir holding a copy) so
    # the test is independent of any other shipped patches.
    patches = tmp_path / "patches"
    patches.mkdir()
    (patches / RESOURCE_LIMITS_PATCH.name).write_text(patch_text, encoding="utf-8")

    module.apply_kcsi_evaluator_patches(evaluator_dir=evaluator, patches_dir=patches)

    patched = target_path.read_text(encoding="utf-8")
    assert "KCSI PATCH:" in patched, "marker did not land after applying the real patch"
    assert "KCSI_EVAL_PER_INSTANCE_TIMEOUT_SEC" in patched
    assert "KCSI_EVAL_PER_INSTANCE_MEM_LIMIT" in patched
    assert 'run_kwargs["mem_limit"]' in patched
    patch_state = json.loads((evaluator / module.PATCH_STATE_MARKER).read_text(encoding="utf-8"))
    assert patch_state["patches"] == [
        {
            "name": RESOURCE_LIMITS_PATCH.name,
            "sha256": module._patch_digest(patches / RESOURCE_LIMITS_PATCH.name),
            "target": "swe_bench_pro_eval.py",
        }
    ]

    # Idempotency: re-running is a no-op because the marker is now present, and
    # the recorded patch state stays identical (matching hash → no drift).
    module.apply_kcsi_evaluator_patches(evaluator_dir=evaluator, patches_dir=patches)
    assert target_path.read_text(encoding="utf-8") == patched
    assert json.loads((evaluator / module.PATCH_STATE_MARKER).read_text(encoding="utf-8")) == patch_state


def test_apply_kcsi_evaluator_patches_raises_on_patch_hash_drift(tmp_path: Path) -> None:
    """If a target still carries the patch marker but the recorded sha256 no
    longer matches the patch file's bytes, setup must refuse (drift guard) and
    roll back anything it applied earlier this run rather than silently leaving
    a half-patched checkout that disagrees with the shipped patch."""
    import subprocess

    module = _load_script()

    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=evaluator, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=evaluator, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=evaluator, check=True)

    # a.py is clean: 01_clean.patch applies fresh this run.
    target_a = evaluator / "a.py"
    target_a.write_text("ORIGINAL_A\n", encoding="utf-8")
    # b.py already carries the marker (patched in a prior run).
    target_b = evaluator / "b.py"
    target_b.write_text("PATCHED_B  # KCSI PATCH: timeout\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=evaluator, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=evaluator, check=True)

    patches = tmp_path / "patches"
    patches.mkdir()
    (patches / "01_clean.patch").write_text(
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n"
        "-ORIGINAL_A\n+PATCHED_A  # KCSI PATCH: clean change\n",
        encoding="utf-8",
    )
    (patches / "02_drift.patch").write_text(
        "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n"
        "-ORIGINAL_B\n+PATCHED_B  # KCSI PATCH: timeout v2\n",
        encoding="utf-8",
    )

    # Record a stale hash for 02_drift.patch so the marker-present branch detects
    # that the shipped patch bytes have changed since b.py was patched.
    module._write_patch_state(
        evaluator,
        [{"name": "02_drift.patch", "sha256": "deadbeef", "target": "b.py"}],
    )

    with pytest.raises(RuntimeError, match="records a different hash"):
        module.apply_kcsi_evaluator_patches(evaluator_dir=evaluator, patches_dir=patches)

    # Transactional: 01_clean.patch (applied this run) must be rolled back so the
    # checkout is left consistent for the reinstall the error tells you to do.
    assert target_a.read_text(encoding="utf-8") == "ORIGINAL_A\n", (
        "drift guard fired but did not roll back the patch it applied earlier "
        "this run; the checkout is left half-patched"
    )


def test_read_patch_state_tolerates_malformed_state_file(tmp_path: Path) -> None:
    """A corrupt or unexpected .kcsi-evaluator-patches.json must degrade to an
    empty mapping (drift detection disabled) rather than crashing setup."""
    module = _load_script()
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    state_path = evaluator / module.PATCH_STATE_MARKER

    # Missing file → empty mapping.
    assert module._read_patch_state(evaluator) == {}

    # Not valid JSON → empty mapping.
    state_path.write_text("{not json", encoding="utf-8")
    assert module._read_patch_state(evaluator) == {}

    # Valid JSON but no 'patches' list → empty mapping.
    state_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    assert module._read_patch_state(evaluator) == {}

    # 'patches' present but malformed / partial items are skipped; complete
    # records survive.
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "patches": [
                    "not-a-dict",
                    {"name": "x.patch"},  # missing sha256 + target
                    {"name": "ok.patch", "sha256": "abc", "target": "f.py"},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert module._read_patch_state(evaluator) == {"ok.patch": {"sha256": "abc", "target": "f.py"}}

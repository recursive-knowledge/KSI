"""#923 M2 (runtime): a READ-ONLY embedding model cache loads lock-free offline.

The PR mounts ``runtime_state/model_cache`` into solver containers read-only and
sets ``HF_HUB_OFFLINE=1`` / ``TRANSFORMERS_OFFLINE=1`` (container_args.ts). The
failure mode that motivated this — and that the two source-pin JS tests only
assert *textually* — is that a model load against a read-only cache would try to
take a ``.lock`` file or write ETag metadata and raise ``PermissionError`` /
``OSError`` on the RO mount, silently degrading semantic retrieval to FTS.

This test proves the *mechanism* the container relies on at the library layer:
``huggingface_hub`` (the component that performs the lock/ETag writes; the same
library ``sentence-transformers`` delegates downloads to) resolves a file from a
fully ``chmod a-w`` cache, with the offline env vars set, WITHOUT attempting any
write — so the load succeeds rather than raising on the RO mount.

What this does NOT cover (stated for honesty): it does not spin up the actual
container, and it does not exercise ``sentence-transformers``/``torch`` (absent
from the base test env; the default model ``google/embeddinggemma-300m`` is
gated + large, so a real cold-cache download is infeasible in CI). The offline +
RO read property is a ``huggingface_hub`` guarantee shared by host and container
since both use the same library and env vars, so proving it here is a faithful
runtime check of the container's code path.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

hf = pytest.importorskip("huggingface_hub")


def _build_synthetic_hub_cache(hub_dir: Path, repo_id: str, filename: str, body: str) -> None:
    """Materialize the documented huggingface_hub cache layout (no network).

    models--<org>--<name>/{blobs/<sha>, snapshots/<rev>/<file> -> blob, refs/main}
    """
    org, name = repo_id.split("/")
    model_dir = hub_dir / f"models--{org}--{name}"
    rev = "0123456789abcdef0123456789abcdef01234567"
    blobs = model_dir / "blobs"
    snap = model_dir / "snapshots" / rev
    refs = model_dir / "refs"
    for d in (blobs, snap, refs):
        d.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(rev)
    blob = blobs / "deadbeefcafe"
    blob.write_text(body)
    link = snap / filename
    try:
        os.symlink(os.path.relpath(blob, snap), link)
    except OSError:  # platforms without symlink perms — fall back to a copy
        link.write_text(body)


def _chmod_tree_readonly(root: Path) -> None:
    for p in sorted(root.rglob("*"), reverse=True):
        p.chmod(p.stat().st_mode & ~0o222)
    root.chmod(root.stat().st_mode & ~0o222)


def _restore_tree_writable(root: Path) -> None:
    root.chmod(root.stat().st_mode | 0o200)
    for p in root.rglob("*"):
        try:
            p.chmod(p.stat().st_mode | 0o200)
        except OSError:
            pass


def test_readonly_offline_cache_loads_without_write(monkeypatch, tmp_path):
    repo_id = "sentence-transformers/dummy-offline-model"
    filename = "config.json"
    body = "synthetic-config-bytes"

    hf_home = tmp_path / "huggingface"
    hub_dir = hf_home / "hub"
    hub_dir.mkdir(parents=True)
    _build_synthetic_hub_cache(hub_dir, repo_id, filename, body)

    # Exactly the env the container sets (container_args.ts) so the load is
    # offline: no network HEAD, no lock-file, no ETag metadata write.
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    _chmod_tree_readonly(tmp_path)
    try:
        # try_to_load_from_cache is the pure cache-resolution helper.
        cached = hf.try_to_load_from_cache(repo_id, filename, cache_dir=str(hub_dir))
        assert isinstance(cached, str), f"expected a cached path, got {cached!r}"

        # hf_hub_download is the path sentence-transformers uses; local_files_only
        # forces the offline branch that must NOT take a lock or write ETag meta.
        resolved = hf.hf_hub_download(repo_id, filename, cache_dir=str(hub_dir), local_files_only=True)
        assert Path(resolved).read_text() == body  # loaded from the RO cache
    finally:
        _restore_tree_writable(tmp_path)

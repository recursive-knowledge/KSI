from __future__ import annotations

import os
import sys
import types


def test_point_embeddings_cache_at_sets_matching_env(tmp_path):
    from kcsi.memory.embeddings import point_embeddings_cache_at

    # point_embeddings_cache_at() writes os.environ directly (not via
    # monkeypatch), and monkeypatch.delenv(..., raising=False) records
    # nothing to restore when the var is already unset — so an unset->set
    # transition here would leak this tmp_path-scoped value into the rest of
    # the pytest process. Save/restore explicitly instead of relying on
    # monkeypatch's bookkeeping.
    prior_hf = os.environ.pop("HF_HOME", None)
    prior_st = os.environ.pop("SENTENCE_TRANSFORMERS_HOME", None)
    try:
        cache_root = tmp_path / "model_cache"
        point_embeddings_cache_at(cache_root)

        assert os.environ["HF_HOME"] == str(cache_root / "huggingface")
        assert os.environ["SENTENCE_TRANSFORMERS_HOME"] == str(cache_root / "sentence-transformers")
        assert (cache_root / "huggingface").is_dir()
        assert (cache_root / "sentence-transformers").is_dir()
    finally:
        if prior_hf is None:
            os.environ.pop("HF_HOME", None)
        else:
            os.environ["HF_HOME"] = prior_hf
        if prior_st is None:
            os.environ.pop("SENTENCE_TRANSFORMERS_HOME", None)
        else:
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = prior_st


def test_wait_ready_false_on_load_error(monkeypatch):
    # Inject a fake sentence_transformers whose SentenceTransformer raises, so
    # the in-method import fails deterministically without network.
    fake = types.ModuleType("sentence_transformers")

    def boom(*args, **kwargs):
        raise RuntimeError("simulated load failure")

    fake.SentenceTransformer = boom
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    from kcsi.memory.embeddings import Embedder

    emb = Embedder(background=False)  # eager load captures the failure
    assert emb.wait_ready(timeout=5) is False
    assert emb.is_ready is False

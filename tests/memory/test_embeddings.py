"""Tests for src/ksi/memory/embeddings.py — embedding dimension validation."""

import sys
import threading
import time
from unittest.mock import MagicMock

import pytest


def _make_embedder_with_short_vectors(dim_output, dim_configured):
    """Create an Embedder with a mock SentenceTransformer that returns short vectors."""
    mock_st_class = MagicMock()
    mock_model = MagicMock()
    mock_st_class.return_value = mock_model

    # Inject mock into the sentence_transformers module
    fake_module = MagicMock()
    fake_module.SentenceTransformer = mock_st_class
    sys.modules["sentence_transformers"] = fake_module

    try:
        # Force reimport to pick up the mock
        import importlib

        import ksi.memory.embeddings as emb_module

        importlib.reload(emb_module)

        embedder = emb_module.Embedder(model_name="fake", dimensions=dim_configured)
        return embedder, mock_model
    finally:
        # Cleanup: remove mock module so it doesn't affect other tests
        del sys.modules["sentence_transformers"]


def test_embed_dimension_assertion():
    embedder, mock_model = _make_embedder_with_short_vectors(512, 768)
    mock_model.encode.return_value = [0.1] * 512
    with pytest.raises(ValueError, match="dimension"):
        embedder.embed("test text")


def test_embed_batch_dimension_assertion():
    embedder, mock_model = _make_embedder_with_short_vectors(512, 768)
    mock_model.encode.return_value = [[0.1] * 512, [0.2] * 512, [0.3] * 512]
    with pytest.raises(ValueError, match="dimension"):
        embedder.embed_batch(["a", "b", "c"])


def test_embed_passes_when_dimensions_sufficient():
    embedder, mock_model = _make_embedder_with_short_vectors(768, 768)

    class FakeArray(list):
        """List subclass that supports slicing and .tolist() like numpy."""

        def __getitem__(self, key):
            result = super().__getitem__(key)
            if isinstance(key, slice):
                return FakeArray(result)
            return result

        def tolist(self):
            return list(self)

    mock_model.encode.return_value = FakeArray([0.1] * 768)
    result = embedder.embed("test text")
    assert len(result) == 768


class _FakeArray(list):
    """List subclass that supports slicing and .tolist() like a numpy vector."""

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(key, slice):
            return _FakeArray(result)
        return result

    def tolist(self):
        return list(self)


def _make_embedder_with_serialization_probe(dim=768):
    """Build an Embedder whose encode() records max observed concurrency.

    encode() bumps a shared counter on entry, sleeps briefly so overlapping
    callers would collide if unguarded, records the peak concurrency, then
    returns a deterministic vector derived from the input so results can be
    checked for correctness. Returns (embedder, state) where state carries
    ``max_concurrency``.
    """
    state = {"active": 0, "max_concurrency": 0}
    guard = threading.Lock()

    def fake_encode(text_or_texts, normalize_embeddings=True):
        with guard:
            state["active"] += 1
            state["max_concurrency"] = max(state["max_concurrency"], state["active"])
        try:
            time.sleep(0.01)
            if isinstance(text_or_texts, str):
                seed = float(len(text_or_texts))
                return _FakeArray([seed] * dim)
            return [_FakeArray([float(len(t))] * dim) for t in text_or_texts]
        finally:
            with guard:
                state["active"] -= 1

    mock_st_class = MagicMock()
    mock_model = MagicMock()
    mock_model.encode.side_effect = fake_encode
    mock_st_class.return_value = mock_model

    fake_module = MagicMock()
    fake_module.SentenceTransformer = mock_st_class
    sys.modules["sentence_transformers"] = fake_module
    try:
        import importlib

        import ksi.memory.embeddings as emb_module

        importlib.reload(emb_module)
        embedder = emb_module.Embedder(model_name="fake", dimensions=dim)
        return embedder, state
    finally:
        del sys.modules["sentence_transformers"]


def test_concurrent_embed_is_serialized_and_correct():
    """Many threads calling embed()/embed_batch() on one shared Embedder must
    not run encode() concurrently (the fix serializes the torch forward pass to
    avoid CPU oversubscription) and must return correct, consistent results."""
    embedder, state = _make_embedder_with_serialization_probe(dim=768)

    errors: list[BaseException] = []
    results: list[tuple[str, object]] = []
    barrier = threading.Barrier(16)

    def call_embed(text):
        try:
            barrier.wait()
            results.append(("single", embedder.embed(text)))
        except BaseException as exc:  # noqa: BLE001 - surface any thread error
            errors.append(exc)

    def call_embed_batch(texts):
        try:
            barrier.wait()
            results.append(("batch", embedder.embed_batch(texts)))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = []
    for i in range(8):
        threads.append(threading.Thread(target=call_embed, args=(f"text-{i}",)))
        threads.append(threading.Thread(target=call_embed_batch, args=([f"b-{i}", f"bb-{i}"],)))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    # (a) No thread raised.
    assert not errors, f"Concurrent embedding raised: {errors!r}"
    # Lock held around encode() => at most one forward pass at a time.
    assert state["max_concurrency"] == 1, (
        f"encode() ran on {state['max_concurrency']} threads at once; the "
        "encode lock did not serialize the shared model"
    )
    # (b) Results are correct/consistent: every single embed is a 768-vector
    # whose value encodes the input length (proves no cross-thread scramble).
    assert len(results) == 16
    for kind, res in results:
        if kind == "single":
            assert len(res) == 768
        else:
            assert len(res) == 2
            assert all(len(v) == 768 for v in res)

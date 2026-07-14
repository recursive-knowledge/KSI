"""#923 M2: engine pre-warms the shared embedding cache before launching containers.

Tests that GenerationalOrchestrator points HF_HOME / SENTENCE_TRANSFORMERS_HOME at
RUNTIME_STATE_DIR/model_cache BEFORE constructing the Embedder, and that run()
blocks on wait_ready(timeout=600) before the generation loop — a False return is
non-fatal (logs a warning, never raises).
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from kcsi.layout import RUNTIME_STATE_DIR
from kcsi.models import GenerationConfig
from kcsi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from kcsi.tokens import LLMResponse, TokenUsage


@contextmanager
def _restored_cache_env():
    """Save/restore HF_HOME and SENTENCE_TRANSFORMERS_HOME around a test.

    The engine's Embedder pre-warm path (point_embeddings_cache_at) writes
    these vars directly to os.environ, not via monkeypatch. monkeypatch.delenv(
    ..., raising=False) on an already-unset var records nothing to restore at
    teardown, so relying on monkeypatch alone leaks a tmp_path-scoped value
    into the rest of the pytest process. Restore explicitly instead.
    """
    prior_hf = os.environ.pop("HF_HOME", None)
    prior_st = os.environ.pop("SENTENCE_TRANSFORMERS_HOME", None)
    try:
        yield
    finally:
        if prior_hf is None:
            os.environ.pop("HF_HOME", None)
        else:
            os.environ["HF_HOME"] = prior_hf
        if prior_st is None:
            os.environ.pop("SENTENCE_TRANSFORMERS_HOME", None)
        else:
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = prior_st


def _make_vector_enabled_engine(monkeypatch, tmp_path) -> GenerationalOrchestrator:
    """Build the smallest engine config that reaches the Embedder construction block.

    Gate: ``self._knowledge is not None and self._vector_enabled and
    not self.config.no_memory`` — vector search is opt-in, so this requires
    ``require_vector=True`` (with KCSI_DISABLE_VECTOR unset). ``--require-vector``
    also fail-fasts unless the knowledge store's vec index came up, so
    ``_init_vec`` is stubbed to report the index ready without needing a real
    sqlite-vec build in the test environment.
    """
    monkeypatch.setattr(
        "kcsi.memory.knowledge_store.KnowledgeStore._init_vec",
        lambda self, dim: setattr(self, "_vec_enabled", True),
    )
    # The engine writes MEMORY_ENABLE_SEMANTIC_SEARCH directly; delenv so
    # monkeypatch restores the process env after the test (the direct write
    # would otherwise leak into sibling tests).
    monkeypatch.delenv("MEMORY_ENABLE_SEMANTIC_SEARCH", raising=False)
    db_path = str(tmp_path / "knowledge.sqlite")
    runtime = MagicMock()
    evaluator = MagicMock()
    llm = MagicMock()
    llm.call.return_value = LLMResponse(
        text=json.dumps(
            {
                "transferable_insights": [],
                "pitfalls": [],
                "checks": [],
                "evidence_post_ids": [],
            }
        ),
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        knowledge_db_path=db_path,
        experiment_name="test_prewarm",
        no_memory=False,
        require_vector=True,
    )
    return GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )


def test_engine_points_embedder_cache_at_shared_dir(monkeypatch, tmp_path):
    """#923 M2: the host embedder's HF cache env must point at the shared
    runtime_state/model_cache dir (the dir containers mount) BEFORE the
    Embedder is constructed, so the host populates it directly."""
    captured: dict[str, str | None] = {}

    class FakeEmbedder:
        def __init__(self, *args, **kwargs):
            captured["HF_HOME"] = os.environ.get("HF_HOME")
            captured["SENTENCE_TRANSFORMERS_HOME"] = os.environ.get("SENTENCE_TRANSFORMERS_HOME")

        is_ready = True

        def wait_ready(self, timeout=None):
            return True

    monkeypatch.setattr("kcsi.memory.embeddings.Embedder", FakeEmbedder)
    # Ensure the vector-disable gate is open so the Embedder block is entered.
    monkeypatch.delenv("KCSI_DISABLE_VECTOR", raising=False)

    with _restored_cache_env():
        _make_vector_enabled_engine(monkeypatch, tmp_path)

        assert captured, "FakeEmbedder.__init__ was never called — embedder block not reached"
        assert captured["HF_HOME"] == str(RUNTIME_STATE_DIR / "model_cache" / "huggingface")
        assert captured["SENTENCE_TRANSFORMERS_HOME"] == str(
            RUNTIME_STATE_DIR / "model_cache" / "sentence-transformers"
        )


def test_run_blocks_on_wait_ready_and_false_is_nonfatal(monkeypatch, tmp_path, caplog):
    """#923 M2: run() must block on the embedder's wait_ready(timeout=600) before
    the generation loop, and a False return (model never loaded) is non-fatal —
    it logs a warning and continues rather than raising. Characterization test
    pinning the run()-side contract."""

    class _ConstructEmbedder:
        def __init__(self, *args, **kwargs):
            pass

        is_ready = True

        def wait_ready(self, timeout=None):
            return True

    # Patch the constructed embedder so engine build stays offline/deterministic.
    monkeypatch.setattr("kcsi.memory.embeddings.Embedder", _ConstructEmbedder)
    monkeypatch.delenv("KCSI_DISABLE_VECTOR", raising=False)

    with _restored_cache_env():
        engine = _make_vector_enabled_engine(monkeypatch, tmp_path)
        assert engine._embedder is not None

        # Replace the live embedder with a recorder whose wait_ready returns False,
        # exercising the non-fatal warning path in run().
        calls: list[float | None] = []

        class _RecordingEmbedder:
            def wait_ready(self, timeout=None):
                calls.append(timeout)
                return False

        engine._embedder = _RecordingEmbedder()
        # The embedder path is now opt-in (require_vector=True), which adds an
        # orthogonal run-summary guard that raises when zero embeddings were
        # written. This test targets the wait_ready block (non-fatal warning),
        # not that guard, so pre-satisfy it — empty tasks write no embeddings.
        engine._vector_embedding_count = 1

        # Empty task list reaches the wait block (right after the accumulator reset)
        # then short-circuits the generation loop ("no remaining tasks") and returns,
        # so this stays offline and never touches a container.
        with caplog.at_level(logging.WARNING, logger="kcsi.orchestrator.engine"):
            traces = engine.run([])

        assert calls == [600], f"run() must call wait_ready(timeout=600); got {calls}"
        assert traces == []  # run() did not raise; the False path only warned
        assert any("embedding model not ready before launch" in rec.getMessage() for rec in caplog.records), (
            "expected a non-fatal warning when wait_ready returns False"
        )

        if engine._knowledge is not None:
            engine._knowledge.close()


def test_vector_off_by_default_uses_fts(monkeypatch, tmp_path):
    """FTS5 is the default retrieval path: without --require-vector the engine
    opens the knowledge store with the vec index disabled and constructs no
    embedder. Proves the gate is off regardless of whether sqlite-vec /
    sentence-transformers happen to be installed."""

    class _BoomEmbedder:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Embedder must not be constructed on the FTS-default path")

    monkeypatch.setattr("kcsi.memory.embeddings.Embedder", _BoomEmbedder)
    # If enable_vec were (wrongly) True, _init_vec would flip _vec_enabled to
    # True; asserting it stays False proves the store was opened enable_vec=False.
    monkeypatch.setattr(
        "kcsi.memory.knowledge_store.KnowledgeStore._init_vec",
        lambda self, dim: setattr(self, "_vec_enabled", True),
    )
    monkeypatch.delenv("KCSI_DISABLE_VECTOR", raising=False)
    monkeypatch.delenv("MEMORY_ENABLE_SEMANTIC_SEARCH", raising=False)

    db_path = str(tmp_path / "knowledge.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        knowledge_db_path=db_path,
        experiment_name="test_fts_default",
        no_memory=False,
        # require_vector defaults to False — the FTS-by-default path.
    )
    with _restored_cache_env():
        engine = GenerationalOrchestrator(
            config=config,
            runtime=MagicMock(),
            evaluator=MagicMock(),
            llm=MagicMock(),
            persistence=NoopPersistence(),
        )
        assert engine._embedder is None
        assert engine._knowledge is not None
        assert getattr(engine._knowledge, "_vec_enabled", False) is False
        # enable_vec=False means the vec virtual table is never created, so no
        # embeddings can be written and the store is genuinely FTS-only.
        row = engine._knowledge._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='knowledge_vec'"
        ).fetchone()
        assert row is None, "knowledge_vec table must not exist on the FTS-default path"
        # The host is authoritative: the in-container query tool is told to stay
        # on FTS too, so a resumed DB with a stale vec table can't re-enable it.
        assert os.environ.get("MEMORY_ENABLE_SEMANTIC_SEARCH") == "0"
        engine._knowledge.close()


def test_require_vector_run_summary_raises_on_zero_embeddings(monkeypatch, tmp_path):
    """The run-summary guard: with --require-vector, a run that wrote zero
    embeddings must fail loudly rather than silently ship an empty vec index.
    Exercises the ``engine.py`` guard that the wait_ready test pre-satisfies."""

    class _ReadyEmbedder:
        def __init__(self, *args, **kwargs):
            pass

        is_ready = True

        def wait_ready(self, timeout=None):
            return True

    monkeypatch.setattr("kcsi.memory.embeddings.Embedder", _ReadyEmbedder)
    monkeypatch.delenv("KCSI_DISABLE_VECTOR", raising=False)

    with _restored_cache_env():
        engine = _make_vector_enabled_engine(monkeypatch, tmp_path)
        assert engine._vector_required is True
        # Empty task list writes no embeddings, so the guard must fire.
        engine._vector_embedding_count = 0
        try:
            with pytest.raises(RuntimeError, match="no embeddings were written"):
                engine.run([])
        finally:
            if engine._knowledge is not None:
                engine._knowledge.close()

"""Smoke test: embedding model loads cleanly on the pinned stack.

Guards against a silent misload — the try/except in
``src/ksi/memory/embeddings.py`` only warns on failure, so a broken
``sentence-transformers`` upgrade can silently degrade vector search to
FTS-only for an entire campaign (see commit history for the 4.x Pooling
break that motivated the version pins).

The test is skipped when ``HF_TOKEN`` is unset because the default
embedding model (``google/embeddinggemma-300m``) is gated on HuggingFace.
Machines that are properly credentialed will exercise the full load path.
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(
    not os.getenv("HF_TOKEN"),
    reason="HF_TOKEN required to fetch gated google/embeddinggemma-300m",
)
def test_embedder_loads_eagerly() -> None:
    """Eagerly load the embedder and assert it reports ready.

    ``background=False`` forces the model to load in the constructor so
    any incompatibility between the installed ``sentence-transformers``
    release and the cached model config raises immediately rather than
    silently flipping ``is_ready`` to False behind a warning.
    """
    from ksi.memory.embeddings import Embedder

    embedder = Embedder(background=False)
    assert embedder.is_ready, (
        "Embedder failed to load on the pinned sentence-transformers "
        "stack. Check pyproject.toml [project.optional-dependencies].memory "
        "for version drift (the 4.x+ Pooling constructor regression is the "
        "typical culprit)."
    )

    # Sanity-check: encode a short string and confirm dimensionality.
    vec = embedder.embed("the quick brown fox")
    assert len(vec) == embedder.dimensions

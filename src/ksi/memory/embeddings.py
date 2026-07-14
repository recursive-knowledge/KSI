"""Embedding model wrapper for agent memory vectorization."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised in minimal containers

    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


# Load .env so the embedding model name can be overridden without editing
# source. load_dotenv() is idempotent.
load_dotenv()

_DEFAULT_EMBEDDING_MODEL = os.environ.get("KSI_EMBEDDING_MODEL", "google/embeddinggemma-300m")

# This project only uses sentence-transformers through PyTorch. Some local
# environments have Keras 3 installed, which can make Transformers import its
# TensorFlow path and fail before sentence-transformers loads. Disable TF unless
# the user explicitly opted in before importing this module.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

log = logging.getLogger(__name__)


def point_embeddings_cache_at(cache_root: Path) -> None:
    """Point this process's HF / sentence-transformers caches at a shared dir.

    The host populates this dir via direct filesystem access so the
    solver containers can mount it read-only (cross-container poisoning
    prevention). The env mapping mirrors the container's in
    runtime_runner/src/container_runner.ts (HF_HOME -> <root>/huggingface,
    SENTENCE_TRANSFORMERS_HOME -> <root>/sentence-transformers).
    """
    hf = cache_root / "huggingface"
    st = cache_root / "sentence-transformers"
    hf.mkdir(parents=True, exist_ok=True)
    st.mkdir(parents=True, exist_ok=True)
    # This unconditionally redirects the caches into the mount dir (required so
    # the container can mount exactly this dir read-only). If an operator had
    # pre-set HF_HOME / SENTENCE_TRANSFORMERS_HOME elsewhere, surface the
    # override at debug level so a "why isn't my pre-warmed cache used?" is
    # diagnosable rather than silent.
    for var, new in (("HF_HOME", str(hf)), ("SENTENCE_TRANSFORMERS_HOME", str(st))):
        prior = os.environ.get(var)
        if prior and prior != new:
            log.debug("[EMBEDDER] overriding operator-set %s=%s -> %s (cache mount)", var, prior, new)
        os.environ[var] = new


class Embedder:
    """Wraps a sentence-transformers model for embedding text.

    The model is loaded lazily on first use (or in a background thread if
    ``background=True`` is passed to the constructor).  This avoids blocking
    experiment startup for 30-60 s while the model loads — tasks can begin
    immediately and embeddings are computed once the model is ready.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_EMBEDDING_MODEL,
        dimensions: int = 768,
        *,
        background: bool = False,
    ) -> None:
        self._model_name = model_name
        self.dimensions = dimensions
        self._model: "SentenceTransformer | None" = None  # type: ignore[name-defined]
        self._load_error: BaseException | None = None
        self._ready = threading.Event()
        # This Embedder is process-wide shared (engine._embedder) and its
        # embed()/embed_batch() are called concurrently from the eval
        # ThreadPoolExecutor (worker count follows max_concurrent_tasks; a
        # non-positive config value falls back to 50).
        # A torch/sentence-transformers model.encode() already fans out across
        # cores internally, so letting N threads call it simultaneously causes
        # CPU oversubscription (N * cores threads thrashing) with no throughput
        # gain. Serialize encode() so at most one thread runs a forward pass at
        # a time — output values are unchanged.
        self._encode_lock = threading.Lock()

        if background:
            self._thread = threading.Thread(
                target=self._load_model,
                daemon=True,
                name="embedder-load",
            )
            self._thread.start()
        else:
            # Eager (blocking) load — preserves old behaviour when needed.
            self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            log.info("[EMBEDDER] Model %s loaded", self._model_name)
        except Exception as exc:
            self._load_error = exc
            log.warning("[EMBEDDER] Failed to load model: %s", exc)
        finally:
            self._ready.set()

    def _get_model(self) -> "SentenceTransformer":  # type: ignore[name-defined]
        """Wait for model and return it, or raise if loading failed."""
        self._ready.wait()
        if self._load_error is not None:
            raise RuntimeError(f"Embedding model failed to load: {self._load_error}") from self._load_error
        assert self._model is not None
        return self._model

    @property
    def is_ready(self) -> bool:
        """Non-blocking check whether the model has finished loading."""
        return self._ready.is_set() and self._load_error is None

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Block until the model finishes loading; return True iff it loaded.

        Never raises — a load failure (e.g. missing HF token) returns False so
        callers can degrade to FTS rather than abort the run.
        """
        self._ready.wait(timeout)
        return self.is_ready

    def embed(self, text: str) -> list[float]:
        model = self._get_model()
        with self._encode_lock:
            vec = model.encode(text, normalize_embeddings=True)
        if len(vec) < self.dimensions:
            raise ValueError(f"Model output dimension {len(vec)} < configured {self.dimensions}")
        return vec[: self.dimensions].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        with self._encode_lock:
            vecs = model.encode(texts, normalize_embeddings=True)
        for i, v in enumerate(vecs):
            if len(v) < self.dimensions:
                raise ValueError(f"Model output dimension {len(v)} < configured {self.dimensions}")
        return [v[: self.dimensions].tolist() for v in vecs]

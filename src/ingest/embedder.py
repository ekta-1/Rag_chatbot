"""Embedding wrapper around ``sentence-transformers/all-MiniLM-L6-v2``.

The model is loaded **once per process** and shared. ``retriever`` imports this
same class rather than building its own, because two instances with different
pooling or normalisation silently destroy retrieval quality -- every query
returns plausible-looking nonsense and nothing errors.
"""

from __future__ import annotations

import logging
import threading

from src.config import CONFIG

log = logging.getLogger(__name__)

BATCH_SIZE = 32
EXPECTED_DIM = 384

_lock = threading.Lock()
_cache: dict[str, "Embedder"] = {}


class Embedder:
    """Lazy singleton per model name."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or CONFIG.embed_model
        self._model = None
        self._tokenizer = None

    # -- lazy loading ------------------------------------------------------

    def _ensure(self) -> None:
        if self._model is not None:
            return
        with _lock:
            if self._model is not None:
                return
            from sentence_transformers import SentenceTransformer

            log.info("loading embedding model %s", self.model_name)
            model = SentenceTransformer(self.model_name)
            # Guard the assumption the whole chunk-size decision rests on. If a
            # future model has a different limit, chunk sizing must be revisited
            # -- better to hear about it here than via mysteriously bad recall.
            log.info("model max_seq_length=%s", model.max_seq_length)
            self._model = model
            self._tokenizer = model.tokenizer

    @property
    def model(self):
        self._ensure()
        return self._model

    @property
    def tokenizer(self):
        """The wordpiece tokenizer. Chunker sizing decisions depend on this."""
        self._ensure()
        return self._tokenizer

    @property
    def max_seq_length(self) -> int:
        self._ensure()
        return int(self._model.max_seq_length)

    # -- embedding ---------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk texts (prefixes already applied by the caller)."""
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=len(texts) > BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vectors = [v.tolist() for v in vectors]
        self._assert_dim(vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a user question.

        Deliberately *not* prefixed: the ``"{scheme} - {section}"`` prefix
        describes a chunk, not a question, and adding it to a query would skew
        the query away from the user's actual words.
        """
        vector = self.model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )
        vector = vector.tolist()
        self._assert_dim([vector])
        return vector

    @staticmethod
    def _assert_dim(vectors: list[list[float]]) -> None:
        for v in vectors:
            if len(v) != EXPECTED_DIM:
                raise ValueError(
                    f"expected {EXPECTED_DIM}-dim vectors from {CONFIG.embed_model}, "
                    f"got {len(v)}. The collection was built for a different model; "
                    "re-run: python -m src.cli ingest --force"
                )


def get_embedder(model_name: str | None = None) -> Embedder:
    """Return the shared Embedder for ``model_name``.

    Ingest and query must agree on the model, so both paths go through here.
    """
    name = model_name or CONFIG.embed_model
    with _lock:
        if name not in _cache:
            _cache[name] = Embedder(name)
        return _cache[name]

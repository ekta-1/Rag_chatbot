"""Retrieval with a score gate (architecture decision D5).

This module owns the *decision* of whether a question is answerable from the
index. The gate lives inside :meth:`Retriever.search` rather than in the caller
so that no code path can retrieve-and-answer without passing through it -- that
single choke point is the main defence against hallucination, and a gate that
callers can forget is not a defence.

An empty result is a valid, meaningful answer: it means the bot should say it
does not know. It is never an error and never triggers a fallback to an
unfiltered search.
"""

from __future__ import annotations

import logging
import re

from src.config import CONFIG, Config
from src.models import Hit

log = logging.getLogger(__name__)

# Scheme names in questions, e.g. "HDFC Large Cap Fund" -> "large_cap". Matched
# loosely on purpose: users say "HDFC Large Cap", "large cap fund", "HDFC ELSS".
_SCHEME_PATTERNS: tuple[tuple[str, str], ...] = (
    ("balanced_advantage", r"balanced\s*advantage"),
    ("small_cap", r"small\s*cap"),
    ("flexi_cap", r"flexi\s*cap"),
    ("large_cap", r"large\s*cap"),
    ("elss", r"\belss\b|tax\s*saver"),
)


class IndexModelMismatch(RuntimeError):
    """The collection was built with a different embedding model than config."""


class EmptyIndexError(RuntimeError):
    """No index has been built yet."""


def detect_scheme_key(query: str) -> str | None:
    """Return the ``scheme_key`` a question names, if any.

    Used for the cheap recall fix in implementation.md 3.4: a question that says
    "HDFC Large Cap" is easier to match if that scheme is named in the embedded
    text. Returns None when the question names no scheme, which is the common
    case ("what is the expense ratio").
    """
    lowered = query.lower()
    for key, pattern in _SCHEME_PATTERNS:
        if re.search(pattern, lowered):
            return key
    return None


def augment_query(query: str, scheme_key: str | None = None) -> str:
    """Append the scheme key to the query text when one is known.

    Deliberately NOT the scheme *name*: the chunks are already prefixed with
    "{scheme_name} -- {section}" at index time, so repeating the name adds
    nothing. The underscore key is a distinct token that the question itself
    never contains, which is what nudges the matching scheme's chunks up.
    """
    if scheme_key is None:
        scheme_key = detect_scheme_key(query)
    return f"{query} {scheme_key}" if scheme_key else query


class Retriever:
    """Scores a question against the index and returns only gated hits."""

    def __init__(self, cfg: Config = CONFIG):
        self.cfg = cfg
        # Import, never re-instantiate: a second SentenceTransformer is a silent
        # retrieval bug the moment the two disagree on pooling or normalisation.
        from src.ingest.embedder import get_embedder
        from src.ingest.store import EmbeddingModelMismatch
        from src.rag.lexical import LexicalGate

        self._embedder = get_embedder()
        self._lexical = LexicalGate.from_collection(cfg)
        try:
            # The write-path guard. Raises if the collection was built with a
            # different model or on a non-cosine space.
            from src.ingest.store import ensure_collection

            self._collection = ensure_collection(cfg, cfg.embed_model)
        except EmbeddingModelMismatch as exc:
            raise IndexModelMismatch(str(exc)) from exc

        count = self._collection.count()
        if count == 0:
            raise EmptyIndexError(
                f"collection {cfg.collection_name!r} is empty. "
                f"Build it first: python -m src.cli ingest"
            )
        log.info("retriever ready: %d chunks in %s", count, cfg.collection_name)

    # -- introspection ------------------------------------------------------

    @property
    def index_size(self) -> int:
        return self._collection.count()

    # -- search -------------------------------------------------------------

    def _score_hit(self, question: str, hit: Hit) -> tuple[bool, float, str]:
        """Evaluate both signals. Returns (passed, coverage, reason-if-failed).

        Both conditions must hold. Kept as one predicate so the reason string is
        always available for the refusal log -- knowing WHICH gate rejected a
        question is what makes a bad threshold debuggable.
        """
        if hit.score < self.cfg.min_similarity:
            return False, 0.0, f"cosine {hit.score:.3f} < {self.cfg.min_similarity}"
        coverage = self._lexical.coverage(question, _embed_text(hit))
        if coverage < self.cfg.min_lexical_coverage:
            return False, coverage, (
                f"coverage {coverage:.3f} < {self.cfg.min_lexical_coverage}"
            )
        return True, coverage, ""

    def search(
        self,
        query: str,
        top_k: int | None = None,
        scheme_key: str | None = None,
        augment: bool = True,
    ) -> list[Hit]:
        """Return hits that clear BOTH gates, best first.

        Args:
            query: the raw user question. Embedded WITHOUT the document prefix,
                because the prefix describes the chunk, not the question.
            top_k: defaults to ``cfg.top_k``.
            scheme_key: restrict to one fund. Inferred from the question text
                when omitted.
            augment: set False to disable the scheme-key recall fix and measure
                its effect (implementation.md 3.4).

        Returns:
            Gated, sorted hits. Empty list when nothing clears both gates.
        """
        if not query or not query.strip():
            return []

        k = top_k or self.cfg.top_k
        candidates = self._candidates(query, scheme_key, augment, k)

        # THE GATE. Kept here on purpose so no caller can bypass it.
        scored = []
        for hit in candidates:
            passed, coverage, _ = self._score_hit(query, hit)
            if passed:
                # Rank by BOTH signals. Cosine alone put the fees chunk above
                # the benchmark chunk for "benchmark of HDFC Balanced
                # Advantage" (0.622/0.695 vs 0.607/1.000) -- the scheme name in
                # the prefix matches every chunk of that fund, so cosine
                # overrates them. Weighting by coverage fixes the ordering
                # without changing Hit.score, which stays the cosine value the
                # thresholds are defined against.
                scored.append((hit, hit.score * coverage))
        scored.sort(key=lambda pair: pair[1], reverse=True)

        hits = [hit for hit, _ in scored[:k]]
        for rank, hit in enumerate(hits, start=1):
            hit.rank = rank

        if not hits and candidates:
            best = candidates[0]
            _, _, reason = self._score_hit(query, best)
            log.info(
                "refused %r: best candidate %s/%s failed on %s",
                query,
                best.chunk.scheme_key,
                best.chunk.section,
                reason,
            )
        return hits

    def _candidates(
        self, query: str, scheme_key: str | None, augment: bool, k: int
    ) -> list[Hit]:
        from src.ingest.store import query as store_query

        effective_query = augment_query(query, scheme_key) if augment else query
        # The raw question, no "{scheme_name} - {section}" prefix.
        vector = self._embedder.embed_query(effective_query)

        # Over-fetch: gating on a shallow top_k would starve the result, because
        # a high-similarity chunk with low coverage must not crowd out a lower-
        # similarity chunk that actually contains the answer.
        return store_query(
            cfg=self.cfg, vector=vector, top_k=max(k * 3, k), scheme_key=scheme_key
        )

    def explain(self, query: str, top_k: int | None = None, scheme_key: str | None = None):
        """Diagnostic view: the candidates with both signals and the verdict.

        Used by the threshold-tuning harness and ``cli ask --explain``. The
        production path is :meth:`search`, which throws this detail away.
        """
        if not query or not query.strip():
            return []
        k = top_k or self.cfg.top_k
        rows = []
        for hit in self._candidates(query, scheme_key, True, k):
            coverage = self._lexical.coverage(query, _embed_text(hit))
            passed, _, reason = self._score_hit(query, hit)
            rows.append(
                {
                    "chunk": hit,
                    "score": hit.score,
                    "coverage": coverage,
                    "passed": passed,
                    "reason": reason,
                    "matched": sorted(
                        self._lexical.matched_terms(query, _embed_text(hit))
                    ),
                }
            )
        return rows


def _embed_text(hit: Hit) -> str:
    """The string the chunk's vector was actually built from."""
    return hit.chunk.embedding_text()

"""IDF-weighted lexical coverage — the second half of the answerability gate.

Why this exists
---------------
Phase 3 tuning showed that a pure cosine threshold CANNOT separate answerable
from unanswerable questions on this corpus. Measured top-1 similarities:

    "expense ratio of HDFC Large Cap Fund"   0.816   answerable
    "riskometer level HDFC Small Cap"         0.625   answerable
    "What is HDFC Bank's FD interest rate?"   0.561   NOT answerable  <-- collides
    "Which fund is best for my portfolio?"    0.486   NOT answerable  <-- collides
    "minimum SIP amount"                      0.445   answerable

The unanswerable finance questions score *higher* than some answerable ones,
because "interest rate" is semantically close to "expense ratio" and "HDFC"
appears in every chunk. No value of MIN_SIMILARITY separates them.

Lexical coverage separates cleanly instead, because the question's *distinctive*
terms must actually appear in the chunk:

    answerable min 0.637  vs  refuse max 0.494   (gap +0.143)

The two signals are complementary and both are required: cosine finds the
plausible chunk, coverage checks that the chunk is really about what was asked.

IDF, not raw term counts
------------------------
Terms like "fund", "HDFC" and "expense" appear in many chunks and must not
count for much; "riskometer" or "lock-in" appear in one and must count for a
lot. Weighting each query term by ``log((N+1)/(df+1))`` handles that without a
hand-maintained stopword list of financial filler.

The index is read from the collection rather than recomputed, so the vocabulary
is exactly the one the vector store was built from.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

log = logging.getLogger(__name__)

# Deliberately small. Only genuinely structural words that carry no topical
# signal. Domain words are NOT here: "lock-in" and "expense" must keep their
# weight, and IDF already down-weights the common ones.
_STOPWORDS = frozenset(
    """
    a an the is are was were be been being of for in on to and or if then than
    that this these those it its as at by from with without about into over
    under my me i we our you your he she they their do does did done can could
    should would will shall may might must have has had not no nor so such
    what which who whom whose when where why how tell show explain please
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric tokens, stopwords and 1-2 char noise removed."""
    return {
        t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2
    }


class LexicalGate:
    """Scores how much of a question's distinctive vocabulary a chunk covers.

    Build once per process from the collection, then :meth:`coverage` is a pure
    function of (question, chunk text).
    """

    def __init__(self, documents: list[str]):
        self._n = max(len(documents), 1)
        self._df: Counter[str] = Counter()
        for doc in documents:
            self._df.update(tokenize(doc))
        self._total_weight = None

    @classmethod
    def from_collection(cls, cfg) -> "LexicalGate":
        """Build IDF statistics from the stored chunks.

        Uses each chunk's *embedding text* (scheme + section + body) rather than
        the bare stored body, because that is the string the vector was built
        from. Scoring one signal against a different string than the other makes
        the gate incoherent -- measuring that during tuning dropped coverage for
        every scheme-named question to ~0.0.
        """
        from src.ingest.store import open_collection

        collection = open_collection(cfg)
        if collection is None:
            return cls([])
        rows = collection.get(include=["documents", "metadatas"])
        docs = [
            f"{meta.get('scheme_name', '')} - {meta.get('section', '')}\n{doc}"
            for doc, meta in zip(rows.get("documents") or [], rows.get("metadatas") or [])
        ]
        gate = cls(docs)
        log.info("lexical gate: %d documents, %d distinct terms", gate._n, len(gate._df))
        return gate

    def idf(self, term: str) -> float:
        """Smoothed inverse document frequency. Always > 0."""
        return math.log((self._n + 1) / (self._df.get(term, 0) + 1)) + 1.0

    def coverage(self, question: str, chunk_embedding_text: str) -> float:
        """Fraction of the question's IDF weight that the chunk accounts for.

        Returns 1.0 for a question with no distinctive terms (nothing to check),
        and 0.0 for a chunk sharing none of them.
        """
        q_terms = tokenize(question)
        if not q_terms:
            return 1.0
        chunk_terms = tokenize(chunk_embedding_text)
        total = sum(self.idf(t) for t in q_terms)
        if total <= 0:
            return 1.0
        hit = sum(self.idf(t) for t in q_terms if t in chunk_terms)
        return hit / total

    def matched_terms(self, question: str, chunk_embedding_text: str) -> set[str]:
        """Which of the question's terms the chunk actually contains.

        For debugging a refusal -- a 0.1 coverage with a high cosine score is
        almost always "similar topic, wrong facts", and this shows why.
        """
        return tokenize(question) & tokenize(chunk_embedding_text)

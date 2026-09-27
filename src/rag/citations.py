"""Citation validation. This module is what makes G1 an invariant.

G1 is "100% of answers carry a source link". Leaving that to the model means
hoping. Here it is enforced: prose without a link that is genuinely one of the
retrieved URLs cannot reach the user, because it is converted to NO_MATCH before
it leaves this function.

The two failure modes worth naming:

* The model recalls a URL from training rather than copying one. It looks
  plausible to a human reviewer, which is exactly why it survives manual QA.
  We catch it by set membership, not by pattern matching.
* The model runs long. Fixed by a sentence-boundary cut, not a character cut, so
  the answer never ends mid-clause.
"""

from __future__ import annotations

import logging
import re

from src.models import Hit
from src.rag.prompts import ADVICE_REFUSAL, NO_MATCH, PII_REFUSAL

logger = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://[^\s\)\]\"'<>]+")

# Trailing punctuation is usually sentence punctuation, not part of the URL.
_URL_TRAILING = ".,;:!?"

MAX_SENTENCES = 3

# The fixed strings are refusals and legitimately carry no citation.
_REFUSAL_MARKERS = (
    "i'm a facts-only assistant",
    "i couldn't find that on the hdfc source pages",
    "please don't share pan",
    "not investment advice",
)


def is_refusal(text: str) -> bool:
    """True for the deterministic refusal paths, which need no citation."""
    low = text.strip().lower()
    return any(marker in low for marker in _REFUSAL_MARKERS)


def _trim_punctuation(url: str) -> str:
    return url.rstrip(_URL_TRAILING)


def _canonical_url(hits: list[Hit]) -> str | None:
    """The URL to fall back on: the top retrieved hit's canonical source_url."""
    for hit in hits:
        url = hit.chunk.source_url
        if url:
            return url
    return None


# A URL together with an optional preceding label, so that removing the model's
# citation does not leave a dangling "Source:" fragment that the sentence
# splitter would then count as an extra sentence.
_URL_WITH_LABEL = re.compile(
    r"(?:\b(?:source|sources|link|links|reference|references|url)\b\s*[:\-]?\s*)?"
    + URL_RE.pattern,
    re.IGNORECASE,
)


def _strip_all_urls(text: str) -> str:
    return _URL_WITH_LABEL.sub("", text).strip()


def _enforce_sentences(text: str) -> tuple[str, bool]:
    """Keep at most MAX_SENTENCES, cutting on a sentence boundary."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p for p in parts if p]
    if len(parts) <= MAX_SENTENCES:
        return text.strip(), False
    kept = " ".join(parts[:MAX_SENTENCES])
    dropped = len(parts) - MAX_SENTENCES
    logger.warning("citation: truncated %d excess sentence(s)", dropped)
    return kept, True


def validate(
    answer_text: str, hits: list[Hit]
) -> tuple[str, str | None, bool]:
    """Return ``(cleaned_text, citation_url, truncated)``.

    Order is fixed and matters: URLs are collected and repaired BEFORE the
    sentence cap, so truncation cannot orphan a link that was already validated.
    """
    text = (answer_text or "").strip()
    if not text:
        return NO_MATCH, None, False

    # Refusals are passed through untouched, but still length-capped.
    if is_refusal(text):
        capped, truncated = _enforce_sentences(text)
        return capped, None, truncated

    retrieved = {hit.chunk.source_url for hit in hits if hit.chunk.source_url}
    found = [_trim_punctuation(u) for u in URL_RE.findall(text)]

    valid = [u for u in found if u in retrieved]
    invalid = [u for u in found if u not in retrieved]

    if invalid:
        # Log the offending URL but never the surrounding prose.
        logger.warning(
            "citation_mismatch: model emitted %d URL(s) not in the retrieved set "
            "(first: %s); substituting the top hit's canonical source_url",
            len(invalid),
            invalid[0],
        )

    if valid:
        citation = valid[0]
    elif invalid:
        # Fabricated link -> replace with the top hit's canonical URL.
        fallback = _canonical_url(hits)
        if not fallback:
            logger.warning("citation: fabricated URL and no canonical source_url; NO_MATCH")
            return NO_MATCH, None, False
        citation = fallback
    else:
        # The model emitted no URL at all. Architecture 5.8 step 3: an uncited
        # non-refusal is a generation failure, not something to paper over by
        # bolting on a link the model did not choose. Shipping a link we invented
        # here would also defeat the point of measuring G1.
        logger.warning("citation: non-refusal answer carried no URL; using NO_MATCH")
        return NO_MATCH, None, False

    # Strip the model's URL tokens, then cap the PROSE, then re-attach the
    # citation. Order matters: capping after re-attaching would let the sentence
    # splitter count "Source: <url>" as a sentence and truncate the citation
    # away, which would silently violate G1 on every 3-sentence answer.
    body = _strip_all_urls(text)
    body, truncated = _enforce_sentences(body)
    body = re.sub(r"\s+", " ", body).strip()
    final = f"{body} Source: {citation}" if body else f"Source: {citation}"
    return final, citation, truncated

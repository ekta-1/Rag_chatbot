"""Citation validation tests. No API key and no network.

validate() is deterministic pure logic over a string and a hit list, so it is
testable without any LLM at all. That is deliberate: G1 ("100% of answers carry
a link") is a property of this function, and it should not need an API key to
be verified.
"""

from __future__ import annotations

import pytest

from src.models import Chunk, Hit
from src.rag.citations import URL_RE, is_refusal, validate
from src.rag.prompts import NO_MATCH

REAL_URL = "https://www.hdfcmf.com/our-funds/equity-funds/hdfc-flexi-cap-fund"


def make_hit(url: str = REAL_URL, section: str = "Exit load") -> Hit:
    chunk = Chunk(
        id=f"test-{section}",
        text="Exit load is 1% if redeemed within 12 months.",
        source_url=url,
        scheme_key="flexi_cap",
        scheme_name="HDFC Flexi Cap Fund",
        category="equity",
        section=section,
        chunk_index=0,
        ingested_at="2026-01-01T00:00:00+00:00",
    )
    return Hit(chunk=chunk, score=0.72)


# --- A correct answer must pass through unchanged in substance -------------

def test_correct_answer_passes_through():
    hits = [make_hit()]
    body = "The exit load is 1% if the units are redeemed within 12 months."
    text, url, truncated = validate(f"{body} Source: {REAL_URL}", hits)

    assert url == REAL_URL
    assert truncated is False
    assert "1%" in text
    assert "12 months" in text
    assert text.count(REAL_URL) == 1


# --- A fabricated URL must be replaced with the top hit's URL -------------

def test_fabricated_url_is_replaced(caplog):
    """The near-miss link is the failure mode manual review misses."""
    fabricated = "https://www.hdfcmf.com/mutual-funds/funds/flexi-cap"
    hits = [make_hit()]
    text, url, truncated = validate(
        f"The exit load is 1%. Source: {fabricated}", hits
    )

    assert url == REAL_URL
    assert fabricated not in text
    assert REAL_URL in text
    assert any("citation_mismatch" in r.message for r in caplog.records)


def test_partial_url_is_not_accepted_as_a_hit_url():
    """A shortened or truncated form of the real URL is still not the real URL."""
    fabricated = "https://www.hdfcmf.com/our-funds/equity-funds/hdfc-flexi"
    hits = [make_hit()]
    _, url, _ = validate(f"Exit load 1%. Source: {fabricated}", hits)
    assert url == REAL_URL


# --- Uncited non-refusal must become NO_MATCH ------------------------------

def test_no_url_non_refusal_becomes_no_match():
    hits = [make_hit()]
    text, url, truncated = validate("The exit load is 1% within 12 months.", hits)

    assert text == NO_MATCH
    assert url is None
    assert truncated is False


def test_uncited_prose_never_reaches_the_user():
    """G1 as an invariant: whatever the model said, a link is attached or the
    answer is replaced."""
    hits = [make_hit()]
    for body in (
        "Exit load is 1%.",
        "Some prose with no link whatsoever.",
        "Numbers 1% and 12 months, still no link.",
    ):
        text, url, _ = validate(body, hits)
        assert REAL_URL in text or text == NO_MATCH


# --- Sentence cap ---------------------------------------------------------

def test_five_sentences_truncated_to_three():
    hits = [make_hit()]
    five = (
        "One. Two. Three. Four. Five. "
        f"Source: {REAL_URL}"
    )
    text, url, truncated = validate(five, hits)

    assert truncated is True
    assert "One." in text and "Two." in text and "Three." in text
    assert "Four." not in text
    assert "Five." not in text


def test_three_sentences_is_not_truncated():
    hits = [make_hit()]
    text, url, truncated = validate(
        f"Exit load is 1%. It applies within 12 months. Nothing after that. Source: {REAL_URL}",
        hits,
    )
    assert truncated is False


def test_truncation_happens_on_a_sentence_boundary():
    """Never cut mid-clause -- a truncated answer must still be a sentence."""
    hits = [make_hit()]
    four = f"A. B. C. D is a long trailing clause that gets cut. Source: {REAL_URL}"
    text, url, truncated = validate(four, hits)
    assert truncated is True
    # The prose stops on a complete sentence; only the citation follows it.
    assert text == f"A. B. C. Source: {REAL_URL}"
    # The 4th sentence is gone entirely, not clipped mid-clause.
    assert "D is a long" not in text


def test_citation_always_survives_truncation():
    """G1 must hold even on an over-long answer -- the link cannot be the thing
    that gets truncated."""
    hits = [make_hit()]
    long_answer = " ".join(f"Sentence {i}." for i in range(1, 9)) + f" Source: {REAL_URL}"
    text, url, truncated = validate(long_answer, hits)
    assert truncated is True
    assert url == REAL_URL
    assert REAL_URL in text


# --- Refusals carry no citation and are passed through --------------------

def test_refusal_passes_through_unchanged():
    from src.rag.prompts import ADVICE_REFUSAL, PII_REFUSAL

    for refusal in (ADVICE_REFUSAL, PII_REFUSAL, NO_MATCH):
        text, url, _ = validate(refusal, [make_hit()])
        assert text == refusal
        assert url is None
        assert is_refusal(refusal)


# --- Multi-hit: only retrieved URLs are acceptable ------------------------

def test_url_from_a_non_retrieved_hit_is_rejected():
    """Citing a URL that was never retrieved is still a fabrication."""
    other = "https://www.hdfcmf.com/our-funds/equity-funds/hdfc-large-cap-fund"
    hits = [make_hit()]  # only the flexi URL was retrieved
    text, url, _ = validate(f"Exit load is 1%. Source: {other}", hits)
    assert url == REAL_URL
    assert other not in text


def test_valid_url_from_second_hit_is_accepted():
    second = "https://www.hdfcmf.com/our-funds/equity-funds/hdfc-large-cap-fund"
    hits = [make_hit(), make_hit(url=second, section="Overview")]
    text, url, _ = validate(f"Expense ratio is 1.03%. Source: {second}", hits)
    assert url == second


# --- URL extraction ------------------------------------------------------

def test_url_re_does_not_swallow_punctuation():
    """URL_RE is pinned verbatim by implementation.md 4.4, so the trailing-dot
    case is absorbed by validate()'s trim rather than by the regex itself."""
    hits = [make_hit()]
    text, url, _ = validate(f"See the facts. Source: {REAL_URL}.", hits)
    assert url == REAL_URL
    assert not url.endswith(".")


def test_url_re_ignores_trailing_paren():
    hits = [make_hit()]
    _, url, _ = validate(f"(source: {REAL_URL})", hits)
    assert url == REAL_URL


def test_answer_with_no_hits_and_no_url_is_no_match():
    text, url, _ = validate("Nothing to cite here.", [])
    assert text == NO_MATCH
    assert url is None


def test_empty_answer_is_no_match():
    text, url, _ = validate("", [make_hit()])
    assert text == NO_MATCH
    assert url is None

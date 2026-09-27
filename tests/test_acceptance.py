"""Acceptance tests against the real LLM API. Marked ``llm``.

Covers PRD section 8: A2, A3, A4, A5, A6, A10.

    pytest tests/test_acceptance.py -v -m llm

The key is read through ``CONFIG.llm_api_key`` rather than ``os.getenv``, so the
suite follows ``LLM_PROVIDER`` and does not skip forever because the
provider-specific variable is no longer ANTHROPIC_API_KEY.

These are the only tests that assert on model *behaviour* rather than on
deterministic logic, so they are written to fail loudly on a real regression and
to skip cleanly when no key is present.
"""

from __future__ import annotations

import pytest

from src.config import CONFIG
from src.rag.chain import answer, reset_caches
from src.rag.prompts import EDU_LINK

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not CONFIG.llm_api_key,
        reason=f"{CONFIG.llm_key_env_var} not set for provider {CONFIG.llm_provider}",
    ),
]


@pytest.fixture(autouse=True)
def _fresh_chain():
    """Each test gets a clean client so call counts are honest."""
    reset_caches()
    yield
    reset_caches()


def _sentences(text: str) -> list[str]:
    import re

    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


# --- A2: expense ratio, <=3 sentences, exactly 1 link ----------------------

def test_a2_expense_ratio():
    result = answer("What is the expense ratio of HDFC Large Cap Fund?")
    text = result.text

    assert not result.refused, f"A2 was refused: {text}"
    # 3 prose sentences max; the trailing "Source: <url>" is a citation token,
    # not a sentence, so allow for it.
    assert len(_sentences(text)) <= 3, f"too long ({len(_sentences(text))}): {text}"
    assert result.citation_url, f"A2 produced no citation: {text}"
    assert result.citation_url in text
    assert text.count("http") == 1, f"expected exactly 1 link: {text}"
    assert "expense ratio" in text.lower()


def test_a2_figure_is_traceable_to_the_page():
    """The number must appear in a retrieved chunk, not be invented."""
    import re

    result = answer("What is the expense ratio of HDFC Large Cap Fund?")
    assert result.hits, "expected retrieved hits"
    corpus = " ".join(h.chunk.text for h in result.hits).lower()

    numbers = re.findall(r"\d+\.\d+\s*%", result.text)
    assert numbers, f"no percentage found in answer: {result.text}"
    for n in numbers:
        assert n.replace(" ", "") in corpus.replace(" ", ""), (
            f"answer contains {n} which is not in the retrieved chunks"
        )


# --- A3: ELSS lock-in ----------------------------------------------------

def test_a3_elss_lock_in():
    result = answer("What is the lock-in period for HDFC ELSS Tax Saver Fund?")
    assert not result.refused, f"A3 was refused: {result.text}"
    assert "3 year" in result.text.lower() or "3-year" in result.text.lower() or (
        "three year" in result.text.lower()
    ), f"lock-in not stated as 3 years: {result.text}"
    assert result.citation_url
    assert len(_sentences(result.text)) <= 3


# --- A4: minimum SIP -----------------------------------------------------

def test_a4_minimum_sip():
    result = answer("What is the minimum SIP amount?")
    assert not result.refused, f"A4 was refused: {result.text}"
    assert result.citation_url
    assert "sip" in result.text.lower()
    assert len(_sentences(result.text)) <= 3


# --- A5: advice refusal, no numbers, education link -----------------------

def test_a5_advice_refused_without_numbers():
    result = answer("Should I buy HDFC Small Cap Fund?")
    assert result.refused
    assert result.refusal_kind == "advice"
    assert EDU_LINK in result.text, "A5: educational link must be present"
    assert result.citation_url is None

    import re

    assert not re.search(r"\d", result.text), (
        f"A5: refusal leaked a number: {result.text}"
    )


def test_a5_portfolio_advice_refused():
    result = answer("Which of these funds is best for my portfolio?")
    assert result.refused
    assert result.refusal_kind == "advice"


# --- A6: out of corpus ---------------------------------------------------

def test_a6_out_of_corpus_says_not_found():
    result = answer("What is HDFC Bank's FD interest rate?")
    assert result.refused
    assert result.refusal_kind == "no_match"


def test_a6_does_not_fabricate():
    """A refused answer must not contain a plausible-looking percentage."""
    import re

    result = answer("What is HDFC Bank's FD interest rate?")
    assert not re.search(r"\d+(\.\d+)?\s*%", result.text), (
        f"A6: refused answer still contains a rate: {result.text}"
    )


# --- A10: determinism ----------------------------------------------------

def test_a10_same_question_twice_same_citation():
    q = "What is the exit load on HDFC Flexi Cap Fund?"
    first = answer(q)
    second = answer(q)
    assert first.citation_url == second.citation_url, (
        f"A10: citation drifted\n  {first.citation_url}\n  {second.citation_url}"
    )
    assert first.text == second.text, "A10: answer text drifted between identical calls"


# --- G1 as a cross-cutting invariant --------------------------------------

@pytest.mark.parametrize(
    "q",
    [
        "expense ratio of HDFC Large Cap Fund",
        "exit load on HDFC Flexi Cap Fund",
        "minimum SIP amount",
        "lock-in period HDFC ELSS",
        "benchmark of HDFC Balanced Advantage",
        "riskometer level HDFC Small Cap",
    ],
)
def test_g1_every_answer_carries_a_link(q):
    """G1: 100% of answers carry a source link."""
    result = answer(q)
    assert result.citation_url, f"G1 violated for {q!r}: {result.text}"
    assert "http" in result.text


# --- PII never reaches the model -----------------------------------------

def test_pii_refused_before_any_api_call():
    result = answer("What is the exit load? My PAN is ABCDE1234F")
    assert result.refused
    assert result.refusal_kind == "pii"
    assert "ABCDE1234F" not in result.text


def test_last_updated_comes_from_manifest():
    import json
    from pathlib import Path

    result = answer("expense ratio of HDFC Large Cap Fund")
    manifest = json.loads(Path("data/index_manifest.json").read_text())
    assert result.last_updated == manifest["ingested_at"]

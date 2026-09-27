"""Chain tests with a stubbed generator. No API key, no network.

The acceptance tests prove the model behaves; these prove the *plumbing* is
correct, so that when a key is added the online tests exercise a pipeline that
is already known to work. In particular they pin the three paths that are easy
to get wrong and hard to debug live: a fabricated URL, a missing URL, and an
API exception.
"""

from __future__ import annotations

import pytest

from src.models import Chunk, Hit
from src.rag import chain
from src.rag.prompts import NO_MATCH

REAL_URL = "https://www.hdfcmf.com/our-funds/equity-funds/hdfc-large-cap-fund"
FAKE_URL = "https://hdfcmf.com/funds/large-cap-direct-growth"


class StubGenerator:
    """Stands in for Generator. Returns canned text, or raises."""

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.call_count = 0
        self.questions: list[str] = []

    def answer(self, question, hits):
        self.call_count += 1
        self.questions.append(question)
        if self.error:
            raise self.error
        return self.reply


@pytest.fixture
def fake_hits():
    chunk = Chunk(
        id="c1",
        text="Direct Growth expense ratio is 1.03%.",
        source_url=REAL_URL,
        scheme_key="large_cap",
        scheme_name="HDFC Large Cap Fund",
        category="equity",
        section="Fees, exit load and investment limits",
        chunk_index=0,
        ingested_at="2026-01-01T00:00:00+00:00",
    )
    return [Hit(chunk=chunk, score=0.81, rank=1)]


@pytest.fixture
def wired(monkeypatch, fake_hits):
    """Install a stub generator and a stub retriever; yield a setter."""
    state = {"gen": None}

    monkeypatch.setattr(chain, "_get_retriever", lambda cfg: type(
        "R", (), {"search": lambda self, q, **kw: fake_hits}
    )())

    def install(gen):
        state["gen"] = gen
        monkeypatch.setattr(chain, "_get_generator", lambda cfg: gen)
        return gen

    return install


# --- Happy path ------------------------------------------------------------

def test_good_answer_carries_the_citation(wired):
    gen = wired(StubGenerator(reply=f"The expense ratio is 1.03%. Source: {REAL_URL}"))
    result = chain.answer("expense ratio of HDFC Large Cap Fund")

    assert not result.refused
    assert result.citation_url == REAL_URL
    assert "1.03%" in result.text
    assert result.hits
    assert gen.call_count == 1


# --- A fabricated URL is repaired, not shipped ----------------------------

def test_fabricated_url_is_repaired(wired):
    wired(StubGenerator(reply=f"The expense ratio is 1.03%. Source: {FAKE_URL}"))
    result = chain.answer("expense ratio of HDFC Large Cap Fund")

    assert result.citation_url == REAL_URL
    assert FAKE_URL not in result.text


# --- An uncited answer becomes NO_MATCH ----------------------------------

def test_uncited_answer_becomes_no_match(wired):
    wired(StubGenerator(reply="The expense ratio is 1.03% for the direct plan."))
    result = chain.answer("expense ratio of HDFC Large Cap Fund")

    assert result.refused
    assert result.refusal_kind == "no_match"
    assert result.text == NO_MATCH
    assert result.citation_url is None
    # hits survive so the debug panel still works on a post-retrieval refusal
    assert result.hits


# --- An API exception never escapes --------------------------------------

def test_api_error_returns_no_match_not_a_traceback(wired):
    wired(StubGenerator(error=RuntimeError("connection reset")))
    result = chain.answer("expense ratio of HDFC Large Cap Fund")  # must not raise

    assert result.refused
    assert result.refusal_kind == "no_match"
    assert result.hits


def test_api_error_called_exactly_once(wired):
    """No retry loop: one failure is one API call."""
    gen = wired(StubGenerator(error=RuntimeError("boom")))
    chain.answer("expense ratio of HDFC Large Cap Fund")
    assert gen.call_count == 1


# --- Ordering: guards cost nothing and never call the model ---------------

def test_advice_never_calls_the_model(wired):
    gen = wired(StubGenerator(reply="should never be used"))
    result = chain.answer("Should I buy HDFC Small Cap Fund?")

    assert result.refused and result.refusal_kind == "advice"
    assert gen.call_count == 0


def test_pii_never_calls_the_model(wired):
    gen = wired(StubGenerator(reply="should never be used"))
    result = chain.answer("exit load? my PAN is ABCDE1234F")

    assert result.refused and result.refusal_kind == "pii"
    assert gen.call_count == 0


def test_out_of_corpus_never_calls_the_model(monkeypatch):
    monkeypatch.setattr(chain, "_get_retriever", lambda cfg: type(
        "R", (), {"search": lambda self, q, **kw: []}
    )())
    gen = StubGenerator(reply="never")
    monkeypatch.setattr(chain, "_get_generator", lambda cfg: gen)
    result = chain.answer("What is HDFC Bank's FD interest rate?")

    assert result.refused and result.refusal_kind == "no_match"
    assert gen.call_count == 0


# --- last_updated comes from the manifest, never the model ---------------

def test_last_updated_matches_manifest(wired):
    import json
    from pathlib import Path

    wired(StubGenerator(reply=f"Expense ratio 1.03%. Source: {REAL_URL}"))
    result = chain.answer("expense ratio of HDFC Large Cap Fund")
    manifest = json.loads(Path("data/index_manifest.json").read_text())

    assert result.last_updated == manifest["ingested_at"]


# --- Empty query ----------------------------------------------------------

def test_empty_query_is_safe(wired):
    gen = wired(StubGenerator(reply="never"))
    result = chain.answer("   ")
    assert result.refused
    assert gen.call_count == 0

"""Retriever tests.

The threshold-tuning tests need the real index and the real model, so they are
marked ``integration``. The pure-logic tests run anywhere.
"""

from __future__ import annotations

import pytest

from src.config import CONFIG
from src.rag.lexical import LexicalGate, tokenize
from src.rag.retriever import (
    EmptyIndexError,
    IndexModelMismatch,
    Retriever,
    augment_query,
    detect_scheme_key,
)


# --------------------------------------------------------------------------
# Scheme detection (no index needed)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question,expected",
    [
        ("expense ratio of HDFC Large Cap Fund", "large_cap"),
        ("exit load on HDFC Flexi Cap Fund", "flexi_cap"),
        ("lock-in period HDFC ELSS", "elss"),
        ("what about the tax saver fund", "elss"),
        ("riskometer level HDFC Small Cap", "small_cap"),
        ("benchmark of HDFC Balanced Advantage", "balanced_advantage"),
        ("small cap vs large cap", "small_cap"),  # first match wins, deterministically
        ("what is the expense ratio", None),
        ("What is HDFC Bank's FD interest rate?", None),
    ],
)
def test_detect_scheme_key(question, expected):
    assert detect_scheme_key(question) == expected


def test_augment_query_only_when_a_scheme_is_named():
    assert augment_query("expense ratio of HDFC Large Cap Fund") == (
        "expense ratio of HDFC Large Cap Fund large_cap"
    )
    assert augment_query("what is the expense ratio") == "what is the expense ratio"


def test_augment_query_respects_an_explicit_scheme():
    assert augment_query("expense ratio?", "elss") == "expense ratio? elss"


# --------------------------------------------------------------------------
# Lexical gate (no index needed)
# --------------------------------------------------------------------------


def test_tokenize_drops_stopwords_and_short_tokens():
    assert tokenize("What is the expense ratio of a HDFC fund?") == {
        "expense",
        "ratio",
        "hdfc",
        "fund",
    }


def test_coverage_is_one_for_a_full_term_match():
    gate = LexicalGate(["expense ratio | 1.03%", "benchmark | NIFTY 100 TRI"])
    assert gate.coverage("expense ratio", "expense ratio | 1.03%") == pytest.approx(1.0)


def test_coverage_is_zero_for_no_shared_terms():
    gate = LexicalGate(["expense ratio | 1.03%", "benchmark | NIFTY 100 TRI"])
    assert gate.coverage("bitcoin halving schedule", "expense ratio | 1.03%") == 0.0


def test_coverage_ranks_rare_terms_above_common_ones():
    """The whole point of IDF: 'fund' appears in every chunk, 'riskometer' in one.

    A term present in every document carries no discriminative power and must
    not let an unrelated chunk look like a good lexical match.
    """
    common = "hdfc mutual fund scheme expense ratio " * 5
    docs = [common, common, common, common, "riskometer level moderately high"]
    gate = LexicalGate(docs)
    # Measured ratio is ~1.8x with this corpus; the requirement is only that a
    # term in every document counts for meaningfully less than a term in one.
    assert gate.idf("riskometer") > gate.idf("fund") * 1.5

    # A query made only of the ubiquitous term must not fully match a chunk
    # that merely contains it -- otherwise every chunk looks equally relevant.
    partial = gate.coverage("fund", "riskometer level moderately high")
    assert partial < 0.2


def test_coverage_of_an_empty_question_is_permissive():
    """Nothing to check must not silently refuse everything."""
    gate = LexicalGate(["expense ratio"])
    assert gate.coverage("a of the", "expense ratio") == 1.0


def test_matched_terms_reports_why_a_hit_failed():
    gate = LexicalGate(["expense ratio | 1.03%"])
    assert gate.matched_terms("expense ratio", "expense ratio | 1.03%") == {
        "expense",
        "ratio",
    }
    assert gate.matched_terms("lock-in period", "expense ratio | 1.03%") == set()


def test_lexical_gate_on_a_missing_collection_is_harmless(tmp_path):
    import dataclasses

    from src.ingest.store import reset

    cfg = dataclasses.replace(CONFIG, chroma_dir=str(tmp_path / "chroma"))
    reset(cfg)
    assert LexicalGate.from_collection(cfg)._n == 1


# --------------------------------------------------------------------------
# Construction guards (integration)
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_retriever_rejects_an_empty_index(tmp_path):
    import dataclasses

    from src.ingest.store import reset

    cfg = dataclasses.replace(CONFIG, chroma_dir=str(tmp_path / "chroma"))
    reset(cfg)
    with pytest.raises(EmptyIndexError, match="ingest"):
        Retriever(cfg)


@pytest.mark.integration
def test_retriever_rejects_a_model_mismatch(tmp_path):
    import dataclasses

    from src.ingest.store import ensure_collection, upsert_chunks
    from tests.test_store import make_chunks, vec

    cfg = dataclasses.replace(
        CONFIG, chroma_dir=str(tmp_path / "chroma"), embed_model="some/other-model"
    )
    ensure_collection(cfg, "sentence-transformers/all-MiniLM-L6-v2")
    upsert_chunks(
        make_chunks(2), [vec(0), vec(1)], cfg, embed_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    with pytest.raises(IndexModelMismatch, match="different model|ingest"):
        Retriever(cfg)


@pytest.mark.integration
def test_retriever_reuses_the_shared_embedder():
    from src.ingest.embedder import get_embedder

    assert Retriever()._embedder is get_embedder()


# --------------------------------------------------------------------------
# The gate itself (integration -- needs the real index)
# --------------------------------------------------------------------------

# The 7 positives from implementation.md 3.2. Expected section is expressed as
# a substring because the live page headings are longer than the doc's shorthand
# ("Fees / charges" is really "Fees, exit load and investment limits").
POSITIVES = [
    ("expense ratio of HDFC Large Cap Fund", "large_cap", "fees"),
    ("exit load on HDFC Flexi Cap Fund", "flexi_cap", "fees"),
    ("minimum SIP amount", None, "fees"),
    ("lock-in period HDFC ELSS", "elss", "fees"),
    ("benchmark of HDFC Balanced Advantage", "balanced_advantage", "benchmark"),
    ("riskometer level HDFC Small Cap", "small_cap", "benchmark"),
    # A6: unanswerable from these 5 pages, and correctly treated as a REFUSAL.
    # Listed here so the test fails loudly if someone "fixes" it by adding a
    # fuzzy fallback that starts answering it.
    ("how to download capital gains statement", None, None),
]

NEGATIVES = [
    "What is HDFC Bank's FD interest rate?",
    "Which fund is best for my portfolio?",
    "What is the weather in Mumbai?",
    "Tell me about Bitcoin.",
    "What is the price of gold today?",
]


@pytest.mark.integration
@pytest.mark.parametrize("question,scheme_key,section", POSITIVES)
def test_positive_queries_retrieve_the_expected_top_chunk(question, scheme_key, section):
    if section is None:
        pytest.skip("A6: not answerable from the source set; covered by test_refusals")
    hits = Retriever().search(question)
    assert hits, f"correct answer was gated out for {question!r}"
    top = hits[0].chunk
    assert section in top.section.lower(), (
        f"{question!r} returned {top.scheme_key}/{top.section!r}, "
        f"expected a {section!r} section"
    )
    if scheme_key:
        assert top.scheme_key == scheme_key, (
            f"{question!r} returned the wrong fund: {top.scheme_key}"
        )


@pytest.mark.integration
@pytest.mark.parametrize("question", NEGATIVES)
def test_negative_queries_return_nothing(question):
    assert Retriever().search(question) == []


@pytest.mark.integration
def test_a6_statement_query_is_refused():
    """The one positive-shaped query with no source coverage must still refuse."""
    assert Retriever().search("how to download capital gains statement") == []


@pytest.mark.integration
def test_gate_explains_which_condition_rejected_a_question():
    rows = Retriever().explain("What is HDFC Bank's FD interest rate?")
    assert rows
    best = rows[0]
    # High cosine, no lexical overlap: the exact failure mode the lexical gate
    # was added for.
    assert best["score"] >= CONFIG.min_similarity
    assert best["coverage"] < CONFIG.min_lexical_coverage
    assert "coverage" in best["reason"]
    assert best["matched"] == [] or "fd" not in best["matched"]


@pytest.mark.integration
def test_hits_are_sorted_and_ranked_contiguously():
    hits = Retriever().search("expense ratio of HDFC Large Cap Fund")
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    blended = [h.score for h in hits]
    assert blended == sorted(blended, reverse=True) or len(hits) == 1


@pytest.mark.integration
def test_scoping_to_a_scheme_excludes_other_funds():
    hits = Retriever().search("expense ratio", scheme_key="elss")
    assert hits
    assert {h.chunk.scheme_key for h in hits} == {"elss"}


@pytest.mark.integration
def test_scheme_named_question_prefers_that_fund():
    hits = Retriever().search("expense ratio of HDFC Flexi Cap Fund")
    assert hits
    assert hits[0].chunk.scheme_key == "flexi_cap"


@pytest.mark.integration
def test_blank_query_returns_nothing():
    assert Retriever().search("   ") == []


@pytest.mark.integration
def test_every_returned_hit_clears_both_gates():
    """The gate is inside search(); this asserts callers cannot get around it."""
    retriever = Retriever()
    for question in ["expense ratio of HDFC Large Cap Fund", "minimum SIP amount"]:
        for hit in retriever.search(question):
            assert hit.score >= CONFIG.min_similarity
            coverage = retriever._lexical.coverage(
                question, hit.chunk.embedding_text()
            )
            assert coverage >= CONFIG.min_lexical_coverage


@pytest.mark.integration
def test_thresholds_have_a_separation_margin():
    """Guards the tuned values against silent drift.

    If someone lowers min_lexical_coverage, or a re-ingest changes the corpus,
    the gap between the two sets can close without any test failing otherwise.
    """
    retriever = Retriever()

    def best(question):
        rows = retriever.explain(question)
        return max((r["coverage"] for r in rows), default=0.0)

    answerable = [best(q) for q, _, s in POSITIVES if s]
    refusable = [best(q) for q in NEGATIVES] + [
        best("how to download capital gains statement")
    ]
    assert min(answerable) > max(refusable), (
        f"sets no longer separate: answerable min {min(answerable):.3f} "
        f"vs refuse max {max(refusable):.3f}"
    )
    assert CONFIG.min_lexical_coverage > max(refusable)
    assert CONFIG.min_lexical_coverage < min(answerable)

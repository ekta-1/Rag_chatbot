"""Extractor tests.

These run entirely against ``tests/data/fixtures/scheme_page.html`` -- no test
in this file may touch the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest.extractor import clean_text, extract, missing_fields
from src.ingest.structured import EXCLUDED_FIELDS, extract_facts, load_server_data
from src.models import Source

FIXTURE = Path(__file__).parent / "data" / "fixtures" / "scheme_page.html"

SOURCE = Source(
    scheme_key="large_cap",
    category="Large Cap",
    scheme_name="HDFC Large Cap Fund",
    url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
)


@pytest.fixture(scope="module")
def html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def document(html: str):
    return extract(html, SOURCE, "2026-09-27T00:00:00+00:00")


# --------------------------------------------------------------------------
# clean_text
# --------------------------------------------------------------------------


def test_clean_text_redacts_long_digit_runs():
    assert "[redacted]" in clean_text("account 123456789012 ok")
    assert "123456789012" not in clean_text("account 123456789012 ok")


def test_clean_text_keeps_financial_figures():
    # A fee percentage must survive -- only 12+ digit runs are redacted.
    assert "1.03%" in clean_text("Expense ratio 1.03%")


def test_clean_text_collapses_whitespace():
    assert clean_text("a   b\t\tc") == "a b c"


# --------------------------------------------------------------------------
# Extraction: what must be gone
# --------------------------------------------------------------------------


def test_nav_and_footer_are_stripped(document):
    text = " ".join(b.text for b in document.blocks)
    assert "Home" not in text
    assert "Groww is a registered investment advisor" not in text


def test_cookie_and_newsletter_noise_is_stripped(document):
    text = " ".join(b.text for b in document.blocks)
    assert "Accept all cookies" not in text
    assert "Subscribe to our newsletter" not in text


def test_script_content_is_stripped(document):
    text = " ".join(b.text for b in document.blocks)
    assert "__SOME_TRACKER__" not in text
    assert "return_stats" not in text


# --------------------------------------------------------------------------
# Extraction: what must survive
# --------------------------------------------------------------------------


def test_section_is_never_empty(document):
    assert all(b.section.strip() for b in document.blocks)


def test_prose_is_captured_under_its_heading(document):
    # Note: iterate blocks, not a {section: text} dict -- several sections hold
    # more than one block and a dict would silently keep only the last.
    benchmark_blocks = [b for b in document.blocks if b.section == "Benchmark"]
    assert benchmark_blocks
    assert any("NIFTY 100 TRI" in b.text for b in benchmark_blocks)


def test_short_factual_values_come_from_the_structured_payload(html: str):
    # Prose blocks under 40 chars are dropped on purpose (they are mostly
    # carousels and promos). Short *facts* therefore have to come from the
    # structured payload instead -- that is the whole point of it.
    combined = "\n".join(b.text for b in extract_facts(html, SOURCE))
    for label in (
        "Expense ratio",
        "Riskometer",
        "Benchmark",
        "Exit load",
        "Minimum SIP amount",
    ):
        assert label in combined, f"{label} must come from the payload"


def test_carousel_noise_is_filtered(document):
    # A block that merely lists other schemes is not a fact about this scheme
    # and must not become a retrievable chunk.
    texts = [b.text for b in document.blocks]
    carousel = [t for t in texts if t.strip() in {"HDFC Value Fund Direct Plan Growth"}]
    assert not carousel, "short carousel entries should have been dropped"


def test_table_rows_keep_label_and_value_on_one_line(document):
    fee_rows = [
        line
        for b in document.blocks
        if b.kind == "table"
        for line in b.text.splitlines()
        if line.startswith("Expense ratio")
    ]
    assert fee_rows, "expected an 'Expense ratio' row"
    # Label and value must stay adjacent on the same line.
    assert "1.03%" in fee_rows[0]


def test_table_row_count_matches_source_rows(document):
    fee_table = next(
        b for b in document.blocks if any("Expense ratio" in l for l in b.text.splitlines())
    )
    # header row + 3 body rows
    assert len(fee_table.text.splitlines()) == 4


def test_list_items_are_extracted(document):
    lists = [b for b in document.blocks if b.kind == "list"]
    assert lists, "expected at least one list block"
    assert any("Reports section" in line for b in lists for line in b.text.splitlines())


def test_statement_download_guidance_survives(document):
    text = " ".join(b.text for b in document.blocks)
    assert "capital gains statement" in text.lower()


# --------------------------------------------------------------------------
# Extraction: robustness
# --------------------------------------------------------------------------


def test_short_blocks_are_dropped(document):
    # Nav crumbs are the target: they must not survive as blocks.
    texts = " ".join(b.text for b in document.blocks)
    assert "Mutual Funds" not in texts


def test_extraction_of_garbage_raises_rather_than_returning_nothing():
    with pytest.raises(ValueError, match="no blocks"):
        extract("<html><body><p>hi</p></body></html>", SOURCE, "t")


def test_extraction_of_malformed_html_does_not_raise():
    doc = extract(
        "<html><body><main><h2>Fees and charges</h2>"
        "<p>Unclosed paragraph with enough text to survive the length filter."
        "</main></body></html>",
        SOURCE,
        "t",
    )
    assert doc.blocks


def test_holdings_are_excluded(document):
    # Thousands of stock names: pure retrieval noise for a fee/rule FAQ.
    text = " ".join(b.text for b in document.blocks)
    assert "Reliance Industries" not in text
    assert not any(b.section.startswith("Holdings") for b in document.blocks)


def test_return_calculator_is_excluded(document):
    # The PRD forbids performance claims; this section is 1Y/3Y return figures.
    text = " ".join(b.text for b in document.blocks)
    assert "+12.40%" not in text
    assert not any("Return calculator" in b.section for b in document.blocks)


def test_fund_comparison_table_is_excluded(document):
    text = " ".join(b.text for b in document.blocks)
    assert "Invesco India Large Cap Fund" not in text


def test_returns_summary_block_is_excluded(document):
    text = " ".join(b.text for b in document.blocks)
    assert "Category average" not in text
    assert "Fund returns +8.7%" not in text


def test_duplicate_address_boilerplate_is_excluded(document):
    # Identical on all five pages: pure top-k budget waste.
    assert not [b for b in document.blocks if b.text.lower().startswith("address")]


def test_useful_prose_survives_the_blocklist(document):
    text = " ".join(b.text for b in document.blocks)
    # The blocklist must not eat genuine scheme content.
    assert "exit load of 1% is charged" in text.lower()
    assert "capital gains statement" in text


def test_prose_has_no_source_wrapping_newlines(document):
    # HTML source line breaks must not leak into prose; tables/lists keep theirs.
    for block in document.blocks:
        if block.kind == "prose":
            assert "\n" not in block.text, f"prose block wrapped: {block.text[:60]!r}"


def test_carousel_of_other_schemes_is_excluded(document):
    # Bare fund-name strings for OTHER schemes: wrong-scheme citation risk.
    texts = [b.text for b in document.blocks]
    assert not [t for t in texts if t.strip() == "HDFC Value Fund Direct Plan Growth"]


def test_drop_counters_are_reported(html: str):
    stats: dict = {}
    extract(html, SOURCE, "t", stats=stats)
    assert stats["dropped_section"] > 0, "holdings/returns sections should be counted"
    assert stats["dropped_content"] > 0, "marketing/returns blocks should be counted"


# --------------------------------------------------------------------------
# Structured facts
# --------------------------------------------------------------------------


def test_server_data_is_parsed(html: str):
    data = load_server_data(html)
    assert data["expense_ratio"] == 1.03
    assert data["benchmark"] == "NIFTY 100 TRI"


def test_facts_are_extracted_with_exact_values(html: str):
    blocks = extract_facts(html, SOURCE)
    assert len(blocks) == 2
    combined = "\n".join(b.text for b in blocks)
    assert "Expense ratio | 1.03%" in combined
    assert "Benchmark | NIFTY 100 TRI" in combined
    assert "Minimum SIP amount | INR 100" in combined


def test_null_lock_in_is_omitted_not_rendered_as_zero(html: str):
    combined = "\n".join(b.text for b in extract_facts(html, SOURCE))
    # This scheme has no lock-in, so the row must be absent rather than "0 years".
    assert "Lock-in period" not in combined


def test_returns_and_ratings_are_never_ingested(html: str):
    combined = "\n".join(b.text for b in extract_facts(html, SOURCE))
    # NOTE: "AUM"/"39933" used to be in this ban list, on the reasoning that fund
    # size is a performance claim. That was wrong twice over. It is a static
    # scheme attribute, nothing computes or compares it, so the PRD does not
    # forbid it -- and because "aum" was in EXCLUDED_FIELDS, the only fund-size
    # string reaching the corpus was Groww's auto-generated summary, which prints
    # the AMC's house-level figure (9,86,237 Cr) on every scheme page. The
    # authoritative value is now ingested from the payload; see
    # tests/test_corpus_integrity.py. Returns and ratings stay banned.
    for banned in ("return_stats", "12.4", "184.5678", "Groww rating"):
        assert banned not in combined
    assert "return_stats" in EXCLUDED_FIELDS
    assert "nav" in EXCLUDED_FIELDS, "NAV is still a performance figure and stays out"


def test_fund_size_is_ingested_from_the_payload(html: str):
    # Counterpart to the ban above: fund size must be present, and must be the
    # payload's value rather than anything scraped from prose.
    combined = "\n".join(b.text for b in extract_facts(html, SOURCE))
    assert "Fund size (AUM) | 39,933.37 Cr" in combined
    assert "9,86,237" not in combined


def test_missing_payload_returns_no_facts():
    assert extract_facts("<html><body><p>nothing here</p></body></html>", SOURCE) == []


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


def test_coverage_finds_the_core_facts(html: str):
    doc = extract(html, SOURCE, "t")
    doc.blocks = extract_facts(html, SOURCE) + doc.blocks
    missing = missing_fields(doc)
    # The fixture has no ELSS lock-in and no client-rendered riskometer prose,
    # but every other answerable field must be present.
    for field in ("expense ratio", "exit load", "minimum sip", "benchmark", "statement"):
        assert field not in missing, f"{field} should be covered"

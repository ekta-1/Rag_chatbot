"""Chunker tests.

Unit tests use a lightweight stand-in tokenizer so the suite stays fast and has
no 90MB model download. One integration-marked test runs the real
all-MiniLM tokenizer, because wordpiece counts are what the size decisions
actually rest on and a stub cannot prove the numbers hold for real.
"""

from __future__ import annotations

import re

import pytest

from src.config import CONFIG
from src.ingest import chunker
from src.ingest.chunker import chunk_document
from src.models import Document, Source, TextBlock, make_chunk_id


# --------------------------------------------------------------------------
# Test double
# --------------------------------------------------------------------------


class StubTokenizer:
    """Whitespace tokenizer with a HuggingFace-compatible surface.

    Good enough to exercise the packing/overlap/split logic. It is NOT a
    wordpiece tokenizer -- that gap is covered by the integration test below.
    """

    def encode(self, text, add_special_tokens=True):
        return (text or "").split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


class SubwordTokenizer:
    """Simulates wordpiece by cutting every 4 characters.

    Needed to exercise the hard-cut path: with a pure whitespace tokenizer a
    single 5000-character "word" is one token and never triggers a split, which
    would leave ``_hard_cut`` untested.
    """

    def encode(self, text, add_special_tokens=True):
        raw = (text or "").replace(" ", "")
        return [raw[i : i + 4] for i in range(0, len(raw), 4)] or [""]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(ids)


@pytest.fixture
def tokenizer():
    return StubTokenizer()


def words(n: int, prefix: str = "word") -> str:
    """``n`` whitespace-separated words -- enough to clear min_chunk_tokens."""
    return " ".join(f"{prefix}{i}" for i in range(n))


def make_doc(blocks: list[TextBlock], scheme_key: str = "large_cap") -> Document:
    return Document(
        source=Source(
            scheme_key=scheme_key,
            category="Large Cap",
            scheme_name="HDFC Large Cap Fund",
            url=f"https://example.invalid/{scheme_key}",
        ),
        blocks=blocks,
        ingested_at="2026-09-27T00:00:00+00:00",
    )


# --------------------------------------------------------------------------
# Invariant 1: never exceed the hard cap
# --------------------------------------------------------------------------


def test_no_chunk_exceeds_hard_cap(tokenizer):
    # One section, far more text than a single chunk can hold.
    prose = " ".join(f"word{i}" for i in range(2000))
    doc = make_doc([TextBlock("Fees", prose, "prose")])
    chunks = chunk_document(doc, CONFIG, tokenizer)
    assert chunks
    for c in chunks:
        n = len(tokenizer.encode(c.text))
        assert n <= CONFIG.chunk_hard_cap_tokens, f"chunk of {n} tokens over cap"


def test_hard_cap_engages_for_unbreakable_runs():
    """A run with no separators must still be cut to the cap.

    Uses SubwordTokenizer because a whitespace tokenizer counts one 5000-char
    "word" as a single token and would never reach the hard-cut path.
    """
    sub = SubwordTokenizer()
    doc = make_doc([TextBlock("Fees", "x" * 5000, "prose")])
    chunks = chunk_document(doc, CONFIG, sub)
    assert chunks
    for c in chunks:
        assert len(sub.encode(c.text)) <= CONFIG.chunk_hard_cap_tokens


def test_tokenizer_is_required():
    doc = make_doc([TextBlock("Fees", words(40), "prose")])
    with pytest.raises(ValueError, match="tokenizer is required"):
        chunk_document(doc, CONFIG, None)


# --------------------------------------------------------------------------
# Invariant 2: a chunk belongs to exactly one section
# --------------------------------------------------------------------------


def test_chunk_never_spans_two_sections(tokenizer):
    blocks = [
        TextBlock("Fees and charges", words(40, "expenseratio"), "prose"),
        TextBlock("Taxation", words(40, "shorttermtax"), "prose"),
        TextBlock("Exit load", words(40, "exitload"), "prose"),
    ]
    chunks = chunk_document(make_doc(blocks), CONFIG, tokenizer)
    assert len(chunks) == 3
    for c in chunks:
        others = [b.text for b in blocks if b.section != c.section]
        assert not any(other in c.text for other in others)


def test_sections_are_never_merged_even_when_tiny(tokenizer):
    # Blocks below min_chunk_tokens are dropped, not merged across sections.
    # A merged chunk could not be cited to one section.
    blocks = [
        TextBlock("A", words(40, "alpha"), "prose"),
        TextBlock("B", words(40, "beta"), "prose"),
    ]
    chunks = chunk_document(make_doc(blocks), CONFIG, tokenizer)
    assert len(chunks) == 2
    assert len({c.section for c in chunks}) == 2


# --------------------------------------------------------------------------
# Invariant 3: never split a table row or list item
# --------------------------------------------------------------------------


def test_table_label_and_value_stay_together(tokenizer):
    rows = "\n".join(
        [
            "Expense ratio | 1.03%",
            "Exit load | Exit load of 1% if redeemed within 1 year",
            "Minimum SIP amount | INR 100",
            "Minimum lump sum investment | INR 100",
            "Stamp duty | 0.005% from July 2020",
        ]
    )
    doc = make_doc([TextBlock("Fees", rows, "table")])
    chunks = chunk_document(doc, CONFIG, tokenizer)
    assert len(chunks) == 1
    assert "Expense ratio | 1.03%" in chunks[0].text


def test_oversized_table_splits_between_rows_not_inside_one(tokenizer):
    rows = "\n".join(f"Expense ratio row {i} | value {i} percent" for i in range(200))
    doc = make_doc([TextBlock("Fees", rows, "table")])
    chunks = chunk_document(doc, CONFIG, tokenizer)
    assert len(chunks) > 1
    for c in chunks:
        for line in c.text.split("\n"):
            if "|" in line:
                # Every emitted table line must still be a complete row with
                # both a label and a value.
                parts = line.split(" | ")
                assert len(parts) == 2
                assert parts[0].strip() and parts[1].strip()


def test_list_items_are_never_bisected(tokenizer):
    items = "\n".join(f"- Instruction number {i} for the user" for i in range(200))
    doc = make_doc([TextBlock("How to", items, "list")])
    chunks = chunk_document(doc, CONFIG, tokenizer)
    assert len(chunks) > 1

    # Every original item must survive somewhere, intact.
    seen = set()
    for c in chunks:
        for line in c.text.split("\n"):
            m = re.search(r"Instruction number (\d+)", line)
            if m:
                seen.add(int(m.group(1)))
    assert seen == set(range(200)), "a list item was lost or torn"

    # The first line of a non-first chunk may be an overlap fragment, which
    # legitimately starts mid-item. But no other line may.
    for c in chunks[1:]:
        for line in c.text.split("\n")[1:]:
            if line.strip():
                assert line.strip().startswith("- "), f"torn list item: {line!r}"


# --------------------------------------------------------------------------
# Overlap
# --------------------------------------------------------------------------


def test_overlap_repeats_content_within_a_section(tokenizer):
    prose = " ".join(f"token{i}" for i in range(600))
    doc = make_doc([TextBlock("Fees", prose, "prose")])
    chunks = chunk_document(doc, CONFIG, tokenizer)
    assert len(chunks) > 1
    # Consecutive chunks in the same section should share some text.
    assert set(chunks[0].text.split()) & set(chunks[1].text.split())


def test_no_overlap_across_a_section_boundary(tokenizer):
    # A chunk must not open with the tail of a *different* section.
    doc = make_doc(
        [
            TextBlock("Fees", words(600, "alpha"), "prose"),
            TextBlock("Taxation", words(60, "beta"), "prose"),
        ]
    )
    chunks = chunk_document(doc, CONFIG, tokenizer)
    tax = next(c for c in chunks if c.section == "Taxation")
    assert "alpha" not in tax.text


def test_first_chunk_of_a_section_has_no_overlap_prefix(tokenizer):
    doc = make_doc([TextBlock("Fees", words(600), "prose")])
    chunks = chunk_document(doc, CONFIG, tokenizer)
    assert chunks[0].text.startswith("word0")


def test_overlap_never_exceeds_its_budget(tokenizer):
    """Regression: a prose line longer than the overlap budget used to be
    taken whole, which produced ~200 tokens of overlap and blew the hard cap."""
    doc = make_doc([TextBlock("Fees", words(2000), "prose")])
    chunks = chunk_document(doc, CONFIG, tokenizer)
    assert len(chunks) > 3
    for c in chunks:
        assert len(tokenizer.encode(c.text)) <= CONFIG.chunk_hard_cap_tokens

    # Directly assert the invariant, not just its downstream symptom.
    for previous in (words(500), words(500).replace(" ", "\n")):
        tail = chunker._overlap_tail(previous, CONFIG.chunk_overlap_tokens, tokenizer)
        assert len(tokenizer.encode(tail)) <= CONFIG.chunk_overlap_tokens


def test_overlap_preserves_table_rows(tokenizer):
    """Regression: a wordpiece-level overlap tail flattened a table into a
    'label | value label | value' mash on the next chunk."""
    rows = "\n".join(f"Expense ratio row {i} | value {i} percent" for i in range(200))
    doc = make_doc([TextBlock("Fees", rows, "table")])
    chunks = chunk_document(doc, CONFIG, tokenizer)
    assert len(chunks) > 1
    for c in chunks:
        for line in c.text.split("\n"):
            if "|" in line:
                parts = line.split(" | ")
                assert len(parts) == 2, f"row mashed by overlap: {line!r}"


# --------------------------------------------------------------------------
# Metadata and identity
# --------------------------------------------------------------------------


def test_every_chunk_has_complete_metadata(tokenizer):
    doc = make_doc(
        [
            TextBlock("Fees", "Expense ratio | 1.03%", "table"),
            TextBlock("Taxation", "Short term gains are taxed at twenty percent.", "prose"),
        ]
    )
    for c in chunk_document(doc, CONFIG, tokenizer):
        assert c.source_url == doc.source.url
        assert c.scheme_key == "large_cap"
        assert c.scheme_name == "HDFC Large Cap Fund"
        assert c.category == "Large Cap"
        assert c.section.strip()
        assert c.ingested_at == doc.ingested_at
        assert c.id
        assert isinstance(c.char_start, int)


def test_chunk_ids_are_deterministic(tokenizer):
    doc = make_doc([TextBlock("Fees", words(60, "expenseratio"), "prose")])
    first = [c.id for c in chunk_document(doc, CONFIG, tokenizer)]
    second = [c.id for c in chunk_document(doc, CONFIG, tokenizer)]
    assert first and first == second
    assert first[0] == make_chunk_id(doc.source.url, "Fees", 0)


def test_char_start_points_at_the_right_block(tokenizer):
    blocks = [
        TextBlock("First", "alpha content here", "prose"),
        TextBlock("Second", "beta content here", "prose"),
    ]
    doc = make_doc(blocks)
    chunks = chunk_document(doc, CONFIG, tokenizer)
    joined = "\n\n".join(b.text for b in blocks)
    for c in chunks:
        # char_start must land on the start of that section's own text
        assert joined[c.char_start :].startswith(c.text.split("\n")[0][:20])


def test_empty_document_returns_no_chunks(tokenizer):
    assert chunk_document(make_doc([]), CONFIG, tokenizer) == []


# --------------------------------------------------------------------------
# Recursive splitting internals
# --------------------------------------------------------------------------


def test_recursive_split_terminates_on_a_pathological_input(tokenizer):
    # No separators present at all: must fall through to _hard_cut, not spin.
    text = "y" * 4000
    pieces = chunker._recursive_split(text, 200, tokenizer)
    assert pieces
    assert all(len(tokenizer.encode(p)) <= 200 for p in pieces)


def test_recursive_split_respects_separator_order(tokenizer):
    text = "First sentence here. Second sentence here. Third sentence here."
    pieces = chunker._recursive_split(text, 6, tokenizer)
    assert len(pieces) > 1
    # Nothing should be silently lost.
    rejoined = " ".join(pieces)
    for word in ("First", "Second", "Third"):
        assert word in rejoined


def test_recursive_split_preserves_all_content(tokenizer):
    text = " ".join(f"word{i}" for i in range(500))
    pieces = chunker._recursive_split(text, 100, tokenizer)
    recovered = set(" ".join(pieces).split())
    original = set(text.split())
    assert original - recovered == set(), "recursive split dropped content"


def test_hard_cut_decodes_to_valid_text(tokenizer):
    # Cutting on ids rather than characters avoids mojibake at chunk edges.
    pieces = chunker._hard_cut("alpha beta gamma delta " * 100, 50, tokenizer)
    assert pieces
    assert all(p.strip() for p in pieces)
    assert "alpha" in pieces[0]


# --------------------------------------------------------------------------
# Short-chunk dropping
# --------------------------------------------------------------------------


def test_tiny_chunks_are_dropped(tokenizer):
    doc = make_doc(
        [
            TextBlock("Fees", "a b c d e", "prose"),  # 5 tokens, below the floor
            TextBlock("Taxation", "Short term gains are taxed at twenty percent here.", "prose"),
        ]
    )
    chunks = chunk_document(doc, CONFIG, tokenizer)
    assert all(len(tokenizer.encode(c.text)) >= CONFIG.min_chunk_tokens for c in chunks)
    assert not any(c.section == "Fees" for c in chunks)


# --------------------------------------------------------------------------
# Real tokenizer
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_real_tokenizer_respects_the_model_limit():
    """The whole size budget exists because all-MiniLM-L6-v2 truncates at 256.

    This is the test that would catch a silent regression if someone raised
    CHUNK_SIZE_TOKENS back toward the PRD's original 300-600.
    """
    from src.ingest.embedder import get_embedder

    embedder = get_embedder()
    tokenizer = embedder.tokenizer
    assert embedder.max_seq_length == 256

    prose = " ".join(f"word{i}" for i in range(1500))
    doc = make_doc([TextBlock("Fees", prose, "prose")])
    chunks = chunk_document(doc, CONFIG, tokenizer)

    assert chunks
    for c in chunks:
        n = len(tokenizer.encode(c.text, add_special_tokens=True))
        assert n <= CONFIG.chunk_hard_cap_tokens
        # Must fit the model, with room to spare.
        assert n <= embedder.max_seq_length

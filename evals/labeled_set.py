"""The labelled evaluation set for retrieval, as the single source of truth.

Both tests/test_retriever.py and scripts/eval_retrieval.py import from here, so
a query can never be fixed in one place and left stale in the other. Phase 6
should build on this too.

Each case carries a SUBSTRING that must appear in the retrieved chunk, not just
an expected section. Section matching alone is a weak test: a chunk can be in
the right section and still not contain the answer, which is the failure that
actually produces a wrong reply. So the evidence string is the fact itself.
"""

from __future__ import annotations

# (question, expected_scheme_key or None, expected_section_substring,
#  evidence_substring that must appear in the top chunk, provenance)
POSITIVES = [
    (
        "expense ratio of HDFC Large Cap Fund",
        "large_cap",
        "fees",
        "Expense ratio",
        "PRD 1 / A2",
    ),
    (
        "exit load on HDFC Flexi Cap Fund",
        "flexi_cap",
        "fees",
        "Exit load",
        "PRD 2 / A3",
    ),
    (
        "minimum SIP amount",
        None,  # any scheme is correct; the amount is the same corpus-wide
        "fees",
        "Minimum SIP amount",
        "PRD 3 / A4",
    ),
    (
        "lock-in period HDFC ELSS",
        "elss",
        "fees",
        "lock-in",
        "PRD 4 / A3",
    ),
    (
        "benchmark of HDFC Balanced Advantage",
        "balanced_advantage",
        "benchmark",
        "Benchmark",
        "PRD 5",
    ),
    (
        "riskometer level HDFC Small Cap",
        "small_cap",
        "benchmark",
        "Riskometer",
        "PRD 5",
    ),
    (
        "stamp duty",
        None,
        "fees",
        "Stamp duty",
        "Problemstatement.md fees/charges coverage",
    ),
    (
        "minimum lump sum investment",
        None,
        "fees",
        "lump sum",
        "Problemstatement.md fees/charges coverage",
    ),
    (
        "who manages HDFC Large Cap Fund",
        "large_cap",
        "fund management",
        "Fund management",
        "Problemstatement.md fund facts",
    ),
]

# A6: real, well-formed, genuinely NOT answerable from these 5 pages. Must be
# refused. Listed as a positive-shaped case so that a future fuzzy fallback which
# "starts answering it" fails loudly instead of quietly inventing an answer.
UNANSWERABLE = [
    ("how to download capital gains statement", "A6 / problemstatement.md item 6"),
    ("how do I download my statement", "problemstatement.md item 6"),
]

# Out-of-corpus and advice-shaped. All must return zero hits.
NEGATIVES = [
    ("What is HDFC Bank's FD interest rate?", "out of corpus, adjacent vocabulary"),
    ("What is the weather in Mumbai?", "out of corpus"),
    ("Tell me about Bitcoin.", "out of corpus"),
    ("What is the price of gold today?", "out of corpus"),
    ("What is the SIP date?", "plausible-sounding, absent from these pages"),
    ("What is the fund manager's email address?", "PII-adjacent, absent"),
]

# The 3 example chips the UI ships with, so the demo surface is covered too.
UI_EXAMPLES = [
    "What is the expense ratio of HDFC Large Cap Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
    "What is HDFC Bank's FD interest rate?",
]

# Query types named in docs/Problemstatement.txt, for a coverage report.
REQUIRED_QUERY_TYPES = {
    "expense ratio": "expense ratio of HDFC Large Cap Fund",
    "exit load": "exit load on HDFC Flexi Cap Fund",
    "minimum SIP": "minimum SIP amount",
    "lock-in (ELSS)": "lock-in period HDFC ELSS",
    "riskometer/benchmark": "riskometer level HDFC Small Cap",
    "how to download statements": "how to download capital gains statement",
}

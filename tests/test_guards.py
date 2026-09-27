"""Guard tests. No API key, no index, no model.

These are the tests that matter most in this phase, and the interesting ones
are the negative cases. A guard that fires on "expense ratio" or "exit load"
breaks the product; a guard that misses a PAN is a privacy incident. Both
directions are asserted explicitly so neither can be traded away later.
"""

from __future__ import annotations

import pytest

from src.rag import guards

# --- PII that must be caught -----------------------------------------------

PII_SAMPLES = [
    ("my PAN is ABCDE1234F", "pan"),
    ("PAN ABCDE1234F please help", "pan"),
    ("aadhaar number 2345 6789 0123", "aadhaar"),
    ("my aadhaar is 234567890123", "aadhaar"),
    ("mail me at ravi.kumar@example.com", "email"),
    ("email: investor@example.co.in", "email"),
    ("call me on 9876543210", "phone"),
    ("whatsapp 8765432109", "phone"),
    ("account number 123456789012", "account"),
    ("my otp is 4821", "otp"),
    ("enter the 6 digit otp 918273", "otp"),
    ("what is the one-time password 554012", "otp"),
    ("verification code 220944", "otp"),
]


@pytest.mark.parametrize("text,kind", PII_SAMPLES)
def test_detect_pii_catches_real_pii(text, kind):
    assert guards.detect_pii(text) == kind


# --- The six PRD section 4 factual questions must NOT look like PII ---------

PRD_FACTUAL_QUESTIONS = [
    "What is the expense ratio of the HDFC Large Cap Fund (Direct – Growth)?",
    "Is there an exit load on HDFC Flexi Cap Fund?",
    "What is the minimum SIP amount?",
    "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
    "What is the benchmark and riskometer level for HDFC Balanced Advantage Fund?",
    "How do I download my capital gains statement?",
]


@pytest.mark.parametrize("q", PRD_FACTUAL_QUESTIONS)
def test_factual_questions_are_not_pii(q):
    assert guards.detect_pii(q) is None


# --- Numbers that appear in the corpus must not be misread as secrets ------

CORPUS_NUMBER_QUESTIONS = [
    "exit load",                       # the spec calls this out explicitly
    "expense ratio of HDFC Large Cap Fund",
    "exit load 1%",
    "what is the expense ratio of 1.03%",
    "stamp duty 0.005% from july 2020",
    "minimum sip 100",
    "expense ratio for the year 2023",
    "launch date 01-Jan-2013",
    "the AUM is 12345 crore",
]


@pytest.mark.parametrize("q", CORPUS_NUMBER_QUESTIONS)
def test_ordinary_numbers_are_not_pii(q):
    """A bare 4-6 digit number is not an OTP without context words."""
    assert guards.detect_pii(q) is None


def test_otp_requires_context():
    assert guards.detect_pii("my otp is 4821") == "otp"
    # Same digits, no context word -> not an OTP.
    assert guards.detect_pii("the value is 4821") is None


# --- Aadhaar must win over the 12-digit account pattern -------------------

def test_aadhaar_reported_before_account():
    assert guards.detect_pii("aadhaar 2345 6789 0123") == "aadhaar"


# --- Advice: the three PRD out-of-scope questions --------------------------

PRD_ADVICE_QUESTIONS = [
    "Should I buy HDFC Small Cap Fund?",
    "Which of these funds is best for my portfolio?",
    "Is now a good time to exit?",
]


@pytest.mark.parametrize("q", PRD_ADVICE_QUESTIONS)
def test_advice_questions_are_flagged(q):
    assert guards.is_advice(q) is True


# --- The seven positive queries must NOT be advice -----------------------

POSITIVE_QUERIES = [
    "expense ratio of HDFC Large Cap Fund",
    "exit load on HDFC Flexi Cap Fund",
    "minimum SIP amount",
    "lock-in period HDFC ELSS",
    "benchmark of HDFC Balanced Advantage",
    "riskometer level HDFC Small Cap",
    "expense ratio of HDFC Large Cap Fund?",
]


@pytest.mark.parametrize("q", POSITIVE_QUERIES)
def test_positive_queries_are_not_advice(q):
    """The spec names this the most damaging bug in the phase."""
    assert guards.is_advice(q) is False


@pytest.mark.parametrize(
    "q",
    [
        "what is the exit load and lock-in period",
        "which scheme has the lowest expense ratio",
        "exit load",
        "how to download capital gains statement",
    ],
)
def test_factual_phrasing_is_not_advice(q):
    assert guards.is_advice(q) is False


@pytest.mark.parametrize(
    "q",
    [
        "what do you recommend",
        "advise me on HDFC ELSS",
        "is HDFC Small Cap suitable for me",
        "can i buy the flexi cap fund",
        "worth investing in large cap?",
        "should we exit HDFC Flexi Cap",
    ],
)
def test_advice_phrasings_are_flagged(q):
    assert guards.is_advice(q) is True


# --- Redaction helper ----------------------------------------------------

def test_redact_never_echoes_the_secret():
    raw = "my PAN is ABCDE1234F and my phone is 9876543210"
    out = guards.redact(raw)
    assert "ABCDE1234F" not in out
    assert "9876543210" not in out
    assert out.startswith("redacted:")


def test_redact_passthrough_when_clean():
    assert guards.redact("exit load on HDFC Flexi Cap Fund") == "no-pii"


def test_empty_input_is_safe():
    assert guards.detect_pii("") is None
    assert guards.is_advice("") is False

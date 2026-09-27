"""Deterministic input guards. Pure Python, no LLM call, no I/O.

These run before retrieval, so a PII-bearing or advice-seeking question costs
nothing and never reaches the API. That ordering is the point: the cheapest
possible rejection is the one that never needs the model.

The hard requirement is ASYMMETRY. A false negative is a privacy incident; a
false positive is an annoying refusal. The patterns are therefore tuned to
miss rather than fire, and the advice keywords are matched with the care that
factual questions demand -- "expense ratio of HDFC Large Cap Fund" and "exit
load" must never be treated as advice.
"""

from __future__ import annotations

import re

# Order matters: the most specific patterns run first so that, e.g., a PAN
# containing digits does not get reported as a phone number.
PII_PATTERNS: dict[str, re.Pattern] = {
    "pan": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "aadhaar": re.compile(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"\b[6-9]\d{9}\b"),
    "account": re.compile(r"\b\d{12}\b"),
}

# A bare 4-6 digit number is NOT an OTP. "exit load 1%" and "expense ratio
# 1.03%" are numbers, not secrets. An OTP is only an OTP when it sits next to
# words that mean "one-time secret".
_OTP_CONTEXT = re.compile(
    r"\b(otp|one[\s-]?time\s*(password|code|pin)|verification\s*code|"
    r"auth(entication)?\s*code|banking\s*code|secure\s*code)\b",
    re.IGNORECASE,
)
_OTP_DIGITS = re.compile(r"\b\d{4,6}\b")

# Advice detection. Matched against the lowercased question.
#
# Deliberately NOT included as bare words: "buy", "sell", "hold", "best".
# "Which equity fund holds the largest..." and "the best performing fund" are
# factual questions, and the PRD forbids return/performance claims anyway --
# so those should be answered from context or refused by the no-match gate, not
# by a keyword that fires on the noun. Every entry below is therefore either a
# multi-word phrase or a verb in an unmistakably imperative construction.
_ADVICE_PHRASES = (
    r"\bshould\s+i\b",
    r"\bshould\s+we\b",
    r"\bshould\s+(buy|sell|hold|switch|invest|exit|redeem|add)\b",
    r"\b(buy|sell|switch|redeem|exit|invest\s+in)\s+(it|this|that|the\s+\w+)\b",
    r"\bwhich\s+(is|one)?\s*best\b",
    r"\bwhat('s|s| is)\s+the\s+best\b",
    r"\bbest\s+(fund|scheme|option|choice)\b",
    r"\bwhich\s+(fund|scheme)\s+(should|do)\b",
    r"\bsuitable\s+for\s+(me|my)\b",
    r"\bgood\s+for\s+(my|me)\b",
    r"\bmy\s+portfolio\b",
    r"\bgood\s+time\s+to\b",
    r"\brecommend\b",
    r"\badvise\b",
    r"\badvice\s+on\b",
    r"\bworth\s+(investing|buying)\b",
    r"\bis\s+it\s+safe\s+to\b",
    r"\bcan\s+i\s+(buy|sell|invest)\b",
)
_ADVICE_RE = re.compile("|".join(_ADVICE_PHRASES))


def detect_pii(text: str) -> str | None:
    """Return the kind of PII found, or None.

    Args:
        text: the raw user question. Callers must NOT log it when this returns
            non-None -- the whole point of detecting it is that it should not be
            stored or echoed anywhere.
    """
    if not text:
        return None

    upper = text.upper()

    # PAN and email first: most specific, and least likely to be a coincidence.
    if PII_PATTERNS["pan"].search(upper):
        return "pan"
    if PII_PATTERNS["email"].search(text):
        return "email"

    # Aadhaar before phone/account: a 12-digit Aadhaar also satisfies \d{12},
    # and reporting "account" for it would be less useful to the user.
    for kind in ("aadhaar", "phone", "account"):
        if PII_PATTERNS[kind].search(text):
            return kind

    # OTP last, and only with context. Without the context check this pattern
    # would fire on every fee percentage in the corpus.
    if _OTP_CONTEXT.search(text) and _OTP_DIGITS.search(text):
        return "otp"

    return None


def is_advice(question: str) -> bool:
    """True for buy/sell/hold/switch/suit-my-portfolio questions."""
    if not question:
        return False
    return bool(_ADVICE_RE.search(question.lower()))


def redact(text: str) -> str:
    """A loggable rendering of a question: pattern kinds only, no values.

    Used so that guard decisions can be audited without ever writing the
    sensitive string itself to a log.
    """
    kinds = detect_pii(text)
    if not kinds:
        return "no-pii"
    return f"redacted:{kinds}"

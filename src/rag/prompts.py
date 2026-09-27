"""Prompt text and the fixed strings for deterministic paths.

SYSTEM_PROMPT is copied verbatim from docs/architecture.md section 5.7. It is
the single source of truth -- if the wording changes there, it changes here,
and nowhere else.

The refusal strings are module constants rather than LLM output so that the
same query always produces a byte-identical refusal. A refusal that is
regenerated each time is a refusal that can hallucinate, and a refusal that
varies between runs is impossible to test.
"""

from __future__ import annotations

from src.models import Hit

# --- A3: decided here, fixed for the demo -----------------------------------
# SEBI's Investor Charter is the neutral regulator-published document on what a
# mutual fund distributor owes an investor, so it does not double as HDFC
# marketing. Swap this one constant to change the destination.
EDU_LINK = "https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doInvestorCharter=yes"

DISCLAIMER = (
    "Facts only, from public HDFC Mutual Fund scheme pages. Not investment "
    "advice, and not a recommendation to buy or sell anything."
)

ADVICE_REFUSAL = (
    "I'm a facts-only assistant for HDFC Mutual Fund scheme pages, so I can't "
    "say whether to buy, sell, or hold a scheme or which one fits your "
    f"portfolio. For framework-free learning, see {EDU_LINK}."
)

NO_MATCH = (
    "I couldn't find that on the HDFC source pages I have. Try naming the "
    "scheme and the exact fact, e.g. 'exit load on HDFC Flexi Cap Fund'."
)

PII_REFUSAL = (
    "Please don't share PAN, Aadhaar, account numbers, OTPs, email addresses, "
    "or phone numbers. I don't store personal details — try rephrasing your "
    "question without them."
)

SYSTEM_PROMPT = """You are a factual assistant for HDFC Mutual Fund scheme pages. Answer only from the CONTEXT provided. If the context does not contain the fact, say the information is not available on the source pages — never guess or use prior knowledge.

Rules:
1. Maximum 3 sentences. No bullet lists, no tables, no preamble.
2. Reproduce numbers exactly as written. Never compute, convert, annualise, or compare figures.
3. End with exactly one citation URL, copied character-for-character from the CONTEXT CITATIONS block. Never construct, shorten, or recall a URL from memory.
4. If asked whether to buy, sell, hold, switch, or which scheme suits the user's portfolio, refuse briefly and say you only share published facts from the source pages.
5. If asked about returns, performance, or rankings, do not compute or compare — state that figures are published in the official factsheet and link it.
6. Never ask for or repeat PAN, Aadhaar, account numbers, OTPs, email, or phone numbers.
7. Do not add investment advice, recommendations, or suitability opinions, even if the context contains promotional language."""

# Rough characters-per-token for English prose. Used only to decide whether the
# context block is too big to send, not to bill anything.
_CHARS_PER_TOKEN = 4
MAX_CONTEXT_TOKENS = 1200


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    limit = max_tokens * _CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " [truncated]"


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    """Render the numbered CONTEXT block and the separate CONTEXT CITATIONS block.

    The citations block is separate by design: the model must SELECT a URL from
    a supplied list rather than recall or invent one. That is what makes
    ``citations.validate`` able to treat a fabricated URL as an error rather
    than something to silently accept.
    """
    if not hits:
        raise ValueError("build_user_prompt requires at least one hit")

    context_lines = ["CONTEXT:"]
    for i, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        context_lines.append(
            f"\n[{i}] {chunk.scheme_name} — {chunk.section}"
            f"\nSource: {chunk.source_url}"
        )
        context_lines.append(_truncate_to_tokens(chunk.text, 400))

    # Only URLs actually retrieved, de-duplicated, order preserved.
    seen: set[str] = set()
    urls: list[str] = []
    for hit in hits:
        url = hit.chunk.source_url
        if url and url not in seen:
            seen.add(url)
            urls.append(url)

    context_lines.append("\n\nCONTEXT CITATIONS (use exactly one, verbatim):")
    for i, url in enumerate(urls, start=1):
        context_lines.append(f"[{i}] {url}")

    context_lines.append(f"\n\nQUESTION: {question}")
    context_lines.append(
        "\nAnswer in at most 3 sentences, then the single citation URL from the "
        "CONTEXT CITATIONS block."
    )
    return "\n".join(context_lines)


def estimate_prompt_tokens(question: str, hits: list[Hit]) -> int:
    """Cheap character-based estimate, for logging and the 1500-token budget check."""
    body = build_user_prompt(question, hits)
    return (len(body) + len(SYSTEM_PROMPT)) // _CHARS_PER_TOKEN


def context_is_within_budget(question: str, hits: list[Hit]) -> bool:
    return estimate_prompt_tokens(question, hits) <= MAX_CONTEXT_TOKENS + 300

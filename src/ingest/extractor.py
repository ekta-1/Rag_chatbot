"""Turn raw HTML into ordered, structured text blocks.

Design note: ``section`` and ``kind`` are preserved rather than flattening the
page into one string (architecture section 5.2). They cost almost nothing and
are what make heading-aware chunking, table-safe splitting, and precise
citations possible in later phases.

Tables are flattened one ``<tr>`` per line with cells joined by ``" | "``. That
is deliberate, not cosmetic: it keeps ``Expense ratio | 0.65%`` as adjacent text
so a later chunk boundary can never separate a label from its value.
"""

from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup, Tag

from src.config import CONFIG
from src.models import Document, Source, TextBlock

log = logging.getLogger(__name__)

STRIP_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg")

# Substrings matched (lowercase) against a tag's class and id attributes.
# The structural entries matter as much as the cookie ones: Groww wraps its
# entire nav and marketing shell in custom elements such as
# ``<div class="header2025_headerContainer__...">``, which ``STRIP_TAGS`` cannot
# catch because the tag name is not literally ``<header>``. Without these, the
# marketing shell is the single largest block of extracted text and matches
# almost any question.
NOISE_MARKERS = (
    # consent / promo
    "cookie",
    "consent",
    "popup",
    "modal",
    "newsletter",
    "subscribe",
    "advert",
    # site chrome
    "header",
    "footer",
    "navbar",
    "navigation",
    "dropdown",
    "breadcrumb",
    "sidebar",
    "searchbar",
    "searchmodal",
    # Carousels of *other* schemes by the same fund manager. These are bare
    # fund-name strings, so a query about HDFC Balanced Advantage would match a
    # block that merely lists it -- wrong scheme, confidently cited.
    "fundmanagement_funds",
    "carousel",
)

HEADING_TAGS = {f"h{i}" for i in range(1, 7)}
CONTAINER_TAGS = {"body", "main", "article", "section", "div", "main"}
LIST_TAGS = {"ul", "ol"}
PROSE_TAGS = {"p", "blockquote", "figcaption", "dd", "dt"}
SKIP_TAGS = {"button", "input", "select", "textarea", "br", "hr", "img", "path", "symbol"}

# Runs of 12+ digits are treated as account-number shaped and redacted.
LONG_DIGIT_RUN = re.compile(r"\d{12,}")
ZERO_WIDTH = re.compile(r"[\u200b-\u200f\ufeff]")
WHITESPACE = re.compile(r"[ \t\r\f\v]+")
BLANK_LINES = re.compile(r"\n{3,}")

# Sections that are noise or actively forbidden for a facts-only assistant.
# Matched case-insensitively as a prefix of the section name.
#   - "Return calculator" / "Returns": the PRD forbids performance claims, and
#     this section literally contains 1Y/3Y return figures.
#   - "Compare similar funds": a comparison table of rival funds' returns.
#   - "Holdings": thousands of stock names; pure retrieval noise for a fee/rule
#     FAQ, and it dwarfs every useful section in the document.
BLOCKED_SECTION_PREFIXES = (
    "return calculator",
    "returns",
    "compare similar funds",
    "holdings",
    "performance",
    "nav history",
    "risk-reward",
)

# Site chrome and marketing that leaks into headingless sections. These blocks
# are the bulk of the page by character count and would otherwise dominate
# retrieval, since they match almost any question.
BLOCKED_CONTENT_MARKERS = (
    "demat account",
    "begin your stock market journey",
    "invest in stocks, etfs",
    "etf screener",
    "stock screener",
    "buy now, pay later",
    "intraday",
    "open a free account",
    "download the groww app",
    "credit loan against securities",
    "personal loan",
    "start a sip",
    "coupon",
    "refer and earn",
    # Performance tables. The PRD forbids performance claims, and these blocks
    # are pure 1Y/3Y/5Y/10Y return figures plus a category rank.
    "fund returns",
    "category average",
    "returns / 1 year",
    "historic returns",
)

# Blocks *starting* with these are dropped. The AMC's postal address is
# identical on all five pages, so leaving it in means every query spends top-k
# slots on five copies of the same boilerplate.
BLOCKED_CONTENT_PREFIXES = ("address",)

# The summary sentence that reports fund size and NAV, matched tightly so it
# cannot swallow surrounding prose. The value itself is deliberately not part
# of the pattern: the number is wrong and may change, but the sentence shape is
# stable, and a pattern that hard-codes "9,86,237" would silently stop working
# the day Groww edits the figure.
#
# The lookahead matters. The sentence also contains a NAV like "1,189.08", whose
# decimal point is not a sentence end, so a plain ``[^.]*?`` stopped there and
# left a bare "08." fragment in the corpus. Requiring the period to be followed
# by whitespace or end-of-string skips decimal points and consumes the whole
# sentence.
GENERATED_SUMMARY_RE = re.compile(
    r"The fund currently has an Asset Under Management\s*\(AUM\)\s*of"
    r".*?\.(?=\s|$)",
    re.IGNORECASE | re.DOTALL,
)

# Fields the chatbot must be able to answer. Reported after each ingest so a
# client-rendered fee table is caught at index time, not at question time.
COVERAGE_FIELDS = {
    "expense ratio": ("expense ratio", "expense ratios"),
    "exit load": ("exit load", "exit loads"),
    "minimum sip": ("minimum sip", "min sip", "minimum investment", "sip amount"),
    "lock-in": ("lock-in", "lock in", "lockin"),
    "benchmark": ("benchmark", "tri benchmark", "nifty"),
    "riskometer": ("riskometer", "risk meter"),
    "statement": ("statement", "capital gains", "tax statement", "download"),
}


def clean_text(raw: str) -> str:
    """Collapse whitespace, strip zero-width characters, redact digit runs."""
    if not raw:
        return ""
    text = ZERO_WIDTH.sub("", raw)
    text = LONG_DIGIT_RUN.sub("[redacted]", text)
    text = WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = BLANK_LINES.sub("\n\n", text)
    return text.strip()


def clean_prose(raw: str) -> str:
    """Clean a paragraph, collapsing the source file's line wrapping.

    HTML source formatting puts newlines mid-sentence. Those are meaningless in
    prose but are structural in tables and lists, so only prose goes through
    here.
    """
    return WHITESPACE.sub(" ", clean_text(raw).replace("\n", " "))


def _attr_text(value) -> str:
    """Normalise a bs4 attribute to a string.

    ``tag.get("class")`` returns a list for multi-valued attributes, which
    breaks any naive string join.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def _is_noise(tag: Tag) -> bool:
    haystack = " ".join(
        _attr_text(tag.get(attr)) for attr in ("class", "id", "data-testid")
    ).lower()
    return any(marker in haystack for marker in NOISE_MARKERS)


def _flatten_table(table: Tag) -> str:
    """One line per ``<tr>``, cells joined by ``" | "``. Never splits a row."""
    lines: list[str] = []
    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"], recursive=False)
        if not cells:
            continue
        values = [clean_text(cell.get_text(" ", strip=True)) for cell in cells]
        values = [v for v in values if v]
        if values:
            lines.append(" | ".join(values))
    return "\n".join(lines)


def _flatten_list(node: Tag) -> str:
    lines: list[str] = []
    for item in node.find_all("li", recursive=False):
        text = clean_text(item.get_text(" ", strip=True))
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines)


def _has_block_children(tag: Tag) -> bool:
    return any(
        isinstance(child, Tag) and child.name in CONTAINER_TAGS | LIST_TAGS | HEADING_TAGS
        for child in tag.children
    )


def _walk(node: Tag, section: str):
    """Yield ``(section, kind, text)`` triples in document order."""
    for child in node.children:
        if not isinstance(child, Tag) or child.decomposed:
            continue

        name = (child.name or "").lower()

        if name in STRIP_TAGS or name in SKIP_TAGS or name == "head":
            continue
        if _is_noise(child):
            continue

        if name in HEADING_TAGS:
            heading = clean_text(child.get_text(" ", strip=True))
            if heading:
                section = heading
            continue

        if name == "table":
            text = clean_text(_flatten_table(child))
            if text:
                yield section, "table", text
            continue

        if name in LIST_TAGS:
            text = clean_text(_flatten_list(child))
            if text:
                yield section, "list", text
            continue

        if name in PROSE_TAGS:
            text = clean_prose(child.get_text(" ", strip=True))
            if text:
                yield section, "prose", text
            continue

        if name in CONTAINER_TAGS:
            if _has_block_children(child):
                yield from _walk(child, section)
            else:
                text = clean_prose(child.get_text(" ", strip=True))
                if text:
                    yield section, "prose", text
            continue

        # Unknown inline-ish element: recurse so nested structure is not lost.
        yield from _walk(child, section)


def _blocked_section(section: str) -> bool:
    lowered = section.strip().lower()
    return any(lowered.startswith(prefix) for prefix in BLOCKED_SECTION_PREFIXES)


def _strip_generated_summary(text: str) -> str:
    """Remove the AI-generated summary's fund-size and NAV claims from prose.

    Groww renders a machine-written summary paragraph per scheme. Its numbers
    are not per-scheme: the identical wrong AUM (9,86,237 Cr) appears on all
    five pages, and it is the AMC's house-level figure. The sentence also
    carries a NAV, which :mod:`src.ingest.structured` excludes on purpose under
    the PRD's no-performance-claims rule, so letting it through prose would
    defeat that exclusion.

    Only that one sentence is removed. The rest of the paragraph is kept, so
    this does not silently discard the manager bio and risk sentence that
    follow it. The authoritative AUM is ingested from ``__NEXT_DATA__``.

    Whitespace is collapsed afterwards: removing a mid-paragraph sentence
    leaves a double space where it was, which would otherwise be stored.
    """
    stripped = GENERATED_SUMMARY_RE.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", stripped)


def _blocked_content(text: str) -> bool:
    lowered = text.lower().lstrip()
    if any(lowered.startswith(prefix) for prefix in BLOCKED_CONTENT_PREFIXES):
        return True
    return any(marker in lowered for marker in BLOCKED_CONTENT_MARKERS)


def extract(
    html: str,
    source: Source,
    ingested_at: str,
    stats: dict | None = None,
) -> Document:
    """Parse HTML into a :class:`Document` of ordered :class:`TextBlock`.

    Malformed HTML does not raise -- BeautifulSoup is lenient. Only a page that
    yields zero blocks is an error, because that means the selectors drifted.

    Args:
        stats: optional dict updated with drop counts, so the CLI can report how
            much was filtered and the team can audit the blocklist.
    """
    counters = stats if stats is not None else {}
    counters.setdefault("dropped_short", 0)
    counters.setdefault("dropped_section", 0)
    counters.setdefault("dropped_content", 0)

    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup.find_all(STRIP_TAGS):
        if not tag.decomposed:
            tag.decompose()
    # Decomposing an outer element also decomposes its descendants, which are
    # still present in the list we are iterating -- skip those explicitly.
    for tag in soup.find_all(True):
        if tag.decomposed:
            continue
        if _attr_text(tag.get("class")) or _attr_text(tag.get("id")):
            if _is_noise(tag):
                tag.decompose()

    root = soup.body or soup
    default_section = "Overview"
    blocks: list[TextBlock] = []
    min_chars = CONFIG.min_block_chars

    for section, kind, text in _walk(root, default_section):
        section = section or default_section
        if len(text) < min_chars:
            counters["dropped_short"] += 1
            continue
        if _blocked_section(section):
            counters["dropped_section"] += 1
            continue
        if _blocked_content(text):
            counters["dropped_content"] += 1
            continue
        text = _strip_generated_summary(text).strip()
        if len(text) < min_chars:
            counters["dropped_short"] += 1
            continue
        blocks.append(TextBlock(section=section, text=text, kind=kind))

    if not blocks:
        raise ValueError(
            f"extraction produced no blocks for {source.scheme_key} ({source.url}); "
            "the page structure has probably changed"
        )

    doc = Document(source=source, blocks=blocks, ingested_at=ingested_at)
    log.info(
        "extracted %d blocks (%d chars) for %s [dropped: %d short, %d section, %d content]",
        len(blocks),
        sum(len(b.text) for b in blocks),
        source.scheme_key,
        counters["dropped_short"],
        counters["dropped_section"],
        counters["dropped_content"],
    )
    return doc


def coverage_report(doc: Document) -> dict[str, bool]:
    """Which answerable fields appear anywhere in the document.

    This is the early-warning system for the risk that fee tables are rendered
    client-side: a field missing here is a field the bot will refuse to answer.
    """
    haystack = " ".join(b.text for b in doc.blocks).lower()
    return {
        field: any(needle in haystack for needle in needles)
        for field, needles in COVERAGE_FIELDS.items()
    }


def missing_fields(doc: Document) -> list[str]:
    return [f for f, present in coverage_report(doc).items() if not present]

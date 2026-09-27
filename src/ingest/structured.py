"""Extract authoritative scheme facts from the page's embedded JSON payload.

Groww is a Next.js app: the figures a mutual fund FAQ actually needs (expense
ratio, exit load, lock-in, minimum SIP, benchmark, riskometer) are **not** in
the visible HTML. They live in the ``__NEXT_DATA__`` script payload as
structured JSON. Scraping the rendered text for them is unreliable, and the
numbers are the one thing this project cannot get wrong.

So the corpus has two halves:

* ``extractor.py``  -- prose from the rendered page (rules, tax treatment,
  how to download a statement).
* this module       -- exact, labelled values straight from the page's own data.

Design notes:
* The field list is an explicit **allowlist**. Anything not listed is never
  ingested, which is how we guarantee the PRD's "no performance claims" rule --
  ``return_stats``, ``simple_return``, ``sip_return``, ``groww_rating`` and
  friends are reachable in this payload and are deliberately left out.
* A missing payload is not an error. The pipeline falls back to prose only.
"""

from __future__ import annotations

import json
import logging
import re

from src.models import Source, TextBlock

log = logging.getLogger(__name__)

NEXT_DATA_RE = re.compile(
    r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S | re.I
)

# Fields that must never be ingested: they are returns, ratings, or price
# history. The PRD forbids computing or comparing returns, so they are excluded
# at the source rather than filtered downstream.
EXCLUDED_FIELDS = frozenset(
    {
        "return_stats",
        "simple_return",
        "sip_return",
        "nav",
        "nav_date",
        "groww_rating",
        "crisil_rating",
        "peerComparison",
        "holdings",
        "analysis",
        "fund_news",
        "portfolio_turnover",
        "historic_exit_loads",
        "historic_fund_expense",
    }
)

# ``aum`` was in EXCLUDED_FIELDS and has been moved to FACT_SECTIONS.
#
# It was excluded under the "no performance claims" rule, but fund size is a
# static attribute of the scheme, not a return: nothing computes or compares it,
# so the PRD does not forbid stating it. Excluding it was actively harmful.
#
# Groww's page renders the AUM twice, and the two disagree:
#
#   * the "Fund size (AUM)" stat card, and the page's own FAQ JSON-LD, both
#     give the scheme's real figure (large_cap: 39933.3663 -> Rs 39,933.37 Cr)
#   * an auto-generated summary paragraph claims a different, much larger
#     number, and prints the SAME wrong figure on all five scheme pages
#     (9,86,237 Cr), which is HDFC Mutual Fund's house-level AUM
#
# With ``aum`` excluded, nothing authoritative reached the corpus, so the only
# fund-size string available to retrieval was that generated sentence -- the
# answer was confidently wrong on all five funds. This module is the only
# trustworthy source for the figure, which is why it is read from here.

# (json key, human label, formatter) grouped into topical sections.
FACT_SECTIONS: dict[str, list[tuple[str, str, str]]] = {
    "Fees, exit load and investment limits": [
        ("expense_ratio", "Expense ratio", "pct"),
        ("exit_load", "Exit load", "text"),
        ("lock_in", "Lock-in period", "duration"),
        ("min_sip_investment", "Minimum SIP amount", "inr"),
        ("min_investment_amount", "Minimum lump sum investment", "inr"),
        ("min_withdrawal", "Minimum withdrawal amount", "inr"),
        ("stamp_duty", "Stamp duty", "text"),
    ],
    "Benchmark, risk and scheme profile": [
        ("aum", "Fund size (AUM)", "cr"),
        ("benchmark", "Benchmark", "text"),
        ("benchmark_name", "Benchmark name", "text"),
        ("nfo_risk", "Riskometer", "text"),
        ("category", "Category", "text"),
        ("sub_category", "Sub-category", "text"),
        ("launch_date", "Launch date", "text"),
        ("fund_manager", "Fund manager", "text"),
        ("registrar_agent", "Registrar and transfer agent", "text"),
        ("description", "Scheme objective", "text"),
    ],
}


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in {"", "-", "NA", "N/A", "null", "None"}
    if isinstance(value, dict):
        # lock_in arrives as {"years": 3, "months": 0, "days": 0} or all-null.
        return all(v in (None, 0, "") for v in value.values())
    return False


def _month_year(iso: str | None) -> str | None:
    """'2022-07-28T18:30:00.000Z' -> 'Jul 2022'."""
    if not iso or not isinstance(iso, str):
        return None
    try:
        year, month = iso[:4], int(iso[5:7])
    except (ValueError, IndexError):
        return None
    names = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    if not 1 <= month <= 12:
        return None
    return f"{names[month - 1]} {year}"


def _fund_manager_value(data: dict) -> str | None:
    """Resolve the fund manager(s) from ``fund_manager_details``.

    The payload carries two disagreeing fields:

    * ``fund_manager`` -- a flat, denormalized string. It appears in the SEO
      meta record and the compare-funds search record, has no tenure, and is
      stale: for large_cap it names 'Prashant Jain', who appears nowhere in
      that scheme's management section.
    * ``fund_manager_details`` -- an array of ``{person_name, date_from, ...}``
      that the page renders directly as the Fund Management accordion, one
      entry per manager with their tenure.

    The rendered array is authoritative, so this reads it and preserves the
    page's own ordering. A scheme may be co-managed: large_cap lists two
    managers and balanced_advantage lists six, so only the first name is not
    an option -- reporting one would understate who runs the money.

    Tenure is included because the page shows it and a user asking "who
    manages this?" usually wants to know whether the tenure is current.
    """
    details = data.get("fund_manager_details")
    if not isinstance(details, list) or not details:
        return None

    parts: list[str] = []
    seen: set[str] = set()
    for entry in details:
        if not isinstance(entry, dict):
            continue
        name = " ".join(str(entry.get("person_name") or "").split())
        if not name or name in seen:
            continue
        seen.add(name)
        since = _month_year(entry.get("date_from"))
        parts.append(f"{name} (since {since})" if since else name)

    return ", ".join(parts) or None


# Facts whose value comes from a structured array rather than a flat field.
# Mapped to the resolver that reads it, so extract_facts() and facts_summary()
# cannot drift apart on the same fact.
ARRAY_RESOLVERS = {"fund_manager": _fund_manager_value}


def _resolve_values(data: dict) -> dict:
    """Overlay array-derived facts on top of the flat payload."""
    resolved = dict(data)
    for key, resolver in ARRAY_RESOLVERS.items():
        value = resolver(data)
        if value:
            resolved[key] = value
    return resolved


def _indian_grouping(value: float, places: int = 2) -> str:
    """Format a number the way Groww does: last three digits, then pairs.

    Indian numbering, not thousands separators -- ``113606.47`` is written
    ``1,13,606.47``, which is what the page's own stat card shows. Using
    ``f"{value:,.0f}"`` here would print ``113,606`` and quietly disagree with
    the source on a figure the user may check by eye.

    >>> _indian_grouping(39933.3663)
    '39,933.37'
    >>> _indian_grouping(113606.46602051)
    '1,13,606.47'
    """
    fixed = f"{value:.{places}f}"
    whole, _, frac = fixed.partition(".")
    sign = ""
    if whole.startswith("-"):
        sign, whole = "-", whole[1:]
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return f"{sign}{whole}.{frac}" if frac else f"{sign}{whole}"


def _format(value, kind: str) -> str | None:
    """Render a raw JSON value as display text, or None to drop the row."""
    if _is_missing(value):
        return None

    if kind == "pct":
        try:
            return f"{float(value):.2f}%"
        except (TypeError, ValueError):
            return str(value)

    if kind == "inr":
        try:
            return f"INR {float(value):,.0f}"
        except (TypeError, ValueError):
            return str(value)

    if kind == "cr":
        try:
            return f"{_indian_grouping(float(value))} Cr"
        except (TypeError, ValueError):
            return str(value)

    if kind == "duration":
        parts = []
        for unit, suffix in (("years", "year"), ("months", "month"), ("days", "day")):
            amount = value.get(unit)
            if amount:
                parts.append(f"{int(amount)} {suffix}{'s' if int(amount) != 1 else ''}")
        return ", ".join(parts) if parts else None

    text = " ".join(str(value).split())
    return text or None


def load_server_data(html: str) -> dict:
    """Return the ``mfServerSideData`` object, or ``{}`` if it is absent."""
    match = NEXT_DATA_RE.search(html or "")
    if not match:
        log.warning("no __NEXT_DATA__ payload found; prose-only extraction")
        return {}
    try:
        payload = json.loads(match.group(1))
        return payload["props"]["pageProps"]["mfServerSideData"] or {}
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        log.warning("could not parse __NEXT_DATA__ (%s); prose-only extraction", exc)
        return {}


def extract_facts(html: str, source: Source) -> list[TextBlock]:
    """Build labelled fact blocks from the embedded payload.

    Returns an empty list when the payload is missing, so the caller can fall
    back to prose extraction. Each block is one topical section rendered as
    ``label | value`` rows, matching the row-wise table convention used by
    :mod:`src.ingest.extractor`.
    """
    data = load_server_data(html)
    if not data:
        return []
    data = _resolve_values(data)

    blocks: list[TextBlock] = []
    for section, fields in FACT_SECTIONS.items():
        rows: list[str] = []
        for key, label, kind in fields:
            if key in EXCLUDED_FIELDS:
                continue
            value = _format(data.get(key), kind)
            if value:
                rows.append(f"{label} | {value}")
        if rows:
            blocks.append(TextBlock(section=section, text="\n".join(rows), kind="table"))

    if not blocks:
        log.warning("payload present but no known facts extracted for %s", source.scheme_key)
    else:
        log.info("extracted %d structured fact blocks for %s", len(blocks), source.scheme_key)
    return blocks


def facts_summary(html: str) -> dict[str, str | None]:
    """Flat ``{label: value}`` view of every extracted fact, for the CLI report."""
    data = load_server_data(html)
    data = _resolve_values(data)
    summary: dict[str, str | None] = {}
    for fields in FACT_SECTIONS.values():
        for key, label, kind in fields:
            if key in EXCLUDED_FIELDS:
                continue
            summary[label] = _format(data.get(key), kind)
    return summary

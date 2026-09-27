"""Corpus-integrity checks against the page's own authoritative values.

Retrieval eval (:mod:`scripts.eval_retrieval`) asks *"did we find the right
chunk?"*. It cannot ask *"is the chunk true?"*, because a fact that was extracted
wrongly is retrieved perfectly: right chunk, rank 1, real URL, one sentence.
That is exactly how a wrong fund size sat in the corpus and answered a user
confidently, backed by a citation that genuinely supported it.

These checks close that gap. They re-read the cached HTML, take the value the
page states authoritatively, and assert the corpus agrees. Two failure modes
are covered:

``conflicting``
    The same labelled fact appears more than once in the corpus with different
    values -- the corpus contradicts itself, and a user cannot tell which is
    right.

``missing``
    The page states the fact but it never reached the corpus, so the chatbot
    will refuse a question the source can answer.

Both are cheap, offline, and read only from ``cache/raw_html`` plus
``data/documents``. Run directly::

    .venv/bin/python -m evals.integrity
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG  # noqa: E402
from src.ingest.fetcher import cache_path_for  # noqa: E402
from src.ingest.structured import (  # noqa: E402
    _format,
    _resolve_values,
    load_server_data,
)
from src.ingest.sources import load_sources  # noqa: E402

# Labelled facts we hold the corpus to. Each maps a friendly name to
# (json key, formatter) -- the same pairing structured.py uses, so a value that
# agrees here agrees with what was ingested.
CHECKED_FACTS: dict[str, tuple[str, str]] = {
    "Fund size (AUM)": ("aum", "cr"),
    "Expense ratio": ("expense_ratio", "pct"),
    "Exit load": ("exit_load", "text"),
    "Minimum SIP amount": ("min_sip_investment", "inr"),
    "Lock-in period": ("lock_in", "duration"),
    "Benchmark": ("benchmark", "text"),
    "Riskometer": ("nfo_risk", "text"),
    "Fund manager": ("fund_manager", "text"),
}

# A "label | value" row as rendered into a fact block.
ROW_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 ()/&.'-]{2,40}?)\s*\|\s*(.+?)\s*$", re.M)


@dataclass
class Issue:
    scheme_key: str
    fact: str
    kind: str  # "conflicting" | "missing"
    detail: str


def _norm(value: str) -> str:
    """Compare on digits and units only, ignoring formatting and separators.

    ``₹39,933.37 Cr`` and ``39,933.37 Cr`` are the same fact; whitespace and
    currency marks should not decide whether we report a conflict.
    """
    return re.sub(r"[\s₹]|cr", "", value.lower())


def authoritative_facts(scheme_key: str) -> dict[str, str]:
    """The value the page itself states, keyed by friendly fact name.

    ``_resolve_values`` matters here. Without it this would read the flat
    ``fund_manager`` field, which is stale on four of five schemes -- and the
    guard would then compare the corpus against the same wrong source and
    report a clean pass. That is how 'Prashant Jain' survived: the corpus and
    the flat field agreed, so consistency was never evidence of correctness.
    """
    path = cache_path_for(scheme_key)
    if not path.exists():
        return {}
    data = _resolve_values(
        load_server_data(path.read_text(encoding="utf-8", errors="replace"))
    )
    out: dict[str, str] = {}
    for label, (key, kind) in CHECKED_FACTS.items():
        value = _format(data.get(key), kind)
        if value:
            out[label] = value
    return out


def corpus_facts(scheme_key: str, documents_dir: Path | str | None = None) -> dict[str, list[str]]:
    """Every value stored in the corpus for each labelled fact."""
    base = Path(documents_dir) if documents_dir is not None else Path(CONFIG.documents_dir)
    path = base / f"{scheme_key}.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    found: dict[str, list[str]] = {}
    for block in doc.get("blocks", []):
        for label, value in ROW_RE.findall(block.get("text", "")):
            if label in CHECKED_FACTS:
                found.setdefault(label, []).append(value)
    return found


def check_scheme(scheme_key: str, documents_dir: Path | str | None = None) -> list[Issue]:
    issues: list[Issue] = []
    truth = authoritative_facts(scheme_key)
    stored = corpus_facts(scheme_key, documents_dir)

    for label, expected in truth.items():
        values = stored.get(label, [])
        if not values:
            issues.append(Issue(scheme_key, label, "missing", f"page says {expected!r}, corpus has nothing"))
            continue
        agree = {_norm(v) for v in values}
        if len(agree) > 1:
            issues.append(
                Issue(scheme_key, label, "conflicting", f"corpus holds {sorted(values)}; page says {expected!r}")
            )
        elif _norm(expected) not in agree:
            issues.append(
                Issue(scheme_key, label, "conflicting", f"corpus has {sorted(values)}; page says {expected!r}")
            )
    return issues


def check_all() -> list[Issue]:
    issues: list[Issue] = []
    for source in load_sources():
        issues.extend(check_scheme(source.scheme_key))
    return issues


def main() -> int:
    sources = [s.scheme_key for s in load_sources()]
    print("=" * 78)
    print("G. CORPUS INTEGRITY -- corpus vs the page's own values")
    print("=" * 78)

    issues = check_all()
    checked = sum(len(authoritative_facts(k)) for k in sources)
    print(f"\n  {len(sources)} schemes, {checked} fact comparisons against cached HTML")

    if not issues:
        print("\n  PASS  every checked fact matches the page, with no self-conflicts\n")
        return 0

    for issue in issues:
        print(f"  FAIL  [{issue.kind:11s}] {issue.scheme_key:20s} {issue.fact}")
        print(f"          {issue.detail}")
    print(f"\n  {len(issues)} integrity failure(s)\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Parse the canonical source list from ``data/sources.md``.

The markdown table in that file is the single source of truth for the corpus.
Nothing else in the codebase may hardcode a URL.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import CONFIG
from src.models import Source

log = logging.getLogger(__name__)

REQUIRED_FIELDS = ("scheme_key", "category", "scheme_name", "url")


def _is_separator_row(cells: list[str]) -> bool:
    return all(set(c) <= set("-: ") and "-" in c for c in cells)


def _parse_row(cells: list[str]) -> Source:
    values = dict(zip(REQUIRED_FIELDS, cells))
    missing = [k for k, v in values.items() if not v]
    if missing:
        raise ValueError(f"sources row missing {missing}: {cells}")
    url = values["url"]
    if not url.startswith("https://"):
        raise ValueError(f"source url must be https, got {url!r}")
    return Source(
        scheme_key=values["scheme_key"],
        category=values["category"],
        scheme_name=values["scheme_name"],
        url=url,
    )


def load_sources(path: str | Path | None = None) -> list[Source]:
    """Parse the markdown table in ``data/sources.md`` into ``Source`` records.

    Raises:
        ValueError: if the file has no parsable table, a row is missing a field,
            a ``scheme_key`` is duplicated, or a url is not https.
    """
    path = Path(path) if path else Path(CONFIG.sources_path)
    if not path.exists():
        raise FileNotFoundError(f"source list not found: {path}")

    sources: list[Source] = []
    seen_keys: set[str] = set()

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < len(REQUIRED_FIELDS):
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        if cells[0] == "scheme_key":  # header row
            continue
        if _is_separator_row(cells[: len(REQUIRED_FIELDS)]):
            continue

        source = _parse_row(cells[: len(REQUIRED_FIELDS)])
        if source.scheme_key in seen_keys:
            raise ValueError(f"duplicate scheme_key in {path}: {source.scheme_key}")
        seen_keys.add(source.scheme_key)
        sources.append(source)

    if not sources:
        raise ValueError(f"no sources parsed from {path}")

    log.info("loaded %d sources from %s", len(sources), path)
    return sources

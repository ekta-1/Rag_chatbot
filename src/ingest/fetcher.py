"""Fetch source pages over HTTP with a cache-first policy.

Cache-first is architecture decision D10: raw HTML is stored under
``cache/raw_html/`` and reused on subsequent runs, so a live demo can never be
broken by a DOM change or a flaky network at presentation time. Re-fetch only
when ``refresh=True``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from src.config import CONFIG

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}
BACKOFF_SECONDS = (2, 4, 8)


def cache_path_for(scheme_key: str, cache_dir: str | Path | None = None) -> Path:
    base = Path(cache_dir) if cache_dir else Path(CONFIG.raw_html_dir)
    return base / f"{scheme_key}.html"


def fetch(
    url: str,
    scheme_key: str,
    *,
    refresh: bool = False,
    delay: float | None = None,
    retries: int | None = None,
    cache_dir: str | Path | None = None,
) -> str:
    """Return raw HTML for ``url``, caching it by ``scheme_key``.

    Args:
        url: page to fetch.
        scheme_key: cache key (the file stem).
        refresh: ignore any cached copy and re-fetch.
        delay: unused here; politeness sleeping belongs to the caller that owns
            the request loop. Accepted so the signature stays stable for tests.
        retries: number of attempts for retryable failures.
        cache_dir: override the cache location (used by tests).

    Raises:
        requests.HTTPError: on a non-retryable status or exhausted retries.
        requests.RequestException: if the network fails on every attempt.
    """
    path = cache_path_for(scheme_key, cache_dir)

    if path.exists() and not refresh:
        html = path.read_text(encoding="utf-8", errors="replace")
        log.info("cache hit for %s (%d bytes)", scheme_key, len(html))
        return html

    attempts = retries if retries is not None else CONFIG.fetch_retries
    headers = {"User-Agent": CONFIG.user_agent, "Accept-Language": "en-IN,en;q=0.9,en-US;q=0.8"}
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            log.info("GET %s (attempt %d/%d)", url, attempt, attempts)
            response = requests.get(
                url, headers=headers, timeout=CONFIG.request_timeout_seconds
            )
            if response.status_code in RETRY_STATUS:
                raise requests.HTTPError(
                    f"retryable status {response.status_code}", response=response
                )
            response.raise_for_status()
            html = response.text

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")
            log.info("cached %s -> %s (%d bytes)", scheme_key, path, len(html))
            return html

        except (requests.RequestException, requests.HTTPError) as exc:
            last_error = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = status is None or status in RETRY_STATUS
            if not retryable or attempt == attempts:
                break
            wait = BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)]
            log.warning("fetch failed for %s (%s); retrying in %ss", scheme_key, exc, wait)
            time.sleep(wait)

    raise RuntimeError(f"failed to fetch {url} after {attempts} attempts: {last_error}")

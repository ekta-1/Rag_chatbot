"""Shared fixtures.

The notable piece here is the LLM rate-limit throttle. The answer layer
deliberately has no retry loop -- a retry during a live demo reads as a hang --
so when the provider rate-limits us the chain falls straight back to NO_MATCH.
That policy is right for the app and wrong for the test suite, which fires ~25
back-to-back calls and would otherwise measure Groq's free-tier RPM cap instead
of the product.

So the throttle lives in the test harness, not in ``src/``. The alternative --
adding a retry loop "just for tests" -- would mean the tested path is not the
shipped path, and the no-retry guarantee in architecture.md 5.7 would no longer
be exercised by anything.

Measured against Groq's free tier (qwen/qwen3.8-27b, 2026-09-27):

    x-ratelimit-limit-requests : 1000      (never the binding constraint)
    x-ratelimit-limit-tokens   : 8000/min  (the binding one)

A single answer costs roughly 2200 tokens -- top_k=4 chunks at up to 400 tokens
of context each, plus up to 300 output -- so the real ceiling is about three
questions per minute, not thirty. Spacing tests by requests would never be
enough; the interval has to be derived from tokens.

This is also a real constraint on a live demo, not just on tests: budget
roughly one question every 20 seconds or the provider returns HTTP 429 and the
chain shows NO_MATCH.
"""

from __future__ import annotations

import os
import time

import pytest

# ~2200 tokens per call against an 8000 tokens/min budget leaves headroom for
# the token accounting to be imprecise. The suite makes ~17 live calls, so a full
# acceptance run takes roughly six minutes.
MIN_SECONDS_BETWEEN_LLM_CALLS = float(
    os.getenv("LLM_TEST_MIN_INTERVAL", "20.0")
)

_last_call_at: float = 0.0


@pytest.fixture(autouse=True)
def throttle_llm_calls(request):
    """Space out live-LLM tests to stay inside the provider's rate limit.

    Skipped entirely for non-LLM tests so the offline suite is unaffected.
    """
    if "llm" not in request.keywords:
        yield
        return

    global _last_call_at
    now = time.monotonic()
    wait = _last_call_at + MIN_SECONDS_BETWEEN_LLM_CALLS - now
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.monotonic()
    yield

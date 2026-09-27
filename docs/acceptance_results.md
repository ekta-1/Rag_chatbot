# Acceptance Results — PRD §8 / `implementation.md` §6.1

**`Actual` and `Pass/Fail` are intentionally empty.** Per `implementation.md` §6.1, A7 is
the single highest-value check in the project and must be done by a human opening the cited
URL — *"Do not mark a row passed on intent."* An automated run cannot substitute for that, and
a row marked green without the URL being opened is worse than an empty row, because it looks
like evidence.

Fill the `Actual` and `Pass/Fail` columns in by hand -- that is what `implementation.md`
§6.1 asks for, and no automated run can stand in for it. The `Automated coverage` column
tells you which rows already have a machine check behind them, so you know what a failure
would contradict.

| Test | Command | Expected | Actual | Pass/Fail | Automated coverage |
|---|---|---|---|---|---|
| **A1** clean clone → `cli ingest` → `cli ask` → UI answers | `python -m src.cli ingest` then `streamlit run src/app.py` | index builds, UI answers | | | none — needs `GROQ_API_KEY` |
| **A2** expense ratio | `python -m src.cli ask "expense ratio of HDFC Large Cap Fund?"` | ≤3 sentences, exactly 1 link, figure matches the page | | | `tests/test_citations.py`, `tests/test_acceptance.py::test_a2_*` (llm) |
| **A3** ELSS lock-in | `python -m src.cli ask "lock-in period for HDFC ELSS Tax Saver Fund?"` | 3 years, exactly 1 link | | | `eval_retrieval.py` (retrieval), `test_acceptance.py::test_a3_elss_lock_in` (llm) |
| **A4** minimum SIP | `python -m src.cli ask "minimum SIP amount for HDFC Flexi Cap Fund?"` | figure + 1 link | | | `eval_retrieval.py` (retrieval), `test_acceptance.py::test_a3_elss_lock_in` (llm) |
| **A5** advice | `python -m src.cli ask "Should I buy HDFC Small Cap Fund?"` | refusal, no numbers, `EDU_LINK` | | | `tests/test_guards.py`, `tests/test_chain.py` (stubbed generator) |
| **A6** out-of-corpus | `python -m src.cli ask "What is HDFC Bank's FD interest rate?"` | not found, no fabrication | | | `tests/test_chain.py`, `scripts/eval_retrieval.py` |
| **A7** every sample answer re-verified by opening the cited URL | open each link in `data/sample_qa.md` | figure present on the live page | | | `evals/integrity.py` compares the corpus to **cached** HTML — not a substitute for opening the live page |
| **A8** PII | paste a PAN and an email into the chat | refused, nothing stored or echoed | | | `tests/test_guards.py`, `tests/test_app.py` |
| **A9** disclaimer on first render | `streamlit run src/app.py` | visible without scrolling | | | `tests/test_app.py::test_disclaimer_visible_on_first_render` |
| **A10** same question twice | ask the same question twice | same answer, same citation | | | `test_acceptance.py::test_a10_*` (llm); `Generator` is temperature 0 and does not retry |

## Live run status (machine-observed, not a sign-off)

A key is configured and the live answer layer has been exercised end to end against Groq
(`qwen/qwen3.8-27b`):

```bash
.venv/bin/python -m pytest tests/test_acceptance.py -v -m llm   # 17 passed in 5m20s
```

All 17 live tests passed, covering A2, A3, A4, A5, A6, A10, G1 (6 queries), the PII
pre-flight refusal, and the manifest-derived `Last updated` line. `data/sample_qa.md` now holds
5/5 verbatim captured answers rather than three `PENDING` rows.

Two things this does **not** establish:

- **A7 is still outstanding.** It requires opening each cited URL on the live page. The tests
  assert the citation is *one of the retrieved URLs*; they cannot assert the figure is still
  published there today.
- **`Actual` / `Pass/Fail` stay blank on purpose.** A green test run is evidence for your
  judgement, not a substitute for it.

### Operational note: the rate limit is tokens, not requests

Groq's free tier allows 1000 requests/min but only **8000 tokens/min**, and one answer costs
~2200 tokens (4 chunks x 400 tokens of context + 300 output). The effective ceiling is
therefore about **three questions per minute**. Ask faster and the chain returns `NO_MATCH`,
because a `429` is handled by the no-retry fallback rather than by waiting.

Budget roughly 20 seconds per demo question. `tests/conftest.py` spaces the live tests to match;
if you see `G1 violated` in a run, check for a `429` in the log before believing it.

## What is already machine-verified

Run these and they are green today, with no key:

```bash
.venv/bin/python -m pytest -q                    # 296 passed, 1 skipped  (5m25s with a key)
.venv/bin/python scripts/eval_retrieval.py       # 25/28
.venv/bin/python -m evals.integrity              # 5 schemes, 36 comparisons
```

The single skip is the A6 statement-download case, deferred deliberately: statements are not in
the five-source corpus, so the expected answer is a refusal and the query is kept out of the
retrieval eval.

### Known retrieval gaps, carried into the demo

These are real and deliberately unfixed. They are listed here so a passing row above is not
mistaken for a claim that retrieval is perfect.

| Query | Behaviour | Cause |
|---|---|---|
| `"stamp duty"` | refused, though coverage is 1.000 | cosine 0.225 loses to `MIN_SIMILARITY=0.30`; no threshold value fixes it without leaking `"SIP date"` at 0.245 |
| `"who manages HDFC Large Cap Fund"` | `Overview` outranks `Fund management` | `manages` ≠ `management` (no stemming); fund-name tokens dilute coverage to a flat 0.584 |
| `"P/E ratio of …"` | 4 hits returned for a fact absent from the corpus | `ratio` matches `Expense ratio | 1.03%`; fund-name tokens alone carry coverage. Risk: the answer layer may quote the expense ratio as a P/E. |

The third is the one to watch in a live demo. It is the only known case where retrieval hands
the model an **invitation to fabricate**, and citation validation would not catch it.

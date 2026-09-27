# Mutual Fund Facts-Only RAG Chatbot

## Overview

A FAQ assistant that answers factual questions about HDFC Mutual Fund schemes from public pages only. Every answer cites one source link. No investment advice.

Pipeline: Loading → Chunking → Embedding (all-MiniLM-L6-v2) → ChromaDB → Retrieval → Answer.

See [docs/PRD.md](docs/PRD.md), [docs/architecture.md](docs/architecture.md) and [docs/implementation.md](docs/implementation.md).

## Scope

**What it covers.** HDFC Mutual Fund. Five schemes, listed in [data/sources.md](data/sources.md):
Large Cap, Flexi Cap, ELSS Tax Saver, Small Cap, and Balanced Advantage — all Direct–Growth.

**Where the facts come from.** The `__NEXT_DATA__` JSON payload embedded in each Groww page, not
scraped visible text. That is the single most important implementation decision in the project:
it is why expense ratio is exactly `1.03%` rather than whatever a regex managed to grab, and it is
why fund size and fund manager were wrong before they were read from here.

**What it will not do.** No investment advice, no recommendation to buy or sell, no return
projections or comparisons, no PII. Answers are capped at three sentences with exactly one source
link, and are labelled with the ingest date.

**Out of scope by construction:** any scheme not in the five above, and any fact those pages do
not publish (portfolio holdings detail, P/E ratio, statement-download instructions). Those are
refused rather than guessed.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add GROQ_API_KEY and set GROQ_MODEL
```

One env file, `.env`, holding settings and credentials. It is git-ignored, so your key is
never committed. `.env` is a dotfile and therefore hidden in Finder — `Cmd+Shift+.` reveals it,
or just run `code .env` / `nano .env`.

An earlier revision split credentials into a separate `llm.env` so the visible file held only
tunables. That is gone: it meant an empty `GROQ_API_KEY=` in the second file could silently mask
a working key in the first, because a later `load_dotenv(..., override=True)` wins on conflict.
One file has no such ordering to get wrong.

| Setting | Where | Notes |
|---|---|---|
| `GROQ_API_KEY` | `.env` | required for answers; get one at console.groq.com/keys |
| `GROQ_MODEL` | `.env` | **verify it** — see below |
| `LLM_PROVIDER` | `.env` | `groq` (default) or `anthropic` |

**Do not trust the model in `.env.example`; verify it.** An unavailable model id fails at the
API with an HTTP 404 whose message ("model does not exist or you do not have access to it")
looks like a key problem. Run:

```bash
.venv/bin/python scripts/list_models.py
```

It lists what your key can reach, flags the non-chat models (speech-to-text, prompt-injection
classifiers) that the same endpoint also serves, and says whether your configured model is
usable.

**The app runs without a key.** Retrieval, both gates, the debug panel and the refusal paths all
work; the answer layer returns "not found" rather than erroring. To get real answers:

```bash
.venv/bin/python scripts/list_models.py   # verifies the key and lists usable model ids
```

## Usage

Build the index once, then run the UI:

```bash
python -m src.cli ingest      # fetch + extract + embed + store (~1 min)
streamlit run src/app.py      # the demo
```

`ingest` is idempotent — re-running it skips work unless you pass `--force`
(rebuild after a model change) or `--refresh` (re-fetch the pages). It is **cache-first**: with
no `--refresh` it reads `cache/raw_html/`, so it completes with the network off.

You can also drive it from the terminal:

```bash
python -m src.cli ask "expense ratio of HDFC Large Cap Fund?"
python -m src.cli ask --explain "exit load on HDFC Flexi Cap Fund"   # gate signals
python -m src.cli ask --retrieve-only "minimum SIP amount"          # no API call
python -m src.cli inspect                                          # index stats
```

### Checking it actually works

Retrieval is verifiable with no key. Start here:

```bash
.venv/bin/python scripts/eval_retrieval.py    # 25/28, incl. all 6 required query types
.venv/bin/python -m evals.integrity           # corpus vs the page's own values
.venv/bin/python -m pytest -q                 # 296 passed, 1 skipped (5m25s; a key is required)
.venv/bin/python scripts/dump_index.py --query "exit load"   # inspect chunks + vectors
.venv/bin/python scripts/rehearse_readme.py   # proves every command above actually runs
```

`scripts/eval_retrieval.py` exits non-zero while the three documented retrieval failures are open,
so it works as a pre-demo gate. `scripts/rehearse_readme.py` exists because §6.1 of the plan says
to fix the README rather than your memory — it executes the commands on this page and fails if one
of them is broken.

### Demo script

See [docs/demo_script.md](docs/demo_script.md) for the timed three-minute storyboard, and
[docs/acceptance_results.md](docs/acceptance_results.md) for the A1–A10 sweep with `Actual`
columns left blank for a human to fill in.

### Retrieval settings

Tuned on this corpus and documented in `docs/architecture.md` §5.6:

| Setting | Value | Why |
|---|---|---|
| `MIN_SIMILARITY` | `0.30` | loose floor; on its own it **cannot** separate answerable from unanswerable questions |
| `MIN_LEXICAL_COVERAGE` | `0.55` | the gate that does the work — IDF-weighted overlap |
| `TOP_K` | `4` | |

**Both gates must pass.** A single cosine threshold does not work here: the
unanswerable "HDFC Bank's FD interest rate?" scores `0.561`, above the answerable
"minimum SIP amount" at `0.445`.

## Sample Q&A

See [data/sample_qa.md](data/sample_qa.md) — 5 rows (3 factual, 1 advice refusal, 1
out-of-corpus). It is **generated** by `scripts/build_sample_qa.py`, which runs the real chain, so
no answer in it is transcribed by hand.

Three factual rows are currently marked `PENDING`: generating them needs `GROQ_API_KEY`. The
advice-refusal and out-of-corpus rows need no key, because both return before generation is
reached, so they are captured verbatim.

Regenerate after adding a key:

```bash
.venv/bin/python scripts/build_sample_qa.py
```

## Disclaimer

> Facts only, from public HDFC Mutual Fund scheme pages. Not investment advice, and not a
> recommendation to buy or sell anything.

The full text, the long version, and the other fixed strings are in
[data/disclaimer.md](data/disclaimer.md), which is generated from `src/rag/prompts.py` so the
running app and the written text cannot drift apart.

## Known limits

**The corpus is 5 pages.** Any question about another scheme, or about these schemes beyond the
facts Groww publishes on them, is refused. That is by design, but it means "not found" usually
means "outside the corpus", not "does not exist".

**Figures can be stale.** Everything comes from the `ingested_at` timestamp in
`data/index_manifest.json`. A fund's expense ratio or AUM changes; the stored copy does not.
Re-run `python -m src.cli ingest --refresh` before relying on a number that matters.

**The same fact appears twice on the source page, and the page disagrees with itself.** Groww
renders fund size and fund manager in both a JSON payload and a generated summary paragraph, and
the generated copy is wrong — it printed the AMC's house-level AUM on all five scheme pages, and
named a former manager on four of five. Both bugs shipped into the corpus before testing caught
them. `evals/integrity.py` now compares the corpus against the page's own authoritative values,
but it reads **cached** HTML, so it cannot detect a page that has since changed. Verify anything
load-bearing against the live page.

**HTML parsing is brittle.** A Groww redesign would break extraction silently or loudly; `ingest`
raises if a page yields zero blocks, but a partial redesign could yield plausible wrong text.

**The embedding model is not finance-tuned.** `all-MiniLM-L6-v2` is a general-purpose sentence
encoder, so "AUM", "fund size" and "corpus size" are not synonyms to it. It works here because a
second, lexical gate does the discriminating work — but cosine on its own is close to useless on
this corpus (lowest inter-chunk similarity is 0.41, above the 0.30 floor).

**Retrieval is not perfect, and three known failures are documented rather than fixed.** See
[docs/acceptance_results.md](docs/acceptance_results.md). The sharpest one: a question about a
fact we do not have (e.g. a P/E ratio) can still retrieve chunks, because the fund's name alone
clears the lexical gate. The fix changes a documented invariant and is awaiting sign-off.

**Answers are short by design and will miss nuance.** Three sentences and one citation. A
question with a legitimate multi-part answer gets a partial answer, not a caveat-laden essay.

**No investment advice, by design.** "Should I buy X?" is refused before retrieval runs, with a
link to SEBI's investor charter. This is a pre-filter, not a prompt instruction, so it holds even
if the model is unhelpful.


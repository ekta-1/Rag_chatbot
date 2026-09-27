# Architecture — Mutual Fund Facts-Only RAG Chatbot

**Companion to** [`PRD.md`](PRD.md) · **Status:** Draft for class demo · **Last updated:** 2026-09-27

This document turns the PRD into a concrete technical design: components, module layout, data contracts,
control flow, and the engineering trade-offs behind each choice. It also carries the **chunking-strategy
decision record** the PRD left open (§6.2).

---

## 1. Architecture at a Glance

A deliberately **linear, two-phase, offline-index / online-query** pipeline. There is no agent loop, no
tool calling, and no multi-step planner — for a 5-page corpus that would add failure modes without
adding accuracy.

```
┌──────────────────────── OFFLINE (once per corpus refresh) ────────────────────────┐
│                                                                                  │
│  data/sources.md                                                                 │
│        │                                                                         │
│        ▼                                                                         │
│  ┌────────────────┐   HTTP + BeautifulSoup    ┌──────────────────┐               │
│  │  Fetch         │ ───────────────────────► │  Extract         │               │
│  │  (requests)    │                          │  (main + section)│               │
│  └────────────────┘                          └──────────────────┘               │
│                                                        │ text blocks             │
│                                                        ▼                         │
│                                              ┌──────────────────┐                │
│                                              │  Chunk           │  ◄── DECISION  │
│                                              │  (section-aware) │      §5        │
│                                              └──────────────────┘                │
│                                                        │ List[Chunk]             │
│                                                        ▼                         │
│                                              ┌──────────────────┐                │
│                                              │  Embed           │                │
│                                              │  all-MiniLM-L6-v2│  384-dim        │
│                                              └──────────────────┘                │
│                                                        │ vectors + metadata      │
│                                                        ▼                         │
│                                              ┌──────────────────┐                │
│                                              │  ChromaDB        │  persistent    │
│                                              │  hdfc_mf_faq     │  ./chroma_db   │
│                                              └──────────────────┘                │
└──────────────────────────────────────────────────────────────────────────────────┘
                                                          │
                        build manifest (chunk count, ingested_at, failures)
                                                          ▼
                                                    data/index_manifest.json
                                                          │
┌───────────────────────── ONLINE (per user question) ─────────────────────────────┐
│                                                                                  │
│  User query                                                                     │
│      │                                                                           │
│      ├─► [0] PII guard ──hit──► refuse, never log                                │
│      │                                                                           │
│      ├─► [0b] Intent gate (advice?) ──hit──► fixed refusal + educational link    │
│      │                                                                           │
│      ▼                                                                           │
│  ┌────────────────┐   same embedding model    ┌──────────────────┐               │
│  │  Embed query   │ ───────────────────────► │  ChromaDB        │               │
│  └────────────────┘                          │  query(top_k=4)  │               │
│                                              └──────────────────┘               │
│                                                        │ hits + distances       │
│                                                        ▼                         │
│                                              ┌──────────────────┐                │
│                                              │  Score gate      │                │
│                                              │  min_similarity  │                │
│                                              └──────────────────┘                │
│                                        ┌───────────────┴─────────────┐          │
│                                   none pass                       pass         │
│                                        │                             │          │
│                                        ▼                             ▼          │
│                              "Not in sources"            ┌──────────────────┐    │
│                              + rephrase hint            │  Prompt builder  │    │
│                                                            └──────────────────┘    │
│                                                                    │             │
│                                                                    ▼             │
│                                                            ┌──────────────────┐    │
│                                                            │  Claude          │    │
│                                                            │  ≤3 sentences    │    │
│                                                            │  + 1 citation    │    │
│                                                            └──────────────────┘    │
│                                                                    │             │
│                                                                    ▼             │
│                                                            ┌──────────────────┐    │
│                                                            │  Citation        │    │
│                                                            │  validator       │    │
│                                                            └──────────────────┘    │
│                                                                    │             │
└────────────────────────────────────────────────────────────────────────────────────┘
                                                                     ▼
                                                        Streamlit chat UI
                                          answer + link + "Last updated from sources: …"
```

**Design principle:** everything that can reject a bad answer is a *deterministic Python check placed
before or after the LLM call*. The LLM is used for exactly one thing — composing short prose from
retrieved text. It is never trusted to decide whether it is allowed to answer.

---

## 2. Component Responsibilities

| # | Component | Responsibility | Explicitly **not** its job |
|---|---|---|---|
| 1 | `config` | Load env vars, defaults, single source of truth for constants | Any network or disk I/O |
| 2 | `sources` | Parse `data/sources.md` into `Source` records | Fetching |
| 3 | `fetcher` | HTTP GET, retries, polite delay, cache raw HTML to disk | Parsing/extraction |
| 4 | `structured` | Pull exact scheme facts from the page's embedded JSON payload | Prose extraction |
| 5 | `extractor` | HTML → clean text, preserving heading + section structure, table-aware | Chunk sizing |
| 6 | `chunker` | Text → `Chunk` objects with metadata | Fetching or embedding |
| 7 | `embedder` | Text → 384-dim vector (cached model instance) | Deciding what to embed |
| 8 | `vectorstore` | ChromaDB upsert / query / reset (single choke point for collection name) | Score policy |
| 9 | `retriever` | Query → filtered, score-gated hits | Generation |
| 10 | `guards` | PII detection, advice/refusal classification | Answer text |
| 11 | `prompt_builder` | Render system + user prompt from hits, in one place | Calling the API |
| 12 | `generator` | Groq chat-completions call, token/latency limits | Interpreting policy |
| 13 | `citation_validator` | Post-hoc check that the URL is real and in the corpus | Rewriting content |
| 14 | `app` (Streamlit) | Render, collect input, show answer + link + debug panel | Business logic |

---

## 3. Module Layout

Extends the existing `src/ingest/` and `src/rag/` split (offline vs. online).

```
src/
├── config.py                    # env + constants
├── models.py                    # dataclasses: Source, Document, Chunk, Hit, Answer
├── ingest/
│   ├── sources.py               # parse data/sources.md -> List[Source]
│   ├── fetcher.py               # requests + cache/raw_html/*.html
│   ├── structured.py            # __NEXT_DATA__ payload -> exact labelled facts
│   ├── extractor.py             # BeautifulSoup -> text blocks (+ section headings)
│   ├── chunker.py               # blocks -> List[Chunk]   (§5)
│   ├── embedder.py              # SentenceTransformer wrapper
│   ├── store.py                 # ChromaDB upsert / query / reset
│   └── pipeline.py              # orchestrates the offline run + manifest
├── rag/
│   ├── retriever.py             # embed query, Chroma query, score gate
│   ├── guards.py                # PII + advice detection
│   ├── prompts.py               # SYSTEM_PROMPT, refusal strings, build_context()
│   ├── generator.py             # Groq / Anthropic call
│   ├── citations.py             # citation validation
│   └── chain.py                 # answer(query) -> Answer  (the online entrypoint)
├── app.py                       # Streamlit UI
└── cli.py                       # `ingest` / `ask` / "demo" subcommands
tests/
├── test_extractor.py            # nav/footer stripped, tables preserved as text
├── test_chunker.py              # no mid-number splits, metadata continuity
├── test_guards.py               # PII + advice detection
├── test_citations.py            # hallucinated URL is rejected
├── test_acceptance.py           # A2–A6 from PRD §8
└── data/fixtures/               # saved HTML so tests never hit the network
```

**Entry points**
- `python -m src.cli ingest` → rebuild the index.
- `python -m src.cli ask "expense ratio of HDFC Large Cap?"` → terminal answer, fast to debug.
- `streamlit run src/app.py` → the demo UI.

The CLI `ask` path and the Streamlit path both call `rag.chain.answer()`. One code path, two front
ends — this is what makes acceptance tests A2–A6 cheap to run.

---

## 3a. Phase 1 Findings — What the Real Pages Actually Contain

Recorded after running ingestion against the live Groww pages, because two of
these changed the design.

### F1. The numbers are not in the HTML — they are in a JSON payload

Groww is a Next.js app. Every fact the FAQ needs (expense ratio, exit load,
lock-in, minimum SIP, benchmark, riskometer) lives in a `<script
id="__NEXT_DATA__">` payload under `props.pageProps.mfServerSideData`, **not**
in the rendered markup. A pure HTML-text scrape of the five pages yields
**zero** occurrences of "expense ratio" and only a handful of "exit load".

Consequence: ingestion was extended with `ingest/structured.py`, which reads
that payload and emits labelled `label | value` rows. The corpus is now
**structured facts + prose**, not prose alone:

```
Fees, exit load and investment limits
  Expense ratio | 1.03%
  Exit load | Exit load of 1% if redeemed within 1 year
  Lock-in period | 3 years          (ELSS only; row omitted when null)
  Minimum SIP amount | INR 100
Benchmark, risk and scheme profile
  Benchmark | NIFTY 100 TRI
  Riskometer | Moderately High Riskometer
```

This retires the PRD's largest stated risk (§11 "dynamic fee tables may not be
in static HTML") and materially reduces the numeric-fidelity risk, because the
figures are copied from a typed field rather than parsed out of prose. Values
are formatted once, in `_format()`, from an explicit allowlist.

**Allowlist, not blocklist.** `FACT_SECTIONS` names every field we ingest.
`EXCLUDED_FIELDS` additionally names what must never be ingested —
`return_stats`, `simple_return`, `sip_return`, `nav`, `groww_rating`,
`crisil_rating`, `peerComparison`, `aum`, `historic_*`. They are all present in
the payload and reachable; excluding them at the source is how the PRD's "no
performance claims" rule is enforced structurally rather than by prompt.

### F2. Unfiltered extraction was 96% noise

Naive extraction of one page produced ~14,000 characters, of which the useful
scheme content was under 1,000. The rest was the site's own shell. Four
filters in `extractor.py` cut the five-page corpus to **55 blocks / ~10,500
chars** (from ~76,000):

| Filter | What it removed | Why it matters |
|---|---|---|
| `NOISE_MARKERS` incl. `header`, `footer`, `dropdown`, `carousel` | The entire marketing nav | Groww wraps its shell in `<div class="header2025_headerContainer__…">` — a *custom element*, so `STRIP_TAGS` (which only matches literal `<header>`) cannot catch it |
| `BLOCKED_SECTION_PREFIXES` | Holdings, Return calculator, Compare similar funds | Return figures and rival-fund comparisons — forbidden by the PRD, and holdings were 24,562 of 39,880 chars on one page |
| `BLOCKED_CONTENT_MARKERS` incl. `fund returns`, `category average` | The performance summary table | Same reason, but it sits in an unlabelled section so a section-level rule misses it |
| `BLOCKED_CONTENT_PREFIXES` = `address` | AMC postal address | Byte-identical on all five pages; would consume 4 of 4 `TOP_K` slots |

`extract()` takes an optional `stats` dict and reports what it dropped, so the
blocklist is auditable rather than mysterious.

**The 40-character floor is deliberate and was tested against a lower value.**
Dropping it to 30 chars to keep short factual lines admitted ~40 extra blocks
per page — almost all of them a carousel of *other* HDFC schemes under "Fund
management". A query about HDFC Balanced Advantage would have retrieved a block
that merely *listed* it, producing a confidently-cited wrong-scheme answer. Short
values do not need the floor relaxed, because F1 supplies them exactly.

### F3. Scope gap: the statement-download question cannot be answered

The brief lists "how to download a capital gains statement" as a required query.
**None of the five pages contain it.** A search of the raw HTML for
`how to download` / `download your` / `tax statement` returns 0 hits on all five
pages; the only `capital gains` string on any page sits inside a modal about
exit-load taxation, not about downloading anything.

`cli ingest` reports this as `WARNING: no source covers statement`. Options, in
the team's hands — this is a scope decision, not an implementation detail:

1. Add a Groww help-centre URL to `data/sources.md` (grows the corpus past the
   brief's "5 pages"; may need a different ingestion profile, since help pages
   are not scheme pages).
2. Add the registrar's (CAMS) statement page.
3. Drop the question from the demo and say so in Known limits.

Recommendation: option 3 for the class demo, with the refusal path demonstrated
instead — it is honest, and the "not in my sources" behaviour is itself a graded
feature. Revisit option 1 if the grader expects that question answered.

### F4. Confirmed non-issue: ELSS-only attributes

The coverage report initially warned that `lock-in` was missing from four
schemes. It is not a gap: `lock_in` is `{"years": null, …}` for non-ELSS
schemes, i.e. genuinely absent. The report now distinguishes *missing
everywhere* (a real ingestion failure) from *missing from some schemes* (often
correct), and says so rather than crying wolf.

---

## 4. Data Contracts

```python
@dataclass(frozen=True)
class Source:
    scheme_key: str          # "large_cap"
    category: str            # "Large Cap"
    scheme_name: str         # "HDFC Large Cap Fund"
    url: str                 # canonical source URL (also the citation)

@dataclass
class Document:              # one per source page
    source: Source
    blocks: list[TextBlock]  # ordered, in page order
    ingested_at: str         # ISO-8601 UTC, stamped at fetch time

@dataclass
class TextBlock:
    section: str             # nearest preceding heading, e.g. "Fees and charges"
    text: str                # cleaned text; tables flattened row-wise
    kind: str                # "prose" | "table" | "list"

@dataclass
class Chunk:
    id: str                  # sha1(source_url + section + chunk_index)
    text: str
    source_url: str
    scheme_key: str
    scheme_name: str
    category: str
    section: str
    chunk_index: int
    ingested_at: str
    char_start: int          # offset into the document — lets us re-verify by hand

@dataclass
class Hit:
    chunk: Chunk
    score: float             # 1 - cosine distance, normalised to [0, 1]
    rank: int

@dataclass
class Answer:
    text: str
    citation_url: str | None
    last_updated: str
    refused: bool
    refusal_kind: str | None # "advice" | "pii" | "no_match" | None
    hits: list[Hit]          # always returned — powers the debug panel
```

**ChromaDB collection** (`COLLECTION_NAME=hdfc_mf_faq`): documents = `chunk.text`, embeddings generated
by the same `SentenceTransformer` (no server-side embedding model, so index and query always agree).
Metadata = all `Chunk` fields except `text`, with `scheme_key` and `category` as filterable fields.

**`data/index_manifest.json`** — written after each ingest so the app can show freshness and the demo
never depends on a rebuild:

```json
{ "ingested_at": "2026-09-27T09:14:02Z",
  "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
  "chunker": "section-aware-recursive",
  "chunk_size_tokens": 450,
  "chunk_overlap_pct": 12,
  "chunk_count": 214,
  "sources": [ { "scheme_key": "large_cap", "url": "...", "chunks": 47, "status": "ok" } ],
  "failures": [] }
```

---

## 5. Stage-by-Stage Design

### 5.1 Loading

`fetcher` issues a `GET` per URL with a browser-like `User-Agent`, a **polite inter-request delay**
(~1.5 s — 5 pages, still respectful), and retry with backoff on 429/5xx. Raw HTML is written to
`cache/raw_html/<scheme_key>.html` and reused on subsequent runs so re-runs are deterministic and the
demo is not at the mercy of a live page at presentation time. Each fetch records `ingested_at`.

**Why cache-first.** The PRD's biggest stated risk is DOM drift. A cached snapshot means a broken
selector can never block a live demo; the fix is a one-command re-ingest at worst.

### 5.2 Extraction

`extractor` removes `script`, `style`, `noscript`, `nav`, `header`, `footer`, and cookie/consent
banners, then walks the remaining DOM **in document order**, emitting `TextBlock`s:

- `h1–h6` update the current `section` for all following blocks.
- `<table>` → `kind="table"`, each `<tr>` flattened to `cell | cell | cell` on its own line. Row-wise
  flattening is deliberate: it keeps `Expense ratio | 0.65%` as an adjacent text pair, so even a chunk
  boundary cannot separate the label from its value.
- Lists → `kind="list"`, one item per line with its bullet character preserved.
- Paragraphs → `kind="prose"`.
- Consecutive whitespace collapsed; empty and near-empty blocks dropped.

**Design choice: keep `kind` and `section` instead of flattening to one big string.** They cost almost
nothing and are what make heading-aware chunking, table-safe splitting, and precise citations possible.

**Known risk (PRD §11).** Groww renders some fee data client-side. The extractor records which expected
fields it could not find, and `ingest` prints a **coverage report** per scheme so a missing expense ratio
is caught at index time rather than becoming a wrong answer at query time.

### 5.3 Chunking — decision record

Per PRD §6.2 the strategy is chosen **after inspecting the real data**. This is the recorded method and
the decision it converges on.

**Evaluation procedure (Phase 2).** For each of the 5 pages, after extraction, record:

1. Total blocks and characters; character count per `section`.
2. Heading depth and how uneven the section sizes are.
3. `kind` mix — especially the share of `table` blocks.
4. Whether a single `section` still mixes topics (e.g. one "Fees and charges" block containing both
   expense ratio and exit load).
5. Where the target facts land: expense ratio, exit load, minimum SIP, ELSS lock-in, benchmark,
   riskometer, statement download.

Then run the three candidates over the same data and score each on: **does a chunk containing a target
fact also contain its label and its value?**

| Candidate | Strength | Weakness here |
|---|---|---|
| `RecursiveCharacterSplitter` | Simple, no dependencies, predictable | Splits mid-section, so a chunk can mix fees with tax prose; can split a flattened table row's neighbours apart |
| `SemanticSplitter` | Groups by sentence similarity; tolerant of uneven sections | Requires a second embedding pass at index time (~4× slower ingest); still has no idea where headings are, so it can drift across a section boundary |
| **Heading-aware + recursive fallback** | Uses the structure we already extracted; keeps "Fees" and "Taxes" chunks pure; cheap; trivially explainable in a demo | Depends on the page having usable headings |

**Decision: heading-aware first, recursive character split as the fallback inside oversized sections.**

Rationale: the sections are already extracted at zero extra cost, and mutual-fund pages are strongly
section-structured — "Fees and charges", "Exit load", "Taxation", "Fund facts" are exactly the units a
user's question targets. Semantic splitting buys marginal quality for a 4× slower ingest and harder
demo narrative. The split stays deliberately boring, which is the right call for a 5-page corpus.

**Parameters (tunable, recorded in the manifest):**
- Target ≈ **200 tokens** per chunk, hard cap **240**, **25-token (12.5%) overlap**.
  *(Corrected in Phase 2 from the original 450/600 — see the note below.)*
- Never break inside a `TextBlock` that is a single table row or a list item.
- If a `section` exceeds the hard cap, recursively split it on paragraph boundaries, then sentences,
  then characters — in that order.
- **Never merge across sections.** A chunk belongs to exactly one `section`, so every chunk's `section`
  metadata is trustworthy and citations can name the section, not just the page.
- Prepend `"{scheme_name} — {section}"` to the embedded text so the vector itself carries scheme and
  topic signal. This measurably helps a question like *"lock-in?"* land on the ELSS tax section instead
  of an unrelated ELSS paragraph.

**Parameter correction, and why the 450/600 figures above were wrong.** Phase 1 established that
`all-MiniLM-L6-v2` truncates at 256 wordpiece tokens. A 450–600 token chunk is therefore embedded with
its tail silently discarded, and every fact in that tail is unretrievable with no error anywhere. The
cap has to sit below the model's own limit, not near it. 200/240/25 leaves headroom for the two
special tokens and for the `"{scheme_name} — {section}"` prefix.

**Phase 2 measurements on the real corpus** (recorded in `data/index_manifest.json`):

| Metric | Value |
|---|---|
| Chunks | **26** across 5 schemes (5/5/5/5/6) |
| Wordpiece tokens min / median / mean / max | 47 / 87 / 91.7 / **153** |
| Over the 240 hard cap | **0** |
| Failures | 0 |

**What the numbers mean.** Max 153 against a 240 cap: nothing came close to the limit, and the mean of
91.7 sits at less than half the 200 target. That is not a tuning failure — it is the honest shape of
this corpus. Each HDFC page yields 9–16 short blocks, and a "Fees, exit load and investment limits"
section *is* a single 61–83 token table. There is simply no 200-token run of prose to find, so
packing is bounded by section size, not by the target. Splitting further would only shred the tables.

The consequence to carry into Phase 3: with 26 chunks, `top_k = 4` returns **15% of the entire corpus**.
Precision has to come from the score gate and the answer prompt, not from chunk granularity.

**Two overlap bugs found and fixed by the tests, worth keeping in mind for any future change:**
1. Taking the overlap tail at the *wordpiece* level flattened a table into a mash —
   `percent Expense ratio row 3 | value 3 percent Expense ratio row 4` — violating the
   never-split-a-row invariant on the *next* chunk. `_overlap_tail` now assembles whole trailing
   lines and only falls back to a token slice for lines too long to fit the budget.
2. A single prose line larger than the whole overlap budget was taken *whole*, producing ~198 tokens
   of overlap and pushing the assembled chunk to 397 tokens — over the hard cap. The line-aware path
   now delegates to a token slice in that case, so overlap can never exceed its budget.

The PRD's open question #1 is therefore **resolved**: heading-aware split with recursive fallback,
200/240/25, verified against the real corpus.

### 5.4 Embedding

`sentence-transformers/all-MiniLM-L6-v2` — 384 dimensions, ~90 MB, CPU-viable, loaded **once** per
process and shared by ingest and query so the vectors are guaranteed consistent.

- Input is the prefixed text from §5.3; the stored `chunk.text` stays unprefixed so the UI shows clean
  quotes in the debug panel.
- Batched (`batch_size=32`) at index time.
- The model name is persisted in the manifest. If `EMBED_MODEL` ever changes, the old collection is
  incompatible — `cli ingest` refuses to write into a collection built with a different model unless
  `--force` is passed. Silent dimension mismatch is the classic way a RAG demo starts returning noise.

*Noted limitation:* all-MiniLM-L6-v2 is a general-purpose English model, not finance-tuned. It handles
"expense ratio", "exit load", and "SIP" well enough at this corpus size, but a reranker is the first
upgrade if recall tests (A2–A4) show misses. See PRD §11.

### 5.5 Vector store

ChromaDB, persistent client at `CHROMA_DIR=./chroma_db`, single collection `COLLECTION_NAME`.
All Chroma access lives in `ingest/store.py` so the collection name and the query shape exist in one
file. IDs are deterministic (`sha1(source_url|section|chunk_index)`), so re-ingesting the same page
**upserts in place** instead of duplicating.

Retrieval returns documents + `distances`; we convert to a comparable `score = 1 - distance` and always
clamp to `[0, 1]` for display.

**The collection must be created with `hnsw:space: "cosine"`.** Chroma's default is squared **L2**,
which makes `1 - distance` a meaningless quantity and would invalidate every threshold in Phase 3. This
was a real bug caught during Phase 2: the first build ran on the default L2 space. The collection
metadata is now set explicitly at creation, `store.ensure_collection` refuses to operate on a
collection whose `hnsw:space` is anything other than `cosine`, and `store.stats()` reports the space so
`cli inspect` shows it. Read-only helpers (`count`, `stats`, `get_chunk`, `delete_scheme`) deliberately
bypass the model guard, so introspection still works right after an intentional model change.

**Stale chunk cleanup.** Deterministic IDs make re-ingest idempotent, which has a sharp edge: if a
source stops producing chunks (page redesigned, extraction broken) its *old* chunks survive the upsert
and the bot keeps answering from them. The pipeline therefore calls `delete_scheme()` for any source
that produced no chunks this run, and records `stale_chunks_removed` in the manifest.

### 5.6 Retrieval

```python
hits = retriever.search(question)      # TOP_K = 4, both gates applied inside
```

- Query is embedded with the **same** model. Any divergence here silently destroys retrieval quality,
  so `retriever` imports the embedder rather than loading its own copy.
- **The gate is the primary hallucination defence.** If nothing clears it, the chain stops and returns a
  "not found" answer without ever calling the LLM. The gate lives *inside* `search()` so no caller can
  retrieve-and-answer around it.
- Optional `where` filter on `scheme_key` when the UI exposes a scheme selector — a cheap precision win
  when a user is on the Large Cap page.
- `TOP_K=4` is deliberate: small enough that 4 chunks of context keep the answer short and the demo
  fast, large enough to survive one bad chunk. Note it returns 15% of a 26-chunk corpus, so the gate
  carries most of the precision work.

#### Phase 3 result: the threshold is TWO gates, not one

The plan was a single `MIN_SIMILARITY` placed between the lowest positive and the highest negative
score. **That is not achievable on this corpus, and pretending otherwise would have shipped a broken
refusal path.** Measured top-1 scores:

| Question | Set | Top-1 cosine |
|---|---|---|
| expense ratio of HDFC Large Cap Fund | answerable | 0.816 |
| exit load on HDFC Flexi Cap Fund | answerable | 0.764 |
| riskometer level HDFC Small Cap | answerable | 0.625 |
| benchmark of HDFC Balanced Advantage | answerable | 0.622 |
| lock-in period HDFC ELSS | answerable | 0.526 |
| minimum SIP amount | answerable | 0.445 |
| **What is HDFC Bank's FD interest rate?** | **refuse** | **0.561** |
| **Which fund is best for my portfolio?** | **refuse** | **0.486** |
| how to download capital gains statement (A6) | refuse | 0.327 |
| Tell me about Bitcoin. | refuse | 0.164 |
| What is the weather in Mumbai? | refuse | 0.153 |
| What is the price of gold today? | refuse | 0.150 |

The two unanswerable *finance* questions outscore three answerable ones. Cosine alone gives a gap of
**−0.116** — there is no separating value. The cause is visible in `cli ask --explain`: the FD question
matches every chunk on the single token `hdfc` (coverage 0.078), and "interest rate" is semantically
close to "expense ratio". Pure embedding similarity cannot tell "asks about a number we don't have"
from "asks about a number we do have".

**The fix is a second, orthogonal signal: IDF-weighted lexical coverage** (`src/rag/lexical.py`) —
the fraction of the question's distinctive vocabulary that the chunk actually contains, each term
weighted by `log((N+1)/(df+1))` so ubiquitous words like *fund* and *HDFC* count for little while
*riskometer* and *lock-in* count for a lot.

| Signal | Answerable min | Refuse max | Gap |
|---|---|---|---|
| cosine | 0.445 | 0.561 | **−0.116** (fails) |
| lexical coverage | **0.637** | **0.494** | **+0.143** (works) |

Both gates must pass. Tuned values, sitting inside the margin with room on each side:

- `MIN_SIMILARITY = 0.30` — a loose floor, well under the answerable minimum of 0.445. It is *not* the
  discriminator; keeping it low means it never rejects a genuinely good chunk.
- `MIN_LEXICAL_COVERAGE = 0.55` — between the refuse maximum (0.494) and the answerable minimum
  (0.637). This is the gate doing the real work.

**Ranking is by `cosine × coverage`, not cosine alone.** Cosine alone put the *fees* chunk above the
*benchmark* chunk for "benchmark of HDFC Balanced Advantage" (0.622/0.695 vs 0.607/1.000): the scheme
name in the §5.3 prefix matches every chunk of that fund, so cosine overrates them all equally and the
discriminating term loses. Weighting by coverage restores the correct order — top-1 accuracy went from
5/6 to **6/6**. `Hit.score` remains the raw cosine value, since that is what the thresholds are defined
against.

**The §3.4 cheap fix did not help, and is off by default.** Appending `scheme_key` to the question text
(architecture §11 step 3) changed scores by ≤0.03 and did not fix either colliding negative — those name
no scheme, so there is nothing to append. It is retained behind `augment=` for measurement but defaults
to the plain question, and a cross-encoder reranker is not warranted on this evidence.

**Result: 6/6 answerable queries return the correct fund and section at rank 1; 6/6 refusals return
`[]`** — the 5 planned negatives plus the A6 statement query, which is a positive-shaped question that
no source covers and must therefore be refused. `test_thresholds_have_a_separation_margin` asserts the
sets still separate, so a corpus change or a careless threshold edit fails the suite rather than
silently degrading the refusal path.

### 5.7 Answer generation

`SYSTEM_PROMPT` (single source of truth in `rag/prompts.py`):

> You are a factual assistant for HDFC Mutual Fund scheme pages. Answer **only** from the CONTEXT
> provided. If the context does not contain the fact, say the information is not available on the
> source pages — never guess or use prior knowledge.
>
> Rules:
> 1. Maximum **3 sentences**. No bullet lists, no tables, no preamble.
> 2. Reproduce numbers **exactly** as written. Never compute, convert, annualise, or compare figures.
> 3. End with exactly one citation URL, copied character-for-character from the CONTEXT CITATIONS
>    block. Never construct, shorten, or recall a URL from memory.
> 4. If asked whether to buy, sell, hold, switch, or which scheme suits the user's portfolio, refuse
>    briefly and say you only share published facts from the source pages.
> 5. If asked about returns, performance, or rankings, do not compute or compare — state that figures
>    are published in the official factsheet and link it.
> 6. Never ask for or repeat PAN, Aadhaar, account numbers, OTPs, email, or phone numbers.
> 7. Do not add investment advice, recommendations, or suitability opinions, even if the context
>    contains promotional language.

The user message carries a compact, numbered `CONTEXT` block (rank, scheme, section, text) and a
`CONTEXT CITATIONS` block listing only the URLs that were actually retrieved. Separate citation block =
the model has to select from a supplied list rather than invent one.

**Fixed strings** for the deterministic paths, so refusals read identically every time:

- `ADVICE_REFUSAL` — "I'm a facts-only assistant for HDFC Mutual Fund scheme pages, so I can't say
  whether to buy, sell, or hold a scheme or which one fits your portfolio. For framework-free learning,
  see {EDU_LINK}."
- `NO_MATCH` — "I couldn't find that on the HDFC source pages I have. Try naming the scheme and the
  exact fact, e.g. 'exit load on HDFC Flexi Cap Fund'."
- `PII_REFUSAL` — "Please don't share PAN, Aadhaar, account numbers, OTPs, email addresses, or phone
  numbers. I don't store personal details — try rephrasing your question without them."

`EDU_LINK` is PRD open question #4 (SEBI investor charter vs. HDFC MF education page) — one constant,
decided before Phase 4.

**Token/latency discipline:** `max_tokens=300` (3 sentences + citation fits comfortably), temperature 0
for A10 determinism, and a single API call per question with no retry loop — a retry inside a live demo
reads as a hang.

### 5.8 Citation validation

After generation, `citations.py` runs deterministically:

1. Extract every `http(s)://...` token from the answer.
2. If the model emitted a URL that is **not in the retrieved set** → strip it and re-render the answer
   with the top hit's canonical `source_url`, and log a `citation_mismatch` warning. This catches the
   "almost-right link" failure that is easy to miss in a demo.
3. If **zero** URLs survive and the answer is not a refusal → treat it as a generation failure and
   return `NO_MATCH` rather than shipping an uncited answer. This makes PRD goal G1 ("100% of answers
   carry a link") an enforced invariant, not a hope.
4. Sentence count is checked the same way; over-length answers are truncated at the third sentence
   boundary, with a `truncated=True` flag for the debug panel.

The UI also renders `Last updated from sources: {manifest.ingested_at}` from the manifest, so it is
consistent across answers rather than depending on the model to remember the format.

### 5.9 Online chain

`rag/chain.py` is the single entrypoint both the CLI and Streamlit call:

```
answer(query):
  1. guards.detect_pii(query)      -> PII_REFUSAL          (never logs the raw string)
  2. guards.is_advice(query)      -> ADVICE_REFUSAL        (keyword + phrasing classifier)
  3. hits = retriever.search(query)
  4. not hits                       -> NO_MATCH
  5. prompt = prompts.build(hits)
  6. text  = generator.call(prompt)          # may raise -> fall back to NO_MATCH, never raw traceback
  7. clean = citations.validate(text, hits)  # fix URL, enforce 1 link + <=3 sentences
  8. return Answer(text, citation_url, manifest.ingested_at, hits=hits)
```

Steps 1, 2, and 4 are **cheap, offline, and deterministic** — most bad queries never reach the model.
Step 6's failure mode is a friendly message plus a logged exception, never a stack trace in the UI.

### 5.10 UI

`src/app.py`, Streamlit, intentionally small:

- `st.set_page_config(page_title="HDFC MF Facts Assistant", layout="centered")`.
- Welcome line naming scope: "HDFC Mutual Fund facts-only assistant — 5 schemes, public pages only."
- Disclaimer rendered first, from a single `DISCLAIMER = "Facts-only. No investment advice."`
  constant, and repeated in the sidebar and footer so it is unmissable (PRD A9).
- **3 example questions** as buttons wired to the chat input (the PRD §4 questions, trimmed to 3).
- Chat area: answer text, then the citation as a clickable link, then
  `Last updated from sources: {date}`.
- `st.expander("🔍 Retrieved chunks (debug)")` listing each hit with rank, score, scheme, section, and
  the raw chunk text — the single best feature for a class demo, because it makes the retrieval visible.
- `st.sidebar` shows corpus scope (5 schemes + links), `ingested_at`, embedding model, `TOP_K`, and
  `MIN_SIMILARITY` — so the architecture is legible on screen.
- Input is never written to disk; `st.session_state` holds the current transcript in memory only.
- Clear-chat button resets session state, reinforcing that nothing persists.

---

## 6. Key Engineering Decisions

| # | Decision | Alternatives | Why | Cost accepted |
|---|---|---|---|---|
| D1 | Offline index, online query | Per-question scraping | Scrape latency and rate-limit risk on every query; facts would change under the answer | Index can go stale — mitigated by `ingested_at` display |
| D2b | Groq over its OpenAI-compatible endpoint | LangChain / LlamaIndex / a framework | For 5 pages a framework hides the very stages the assignment is graded on; fewer deps, fewer version surprises, total control of prompt and citation logic. `requests` was already a dependency, so the provider swap added no package. `LLM_PROVIDER=anthropic` keeps the old path alive |
| D2 | Hand-rolled RAG with direct Chroma + a direct HTTP call | LangChain / LlamaIndex / a framework | For 5 pages a framework hides the very stages the assignment is graded on; fewer deps, fewer version surprises, total control of prompt and citation logic | ~200 lines of glue we own |
| D3 | all-MiniLM-L6-v2 | A finance-tuned or OpenAI embedding | Free, offline, CPU-friendly, no extra API cost or key | Lower recall on finance jargon; reranker is the upgrade path |
| D4 | Heading-aware chunking | Recursive-only, semantic | Uses structure we already extracted; keeps topic-pure chunks; 4× cheaper ingest than semantic; explainable in a demo | Breaks if a page has no usable headings → recursive fallback |
| D5 | Score-gated retrieval, no LLM fallback | Always generate, "let the model decide" | The gate is deterministic, free, and testable — the only reliable way to pass A6 | Needs threshold tuning; too high refuses valid questions |
| D6 | Deterministic PII + advice guards in Python | Relying on the system prompt | A prompt request is not an enforcement mechanism; regex on known formats is auditable and testable | Keyword lists need maintenance; may miss phrasings |
| D7 | Post-hoc citation validation + repair | Trusting the model's URL | Enforces G1 as an invariant and catches near-miss URLs that look plausible to a human reviewer | Occasionally overrides a correct answer (fails safe, toward the canonical top hit) |
| D8 | Fixed refusal strings | Model-generated refusals | A refusal must be identical every time — it's a compliance surface, not a creative one | Slightly less tailored; the educational link is constant |
| D9 | `sha1`-based deterministic chunk IDs | UUIDs / autoincrement | Re-ingest upserts in place → no duplicate chunks across refreshes | Must include enough identity fields; collision risk negligible here |
| D10 | Cache raw HTML | Fetch live every run | Demo never breaks on DOM drift or a flaky network; tests never hit the network | Must remember to clear the cache after a fix |
| D11 | Single `chain.answer()` for CLI and UI | Logic in the Streamlit script | Acceptance tests run headless in milliseconds; the CLI becomes a debugging tool | One extra layer of indirection |
| D12 | Chroma behind `store.py` | Chroma calls scattered | Collection name, ID scheme, and query shape in one file → swappable for pgvector/FAISS later | Slight indirection |

---

## 7. Data Flow — Worked Example

Question: **"Is there an exit load on HDFC Flexi Cap Fund?"**

| Step | Component | Action | Result |
|---|---|---|---|
| 1 | `guards.detect_pii` | Regex for PAN/email/phone/OTP | No match → continue |
| 2 | `guards.is_advice` | "should I / which is best / advise" | No match → continue |
| 3 | `retriever` | Embed query | `[0.043, -0.021, …]` (384-d) |
| 4 | `vectorstore` | `query(n_results=4)` | 4 hits, e.g. `large_cap 0.41`, `flexi_cap 0.78`, … |
| 5 | Score gate | `score >= 0.35` | 3 survive; top hit `flexi_cap`, section "Exit load" |
| 6 | `prompt_builder` | Render 3 blocks + citations list | Context ≈ 700 tokens |
| 7 | `generator` | Claude, temp 0, `max_tokens=300` | "Yes — HDFC Flexi Cap Fund (Direct – Growth) charges an exit load of 1% if units are redeemed within 12 months. No exit load applies after 12 months. Source: https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth" |
| 8 | `citation_validator` | URL ∈ retrieved set? Sentences ≤ 3? | Pass, unchanged |
| 9 | UI | Render + link + `ingested_at` | Answer shown, A2-style check satisfied |

If step 7's model had emitted `...hdfc-flexi-cap-fund`, step 8 replaces it with the canonical
`flexi_cap` URL and logs the mismatch.

---

## 8. Failure Modes & Handling

| Failure | Detection | Handling | User sees |
|---|---|---|---|
| Page fetch 4xx/5xx | Non-200 / exception | Retry ×3 with backoff; record in `failures[]`; continue with remaining pages | (Ingest-time report) |
| Selector drift → empty extract | Zero blocks for a page | Fail loudly at ingest; fall back to cached snapshot | (Ingest-time report) |
| Fee data missing (client-rendered) | Coverage report per scheme | Warn at ingest; consider adding the official factsheet URL to `sources.md` | Narrower answers, never invented |
| Chroma empty on first run | Zero collections at query time | UI instructs: `python -m src.cli ingest` | Setup hint |
| Embedding model changed | Model name in manifest ≠ current | Refuse to query, prompt re-ingest with `--force` | Clear error, no silent garbage |
| Top score below threshold | Score gate | `NO_MATCH` | "Couldn't find that…" + rephrase hint |
| Out-of-corpus question | Score gate | `NO_MATCH` | Same as above (PRD A6) |
| Model returns prose from memory | Not in context | Post-check is hard here; mitigated by short max_tokens, strict prompt, and manual re-verification (A7) | Possible — **known limit** |
| Hallucinated / malformed citation | Citation validator | Repair to top-hit URL, or fall back to `NO_MATCH` | Always a valid link |
| Answer over 3 sentences | Citation validator | Truncate at sentence 3, flag | Still ≤3 sentences |
| Provider API error / rate limit | Exception | Log, return `NO_MATCH` message | Friendly, no stack trace. Caveat: a `429` reads as "not found" even though the fact was retrieved; budget ~20 s/question against Groq's 8000 tokens/min |
| Missing `GROQ_API_KEY` | First answer attempt | Log, return `NO_MATCH` message | Ingest and retrieval still work |

**Logging:** structured logs to stdout with stage, `scheme_key`, score, and latency. Raw user queries
containing PII are **never** logged — the guard returns before the logging call.

---

## 9. Testing Strategy

| Layer | What it covers | Notes |
|---|---|---|
| Unit | `extractor` strips nav/footer, preserves table rows; `chunker` never splits a table row, metadata always present | Uses `tests/data/fixtures/*.html` — no network |
| Unit | `guards.detect_pii` on PAN/email/phone/OTP samples; `is_advice` on the PRD's refusal set | Fast, no API |
| Unit | `citations.validate` — strips a hallucinated URL, truncates 4 sentences, converts a no-URL answer to `NO_MATCH` | Fast, no API |
| Integration | `retriever` returns ≥1 above-threshold hit for A2–A4 and none for A6 after threshold tuning | Needs a built index; marked `@pytest.mark.integration` |
| Acceptance | PRD §8 A2–A6 end-to-end through `chain.answer()` | Needs `GROQ_API_KEY`; marked `@pytest.mark.llm` |
| Manual | A7 re-verify every sample answer against the live URL; A9 disclaimer; A10 determinism | Human, at the end of Phase 6 |

`pytest -m "not llm and not integration"` is the fast pre-commit gate and needs neither API key nor index.

---

## 10. Security & Privacy

| Concern | Control |
|---|---|
| PII ingestion | Regex guard runs **before** retrieval; input is never logged, persisted, or embedded |
| PII in source pages | Groww scheme pages carry no user data; if any appears, `extractor` drops long digit runs during cleanup |
| API key | Read from `.env` only; `.env` is gitignored; never logged, never sent to the browser |
| Prompt injection | Corpus is 5 fixed first-party pages under our control; system prompt forbids obeying instructions found in context |
| Outbound requests | Only to the 5 listed URLs plus the configured LLM API; nothing else is fetched |
| Data retention | Streamlit transcript lives in `st.session_state` in memory; clear-chat wipes it |

---

## 11. Scaling Path (if scope ever grows)

Ordered by cost-to-benefit, so the demo path is never blocked by these:

1. **Reranker** (cross-encoder) over the `TOP_K=4` candidates — biggest recall win, ~1 extra local model.
2. **Hybrid retrieval** — BM25 (lexical) unioned with dense. Fees and lock-in periods are exact-match
   numbers; lexical search nails them where embeddings blur.
3. **Query expansion** — append the selected `scheme_key` to the query to sharpen multi-scheme questions.
4. **Fact sheet expansion** — add official HDFC factsheet/SID pages to `sources.md` for higher-authority
   fee and benchmark data.
5. **Multi-AMC** — no code change needed: `sources.md` is already the corpus contract. Only the
   `scheme_key` filter and the UI copy would need widening.
6. **Metadata filtering at query time** — already supported via Chroma `where`; the UI just needs to
   expose it.

None of these are needed for the 5-page demo. Listed so the architecture reads as a platform, not a
script.

---

## 12. Open Decisions Carried Forward

Phase 6 close-out. Rows marked **Open** still need a decision; everything else records what was
actually built, including where the answer differed from the plan.

| # | Question | Recorded answer | Status |
|---|---|---|---|
| A1 | Concrete chunk size / overlap | `200` target, `240` hard cap, `25` overlap, forced by all-MiniLM-L6-v2's 256-token limit. **Correction to §0.1:** character-based sizing was the original assumption; token-based is what shipped, because a "200-character" chunk is ~50 tokens and leaves the model half unused. Actual corpus: 26 chunks, mean 87 tokens — well under the target, because the source blocks are small tables. Granularity is a feature here: it keeps the gates precise. | **Done** (Phase 1) |
| A2 | A single similarity threshold? | **No, and this was measured.** A single threshold provably cannot separate answerable from unanswerable on this corpus: the unanswerable "HDFC Bank's FD interest rate?" scores `0.561` against an answerable "minimum SIP amount" at `0.445` (gap −0.116). Shipped two required gates instead — `MIN_SIMILARITY=0.30` (loose floor) and `MIN_LEXICAL_COVERAGE=0.55` (IDF-weighted, gap +0.143). See §5.6. **Not signed off by the team**, and the values are displayed in the UI sidebar. | **Done, pending sign-off** |
| A3 | `EDU_LINK` for the advice refusal | SEBI's investor charter: `https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doInvestorCharter=yes`. Regulator-hosted, stable, and about investor education rather than any one fund — so it cannot be mistaken for HDFC endorsing a recommendation. | **Done** (Phase 4) |
| A4 | Commit `chroma_db/` or require `ingest`? | **Do not commit the binaries.** They are a derived artefact of the model and the source pages, and a stale committed index is worse than no index: it answers questions from figures that have since changed. The one-liner is `python -m src.cli ingest` (~1 min, cache-first), so **pre-ingest immediately before the demo** and verify `cli inspect` reports the expected chunk count. `cache/raw_html/` is committed deliberately — it makes the demo reproducible with the network off (D10). | **Done** (Phase 5) |
| A5 | Factsheet URLs for missing fields? | **Not needed.** The `__NEXT_DATA__` payload supplies every required field exactly, so no factsheet fallback was needed. The one field it does *not* cover is statement-download guidance, which became A6. | **Done** (Phase 1) |
| A6 | Statement-download question is unanswerable from the 5 pages | **Kept as a refusal, no source added.** Adding a CAMS or help-centre URL would mean a non-Groww source, and the corpus is deliberately 5 public scheme pages. The question is in the required query types and the bot answers "not found", which is the correct behaviour for a source-bounded assistant. Reconsider only if the brief widens. | **Done** (Phase 4) |

### Two data-integrity corrections, found by testing

Neither was visible from the code alone; both produced confidently wrong, correctly-cited
answers. Recorded here because the lesson generalises past this project.

| What | Root cause | Fix |
|---|---|---|
| Fund size (AUM) was wrong on **all five** schemes — the AMC's house-level figure (`9,86,237 Cr`) instead of each scheme's | `aum` sat in `EXCLUDED_FIELDS`, so the only fund-size string reaching the corpus was Groww's auto-generated summary paragraph, which prints the house figure on every page. The authoritative value was in `__NEXT_DATA__` all along and was never read. | Read `aum` from the payload; strip the generated sentence from prose. Fund size is a static scheme attribute, not a return, so the PRD never forbade stating it. |
| Fund manager was wrong on **four of five** schemes | Two disagreeing fields. `fund_manager` is a flat denormalized string that lives in SEO/compare-funds records and is stale. `fund_manager_details[]` is the array the page actually renders as the Fund Management accordion, with tenure. The extractor trusted the flat one. | Read `fund_manager_details`, keep every manager and the tenure. A scheme may be co-managed — large_cap has 2, balanced_advantage has 6 — so taking only the first name would understate who runs the money. |

The generalisable lesson: **a denormalized convenience field is not a source of truth**, and a
guard that compares the corpus against the same wrong field will report a clean pass. That is
why `evals/integrity.py` resolves array-derived facts through the same code path as ingestion
rather than re-reading the raw field.

### Still open

| # | Question | Why it is still open |
|---|---|---|
| — | Three-state coverage rule to replace the AND gate | The AND gate treats coverage 1.000 and 0.584 as equivalent, which causes three known failures (`docs/acceptance_results.md`). The fix changes the "both gates must pass" invariant in §5.6, the README and the UI sidebar, so it needs sign-off. **Not implemented.** |
| — | `augment` default | `Retriever.search` defaults to `augment=True`; §5.6 and the Phase 3 notes say augmentation is off by default. One of the two is wrong. Unchanged pending a decision. |
| — | Team sign-off on `0.30` / `0.55` | A2 was resolved by measurement, not by approval. The values are already user-visible in the UI. |


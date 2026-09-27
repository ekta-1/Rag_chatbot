# Implementation Guide — Mutual Fund Facts-Only RAG Chatbot

**Companion to** [`PRD.md`](PRD.md) and [`architecture.md`](architecture.md) · **Status:** Ready to build
· **Last updated:** 2026-09-27

Phase-by-phase build instructions. Each phase lists the exact files to create, the contracts each module
must satisfy, the traps to avoid, and the commands that prove the phase works. Work them **in order** —
every phase depends on the manifest and index produced by the one before it.

**How to use this with Cursor:** open the project root, then paste the *Cursor prompt* block at the end
of each phase verbatim. The prompts are written to be self-contained. After each phase, run the
**Verify** commands yourself and read the output before moving on.

---

## 0. Before You Start — Read This

### 0.1 Correction to the PRD: chunk size

The PRD (§6.2) and architecture (§5.3) originally targeted **300–600 token chunks**. That is wrong for
the chosen embedding model:

> **`sentence-transformers/all-MiniLM-L6-v2` has `max_seq_length = 256` wordpiece tokens.**

Any chunk longer than 256 tokens is **silently truncated** at embedding time. The tail of a 450-token
chunk contributes nothing to retrieval — so a fact sitting at the end of a section becomes effectively
invisible, and the failure looks like "the bot just doesn't know" rather than an error.

**Corrected parameters (use these everywhere):**

| Parameter | Value | Note |
|---|---|---|
| `CHUNK_SIZE_TOKENS` | **200** | Comfortably under 256 after prefix + special tokens |
| `CHUNK_HARD_CAP` | **240** | Absolute ceiling |
| `CHUNK_OVERLAP_TOKENS` | **25** | ~12% of 200, same ratio as the PRD intended |

Use the real tokenizer to count, not `len(text.split())` — wordpiece ≠ words:

```python
tokenizer = SentenceTransformer(EMBED_MODEL).tokenizer
n = len(tokenizer.encode(text, add_special_tokens=True))
```

**Action:** update `CHUNK_SIZE_TOKENS = 200` / `CHUNK_OVERLAP_TOKENS = 25` into PRD §6.2 and
architecture §5.3 when you finish Phase 2, and note the 256-token reason there.

### 0.2 No LangChain — hand-roll the splitter

Architecture decision **D2** rules out RAG frameworks so the pipeline's stages stay visible. That means
`RecursiveCharacterSplitter` is **not** available. Write a ~25-line recursive splitter in
`src/ingest/chunker.py` over plain character separators. Do **not** add `langchain` or
`langchain-text-splitters` to `requirements.txt`.

### 0.3 Ground rules

- One module, one responsibility. No business logic in `app.py` or `cli.py`.
- **Type-hint every public function.** These signatures are the contract; other modules import them.
- No `print()` in library code — use `logging`. CLI/UI are the only printers.
- No network access from tests. Save HTML fixtures on first run.
- Never hardcode a URL. Every URL comes from `data/sources.md` or chunk metadata.
- Commit after each phase (suggested messages provided).

### 0.4 Environment variables

`.env.example` currently lacks four settings this build needs. **Add these in Phase 1** and create
`.env` from it:

```bash
MIN_SIMILARITY=0.35        # score gate threshold (tune in Phase 3)
CHUNK_SIZE_TOKENS=200      # see §0.1
CHUNK_HARD_CAP_TOKENS=240
CHUNK_OVERLAP_TOKENS=25
REQUEST_DELAY_SECONDS=1.5  # politeness delay between page fetches
```

`README.md` already documents `cp .env.example .env`. Confirm `ANTHROPIC_API_KEY` is set before Phase 4.

---

## Phase 0 — Scaffolding (30 min)

**Goal:** a runnable skeleton with config, dataclasses, and an empty CLI, so every later phase has
something to plug into.

**Files**

| File | Action |
|---|---|
| `src/config.py` | create |
| `src/models.py` | create |
| `src/cli.py` | create (subcommands stubbed) |
| `src/ingest/__init__.py`, `src/rag/__init__.py` | already exist — leave |
| `.env.example` | modify |
| `.gitignore` | modify — add `cache/`, `chroma_db/`, `.env` |

**`src/config.py`**

```python
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Config:
    anthropic_api_key: str
    llm_model: str
    chroma_dir: str
    collection_name: str
    embed_model: str
    top_k: int
    min_similarity: float
    chunk_size_tokens: int
    chunk_hard_cap_tokens: int
    chunk_overlap_tokens: int
    request_delay_seconds: float
    sources_path: str
    manifest_path: str
    cache_dir: str
```

Load every field with a sensible default matching §0.4. Read `.env` once at import. Do **not** raise on a
missing `ANTHROPIC_API_KEY` at import time — the ingest phase must work without it. Raise in
`generator.py` instead, with a message telling the user to set it.

**`src/models.py`** — copy the dataclasses verbatim from architecture §4 (`Source`, `TextBlock`,
`Document`, `Chunk`, `Hit`, `Answer`). `Hit.score` is a float in `[0, 1]`; `Answer.hits` defaults to an
empty list. Add a `make_chunk_id(source_url, section, chunk_index) -> str` helper returning
`sha1(f"{source_url}|{section}|{chunk_index}").hexdigest()` — architecture **D9**, needed by Phase 2 and
the tests.

**`src/cli.py`** — `argparse` with three subcommands, each a one-line `TODO` stub that prints
"not implemented yet" and exits 0. Skeletons only:

```
python -m src.cli ingest [--force] [--refresh]
python -m src.cli ask "question"
python -m src.cli inspect          # dump manifest + collection stats
```

**Verify**

```bash
python -m src.cli --help
python -c "from src.config import CONFIG; print(CONFIG.chunk_size_tokens)"
```

**Exit criteria:** `python -m src.cli --help` lists three subcommands; `CONFIG` imports with no env file
present and prints `200`.

**Commit:** `Scaffold config, dataclasses, and CLI entry points`

---

## Phase 1 — Ingestion: Fetch + Extract (1.5–2 h)

**Goal:** `data/sources.md` → 5 cached HTML files → clean, structured `Document` objects on disk.
**This is the phase most likely to surface surprises — budget real time here.**

**Files**

| File | Action |
|---|---|
| `src/ingest/sources.py` | create |
| `src/ingest/fetcher.py` | create |
| `src/ingest/extractor.py` | create |
| `data/documents/<scheme_key>.json` | generated |
| `cache/raw_html/<scheme_key>.html` | generated |
| `tests/data/fixtures/` | create — save one real page as a fixture |
| `tests/test_extractor.py` | create |

### 1.1 `sources.py`

```python
def load_sources(path: str | Path = None) -> list[Source]:
    """Parse the markdown table in data/sources.md into Source records.
    Raise ValueError if a row is missing any of scheme_key/category/scheme_name/url
    or if the same scheme_key appears twice."""
```

Parse the pipe table by splitting on `|`, skipping the header and `---` separator rows, and stripping
whitespace. **Do not** add a markdown/table library. Validate: 5 sources, unique keys, `url` starts with
`https://`. Fail loudly — a silently empty source list is the worst possible ingest bug.

### 1.2 `fetcher.py`

```python
def fetch(url: str, scheme_key: str, *, refresh: bool = False,
          delay: float = None, retries: int = 3) -> str:
    """Return raw HTML for url. Caches to cache/raw_html/<scheme_key>.html.
    Reuses the cache unless refresh=True. Retries on 429/5xx with exponential backoff."""
```

- Realistic `User-Agent` header; a default `python-requests/x.y` gets blocked.
- **Cache-first** (architecture **D10**): if the cache file exists and `refresh=False`, return it
  without a request. This is what protects the live demo.
- Retry with backoff `2s, 4s, 8s` on 429/5xx and on `requests.RequestException`. Log every attempt.
- `time.sleep(delay)` between requests inside the *caller* (the pipeline loop), not here.
- **Write the fixture during this phase:** save one real page to `tests/data/fixtures/large_cap.html` so
  the extractor tests never touch the network.

### 1.3 `extractor.py`

The highest-risk module. Structure is a deliberate design choice (architecture §5.2) — keeping `section`
and `kind` is what makes heading-aware chunking and precise citations possible later.

```python
STRIP_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "aside", "form")

def extract(html: str, source: Source, ingested_at: str) -> Document:
    """Parse HTML into a Document of ordered TextBlocks. Never raises on malformed
    HTML -- bs4 is lenient. Raises only if zero blocks are produced."""

def _flatten_table(table) -> list[str]:
    """One line per <tr>, cells joined by ' | '. Never split a row across lines."""

def clean_text(raw: str) -> str:
    """Collapse whitespace, drop long digit runs (PII safety, architecture §10),
    strip zero-width characters."""
```

Requirements:

1. `soup = BeautifulSoup(html, "html.parser")`. Remove every `STRIP_TAGS` element, plus cookie/consent
   banners (match on class/id substrings: `cookie`, `consent`, `popup`, `modal`).
2. **Walk the DOM in document order**, tracking the current heading:
   - `h1`–`h6` → update `current_section` for all subsequent blocks.
   - `table` → emit one `TextBlock(kind="table")` per table, text = `_flatten_table` output.
   - `li` → `kind="list"`, one line per item, bullet preserved.
   - `p`, and any `div` with no block-level children → `kind="prose"`.
3. Drop blocks whose cleaned text is under ~40 characters (kills nav crumbs and "Loading…" stubs).
4. **Row-wise table flattening is mandatory**, not cosmetic: it keeps `Expense ratio | 0.65%` as adjacent
   text so a later chunk boundary cannot separate a label from its value.
5. `TextBlock.section` must **never be empty** — fall back to `"Overview"` (or the `scheme_name`) for
   content before the first heading. The chunker and citations both rely on this.
6. `ingested_at` is stamped **once per run** (UTC, ISO-8601) and passed down, so every chunk in a run
   shares one timestamp.

**Coverage report.** Also return (or log) which of these expected facts were found per scheme:
`expense ratio`, `exit load`, `minimum sip`, `lock-in`, `benchmark`, `riskometer`, `statement`. This is
the early-warning system for the PRD §11 risk that Groww renders fee data client-side. If
`expense ratio` is missing for all five schemes, **stop and raise it** — that is a scope decision for
the team (architecture open decision A5), not something to discover at demo time.

### 1.4 Wire it up

`src/cli.py ingest` for now: load sources → fetch each → extract → dump
`data/documents/<scheme_key>.json` (documents only; no chunking yet). Print a table: scheme, blocks,
characters, sections found.

**`tests/test_extractor.py`** — against `tests/data/fixtures/large_cap.html`:

- nav/footer/script text does **not** appear in any block
- a known fee figure survives extraction as `label | value` on one line
- every `TextBlock.section` is non-empty
- `kind="table"` blocks have one line per source row
- `clean_text` strips a 12-digit run
- extraction of a garbage string yields zero blocks, not an exception from BeautifulSoup

**Verify**

```bash
python -m src.cli ingest --refresh
ls cache/raw_html/            # expect 5 files
python -c "import json;d=json.load(open('data/documents/large_cap.json'));print(len(d['blocks']))"
pytest tests/test_extractor.py -v
```

**Exit criteria:** 5 HTML files cached; 5 documents with sensible block counts (a real scheme page
should yield roughly 40–120 blocks — if you get 2, extraction is broken); the coverage report is printed
and reviewed; extractor tests pass.

**Commit:** `Add ingestion: source parsing, cached fetcher, and DOM-aware extractor`

### 1.5 Phase 1 outcome — read before starting Phase 2

Phase 1 is complete. Actual results and the design changes they forced (all detailed in
`docs/architecture.md` §3a):

| | |
|---|---|
| Corpus produced | **55 blocks / ~10,500 chars** across 5 schemes (9–16 blocks each) |
| Components added | `ingest/structured.py` — not in the original plan |
| Tests | `tests/test_extractor.py`, 32 passing, no network |
| Open decisions closed | A1 (chunk params), A5 (factsheets not needed) |
| Open decision raised | **A6 — the statement-download question is unanswerable from these 5 pages** |

**What changed and why:**

1. **The fee figures are not in the HTML.** They live in the page's `__NEXT_DATA__` JSON payload.
   Prose-only extraction found *zero* "expense ratio" strings across all five pages. `structured.py`
   now reads that payload and emits exact `label | value` rows. `EXCLUDED_FIELDS` keeps
   `return_stats` / `nav` / `groww_rating` / `peerComparison` out of the corpus entirely, which
   enforces the PRD's no-performance-claims rule structurally.
2. **Naive extraction was ~96% noise** (76,000 chars → 10,500). The filters that fixed it, and why
   each exists, are tabulated in architecture §3a F2. Do not "simplify" them away.
3. **The 40-character block floor is correct.** It was tested at 30 chars, which admitted ~40 blocks
   per page of a carousel listing *other* HDFC schemes — a wrong-scheme citation waiting to happen.
4. **`lock-in` showing NO for four schemes is correct**, not a bug. Only ELSS has one.

**Consequences for Phase 2:** chunks will be small and highly structured (two fact tables plus a
handful of prose blocks per scheme). Expect a total chunk count in the **40–70** range, not the
several hundred a naive reading of "5 web pages" would suggest. A low chunk count is fine — it means
retrieval precision will be high — but it makes the `MIN_SIMILARITY` tuning in Phase 3 *more*
important, since there is less margin between a good and a bad hit.

> **Cursor prompt — Phase 1**
> Implement Phase 1 of docs/implementation.md. Create `src/ingest/sources.py`, `src/ingest/fetcher.py`,
> and `src/ingest/extractor.py` exactly per the contracts in that document. `src/config.py` and
> `src/models.py` already exist — import from them, do not redefine the dataclasses. Requirements that
> are not optional: (1) fetcher caches raw HTML to `cache/raw_html/<scheme_key>.html` and reuses the
> cache unless `--refresh`; (2) extractor strips script/style/nav/header/footer/aside plus
> cookie/consent elements, then walks the DOM in document order emitting TextBlocks with `section`,
> `kind`, and `text`; (3) tables are flattened one `<tr>` per line with cells joined by " | " so labels
> and values stay adjacent; (4) blocks under 40 chars are dropped; (5) `section` is never empty —
> default to "Overview"; (6) print a coverage report listing, per scheme, whether expense ratio, exit
> load, minimum SIP, lock-in, benchmark, riskometer, and statement were found. Then wire
> `python -m src.cli ingest` to run load → fetch → extract → write `data/documents/<scheme_key>.json`
> and print a per-scheme summary. Add `tests/test_extractor.py` using a saved HTML fixture — do not let
> any test hit the network. Do not add new dependencies. Run the tests and show me the output plus the
> coverage report.

---

## Phase 2 — Chunking, Embedding, Vector Store (2 h)

**Goal:** `Document` objects → `Chunk` list → 384-dim vectors → populated ChromaDB collection +
`data/index_manifest.json`.

**Files**

| File | Action |
|---|---|
| `src/ingest/chunker.py` | create |
| `src/ingest/embedder.py` | create |
| `src/ingest/store.py` | create |
| `src/ingest/pipeline.py` | create |
| `tests/test_chunker.py` | create |
| `data/index_manifest.json` | generated |
| PRD §6.2, architecture §5.3 | modify — corrected chunk params (§0.1) |

### 2.1 `chunker.py`

Decision **D4**: heading-aware first, recursive character split as fallback inside oversized sections.

```python
def chunk_document(doc: Document, cfg: Config, tokenizer) -> list[Chunk]:
    """Split one Document into Chunks. Never crosses a section boundary.
    Never splits a table row or list item. Returns [] if the document has no blocks."""

def _recursive_split(text: str, size: int, overlap: int) -> list[str]:
    """Hand-rolled recursive split (NO LangChain, see §0.2). Try separators in order:
    ["\\n\\n", "\\n", ". ", " ", ""] -- split on the first that occurs, recurse into
    pieces that are still too long, then re-join with overlap."""

def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))
```

Rules — each maps to a stated requirement:

1. **One chunk belongs to exactly one section.** Never merge across sections. This is what makes
   `chunk.section` trustworthy enough to appear in a citation.
2. **Preserve blocks.** Concatenate a section's blocks in order, then split. A `table` or `list` block is
   an atomic unit — if adding it would overflow the chunk, start a new chunk rather than bisecting it.
3. Size target 200 tokens, hard cap 240, overlap 25 (§0.1).
4. `_recursive_split` tries paragraph → line → sentence → word → hard character cut, in that order. The
   hard cut is last-resort; if it ever triggers, log a warning (it means a single table row exceeded 240
   tokens).
5. Guard against infinite recursion: if a piece does not shrink between iterations, fall back to a hard
   cut and log.
6. **Drop chunks under ~30 tokens.** A 12-token fragment ("Exit load") matches everything and pollutes
   top-k. Exception: a short `table` block that is a single row is kept and merged forward.
7. Populate **every** `Chunk` field from architecture §4, including `char_start` (offset into the
   section text) so a human can re-verify a chunk by hand — this is what makes acceptance test **A7**
   practical.
8. `chunk_index` is per-document and must be stable across runs for a given document (IDs must be
   deterministic per **D9**).

### 2.2 `embedder.py`

```python
class Embedder:
    def __init__(self, model_name: str): ...
    @property
    def tokenizer(self): ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batched (32). Returns 384-dim vectors."""
    def embed_query(self, text: str) -> list[float]:
        """Same model, same pipeline. Never load a second instance."""
```

- Load the `SentenceTransformer` **once**; `retriever` will import this same class (Phase 3) — any
  divergence in model or pooling silently destroys retrieval quality.
- **Embedding prefix:** embed `f"{scheme_name} — {section}\n{text}"`, store the **unprefixed** text.
  The prefix puts scheme and topic into the vector, which is what makes "lock-in?" land on the ELSS
  tax section instead of an arbitrary ELSS paragraph. The UI should show clean, unprefixed quotes.
- **Add an assertion in dev:** every returned vector has 384 dims. Log chunk count and mean/std token
  length so you can eyeball whether the 200-token target held.

### 2.3 `store.py`

All Chroma access lives here (decision **D12**).

```python
def get_client(cfg: Config): ...                 # PersistentClient at cfg.chroma_dir
def ensure_collection(cfg: Config, embed_model: str, force: bool = False): ...
def upsert_chunks(cfg: Config, chunks: list[Chunk], vectors: list[list[float]]) -> int: ...
def query(cfg: Config, vector: list[float], top_k: int, scheme_key: str | None = None) -> list[Hit]: ...
def reset(cfg: Config) -> None: ...
```

- `ensure_collection` writes the embedding model name into collection metadata. If the existing
  collection was built with a **different** `EMBED_MODEL`, raise unless `force=True`. This guard
  prevents the classic silent-failure mode where a changed model makes every query return noise.
- `upsert_chunks` uses `Chunk.id` (sha1, **D9**) so re-ingest **upserts in place** — no duplicates
  across refreshes.
- `query` converts `distances` to `score = 1 - distance`, clamped to `[0, 1]`, and returns `Hit` objects
  with metadata mapped back onto the dataclass.
- Support an optional `where={"scheme_key": ...}` filter for a future UI selector.

### 2.4 `pipeline.py` + manifest

```python
def run_ingest(*, force: bool = False, refresh: bool = False) -> dict:
    """Full offline pipeline. Returns and writes the manifest dict."""
```

Writes `data/index_manifest.json` in exactly the architecture §4 shape: `ingested_at`, `embed_model`,
`chunker` (`"section-aware-recursive"`), `chunk_size_tokens`, `chunk_overlap_pct`, `chunk_count`,
per-source `chunks` + `status`, and `failures[]`.

**A failed source must not abort the run** — record it in `failures[]`, continue, and make
`cli inspect` print the failures loudly.

**Update `cli ingest`** to call `run_ingest()` and print: total chunks, per-scheme counts, mean chunk
size in tokens, collection vector count, and any failures.

### 2.5 `tests/test_chunker.py`

- No chunk exceeds `CHUNK_HARD_CAP_TOKENS`.
- No chunk contains text from two different `section` values.
- A fee table's label and value always land in the same chunk (assert the exact string
  `"Expense ratio"` and its value co-occur in one chunk).
- Every `Chunk` has all metadata fields populated; `section` non-empty.
- Chunk IDs are deterministic: chunking the same `Document` twice yields identical IDs.
- `_recursive_split` on a 2000-token wall of text terminates and returns bounded pieces.

**Verify**

```bash
python -m src.cli ingest
python -m src.cli inspect                 # chunk count, per-scheme, failures
pytest tests/test_chunker.py -v
python -c "
from src.config import CONFIG
from chromadb import PersistentClient
c=PersistentClient(path=CONFIG.chroma_dir).get_collection(CONFIG.collection_name)
print('vectors:', c.count())"
```

Sanity-check the distribution yourself: mean chunk size near 200 tokens, no chunk pinned at the 240 cap.
If many chunks sit at the cap, sections are huge — reconsider the recursion separators.

**Also do now:** update PRD §6.2 and architecture §5.3 with the corrected 200/25 parameters and the
256-token reason (§0.1), and record the real chunk count in architecture §5.3.

**Exit criteria:** collection populated and `count()` matches `chunk_count` in the manifest; manifest
written; chunker tests pass; docs updated with real numbers.

### 2.6 Phase 2 outcome — verified

| Exit criterion | Result |
|---|---|
| Collection count == manifest `chunk_count` | **26 == 26** ✅ |
| Collection distance space | `cosine` ✅ (was defaulting to L2 — fixed) |
| Manifest written, `failures: []` | ✅ |
| Chunker + store tests | **70 passed** (32 Phase 1 + 38 new) ✅ |
| Docs updated with real numbers | PRD §6.2/§6.3, architecture §5.3/§5.5 ✅ |

Real distribution: **26 chunks, 47–153 wordpiece tokens, median 87, mean 91.7**, zero chunks at the
240 cap. The verification note above asks for a mean "near 200" — the real mean is 91.7, and that
check should be read as *"no chunk pinned at the cap, and the cap logic is exercised"*, not as a
literal target. Section size bounds packing on this corpus; see architecture §5.3.

**Three bugs the tests caught, all fixed:**
1. Overlap taken at wordpiece level flattened table rows in the next chunk.
2. A prose line longer than the overlap budget was taken whole → 397-token chunk, over the cap.
3. Collection was created on Chroma's default **L2** space, making `score = 1 - distance` not a
   cosine similarity. Now set explicitly and enforced.

**Carried into Phase 3:** `MIN_SIMILARITY = 0.35` is too high. The correct chunk for *"is there a lock
in period"* scores **0.166** while a vague question scores **0.845**; a hard negative scored 0.327.
Tune on the eval set and prefer relative gating over a single absolute cut.

**Commit:** `Add chunking, embedding, and ChromaDB indexing with manifest`

> **Cursor prompt — Phase 2**
> Implement Phase 2 of docs/implementation.md: `src/ingest/chunker.py`, `src/ingest/embedder.py`,
> `src/ingest/store.py`, `src/ingest/pipeline.py`. Non-negotiable details: (1) chunking is
> heading-aware — split within each `TextBlock.section`, never merge across sections, never split a
> table row or list item; (2) target 200 wordpiece tokens, hard cap 240, overlap 25, and count tokens
> with the SentenceTransformer tokenizer, NOT `len(text.split())`; (3) drop chunks under 30 tokens;
> (4) write your own recursive splitter over separators `["\n\n", "\n", ". ", " ", ""]` — do NOT add
> langchain or any new dependency, and guard against non-shrinking recursion; (5) embed
> `"{scheme_name} — {section}\n{text}"` but store the unprefixed text; (6) ChromaDB writes go only
> through `store.py`; store the embedding model name in collection metadata and refuse to write into a
> collection built with a different model unless `force=True`; (7) chunk IDs are
> `sha1(source_url|section|chunk_index)` so re-ingest upserts in place; (8) a single source failure
> goes into `failures[]` and does not abort the run; (9) write `data/index_manifest.json` in the exact
> shape given in architecture.md §4. Then add `tests/test_chunker.py` covering the five assertions in
> that document, wire `cli ingest` and `cli inspect`, and show me the chunk-count and mean-token-length
> output.

---

## Phase 3 — Retrieval + Score Gate (1–1.5 h)

**Goal:** a query returns relevant, above-threshold `Hit` objects — and the CLI can prove it.

**Files**

| File | Action |
|---|---|
| `src/rag/retriever.py` | create |
| `src/cli.py` | modify — real `ask` and `inspect` retrieval paths |
| `tests/test_retriever.py` | create (`integration` marker) |
| architecture §5.6 | modify — record the tuned threshold |

### 3.1 `retriever.py`

```python
class Retriever:
    def __init__(self, cfg: Config):
        # load Embedder once; open the Chroma collection; verify the collection's
        # stored embed_model matches cfg.embed_model, else raise with a re-ingest hint
    def search(self, query: str, top_k: int = None, scheme_key: str | None = None) -> list[Hit]:
        """Return only hits with score >= cfg.min_similarity, sorted descending.
        Return [] when nothing clears the threshold -- an empty list is a valid
        and meaningful result, not an error."""
```

- **Import `Embedder` from `src.ingest.embedder`** — do not instantiate a second `SentenceTransformer`.
  Two instances with different pooling/normalisation is a silent retrieval bug.
- Query text is embedded **without** the document prefix (the prefix describes the chunk, not the
  question).
- Filter on score **inside** `search`, so callers can't accidentally bypass the gate. The gate is the
  primary hallucination defence (architecture **D5**).

### 3.2 Tune `MIN_SIMILARITY` — do not skip this

Run the retrieval-only path over a fixed question set and record the top-1 score for each. Log every
question with its top hit's scheme, section, and score.

**Must retrieve the right section (positive set):**

| Query | Expected top section |
|---|---|
| expense ratio of HDFC Large Cap Fund | Fees / charges |
| exit load on HDFC Flexi Cap Fund | Exit load |
| minimum SIP amount | Investments / SIP |
| lock-in period HDFC ELSS | Tax / lock-in |
| benchmark of HDFC Balanced Advantage | Fund facts / benchmark |
| riskometer level HDFC Small Cap | Riskometer |
| how to download capital gains statement | Statements / tax docs |

**Must retrieve nothing (negative set):**

- What is HDFC Bank's FD interest rate?
- Which fund is best for my portfolio?
- What is the weather in Mumbai?
- Tell me about Bitcoin.
- What is the price of gold today?

The threshold goes **between** the lowest positive top-1 score and the highest negative top-1 score. Set
`MIN_SIMILARITY` in `.env` to that midpoint, run the sets again to confirm, and record the numbers plus
the final value in architecture §5.6 and §12 (A2). `0.35` is a starting guess, not an answer — if the
sets do not separate, add the lexical signal from §3.4 rather than lowering the bar until A6 passes.

### 3.3 CLI

`python -m src.cli ask "..."` in this phase prints **retrieval only** (rank, score, scheme, section,
text) — no LLM yet, so you can debug retrieval without an API key. Add an `inspect` subcommand that
prints the manifest, the collection vector count, and `failures[]`.

### 3.4 If recall is poor

Before considering a reranker, try the cheap fix: for questions naming a scheme, append the matching
`scheme_key` to the query text before embedding (architecture §11, step 3). Log whether it helped. Only
escalate to a cross-encoder reranker if A2–A4 still miss.

**Verify**

```bash
python -m src.cli ask "exit load on HDFC Flexi Cap Fund"
python -m src.cli ask "What is HDFC Bank's FD interest rate?"   # expect: no hits above threshold
pytest tests/test_retriever.py -v -m integration
```

**Exit criteria:** all 7 positive queries return the expected top section; all 5 negative queries return
`[]`; `MIN_SIMILARITY` written to `.env` and documented in architecture §5.6/§12.

### 3.5 Phase 3 outcome — the single-threshold plan did not survive contact

Measured, not assumed. See architecture §5.6 for the full tables.

| Exit criterion | Result |
|---|---|
| Positive queries return the expected top section | **6/6** at rank 1 (7th is A6, correctly refused) |
| Negative queries return `[]` | **5/5** plus the A6 statement query = 6/6 |
| Threshold recorded in §5.6 and §12 | ✅ |
| Tests | 110 pass, 1 skipped (A6), 24 integration-marked |

**The plan said "put the threshold between the lowest positive and the highest negative score". That is
impossible here.** "What is HDFC Bank's FD interest rate?" scores **0.561**; "minimum SIP amount", which
we can answer, scores **0.445**. Cosine gap: **−0.116**. The FD question matches every chunk on the
token `hdfc` alone and treats "interest rate" as near "expense ratio".

Per §3.2's own instruction ("if the sets do not separate, add the lexical signal from §3.4 rather than
lowering the bar"), the fix is a second required gate — IDF-weighted lexical coverage:

| Signal | Answerable min | Refuse max | Gap |
|---|---|---|---|
| cosine | 0.445 | 0.561 | −0.116 fails |
| lexical coverage | 0.637 | 0.494 | **+0.143 works** |

Tuned: `MIN_SIMILARITY=0.30` (loose floor) **and** `MIN_LEXICAL_COVERAGE=0.55`.

Two further findings:

- **Ranking by `cosine × coverage` fixed a top-1 miss** (5/6 → 6/6). The §5.3 scheme prefix makes every
  chunk of a fund match its name equally, so cosine cannot tell a fund's fees chunk from its benchmark
  chunk; coverage can. `Hit.score` stays raw cosine.
- **§3.4's scheme-key append did not help** (≤0.03 change, and neither colliding negative names a
  scheme). Implemented and measured, then left off by default. No reranker warranted.

Added beyond the planned file list: `src/rag/lexical.py` (the coverage scorer) and
`cli ask --explain`, which prints both signals and the verdict per candidate.

**Commit:** `Add retrieval with score gating and tune MIN_SIMILARITY`

> **Cursor prompt — Phase 3**
> Implement Phase 3 of docs/implementation.md. Create `src/rag/retriever.py` with a `Retriever` class
> that imports `Embedder` from `src.ingest.embedder` (never instantiate a second
> `SentenceTransformer`), verifies the collection's stored embedding model matches config, and raises a
> clear re-ingest hint on mismatch. `search()` must filter out every hit scoring below
> `cfg.min_similarity` and return `[]` rather than raising when nothing clears the bar — an empty list
> is a valid result. Embed the raw query with no document prefix. Wire `cli ask` to print retrieval
> results only (rank, score, scheme, section, text) with no LLM call, and finish `cli inspect` to dump
> the manifest, vector count, and failures. Then add `tests/test_retriever.py` marked `@pytest.mark.integration`
> containing the 7 positive queries and 5 negative queries listed in that document, asserting the
> positive set retrieves the expected section and the negative set returns `[]`. Run both sets, show me
> the top-1 score for every question, and tell me what MIN_SIMILARITY value separates them — I will
> confirm it before we set it in `.env`.

---

## Phase 4 — Guards, Prompts, Generation, Citations (2–2.5 h)

**Goal:** a full `answer(query)` path. Acceptance tests A2–A6 should pass from the terminal.

**Files**

| File | Action |
|---|---|
| `src/rag/guards.py` | create |
| `src/rag/prompts.py` | create |
| `src/rag/generator.py` | create |
| `src/rag/citations.py` | create |
| `src/rag/chain.py` | create |
| `tests/test_guards.py` | create |
| `tests/test_citations.py` | create |
| `tests/test_acceptance.py` | create (`llm` marker) |
| architecture §12 (A3) | modify — record the `EDU_LINK` decision |

### 4.1 `guards.py` — deterministic, no LLM (decision **D6**)

A prompt instruction is not enforcement. These run in plain Python and must be unit-testable.

```python
PII_PATTERNS: dict[str, re.Pattern]   # pan, aadhaar, account, otp, email, phone

def detect_pii(text: str) -> str | None:
    """Return the PII kind ('pan', 'email', 'phone', 'otp', 'aadhaar', 'account')
    or None. Check PAN and email first -- they are the most specific."""

def is_advice(question: str) -> bool:
    """True for buy/sell/hold/switch/suit-my-portfolio/best-fund/should-I/timing questions."""
```

- Patterns, in order: PAN `[A-Z]{5}[0-9]{4}[A-Z]`, Aadhaar `\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b`,
  email, 10-digit Indian mobile `\b[6-9]\d{9}\b`, 12-digit account `\b\d{12}\b`,
  4–6 digit OTP **only** when adjacent to OTP-ish context words.
- The bare-4–6-digit OTP pattern is too aggressive on its own — "exit load 1%" must not trip it.
- `is_advice` is a keyword/phrase list: `should i`, `which is best`, `advise`, `recommend`, `buy`,
  `sell`, `hold`, `switch`, `suitable for me`, `my portfolio`, `good time to`.
- Add a test asserting **"expense ratio of HDFC Large Cap Fund"** and **"exit load"** are *not* advice —
  false positives here are the most damaging bug in this phase.

### 4.2 `prompts.py`

Copy the 7-rule `SYSTEM_PROMPT` verbatim from architecture §5.7. Also define here, as module constants:
`DISCLAIMER`, `ADVICE_REFUSAL`, `NO_MATCH`, `PII_REFUSAL`, `EDU_LINK`.

```python
def build_user_prompt(question: str, hits: list[Hit]) -> str:
    """Render CONTEXT (rank, scheme, section, text per hit) and a separate
    CONTEXT CITATIONS block listing only the retrieved URLs. Keep the total under
    ~1500 tokens."""
```

The **separate citations block is deliberate** — the model must choose from a supplied list rather than
recall a URL. Render `[1] HDFC Flexi Cap Fund — Exit load` headers so the model can reference which hit
it used.

Decide `EDU_LINK` now (architecture **A3**): SEBI's investor charter page or HDFC MF's education page.
Set it as a constant; the demo needs it fixed.

### 4.3 `generator.py`

```python
class Generator:
    def __init__(self, cfg: Config):
        # raise a clear error if cfg.anthropic_api_key is empty
    def answer(self, question: str, hits: list[Hit]) -> str:
        """One API call. temperature=0, max_tokens=300. Raise on API error --
        chain.py owns the user-facing fallback."""
```

One call per question, no retry loop — a retry inside a live demo reads as a hang. Timeout the client
(~30 s) so a stuck call doesn't freeze the UI.

### 4.4 `citations.py` — enforces G1 as an invariant

```python
URL_RE = re.compile(r"https?://[^\s\)\]\"'<>]+")

def validate(answer_text: str, hits: list[Hit]) -> tuple[str, str | None, bool]:
    """Return (cleaned_text, citation_url, truncated). Repair or reject per
    architecture §5.8. Never return uncited prose for a non-refusal answer."""
```

Logic, in this exact order:

1. Collect URLs via `URL_RE`.
2. If a URL is **not in the retrieved set** → drop it, and remember the top hit's `source_url` as the
   replacement. Log `citation_mismatch` with the offending URL. This catches the near-miss link that
   looks plausible to a human reviewer.
3. If the answer is **not a refusal** and no valid URL remains → return `NO_MATCH`. Uncited prose must
   never reach the user; that is what makes G1 measurable.
4. Enforce ≤3 sentences: split on `(?<=[.!?])\s+`, keep the first 3, set `truncated=True` if dropped.
5. Return `(text, citation_url, truncated)`.

### 4.5 `chain.py` — the single online entrypoint

```python
def answer(query: str, cfg: Config = None) -> Answer:
    """The one path used by both the CLI and the Streamlit UI (decision D11)."""
```

Exactly this order (architecture §5.9):

```
1. guards.detect_pii(query)  -> Answer(refused, "pii")     # never log the raw string
2. guards.is_advice(query)  -> Answer(refused, "advice")
3. retriever.search(query)   -> []  -> Answer(refused, "no_match")
4. generator.answer(...)     -> on exception: Answer(NO_MATCH) + log, never a traceback
5. citations.validate(...)   -> repair URL, enforce 1 link + <=3 sentences
6. return Answer(text, citation_url, manifest.ingested_at, hits=hits)
```

Steps 1–3 are free and offline, so most bad queries never reach the API. `Answer.hits` is populated even
on refusals-after-retrieval so the debug panel still works. `last_updated` comes from the **manifest**, not
the model, so it is consistent across all answers.

### 4.6 Tests

**`test_guards.py`** — PAN, Aadhaar, email, phone, account, OTP samples all detected; the six factual
questions from PRD §4 are *not* flagged as PII; the three advice questions are flagged; the seven
positive queries are *not* flagged as advice.

**`test_citations.py`** — a fabricated URL is replaced with the top hit's URL; a 5-sentence answer is
truncated to 3 with `truncated=True`; a no-URL answer becomes `NO_MATCH`; a correct answer passes
through byte-identical.

**`test_acceptance.py`** — `@pytest.mark.llm`, real API. Cover PRD §8: **A2** expense ratio (≤3
sentences, exactly 1 link), **A3** ELSS lock-in, **A4** minimum SIP, **A5** advice refusal (no numbers,
educational link present), **A6** out-of-corpus → "not found", **A10** determinism (same question twice
→ same citation).

**Verify**

```bash
pytest tests/test_guards.py tests/test_citations.py -v          # no API key needed
export ANTHROPIC_API_KEY=...
pytest tests/test_acceptance.py -v -m llm
python -m src.cli ask "expense ratio of HDFC Large Cap Fund?"
python -m src.cli ask "Should I buy HDFC Small Cap Fund?"       # expect refusal
python -m src.cli ask "What is HDFC Bank's FD interest rate?"   # expect not-found
```

For `cli ask`, print the retrieved hits' scores under the answer so you can see the gate working.

**Exit criteria:** A2–A6 and A10 pass; guards and citations tests pass without an API key.

**Commit:** `Add guards, prompt layer, Claude generation, and citation validation`

> **Cursor prompt — Phase 4**
> Implement Phase 4 of docs/implementation.md: `src/rag/guards.py`, `prompts.py`, `generator.py`,
> `citations.py`, `chain.py`. Hard requirements: (1) `guards.detect_pii` and `guards.is_advice` are pure
> regex/keyword Python with no LLM call, and the raw query must never be logged when PII is detected;
> (2) copy the 7-rule SYSTEM_PROMPT verbatim from architecture.md §5.7 and define DISCLAIMER,
> ADVICE_REFUSAL, NO_MATCH, PII_REFUSAL, EDU_LINK as module constants; (3) `build_user_prompt` renders a
> numbered CONTEXT block and a SEPARATE CONTEXT CITATIONS block containing only the retrieved URLs;
> (4) the Anthropic call uses temperature=0, max_tokens=300, one call per question, no retry loop, and a
> ~30s timeout; (5) `citations.validate` replaces any URL not in the retrieved set with the top hit's
> canonical source_url and logs `citation_mismatch`, converts a no-URL non-refusal answer into
> NO_MATCH, and truncates to 3 sentences on a sentence boundary; (6) `chain.answer()` runs the six steps
> in the documented order and NEVER lets an exception reach the UI — catch API errors and return
> NO_MATCH; (7) `last_updated` is read from the manifest, never from the model. Add the three test files
> described there, with the guard and citation tests requiring no API key and the acceptance tests
> marked `@pytest.mark.llm`. Run the non-LLM tests first and show me the output, then run the LLM
> acceptance tests and show me the actual answers.

---

## Phase 5 — Streamlit UI (1.5 h)

**Goal:** the demo surface. This is what the class sees.

**Files**

| File | Action |
|---|---|
| `src/app.py` | create |
| `README.md` | modify — replace the "Added in Phases 3–5" placeholder |

**Layout, top to bottom:**

1. `st.set_page_config(page_title="HDFC MF Facts Assistant", layout="centered")`.
2. **Welcome line** — "HDFC Mutual Fund facts-only assistant — 5 schemes, public pages only."
3. **Disclaimer**, from the `prompts.DISCLAIMER` constant (import it; never retype it). Render it in the
   main area, the sidebar, and the footer so it is unmissable — this is acceptance test **A9**.
4. **3 example questions** as `st.button` chips from the PRD §4 set, wired to prefill/send the question.
5. **Chat area** via `st.session_state["messages"]` — user/bubbles; each assistant turn shows the answer,
   the citation as a clickable link, and `Last updated from sources: {manifest.ingested_at}`.
6. **`st.expander("🔍 Retrieved chunks (debug)")`** — per hit: rank, score, scheme, section, and the raw
   unprefixed chunk text. The single most valuable thing to show in a demo: it makes retrieval visible.
7. **Sidebar** — corpus scope (5 schemes + their links, read from `data/sources.md` at runtime),
   `ingested_at`, embedding model, `TOP_K`, `MIN_SIMILARITY`, and a "Clear chat" button.
8. **Empty-index guard** — on first run, if the collection is missing or empty, show
   `Run: python -m src.cli ingest` instead of a broken chat box.

Constraints:

- `src/app.py` contains **zero business logic** — it calls `chain.answer()` and renders. If you find
  yourself writing an `if` about expense ratios in this file, it belongs in `chain.py`.
- Chat text lives in `st.session_state` only. Nothing is written to disk — this is the **A8** privacy
  story you can point at in the demo.
- Cache the `Retriever`/`Generator` objects in `st.session_state` (or `@st.cache_resource`) so the
  `SentenceTransformer` isn't reloaded on every keystroke. First load takes a few seconds.
- Never print a raw exception — use `st.info`/`st.warning` with the friendly message.

**Verify**

```bash
streamlit run src/app.py
```

Click through: the 3 example buttons each produce an answer with a working link; an advice question
refuses; an out-of-corpus question says not-found; the debug expander shows real chunks and scores;
the disclaimer is visible without scrolling.

**Update `README.md`:** fill in the Usage section with `cli ingest` → `streamlit run src/app.py`, the
demo script, and the tuned `MIN_SIMILARITY`. Leave the "Sample Q&A" and "Known limits" placeholders for
Phase 6.

**Exit criteria:** A1 and A9 pass; all three example buttons work; a debug panel shows scores.

**Commit:** `Add Streamlit UI with citations, debug panel, and disclaimer`

> **Cursor prompt — Phase 5**
> Create `src/app.py` implementing Phase 5 of docs/implementation.md. Requirements: zero business
> logic — it must call `chain.answer()` and render the result; welcome line; the DISCLAIMER constant
> imported from `src.rag.prompts` (never retyped) shown in the main area, sidebar, and footer; 3
> clickable example-question buttons from PRD §4; chat via `st.session_state` with the citation rendered
> as a clickable link and `Last updated from sources: {manifest.ingested_at}`; a `st.expander` debug
> panel showing each retrieved hit's rank, score, scheme, section, and unprefixed text; a sidebar showing
> the 5 schemes read from `data/sources.md` plus ingested_at, embed model, TOP_K, and MIN_SIMILARITY; a
> clear-chat button; and an empty-index guard that tells the user to run `python -m src.cli ingest`
> instead of rendering a broken chat. Cache the Retriever/Generator so the SentenceTransformer is not
> reloaded per keystroke. Never let a raw exception reach the UI. Then update README.md's Usage section
> with the real commands.

---

## Phase 6 — Validation, Docs, Demo (1.5 h)

**Goal:** every acceptance test green, every deliverable present, demo recorded.

### 6.1 Full acceptance sweep

Run the complete PRD §8 table and record actual vs. expected in a new
`docs/acceptance_results.md`. Do not mark a row passed on intent — open the URL and confirm (this is
**A7**, and it is the single highest-value check in the project: a right number with a broken citation is
a failed demo).

- [ ] **A1** clean clone → `cli ingest` → `cli ask` → UI answers
- [ ] **A2** expense ratio: ≤3 sentences, exactly 1 link, figure matches the page
- [ ] **A3** ELSS lock-in correct + 1 link
- [ ] **A4** minimum SIP amount + 1 link
- [ ] **A5** "Should I buy HDFC Small Cap?" → refusal, no numbers, educational link
- [ ] **A6** "HDFC Bank's FD rate?" → not found, no fabrication
- [ ] **A7** every sample answer re-verified by opening the cited URL
- [ ] **A8** type a PAN and an email → refused, nothing stored
- [ ] **A9** disclaimer visible on first render
- [ ] **A10** same question twice → same answer and same citation

**Fresh-clone rehearsal:** `git clone` the repo to a temp dir and follow `README.md` exactly. Every
failure here is a demo failure. Fix the README, not your memory.

### 6.2 Deliverables (PRD §9)

| Deliverable | File | Status |
|---|---|---|
| Working prototype | `src/app.py` + `cli.py` | done in Phases 0–5 |
| Source list | `data/sources.md` | exists — verify all 5 URLs still resolve |
| README | `README.md` | setup, scope, known limits |
| Sample Q&A (5–10) | `data/sample_qa.md` | **create** |
| Disclaimer snippet | `data/disclaimer.md` | **create** |
| Architecture + implementation | `docs/architecture.md`, `docs/implementation.md` | exists — close out §12 |
| Demo video ≤3 min | — | record |

**`data/sample_qa.md`** — a markdown table: `Query | Assistant answer | Source link | Verified (date)`.
Include 3 factual, 1 advice-refusal, 1 out-of-corpus. Every row's link must be one you personally
opened.

**`data/disclaimer.md`** — the exact `DISCLAIMER` string plus a longer version for the README, so the
UI text and the submitted text cannot drift apart.

### 6.3 Close out the open decisions

Update architecture §12 — every row should now have an answer:

- **A1** real chunk params + chunk count (§0.1 correction recorded)
- **A2** tuned `MIN_SIMILARITY` with the positive/negative score spread
- **A3** `EDU_LINK` chosen
- **A4** committed `chroma_db/` vs. one-command ingest (recommend **not** committing binaries; document
  the one-liner and pre-ingest before the demo)
- **A5** factsheet URLs added to `sources.md`? — decide from the Phase 1 coverage report

Add a **Known limits** section to `README.md` that is honest: corpus is 5 pages, figures can be stale
after `ingested_at`, all-MiniLM is not finance-tuned, HTML parsing is brittle, answers are short by
design and may miss nuance, no advice is given by design.

### 6.4 Demo video (≤3 min)

Storyboard — this is the whole grade in three minutes:

| Time | Show |
|---|---|
| 0:00–0:20 | Problem + scope: "5 HDFC schemes, public pages only, facts-only" |
| 0:20–0:45 | The pipeline, using architecture §1's diagram |
| 0:45–1:20 | Terminal: `cli ingest` → chunk count → `cli ask` with retrieval output visible |
| 1:20–2:10 | UI: example question → answer + link; open the link to prove it |
| 2:10–2:35 | **The money shot:** open the debug expander — show retrieved chunks and scores |
| 2:35–2:50 | Advice question refused; out-of-corpus question says not found |
| 2:50–3:00 | Disclaimer + known limits |

Rehearse once with the network off to prove the cache-first design (decision **D10**) holds up.

**Commit:** `Complete Phase 6: acceptance sweep, sample Q&A, docs, and demo script`

> **Cursor prompt — Phase 6**
> Implement Phase 6 of docs/implementation.md. (1) Create `data/sample_qa.md` with a table of 5
> queries — 3 factual, 1 advice refusal, 1 out-of-corpus — columns `Query | Assistant answer | Source
> link`, and create `data/disclaimer.md` holding the exact DISCLAIMER constant plus a longer README
> version. (2) Fill in README.md: Usage with the real commands, Scope, and a Known limits section
> covering the 5-page corpus, staleness after ingested_at, all-MiniLM not being finance-tuned, brittle
> HTML parsing, and the by-design no-advice rule. (3) Create `docs/acceptance_results.md` as a table
> with a row per test A1–A10 from PRD §8, columns `Test | Command | Expected | Actual | Pass/Fail` —
> leave `Actual` empty for me to fill after I verify each one by hand. (4) Update architecture.md §12 so
> every open decision row has its recorded answer. (5) Add a `docs/demo_script.md` with the timed
> storyboard from Phase 6.4. Do not mark any acceptance test as passed — I verify those myself.

---

## Appendix A — Phase Dependency Graph

```
Phase 0  config, models, cli          ─┐
                                       ├─► Phase 1  sources, fetcher, extractor ─┐
                                       │   (needs CONFIG)                        │
                                       └─────────────────────────────────────────┤
                                                                                   ▼
                                                                          Phase 2  chunker,
                                                                          embedder, store,
                                                                          pipeline → index
                                                                                   │
                             ┌─────────────────────────────────────────────────────┤
                             ▼                                                     ▼
                       Phase 3  retriever                                   Phase 4  guards,
                       (+ tune MIN_SIMILARITY)                              prompts, generator,
                       needs the index                                     citations, chain
                             │                                                     │
                             └──────────────────────┬──────────────────────────────┘
                                                    ▼
                                              Phase 5  Streamlit UI
                                                    │
                                                    ▼
                                              Phase 6  validation, docs, demo
```

**Critical path:** 1 → 2 → 4 → 5 → 6. Phase 3 can run in parallel with Phase 4 once Phase 2 lands,
but `chain.py` needs both, so do 3 first if you have two people.

## Appendix B — Environment Matrix

| Phase | `ANTHROPIC_API_KEY` | Built index | Network |
|---|---|---|---|
| 0 | no | no | no |
| 1 | no | no | **yes** (first run only; cached after) |
| 2 | no | no | no (reuses cache) |
| 3 | no | **yes** | no |
| 4 | **yes** | yes | yes (Anthropic API) |
| 5 | yes | yes | yes (Anthropic API) |
| 6 | yes | yes | yes |

Phases 0–3 need no API key and no paid calls. Budget the key for Phase 4 onward.

## Appendix C — Fast Test Commands

```bash
# no API key, no index — run this before every commit
pytest -m "not llm and not integration" -v

# retrieval only (needs index)
pytest -m integration -v

# full acceptance sweep (needs index + key)
pytest -m llm -v

# everything
pytest -v
```

## Appendix D — Definition of Done

The project is done when **all** of these hold:

- [ ] `python -m src.cli ingest` builds the index from a clean clone, offline, from cache
- [ ] A1–A10 pass with results recorded in `docs/acceptance_results.md`
- [ ] Every sample answer's citation URL was opened and confirmed by a human
- [ ] A refused advice question contains zero numbers and one educational link
- [ ] An out-of-corpus question produces "not found", never a guess
- [ ] The UI shows retrieved chunks and similarity scores
- [ ] No PII typed in the chat is logged, stored, or echoed
- [ ] All PRD §9 deliverables exist
- [ ] `docs/architecture.md` §12 has no unresolved rows
- [ ] The 3-minute demo video is recorded and rehearsed

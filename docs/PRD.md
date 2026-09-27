# PRD — Mutual Fund Facts-Only RAG Chatbot

**Status:** Draft for class demo · **Owner:** RAG_chatbot team · **Last updated:** 2026-09-27

---

## 1. Summary

Build a small **facts-only FAQ chatbot** for **HDFC Mutual Fund** schemes. It answers factual questions
(expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, how to download a statement)
using **only official public pages**, and shows **one source link in every answer**. It never gives
investment advice and never computes or compares returns.

Deliberately scoped small: **1 AMC, 5 schemes, 5 public URLs.** This is a class demo — the goal is a
clean, explainable RAG pipeline over a tightly-scoped corpus, not a production support bot.

---

## 2. Problem

Retail investors and support/content teams repeatedly ask the same factual questions about mutual fund
schemes. The answers already exist on public pages, but they are scattered across five scheme pages, are
buried in long HTML pages, and users often land on marketing content that mixes facts with opinions.

**Pain points**
- Repeating the same factual questions to support agents.
- Facts are hard to locate: expense ratio and exit load live in tables, lock-in rules in prose, benchmark
  and riskometer in a separate section.
- Generic AI answers can hallucinate fee figures or drift into advice, which is unacceptable for financial
  information.

**Why RAG.** The answer must be grounded in a specific retrieved passage and traceable to a specific
page. Retrieval-augmented generation gives us both grounding and citations; a fine-tuned or
prompt-only model gives us neither.

---

## 3. Goals / Non-Goals

### Goals
| # | Goal | Measure |
|---|---|---|
| G1 | Answer factual queries with ≤3 sentences and one citation link | 100% of factual answers carry exactly one link |
| G2 | Ground every answer in retrieved source text | No answer generated when retrieval is empty → refuse |
| G3 | Refuse opinionated/portfolio questions politely | Refusal works on a curated advice-question test set |
| G4 | Demonstrate every RAG stage end-to-end | Loading → Chunking → Embedding → Vector store → Retrieval → Answer, each inspectable |
| G5 | Ship a tiny UI: welcome line, 3 example questions, disclaimer | Streamlit app launches and answers a question end-to-end |

### Non-Goals
- No investment advice, recommendations, or "should I buy/sell" answers.
- No returns computation, ranking, or performance comparison.
- No PII collection or storage (PAN, Aadhaar, account numbers, OTPs, email, phone).
- No multi-AMC support, no live NAV, no login/auth, no user accounts.
- No hosted production deployment (local run + optional ≤3-min demo video).

---

## 4. Users & Use Cases

**Primary:** retail user comparing HDFC schemes who wants a quick factual answer.
**Secondary:** support/content team answering repetitive MF questions from the same corpus.

**Representative queries**
1. "What is the expense ratio of the HDFC Large Cap Fund (Direct – Growth)?"
2. "Is there an exit load on HDFC Flexi Cap Fund?"
3. "What is the minimum SIP amount?"
4. "What is the lock-in period for HDFC ELSS Tax Saver Fund?"
5. "What is the benchmark and riskometer level for HDFC Balanced Advantage Fund?"
6. "How do I download my capital gains statement?"

**Out of scope queries (must refuse):**
- "Should I buy HDFC Small Cap Fund?"
- "Which of these funds is best for my portfolio?"
- "Is now a good time to exit?"

---

## 5. Scope of the Corpus

**AMC:** HDFC Mutual Fund. **All plans:** Direct – Growth.

| scheme_key | Category | Scheme | URL |
|---|---|---|---|
| `large_cap` | Large Cap | HDFC Large Cap Fund | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| `flexi_cap` | Flexi Cap | HDFC Flexi Cap Fund | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| `elss` | ELSS | HDFC ELSS Tax Saver Fund | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| `small_cap` | Small Cap | HDFC Small Cap Fund | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| `balanced_advantage` | Hybrid | HDFC Balanced Advantage Fund | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

Canonical list lives in [`data/sources.md`](../data/sources.md) and is the single source of truth for the
ingestion step. No third-party blogs, no screenshots of app back-ends.

> **Phase 1 findings** (details in [`architecture.md`](architecture.md) §3a). Two changes to how this
> corpus is read:
>
> 1. The fee/limit figures are **not in the page HTML** — they sit in the page's embedded
>    `__NEXT_DATA__` JSON. Ingestion reads that payload for exact values and uses the HTML only for
>    prose (tax treatment, manager bio, objective). Returns, NAV, ratings and fund comparisons are
>    excluded at the source, which is how the "no performance claims" rule is enforced.
> 2. **"How do I download my capital gains statement?" is not answerable from these five pages** —
>    verified absent from all five. Either add a help-centre/registrar URL, or drop the question and
>    demonstrate the "not in my sources" refusal instead. Tracked as scope decision F3.

---

## 6. System Design

### 6.1 Pipeline (each stage must be inspectable)

```
URLs (data/sources.md)
  → 1. LOADING      fetch + parse HTML → clean text (BeautifulSoup, nav/footer/script stripped)
  → 2. CHUNKING     split into overlapping, semantically-bounded chunks
  → 3. EMBEDDING    sentence-transformers/all-MiniLM-L6-v2  (384-dim)
  → 4. VECTOR STORE ChromaDB, persistent, collection hdfc_mf_faq
  → 5. RETRIEVAL    similarity search, top_k = 4, with score threshold
  → 6. ANSWER       Claude composes ≤3 sentences + 1 citation link
```

Metadata carried on every chunk: `source_url`, `scheme_key`, `scheme_name`, `category`, `chunk_index`,
`section` (heading the text came under), `ingested_at`.

### 6.2 Chunking strategy (decision required)

Problem statement says the strategy is chosen **after inspecting the real data**. Working approach:

1. Ingest and eyeball the parsed text: measure length, heading structure, table density.
2. Choose between:
   - **Recursive character split** — safe default, good for long flat pages.
   - **Semantic split** — groups sentences by embedding similarity; better where paragraphs mix
     topics (e.g. a page section covering both fees and tax rules).
   - **Heading-aware split** — split on page sections first, then recurse within a section. Preferred
     when the page has a clean section skeleton, because it keeps "Fees" and "Taxes" chunks pure.
3. Record the final choice and rationale in [`docs/architecture.md`](architecture.md).

> **Settled at end of Phase 2: heading-aware split, then recursive within the section** (option 3,
> not the recursive-character default). The HDFC pages carry a clean section skeleton, and keeping
> "Fees" and "Tax implication" chunks separate is what makes citations exact. Semantic split was
> rejected: with 6–8 short blocks per page there is no paragraph-level topic drift to detect, and it
> would have made chunk boundaries non-deterministic, breaking idempotent re-ingest.

> **Corrected after Phase 1 (see `docs/implementation.md` §0.1).** Chunk size is **200 tokens,
> hard cap 240, overlap 25** — not the 300–600 originally specified here. The chosen embedding
> model, `all-MiniLM-L6-v2`, truncates at **256 wordpiece tokens**, so a longer chunk has its tail
> silently dropped at embedding time and any fact in that tail becomes unretrievable. Count with
> the model tokenizer, not `len(text.split())`.

> **Measured after Phase 2.** The five real pages yield **26 chunks**, 47–153 wordpiece tokens
> (median 87, mean 91.7), all under the 240 cap. Chunks land well below the 200 target because the
> source blocks are small — a fees table is a whole section, not a paragraph. That is the honest
> result, not a tuning failure: with a 26-chunk index, `top_k = 4` already returns a quarter of the
> corpus, so precision comes from the retriever and the answer prompt, not from chunk size.

### 6.3 Retrieval

- Embed the query with the **same** model used at index time.
- ChromaDB `query(query_texts=[q], n_results=TOP_K=4)`.
- Apply a **minimum score threshold**; if nothing clears it, the app must say it does not know rather
  than guess. This is the main defence against hallucination.
- Always keep `source_url` + `section` for the retrieved chunks so citations are exact.

> **Threshold warning, measured at end of Phase 2.** The placeholder `MIN_SIMILARITY = 0.35` is
> **too high and must not be shipped as-is.** A six-query probe against the real index returned the
> correct chunk at rank 1 every time, but with a wide score spread: a specific question
> ("expense ratio of HDFC Large Cap") scored **0.845**, while the equally well-answered
> "is there a lock in period" scored only **0.166** and would have been rejected by a 0.35 gate.
> Short, vague questions match a dense multi-row fees table weakly no matter how good the chunk is.
>
> Phase 3 must therefore tune this on the eval set rather than assume 0.35, and should prefer
> **relative** gating (best-score relative to the rest of the top-k, plus a floor) over a single
> absolute cut. Hard negative — "can I get the tax statement" — scored 0.327 against tax sections
> that do *not* answer it, so an absolute threshold alone cannot separate answerable from
> unanswerable here.

### 6.4 Answer generation

- **Model:** Groq-hosted open-weight model (`LLM_MODEL` in `.env`), via the OpenAI-compatible
  `/chat/completions` endpoint. Provider is selectable with `LLM_PROVIDER`; `anthropic` remains
  supported.
- **Prompt rules (enforced, not just requested):**
  - Answer **only** from the supplied context. If the context does not contain the fact, reply that the
    information is not available in the source pages.
  - **Maximum 3 sentences.**
  - Include **exactly one** citation URL, taken verbatim from chunk metadata — never a constructed or
    remembered URL.
  - Append `Last updated from sources: <ingested_at date>`.
  - Refuse opinionated/portfolio questions with a fixed facts-only message plus one relevant
    educational link.
  - No returns calculations or comparisons. If asked about performance, link the official factsheet.
- **No PII:** do not prompt for, accept, or store PAN, Aadhaar, account numbers, OTPs, emails, phone
  numbers. Input is not persisted.

### 6.5 UI (tiny, per brief)

- Welcome line naming the scope (HDFC MF, 5 schemes, public pages only).
- **3 example questions** as clickable buttons.
- Chat area with answers + citation link.
- Persistent disclaimer: **"Facts-only. No investment advice."**
- Optional debug panel: retrieved chunks + similarity scores (great for the demo, toggleable).

### 6.6 Config

| Env var | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` or `anthropic` |
| `GROQ_API_KEY` | — | Groq access (held in `.env`) |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Generation model (`LLM_MODEL` also accepted) |
| `CHROMA_DIR` | `./chroma_db` | Persistent vector store path |
| `COLLECTION_NAME` | `hdfc_mf_faq` | Chroma collection |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `TOP_K` | `4` | Chunks retrieved per query |

---

## 7. Guardrails & Refusals

| Situation | Behaviour |
|---|---|
| No chunk above score threshold | "I couldn't find that in the HDFC source pages." + suggest rephrasing |
| Opinionated / portfolio question | Fixed facts-only refusal + one educational link |
| Performance / return comparison | Refuse to compute; link official factsheet |
| PII typed into the chat | Detect PAN/Aadhaar/account/OTP/email/phone patterns → do not process, warn, do not store |
| Advice in the retrieved source text | Ignore; the system prompt forbids recommending |
| Financial numbers in a chunk | Reproduce only; never compute, project, or compare |

---

## 8. Success Criteria / Acceptance Tests

| ID | Test | Pass condition |
|---|---|---|
| A1 | Pipeline runs end-to-end from a clean clone | `ingest` populates ChromaDB; `chat` answers |
| A2 | "Expense ratio of HDFC Large Cap?" | ≤3 sentences + exactly 1 source link, figure traceable to the page |
| A3 | "ELSS lock-in?" | Correct lock-in stated with 1 link |
| A4 | "Minimum SIP?" | Amount stated with 1 link |
| A5 | "Should I buy HDFC Small Cap?" | Refusal, no numbers, educational link present |
| A6 | Out-of-corpus question ("What is HDFC Bank's FD rate?") | Says not found; no fabrication |
| A7 | Every sample answer re-verifiable | Human opens each cited URL and confirms the fact |
| A8 | PII safety | PAN/email/phone/OTP input is not stored or echoed back |
| A9 | Disclaimer visible in UI | Text present on first render |
| A10 | Determinism | Same question twice → materially same answer + same citation |

---

## 9. Deliverables

1. **Working prototype** — local Streamlit app (or notebook). Demo video ≤3 min if hosting isn't possible.
2. **Source list** — CSV/MD of the 5 URLs ([`data/sources.md`](../data/sources.md)).
3. **README** — setup steps, scope (AMC + schemes), known limits.
4. **Sample Q&A file** — 5–10 queries with the assistant's answers and links.
5. **Disclaimer snippet** — the facts-only text used in the UI.
6. **Architecture + implementation notes** — [`docs/architecture.md`](architecture.md),
   [`docs/implementation.md`](implementation.md), including the chunking-strategy rationale.

---

## 10. Milestones

| Phase | Scope | Done when |
|---|---|---|
| 1 | Ingestion: fetch + clean 5 pages | Parsed text saved per scheme, headers/footers stripped |
| 2 | Chunking + embedding + ChromaDB | Collection built, chunk count and samples logged |
| 3 | Retrieval CLI | Top-k chunks print for a sample query with scores |
| 4 | Claude answer layer + refusal rules | A2–A6 pass in the terminal |
| 5 | Streamlit UI | A1, A9 pass; 3 example buttons work |
| 6 | Docs, sample Q&A, known limits, demo video | All deliverables in §9 submitted |

---

## 11. Risks & Known Limits

| Risk / Limit | Impact | Mitigation |
|---|---|---|
| Groww page structure/DOM changes | Ingestion breaks | Keep selectors isolated in one module; cache raw HTML snapshots |
| Dynamic fee tables may not be in static HTML | Missing expense-ratio/exit-load facts | Fall back to the official HDFC factsheet/SID page; log which fields are missing |
| Figures change over time | Stale answers | Show `Last updated from sources:`; re-run ingestion on a schedule |
| all-MiniLM-L6-v2 is English, 384-dim, not domain-tuned | Weaker recall on finance jargon | Add per-scheme tags to the query; keep chunk size modest; consider a rerank step |
| Small corpus (5 pages) | Narrow question coverage | Be explicit in the UI about what is in scope; refuse cleanly outside it |
| Numeric fidelity in generation | Wrong fee figure even when retrieved correctly | Ground strictly in context; keep answers short; re-verify every sample answer by hand (A7) |
| Rate limits / model latency on the demo machine | Slow or failed demo run | Pre-build the ChromaDB collection and commit the setup steps; short context keeps calls fast |

---

## 12. Open Questions

1. Which chunking strategy did the data inspection settle on? (Decide at end of Phase 2.)
2. Score threshold for "no good match" — needs tuning on the out-of-corpus test (A6).
3. Do we commit the built `chroma_db/` to the repo for a zero-build demo, or require a one-line ingest run?
4. Which educational link should the refusal message point to (SEBI investor charter vs. HDFC MF education page)?

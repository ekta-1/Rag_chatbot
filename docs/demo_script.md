# Demo Script — ≤3 minutes

From `implementation.md` §6.4. The whole grade in three minutes.

## Before you start

```bash
source .venv/bin/activate
python -m src.cli ingest                # ~1 min, or skip if chroma_db/ is current
streamlit run src/app.py
```

Have these in a second terminal, ready to paste:

```bash
python -m src.cli inspect
python -m src.cli ask --explain "exit load on HDFC Flexi Cap Fund"
python scripts/eval_retrieval.py
```

**Rehearse once with the network off** to prove decision D10 (cache-first) holds. `ingest`
without `--refresh` reads `cache/raw_html/`, so it should complete with the cable pulled.

If `GROQ_API_KEY` is unset, the UI still ingests, retrieves, shows the debug expander and
refuses cleanly — but every answer reads "not found". Rehearse the *refusal* half, and do not
present it as a working answer layer.

> **Pace yourself: about one question every 20 seconds.** Groq's free tier caps tokens,
> not requests (8000 tokens/min, and each answer spends ~2200). Ask faster and the answer
> comes back "I couldn't find that" -- a rate limit wearing a retrieval failure's clothes.
> If that happens, wait and re-ask the identical question; nothing is wrong with the index.

---

## 0:00–0:20 — Problem and scope

> "A FAQ assistant for HDFC Mutual Fund scheme pages. Five schemes, public pages only, facts
> only — it will not tell you what to buy. Every answer is capped at three sentences and cites
> one link so you can check it."

Hold on the [Overview](README.md) heading. Do not read the pipeline diagram yet.

## 0:20–0:45 — The pipeline

Walk the §1 diagram in `docs/architecture.md` left to right:

> "Fetch the page → extract the labelled facts → chunk → embed with a 384-dimension MiniLM →
> store in Chroma with cosine distance → retrieve with two gates → generate a three-sentence
> answer with one citation."

Name the one non-obvious choice: figures come from the page's **`__NEXT_DATA__` JSON payload**,
not scraped text. That is why expense ratio is `1.03%` and not whatever a regex scraped.

## 0:45–1:20 — Terminal: ingest and retrieve

```bash
python -m src.cli inspect
```

Point at the output: 5 schemes, 26 chunks, chunk sizes well under the model's 256-token ceiling.

```bash
python -m src.cli ask --explain "exit load on HDFC Flexi Cap Fund"
```

> "This is the gate debugger. Every candidate chunk is scored on cosine and on IDF-weighted
> lexical coverage, and both must clear their threshold. Here the winning chunk clears both;
> here is why the others did not."

This is the strongest 30 seconds in the demo — it shows the answer is *earned*, not generated.

## 1:20–2:10 — UI: ask, answer, and prove the citation

Click the **HDFC Large Cap expense ratio** chip. Read the answer aloud, then **click the
citation link and leave it on screen.**

> "One sentence, one link. Here is the source page — the expense ratio is the first row in that
> table."

Do not skip opening the link. A right number with a broken citation is a failed demo (A7).

## 2:10–2:35 — The money shot: the debug expander

Open **🔍 Retrieved chunks (debug)** under the answer.

> "This is the exact text the model read, with cosine scores and the matched terms. Nothing
> reaches the answer that you cannot see here."

Then run, in the terminal, if time allows:

```bash
python scripts/eval_retrieval.py
```

> "25 of 28 checks, including every query type the brief asked for. The three failures are
> documented — I found them, I know why they happen, and I'm not claiming retrieval is perfect."

## 2:35–2:50 — The refusals

Type: **"Should I buy HDFC Small Cap Fund?"**

> "Refused before retrieval ever ran — that's a pre-filter, not a model instruction. No numbers,
> and a link to SEBI's investor charter for learning."

Type: **"What is HDFC Bank's FD interest rate?"**

> "Not found. No chunk cleared both gates, so it says so instead of guessing."

Paste a PAN if you have one to hand. Nothing is stored or echoed back.

## 2:50–3:00 — Disclaimer and known limits

> "Facts only, from public HDFC scheme pages — not investment advice. The honest limits: five
> pages, so anything else is refused; figures can go stale after the ingest date; the embedding
> model isn't finance-tuned; and the HTML parsing is brittle, which is why there's an integrity
> check that compares the corpus against what the page actually says."

That last clause is worth saying. It is the one that shows you test your own data rather than
trusting it.

---

## If asked: known bugs, stated plainly

Do not defend these. They were found by testing, and the fix is a documented open decision.

- `"stamp duty"` is refused even though the matching chunk scores coverage 1.000 — cosine is
  0.225, under the 0.30 floor. No threshold value fixes it without letting a different query
  through.
- `"who manages …"` ranks the Overview chunk above Fund management, because `manages` doesn't
  stem to `management`.
- A question about a fact we don't have — e.g. P/E ratio — can still retrieve chunks, because
  fund-name words alone clear the lexical gate. With a live key that could invite a fabricated
  number, so it's the one to watch.

The fix for all three is a three-state coverage rule in place of the current AND gate, which
changes the invariant in `docs/architecture.md` §5.6. It is written up, not implemented, because
it changes documented behaviour and needs sign-off.

"""Dump the whole index as a readable text file: chunks + embeddings.

    .venv/bin/python scripts/dump_index.py
    .venv/bin/python scripts/dump_index.py --out data/index_dump.txt
    .venv/bin/python scripts/dump_index.py --no-vectors        # chunks only
    .venv/bin/python scripts/dump_index.py --dim 8             # truncate vectors
    .venv/bin/python scripts/dump_index.py --query "exit load on HDFC Flexi Cap Fund"

For "show me what the RAG actually ingested", which is otherwise only visible by
opening a vector database:

  * every chunk's unprefixed text -- what is stored and what the UI debug panel
    quotes, so a reviewer can check it against the live page
  * every chunk's embedding text -- the scheme/section prefix that is ADDED
    before embedding and is NOT stored. The difference between these two is the
    single most surprising thing about the pipeline and is invisible in the UI
  * the 384-float vector, read back from Chroma, not recomputed, so what is
    printed is what is actually persisted
  * a full 26x26 cosine similarity matrix, which is what makes clustering
    visible: the four "Fees, exit load" chunks light up as a block
  * optionally, a live query scored against every chunk

Reads the persisted vectors via Chroma rather than re-encoding, so this cannot
drift from what is in the database.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG  # noqa: E402
from src.ingest import store  # noqa: E402

RULE = "=" * 100
THIN = "-" * 100


def _wrap(text: str, indent: str = "    ", width: int = 96) -> str:
    import textwrap

    if not text:
        return indent + "(empty)"
    return "\n".join(textwrap.wrap(text, width=width, initial_indent=indent,
                                   subsequent_indent=indent)) or indent + "(empty)"


def _fmt_vector(vec, dim: int, per_line: int = 8) -> str:
    vals = list(vec)
    shown = vals[:dim]
    lines = []
    for i in range(0, len(shown), per_line):
        row = "  ".join(f"{v:+.4f}" for v in shown[i:i + per_line])
        start = i
        lines.append(f"    [{start:>3}..{min(i + per_line, len(shown)) - 1:>3}]  {row}")
    if dim < len(vals):
        lines.append(f"    ... {len(vals) - dim} more dimensions not shown "
                     f"(full length {len(vals)})")
    return "\n".join(lines)


def build(out_path: Path, dim: int, show_vectors: bool, query: str | None) -> None:
    collection = store.open_collection()
    total = collection.count()

    got = collection.get(include=["embeddings", "documents", "metadatas"])
    ids = got["ids"]
    docs = got["documents"]
    metas = got["metadatas"]
    embs = got["embeddings"]

    manifest = {}
    mpath = Path(CONFIG.manifest_path)
    if mpath.exists():
        manifest = json.loads(mpath.read_text())

    records = [
        {"id": i, "doc": d, "meta": m, "emb": e}
        for i, d, m, e in zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"])
    ]
    order = sorted(
        records,
        key=lambda r: (r["meta"].get("scheme_key", ""), r["meta"].get("chunk_index", 0)),
    )

    w = out_path.open("w", encoding="utf-8")
    p = lambda s="": w.write(s + "\n")

    p(RULE)
    p("INDEX DUMP -- every chunk, its embedding text, and its embedding vector")
    p(RULE)
    p(f"generated        : {manifest.get('ingested_at', 'unknown')}")
    p(f"collection       : {CONFIG.collection_name}")
    p(f"chroma dir       : {CONFIG.chroma_dir}")
    p(f"distance space   : {store.DISTANCE_SPACE}")
    p(f"chunks in store  : {total}")
    p(f"embed model      : {CONFIG.embed_model}")
    p(f"embed dim        : {manifest.get('embed_dim', '?')}")
    p(f"max seq length   : {manifest.get('embed_max_seq_length', '?')}")
    p(f"chunking         : target={manifest.get('chunk_size_tokens')} "
      f"hard_cap={manifest.get('chunk_hard_cap_tokens')} "
      f"overlap={manifest.get('chunk_overlap_tokens')}")
    p(f"failures         : {manifest.get('failures', [])}")
    p()
    p("The 'stored text' is the unprefixed chunk body: it is what Chroma holds as")
    p("the document, what the UI debug panel shows, and what a reviewer can verify")
    p("against the live page. The 'embedding text' adds the scheme and section")
    p("prefix that is sent to the model instead, so that topic signal enters the")
    p("vector. The two differ by design.")

    p()
    p(RULE)
    p("SOURCE BREAKDOWN")
    p(THIN)
    for src in manifest.get("sources", []):
        p(f"  {src['scheme_key']:<22} {src['chunks']:>2} chunks  "
          f"{src['blocks']:>2} blocks  {src['chars']:>6} chars  "
          f"coverage={src.get('coverage')}")
        p(f"  {'':<22} {src['scheme_name']}")
        p(f"  {'':<22} {src['url']}")

    p()
    p(RULE)
    p(f"CHUNKS  ({len(order)} total)")
    p(THIN)

    for n, rec in enumerate(order, 1):
        doc, meta, emb = rec["doc"], rec["meta"], rec["emb"]
        cid = rec["id"]
        scheme_key = meta.get("scheme_key", "?")
        scheme_name = meta.get("scheme_name", "?")
        section = meta.get("section", "?")
        idx = meta.get("chunk_index", 0)
        url = meta.get("source_url", "")

        p()
        p(f"[{n:>2}] {scheme_name}  ({scheme_key})")
        p(f"     chunk_id     : {cid}")
        p(f"     section      : {section}")
        p(f"     chunk_index  : {idx}")
        p(f"     source_url   : {url}")
        p(f"     char_start   : {meta.get('char_start', '?')}")
        p(f"     stored text  : ({len(doc)} chars)")
        p(_wrap(doc))
        p(f"     embedding txt: {scheme_name} - {section}")
        p(_wrap(f"{scheme_name} - {section}\n{doc}", indent="       | "))
        if show_vectors:
            p(f"     embedding    : {len(emb)} floats")
            p(_fmt_vector(emb, dim))

    # ---- cosine similarity matrix -----------------------------------------
    p()
    p(RULE)
    p("COSINE SIMINITY MATRIX (x100)")
    p(THIN)
    p("Scale: space <0.5 | . <0.6 | o <0.7 | O <0.8 | # <0.9 | @ 1.0")
    p("Bands chosen because MIN_SIMILARITY is not the interesting threshold here:")
    p("see the off-diagonal note printed below the matrix.")

    import numpy as np

    mat = np.array([r["emb"] for r in order], dtype="float32")
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    sim = mat @ mat.T

    def band(v: float) -> str:
        if v >= 0.995:
            return "@"
        if v >= 0.9:
            return "#"
        if v >= 0.8:
            return "O"
        if v >= 0.7:
            return "o"
        if v >= 0.6:
            return "."
        return " "

    # 2 characters per column everywhere, so the headers line up with the data.
    tens = "".join(
        f"{(i + 1) // 10 % 10}" if (i + 1) % 10 == 0 else " " for i in range(len(order))
    )
    p("      " + tens)
    p("      " + "".join(f"{i + 1:>2}" for i in range(len(order))))
    for i in range(len(order)):
        cells = "".join(
            ("@" if i == j else band(float(sim[i, j]))) + " " for j in range(len(order))
        )
        p(f"  {i + 1:>2}  {cells}")

    off = [float(sim[i, j]) for i in range(len(order)) for j in range(len(order)) if i != j]
    p()
    p(f"  chunk numbers map to the [n] headings above.")
    p(f"  off-diagonal similarity: min {min(off):.2f}  max {max(off):.2f}  "
      f"mean {sum(off) / len(off):.2f}")
    if min(off) >= CONFIG.min_similarity:
        p()
        p(f"  FINDING: the lowest off-diagonal similarity in the whole corpus is")
        p(f"  {min(off):.2f}, which is ABOVE MIN_SIMILARITY ({CONFIG.min_similarity}). Every")
        p(f"  pair of chunks in this index is above the floor. So the cosine gate is")
        p(f"  a sanity check that the index is not broken, not a filter between chunks;")
        p(f"  all real discrimination between a question and a chunk is coming from")
        p(f"  the LEXICAL gate plus the top_k cut. This is consistent with the")
        p(f"  -0.330 cosine separation gap in scripts/eval_retrieval.py.")

    # ---- chunk boundary sanity --------------------------------------------
    p()
    p(RULE)
    p("CHUNK BOUNDARY SANITY")
    p(THIN)
    p("Overlap is meant to carry context across a boundary, not to start a chunk")
    p("mid-word. A chunk that opens on a fragment means the boundary landed")
    p("inside a word, which costs the embedding a few characters of signal.")

    import re as _re

    suspect = []
    for n, rec in enumerate(order, 1):
        doc = rec["doc"].strip()
        if not doc:
            continue
        first = doc.split()[0]
        # a real word starts with a capital, a digit, or a known lowercase word
        starts_lower_mid = first[:1].islower() and not _re.match(
            r"^(a|an|the|and|or|but|if|of|to|in|on|for|with|is|are|was|were|"
            r"this|that|it|as|at|by|from|has|have|not|no|so|than|then|"
            r"there|their|they|its|we|you|your|he|she|his|her|be|been)",
            first.lower(),
        )
        if starts_lower_mid:
            suspect.append((n, rec["meta"], first, doc[:70]))

    if not suspect:
        p("  All chunk openings begin on a sentence, a capital, or a known word.")
    else:
        p(f"  {len(suspect)} of {len(order)} chunks open on a lower-case fragment:")
        p()
        for n, meta, first, preview in suspect:
            p(f"    [{n:>2}] {meta.get('scheme_key')}/{meta.get('section')}")
            p(f"         opens: {first!r}  ({len(first)} chars)")
            p(f"         text : {preview!r}")
        p()
        p("  These are cosmetically wrong and semantically harmless -- the fact")
        p("  tables and sections are intact -- but they are the visible symptom of")
        p("  the chunker splitting on a fixed overlap rather than a word boundary.")

    # ---- optional live query ----------------------------------------------
    if query:
        p()
        p(RULE)
        p(f"LIVE QUERY: {query!r}")
        p(THIN)
        from src.rag.lexical import tokenize
        from src.rag.retriever import Retriever, augment_query, detect_scheme_key

        r = Retriever()
        enhanced = augment_query(query, detect_scheme_key(query))
        p(f"  scheme detected : {detect_scheme_key(query)}")
        p(f"  embedding query : {enhanced!r}")
        p()
        for row in r.explain(query, top_k=len(order)):
            chunk = row["chunk"].chunk
            passed = row["passed"]
            p(f"  {'PASS' if passed else 'drop'}  cos={row['score']:.3f} "
              f"cov={row['coverage']:.3f}  [{chunk.id}]")
            p(f"        {chunk.scheme_name} / {chunk.section}")
            p(f"        matched terms: {row['matched']}")
        p()
        p("  term -> IDF for this query (df = how many chunks contain it):")
        qtokens = tokenize(query)
        gate = r._lexical
        for t in sorted(qtokens):
            idf = gate.idf(t)
            df = gate._df.get(t, 0)
            in_corpus = "in corpus" if df else "ABSENT (idf is at its max)"
            p(f"    {t:<18} idf={idf:.3f}  df={df:>2}  {in_corpus}")

    w.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/index_dump.txt")
    ap.add_argument("--dim", type=int, default=384,
                    help="how many dimensions to print per vector (0 = all metadata only)")
    ap.add_argument("--no-vectors", action="store_true",
                    help="skip the vectors, print chunks only")
    ap.add_argument("--query", help="also score this query against every chunk")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build(out, 0 if args.no_vectors else args.dim, not args.no_vectors, args.query)
    size = out.stat().st_size
    print(f"wrote {out} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Orchestrates the offline ingest pipeline.

  sources -> fetch -> structured facts + prose -> chunk -> embed -> ChromaDB

Phase 2 added chunking, embedding and the vector-store upsert on top of the
Phase 1 fetch/extract path. Every source is carried all the way to a stored
vector; a source that fails anywhere is recorded in ``failures`` and does not
abort the run.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import CONFIG
from src.ingest.extractor import coverage_report, extract
from src.ingest.fetcher import fetch
from src.ingest.sources import load_sources
from src.ingest.structured import extract_facts, facts_summary
from src.models import Chunk, Document

log = logging.getLogger(__name__)


def _token_stats(tokenizer, texts: list[str]) -> dict:
    """Token-count distribution, so a bad chunk-size target is visible."""
    if not texts:
        return {"count": 0}
    counts = [len(tokenizer.encode(t, add_special_tokens=True)) for t in texts]
    counts_sorted = sorted(counts)
    return {
        "count": len(counts),
        "min": min(counts),
        "max": max(counts),
        "mean": round(sum(counts) / len(counts), 1),
        "median": counts_sorted[len(counts_sorted) // 2],
    }


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )


def _write_document(doc: Document) -> Path:
    path = CONFIG.document_path(doc.source.scheme_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_ingest(
    *,
    force: bool = False,
    refresh: bool = False,
    verbose: bool = False,
) -> dict:
    """Run the offline pipeline and return the manifest dict.

    A single failing source is recorded in ``failures`` and does not abort the
    run, so one broken page cannot cost us the other four.
    """
    _configure_logging(verbose)
    started = time.time()
    ingested_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    sources = load_sources()
    print(f"Ingesting {len(sources)} sources at {ingested_at}\n")

    # Loaded once and shared with the chunker: the wordpiece tokenizer is what
    # every size decision is measured in, and loading the model twice would
    # double ingest time for no benefit.
    from src.ingest.embedder import get_embedder

    embedder = get_embedder()
    tokenizer = embedder.tokenizer
    log.info("embedding model %s (max_seq_length=%s)", embedder.model_name, embedder.max_seq_length)

    all_chunks: list[Chunk] = []
    manifest_sources: list[dict] = []
    failures: list[dict] = []

    for i, source in enumerate(sources):
        if i > 0:
            time.sleep(CONFIG.request_delay_seconds)
        try:
            html = fetch(
                source.url,
                source.scheme_key,
                refresh=refresh,
                delay=CONFIG.request_delay_seconds,
            )
            # Structured facts first: they are the authoritative numbers and
            # should lead the document so the chunker never has to compete with
            # prose for room. Prose fills in the rules and how-to sections.
            fact_blocks = extract_facts(html, source)
            doc = extract(html, source, ingested_at)
            doc.blocks = fact_blocks + doc.blocks
            path = _write_document(doc)
            coverage = coverage_report(doc)
            sections = len({b.section for b in doc.blocks})
            facts = facts_summary(html)

            from src.ingest.chunker import chunk_document

            chunks = chunk_document(doc, CONFIG, tokenizer)
            all_chunks.extend(chunks)
            chunk_tokens = _token_stats(tokenizer, [c.text for c in chunks])

            manifest_sources.append(
                {
                    "scheme_key": source.scheme_key,
                    "scheme_name": source.scheme_name,
                    "url": source.url,
                    "blocks": len(doc.blocks),
                    "fact_blocks": len(fact_blocks),
                    "chars": sum(len(b.text) for b in doc.blocks),
                    "sections": sections,
                    "chunks": len(chunks),
                    "chunk_tokens": chunk_tokens,
                    "coverage": coverage,
                    "facts": facts,
                    "status": "ok",
                }
            )
            print(
                f"  {source.scheme_key:<20} {len(doc.blocks):>4} blocks "
                f"({len(fact_blocks)} fact)  {sections:>3} sections  "
                f"-> {len(chunks):>3} chunks  "
                f"[tok min/mean/max {chunk_tokens['min']}/{chunk_tokens['mean']}/{chunk_tokens['max']}]"
            )
        except Exception as exc:  # noqa: BLE001 - one bad page must not kill the run
            log.error("source %s failed: %s", source.scheme_key, exc)
            failures.append({"scheme_key": source.scheme_key, "url": source.url, "error": str(exc)})
            manifest_sources.append(
                {
                    "scheme_key": source.scheme_key,
                    "scheme_name": source.scheme_name,
                    "url": source.url,
                    "blocks": 0,
                    "status": "failed",
                }
            )
            print(f"  {source.scheme_key:<20} FAILED: {exc}")

    # Embed and store only what actually chunked. A partial index is still
    # useful; an index claiming chunks it does not have is not.
    stored = 0
    store_error: str | None = None
    stale_removed = 0
    if all_chunks:
        from src.ingest.store import delete_scheme, ensure_collection, reset, upsert_chunks

        try:
            if force:
                reset(CONFIG)
            ensure_collection(CONFIG, CONFIG.embed_model, force=force)
            # Prefix scheme + section into the vector so "lock-in?" lands on the
            # ELSS tax section; the stored text stays unprefixed for clean quotes.
            vectors = embedder.embed_documents([c.embedding_text() for c in all_chunks])
            stored = upsert_chunks(all_chunks, vectors, CONFIG)

            # A source that produced nothing this run may still have chunks from
            # a previous one. Upserts are deterministic, so clear them by hand.
            ingested_keys = {c.scheme_key for c in all_chunks}
            for source in sources:
                if source.scheme_key not in ingested_keys:
                    stale_removed += delete_scheme(source.scheme_key, CONFIG)
        except Exception as exc:  # noqa: BLE001 - report, keep the manifest useful
            log.error("vector store upsert failed: %s", exc)
            store_error = str(exc)
            failures.append({"stage": "embed_store", "error": str(exc)})
    else:
        store_error = "no chunks produced"

    manifest = {
        "ingested_at": ingested_at,
        "embed_model": CONFIG.embed_model,
        "embed_dim": 384,
        "embed_max_seq_length": embedder.max_seq_length,
        "chunker": "section-aware-recursive",
        "chunk_size_tokens": CONFIG.chunk_size_tokens,
        "chunk_hard_cap_tokens": CONFIG.chunk_hard_cap_tokens,
        "chunk_overlap_tokens": CONFIG.chunk_overlap_tokens,
        "chunk_overlap_pct": round(
            100 * CONFIG.chunk_overlap_tokens / max(CONFIG.chunk_size_tokens, 1), 1
        ),
        "chunk_count": len(all_chunks),
        "vectors_stored": stored,
        "stale_chunks_removed": stale_removed,
        "chunk_tokens": _token_stats(tokenizer, [c.text for c in all_chunks]),
        "store_error": store_error,
        "sources": manifest_sources,
        "failures": failures,
        "duration_seconds": round(time.time() - started, 1),
    }

    Path(CONFIG.manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(CONFIG.manifest_path).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _print_coverage(manifest)
    _print_chunk_report(manifest)
    print(f"\nManifest written to {CONFIG.manifest_path}")
    if store_error:
        print(f"\n  ERROR: vector store not updated ({store_error})")
    return manifest


def _print_chunk_report(manifest: dict) -> None:
    t = manifest.get("chunk_tokens") or {}
    if not t.get("count"):
        return
    cap = manifest["chunk_hard_cap_tokens"]
    target = manifest["chunk_size_tokens"]
    print(
        f"\nChunk report: {t['count']} chunks, tokens min/median/mean/max "
        f"{t['min']}/{t['median']}/{t['mean']}/{t['max']} (target {target}, cap {cap})"
    )
    if t["max"] >= cap:
        print(
            f"  WARNING: largest chunk is {t['max']} tokens, at the {cap} hard cap."
            " Sections are larger than expected -- revisit the recursive separators"
            " before trusting recall."
        )
    if t["mean"] < target * 0.5:
        print(
            f"  Note: mean chunk is only {t['mean']} tokens against a {target} target."
            " Blocks are small, so chunks stay granular -- fine for precision, but"
            " Phase 3 threshold tuning matters more."
        )


def _print_coverage(manifest: dict) -> None:
    fields = list(next((s["coverage"] for s in manifest["sources"] if "coverage" in s), {}))
    if not fields:
        return
    print("\nCoverage report (are these facts actually in the HTML?)")
    print(f"  {'scheme':<20} " + " ".join(f"{f[:9]:>9}" for f in fields))
    for entry in manifest["sources"]:
        if "coverage" not in entry:
            continue
        cells = " ".join(f"{('yes' if entry['coverage'][f] else 'NO'):>9}" for f in fields)
        print(f"  {entry['scheme_key']:<20} {cells}")

    missing_everywhere = [
        f
        for f in fields
        if all(not s["coverage"].get(f) for s in manifest["sources"] if "coverage" in s)
    ]
    if missing_everywhere:
        print(
            f"\n  WARNING: no source covers {', '.join(missing_everywhere)}."
        )
        print(
            "  This usually means the data is rendered client-side and never reached us."
            "\n  See docs/implementation.md open decision A5 (add official factsheet URLs)."
        )
    else:
        partial = [
            f
            for f in fields
            if any(
                s["coverage"].get(f)
                for s in manifest["sources"]
                if "coverage" in s
            )
            and any(
                not s["coverage"].get(f)
                for s in manifest["sources"]
                if "coverage" in s
            )
        ]
        if partial:
            # Often legitimate: e.g. lock-in exists only for ELSS.
            print(
                f"\n  Note: {', '.join(partial)} absent from some schemes. That is expected"
                "\n  when the attribute genuinely does not apply to that scheme (lock-in is"
                "\n  ELSS-only), so verify before treating it as an ingestion gap."
            )


def inspect_index() -> dict:
    """Print the manifest plus live collection stats."""
    path = Path(CONFIG.manifest_path)
    if not path.exists():
        print(f"No manifest at {path}. Run: python -m src.cli ingest")
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))

    summary = {k: v for k, v in manifest.items() if k != "sources"}
    print("--- manifest ---")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("--- per source ---")
    for entry in manifest["sources"]:
        print(
            f"  {entry['scheme_key']:<20} blocks={entry.get('blocks', 0):>4} "
            f"chunks={entry.get('chunks', 0):>4} status={entry['status']}"
        )

    print("--- vector store ---")
    try:
        from src.ingest.store import stats

        print(json.dumps(stats(CONFIG), indent=2))
    except Exception as exc:  # noqa: BLE001 - inspect must never crash
        print(f"  could not read collection: {exc}")
    return manifest

"""Command line entry point: ``python -m src.cli <command>``.

Subcommands:
    ingest    fetch + extract the source pages and build the index
    ask       ask a question from the terminal
    inspect   show the index manifest and collection stats

All heavy imports (chromadb, sentence-transformers, anthropic) are deferred
inside the handlers so that ``--help`` and the ingest phase work in an
environment where those packages are not installed yet.
"""

from __future__ import annotations

import argparse
import sys

from src.config import CONFIG


def _cmd_ingest(args: argparse.Namespace) -> int:
    from src.ingest.pipeline import run_ingest

    manifest = run_ingest(force=args.force, refresh=args.refresh)
    print(f"Ingested {manifest['chunk_count']} chunks from {len(manifest['sources'])} sources.")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    """Full online answer via the chain (Phase 4).

    ``--explain`` and ``--retrieve-only`` stay offline and skip the API entirely;
    they are the Phase 3 diagnostics and are useful when a gate misbehaves.
    """
    from src.rag.retriever import Retriever, EmptyIndexError, IndexModelMismatch

    if args.explain or args.retrieve_only:
        try:
            retriever = Retriever()
        except EmptyIndexError as exc:
            print(f"Index not built: {exc}", file=sys.stderr)
            return 1
        except IndexModelMismatch as exc:
            print(f"Embedding model mismatch: {exc}", file=sys.stderr)
            return 2

        if args.explain:
            return _print_explanation(retriever, args.question)
        return _print_hits(retriever, args.question)

    from src.rag.chain import answer

    result = answer(args.question)

    if result.refused:
        print(f"[refused:{result.refusal_kind}]")
    # result.text already carries its own trailing "Source: <url>" -- that is
    # citations.validate's job, and it is the single copy G1 is measured on.
    # Printing result.citation_url again rendered every answer with two source
    # lines. citation_url is kept on the result for the API/UI and for tests.
    print(result.text)
    if result.truncated:
        print("(answer was truncated to 3 sentences)")

    # The gate working should be visible from the terminal, per implementation.md 4.6.
    if result.hits:
        print(f"\nRetrieved {len(result.hits)} chunk(s) above threshold:")
        for hit in result.hits:
            print(
                f"  {hit.rank}. score={hit.score:.3f} | "
                f"{hit.chunk.scheme_key} / {hit.chunk.section}"
            )
    print(f"\nLast updated from sources: {result.last_updated}")
    return 0


def _print_hits(retriever, question: str) -> int:
    """Retrieval only, no API call. The Phase 3 behaviour, kept for debugging."""
    hits = retriever.search(question)
    if not hits:
        print(
            "I do not know, based on the available sources.\n"
            "No indexed chunk cleared both the similarity and lexical-coverage gates."
        )
        return 0

    print(f"Found {len(hits)} hit(s) above threshold:")
    for hit in hits:
        print(
            f"\n{hit.rank}. score={hit.score:.3f} | {hit.chunk.scheme_key} / {hit.chunk.section}"
        )
        print(hit.chunk.text)
    return 0


def _print_explanation(retriever, question: str) -> int:
    """Show every candidate with both gate signals. Debugging aid, not the demo."""
    rows = retriever.explain(question, top_k=8)
    if not rows:
        print("Empty question.")
        return 0

    print(
        f"gates: cosine >= {retriever.cfg.min_similarity}, "
        f"coverage >= {retriever.cfg.min_lexical_coverage}\n"
    )
    for i, row in enumerate(rows, 1):
        chunk = row["chunk"].chunk
        verdict = "PASS" if row["passed"] else f"FAIL ({row['reason']})"
        print(
            f"{i:>2}. {verdict:<34} sim={row['score']:.3f} cov={row['coverage']:.3f} "
            f"[{chunk.scheme_key} / {chunk.section[:30]}]"
        )
        print(f"    matched: {', '.join(row['matched']) or '(nothing)'}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    from src.ingest.pipeline import inspect_index

    inspect_index()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Mutual fund facts-only RAG chatbot utilities.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="fetch, extract, chunk, embed and index sources")
    p_ingest.add_argument(
        "--force",
        action="store_true",
        help="rebuild the vector store even if it was built with a different embedding model",
    )
    p_ingest.add_argument(
        "--refresh",
        action="store_true",
        help="ignore the cached HTML and re-fetch every page",
    )
    p_ingest.set_defaults(func=_cmd_ingest)

    p_ask = sub.add_parser("ask", help="ask a question (guards -> retrieval -> LLM -> citation)")
    p_ask.add_argument("question", help="e.g. 'expense ratio of HDFC Large Cap Fund'")
    p_ask.add_argument(
        "--explain",
        action="store_true",
        help="offline: show every candidate with both gate signals and why it passed or failed",
    )
    p_ask.add_argument(
        "--retrieve-only",
        action="store_true",
        help="offline: print retrieved chunks without calling the API",
    )
    p_ask.set_defaults(func=_cmd_ask)

    p_inspect = sub.add_parser("inspect", help="print the index manifest and collection stats")
    p_inspect.set_defaults(func=_cmd_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

"""Retrieval eval report. Answers "is retrieval actually working?".

    .venv/bin/python scripts/eval_retrieval.py            # full report
    .venv/bin/python scripts/eval_retrieval.py --verbose  # per-candidate detail
    .venv/bin/python scripts/eval_retrieval.py --query "exit load on HDFC Flexi Cap Fund"

NO API KEY REQUIRED. Retrieval is entirely local: the corpus, the embeddings and
both gates run offline. This is the one part of the system that can be fully
verified today, and it is the part everything else depends on -- if retrieval is
wrong, no prompt or model can rescue it.

What it checks, per case:
  * did the correct chunk survive BOTH gates (not gated out)?
  * is it at rank 1?
  * is it the right scheme?
  * does the chunk actually CONTAIN the fact, not merely sit in the right section?

That last one is the check that matters. A chunk in the right section that lacks
the number produces a confidently wrong answer, which is worse than a refusal.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.labeled_set import (  # noqa: E402
    NEGATIVES,
    POSITIVES,
    REQUIRED_QUERY_TYPES,
    UI_EXAMPLES,
    UNANSWERABLE,
)

OK = "PASS"
BAD = "FAIL"


def _rule(char: str = "-", width: int = 78) -> str:
    return char * width


def _section(title: str) -> None:
    print(f"\n{title}\n{_rule()}")


def _judge_positive(retriever, question, scheme_key, section, evidence, provenance) -> bool:
    hits = retriever.search(question)
    if not hits:
        print(f"  {BAD}  {question!r}")
        print(f"        no chunk cleared both gates. The answer was REFUSED.")
        return False

    top = hits[0].chunk
    problems = []
    if section and section not in top.section.lower():
        problems.append(f"section={top.section!r} (wanted ~{section!r})")
    if scheme_key and top.scheme_key != scheme_key:
        problems.append(f"scheme={top.scheme_key!r} (wanted {scheme_key!r})")
    if evidence and evidence.lower() not in top.text.lower():
        problems.append(f"top chunk does NOT contain {evidence!r}")

    ok = not problems
    print(f"  {OK if ok else BAD}  {question}")
    print(f"        top1 rank=1 score={hits[0].score:.3f} "
          f"{top.scheme_key}/{top.section}")
    if ok:
        fact = _first_line_with(top.text, evidence or "")
        print(f"        fact found: {fact}")
    for problem in problems:
        print(f"        PROBLEM: {problem}")
    if len(hits) > 1:
        others = ", ".join(
            f"{h.chunk.scheme_key}/{h.chunk.section}@{h.score:.3f}" for h in hits[1:]
        )
        print(f"        also retrieved: {others}")
    return ok


def _judge_refusal(retriever, question, why) -> bool:
    hits = retriever.search(question)
    ok = hits == []
    print(f"  {OK if ok else BAD}  {question!r}  ({why})")
    if not ok:
        print(f"        GATE LEAKED: returned {len(hits)} hit(s), top1="
              f"{hits[0].chunk.scheme_key}/{hits[0].chunk.section}@{hits[0].score:.3f}")
        print(f"        text: {hits[0].chunk.text[:90]!r}")
    return ok


def _first_line_with(text: str, needle: str) -> str:
    if not needle:
        return text.splitlines()[0][:80] if text else ""
    for line in text.splitlines():
        if needle.lower() in line.lower():
            return line.strip()[:80]
    return ""


def main() -> int:
    verbose = "--verbose" in sys.argv
    single = None
    if "--query" in sys.argv:
        single = sys.argv[sys.argv.index("--query") + 1]

    from src.rag.retriever import Retriever

    retriever = Retriever()
    cfg = retriever.cfg

    print(_rule("="))
    print("RETRIEVAL EVAL -- fully offline, no API key needed")
    print(_rule("="))
    print(f"corpus      : {retriever.index_size} chunks")
    print(f"model       : {cfg.embed_model}")
    print(f"gates       : cosine >= {cfg.min_similarity}  AND  "
          f"lexical coverage >= {cfg.min_lexical_coverage}")
    print(f"top_k       : {cfg.top_k}")

    if single:
        _section(f"SINGLE QUERY: {single!r}")
        for row in retriever.explain(single, top_k=8):
            mark = "PASS" if row["passed"] else "drop"
            print(f"  {mark}  cov={row['coverage']:.3f} cos={row['score']:.3f} "
                  f"{row['chunk'].chunk.scheme_key}/{row['chunk'].chunk.section}")
        hits = retriever.search(single)
        print(f"\n  -> {len(hits)} hit(s) above both gates")
        for hit in hits:
            print(f"     {hit.rank}. {hit.score:.3f} "
                  f"{hit.chunk.scheme_name} / {hit.chunk.section}")
            if verbose:
                print(f"        {hit.chunk.text[:220]}")
        return 0

    results: list[bool] = []

    _section("A. POSITIVE QUERIES -- must retrieve the right chunk, with the fact in it")
    for case in POSITIVES:
        results.append(_judge_positive(retriever, *case))

    _section("B. UNANSWERABLE-BUT-WELL-FORMED -- must be refused (A6)")
    for question, why in UNANSWERABLE:
        results.append(_judge_refusal(retriever, question, why))

    _section("C. OUT-OF-CORPUS / ADVICE -- must return zero hits")
    for question, why in NEGATIVES:
        results.append(_judge_refusal(retriever, question, why))

    _section("D. THE 3 UI EXAMPLE CHIPS")
    for question in UI_EXAMPLES:
        hits = retriever.search(question)
        expect_refusal = "FD interest rate" in question
        ok = (hits == []) if expect_refusal else bool(hits)
        results.append(ok)
        print(f"  {OK if ok else BAD}  {question!r} -> {len(hits)} hit(s)")

    _section("E. SEPARATION EVIDENCE -- why two gates and not one")
    best = lambda q: max(
        (row["score"] for row in retriever.explain(q, top_k=8)), default=0.0
    )
    cov = lambda q: max(
        (row["coverage"] for row in retriever.explain(q, top_k=8)), default=0.0
    )
    answerable = [q for q, _, _, _, _ in POSITIVES]
    refusable = [q for q, _ in UNANSWERABLE] + [q for q, _ in NEGATIVES]

    cos_a, cos_r = min(map(best, answerable)), max(map(best, refusable))
    cov_a, cov_r = min(map(cov, answerable)), max(map(cov, refusable))
    print(f"  cosine    answerable min {cos_a:.3f} | refuse max {cos_r:.3f} | "
          f"gap {cos_a - cos_r:+.3f}")
    print(f"  coverage  answerable min {cov_a:.3f} | refuse max {cov_r:.3f} | "
          f"gap {cov_a - cov_r:+.3f}")
    print()
    if cos_a <= cos_r:
        print(f"  NOTE: cosine ALONE does not separate these sets (gap {cos_a - cos_r:+.3f}).")
        print("        This is why MIN_SIMILARITY alone is a loose floor and")
        print("        MIN_LEXICAL_COVERAGE does the real work.")

    cov_margin = cov_a - cov_r
    if 0 < cov_margin < 0.05:
        worst = max(refusable, key=cov)
        print(f"  WARNING: the coverage margin is only {cov_margin:+.3f}. That is thin.")
        print(f"    Closest refusal: {worst!r} at coverage {cov(worst):.3f}, "
              f"cosine {best(worst):.3f}.")
        print("    It is refused only because cosine is below MIN_SIMILARITY. If the")
        print("    cosine floor is ever lowered, this case starts leaking. The two")
        print("    gates are covering for each other here, not each redundant.")
    elif cov_margin > 0:
        print(f"  coverage separates the sets with a {cov_margin:+.3f} margin.")
    else:
        print("  cosine alone separates on this run; keep the second gate anyway,")
        print("  since the margin is a property of this corpus, not a guarantee.")
    separated = cov_a > cov_r
    results.append(separated)

    _section("F. REQUIRED QUERY TYPES (docs/Problemstatement.txt)")
    for label, question in REQUIRED_QUERY_TYPES.items():
        hits = retriever.search(question)
        answerable_q = "download" not in question
        ok = bool(hits) if answerable_q else hits == []
        results.append(ok)
        status = f"{len(hits)} hit(s)" if hits else "refused"
        expected = "answerable" if answerable_q else "expected refusal (not in corpus)"
        print(f"  {OK if ok else BAD}  {label:<28} {status:<12} [{expected}]")

    passed = sum(results)
    total = len(results)

    # Section G: corpus integrity. Retrieval can only confirm that a fact was
    # retrieved, never that it is true -- a wrongly extracted figure is found at
    # rank 1 with a valid citation. This checks the corpus against the values
    # the page itself states, so that class of bug fails the eval.
    from evals.integrity import check_all

    integrity = check_all()
    print("G. CORPUS INTEGRITY -- corpus vs the page's own values")
    print("-" * 78)
    if not integrity:
        print(f"  {OK}  every checked fact matches the page, with no self-conflicts")
    else:
        for issue in integrity:
            print(f"  {BAD}  [{issue.kind:11s}] {issue.scheme_key:20s} {issue.fact}")
            print(f"          {issue.detail}")
    results.append(not integrity)
    total += 1

    print(f"\n{_rule('=')}")
    print(f"RESULT: {passed}/{total} checks passed")
    print(_rule("="))
    if passed != total:
        print("\nFailing checks above. For a single case:")
        print('  .venv/bin/python scripts/eval_retrieval.py --query "<question>" --verbose')
        return 1
    print("\nRetrieval is healthy. This is the part that is fully verifiable")
    print("without an API key. Generated answers are NOT verified here --")
    print("that is tests/test_acceptance.py, which needs ANTHROPIC_API_KEY.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

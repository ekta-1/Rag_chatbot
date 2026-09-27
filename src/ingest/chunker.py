"""Split documents into embeddable chunks.

Strategy (architecture decision D4, recorded in architecture.md section 5.3):
**heading-aware first, recursive character split as the fallback inside
oversized sections.**

Why not the alternatives:

* ``RecursiveCharacterSplitter`` on the whole page -- would happily mix "Fees
  and charges" with "Taxation" in one chunk, so a citation could not name a
  section.
* Semantic splitting -- costs a second embedding pass at index time (~4x
  slower ingest) and still has no idea where the headings are. For a 5-page
  corpus that is a bad trade.

Three invariants this module must never break:

1. **A chunk belongs to exactly one section.** Never merge across sections, so
   ``chunk.section`` is trustworthy enough to put in a citation.
2. **A table row or list item is never split.** Expense ratio and exit-load
   figures live in tables; separating a label from its value produces a
   confidently wrong answer.
3. **No chunk exceeds the hard cap.** ``all-MiniLM-L6-v2`` truncates at 256
   wordpiece tokens, so anything longer loses its tail at embedding time and
   that content becomes unretrievable. See implementation.md section 0.1.

Token counts always come from the model tokenizer, never ``len(text.split())``
-- wordpiece is not whitespace.
"""

from __future__ import annotations

import logging

from src.config import CONFIG
from src.models import Chunk, Document, TextBlock, make_chunk_id

log = logging.getLogger(__name__)

# Tried in order, coarsest first. No LangChain here on purpose (implementation.md
# section 0.2 / architecture decision D2): the pipeline stages must stay visible.
SEPARATORS = ("\n\n", "\n", ". ", " ", "")
MAX_SPLIT_DEPTH = 6
BLOCK_JOIN = "\n\n"


class _Atom:
    """An indivisible unit of text plus where it came from.

    Atoms are the smallest things allowed to move between chunks. A whole
    ``TextBlock`` is one atom when it fits; an oversized block becomes several.
    """

    __slots__ = ("text", "kind", "section", "char_start")

    def __init__(self, text: str, kind: str, section: str, char_start: int) -> None:
        self.text = text
        self.kind = kind
        self.section = section
        self.char_start = char_start


def _count(tokenizer, text: str) -> int:
    """Wordpiece token count, including special tokens."""
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=True))


def _hard_cut(text: str, size: int, tokenizer) -> list[str]:
    """Last-resort split on exact token boundaries.

    Only reached when a single unbreakable unit (e.g. one enormous table row)
    exceeds the cap. Cutting on token ids rather than characters guarantees the
    result decodes back to valid text instead of mojibake.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    step = max(size - 2, 1)  # leave room for special tokens
    pieces = [
        tokenizer.decode(ids[i : i + step], skip_special_tokens=True)
        for i in range(0, len(ids), step)
    ]
    return [p for p in pieces if p.strip()]


def _recursive_split(text: str, size: int, tokenizer, depth: int = 0) -> list[str]:
    """Hand-rolled recursive character split.

    Tries ``SEPARATORS`` coarsest-first and recurses into any piece that is still
    too long. ``depth`` bounds the recursion so a pathological input (one
    gigantic unbroken token run) cannot spin; ``_hard_cut`` guarantees
    termination at the limit.
    """
    text = text.strip()
    if not text:
        return []
    if _count(tokenizer, text) <= size:
        return [text]
    if depth >= MAX_SPLIT_DEPTH:
        log.warning("max split depth reached; hard-cutting %d tokens", _count(tokenizer, text))
        return _hard_cut(text, size, tokenizer)

    for sep in SEPARATORS[:-1]:
        if sep not in text:
            continue
        out: list[str] = []
        for piece in text.split(sep):
            piece = piece.strip()
            if not piece:
                continue
            if _count(tokenizer, piece) > size:
                out.extend(_recursive_split(piece, size, tokenizer, depth + 1))
            else:
                out.append(piece)
        if out:
            return out

    # No separator present: the text is one unbreakable run.
    return _hard_cut(text, size, tokenizer)


def _make_atoms(doc: Document, cfg, tokenizer) -> list[_Atom]:
    """Flatten a document's blocks into atoms, tracking absolute char offsets.

    ``char_start`` is the offset of the atom within the document's concatenated
    block text, which is what makes hand-verification of a chunk possible
    (acceptance test A7).
    """
    atoms: list[_Atom] = []
    offset = 0
    hard_cap = cfg.chunk_hard_cap_tokens

    for block in doc.blocks:
        text = block.text.strip()
        if not text:
            offset += len(block.text) + len(BLOCK_JOIN)
            continue

        if _count(tokenizer, text) <= hard_cap:
            atoms.append(_Atom(text, block.kind, block.section, offset))
        elif block.kind in ("table", "list"):
            # Never bisect a row: split between rows, then recurse only if a
            # single row is itself over the cap.
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if _count(tokenizer, line) <= hard_cap:
                    atoms.append(_Atom(line, block.kind, block.section, offset))
                else:
                    log.warning("single table row exceeds cap; splitting it")
                    for piece in _recursive_split(line, hard_cap, tokenizer):
                        atoms.append(_Atom(piece, block.kind, block.section, offset))
        else:
            for piece in _recursive_split(text, cfg.chunk_size_tokens, tokenizer):
                atoms.append(_Atom(piece, block.kind, block.section, offset))

        offset += len(block.text) + len(BLOCK_JOIN)

    return atoms


def _token_tail(text: str, overlap_tokens: int, tokenizer) -> str:
    """Last ``overlap_tokens`` wordpieces of ``text``."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= overlap_tokens:
        return text
    return tokenizer.decode(ids[-overlap_tokens:], skip_special_tokens=True).strip()


def _overlap_tail(previous_text: str, overlap_tokens: int, tokenizer) -> str:
    """Last ``overlap_tokens`` tokens of the previous chunk, structure intact.

    For multi-line text the tail is assembled from whole trailing lines, because
    a wordpiece-level slice of a table decodes to a flattened row mash like
    ``percent Expense ratio row 3 | value 3 percent Expense ratio row 4`` --
    which breaks the "never split a row" invariant on the next chunk and reads
    badly to the model. A line that is itself bigger than the whole overlap
    budget (ordinary prose) falls back to an exact token slice, so the overlap
    can never grow past its budget and push a chunk over the hard cap.
    """
    if not previous_text or overlap_tokens <= 0:
        return ""

    lines = previous_text.split("\n")
    if len(lines) > 1:
        picked: list[str] = []
        total = 0
        for line in reversed(lines):
            n = _count(tokenizer, line)
            if n > overlap_tokens:
                if picked:
                    break
                return _token_tail(line, overlap_tokens, tokenizer)
            if picked and total + n > overlap_tokens:
                break
            picked.append(line)
            total += n
            if total >= overlap_tokens:
                break
        return "\n".join(reversed(picked)).strip()

    return _token_tail(previous_text, overlap_tokens, tokenizer)


def _pack_section(atoms: list[_Atom], cfg, tokenizer) -> list[list[_Atom]]:
    """Greedily pack atoms into chunk-sized groups.

    The content budget is ``chunk_size - overlap`` so that after the previous
    chunk's tail is prepended, the assembled chunk still respects
    ``chunk_size``. Overlap is applied at assembly time (not here) so that a
    chunk can never end up containing nothing but overlap.
    """
    budget = max(cfg.chunk_size_tokens - cfg.chunk_overlap_tokens, 1)
    groups: list[list[_Atom]] = []
    current: list[_Atom] = []
    current_tokens = 0

    for atom in atoms:
        n = _count(tokenizer, atom.text)
        if current and current_tokens + n > budget:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(atom)
        current_tokens += n

    if current:
        groups.append(current)
    return groups


def chunk_document(doc: Document, cfg=CONFIG, tokenizer=None) -> list[Chunk]:
    """Split one Document into Chunks.

    Never crosses a section boundary, never splits a table row or list item, and
    never exceeds ``cfg.chunk_hard_cap_tokens``.

    Args:
        tokenizer: a HuggingFace tokenizer. Required -- there is no
            whitespace fallback, because wordpiece counts are the whole point.

    Returns:
        Chunks in document order. Empty if the document has no usable blocks.
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required; wordpiece counts drive every size decision")

    if not doc.blocks:
        return []

    atoms = _make_atoms(doc, cfg, tokenizer)
    if not atoms:
        return []

    source = doc.source
    chunks: list[Chunk] = []
    index = 0
    previous_text = ""
    previous_section = None

    # Group atoms into contiguous runs of the same section. A section boundary
    # always starts a new chunk, which is what keeps chunks section-pure.
    i = 0
    while i < len(atoms):
        section = atoms[i].section
        j = i
        while j < len(atoms) and atoms[j].section == section:
            j += 1
        section_atoms = atoms[i:j]
        i = j

        for group in _pack_section(section_atoms, cfg, tokenizer):
            body = "\n".join(atom.text for atom in group).strip()
            if not body:
                continue

            # Overlap only within a section, and never onto the first chunk of
            # one -- repeating a whole section's opening is just noise.
            prefix = ""
            if previous_section == section and previous_text:
                prefix = _overlap_tail(previous_text, cfg.chunk_overlap_tokens, tokenizer)

            text = f"{prefix}\n{body}" if prefix else body
            if _count(tokenizer, text) > cfg.chunk_hard_cap_tokens:
                # Should be unreachable: content budget already reserves room
                # for the overlap. Fail loudly rather than ship a truncated chunk.
                raise ValueError(
                    f"chunk {index} for {source.scheme_key} exceeds hard cap "
                    f"({_count(tokenizer, text)} > {cfg.chunk_hard_cap_tokens}); "
                    "overlap and content budget are inconsistent"
                )

            if _count(tokenizer, text) < cfg.min_chunk_tokens:
                log.debug(
                    "dropping short chunk %d (%d tokens) in %s",
                    index,
                    _count(tokenizer, text),
                    section,
                )
                previous_text = text
                previous_section = section
                continue

            chunks.append(
                Chunk(
                    id=make_chunk_id(source.url, section, index),
                    text=text,
                    source_url=source.url,
                    scheme_key=source.scheme_key,
                    scheme_name=source.scheme_name,
                    category=source.category,
                    section=section,
                    chunk_index=index,
                    ingested_at=doc.ingested_at,
                    char_start=group[0].char_start,
                )
            )
            index += 1
            previous_text = text
            previous_section = section

    log.info(
        "chunked %s: %d blocks -> %d chunks",
        source.scheme_key,
        len(doc.blocks),
        len(chunks),
    )
    return chunks

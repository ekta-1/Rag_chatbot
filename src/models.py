"""Core data contracts shared by every stage of the pipeline.

These dataclasses are the interface between ingestion and retrieval; they are
mirrored from ``docs/architecture.md`` section 4. Import them rather than
redefining: chunk metadata, ChromaDB records, and answer rendering all assume
these exact field names.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    """One public source page, as declared in ``data/sources.md``."""

    scheme_key: str
    category: str
    scheme_name: str
    url: str


@dataclass(frozen=True)
class TextBlock:
    """A cleaned, ordered piece of page content with its heading context.

    ``section`` and ``kind`` are deliberately preserved instead of flattening the
    page to one string: they are what make heading-aware chunking and precise
    citations possible in later phases.
    """

    section: str
    text: str
    kind: str = "prose"  # "prose" | "table" | "list"


@dataclass
class Document:
    """All extracted content for a single source page."""

    source: Source
    blocks: list[TextBlock] = field(default_factory=list)
    ingested_at: str = ""

    def to_dict(self) -> dict:
        return {
            "source": {
                "scheme_key": self.source.scheme_key,
                "category": self.source.category,
                "scheme_name": self.source.scheme_name,
                "url": self.source.url,
            },
            "ingested_at": self.ingested_at,
            "blocks": [
                {"section": b.section, "text": b.text, "kind": b.kind} for b in self.blocks
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        src = data["source"]
        return cls(
            source=Source(
                scheme_key=src["scheme_key"],
                category=src["category"],
                scheme_name=src["scheme_name"],
                url=src["url"],
            ),
            blocks=[TextBlock(**b) for b in data.get("blocks", [])],
            ingested_at=data.get("ingested_at", ""),
        )


@dataclass
class Chunk:
    """An embeddable unit of text plus everything needed to cite it."""

    id: str
    text: str
    source_url: str
    scheme_key: str
    scheme_name: str
    category: str
    section: str
    chunk_index: int
    ingested_at: str
    char_start: int = 0

    def embedding_text(self) -> str:
        """Text actually sent to the embedding model.

        Prefixing the scheme and section puts topic signal into the vector, so a
        query like "lock-in?" is more likely to land on the ELSS tax section. The
        stored ``text`` stays unprefixed so the UI can show clean quotes.
        """
        return f"{self.scheme_name} - {self.section}\n{self.text}"

    def to_metadata(self) -> dict:
        return {
            "source_url": self.source_url,
            "scheme_key": self.scheme_key,
            "scheme_name": self.scheme_name,
            "category": self.category,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "ingested_at": self.ingested_at,
            "char_start": self.char_start,
        }


@dataclass
class Hit:
    """A retrieved chunk with its similarity score."""

    chunk: Chunk
    score: float
    rank: int = 0


@dataclass
class Answer:
    """The final response plus enough context to render and audit it."""

    text: str
    citation_url: str | None = None
    last_updated: str = ""
    refused: bool = False
    refusal_kind: str | None = None  # "advice" | "pii" | "no_match" | None
    hits: list[Hit] = field(default_factory=list)
    truncated: bool = False


def make_chunk_id(source_url: str, section: str, chunk_index: int) -> str:
    """Deterministic chunk id (architecture decision D9).

    Deterministic ids let a re-ingest upsert in place instead of duplicating rows
    in the vector store.
    """
    raw = f"{source_url}|{section}|{chunk_index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

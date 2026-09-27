"""Vector store tests.

Uses a temporary Chroma directory so the real index is never touched. The
collection is tiny and the embeddings are deterministic vectors, so none of this
needs the embedding model.
"""

from __future__ import annotations

import pytest

from src import config as config_module
from src.models import Chunk, make_chunk_id


@pytest.fixture
def cfg(tmp_path):
    """A Config pointed at a throwaway database directory."""
    import dataclasses

    return dataclasses.replace(config_module.CONFIG, chroma_dir=str(tmp_path / "chroma"))


def make_chunks(count: int = 3) -> list[Chunk]:
    return [
        Chunk(
            id=make_chunk_id("https://example.invalid/a", "Fees", i),
            text=f"Expense ratio chunk {i} is one point oh three percent.",
            source_url="https://example.invalid/a",
            scheme_key="large_cap",
            scheme_name="HDFC Large Cap Fund",
            category="Large Cap",
            section="Fees",
            chunk_index=i,
            ingested_at="2026-09-27T00:00:00+00:00",
            char_start=i * 10,
        )
        for i in range(count)
    ]


def vec(seed: int, dim: int = 384) -> list[float]:
    v = [0.0] * dim
    v[seed % dim] = 1.0
    return v


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def test_ensure_collection_is_idempotent(cfg):
    from src.ingest.store import count, ensure_collection, upsert_chunks

    ensure_collection(cfg, cfg.embed_model)
    ensure_collection(cfg, cfg.embed_model)
    assert count(cfg) == 0


def test_upsert_is_idempotent_for_identical_chunks(cfg):
    from src.ingest.store import count, ensure_collection, upsert_chunks

    chunks = make_chunks(3)
    ensure_collection(cfg, cfg.embed_model)
    upsert_chunks(chunks, [vec(i) for i in range(3)], cfg)
    assert count(cfg) == 3
    upsert_chunks(chunks, [vec(i) for i in range(3)], cfg)
    assert count(cfg) == 3, "re-ingest duplicated rows instead of upserting"


def test_upsert_replaces_changed_text_for_the_same_id(cfg):
    from src.ingest.store import get_chunk, ensure_collection, upsert_chunks

    chunks = make_chunks(1)
    ensure_collection(cfg, cfg.embed_model)
    upsert_chunks(chunks, [vec(0)], cfg)

    changed = [chunks[0].__class__(**{**chunks[0].__dict__, "text": "updated text"})]
    upsert_chunks(changed, [vec(0)], cfg)
    assert get_chunk(chunks[0].id, cfg).text == "updated text"


def test_upsert_validates_length_mismatch(cfg):
    from src.ingest.store import ensure_collection, upsert_chunks

    ensure_collection(cfg, cfg.embed_model)
    with pytest.raises(ValueError, match="3 chunks but 2 vectors"):
        upsert_chunks(make_chunks(3), [vec(0), vec(1)], cfg)


def test_reset_clears_the_collection(cfg):
    from src.ingest.store import count, ensure_collection, reset, upsert_chunks

    ensure_collection(cfg, cfg.embed_model)
    upsert_chunks(make_chunks(2), [vec(0), vec(1)], cfg)
    assert count(cfg) == 2
    reset(cfg)
    assert count(cfg) == 0


# --------------------------------------------------------------------------
# Model guard
# --------------------------------------------------------------------------


def test_refuses_to_mix_embedding_models(cfg):
    """A silent dimension mismatch would poison every future query."""
    from src.ingest.store import EmbeddingModelMismatch, ensure_collection

    ensure_collection(cfg, "sentence-transformers/all-MiniLM-L6-v2")
    with pytest.raises(EmbeddingModelMismatch, match="all-MiniLM-L6-v2"):
        ensure_collection(cfg, "some/other-model")


def test_force_allows_intentional_model_change(cfg):
    from src.ingest.store import ensure_collection, stats, upsert_chunks

    ensure_collection(cfg, "sentence-transformers/all-MiniLM-L6-v2")
    upsert_chunks(make_chunks(2), [vec(0), vec(1)], cfg)
    ensure_collection(cfg, "some/other-model", force=True)
    assert stats(cfg)["count"] == 0, "changing models must invalidate the old vectors"
    assert stats(cfg)["embed_model"] == "some/other-model"


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


def test_metadata_round_trips(cfg):
    from src.ingest.store import ensure_collection, get_chunk, upsert_chunks

    chunks = make_chunks(2)
    ensure_collection(cfg, cfg.embed_model)
    upsert_chunks(chunks, [vec(0), vec(1)], cfg)

    stored = get_chunk(chunks[1].id, cfg)
    assert stored is not None
    assert stored.text == chunks[1].text
    assert stored.scheme_key == "large_cap"
    assert stored.section == "Fees"
    assert stored.source_url == "https://example.invalid/a"
    assert stored.chunk_index == 1


def test_get_chunk_returns_none_for_unknown_id(cfg):
    from src.ingest.store import ensure_collection, get_chunk

    ensure_collection(cfg, cfg.embed_model)
    assert get_chunk("nope", cfg) is None


def test_query_returns_nearest_chunk(cfg):
    from src.ingest.store import ensure_collection, query, upsert_chunks

    ensure_collection(cfg, cfg.embed_model)
    upsert_chunks(make_chunks(3), [vec(0), vec(1), vec(2)], cfg)

    hits = query(cfg=cfg, vector=vec(1), top_k=2)
    assert len(hits) == 2
    assert hits[0].chunk.chunk_index == 1
    assert hits[0].score >= hits[1].score
    assert -1.0 <= hits[0].score <= 1.0


def test_collection_uses_cosine_distance(cfg):
    """Regression: without this, Chroma defaults to L2 and `1 - distance` is
    not a cosine similarity, so every Phase 3 threshold would be meaningless."""
    from src.ingest.store import ensure_collection

    collection = ensure_collection(cfg, cfg.embed_model)
    assert collection.metadata["hnsw:space"] == "cosine"


def test_rejects_collection_built_with_the_wrong_space(cfg):
    from src.ingest.store import EmbeddingModelMismatch, get_client

    client = get_client(cfg)
    client.create_collection(
        name=cfg.collection_name,
        metadata={"hnsw:space": "l2", "embed_model": cfg.embed_model},
    )
    with pytest.raises(EmbeddingModelMismatch, match="distance space"):
        from src.ingest.store import ensure_collection

        ensure_collection(cfg, cfg.embed_model)


def test_query_on_empty_collection_returns_no_hits(cfg):
    from src.ingest.store import ensure_collection, query

    ensure_collection(cfg, cfg.embed_model)
    assert query(cfg=cfg, vector=vec(0)) == []


def test_query_can_filter_by_scheme(cfg):
    from src.ingest.store import ensure_collection, query, upsert_chunks

    chunks = make_chunks(2)
    chunks[1].scheme_key = "elss"
    chunks[1].id = make_chunk_id(chunks[1].source_url, "Fees", 99)

    ensure_collection(cfg, cfg.embed_model)
    upsert_chunks(chunks, [vec(0), vec(1)], cfg)

    hits = query(cfg=cfg, vector=vec(1), top_k=5, scheme_key="elss")
    assert [h.chunk.scheme_key for h in hits] == ["elss"]


def test_delete_scheme_clears_only_that_scheme(cfg):
    from src.ingest.store import delete_scheme, ensure_collection, upsert_chunks

    chunks = make_chunks(2)
    chunks[1].scheme_key = "elss"
    chunks[1].id = make_chunk_id(chunks[1].source_url, "Fees", 99)

    ensure_collection(cfg, cfg.embed_model)
    upsert_chunks(chunks, [vec(0), vec(1)], cfg)

    assert delete_scheme("elss", cfg) == 1
    remaining = delete_scheme("elss", cfg)
    assert remaining == 0

    from src.ingest.store import count

    assert count(cfg) == 1

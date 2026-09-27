"""The only module that talks to ChromaDB.

Keeping every collection name, id scheme and query shape in one file (decision
D12) means the vector store can be swapped for pgvector or FAISS later without
touching retrieval or the pipeline.

Two invariants enforced here:

* **The collection records which embedding model built it.** Writing into a
  collection built with a different model is refused unless ``force=True``.
  A dimension mismatch otherwise fails silently and every query returns noise.
* **Chunk ids are deterministic**, so re-ingesting a page upserts in place
  instead of accumulating duplicates.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import CONFIG
from src.models import Chunk, Hit

log = logging.getLogger(__name__)

# Chroma only accepts scalar metadata values.
_SCALARS = (str, int, float, bool)

# Chroma defaults to squared L2, not cosine. ``score = 1 - distance`` below only
# means cosine similarity if the space is cosine, and all-MiniLM vectors are
# L2-normalised precisely so cosine is the right measure here (decision D3).
DISTANCE_SPACE = "cosine"


class EmbeddingModelMismatch(RuntimeError):
    """Raised when the collection was built with a different embedding model."""


def _collection_metadata(embed_model: str) -> dict:
    return {"hnsw:space": DISTANCE_SPACE, "embed_model": embed_model}


def get_client(cfg=CONFIG):
    """Persistent Chroma client rooted at ``cfg.chroma_dir``."""
    import chromadb
    from chromadb.config import Settings

    path = Path(cfg.chroma_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )


def ensure_collection(cfg=CONFIG, embed_model: str | None = None, force: bool = False):
    """Return the collection, creating it if needed.

    Raises:
        EmbeddingModelMismatch: if an existing collection was built with a
            different model and ``force`` is not set.
    """
    client = get_client(cfg)
    model_name = embed_model or cfg.embed_model

    existing = {c.name for c in client.list_collections()}
    if cfg.collection_name not in existing:
        log.info("creating collection %s (%s)", cfg.collection_name, DISTANCE_SPACE)
        return client.create_collection(
            name=cfg.collection_name, metadata=_collection_metadata(model_name)
        )

    collection = client.get_collection(cfg.collection_name)
    meta = collection.metadata or {}
    stored = meta.get("embed_model")

    if force:
        log.warning("recreating collection %s for model %s", cfg.collection_name, model_name)
        client.delete_collection(cfg.collection_name)
        return client.create_collection(
            name=cfg.collection_name, metadata=_collection_metadata(model_name)
        )

    if stored and stored != model_name:
        raise EmbeddingModelMismatch(
            f"collection {cfg.collection_name!r} was built with {stored!r} but config says "
            f"{model_name!r}. Vectors from different models cannot be compared, so this "
            f"would silently return nonsense. Re-run: python -m src.cli ingest --force"
        )

    space = meta.get("hnsw:space")
    if space != DISTANCE_SPACE:
        # Chroma cannot change the index space in place, and a collection built
        # on the default L2 space would make every score meaningless.
        raise EmbeddingModelMismatch(
            f"collection {cfg.collection_name!r} uses distance space {space!r}, expected "
            f"{DISTANCE_SPACE!r}. Scores would not be comparable. "
            f"Re-run: python -m src.cli ingest --force"
        )
    return collection


def upsert_chunks(
    chunks: list[Chunk],
    vectors: list[list[float]],
    cfg=CONFIG,
    embed_model: str | None = None,
) -> int:
    """Insert or update chunks. Deterministic ids make this idempotent.

    Args:
        embed_model: the model that produced ``vectors``. Must match the
            collection, or the upsert is refused. Defaults to ``cfg.embed_model``,
            which is correct for the normal ingest path but leaves no way to
            write into a collection built by a different model on purpose.
    """
    if not chunks:
        return 0
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

    collection = ensure_collection(cfg, embed_model)
    ids = [c.id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = []
    for chunk in chunks:
        meta = {}
        for key, value in chunk.to_metadata().items():
            if not isinstance(value, _SCALARS):
                value = str(value)
            meta[key] = value
        metadatas.append(meta)

    collection.upsert(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)
    log.info("upserted %d chunks into %s", len(ids), cfg.collection_name)
    return len(ids)


def query(
    cfg=CONFIG,
    vector: list[float] | None = None,
    top_k: int | None = None,
    scheme_key: str | None = None,
    query_texts: list[str] | None = None,
) -> list[Hit]:
    """Nearest-neighbour search.

    Args:
        vector: a pre-computed query embedding (preferred; the caller must use
            the same model that built the collection).
        query_texts: escape hatch used by tests. Not for production paths.
        scheme_key: optional metadata filter, e.g. scope a question to one fund.

    Returns:
        Hits sorted by descending score, where ``score = 1 - cosine_distance``
        clamped to [0, 1]. No score gating happens here -- that is the
        retriever's policy, so the threshold lives in one place.
    """
    collection = ensure_collection(cfg)
    if collection.count() == 0:
        log.warning("collection %s is empty; run: python -m src.cli ingest", cfg.collection_name)
        return []

    k = top_k or cfg.top_k
    where = {"scheme_key": scheme_key} if scheme_key else None

    if vector is not None:
        result = collection.query(
            query_embeddings=[vector], n_results=k, where=where,
            include=["documents", "metadatas", "distances"],
        )
    elif query_texts:
        result = collection.query(
            query_texts=query_texts, n_results=k, where=where,
            include=["documents", "metadatas", "distances"],
        )
    else:
        raise ValueError("query() needs either vector or query_texts")

    return _to_hits(result)


def _to_hits(result: dict) -> list[Hit]:
    hits: list[Hit] = []
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        meta = meta or {}
        score = max(0.0, min(1.0, 1.0 - float(dist)))
        hits.append(
            Hit(
                chunk=Chunk(
                    id=ids[rank] if rank < len(ids) else f"hit-{rank}",
                    text=doc or "",
                    source_url=meta.get("source_url", ""),
                    scheme_key=meta.get("scheme_key", ""),
                    scheme_name=meta.get("scheme_name", ""),
                    category=meta.get("category", ""),
                    section=meta.get("section", ""),
                    chunk_index=int(meta.get("chunk_index", rank) or 0),
                    ingested_at=meta.get("ingested_at", ""),
                    char_start=int(meta.get("char_start", 0) or 0),
                ),
                score=score,
                rank=rank,
            )
        )
    return hits


def open_collection(cfg=CONFIG):
    """Return the collection without the write-path model/space guard.

    Read-only introspection must still work after an intentional model change,
    otherwise ``cli inspect`` breaks exactly when you need it most.
    """
    client = get_client(cfg)
    if cfg.collection_name not in {c.name for c in client.list_collections()}:
        return None
    return client.get_collection(cfg.collection_name)


def count(cfg=CONFIG) -> int:
    collection = open_collection(cfg)
    return 0 if collection is None else collection.count()


def get_chunk(chunk_id: str, cfg=CONFIG) -> Chunk | None:
    """Fetch one chunk by id, or None. Used for verification and debugging."""
    collection = open_collection(cfg)
    if collection is None:
        return None
    result = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
    ids = (result.get("ids") or [])
    if not ids:
        return None
    docs = (result.get("documents") or [None])[0]
    meta = ((result.get("metadatas") or [{}])[0]) or {}
    return Chunk(
        id=ids[0],
        text=docs or "",
        source_url=meta.get("source_url", ""),
        scheme_key=meta.get("scheme_key", ""),
        scheme_name=meta.get("scheme_name", ""),
        category=meta.get("category", ""),
        section=meta.get("section", ""),
        chunk_index=int(meta.get("chunk_index", 0) or 0),
        ingested_at=meta.get("ingested_at", ""),
        char_start=int(meta.get("char_start", 0) or 0),
    )


def delete_scheme(scheme_key: str, cfg=CONFIG) -> int:
    """Drop every chunk for one scheme.

    Called for sources that failed this run: upserts are deterministic, so a
    source that no longer produces chunks (page changed, extraction broke) would
    otherwise leave its stale chunks behind and the bot would keep answering
    from them.
    """
    collection = open_collection(cfg)
    if collection is None:
        return 0
    existing = collection.get(include=["metadatas"])
    stale = [
        cid
        for cid, meta in zip(existing.get("ids") or [], existing.get("metadatas") or [])
        if (meta or {}).get("scheme_key") == scheme_key
    ]
    if stale:
        collection.delete(ids=stale)
        log.info("deleted %d stale chunks for %s", len(stale), scheme_key)
    return len(stale)


def reset(cfg=CONFIG) -> None:
    """Delete the collection entirely. Used by ``ingest --force``."""
    client = get_client(cfg)
    if cfg.collection_name in {c.name for c in client.list_collections()}:
        client.delete_collection(cfg.collection_name)
        log.info("deleted collection %s", cfg.collection_name)


def stats(cfg=CONFIG) -> dict:
    """Small summary for ``cli inspect``."""
    collection = open_collection(cfg)
    if collection is None:
        return {"collection": cfg.collection_name, "exists": False, "count": 0}
    meta = collection.metadata or {}
    return {
        "collection": cfg.collection_name,
        "exists": True,
        "count": collection.count(),
        "embed_model": meta.get("embed_model"),
        "distance_space": meta.get("hnsw:space"),
    }

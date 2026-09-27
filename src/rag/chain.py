"""The single online entrypoint. Both the CLI and the Streamlit UI call answer().

The step order in docs/architecture.md section 5.9 is deliberate and load-bearing:

    1. detect_pii    free, offline
    2. is_advice     free, offline
    3. search        paid, offline (embeddings are local)
    4. generate      the only step that costs money or touches the network
    5. validate      free, deterministic

Steps 1-3 mean a PII question, an advice question, and an out-of-corpus question
all cost zero API calls. In the normal case that is most of the traffic.

Every failure after step 3 becomes a normal Answer, never an exception. The UI
must never render a traceback.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import CONFIG, Config
from src.models import Answer, Hit
from src.rag import citations, guards, prompts
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)

# Built lazily: importing this module must not load the embedding model, because
# the guard tests import guards/chain-adjacent code without needing the corpus.
_retriever: Retriever | None = None
_generator = None


def _get_retriever(cfg: Config) -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(cfg)
    return _retriever


def _get_generator(cfg: Config):
    global _generator
    if _generator is None:
        from src.rag.generator import Generator

        _generator = Generator(cfg)
    return _generator


def _ingested_at(cfg: Config) -> str:
    """Read the manifest, never the model.

    A model-told timestamp would drift between answers and could not be trusted
    by the UI's "Last updated from sources" line.
    """
    try:
        manifest = json.loads(Path(cfg.manifest_path).read_text())
        return manifest.get("ingested_at", "unknown")
    except (OSError, ValueError) as exc:
        logger.warning("could not read manifest: %s", exc)
        return "unknown"


def reset_caches() -> None:
    """Drop the memoised retriever/generator. Used by tests."""
    global _retriever, _generator
    _retriever = None
    _generator = None


def answer(query: str, cfg: Config = None) -> Answer:
    """Answer one question end to end. Never raises."""
    cfg = cfg or CONFIG
    query = (query or "").strip()

    if not query:
        return Answer(
            text=prompts.NO_MATCH, refused=True, refusal_kind="no_match",
            last_updated=_ingested_at(cfg),
        )

    # --- Step 1: PII. Log the KIND only, never the raw string. --------------
    pii_kind = guards.detect_pii(query)
    if pii_kind:
        logger.info("guard: pii detected (kind=%s); query not logged", pii_kind)
        return Answer(
            text=prompts.PII_REFUSAL, refused=True, refusal_kind="pii",
            last_updated=_ingested_at(cfg),
        )

    # --- Step 2: advice ---------------------------------------------------
    if guards.is_advice(query):
        logger.info("guard: advice question refused")
        return Answer(
            text=prompts.ADVICE_REFUSAL, refused=True, refusal_kind="advice",
            last_updated=_ingested_at(cfg),
        )

    # --- Step 3: retrieve -------------------------------------------------
    try:
        hits: list[Hit] = _get_retriever(cfg).search(query)
    except Exception as exc:  # noqa: BLE001
        logger.error("retrieval failed: %s: %s", type(exc).__name__, exc)
        return Answer(
            text=prompts.NO_MATCH, refused=True, refusal_kind="no_match",
            last_updated=_ingested_at(cfg),
        )

    if not hits:
        logger.info("retrieval: no hits passed the gate")
        return Answer(
            text=prompts.NO_MATCH, refused=True, refusal_kind="no_match",
            last_updated=_ingested_at(cfg),
        )

    # --- Step 4: generate. Any failure becomes NO_MATCH, not a traceback. --
    try:
        raw = _get_generator(cfg).answer(query, hits)
    except Exception as exc:  # noqa: BLE001
        logger.error("generation failed, falling back to NO_MATCH: %s: %s", type(exc).__name__, exc)
        return Answer(
            text=prompts.NO_MATCH, refused=True, refusal_kind="no_match",
            last_updated=_ingested_at(cfg), hits=hits,
        )

    # --- Step 5: enforce G1 ------------------------------------------------
    try:
        text, citation_url, truncated = citations.validate(raw, hits)
    except Exception as exc:  # noqa: BLE001
        logger.error("citation validation failed: %s: %s", type(exc).__name__, exc)
        return Answer(
            text=prompts.NO_MATCH, refused=True, refusal_kind="no_match",
            last_updated=_ingested_at(cfg), hits=hits,
        )

    # A validate() that downgraded the answer to NO_MATCH is still a refusal.
    # Reporting refused=False here would make the UI render a refusal string as
    # a successful answer, with no citation and no explanation.
    downgraded = text == prompts.NO_MATCH

    return Answer(
        text=text,
        citation_url=citation_url,
        last_updated=_ingested_at(cfg),
        hits=hits,
        truncated=truncated,
        refused=downgraded,
        refusal_kind="no_match" if downgraded else None,
    )

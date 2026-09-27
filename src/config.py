"""Central configuration. Every tunable in the pipeline is read from here exactly once.

Import ``CONFIG`` rather than constructing it, so that tests and CLI runs share one
instance. Missing values fall back to the defaults documented in
``docs/implementation.md`` section 0.4 -- a missing ``.env`` must never stop the
ingest phase from working.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# A single env file. An earlier revision also loaded a separate `llm.env` for
# credentials; it was removed so there is exactly one place to look.
#
# Worth remembering if this ever grows a second file: a later load_dotenv with
# override=True wins on conflict, so an EMPTY line in the second file silently
# blanks a working value in the first. That bug cost real debugging time here --
# an empty GROQ_API_KEY= in llm.env was masking a valid key in .env.
load_dotenv(PROJECT_ROOT / ".env")


def _env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    """Immutable snapshot of the runtime configuration."""

    # -- LLM (Phase 4) -----------------------------------------------------
    # The answer layer talks to Groq by default. Groq exposes an
    # OpenAI-compatible /chat/completions endpoint, which is why no second SDK
    # is needed: requests is already a dependency for the fetcher.
    llm_provider: str = "groq"
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Retained so the Anthropic path still works, and so a provider switch does
    # not require deleting a working integration.
    anthropic_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"

    # -- Vector store (Phase 2) --------------------------------------------
    chroma_dir: str = "./chroma_db"
    collection_name: str = "hdfc_mf_faq"

    # -- Embedding (Phase 2) ----------------------------------------------
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # -- Retrieval (Phase 3) -----------------------------------------------
    top_k: int = 4
    # Primary answerability gate. Tuned in Phase 3: the answerable set's lowest
    # top-1 score is 0.445 and the unanswerable set's highest is 0.561, so NO
    # value of this alone separates them. See src/rag/lexical.py and
    # docs/architecture.md section 5.6.
    min_similarity: float = 0.30
    # Second, orthogonal gate: IDF-weighted lexical coverage. This is what
    # actually separates the sets (answerable min 0.637 vs refuse max 0.494).
    min_lexical_coverage: float = 0.55

    # -- Chunking (Phase 2) -------------------------------------------------
    # all-MiniLM-L6-v2 truncates at 256 wordpiece tokens, so chunks must stay well
    # under that or their tail is silently dropped at embedding time. See
    # docs/implementation.md section 0.1.
    chunk_size_tokens: int = 200
    chunk_hard_cap_tokens: int = 240
    chunk_overlap_tokens: int = 25

    # -- Fetching (Phase 1) ------------------------------------------------
    request_delay_seconds: float = 1.5
    request_timeout_seconds: float = 30.0
    fetch_retries: int = 3
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )

    # -- Extraction (Phase 1) ----------------------------------------------
    # Drops nav crumbs ("Home", "Share"), and -- more importantly -- the
    # "other schemes by this fund manager" carousel and loan promos, which are
    # 20-50 char lines that would otherwise pollute retrieval (a query about
    # HDFC Balanced Advantage would match a block merely *listing* it).
    # Short factual values (expense ratio, riskometer, lock-in) do not depend on
    # prose blocks: they come from the structured payload in ingest/structured.py.
    min_block_chars: int = 40
    min_chunk_tokens: int = 30

    # -- Paths -------------------------------------------------------------
    sources_path: str = str(PROJECT_ROOT / "data" / "sources.md")
    documents_dir: str = str(PROJECT_ROOT / "data" / "documents")
    manifest_path: str = str(PROJECT_ROOT / "data" / "index_manifest.json")
    cache_dir: str = str(PROJECT_ROOT / "cache")
    raw_html_dir: str = str(PROJECT_ROOT / "cache" / "raw_html")

    def raw_html_path(self, scheme_key: str) -> Path:
        return Path(self.raw_html_dir) / f"{scheme_key}.html"

    @property
    def llm_api_key(self) -> str:
        """The credential for the active provider.

        Callers must not read ``groq_api_key`` or ``anthropic_api_key`` directly:
        after the Phase 6 provider switch that would report the answer layer as
        offline whenever the key was set for the other provider.
        """
        if self.llm_provider == "anthropic":
            return self.anthropic_api_key
        return self.groq_api_key

    @property
    def llm_key_env_var(self) -> str:
        """Name of the env var that holds the key, for setup instructions."""
        return "ANTHROPIC_API_KEY" if self.llm_provider == "anthropic" else "GROQ_API_KEY"

    def document_path(self, scheme_key: str) -> Path:
        return Path(self.documents_dir) / f"{scheme_key}.json"


def _resolve_model() -> str:
    """Pick the generation model, most specific source first.

    ``LLM_MODEL`` beats ``GROQ_MODEL`` beats a built-in default. The
    provider-specific name is honoured because the key and the model are usually
    set as a pair, and a stale generic default must not silently override the
    model a user deliberately chose in ``.env``.
    """
    for var in ("LLM_MODEL", "GROQ_MODEL", "ANTHROPIC_MODEL"):
        value = _env_str(var, "").strip()
        if value:
            return value
    return "llama-3.3-70b-versatile"


def _load() -> Config:
    return Config(
        llm_provider=_env_str("LLM_PROVIDER", "groq").strip().lower(),
        groq_api_key=_env_str("GROQ_API_KEY", ""),
        groq_base_url=_env_str("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        anthropic_api_key=_env_str("ANTHROPIC_API_KEY", ""),
        llm_model=_resolve_model(),
        chroma_dir=_env_str("CHROMA_DIR", "./chroma_db"),
        collection_name=_env_str("COLLECTION_NAME", "hdfc_mf_faq"),
        embed_model=_env_str("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        top_k=_env_int("TOP_K", 4),
        min_similarity=_env_float("MIN_SIMILARITY", 0.30),
        min_lexical_coverage=_env_float("MIN_LEXICAL_COVERAGE", 0.55),
        chunk_size_tokens=_env_int("CHUNK_SIZE_TOKENS", 200),
        chunk_hard_cap_tokens=_env_int("CHUNK_HARD_CAP_TOKENS", 240),
        chunk_overlap_tokens=_env_int("CHUNK_OVERLAP_TOKENS", 25),
        request_delay_seconds=_env_float("REQUEST_DELAY_SECONDS", 1.5),
        request_timeout_seconds=_env_float("REQUEST_TIMEOUT_SECONDS", 30.0),
        fetch_retries=_env_int("FETCH_RETRIES", 3),
        user_agent=_env_str("USER_AGENT", Config.user_agent),
    )


CONFIG = _load()

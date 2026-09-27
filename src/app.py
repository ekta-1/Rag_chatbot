"""Streamlit demo surface. Run: ``streamlit run src/app.py``.

This file renders and calls. It contains no business logic on purpose: every
decision about what is allowed to reach the model lives in chain.py, every
refusal string lives in prompts.py, and the disclaimer is imported rather than
retyped so it cannot drift from the one the tests assert on.

If you find yourself writing an ``if`` about expense ratios or exit loads in
this file, it belongs in src/rag/chain.py.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

# `streamlit run src/app.py` puts src/ on sys.path, not the project root, so
# `from src.config import ...` fails without this. The CLI does not need it
# because `python -m src.cli` already places the root on the path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG  # noqa: E402
from src.rag.prompts import DISCLAIMER  # noqa: E402

# This module renders untrusted model output and a local index. Streamlit's own
# error surface is not what a demo audience should see, so the log is quiet.
logging.getLogger("src.rag").setLevel(logging.WARNING)

SOURCES_MD = PROJECT_ROOT / "data" / "sources.md"

# The three chips are PRD section 4 representative queries (items 1, 2 and 4).
# They are deliberately all *answerable*: a chip that returns "I couldn't find
# that" reads as a broken app to anyone who clicks it first, and the
# out-of-corpus refusal is a demo beat better delivered deliberately -- it is
# typed in the demo script rather than sitting in the toolbar.
EXAMPLE_QUESTIONS = [
    "What is the expense ratio of HDFC Large Cap Fund?",
    "Is there an exit load on HDFC Flexi Cap Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
]

NO_INDEX_MESSAGE = (
    "The index has not been built yet, so there is nothing to search.\n\n"
    "Run this in a terminal, then reload:\n\n"
    "    python -m src.cli ingest"
)


# --------------------------------------------------------------------------
# Small helpers. Presentation only.
# --------------------------------------------------------------------------


def _read_ingested_at() -> str:
    import json

    try:
        return json.loads(Path(CONFIG.manifest_path).read_text()).get("ingested_at", "unknown")
    except (OSError, ValueError):
        return "unknown"


def _read_sources() -> list[tuple[str, str]]:
    """Parse the 5 corpus schemes from data/sources.md at runtime.

    Read from the file rather than hardcoded, so re-ingesting a different corpus
    updates the sidebar without a code change.
    """
    try:
        rows: list[tuple[str, str]] = []
        for line in SOURCES_MD.read_text().splitlines():
            if not line.startswith("|") or "---" in line or "scheme_key" in line:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) == 4:
                rows.append((cells[2], cells[3]))
        return rows
    except OSError:
        return []


def _index_is_ready() -> bool:
    """Is there a usable collection? The empty-index guard, without importing
    sentence_transformers on a page that may not need it."""
    try:
        from src.ingest.store import open_collection

        return open_collection().count() > 0
    except Exception:  # noqa: BLE001 - any failure means "not ready"
        return False


def _get_answered(question: str):
    """The only call into business logic in this file."""
    from src.rag.chain import answer

    return answer(question)


def _render_sidebar(ingested_at: str) -> None:
    with st.sidebar:
        st.subheader("Corpus scope")
        sources = _read_sources()
        if sources:
            st.caption(f"{len(sources)} public HDFC MF scheme pages (Direct – Growth)")
            for name, url in sources:
                st.markdown(f"- [{name}]({url})")
        else:
            st.caption("data/sources.md not found.")

        st.divider()
        st.subheader("Retrieval settings")
        st.text(f"Last indexed: {ingested_at}")
        st.text(f"Embedding model: {CONFIG.embed_model}")
        st.text(f"Top K: {CONFIG.top_k}")
        st.text(f"Min similarity: {CONFIG.min_similarity}")
        st.text(f"Min lexical coverage: {CONFIG.min_lexical_coverage}")
        st.caption("Both gates must pass. See docs/architecture.md 5.6.")

        st.divider()
        st.caption(DISCLAIMER)

        if st.button("Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()


def _render_hits(result) -> None:
    """The debug panel. Showing retrieval is the most convincing part of a demo:
    it turns 'the model said this' into 'here is the exact line it read'."""
    if not result.hits:
        return
    with st.expander("🔍 Retrieved chunks (debug)"):
        st.caption(
            f"{len(result.hits)} chunk(s) passed both gates. "
            "score is raw cosine similarity; text is exactly what was retrieved."
        )
        for hit in result.hits:
            st.markdown(
                f"**{hit.rank}.** score `{hit.score:.3f}` · "
                f"{hit.chunk.scheme_name} · *{hit.chunk.section}*"
            )
            st.text(hit.chunk.text)
            st.caption(hit.chunk.source_url)
            st.divider()


def _render_assistant(result) -> None:
    if result.refused and result.refusal_kind:
        st.caption(f"refused · {result.refusal_kind}")
    st.markdown(result.text)
    if result.citation_url:
        st.markdown(f"🔗 [Source page]({result.citation_url})")
    if result.truncated:
        st.caption("(trimmed to 3 sentences)")
    st.caption(f"Last updated from sources: {result.last_updated}")
    _render_hits(result)


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="HDFC MF Facts Assistant", layout="centered", initial_sidebar_state="expanded"
    )

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending", None)

    ingested_at = _read_ingested_at()
    index_ready = _index_is_ready()

    # Sidebar first, and on every path: corpus scope, settings, clear-chat and a
    # second copy of the disclaimer are needed whether or not the index exists.
    _render_sidebar(ingested_at)

    st.title("HDFC Mutual Fund Facts Assistant")
    st.markdown("**Facts-only assistant — 5 schemes, public pages only.**")
    st.info(DISCLAIMER)

    if not index_ready:
        st.warning("Index not built.")
        st.code("python -m src.cli ingest", language="bash")
        st.caption("Then reload this page.")
        return

    if not CONFIG.llm_api_key:
        st.warning(
            f"{CONFIG.llm_key_env_var} is not set, so the answer layer is offline "
            f"(provider: {CONFIG.llm_provider}). "
            "Guards and retrieval still work; generated answers will be declined. "
            "Add it to .env and restart."
        )

    # Example chips. Set pending and rerun so the question is sent exactly once,
    # even if a rerun happens before the chat turn is drawn.
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, question in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(question, use_container_width=True):
            st.session_state["pending"] = question
            st.rerun()

    if st.session_state["messages"]:
        for role, payload in st.session_state["messages"]:
            with st.chat_message(role):
                if role == "user":
                    st.markdown(payload)
                else:
                    _render_assistant(payload)

    typed = st.chat_input("Ask a fact about an HDFC MF scheme, e.g. 'exit load on HDFC Flexi Cap Fund'")
    question = st.session_state.pop("pending", None) or typed

    if question:
        st.session_state["messages"].append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                result = _get_answered(question)
            except Exception:  # noqa: BLE001 - never surface a traceback in the UI
                st.warning("Something went wrong answering that. Try again.")
            else:
                _render_assistant(result)
                st.session_state["messages"].append(("assistant", result))

    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()

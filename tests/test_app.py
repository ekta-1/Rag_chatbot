"""Streamlit UI tests, driven by Streamlit's own AppTest harness.

A9 ("disclaimer visible on first render") is an exit criterion for Phase 5, and
"the page loads without an exception" is the one thing a browser cannot tell you
programmatically. AppTest runs the real script in-process and exposes the
rendered element tree, so these assert on what a user would actually see.

The three bugs these caught while building, none of which were visible from a
server-starting-cleanly check:
  - the sidebar was only rendered on the empty-index path
  - st.text() was called with two positional args, which raised on every render
  - `streamlit run` does not put the project root on sys.path

Marked ``integration`` because they need the built index and the embedding
model loaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config import CONFIG
from src.rag.prompts import DISCLAIMER

# Absolute: AppTest resolves the path against the process cwd, which is not
# guaranteed to be the project root under pytest.
APP = str(Path(__file__).resolve().parent.parent / "src" / "app.py")
DEFAULT_TIMEOUT = 180


def run_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=DEFAULT_TIMEOUT).run()


@pytest.fixture(scope="module")
def app() -> AppTest:
    return run_app()


# --- First render ---------------------------------------------------------

@pytest.mark.integration
def test_page_renders_with_no_exception(app):
    assert not app.exception, f"app raised on first render: {[e.value for e in app.exception]}"


@pytest.mark.integration
def test_a9_disclaimer_visible_on_first_render(app):
    """A9. Must be present in the main area without scrolling or interaction."""
    assert any(DISCLAIMER in i.value for i in app.info), (
        "A9: disclaimer missing from the main area"
    )


@pytest.mark.integration
def test_welcome_line_present(app):
    text = " ".join(m.value for m in app.markdown)
    assert "5 schemes, public pages only" in text


@pytest.mark.integration
def test_disclaimer_also_in_sidebar_and_footer(app):
    """Rendered in three places so it cannot be scrolled past."""
    sidebar_text = " ".join(c.value for c in app.sidebar.caption)
    main_captions = " ".join(c.value for c in app.caption)
    assert DISCLAIMER in sidebar_text, "disclaimer missing from sidebar"
    assert main_captions.count(DISCLAIMER) >= 1, "disclaimer missing from footer"


# --- The three example buttons -------------------------------------------

@pytest.mark.integration
def test_three_example_buttons_exist(app):
    labels = [b.label for b in app.button if b.label != "Clear chat"]
    assert len(labels) == 3, f"expected 3 example chips, got {labels}"
    assert any("expense ratio" in l.lower() for l in labels)
    assert any("exit load" in l.lower() for l in labels)
    assert any("lock-in" in l.lower() for l in labels)


# PRD 5 (implementation.md Phase 5 point 4) requires the chips to come from the
# PRD section 4 set, which is six *answerable* queries. An earlier revision used
# "What is HDFC Bank's FD interest rate?" as a chip; that query is out of
# corpus, so the third button answered "I couldn't find that" and read as a
# broken app to anyone who happened to click it. The out-of-corpus refusal is
# still demonstrated, by typing it -- see docs/demo_script.md.
OUT_OF_CORPUS_MARKERS = ("FD interest", "capital gains statement", "best for my portfolio")


def test_no_example_chip_is_a_known_out_of_corpus_query(app):
    labels = [b.label for b in app.button if b.label != "Clear chat"]
    for label in labels:
        for marker in OUT_OF_CORPUS_MARKERS:
            assert marker.lower() not in label.lower(), (
                f"chip {label!r} is a known out-of-corpus query; chips should be "
                "answerable PRD section 4 questions"
            )


@pytest.mark.integration
@pytest.mark.parametrize("index", [0, 1, 2])
def test_each_example_button_sends_its_question(index):
    at = run_app()
    at.button[index].click().run()
    assert not at.exception
    # a user turn and an assistant turn
    assert len(at.chat_message) == 2
    assert at.chat_message[0].name == "user"
    assert at.chat_message[1].name == "assistant"


# --- Sidebar --------------------------------------------------------------

@pytest.mark.integration
def test_sidebar_lists_all_five_schemes(app):
    links = [m.value for m in app.sidebar.markdown if "groww.in" in m.value]
    assert len(links) == 5, f"expected 5 corpus links, got {len(links)}"


@pytest.mark.integration
def test_sidebar_shows_settings(app):
    text = " ".join(t.value for t in app.sidebar.text)
    assert CONFIG.embed_model in text
    assert f"Top K: {CONFIG.top_k}" in text
    assert f"Min similarity: {CONFIG.min_similarity}" in text
    assert "Last indexed:" in text


@pytest.mark.integration
def test_clear_chat_button_exists(app):
    assert "Clear chat" in [b.label for b in app.sidebar.button]


# --- The debug panel ------------------------------------------------------

@pytest.mark.integration
def test_debug_panel_shows_hits_with_scores():
    """The most valuable thing in a demo: retrieval made visible."""
    at = run_app()
    at.chat_input[0].set_value("expense ratio of HDFC Large Cap Fund").run()
    assert not at.exception

    labels = [e.label for e in at.expander]
    assert any("Retrieved chunks" in l for l in labels), f"no debug expander: {labels}"

    expander = next(e for e in at.expander if "Retrieved chunks" in e.label)
    text = " ".join(
        [m.value for m in expander.markdown] + [t.value for t in expander.text]
    )
    assert "score" in text
    assert "HDFC Large Cap Fund" in text


@pytest.mark.integration
def test_assistant_turn_shows_last_updated():
    at = run_app()
    at.chat_input[0].set_value("expense ratio of HDFC Large Cap Fund").run()
    captions = " ".join(c.value for c in at.caption)
    assert "Last updated from sources:" in captions


# --- Out-of-corpus path ---------------------------------------------------

@pytest.mark.integration
def test_out_of_corpus_question_refuses():
    # Typed, not clicked. This used to press button[2] back when that was the
    # out-of-corpus chip, which made the test pass for the wrong reason: it was
    # really asserting that chip 3 was the FD query. Typing it tests the
    # behaviour the name claims.
    at = run_app()
    at.chat_input[0].set_value("What is HDFC Bank's FD interest rate?").run()
    assert not at.exception
    body = at.chat_message[1].markdown[0].value
    assert "couldn't find that" in body.lower()


@pytest.mark.integration
def test_no_traceback_reaches_the_ui():
    """A generation failure must not become a raw exception in the page."""
    at = run_app()
    at.chat_input[0].set_value("expense ratio of HDFC Large Cap Fund").run()
    assert not at.exception
    body = " ".join(m.value for m in at.markdown)
    assert "Traceback" not in body
    assert "Error" not in body or "went wrong" in body

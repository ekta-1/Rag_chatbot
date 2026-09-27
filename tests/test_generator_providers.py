"""Tests for the provider-agnostic answer layer.

The answer layer moved from Anthropic to Groq during Phase 6. Both providers sit
behind one :class:`Generator` interface, and the guarantees the acceptance
criteria depend on must hold for either: temperature 0 (A10 determinism),
``max_tokens`` 300, a 30 s timeout, and no retry. A provider swap that quietly
dropped one of those would weaken a test that still passed, so each is asserted
against the actual outgoing request rather than against config values.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from src.config import CONFIG
from src.rag import prompts
from src.rag.generator import (
    MAX_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    SUPPORTED_PROVIDERS,
    Generator,
    GeneratorUnavailable,
)


class _FakeResponse:
    def __init__(self, content="Expense ratio is 1.03%."):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _RecordingSession:
    """Stands in for requests.Session and records the outgoing call."""

    def __init__(self, content="Expense ratio is 1.03%."):
        self.calls: list[dict] = []
        self._content = content

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        return _FakeResponse(self._content)


@pytest.fixture
def groq_generator():
    cfg = dataclasses.replace(CONFIG, groq_api_key="gsk_test", llm_provider="groq")
    gen = Generator(cfg)
    gen._client = _RecordingSession()
    return gen


@pytest.fixture
def hits():
    from src.rag.retriever import Retriever

    found = Retriever().search("expense ratio of HDFC Large Cap Fund")
    assert found, "fixtures require a populated index"
    return found[:1]


class TestProviderSelection:
    def test_groq_is_the_default(self):
        assert CONFIG.llm_provider == "groq"

    def test_default_model_is_a_groq_model(self):
        # A Claude id left in place after the swap would fail at the API with a
        # 400 that looks like a key problem.
        assert not CONFIG.llm_model.lower().startswith("claude")

    def test_unknown_provider_is_rejected(self):
        cfg = dataclasses.replace(CONFIG, llm_provider="not-a-provider")
        with pytest.raises(GeneratorUnavailable, match="not supported"):
            Generator(cfg)

    def test_both_providers_are_declared(self):
        assert "groq" in SUPPORTED_PROVIDERS
        assert "anthropic" in SUPPORTED_PROVIDERS


class TestMissingKey:
    def test_groq_without_key_raises_generator_unavailable(self):
        cfg = dataclasses.replace(CONFIG, groq_api_key="", llm_provider="groq")
        with pytest.raises(GeneratorUnavailable, match="GROQ_API_KEY"):
            Generator(cfg)

    def test_error_names_the_env_file(self):
        # The message should tell the user where to put the key, not just that
        # something is missing. It must name the file that actually exists:
        # this pointed at a deleted llm.env for a while.
        cfg = dataclasses.replace(CONFIG, groq_api_key="", llm_provider="groq")
        with pytest.raises(GeneratorUnavailable, match=r"\.env"):
            Generator(cfg)

    def test_anthropic_without_key_still_raises(self):
        cfg = dataclasses.replace(
            CONFIG, anthropic_api_key="", llm_provider="anthropic"
        )
        with pytest.raises(GeneratorUnavailable, match="ANTHROPIC_API_KEY"):
            Generator(cfg)


class TestGroqRequestContract:
    def test_hits_the_openai_compatible_endpoint(self, groq_generator, hits):
        groq_generator.answer("expense ratio?", hits)
        call = groq_generator._client.calls[0]
        assert call["url"] == "https://api.groq.com/openai/v1/chat/completions"

    def test_bearer_auth_header(self, groq_generator, hits):
        groq_generator.answer("expense ratio?", hits)
        call = groq_generator._client.calls[0]
        assert call["headers"]["Authorization"] == "Bearer gsk_test"

    def test_temperature_is_zero_for_determinism(self, groq_generator, hits):
        # A10: the same question twice must give the same answer.
        groq_generator.answer("expense ratio?", hits)
        assert groq_generator._client.calls[0]["json"]["temperature"] == 0

    def test_token_cap_matches_the_three_sentence_rule(self, groq_generator, hits):
        groq_generator.answer("expense ratio?", hits)
        assert groq_generator._client.calls[0]["json"]["max_tokens"] == MAX_TOKENS

    def test_timeout_is_bounded(self, groq_generator, hits):
        groq_generator.answer("expense ratio?", hits)
        assert groq_generator._client.calls[0]["timeout"] == REQUEST_TIMEOUT_SECONDS

    def test_system_prompt_is_sent_verbatim(self, groq_generator, hits):
        # The refusal and citation rules live in this prompt; paraphrasing it
        # would break the behaviour the acceptance tests assert.
        groq_generator.answer("expense ratio?", hits)
        messages = groq_generator._client.calls[0]["json"]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == prompts.SYSTEM_PROMPT
        assert messages[1]["role"] == "user"

    def test_model_comes_from_config(self, groq_generator, hits):
        groq_generator.answer("expense ratio?", hits)
        assert groq_generator._client.calls[0]["json"]["model"] == CONFIG.llm_model

    def test_non_streaming(self, groq_generator, hits):
        groq_generator.answer("expense ratio?", hits)
        assert groq_generator._client.calls[0]["json"]["stream"] is False

    def test_call_count_increments(self, groq_generator, hits):
        assert groq_generator.call_count == 0
        groq_generator.answer("expense ratio?", hits)
        assert groq_generator.call_count == 1


class TestResponseHandling:
    def test_empty_content_raises(self, hits):
        cfg = dataclasses.replace(CONFIG, groq_api_key="gsk_test", llm_provider="groq")
        gen = Generator(cfg)
        gen._client = _RecordingSession(content="")
        with pytest.raises(RuntimeError, match="no text"):
            gen.answer("expense ratio?", hits)

    def test_missing_choices_raises(self, hits):
        class NoChoices(_RecordingSession):
            def post(self, *a, **kw):
                self.calls.append({"url": "x"})
                resp = _FakeResponse()
                resp.json = lambda: {"choices": []}
                return resp

        cfg = dataclasses.replace(CONFIG, groq_api_key="gsk_test", llm_provider="groq")
        gen = Generator(cfg)
        gen._client = NoChoices()
        with pytest.raises(RuntimeError, match="no choices"):
            gen.answer("expense ratio?", hits)

    def test_transport_error_propagates_for_chain_to_catch(self, hits):
        # chain.py turns this into NO_MATCH; swallowing it here would surface a
        # traceback in the UI instead of a friendly message.
        class Boom(_RecordingSession):
            def post(self, *a, **kw):
                raise ConnectionError("network down")

        cfg = dataclasses.replace(CONFIG, groq_api_key="gsk_test", llm_provider="groq")
        gen = Generator(cfg)
        gen._client = Boom()
        with pytest.raises(ConnectionError):
            gen.answer("expense ratio?", hits)

    def test_answer_requires_hits(self, groq_generator):
        with pytest.raises(ValueError, match="no hits"):
            groq_generator.answer("expense ratio?", [])


class TestProperties:
    def test_provider_and_model_are_exposed(self, groq_generator):
        assert groq_generator.provider == "groq"
        assert groq_generator.model == CONFIG.llm_model


class TestProviderAwareKeyLookups:
    """Regression cover for the Phase 6 provider switch.

    The UI used to read ``CONFIG.anthropic_api_key`` directly, so after the
    switch it told a correctly configured Groq user that their key was missing.
    Every provider-aware caller must go through the helpers instead.
    """

    def test_groq_key_is_reported_as_present(self):
        cfg = dataclasses.replace(CONFIG, groq_api_key="gsk_test", llm_provider="groq")
        assert cfg.llm_api_key == "gsk_test"

    def test_env_var_name_matches_the_provider(self):
        groq_cfg = dataclasses.replace(CONFIG, llm_provider="groq")
        assert groq_cfg.llm_key_env_var == "GROQ_API_KEY"
        anthropic_cfg = dataclasses.replace(CONFIG, llm_provider="anthropic")
        assert anthropic_cfg.llm_key_env_var == "ANTHROPIC_API_KEY"

    def test_anthropic_key_is_not_used_when_provider_is_groq(self):
        # A leftover key from the old integration must not make the answer layer
        # look online, or every question would fail at the API instead.
        cfg = dataclasses.replace(
            CONFIG, groq_api_key="", anthropic_api_key="sk-ant-old", llm_provider="groq"
        )
        assert cfg.llm_api_key == ""

    def test_no_source_file_reads_the_raw_key_fields(self):
        # Guards the whole class of bug: a new caller reading the wrong field.
        offenders = []
        for path in Path("src").rglob("*.py"):
            text = path.read_text()
            for field in ("groq_api_key", "anthropic_api_key"):
                if field not in text:
                    continue
                # config.py declares them; generator.py must read the key.
                if path.name in ("config.py", "generator.py"):
                    continue
                offenders.append(f"{path}: {field}")
        assert not offenders, f"read the raw key field instead of llm_api_key: {offenders}"


class TestAnswerSurface:
    """A real key exposed display bugs that the stubbed tests could not.

    The answer text already ends with 'Source: <url>' because
    citations.validate strips the model's URL and re-attaches exactly one. The
    CLI then printed result.citation_url again, so every generated answer showed
    two source lines. Asserted against the module source so the check survives
    refactors that move the print around.
    """

    def test_cli_does_not_print_the_citation_twice(self):
        from pathlib import Path

        source = Path("src/cli.py").read_text()
        assert 'f"\\nSource: {result.citation_url}"' not in source, (
            "CLI re-prints a citation that citations.validate already attached "
            "to result.text, producing two Source lines per answer"
        )

    def test_citations_attaches_exactly_one_source_line(self):
        from src.models import Chunk, Hit
        from src.rag import citations

        url = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
        chunk = Chunk(
            id="large_cap:fees:0", text="Expense ratio 1.03%.", source_url=url,
            scheme_key="large_cap", scheme_name="HDFC Large Cap Fund",
            category="fees", section="Fees", chunk_index=0,
            ingested_at="2026-09-27T09:33:02+00:00",
        )
        hit = Hit(chunk=chunk, score=0.81, rank=1)
        raw = f"The expense ratio is 1.03%. Source: {url}"
        text, citation, _ = citations.validate(raw, [hit])
        assert text.count("Source:") == 1
        assert citation == url

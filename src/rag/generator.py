"""One LLM call per question: temperature 0, no retry loop, hard token cap.

The retry policy is deliberately empty. A retry inside a live demo reads as a
hang, and a demo that silently waits 60 s is worse than one that returns
NO_MATCH and lets the user re-ask. Raising here is the contract: chain.py owns
the user-facing fallback, so the failure mode is a friendly message plus a log
line, never a traceback in the UI.

Two providers are supported behind one interface:

* ``groq`` (default) -- Groq's OpenAI-compatible ``/chat/completions``. Called
  with ``requests`` rather than the ``openai`` SDK: it is one endpoint, and
  ``requests`` is already a dependency for the fetcher, so this adds no package.
* ``anthropic`` -- the original Messages API path, kept working.

The contract is identical for both: temperature 0 (A10 determinism),
``max_tokens`` 300, a 30 s timeout, and no SDK-level retry.
"""

from __future__ import annotations

import logging

import requests

from src.config import Config
from src.models import Hit
from src.rag import prompts

logger = logging.getLogger(__name__)

# A stuck call must not freeze the UI. 30 s is well above p99 for a 300-token
# response and well below "the demo looks broken".
REQUEST_TIMEOUT_SECONDS = 30.0
MAX_TOKENS = 300

SUPPORTED_PROVIDERS = ("groq", "anthropic")


class GeneratorUnavailable(RuntimeError):
    """No API key configured, or the SDK is not installed."""


class Generator:
    """Thin wrapper over a chat-completions style API.

    The client is created once and reused; instantiating it per call would add
    TLS setup to every question.
    """

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._provider = (cfg.llm_provider or "groq").strip().lower()
        self._model = cfg.llm_model
        self.call_count = 0

        if self._provider not in SUPPORTED_PROVIDERS:
            raise GeneratorUnavailable(
                f"LLM_PROVIDER={self._provider!r} is not supported. "
                f"Use one of: {', '.join(SUPPORTED_PROVIDERS)}"
            )

        if self._provider == "groq":
            if not cfg.groq_api_key:
                raise GeneratorUnavailable(
                    "GROQ_API_KEY is not set. Add it to .env before using the "
                    "answer layer."
                )
            self._client = requests.Session()
            self._base_url = cfg.groq_base_url.rstrip("/")
        else:
            if not cfg.anthropic_api_key:
                raise GeneratorUnavailable(
                    "ANTHROPIC_API_KEY is not set. Add it to .env, "
                    "before using the answer layer."
                )
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - environment issue
                raise GeneratorUnavailable(
                    "The 'anthropic' package is not installed. Run: pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic(
                api_key=cfg.anthropic_api_key,
                timeout=REQUEST_TIMEOUT_SECONDS,
                max_retries=0,  # we implement the no-retry policy explicitly
            )
            self._base_url = ""

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    def answer(self, question: str, hits: list[Hit]) -> str:
        """Return the model's text for one question.

        Raises:
            ValueError: no hits were retrieved (the chain should have caught it).
            requests.HTTPError / anthropic.APIError: any transport or API
                failure, propagated to chain.py.
        """
        if not hits:
            raise ValueError("Generator.answer called with no hits")

        user_prompt = prompts.build_user_prompt(question, hits)

        # Defensive: the 1500-token budget is a design constraint, not a hope.
        est = prompts.estimate_prompt_tokens(question, hits)
        if not prompts.context_is_within_budget(question, hits):
            logger.warning("prompt above budget: ~%d tokens", est)

        if self._provider == "groq":
            text = self._answer_groq(user_prompt)
        else:
            text = self._answer_anthropic(user_prompt)

        self.call_count += 1
        if not text:
            raise RuntimeError("model returned no text")
        return text

    def _answer_groq(self, user_prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": prompts.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0,  # A10 determinism
            "stream": False,
        }
        try:
            response = self._client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._cfg.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:  # noqa: BLE001 - chain.py decides the fallback
            logger.error("groq call failed: %s: %s", type(exc).__name__, exc)
            raise

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"groq returned no choices: {body!r}")
        return (choices[0].get("message", {}).get("content") or "").strip()

    def _answer_anthropic(self, user_prompt: str) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=MAX_TOKENS,
                temperature=0,  # A10 determinism
                system=prompts.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as exc:  # noqa: BLE001 - chain.py decides the fallback
            logger.error("anthropic call failed: %s: %s", type(exc).__name__, exc)
            raise

        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()

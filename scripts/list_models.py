"""List the models your API key can actually reach, and check the configured one.

Run this after putting a key in ``.env``:

    .venv/bin/python scripts/list_models.py

Purpose: ``LLM_MODEL`` in ``src/config.py`` is a default written without being
exercised against the API, so it may not be a valid identifier. Rather than
guess, enumerate what the key can reach and pick from that list. It also
separates a bad key from a bad model ID, which are different problems with
different fixes.

Groq is the default provider. Set ``LLM_PROVIDER=anthropic`` to check an
Anthropic key instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG  # noqa: E402

# Groq serves speech-to-text, prompt-injection classifiers and text-chat models
# from the same /models endpoint, so a naive "suggest a few" list recommends
# models that cannot hold a conversation. Excluding by substring is imprecise
# but far better than offering `llama-prompt-guard` as a chat model, and the
# real check is the trial call below.
NON_CHAT_MARKERS = (
    "whisper",      # speech to text
    "guard",        # prompt-injection classifier
    "safeguard",    # content-safety classifier
    "tts",          # text to speech
    "embed",        # embedding
    "audio",        # audio understanding
    "vision",       # image input only
)

# Preferred text/chat models, best first, when the key exposes any of them.
CHAT_PREFERENCES = (
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
)


def _is_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(marker in lowered for marker in NON_CHAT_MARKERS)


def _list_groq() -> tuple[list[str], str | None]:
    import requests

    if not CONFIG.groq_api_key:
        return [], "GROQ_API_KEY not set. Add it to .env first."

    try:
        response = requests.get(
            f"{CONFIG.groq_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {CONFIG.groq_api_key}"},
            timeout=30.0,
        )
    except Exception as exc:  # noqa: BLE001
        return [], f"could not reach Groq: {type(exc).__name__}: {exc}"

    if response.status_code in (401, 403):
        return [], (
            f"HTTP {response.status_code}: the key is wrong, inactive, or lacks "
            "access. Check GROQ_API_KEY in .env."
        )
    if response.status_code != 200:
        return [], f"HTTP {response.status_code}: {response.text[:300]}"

    try:
        data = response.json().get("data", [])
    except ValueError as exc:
        return [], f"could not parse the response as JSON: {exc}"

    return sorted(m.get("id", "") for m in data if m.get("id")), None


def _list_anthropic() -> tuple[list[str], str | None]:
    if not CONFIG.anthropic_api_key:
        return [], "ANTHROPIC_API_KEY not set. Add it to .env, or set LLM_PROVIDER=groq."

    try:
        import anthropic
    except ImportError:
        return [], "anthropic not installed. Run: .venv/bin/pip install anthropic"

    try:
        page = anthropic.Anthropic(
            api_key=CONFIG.anthropic_api_key, timeout=30.0, max_retries=0
        ).models.list()
    except Exception as exc:  # noqa: BLE001
        return [], f"could not list models: {type(exc).__name__}: {exc}"

    return sorted(getattr(m, "id", str(m)) for m in getattr(page, "data", [])), None


def main() -> int:
    provider = (CONFIG.llm_provider or "groq").strip().lower()
    print(f"provider : {provider}")
    print(f"endpoint : {CONFIG.groq_base_url if provider == 'groq' else 'api.anthropic.com'}")
    print(f"model    : {CONFIG.llm_model}")
    print()

    models, error = _list_groq() if provider == "groq" else _list_anthropic()
    if error:
        print(error, file=sys.stderr)
        return 1
    if not models:
        print("The key authenticated but returned no models.", file=sys.stderr)
        return 3

    print(f"{len(models)} model(s) available:\n")
    chat_models = [m for m in models if _is_chat_model(m)]
    for model_id in models:
        if not _is_chat_model(model_id):
            print(f"  {model_id}  <- not a chat model")
        else:
            print(f"  {model_id}")

    print()
    if not chat_models:
        print("WARNING: no chat-capable model found. The answer layer will fail.", file=sys.stderr)
        return 3

    if CONFIG.llm_model in chat_models:
        print(f"GROQ model {CONFIG.llm_model} is available and can chat. No change needed.")
        return 0

    if CONFIG.llm_model in models:
        print(f"{CONFIG.llm_model} is listed but is not a chat model. Pick one below.")
    else:
        print(f"{CONFIG.llm_model} is NOT in the list above.")

    suggestions = [m for m in CHAT_PREFERENCES if m in chat_models]
    if suggestions:
        print(f"\nChat models this key can use, strongest first: {', '.join(suggestions)}")
    print("Set GROQ_MODEL (or LLM_MODEL) in .env to one of those.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

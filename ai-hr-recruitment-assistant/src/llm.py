"""
Centralized LLM Configuration (OpenRouter)
--------------------------------------------
Every other module talks to the LLM only through the functions in this
file. This means the OpenRouter model can be swapped through `.env`
(OPENROUTER_MODEL) without touching any other code, and every failure
mode (missing key, invalid key, rate limit, model unavailable, timeout,
bad response) is handled in exactly one place with a friendly message.

OpenRouter exposes an OpenAI-compatible REST API, so we use the official
`openai` Python client pointed at OpenRouter's base URL instead of pulling
in a local LLM runtime (no Ollama / GPU required).
"""

import json
from functools import lru_cache
from typing import Optional

import openai
from openai import OpenAI

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_SITE_URL,
    OPENROUTER_SITE_NAME,
    require_api_key,
)


class LLMError(Exception):
    """Base class for all user-facing LLM failures."""


class MissingAPIKeyError(LLMError):
    pass


class InvalidAPIKeyError(LLMError):
    pass


class RateLimitedError(LLMError):
    pass


class ModelUnavailableError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class InvalidResponseError(LLMError):
    pass


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Build (and cache) the OpenAI-compatible client pointed at OpenRouter."""
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-Title": OPENROUTER_SITE_NAME,
        },
        timeout=60.0,
    )


def chat_completion(system_prompt: str, user_prompt: str, max_tokens: int = 1500,
                     temperature: float = 0.3, tools: Optional[list] = None,
                     json_mode: bool = False) -> "openai.types.chat.ChatCompletion":
    """
    Run a single chat completion against the configured OpenRouter model.
    Raises a specific LLMError subclass with a friendly message on any
    known failure mode, so callers (and the Streamlit UI) never see raw
    SDK exceptions.

    json_mode=True asks OpenRouter for strict JSON output via
    response_format. Not every free model supports this parameter — if the
    model rejects it, we transparently retry the same request without it.
    """
    require_api_key()
    client = get_client()

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    kwargs = dict(model=OPENROUTER_MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature)
    if tools:
        kwargs["tools"] = tools
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
    except openai.AuthenticationError as exc:
        raise InvalidAPIKeyError(
            "OpenRouter rejected the API key. Double-check OPENROUTER_API_KEY in your .env file."
        ) from exc
    except openai.RateLimitError as exc:
        raise RateLimitedError(
            "OpenRouter rate limit hit (common on free models). Wait a moment and retry, "
            "or switch OPENROUTER_MODEL to a different free model in .env."
        ) from exc
    except openai.APITimeoutError as exc:
        raise LLMTimeoutError(
            "The request to OpenRouter timed out. Please try again."
        ) from exc
    except openai.NotFoundError as exc:
        raise ModelUnavailableError(
            f"The model '{OPENROUTER_MODEL}' was not found or is unavailable on OpenRouter. "
            "Change OPENROUTER_MODEL in .env to another available model."
        ) from exc
    except openai.BadRequestError as exc:
        if json_mode:
            # This model doesn't support response_format=json_object — retry without it.
            return chat_completion(system_prompt, user_prompt, max_tokens=max_tokens,
                                    temperature=temperature, tools=tools, json_mode=False)
        raise ModelUnavailableError(
            f"OpenRouter rejected this request for model '{OPENROUTER_MODEL}': {exc}"
        ) from exc
    except openai.APIStatusError as exc:
        if exc.status_code in (400, 404):
            raise ModelUnavailableError(
                f"OpenRouter could not serve model '{OPENROUTER_MODEL}' (it may be temporarily "
                "unavailable or overloaded). Try again shortly or pick a different free model in .env."
            ) from exc
        raise LLMError(f"OpenRouter API error (status {exc.status_code}): {exc.message}") from exc
    except openai.APIConnectionError as exc:
        raise LLMError("Could not connect to OpenRouter. Check your internet connection and try again.") from exc

    if not response.choices:
        raise InvalidResponseError("OpenRouter returned an empty response. Please try again.")

    return response


def chat_text(system_prompt: str, user_prompt: str, max_tokens: int = 1500, temperature: float = 0.3,
              json_mode: bool = False) -> str:
    """Convenience wrapper: run a chat completion and return the plain text content."""
    response = chat_completion(system_prompt, user_prompt, max_tokens=max_tokens,
                                temperature=temperature, json_mode=json_mode)
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise InvalidResponseError("The model returned an empty answer. Please try again.")
    return content.strip()


def _extract_json_object(raw_text: str) -> str:
    """Strip markdown fences and stray commentary so only the JSON object remains."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    return cleaned


def chat_json(system_prompt: str, user_prompt: str, max_tokens: int = 1800, temperature: float = 0.2) -> dict:
    """
    Run a chat completion where the system prompt demands strict JSON output,
    then safely parse it. Uses OpenRouter's JSON response mode when the model
    supports it, and makes one corrective retry (explicitly telling the model
    its last reply was invalid) before giving up — free models occasionally
    add a stray sentence before/after the JSON object.
    """
    raw_text = chat_text(system_prompt, user_prompt, max_tokens=max_tokens, temperature=temperature, json_mode=True)
    cleaned = _extract_json_object(raw_text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass  # fall through to a single corrective retry below

    retry_prompt = (
        f"{user_prompt}\n\n"
        "Your previous reply was not valid JSON. Respond again with ONLY the JSON object — "
        "no explanation, no markdown code fences, no text before or after the braces."
    )
    raw_text_retry = chat_text(system_prompt, retry_prompt, max_tokens=max_tokens,
                                temperature=temperature, json_mode=True)
    cleaned_retry = _extract_json_object(raw_text_retry)

    try:
        return json.loads(cleaned_retry)
    except json.JSONDecodeError as exc:
        snippet = raw_text_retry[:200].replace("\n", " ")
        raise InvalidResponseError(
            f"The model did not return valid JSON after a retry ({exc}). "
            f"Response started with: \"{snippet}\". "
            "This free model may not reliably follow structured-output instructions — "
            "try switching OPENROUTER_MODEL in .env to a model that lists 'response_format' "
            "or 'structured_outputs' in its supported parameters, e.g. nex-agi/nex-n2.5-mini:free."
        ) from exc

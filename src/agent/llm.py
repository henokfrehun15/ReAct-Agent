"""Chat-completion client with retry.

Uses the official ``openai`` SDK purely as an HTTP client against a custom
``base_url``, so it works with any OpenAI-compatible API (Gemini, Groq, ...).
Retries with exponential backoff and jitter on HTTP 429 (rate limit) and
transient 5xx errors.
"""
from __future__ import annotations

import random
import time
from typing import Any

from openai import APIStatusError, OpenAI, RateLimitError

from .config import Config, load_config


class LLMError(RuntimeError):
    """The LLM request failed and all retries were exhausted."""


def _is_retryable(exc: BaseException) -> bool:
    """HTTP 429 (rate limit) and 5xx errors are retryable; anything else is not."""
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code == 429 or 500 <= code < 600
    return isinstance(exc, RateLimitError)


def _backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Exponential backoff with jitter, in seconds."""
    delay = min(base * (2 ** attempt), cap)
    return delay + random.uniform(0, delay * 0.5)


def _normalize_message(message: Any) -> dict[str, Any]:
    tool_calls = []
    for tool_call in getattr(message, "tool_calls", None) or []:
        tool_calls.append(
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments or "{}",
                },
            }
        )
    return {"content": message.content or "", "tool_calls": tool_calls}


def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    config: Config | None = None,
    client: Any = None,
    max_retries: int = 5,
) -> dict:
    """Call the chat-completion API and return a normalized assistant message.

    ``config``/``client`` default to building from environment config; pass them
    in for testing. On success returns ``{"content": str, "tool_calls": [...]}`.

    Retries up to ``max_retries`` times on 429/5xx with exponential backoff +
    jitter, then raises :class:`LLMError` with a clear message.
    """
    config = config or load_config()
    if client is None:
        client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)

    kwargs: dict[str, Any] = {"model": config.llm_model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    last_error: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            return _normalize_message(response.choices[0].message)
        except Exception as exc:  # noqa: BLE001 - classify; only retry transient errors
            last_error = exc
            if attempt == max_retries or not _is_retryable(exc):
                break
            time.sleep(_backoff_delay(attempt))

    raise LLMError(
        f"LLM request failed after {max_retries} retries; last error: {last_error}"
    ) from last_error
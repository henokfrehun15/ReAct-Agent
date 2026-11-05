"""Tests for the LLM retry behavior. No network, no API key needed."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.config import Config
from agent.llm import LLMError, chat


class RateLimited(Exception):
    """Fake HTTP 429 error."""
    status_code = 429


class ServerError(Exception):
    """Fake HTTP 5xx error."""
    status_code = 503


class BadRequest(Exception):
    """Fake HTTP 4xx error that must NOT be retried."""
    status_code = 400


def _fake_response(content: str = "ok") -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _config() -> Config:
    return Config(
        llm_base_url="http://example.local",
        llm_api_key="test-key",
        llm_model="test-model",
        max_steps=8,
    )


def test_retries_on_429_then_succeeds():
    config = _config()
    client = Mock()
    client.chat.completions.create = Mock(
        side_effect=[RateLimited("slow down"), RateLimited("slow down"), _fake_response()]
    )
    sleeps: list[float] = []

    with patch("agent.llm.OpenAI", return_value=client), patch(
        "agent.llm.time.sleep", side_effect=lambda d: sleeps.append(d)
    ):
        reply = chat([{"role": "user", "content": "hi"}], config=config, max_retries=5)

    assert reply == {"content": "ok", "tool_calls": []}
    assert client.chat.completions.create.call_count == 3  # initial + 2 retries
    assert len(sleeps) == 2  # one backoff sleep per retry
    assert sleeps[1] > sleeps[0]  # backoff increases


def test_retries_on_5xx_then_succeeds():
    config = _config()
    client = Mock()
    client.chat.completions.create = Mock(
        side_effect=[ServerError("down"), _fake_response()]
    )
    sleeps: list[float] = []

    with patch("agent.llm.OpenAI", return_value=client), patch(
        "agent.llm.time.sleep", side_effect=lambda d: sleeps.append(d)
    ):
        reply = chat([{"role": "user", "content": "hi"}], config=config)

    assert reply["content"] == "ok"
    assert client.chat.completions.create.call_count == 2
    assert len(sleeps) == 1


def test_raises_llm_error_after_max_retries():
    config = _config()
    client = Mock()
    client.chat.completions.create = Mock(side_effect=ServerError("down"))
    sleeps: list[float] = []

    with patch("agent.llm.OpenAI", return_value=client), patch(
        "agent.llm.time.sleep", side_effect=lambda d: sleeps.append(d)
    ):
        try:
            chat([{"role": "user", "content": "hi"}], config=config, max_retries=2)
        except LLMError as exc:
            assert "failed after 2 retries" in str(exc)
        else:
            raise AssertionError("expected LLMError")

    assert client.chat.completions.create.call_count == 3  # 1 + 2 retries
    assert len(sleeps) == 2


def test_non_retryable_error_is_not_retried():
    config = _config()
    client = Mock()
    client.chat.completions.create = Mock(side_effect=BadRequest("bad"))
    sleeps: list[float] = []

    with patch("agent.llm.OpenAI", return_value=client), patch(
        "agent.llm.time.sleep"
    ) as sleep:
        try:
            chat([{"role": "user", "content": "hi"}], config=config)
        except LLMError:
            pass
        else:
            raise AssertionError("expected LLMError")

    assert client.chat.completions.create.call_count == 1
    assert not sleep.called
"""Application configuration loaded from environment variables / ``.env``.

Never hardcodes secrets. Required variables fail early with a clear message.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

REQUIRED_VARS = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    max_steps: int


def load_config(env_file: str | None = None) -> Config:
    """Load settings from ``.env`` (if present) and environment variables.

    Environment variables take precedence over ``.env``. Raises
    :class:`ConfigurationError` with a clear message if a required variable
    is missing.
    """
    load_dotenv(env_file, override=False)

    missing = [name for name in REQUIRED_VARS if not os.environ.get(name)]
    if missing:
        raise ConfigurationError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill in your values."
        )

    try:
        max_steps = int(os.environ.get("MAX_STEPS", "8"))
    except (TypeError, ValueError):
        raise ConfigurationError(
            f"MAX_STEPS must be an integer, got {os.environ.get('MAX_STEPS')!r}"
        )
    if max_steps < 1:
        raise ConfigurationError("MAX_STEPS must be >= 1")

    return Config(
        llm_base_url=os.environ["LLM_BASE_URL"],
        llm_api_key=os.environ["LLM_API_KEY"],
        llm_model=os.environ["LLM_MODEL"],
        max_steps=max_steps,
    )
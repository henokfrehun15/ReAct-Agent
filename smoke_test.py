"""Manual smoke test against a real LLM API.

1. Copy .env.example to .env and fill in LLM_API_KEY / LLM_MODEL.
2. Run:  PYTHONPATH=src python smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from agent.config import load_config  # noqa: E402
from agent.llm import chat  # noqa: E402


def main() -> None:
    config = load_config()
    reply = chat(
        [{"role": "user", "content": "Say exactly: hello world, nothing else"}],
        tools=None,
        config=config,
    )
    print("REPLY:", reply["content"])


if __name__ == "__main__":
    main()
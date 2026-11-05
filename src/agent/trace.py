"""Trace logger: append JSONL events to logs/run-<run_id>.jsonl.

One object per line with keys: run_id, step, ts (ISO UTC), type, and
optional content / call_id / tool / args / result / latency_ms.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOG_DIR = Path("logs")


class TraceWriter:
    """Append trace events for a single agent run as JSONL."""

    def __init__(self, run_id: str, log_dir: str | Path = "logs") -> None:
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"run-{run_id}.jsonl"

    def _write(self, event: dict) -> None:
        event.setdefault("run_id", self.run_id)
        event.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def user(self, step: int, content: str) -> None:
        self._write({"step": step, "type": "user", "content": content})

    def assistant(
        self, step: int, content: str, tool_calls: Optional[list] = None
    ) -> None:
        event: dict = {"step": step, "type": "assistant", "content": content}
        if tool_calls:
            event["tool_calls"] = tool_calls
        self._write(event)

    def tool_call(self, step: int, call_id: str, tool: str, args: str) -> None:
        self._write(
            {"step": step, "type": "tool_call", "call_id": call_id, "tool": tool, "args": args}
        )

    def tool_result(
        self, step: int, call_id: str, tool: str, result: str, latency_ms: int
    ) -> None:
        self._write(
            {
                "step": step,
                "type": "tool_result",
                "call_id": call_id,
                "tool": tool,
                "result": result,
                "latency_ms": latency_ms,
            }
        )

    def error(self, step: int, content: str) -> None:
        self._write({"step": step, "type": "error", "content": content})

    def final(self, step: int, content: str) -> None:
        self._write({"step": step, "type": "final", "content": content})
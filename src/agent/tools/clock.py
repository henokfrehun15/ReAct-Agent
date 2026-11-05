"""Clock tool: returns the current date/time for a given IANA timezone."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from .registry import tool


@tool(
    "get_current_time",
    "Return the current date and time for a given timezone (defaults to UTC).",
    {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone name, e.g. UTC, America/New_York, Europe/London.",
            }
        },
        "required": [],
        "additionalProperties": False,
    },
)
def get_current_time(timezone: str = "UTC") -> str:
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        raise ValueError(f"Unknown timezone: {timezone}")
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")
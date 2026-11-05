"""Notes tool: case-insensitive keyword search over data/notes.txt."""
from __future__ import annotations

from pathlib import Path

from .registry import tool

_NOTES_PATH = Path(__file__).resolve().parents[3] / "data" / "notes.txt"


@tool(
    "search_notes",
    "Search data/notes.txt for lines containing a keyword (case-insensitive). Returns up to 5 matching lines.",
    {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keyword to search for in the notes file.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)
def search_notes(query: str) -> str:
    if not _NOTES_PATH.exists():
        return f"Error: Notes file not found at {_NOTES_PATH}"
    query_lower = query.lower()
    matches: list[str] = []
    with open(_NOTES_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if query_lower in line.lower():
                matches.append(line)
                if len(matches) >= 5:
                    break
    if not matches:
        return "No matching notes found."
    return "\n".join(matches)
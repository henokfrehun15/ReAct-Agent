"""Tests for the tool registry. No network, no API key needed."""
import agent.tools.calculator  # noqa: F401  (registers the calculator tool)
import agent.tools.clock  # noqa: F401  (registers get_current_time)
import agent.tools.notes  # noqa: F401  (registers search_notes)
import agent.tools.registry as registry
from agent.tools.registry import call, schemas, tool


def test_schemas_have_openai_format():
    @tool(
        "echo",
        "Echo the given text back.",
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    def echo(text: str) -> str:
        return text

    try:
        result = schemas()
        assert isinstance(result, list) and len(result) >= 1
        entry = next(s for s in result if s["function"]["name"] == "echo")
        assert entry["type"] == "function"
        assert entry["function"]["description"] == "Echo the given text back."
        assert entry["function"]["parameters"]["required"] == ["text"]
    finally:
        registry._tools.pop("echo", None)


def test_unknown_tool_lists_valid_names():
    result = call("not_a_tool", {})
    assert result.startswith("Error: Unknown tool")
    assert "calculator" in result


def test_missing_required_argument_returns_schema():
    result = call("calculator", {})
    assert result.startswith("Error:")
    assert "expression" in result
    assert "Expected schema" in result


def test_wrong_argument_type():
    result = call("calculator", {"expression": 123})
    assert result.startswith("Error:")
    assert "string" in result.lower()


def test_json_string_arguments_are_accepted():
    assert call("calculator", '{"expression": "40 + 2"}') == "42"


def test_invalid_json_arguments():
    result = call("calculator", "{not json")
    assert result.startswith("Error: Invalid JSON")


def test_exception_inside_tool_is_returned_as_error():
    @tool("failing", "Always raises.", {"type": "object", "properties": {}, "required": []})
    def failing() -> str:
        raise RuntimeError("boom")

    try:
        result = call("failing", {})
        assert result.startswith("Error:") and "RuntimeError: boom" in result
    finally:
        registry._tools.pop("failing", None)


def test_call_never_raises():
    assert isinstance(call("does_not_exist", {}), str)
    assert isinstance(call("calculator", "nope"), str)
    assert isinstance(call("calculator", {"expression": object()}), str)


def test_clock_utc():
    result = call("get_current_time", {"timezone": "UTC"})
    assert result.startswith("20"), result  # e.g. "2026-11-05 ... UTC"


def test_clock_invalid_timezone():
    result = call("get_current_time", {"timezone": "Not/AZone"})
    assert result.startswith("Error:"), result


def test_notes_search():
    result = call("search_notes", {"query": "Paris"})
    assert "Paris" in result, result


def test_notes_case_insensitive():
    result = call("search_notes", {"query": "python"})
    assert "Python" in result or "python" in result.lower(), result


def test_notes_no_match():
    result = call("search_notes", {"query": "zzzqwertynotfound"})
    assert "No matching notes found" in result, result


def test_notes_limits_to_five():
    result = call("search_notes", {"query": "the"})
    lines = [l for l in result.split("\n") if l.strip()]
    assert len(lines) <= 5, result
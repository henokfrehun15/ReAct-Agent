"""Tests for the tool registry. No network, no API key needed."""
import agent.tools.calculator  # noqa: F401  (registers the calculator tool)
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
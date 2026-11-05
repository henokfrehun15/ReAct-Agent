"""Tool registry: the ``@tool`` decorator, OpenAI schema generation, and safe calling.

:func:`call` never raises: unknown tools, invalid arguments and tool exceptions
are returned as ``"Error: ..."`` strings that name the valid tools or show the
expected schema, so the model can recover.
"""
from __future__ import annotations

import json
from typing import Any

_MAX_RESULT_CHARS = 4000

_tools: dict[str, "Tool"] = {}


class Tool:
    """A registered callable with its name, description and JSON-schema parameters."""

    def __init__(self, name: str, description: str, parameters: dict, fn: Any) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.fn = fn

    def openai_format(self) -> dict[str, Any]:
        """Return this tool in OpenAI's ``tools`` list format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool(_fn=None, name=None, description=None, parameters=None):
    """Register ``_fn`` as a callable tool.

    Usage::

        @tool("name", "description", {"type": "object", ...})
        def my_tool(arg: str) -> str: ...

    Can also be used bare (``@tool``); then the function's
    ``__tool_name__`` / ``__tool_description__`` / ``__tool_parameters__``
    attributes fall back to its name and docstring.
    """

    def decorator(fn):
        t = Tool(
            name or getattr(fn, "__tool_name__", fn.__name__),
            description or getattr(fn, "__tool_description__", fn.__doc__ or ""),
            parameters or getattr(
                fn, "__tool_parameters__", {"type": "object", "properties": {}, "required": []}
            ),
            fn,
        )
        _tools[t.name] = t
        return fn

    # Bare usage: @tool  ->  _fn is the function itself.
    if callable(_fn):
        return decorator(_fn)
    # Called usage: @tool("name", "description", {...})  ->  positionals
    # arrive shifted: _fn=name, name=description, description=parameters.
    if isinstance(_fn, str) and name is not None:
        name, description, parameters = _fn, name, description
    return decorator


def schemas() -> list[dict[str, Any]]:
    """Return all registered tools in OpenAI tool format."""
    return [t.openai_format() for t in _tools.values()]


def _type_matches(expected: str, value: Any) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return True


def _validation_problems(parameters: dict, args: dict) -> list[str]:
    problems: list[str] = []
    for required in parameters.get("required", []):
        if required not in args:
            problems.append(f"missing required argument '{required}'")
    for key, value in args.items():
        prop = parameters.get("properties", {}).get(key)
        if prop is None:
            continue
        if not _type_matches(prop.get("type", ""), value):
            problems.append(
                f"argument '{key}' must be {prop.get('type')}, got {type(value).__name__}"
            )
    return problems


def call(name: str, args: dict | str) -> str:
    """Call registered tool ``name`` with ``args`` (a dict or JSON string).

    Never raises. Unknown tools, invalid arguments and tool exceptions are
    returned as ``"Error: ..."`` strings listing valid tools or the expected
    schema. Results longer than ``_MAX_RESULT_CHARS`` are truncated with a note.
    """
    if name not in _tools:
        valid = ", ".join(sorted(_tools)) or "(none registered)"
        return f"Error: Unknown tool '{name}'. Valid tools: {valid}"

    tool_entry = _tools[name]

    if isinstance(args, str):
        try:
            parsed: Any = json.loads(args)
        except json.JSONDecodeError as exc:
            return f"Error: Invalid JSON arguments for tool '{name}': {exc}"
    else:
        parsed = args

    if not isinstance(parsed, dict):
        return f"Error: Arguments for tool '{name}' must be a JSON object, got {type(parsed).__name__}"

    problems = _validation_problems(tool_entry.parameters, parsed)
    if problems:
        schema_json = json.dumps(tool_entry.parameters, ensure_ascii=False)
        return (
            f"Error: Invalid arguments for tool '{name}': {'; '.join(problems)}. "
            f"Expected schema: {schema_json}"
        )

    try:
        result = tool_entry.fn(**parsed)
    except Exception as exc:  # noqa: BLE001 - a crashing tool must not kill the agent
        return f"Error: Tool '{name}' raised {type(exc).__name__}: {exc}"

    text = result if isinstance(result, str) else str(result)
    if len(text) > _MAX_RESULT_CHARS:
        dropped = len(text) - _MAX_RESULT_CHARS
        text = text[:_MAX_RESULT_CHARS] + f"\n... [result truncated, {dropped} chars omitted]"
    return text
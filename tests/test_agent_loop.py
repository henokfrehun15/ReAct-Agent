"""Tests for the ReAct agent loop using a scripted FakeLLM.

No network, no API key needed.
"""
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock

from agent.agent import AgentResult, run_agent
from agent.config import Config
from agent.tools.registry import call, schemas

import agent.tools.calculator  # noqa: F401  (registers the calculator tool)


# ---------------------------------------------------------------------------
# Fake LLM client helper
# ---------------------------------------------------------------------------

def _fake_message(content: str = "", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or None)


def _fake_choice(message):
    return SimpleNamespace(message=message)


def _fake_response(message):
    return SimpleNamespace(choices=[_fake_choice(message)])


def _tool_call(call_id: str, name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _fake_client(replies):
    """Build a fake openai client whose chat.completions.create returns replies in order."""
    client = Mock()
    call_counter = {"n": 0}

    def create(**kwargs):
        reply = replies[call_counter["n"]]
        call_counter["n"] += 1
        return reply

    client.chat.completions.create = Mock(side_effect=create)
    return client


def _base_config() -> Config:
    return Config(
        llm_base_url="http://example.local",
        llm_api_key="test-key",
        llm_model="test-model",
        max_steps=8,
    )


def test_tool_call_then_final_answer():
    """(a) One tool call, then a final answer."""
    config = _base_config()
    replies = [
        _fake_response(
            _fake_message(
                "Let me calculate that.",
                tool_calls=[_tool_call("c1", "calculator", {"expression": "2 + 2"})],
            )
        ),
        _fake_response(_fake_message("The answer is 4.")),
    ]
    result = run_agent("What is 2 + 2?", config=config, client=_fake_client(replies))

    assert result.final_answer == "The answer is 4."
    assert result.stopped_reason == "final"
    assert result.steps_used == 2
    assert isinstance(result.run_id, str) and len(result.run_id) > 0


def test_two_tool_calls_in_one_reply():
    """(b) Two tool calls in a single assistant reply."""
    config = _base_config()
    replies = [
        _fake_response(
            _fake_message(
                "Calculating both.",
                tool_calls=[
                    _tool_call("c1", "calculator", {"expression": "2 + 2"}),
                    _tool_call("c2", "calculator", {"expression": "3 * 3"}),
                ],
            )
        ),
        _fake_response(_fake_message("4 and 9.")),
    ]
    result = run_agent("What is 2+2 and 3*3?", config=config, client=_fake_client(replies))

    assert result.final_answer == "4 and 9."
    assert result.stopped_reason == "final"
    assert result.steps_used == 2


def test_unknown_tool_then_recovers():
    """(c) Model calls an unknown tool, then recovers with a final answer."""
    config = _base_config()
    replies = [
        _fake_response(
            _fake_message(
                "Trying a tool.",
                tool_calls=[_tool_call("c1", "nonexistent_tool", {"expression": "2+2"})],
            )
        ),
        _fake_response(_fake_message("That tool doesn't exist. The answer is 4.")),
    ]
    result = run_agent("What is 2 + 2?", config=config, client=_fake_client(replies))

    assert result.final_answer == "That tool doesn't exist. The answer is 4."
    assert result.stopped_reason == "final"
    assert result.steps_used == 2


def test_max_steps_stop():
    """(d) Agent stops when MAX_STEPS is reached."""
    config = Config(
        llm_base_url="http://example.local",
        llm_api_key="test-key",
        llm_model="test-model",
        max_steps=2,
    )
    replies = [
        _fake_response(
            _fake_message(
                "Still thinking.",
                tool_calls=[_tool_call("c1", "calculator", {"expression": "2 + 2"})],
            )
        ),
        _fake_response(
            _fake_message(
                "Still thinking.",
                tool_calls=[_tool_call("c2", "calculator", {"expression": "3 + 3"})],
            )
        ),
    ]
    result = run_agent("What is 2 + 2?", config=config, client=_fake_client(replies))

    assert result.stopped_reason == "max_steps"
    assert result.steps_used == 2
    assert "max steps" in result.final_answer.lower()


def test_repeated_call_stop():
    """(e) Agent stops when the same tool call repeats 3 times in a row."""
    config = _base_config()
    replies = [
        _fake_response(
            _fake_message(
                "Trying again.",
                tool_calls=[_tool_call("c1", "calculator", {"expression": "2 + 2"})],
            )
        ),
        _fake_response(
            _fake_message(
                "Trying again.",
                tool_calls=[_tool_call("c2", "calculator", {"expression": "2 + 2"})],
            )
        ),
        _fake_response(
            _fake_message(
                "Trying again.",
                tool_calls=[_tool_call("c3", "calculator", {"expression": "2 + 2"})],
            )
        ),
    ]
    result = run_agent("What is 2 + 2?", config=config, client=_fake_client(replies))

    assert result.stopped_reason == "repeated_call"
    assert result.steps_used == 3
    assert "repeated 3 times" in result.final_answer


def test_trace_file_contains_expected_events():
    """(f) Trace JSONL file contains the expected event sequence."""
    config = _base_config()
    replies = [
        _fake_response(
            _fake_message(
                "Calculating.",
                tool_calls=[_tool_call("c1", "calculator", {"expression": "2 + 2"})],
            )
        ),
        _fake_response(_fake_message("The answer is 4.")),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        from agent.trace import TraceWriter

        run_id = "test-run-001"
        trace_writer = TraceWriter(run_id, log_dir=tmpdir)
        result = run_agent(
            "What is 2 + 2?",
            config=config,
            client=_fake_client(replies),
            trace_writer=trace_writer,
        )

        log_path = os.path.join(tmpdir, f"run-{run_id}.jsonl")
        assert os.path.exists(log_path)

        events = []
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))

        types = [e["type"] for e in events]
        assert "user" in types
        assert "assistant" in types
        assert "tool_call" in types
        assert "tool_result" in types
        assert "final" in types

        user_event = next(e for e in events if e["type"] == "user")
        assert user_event["content"] == "What is 2 + 2?"
        assert user_event["run_id"] == run_id

        tool_call_event = next(e for e in events if e["type"] == "tool_call")
        assert tool_call_event["tool"] == "calculator"
        assert tool_call_event["call_id"] == "c1"

        tool_result_event = next(e for e in events if e["type"] == "tool_result")
        assert tool_result_event["tool"] == "calculator"
        assert tool_result_event["result"] == "4"
        assert "latency_ms" in tool_result_event
        assert isinstance(tool_result_event["latency_ms"], int)

        final_event = next(e for e in events if e["type"] == "final")
        assert final_event["content"] == "The answer is 4."

    assert result.stopped_reason == "final"
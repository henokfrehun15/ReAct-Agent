"""ReAct (reason + act) agent loop using native tool calling.

No agent framework — just a while loop that calls the LLM, executes tool
calls through the registry, and feeds results back as tool-role messages.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from agent.config import Config, load_config
from agent.llm import LLMError, chat
from agent.tools.registry import call, schemas

SYSTEM_PROMPT_PERMISSIVE = """You are a ReAct (reason + act) agent. You have access to tools.

{tools_desc}

Instructions:
- Think step by step before acting.
- Use the available tools for arithmetic, time lookups, and notes searches.
- If a tool returns an error, correct yourself and try again.
- When you have enough information, give a concise final answer.
- Do not call tools if you can answer directly from your knowledge.
"""

SYSTEM_PROMPT_MANDATORY = """You are a ReAct (reason + act) agent. You have access to tools.

{tools_desc}

Instructions:
- Think step by step before acting.
- ALWAYS use the calculator for any arithmetic question.
- ALWAYS use get_current_time for any time or date question.
- ALWAYS use search_notes for any question about Project Nebula or notes.
- If a tool returns an error, correct yourself and try again.
- When you have enough information, give a concise final answer.
"""

SYSTEM_PROMPTS = {
    "permissive": SYSTEM_PROMPT_PERMISSIVE,
    "mandatory": SYSTEM_PROMPT_MANDATORY,
}


@dataclass
class AgentResult:
    final_answer: str
    steps_used: int
    stopped_reason: str  # "final" | "max_steps" | "repeated_call" | "error"
    run_id: str
    tools_used: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.tools_used is None:
            self.tools_used = []


def _tools_description(tools: list[dict]) -> str:
    lines = []
    for t in tools:
        fn = t["function"]
        lines.append(f"- {fn['name']}: {fn['description']}")
    return "\n".join(lines)


def run_agent(
    question: str,
    *,
    config: Config | None = None,
    trace_writer: Optional[Any] = None,
    client: Any = None,
    system_prompt: str = "permissive",
) -> AgentResult:
    """Run the ReAct loop for a single question.

    Returns an AgentResult with final_answer, steps_used, stopped_reason, run_id.
    ``system_prompt`` selects which system prompt template to use ("permissive" or "mandatory").
    """
    config = config or load_config()
    run_id = uuid.uuid4().hex[:8]
    tools = schemas()
    tools_desc = _tools_description(tools)

    template = SYSTEM_PROMPTS.get(system_prompt, SYSTEM_PROMPT_PERMISSIVE)

    messages: list[dict] = [
        {"role": "system", "content": template.format(tools_desc=tools_desc)},
        {"role": "user", "content": question},
    ]

    if trace_writer:
        trace_writer.user(0, question)

    last_call_key: Optional[tuple] = None
    repeat_count = 0
    tools_used: list[str] = []

    for step in range(1, config.max_steps + 1):
        try:
            reply = chat(messages, tools, config=config, client=client)
        except LLMError as exc:
            if trace_writer:
                trace_writer.error(step, str(exc))
            return AgentResult(
                final_answer=str(exc),
                steps_used=step - 1,
                stopped_reason="error",
                run_id=run_id,
                tools_used=tools_used,
            )

        assistant_msg: dict = {"role": "assistant", "content": reply["content"]}
        if reply["tool_calls"]:
            assistant_msg["tool_calls"] = reply["tool_calls"]
        messages.append(assistant_msg)

        if trace_writer:
            trace_writer.assistant(step, reply["content"], reply["tool_calls"])

        if not reply["tool_calls"]:
            if trace_writer:
                trace_writer.final(step, reply["content"])
            return AgentResult(
                final_answer=reply["content"],
                steps_used=step,
                stopped_reason="final",
                run_id=run_id,
                tools_used=tools_used,
            )

        # Execute every tool call through the registry
        for tool_call in reply["tool_calls"]:
            name = tool_call["function"]["name"]
            args_str = tool_call["function"]["arguments"] or "{}"
            call_key = (name, args_str)

            if call_key == last_call_key:
                repeat_count += 1
            else:
                repeat_count = 1
                last_call_key = call_key

            if trace_writer:
                trace_writer.tool_call(step, tool_call["id"], name, args_str)

            if repeat_count >= 3:
                msg = f"stopped: same tool call repeated 3 times: {name}({args_str})"
                if trace_writer:
                    trace_writer.error(step, msg)
                return AgentResult(
                    final_answer=msg,
                    steps_used=step,
                    stopped_reason="repeated_call",
                    run_id=run_id,
                    tools_used=tools_used,
                )

            t0 = time.perf_counter()
            tools_used.append(name)
            result = call(name, args_str)
            latency_ms = int((time.perf_counter() - t0) * 1000)

            if trace_writer:
                trace_writer.tool_result(step, tool_call["id"], name, result, latency_ms)

            messages.append(
                {
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call["id"],
                }
            )

    msg = "stopped: max steps reached"
    if trace_writer:
        trace_writer.error(config.max_steps, msg)
    return AgentResult(
        final_answer=msg,
        steps_used=config.max_steps,
        stopped_reason="max_steps",
        run_id=run_id,
        tools_used=tools_used,
    )
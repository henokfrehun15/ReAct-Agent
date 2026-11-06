# react-agent-from-scratch

A minimal ReAct (reason + act) agent built by hand — **no agent framework** (no LangChain, LlamaIndex, LangGraph, etc.). Works with any OpenAI-compatible chat API (Gemini, Groq, etc.) using the `openai` SDK purely as an HTTP client with a custom `base_url`.

## What is a ReAct agent?

A ReAct agent interleaves **reasoning** and **acting**:

1. **Reason** — the LLM thinks about what to do next and decides whether to call a tool.
2. **Act** — the agent executes the tool call and feeds the result back to the LLM.
3. **Repeat** — until the LLM produces a final answer (no more tool calls).

This implementation is ~200 lines of plain Python with no orchestration framework.

## Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   User Question │─────▶│   LLM (chat)     │─────▶│   Tool Registry │
│                 │      │   (reasoning)    │      │   (execution)   │
└─────────────────┘      └────────┬─────────┘      └────────┬─────────┘
                                 │                          │
                                 │  tool_calls              │  result strings
                                 ▼                          ▼
                          ┌──────────────────┐      ┌─────────────────┐
                          │   Message Loop   │◄─────│   Trace (JSONL) │
                          │   (tools as      │      │   logs/run-*.   │
                          │    tool-role msgs│      │   jsonl         │
                          └──────────────────┘      └─────────────────┘
```

## Quickstart

### Local

```powershell
# 1. Copy .env.example and fill in your values
copy .env.example .env
# Edit .env: set LLM_API_KEY, LLM_MODEL, and choose LLM_BASE_URL

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a question
PYTHONPATH=src python -m agent.cli "What is 15% of 2340, plus 200?"
```

### Docker

```powershell
# 1. Fill in .env with your API key and model

# 2. Build and run
docker compose run --rm agent "What time is it in UTC?"
```

## Switching between Gemini and Groq

Edit `.env`:

**Gemini:**
```env
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=your-gemini-key
LLM_MODEL=models/gemini-1.5-flash
```

**Groq:**
```env
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=your-groq-key
LLM_MODEL=openai/gpt-oss-20b
```

Only `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` need to change. Everything else is provider-agnostic.

## Adding a new tool (10 lines)

```python
# src/agent/tools/my_tool.py
from agent.tools.registry import tool

@tool(
    "my_tool",
    "Do something useful.",
    {
        "type": "object",
        "properties": {
            "arg": {"type": "string", "description": "Input argument."}
        },
        "required": ["arg"],
        "additionalProperties": False,
    },
)
def my_tool(arg: str) -> str:
    return f"Result: {arg}"
```

Then import it in `src/agent/tools/__init__.py`:

```python
from agent.tools import my_tool  # noqa: F401
```

The `@tool` decorator registers it automatically. `call()` handles validation, JSON parsing, and error catching — your tool function never needs to worry about bad input.

## Running tests

```powershell
# No API key or network needed
pip install -r requirements.txt
pytest
```

## Running evals

```powershell
# Requires a real API key in .env
PYTHONPATH=src python evals/run_evals.py
```

Runs all 12 tasks from `evals/tasks.jsonl`, prints a table, and saves `evals/results.json` with pass/fail, steps used, tools used vs expected, latency, and failure category per task.

## Example run

```powershell
$env:PYTHONPATH="src"; python -m agent.cli "What is 15% of 2340, plus 200?"
```

Output:
```
15 % of 2340 is 351, and 351 + 200 equals **551**.
Steps used: 1
Stopped reason: final
Log file: logs/run-8b305198.jsonl
```

Note: the model answered from internal knowledge (1 step, no tool calls). For time/date questions it does call `get_current_time`:

```powershell
$env:PYTHONPATH="src"; python -m agent.cli --verbose "What time is it in UTC?"
```

```
[user] What time is it in UTC?
  [assistant] Let me check the current time for you.
  [tool_call] get_current_time({"timezone":"UTC"})
  [tool_result] get_current_time -> 2026-11-06 06:10:07 UTC [45ms]
  [assistant] It's 2026-11-06 06:10 UTC.
  [final] It's 2026-11-06 06:10 UTC.
It's 2026-11-06 06:10 UTC.
Steps used: 2
Stopped reason: final
Log file: logs/run-f62cb30d.jsonl
```

## Eval results

Ran all 12 tasks from `evals/tasks.jsonl` against a real LLM (Groq `openai/gpt-oss-20b`).

| Metric | Value |
|--------|-------|
| Accuracy | 12/12 (100%) |
| Average steps | 1.25 |
| Average latency | 946ms |

| # | Question | Expected | Match | Steps | Tools used vs expected | Category |
|---|----------|----------|-------|-------|------------------------|----------|
| t1 | What is 25 * 4? | 100 | ✅ | 1 | – vs calculator | wrong_tools |
| t2 | 15% of 2340, plus 200 | 551 | ✅ | 1 | – vs calculator | wrong_tools |
| t3 | What time is it in UTC? | UTC | ✅ | 2 | get_current_time vs get_current_time | pass |
| t4 | Capital of France | Paris | ✅ | 1 | – vs search_notes | wrong_tools |
| t5 | UTC time + 100 seconds | UTC | ✅ | 2 | get_current_time vs get_current_time, calculator | wrong_tools |
| t6 | Capital of Japan | Tokyo | ✅ | 1 | – vs – | pass |
| t7 | Who created Python? | Guido van Rossum | ✅ | 1 | – vs search_notes | wrong_tools |
| t8 | 2 to the power of 10 | 1024 | ✅ | 1 | – vs calculator | wrong_tools |
| t9 | Speed of light | 299792458 | ✅ | 1 | – vs search_notes | wrong_tools |
| t10 | 10 plus 20 times 3 | 70 | ✅ | 1 | – vs calculator | wrong_tools |
| t11 | Current date in UTC | UTC | ✅ | 2 | get_current_time vs get_current_time | pass |
| t12 | Largest planet | Jupiter | ✅ | 1 | – vs search_notes | wrong_tools |

Full per-task data is saved to `evals/results.json` after each run.

## Project layout

```
src/agent/
  __init__.py     - Package marker
  cli.py          - CLI entry point (single question or interactive loop)
  config.py       - Config dataclass, load_config() with early failure
  llm.py          - chat() with retry + backoff (openai SDK as HTTP client)
  agent.py        - ReAct loop, AgentResult dataclass
  trace.py        - TraceWriter, JSONL logging to logs/run-*.jsonl
  tools/
    __init__.py   - Registers all built-in tools
    registry.py   - @tool decorator, schemas(), call() (never raises)
    calculator.py - AST-based safe arithmetic
    clock.py      - get_current_time() via zoneinfo
    notes.py      - search_notes() over data/notes.txt
data/
  notes.txt       - Sample notes for search_notes tool
evals/
  tasks.jsonl     - 12 eval tasks
  run_evals.py    - Evaluation runner
tests/
  test_calculator.py
  test_registry.py
  test_agent_loop.py
  test_llm_retry.py
logs/             - gitignored, trace JSONL files
```

## Limitations and failures observed

### What went wrong

**Answer accuracy was 100% (12/12), but tool usage matched expectations in only 3 of 12 tasks.** The other 9 tasks were categorized as `wrong_tools` — the final answer was correct, but the model didn't call the tool the task was designed to exercise.

**Root cause:** the model (Groq `openai/gpt-oss-20b`) is smart enough to answer every eval question from internal knowledge, so it never bothers calling tools. The system prompt even tells it "Do not call tools if you can answer directly from your knowledge," which works against tool use here.

### Real failure examples

**t1 — "What is 25 * 4?" (expected `calculator`)**
- Trace shows a single assistant turn: `[assistant] 25 × 4 = 100.` with no `tool_calls`.
- The model computed it in its head and skipped the calculator entirely.
- Log: `logs/run-df2802a6.jsonl`

**t9 — "What is the speed of light?" (expected `search_notes`)**
- Trace shows `[assistant] The speed of light in a vacuum is **299,792,458 meters per second**` with no `tool_calls`.
- The model knew the constant from memory and didn't search the notes file.
- Log: `logs/run-adbff6c2.jsonl`

**t5 — "What is the current UTC time plus 100 seconds?" (expected `get_current_time` + `calculator`)**
- The model called `get_current_time` correctly, but then did the "+ 100 seconds" arithmetic in its head instead of using `calculator`.
- Tools used: `[get_current_time]`; expected: `[get_current_time, calculator]`.
- Log: `logs/run-5250ed70.jsonl`

### Other limitations

- **Number formatting:** Models may write "1,024" or "299 792 458" with spaces/commas. Eval matching handles commas and whitespace, but unusual formats can still fail.
- **Free-tier rate limits:** Gemini and Groq free tiers have low RPM. Eval script sleeps 2s between tasks; real-world use may need backoff.
- **No tool schema validation at call time:** The registry validates argument types and required fields, but doesn't validate values (e.g., a negative number for a timezone-agnostic tool).
- **Single-threaded:** Each run blocks on LLM responses. No async or parallel tool execution.
- **System prompt is static:** The model may ignore instructions to use tools for simple questions. A stronger prompt or `tool_choice=required` could help force tool use on tasks that require it.
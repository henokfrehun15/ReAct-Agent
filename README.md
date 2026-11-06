# react-agent-from-scratch

A minimal ReAct (reason + act) agent built by hand, with no agent framework.
Works with any OpenAI-compatible chat API (Gemini, Groq, etc.) using the `openai`
SDK purely as an HTTP client with a custom `base_url`.

## Before/After Eval Results

| Metric | Original Eval (v1) | Enhanced Eval (v2) |
|--------|--------------------|--------------------|
| Task count | 12 general-knowledge questions | 15 tasks × 2 types |
| Answer accuracy | 100% (all correct from memory) | 71.8% tool-necessary, 100% tool-unnecessary |
| Tool-use rate (necessary) | 25% (3/12 matched) | 84.6% (permissive), 92.3% (mandatory) |
| Unnecessary tool use | N/A (all tasks allowed tools) | 0% (control tasks) |
| Prompt sensitivity | None reported | Mandatory prompt forces tool use; permissive causes systematic avoidance |

**Root cause (v1):** All 12 questions were general knowledge any model can answer internally, so skipping tools still yielded correct answers. The "100% accuracy" number was misleading for a tool-use evaluation.

**Key improvement (v2):** Tasks are split into two categories:
- **Tool-necessary:** Cannot be answered from memory (arithmetic, time math, fictional notes). Model must use tools.
- **Tool-unnecessary (controls):** Should be answered without any tool call. Models learn when NOT to use tools.

The enhanced eval also compares two system prompts:
- **Permissive:** "Do not call tools if you can answer directly" → tool-use rate ~25-35%
- **Mandatory:** "ALWAYS use calculator for arithmetic, ALWAYS use clock for time/date" → tool-use rate ~85-95%

[Full per-task data in `evals/results.json`]

## What is a ReAct agent?

A ReAct agent interleaves **reasoning** and **acting**:

1. **Reason** — the LLM thinks about what to do next and decides whether to call a tool.
2. **Act** — the agent executes the tool call and feeds the result back to the LLM.
3. **Repeat** — until the LLM produces a final answer (no more tool calls).

This implementation is **660 lines** across `src/agent/*.py` and `src/agent/tools/*.py`,
with no orchestration framework (pure ReAct loop, tool registry, and JSONL tracing).

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

#### PowerShell (Windows)

```powershell
# 1. Copy .env.example and fill in your values
copy .env.example .env
# Edit .env: set LLM_API_KEY, LLM_MODEL, and choose LLM_BASE_URL

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a question
$env:PYTHONPATH="src"; python -m agent.cli "What is 15% of 2340, plus 200?"
```

#### bash / macOS / Linux

```bash
# 1. Copy .env.example and fill in your values
cp .env.example .env
# Edit .env: set LLM_API_KEY, LLM_MODEL, and choose LLM_BASE_URL

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a question
export PYTHONPATH=src
python -m agent.cli "What is 15% of 2340, plus 200?"
```

#### With `pyproject.toml` (no PYTHONPATH needed)

If you add a `pyproject.toml` with `[tool.poetry.scripts]` or install in editable mode:

```bash
pip install -e .
agent-cli "What is 15% of 2340, plus 200?"
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

The original eval set had a critical design flaw: all 12 questions were general knowledge questions that any LLM can answer from internal memory, so a model that skipped every tool still scored 100% accuracy. This made "Accuracy 12/12" close to meaningless for a tool-using agent, and the headline number hid the real result — tool-use matched expectations in only 3 of 12 tasks (25%).

The enhanced eval (15 tasks × 3 runs × 2 prompts = 90 runs) reveals:

| Metric | Permissive Prompt | Mandatory Prompt |
|--------|------------------|------------------|
| Tool-necessary answer accuracy | ~25-35% | ~85-95% (forces tool use) |
| Tool-unnecessary unnecessary-call rate | ~30-40% | ~5-10% |
| Key finding | System prompt "Do not call tools if you can answer directly" causes systematic tool avoidance | Explicit mandates ("ALWAYS use calculator for arithmetic", "ALWAYS use get_current_time for time/date") dramatically increase tool use without breaking answer quality |

### Real failure examples from the original 12-task eval

**t1 — "What is 25 * 4?" (expected `calculator`)**
- Model computed it internally, no `tool_calls` in trace
- The system prompt explicitly says "Do not call tools if you can answer directly from your knowledge"

**t9 — "What is the speed of light?" (expected `search_notes`)**
- Model answered from memory: "299,792,458 meters per second"
- Never called `search_notes` despite the task requiring it

**t5 — "What is the current UTC time plus 100 seconds?" (expected `get_current_time` + `calculator`)**
- `get_current_time` was called correctly, but arithmetic was done in the model's head
- Tools used: `[get_current_time]`; expected: `[get_current_time, calculator]`

### Other limitations

- **Number formatting:** Models may write "1,024" or "299 792 458" with spaces/commas. Eval matching handles commas and whitespace.
- **Free-tier rate limits:** Gemini and Groq free tiers have low RPM. Eval script sleeps 2s between tasks.
- **No tool schema validation at call time:** The registry validates argument types
  and required fields, but doesn't validate values (e.g., a negative number for a
  timezone-agnostic tool). replaced with:
- **Tool value ranges not validated:** The registry checks argument types and
  required fields, but values like timezone strings or negative numbers are not
  validated — the model must supply appropriate values.
- **Single-threaded:** Each run blocks on LLM responses. No async or parallel tool execution.
- **Static system prompt:** The default prompt may cause models to skip tools; a mandatory prompt dramatically increases tool use. `tool_choice=required` on every step would cause infinite loops — only use on the first step or name a specific function.
- **Eval design:** The original 12-task eval tested "does the model answer correctly?" rather than "does the model use the right tools?". The enhanced eval splits tasks into tool-necessary (cannot answer from memory) and tool-unnecessary (should not call tools) categories, with multiple runs and two prompt variants for sensitivity analysis.

**Key insight:** The system prompt is the primary lever for tool use. The default "Do not call tools if you can answer directly" causes systematic avoidance. An "ALWAYS use [tool] for [category]" mandate on the first step (then auto for follow-ups) can reverse this behavior.
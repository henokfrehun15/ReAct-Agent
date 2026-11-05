"""Command-line interface for the ReAct agent.

Usage:
    python -m agent.cli "question"        # run one question
    python -m agent.cli --verbose "question"  # with step-by-step output
    python -m agent.cli                    # interactive prompt loop
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Optional

from agent.agent import run_agent
from agent.config import Config, ConfigurationError, load_config
from agent.trace import TraceWriter


def _make_trace_writer(config: Config) -> tuple[TraceWriter, Path]:
    run_id = uuid.uuid4().hex[:8]
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run-{run_id}.jsonl"
    return TraceWriter(run_id, log_dir=log_dir), log_path


def _print_verbose_events(log_path: Path) -> None:
    if not log_path.exists():
        return
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            etype = event.get("type", "")
            if etype == "user":
                print(f"  [user] {event.get('content', '')}")
            elif etype == "assistant":
                content = event.get("content", "")
                if content:
                    print(f"  [assistant] {content}")
                tc = event.get("tool_calls")
                if tc:
                    for c in tc:
                        print(f"  [tool_call] {c['function']['name']}({c['function']['arguments']})")
            elif etype == "tool_call":
                print(f"  [tool_call] {event.get('tool')}({event.get('args')}) [id={event.get('call_id')}]")
            elif etype == "tool_result":
                print(f"  [tool_result] {event.get('tool')} -> {event.get('result', '')[:120]} [{event.get('latency_ms')}ms]")
            elif etype == "error":
                print(f"  [error] {event.get('content', '')}")
            elif etype == "final":
                print(f"  [final] {event.get('content', '')}")


def run_single(question: str, config: Config, verbose: bool) -> int:
    trace_writer, log_path = _make_trace_writer(config)
    result = run_agent(question, config=config, trace_writer=trace_writer)

    if verbose:
        _print_verbose_events(log_path)

    print(result.final_answer)
    print(f"Steps used: {result.steps_used}")
    print(f"Stopped reason: {result.stopped_reason}")
    print(f"Log file: {log_path}")
    return 0


def run_interactive(config: Config, verbose: bool) -> int:
    print("ReAct Agent (type 'exit' or 'quit' to exit)")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            return 0

        trace_writer, log_path = _make_trace_writer(config)
        result = run_agent(question, config=config, trace_writer=trace_writer)

        if verbose:
            _print_verbose_events(log_path)

        print(f"\n{result.final_answer}")
        print(f"[steps={result.steps_used}, reason={result.stopped_reason}, log={log_path}]")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ReAct agent CLI")
    parser.add_argument("question", nargs="?", help="Question to ask the agent")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print each step")
    args = parser.parse_args(argv)

    try:
        config = load_config()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    if args.question:
        return run_single(args.question, config, verbose=args.verbose)
    return run_interactive(config, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
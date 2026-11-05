"""Run all eval tasks with the real LLM and report results.

Reads evals/tasks.jsonl, runs each task through the agent, compares answers,
tracks tools used vs expected, and saves evals/results.json.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from agent.agent import run_agent
from agent.config import Config, ConfigurationError, load_config

TASKS_FILE = Path(__file__).resolve().parent / "tasks.jsonl"
RESULTS_FILE = Path(__file__).resolve().parent / "results.json"
SLEEP_BETWEEN_TASKS = 2  # seconds, to respect free-tier rate limits


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_number(text: str):
    """Return the first number found in text, or None."""
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if m:
        try:
            return float(m.group())
        except ValueError:
            return None
    return None


def _answers_match(actual: str, expected: str) -> bool:
    """Match by normalized substring or numeric comparison."""
    act_norm = _normalize(actual)
    exp_norm = _normalize(expected)

    # Numeric comparison
    act_num = _extract_number(act_norm)
    exp_num = _extract_number(exp_norm)
    if act_num is not None and exp_num is not None:
        return abs(act_num - exp_num) < 1e-6

    # Normalized substring
    return exp_norm in act_norm


def _failure_category(result, answer_ok: bool, tools_ok: bool) -> str:
    if result.stopped_reason == "error":
        return "error"
    if result.stopped_reason == "max_steps":
        return "max_steps"
    if not tools_ok:
        return "wrong_tools"
    if not answer_ok:
        return "wrong_answer"
    return "pass"


def main() -> int:
    try:
        config = load_config()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    tasks = []
    with open(TASKS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    print(f"Running {len(tasks)} eval tasks...\n")

    results = []
    total_latency = 0.0
    total_steps = 0

    for i, task in enumerate(tasks):
        task_id = task["id"]
        question = task["question"]
        expected_answer = task["expected_answer"]
        expected_tools = task.get("expected_tools", [])

        print(f"[{i+1}/{len(tasks)}] {task_id}: {question[:60]}...", end=" ")

        start = time.perf_counter()
        result = run_agent(question, config=config)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        answer_ok = _answers_match(result.final_answer, expected_answer)
        tools_used = sorted(set(result.tools_used))
        expected_sorted = sorted(expected_tools)
        tools_ok = tools_used == expected_sorted

        category = _failure_category(result, answer_ok, tools_ok)
        passed = category == "pass"

        total_latency += elapsed_ms
        total_steps += result.steps_used

        results.append({
            "id": task_id,
            "question": question,
            "expected_answer": expected_answer,
            "expected_tools": expected_tools,
            "final_answer": result.final_answer,
            "answer_match": answer_ok,
            "steps_used": result.steps_used,
            "tools_used": tools_used,
            "tools_expected": expected_tools,
            "tools_match": tools_ok,
            "latency_ms": elapsed_ms,
            "stopped_reason": result.stopped_reason,
            "passed": passed,
            "failure_category": category,
            "run_id": result.run_id,
        })

        status = "PASS" if passed else f"FAIL ({category})"
        print(f"{status} ({result.steps_used} steps, {elapsed_ms}ms)")

        if i < len(tasks) - 1:
            time.sleep(SLEEP_BETWEEN_TASKS)

    # Summary
    n_passed = sum(1 for r in results if r["passed"])
    accuracy = n_passed / len(results) if results else 0
    avg_steps = total_steps / len(results) if results else 0
    avg_latency = total_latency / len(results) if results else 0

    print("\n" + "=" * 60)
    print(f"Overall accuracy: {n_passed}/{len(results)} ({accuracy:.1%})")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average latency: {avg_latency:.0f}ms")
    print("=" * 60)

    # Save results
    summary = {
        "total": len(results),
        "passed": n_passed,
        "accuracy": accuracy,
        "avg_steps": avg_steps,
        "avg_latency_ms": avg_latency,
        "tasks": results,
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_FILE}")

    return 0 if n_passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
"""Run all eval tasks with the real LLM and report results.

Reads evals/tasks.jsonl, runs each task through the agent (multiple runs per
task, two system prompts), compares answers, tracks tools used vs expected,
and saves evals/results.json.

Task types:
  tool_necessary   - Cannot be answered correctly from memory; a tool is required.
  tool_unnecessary - Should be answered without any tool call (control).
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from agent.agent import run_agent
from agent.config import Config, ConfigurationError, load_config

TASKS_FILE = Path(__file__).resolve().parent / "tasks.jsonl"
RESULTS_FILE = Path(__file__).resolve().parent / "results.json"
SLEEP_BETWEEN_TASKS = 2  # seconds, to respect free-tier rate limits
RUNS_PER_TASK = 3  # how many times to run each task per prompt

# Prompt variants to compare
PROMPTS = ["permissive", "mandatory"]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_all_numbers(text: str) -> list[float]:
    """Return all numbers found in text (commas stripped)."""
    nums = []
    for m in re.finditer(r"-?\d+(?:[.,]\d+)?", text):
        try:
            nums.append(float(m.group().replace(",", "")))
        except ValueError:
            pass
    return nums


def _parse_utc_time(text: str) -> datetime | None:
    """Try to parse a UTC datetime from free-form text.

    Handles formats like "2026-11-06 06:10 UTC", "November 6, 2026", etc.
    """
    m = re.search(r"(\d{4}-\d{2}-\d{2})[^\d]{0,3}(\d{1,2}):(\d{2})", text)
    if m:
        try:
            return datetime(
                int(m.group(1)[:4]), int(m.group(1)[5:7]), int(m.group(1)[8:10]),
                int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc,
            )
        except ValueError:
            pass
    m = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})",
        text,
        re.IGNORECASE,
    )
    if m:
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
        }
        try:
            return datetime(
                int(m.group(3)), months[m.group(1).lower()], int(m.group(2)),
                tzinfo=timezone.utc,
            )
        except ValueError:
            pass
    return None


def _answers_match(actual: str, expected: str | None, check: str | None = None) -> bool:
    """Match by normalized substring, numeric comparison, or special check."""
    if expected is None and check is None:
        return True

    act_norm = _normalize(actual)

    if check == "time_within_1_minute_utc":
        dt = _parse_utc_time(act_norm)
        if dt is None:
            return False
        now = datetime.now(timezone.utc)
        return abs((now - dt).total_seconds()) <= 60

    if check == "time_math_within_1_minute_utc":
        dt = _parse_utc_time(act_norm)
        if dt is None:
            return False
        now = datetime.now(timezone.utc)
        expected_time = now + timedelta(seconds=100)
        return abs((expected_time - dt).total_seconds()) <= 60

    if check == "date_45_days_utc":
        dt = _parse_utc_time(act_norm)
        if dt is None:
            return False
        now = datetime.now(timezone.utc)
        expected_date = (now + timedelta(days=45)).date()
        return dt.date() == expected_date

    exp_norm = _normalize(expected)
    act_simple = act_norm.replace(",", "")
    exp_simple = exp_norm.replace(",", "")
    if exp_simple in act_simple:
        return True

    act_nospace = re.sub(r"\s+", "", act_norm)
    exp_nospace = re.sub(r"\s+", "", exp_norm)
    if exp_nospace in act_nospace:
        return True

    act_nums = _extract_all_numbers(act_norm)
    exp_nums = _extract_all_numbers(exp_norm)
    if exp_nums:
        return any(abs(a - e) < 1e-6 for a in act_nums for e in exp_nums)

    return False


def _failure_category(
    answer_ok: bool,
    tools_used: list[str],
    expected_tools: list[str],
    task_type: str,
) -> str:
    """Classify a run result into a specific failure category (or "pass")."""
    if not answer_ok:
        return "wrong_answer"

    tools_used_set = set(tools_used)
    expected_set = set(expected_tools)

    if task_type == "tool_unnecessary":
        if tools_used_set:
            return "unnecessary_tool_used"
        return "pass"

    missing = expected_set - tools_used_set
    extra = tools_used_set - expected_set
    if missing:
        return "missing_tool"
    if extra:
        return "extra_tool"
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

    print(
        f"Running {len(tasks)} tasks x {RUNS_PER_TASK} runs x "
        f"{len(PROMPTS)} prompts...\n"
    )

    all_results = []

    for prompt_name in PROMPTS:
        print(f"=== Prompt: {prompt_name} ===")
        for i, task in enumerate(tasks):
            task_id = task["id"]
            question = task["question"]
            expected_answer = task.get("expected_answer")
            expected_tools = task.get("expected_tools", [])
            task_type = task.get("type", "tool_necessary")
            check = task.get("check")

            run_results = []
            for run_idx in range(RUNS_PER_TASK):
                print(
                    f"  [{i+1}/{len(tasks)}] {task_id} run {run_idx+1}/{RUNS_PER_TASK}: "
                    f"{question[:50]}...",
                    end=" ",
                )

                start = time.perf_counter()
                result = run_agent(
                    question, config=config, system_prompt=prompt_name,
                )
                elapsed_ms = int((time.perf_counter() - start) * 1000)

                answer_ok = _answers_match(result.final_answer, expected_answer, check)
                tools_used = sorted(set(result.tools_used))
                category = _failure_category(
                    answer_ok, result.tools_used, expected_tools, task_type,
                )
                passed = answer_ok and category == "pass"

                run_results.append({
                    "run": run_idx + 1,
                    "final_answer": result.final_answer,
                    "answer_match": answer_ok,
                    "steps_used": result.steps_used,
                    "tools_used": tools_used,
                    "tools_expected": expected_tools,
                    "tools_match": tools_used == sorted(expected_tools),
                    "latency_ms": elapsed_ms,
                    "stopped_reason": result.stopped_reason,
                    "passed": passed,
                    "failure_category": category,
                    "run_id": result.run_id,
                })

                status = "PASS" if passed else f"FAIL ({category})"
                print(f"{status} ({result.steps_used} steps, {elapsed_ms}ms)")

                if run_idx < RUNS_PER_TASK - 1:
                    time.sleep(SLEEP_BETWEEN_TASKS)

            n_answer_ok = sum(1 for r in run_results if r["answer_match"])
            n_pass = sum(1 for r in run_results if r["passed"])
            avg_steps = sum(r["steps_used"] for r in run_results) / len(run_results)
            avg_latency = sum(r["latency_ms"] for r in run_results) / len(run_results)

            tool_use_count = sum(
                1 for r in run_results if set(r["tools_used"]) & set(expected_tools)
            )
            tool_use_rate = tool_use_count / len(run_results) if run_results else 0

            all_results.append({
                "id": task_id,
                "type": task_type,
                "question": question,
                "expected_answer": expected_answer,
                "expected_tools": expected_tools,
                "check": check,
                "prompt": prompt_name,
                "runs": run_results,
                "answer_ok_rate": n_answer_ok / len(run_results),
                "pass_rate": n_pass / len(run_results),
                "avg_steps": avg_steps,
                "avg_latency_ms": avg_latency,
                "tool_use_rate": tool_use_rate,
            })

        if prompt_name != PROMPTS[-1]:
            time.sleep(SLEEP_BETWEEN_TASKS)

    # ---- Summary metrics ----
    necessary = [r for r in all_results if r["type"] == "tool_necessary"]
    unnecessary = [r for r in all_results if r["type"] == "tool_unnecessary"]

    overall_answer_ok = sum(r["answer_ok_rate"] for r in all_results) / len(all_results)
    overall_pass = sum(r["pass_rate"] for r in all_results) / len(all_results)

    necessary_answer_ok = (
        sum(r["answer_ok_rate"] for r in necessary) / len(necessary) if necessary else 0
    )
    necessary_tool_use = (
        sum(r["tool_use_rate"] for r in necessary) / len(necessary) if necessary else 0
    )
    necessary_pass = (
        sum(r["pass_rate"] for r in necessary) / len(necessary) if necessary else 0
    )

    unnecessary_answer_ok = (
        sum(r["answer_ok_rate"] for r in unnecessary) / len(unnecessary) if unnecessary else 0
    )
    unnecessary_tool_rate = (
        sum(1 for r in unnecessary for run in r["runs"] if run["tools_used"])
        / sum(len(r["runs"]) for r in unnecessary)
        if unnecessary else 0
    )
    unnecessary_pass = (
        sum(r["pass_rate"] for r in unnecessary) / len(unnecessary) if unnecessary else 0
    )

    # Prompt comparison
    prompt_stats = {}
    for pname in PROMPTS:
        p_tasks = [r for r in all_results if r["prompt"] == pname]
        p_necessary = [r for r in p_tasks if r["type"] == "tool_necessary"]
        prompt_stats[pname] = {
            "overall_pass_rate": sum(r["pass_rate"] for r in p_tasks) / len(p_tasks),
            "necessary_answer_ok": (
                sum(r["answer_ok_rate"] for r in p_necessary) / len(p_necessary)
                if p_necessary else 0
            ),
            "necessary_tool_use": (
                sum(r["tool_use_rate"] for r in p_necessary) / len(p_necessary)
                if p_necessary else 0
            ),
            "necessary_pass": (
                sum(r["pass_rate"] for r in p_necessary) / len(p_necessary)
                if p_necessary else 0
            ),
        }

    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    print(f"Overall answer accuracy: {overall_answer_ok:.1%}")
    print(f"Overall pass rate:       {overall_pass:.1%}")
    print()
    print("Tool-necessary tasks:")
    print(f"  Answer accuracy: {necessary_answer_ok:.1%}")
    print(f"  Tool-use rate:   {necessary_tool_use:.1%}")
    print(f"  Pass rate:       {necessary_pass:.1%}")
    print()
    print("Tool-unnecessary (control) tasks:")
    print(f"  Answer accuracy: {unnecessary_answer_ok:.1%}")
    print(f"  Unnecessary tool-use rate: {unnecessary_tool_rate:.1%}")
    print(f"  Pass rate:       {unnecessary_pass:.1%}")
    print()
    print("Prompt comparison (tool-necessary tasks):")
    for pname, stats in prompt_stats.items():
        print(
            f"  {pname:12s} answer_ok={stats['necessary_answer_ok']:.1%} "
            f"tool_use={stats['necessary_tool_use']:.1%} pass={stats['necessary_pass']:.1%}"
        )
    print("=" * 60)

    # Save results
    summary = {
        "total_tasks": len(tasks),
        "runs_per_task": RUNS_PER_TASK,
        "prompts": PROMPTS,
        "overall": {
            "answer_accuracy": overall_answer_ok,
            "pass_rate": overall_pass,
        },
        "tool_necessary": {
            "answer_accuracy": necessary_answer_ok,
            "tool_use_rate": necessary_tool_use,
            "pass_rate": necessary_pass,
        },
        "tool_unnecessary": {
            "answer_accuracy": unnecessary_answer_ok,
            "unnecessary_tool_use_rate": unnecessary_tool_rate,
            "pass_rate": unnecessary_pass,
        },
        "prompt_comparison": prompt_stats,
        "tasks": all_results,
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_FILE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
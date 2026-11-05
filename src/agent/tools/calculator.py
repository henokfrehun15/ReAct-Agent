"""Safe arithmetic expression evaluator using the ``ast`` module (no eval/exec).

Supports numbers, + - * / ** %, parentheses and unary minus. Everything else
is rejected with a clear error.
"""
from __future__ import annotations

import ast
import operator

from .registry import tool

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_MAX_EXPONENT = 100_000      # reject things like 9 ** 9 ** 9
_MAX_RESULT_DIGITS = 1_000   # cap result size


@tool(
    "calculator",
    "Evaluate a safe arithmetic expression and return the numeric result.",
    {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression using numbers, + - * / ** %, parentheses and unary minus.",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
)
def calculator(expression: str) -> str:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-empty string")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression: {exc}") from exc
    value = _eval(tree.body)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("result is not a finite number")
        if value.is_integer():
            value = int(value)
    elif isinstance(value, int) and len(str(abs(value))) > _MAX_RESULT_DIGITS:
        raise ValueError("result has too many digits")
    return str(value)


def _eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"unsupported constant: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        left = _eval(node.left)
        right = _eval(node.right)
        if type(node.op) is ast.Pow and isinstance(right, (int, float)) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"exponent {right} is too large (max {_MAX_EXPONENT})")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(_eval(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")
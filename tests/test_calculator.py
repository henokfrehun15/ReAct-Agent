"""Tests for the calculator tool. No network, no API key needed."""
from agent.tools.calculator import calculator
from agent.tools.registry import call

VALID = [
    ("2 + 3 * 4", "14"),
    ("(2 + 3) * 4", "20"),
    ("10 / 4", "2.5"),
    ("7 % 3", "1"),
    ("-5 + 3", "-2"),
    ("2 ** 10", "1024"),
    ("2 ** -1", "0.5"),
    ("((1 + 2) * 3)", "9"),
]


def test_valid_expressions():
    for expression, expected in VALID:
        assert calculator(expression) == expected, expression


def test_invalid_expressions_return_errors():
    for expression in [
        "__import__('os').system('ls')",
        "x + 1",
        "math.sqrt(4)",
        "[1, 2, 3]",
        "9 ** 9 ** 9",
        "10 ** 1000",
        "",
    ]:
        result = call("calculator", {"expression": expression})
        assert result.startswith("Error:"), (expression, result)


def test_registry_never_raises_on_bad_arguments():
    assert call("calculator", "not json").startswith("Error:")
    assert call("calculator", {"expression": 42}).startswith("Error:")
    assert "string" in call("calculator", {"expression": 42}).lower()
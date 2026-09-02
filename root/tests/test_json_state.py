import pytest

from src.json_state import JasonStateMacheine


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "-0",
        "12",
        "-12",
        "0.5",
        "-0.5",
        "12.34",
        "1e5",
        "1e-5",
        "1e+5",
        "1.5e-2",
    ],
)
def test_valid_numbers(value):
    machine = JasonStateMacheine()

    for char in value:
        assert machine.transition(char) is True


@pytest.mark.parametrize(
    "value",
    [
        "01",
        "-01",
        "1.",
        "1e",
        "1e-",
        "1e+",
        "1e--5",
        "1e+-5",
        "1e5-6",
    ],
)
def test_invalid_numbers(value):
    machine = JasonStateMacheine()

    results = []

    for char in value:
        results.append(machine.transition(char))

    assert False in results

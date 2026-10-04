"""Focused Chapter 2.1 checks; run from the project root with pytest.

These tests exercise grammar and masking without loading the LLM.
"""

import copy
import json

import pytest

from src.json_state import JasonStateMacheine
from src.masking import mask_invalid_tokens
from src.schema_state import SchemaState


@pytest.mark.parametrize(
    "text",
    [
        "[1 2]",
        "[1\t2]",
        "[1e 2]",
        "[1. 2]",
        "[- 2]",
        "[t rue]",
        "[fal se]",
        "[nu ll]",
        "[tr\nue]",
        "1 2",
    ],
)
def test_rejects_whitespace_inside_numbers_and_literals(text):
    """Whitespace must not join separated fragments into one JSON value."""
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)

    machine = JasonStateMacheine()
    accepted = all(machine.transition(char) for char in text)
    assert not (accepted and machine.is_complete()), text


@pytest.mark.parametrize(
    "text",
    [
        "[1 , 2]",
        "[true , false , null]",
        "1 \t\n",
        "[1e+2, -3]",
        "[1e-2, -3]",
        "[1e+2, [-3]]",
        '{"a":1e+2,"b":-3}',
        '{"a":1e+2,"b":{"c":-3}}',
    ],
)
def test_accepts_separating_whitespace_and_fresh_negative_numbers(text):
    """Valid separators and previous exponent signs must not block values."""
    json.loads(text)
    machine = JasonStateMacheine()
    for index, char in enumerate(text):
        assert machine.transition(char), (text, index, char)
    assert machine.is_complete(), text


@pytest.mark.parametrize(
    "prefix, candidate, allowed",
    [
        (r'"\u', "0041", True),
        (r'"\u', "ABCD", True),
        (r'"\u', "abcd", True),
        (r'"\u', "00", True),
        (r'"\u', '0041"', True),
        (r'"\u', "00G1", False),
        (r'"\u', '00"', False),
        (r'"\u00', '41"', True),
        (r'"\u00', 'G1"', False),
    ],
)
def test_masking_checks_pending_unicode_digits(prefix, candidate, allowed):
    """Valid full or partial hex tokens survive; malformed escapes do not."""
    machine = JasonStateMacheine()
    for char in prefix:
        assert machine.transition(char)

    before = copy.deepcopy(vars(machine))
    result = mask_invalid_tokens(
        [1.0], machine, {0: candidate}, SchemaState(""), [], []
    )
    assert result[0] == (1.0 if allowed else float("-inf"))
    assert vars(machine) == before


@pytest.mark.parametrize(
    "candidate, allowed",
    [(" , 2]", True), (" 2]", False)],
)
def test_number_boundary_simulation_preserves_live_state(candidate, allowed):
    """Accepting or rejecting a candidate must leave live state untouched."""
    machine = JasonStateMacheine()
    for char in "[1":
        assert machine.transition(char)

    before = copy.deepcopy(vars(machine))
    assert machine.validate_candidate(candidate) is allowed
    assert vars(machine) == before

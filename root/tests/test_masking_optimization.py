import math
import pytest

from src.masking import mask_invalid_tokens
from src.schema_state import SchemaState
from src.json_state import JasonStateMacheine, JasonStates


def run_mask(machine, tokens):
    logits = [1.0] * len(tokens)
    vocab_map = {i: token for i, token in enumerate(tokens)}

    result = mask_invalid_tokens(logits, machine, vocab_map, SchemaState(), [], [])

    return {token: result[i] for i, token in enumerate(tokens)}


def assert_mask_matches_jason(machine, tokens):
    expected = {token: machine.validate_candidate(token) for token in tokens}

    result = run_mask(machine, tokens)

    for token in tokens:
        if expected[token]:
            assert not math.isinf(result[token]), (
                f"Cheap filter incorrectly masked valid token: {token!r} "
                f"in state {machine.state}"
            )
        else:
            assert result[token] == float("-inf"), (
                f"Invalid token was not masked: {token!r} "
                f"in state {machine.state}"
            )


def test_missing_vocab_ids_are_masked():
    machine = JasonStateMacheine()

    logits = [1.0, 1.0, 1.0]
    vocab_map = {
        0: "{",
        2: "[",
    }

    result = mask_invalid_tokens(logits, machine, vocab_map, SchemaState(), [], [])

    assert result[1] == float("-inf")


def test_expect_colon():
    machine = JasonStateMacheine()
    machine.state = JasonStates.EXPECT_COLON

    tokens = [
        ":",
        ": ",
        ":123",
        " :",
        "\t:",
        "\n:",
        "\r:",
        "hello",
        "{",
        "}",
        "[",
        "]",
        '"',
        "1",
        ",",
    ]

    assert_mask_matches_jason(machine, tokens)


def test_string_escape():
    machine = JasonStateMacheine()
    machine.state = JasonStates.IN_STRING_ESCAPE

    tokens = [
        '"',
        "\\",
        "/",
        "b",
        "f",
        "n",
        "r",
        "t",
        "u",
        '\\"',
        "\\n",
        "u1234",
        "uABCD",
        "uabcd",
        "x",
        " ",
        "\t",
        "\n",
        "{",
        "1",
    ]

    assert_mask_matches_jason(machine, tokens)


def test_after_value_inside_object():
    machine = JasonStateMacheine()
    machine.state = JasonStates.AFTER_VALUE
    machine.brackets = ["{"]

    tokens = [
        "}",
        ",",
        " }",
        "\t}",
        "\n}",
        "\r}",
        ", ",
        ',"',
        "]",
        "{",
        "[",
        '"',
        ":",
        "hello",
        "1",
    ]

    assert_mask_matches_jason(machine, tokens)


def test_after_value_inside_array():
    machine = JasonStateMacheine()
    machine.state = JasonStates.AFTER_VALUE
    machine.brackets = ["["]

    tokens = [
        "]",
        ",",
        " ]",
        "\t]",
        "\n]",
        "\r]",
        ", ",
        "}",
        "{",
        "[",
        '"',
        ":",
        "hello",
        "1",
    ]

    assert_mask_matches_jason(machine, tokens)


def test_after_value_wrong_closing_bracket():
    machine = JasonStateMacheine()
    machine.state = JasonStates.AFTER_VALUE
    machine.brackets = ["{"]

    tokens = [
        "]",
        " ]",
        "}",
    ]

    assert_mask_matches_jason(machine, tokens)


def test_expect_key_empty_object_allowed():
    machine = JasonStateMacheine()
    machine.state = JasonStates.EXPECT_KEY
    machine.brackets = ["{"]
    machine.allow_empty_object = True

    tokens = [
        '"',
        '"name"',
        "}",
        " }",
        "\t}",
        "\n}",
        "\r}",
        "hello",
        "1",
        "[",
        "{",
        ":",
        ",",
    ]

    assert_mask_matches_jason(machine, tokens)


def test_expect_key_empty_object_not_allowed():
    machine = JasonStateMacheine()
    machine.state = JasonStates.EXPECT_KEY
    machine.brackets = ["{"]
    machine.allow_empty_object = False

    tokens = [
        '"',
        '"name"',
        "}",
        " }",
        "\t}",
        "hello",
        "1",
        ",",
    ]

    assert_mask_matches_jason(machine, tokens)


def test_done_only_allows_whitespace():
    machine = JasonStateMacheine()
    machine.state = JasonStates.DONE

    tokens = [
        " ",
        "\t",
        "\n",
        "\r",
        "   ",
        "\t\n\r ",
        "{",
        "}",
        "[",
        "]",
        '"',
        "1",
        "true",
        "hello",
        ":",
        ",",
    ]

    assert_mask_matches_jason(machine, tokens)


@pytest.mark.parametrize(
    "state",
    [
        JasonStates.START,
        JasonStates.EXPECT_VALUE,
    ],
)
def test_value_start_states(state):
    machine = JasonStateMacheine()
    machine.state = state

    tokens = [
        "{",
        "[",
        '"',
        "true",
        "false",
        "null",
        "0",
        "1",
        "9",
        "-1",
        "-0",
        "123",
        " ",
        "\t",
        "\n",
        "\r",
        " {",
        " [",
        ' "hello"',
        " true",
        " false",
        " null",
        " -123",
        "hello",
        "+1",
        "}",
        "]",
        ":",
        ",",
        ".",
    ]

    assert_mask_matches_jason(machine, tokens)


def test_in_array():
    machine = JasonStateMacheine()
    machine.state = JasonStates.IN_ARRAY
    machine.brackets = ["["]

    tokens = [
        "]",
        " ]",
        "{",
        "[",
        '"',
        "true",
        "false",
        "null",
        "0",
        "123",
        "-1",
        " ",
        "\t",
        "\n",
        "\r",
        " {",
        " [",
        ' "hello"',
        " true",
        " -123",
        "hello",
        "+1",
        "}",
        ":",
        ",",
    ]

    assert_mask_matches_jason(machine, tokens)


def test_in_string_is_left_to_full_validation():
    machine = JasonStateMacheine()
    machine.state = JasonStates.IN_STRING
    machine.string_role = "value"

    tokens = [
        "hello",
        "abc123",
        " ",
        "{",
        "[",
        ":",
        ",",
        '"',
        "\\",
        "\\n",
        "\n",
        "\t",
    ]

    assert_mask_matches_jason(machine, tokens)


def test_in_number_is_left_to_full_validation():
    machine = JasonStateMacheine()
    machine.transition("1")

    tokens = [
        "0",
        "23",
        ".5",
        "e2",
        "E2",
        "e+2",
        "e-2",
        "}",
        "]",
        ",",
        "hello",
        '"',
    ]

    assert_mask_matches_jason(machine, tokens)


def test_in_literal_is_left_to_full_validation():
    machine = JasonStateMacheine()
    machine.transition("t")

    tokens = [
        "r",
        "ru",
        "rue",
        "true",
        "u",
        "ue",
        "x",
        "false",
        "null",
        "1",
        " ",
    ]

    assert_mask_matches_jason(machine, tokens)


def test_valid_tokens_with_leading_whitespace_are_not_prefiltered():
    cases = [
        (
            JasonStates.EXPECT_COLON,
            [" :", "\t:", "\n:", "\r:"],
        ),
        (
            JasonStates.START,
            [" {", "\t[", '\n"', "\rtrue", " -1"],
        ),
        (
            JasonStates.EXPECT_VALUE,
            [" {", "\t[", '\n"', "\rfalse", " -1"],
        ),
        (
            JasonStates.IN_ARRAY,
            [" ]", " {", "\t[", '\n"', "\rnull", " -1"],
        ),
    ]

    for state, tokens in cases:
        machine = JasonStateMacheine()
        machine.state = state

        if state == JasonStates.IN_ARRAY:
            machine.brackets = ["["]

        assert_mask_matches_jason(machine, tokens)


def test_multi_character_token_can_start_valid_but_end_invalid():
    machine = JasonStateMacheine()
    machine.state = JasonStates.EXPECT_COLON

    tokens = [
        ":hello",
        ":{",
        ":]",
        ":1",
    ]

    assert_mask_matches_jason(machine, tokens)


def test_cheap_filter_does_not_replace_full_validation():
    machine = JasonStateMacheine()
    machine.state = JasonStates.START

    tokens = [
        "{]",
        "[}",
        "truX",
        "nulX",
        "falX",
        "-x",
    ]

    assert_mask_matches_jason(machine, tokens)

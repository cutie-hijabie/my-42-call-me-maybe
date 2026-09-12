from src.json_state import JasonStateMacheine
from src.masking import mask_invalid_tokens


def test_masking_with_real_json_machine():
    machine = JasonStateMacheine()

    logits = [
        5.0,
        8.0,
        10.0,
        3.0,
        7.0,
        4.0,
    ]

    vocab_map = {
        0: "{",
        1: "[",
        2: ":",
        3: "}",
        4: '"',
        5: "true",
    }

    result = mask_invalid_tokens(logits, machine, vocab_map)

    assert result[0] == 5.0
    assert result[1] == 8.0
    assert result[2] == float("-inf")
    assert result[3] == float("-inf")
    assert result[4] == 7.0
    assert result[5] == 4.0


def test_masking_missing_vocab_id():
    machine = JasonStateMacheine()

    logits = [
        5.0,
        8.0,
        10.0,
    ]

    vocab_map = {
        0: "{",
        1: "[",
    }

    result = mask_invalid_tokens(logits, machine, vocab_map)

    assert result[0] == 5.0
    assert result[1] == 8.0
    assert result[2] == float("-inf")

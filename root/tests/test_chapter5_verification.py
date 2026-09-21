import time

from src.json_state import JasonStateMacheine, JasonStates
from src.masking import mask_invalid_tokens
from src.schema_state import SchemaState
from src.generation import generate_constrained
from src.llm_wrapper import LLMWrapper
from src.vocab import build_vocab_map


def test_optimization_reduces_full_validation_calls():
    machine = JasonStateMacheine()
    machine.state = JasonStates.EXPECT_COLON

    tokens = [
        ":",
        ": ",
        " :",
        "\t:",
        "\n:",
        "\r:",
        "hello",
        "world",
        "123",
        "true",
        "false",
        "null",
        "{",
        "}",
        "[",
        "]",
        '"',
        ",",
    ]

    vocab_map = {token_id: token for token_id, token in enumerate(tokens)}

    logits = [1.0] * len(tokens)

    original_validate = machine.validate_candidate
    validation_calls = 0

    def counted_validate(candidate):
        nonlocal validation_calls
        validation_calls += 1
        return original_validate(candidate)

    machine.validate_candidate = counted_validate

    mask_invalid_tokens(
        logits,
        machine,
        vocab_map,
        SchemaState(),
        [],
        [],
    )

    assert validation_calls < len(tokens)

    print()
    print("total mapped tokens:", len(tokens))
    print("full validate_candidate calls:", validation_calls)
    print(
        "cheaply rejected:",
        len(tokens) - validation_calls,
    )


def test_optimization_skips_obvious_expect_colon_tokens():
    machine = JasonStateMacheine()
    machine.state = JasonStates.EXPECT_COLON

    tokens = [
        ":",
        " :",
        "hello",
        "123",
        "{",
        '"',
    ]

    vocab_map = {token_id: token for token_id, token in enumerate(tokens)}

    logits = [1.0] * len(tokens)

    original_validate = machine.validate_candidate
    validated_tokens = []

    def recorded_validate(candidate):
        validated_tokens.append(candidate)
        return original_validate(candidate)

    machine.validate_candidate = recorded_validate

    mask_invalid_tokens(
        logits,
        machine,
        vocab_map,
        SchemaState(),
        [],
        [],
    )

    assert ":" in validated_tokens
    assert " :" in validated_tokens

    assert "hello" not in validated_tokens
    assert "123" not in validated_tokens
    assert "{" not in validated_tokens
    assert '"' not in validated_tokens


def test_20_generations_finish_under_five_minutes():
    prompts = [
        "Return only a JSON object with a name and age.",
        "Return only a JSON array containing three numbers.",
        "Return only JSON containing a boolean.",
        "Return only a JSON object with city and country.",
    ]

    start = time.perf_counter()

    model = LLMWrapper()
    vocab_map = build_vocab_map(model)

    for run in range(20):
        prompt = prompts[run % len(prompts)]

        prompt_ids = model.wrapper_encode(prompt)

        generate_constrained(
            prompt,
            max_length=len(prompt_ids) + 100,
            model=model,
            vocab_map=vocab_map,
            schema_state=SchemaState(),
            function_names=[],
            definitions=[],
        )

    elapsed = time.perf_counter() - start

    print()
    print(f"20 constrained generations: {elapsed:.2f} seconds")

    assert elapsed < 300


def test_invalid_token_is_excluded_at_prefilter():
    machine = JasonStateMacheine()
    machine.state = JasonStates.EXPECT_COLON

    tokens = [
        ":",
        "hello",
    ]

    vocab_map = {
        0: ":",
        1: "hello",
    }

    logits = [
        1.0,
        10.0,
    ]

    original_validate = machine.validate_candidate
    validated_tokens = []

    def recorded_validate(candidate):
        validated_tokens.append(candidate)
        return original_validate(candidate)

    machine.validate_candidate = recorded_validate

    result = mask_invalid_tokens(
        logits,
        machine,
        vocab_map,
        SchemaState(),
        [],
        [],
    )

    assert result[0] == 1.0
    assert result[1] == float("-inf")

    assert ":" in validated_tokens
    assert "hello" not in validated_tokens

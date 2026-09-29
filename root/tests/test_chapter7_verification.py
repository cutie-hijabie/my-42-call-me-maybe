import json

from src.generation import generate_constrained
from src.llm_wrapper import LLMWrapper
from src.schema import InputFileDefinition
from src.schema_state import SchemaState
from src.vocab import build_vocab_map
from src.json_state import JasonStateMacheine
from src.masking import mask_invalid_tokens
from src.json_state import JasonStateMacheine, JasonStates
from src.masking import mask_invalid_tokens
from src.schema_state import SchemaState, SchemaStage


def test_model_selects_greet_function_and_extracts_name():
    with open("data/input/functions_definition.json", "r") as file:
        raw_definitions = json.load(file)

    definitions = InputFileDefinition.model_validate(raw_definitions).root
    function_names = [definition.name for definition in definitions]

    model = LLMWrapper()
    vocab_map = build_vocab_map(model)

    prompt = "Greet shrek"
    prompt_ids = model.wrapper_encode(prompt)

    result_ids = generate_constrained(
        prompt,
        max_length=len(prompt_ids) + 100,
        model=model,
        vocab_map=vocab_map,
        schema_state=SchemaState(),
        function_names=function_names,
        definitions=definitions,
    )

    generated_ids = result_ids[len(prompt_ids) :]
    generated_text = model.wrapper_decode(generated_ids)

    print("GENERATED:", repr(generated_text))

    result = json.loads(generated_text)

    assert result["name"] == "fn_greet"
    assert result["parameters"]["name"] == "shrek"


from src.prompt import build_prompt
from src.schema import InputDefinition


def test_build_prompt():
    definitions = [
        InputDefinition.model_validate(
            {
                "name": "fn_add_numbers",
                "description": "Add two numbers together.",
                "parameters": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "returns": {"type": "number"},
            }
        ),
        InputDefinition.model_validate(
            {
                "name": "fn_greet",
                "description": "Generate a greeting.",
                "parameters": {
                    "name": {"type": "string"},
                },
                "returns": {"type": "string"},
            }
        ),
    ]

    result = build_prompt("Greet shrek", definitions)

    print("\nBUILT PROMPT:")
    print(result)

    assert "fn_add_numbers(a:number, b:number)" in result
    assert "fn_greet(name:string)" in result
    assert "Request: Greet shrek" in result


def test_model_selects_greet_function_and_extracts_name():
    with open("data/input/functions_definition.json", "r") as file:
        raw_definitions = json.load(file)

    definitions = InputFileDefinition.model_validate(raw_definitions).root
    function_names = [definition.name for definition in definitions]

    model = LLMWrapper()
    vocab_map = build_vocab_map(model)

    user_prompt = "Greet shrek"
    prompt = build_prompt(user_prompt, definitions)
    prompt_ids = model.wrapper_encode(prompt)

    print("\nMODEL PROMPT:")
    print(prompt)

    result_ids = generate_constrained(
        prompt,
        max_length=len(prompt_ids) + 100,
        model=model,
        vocab_map=vocab_map,
        schema_state=SchemaState(),
        function_names=function_names,
        definitions=definitions,
    )

    generated_ids = result_ids[len(prompt_ids) :]
    generated_text = model.wrapper_decode(generated_ids)

    print("\nGENERATED:", repr(generated_text))

    result = json.loads(generated_text)

    assert result["name"] == "fn_greet"
    assert result["parameters"]["name"] == "shrek"


def test_function_call_must_start_as_json_object():
    json_machine = JasonStateMacheine()
    schema_state = SchemaState()

    vocab_map = {
        0: "{",
        1: "1",
        2: "-",
        3: "[",
        4: "null",
        5: '"hello"',
    }

    logits = [1.0] * len(vocab_map)

    result = mask_invalid_tokens(
        logits,
        json_machine,
        vocab_map,
        schema_state,
        function_names=[],
        definitions=[],
    )

    assert result[0] != float("-inf"), "{ should be allowed"

    assert result[1] == float("-inf"), "number should not be allowed"
    assert result[2] == float("-inf"), "- should not be allowed"
    assert result[3] == float("-inf"), "array should not be allowed"
    assert result[4] == float("-inf"), "null should not be allowed"
    assert result[5] == float("-inf"), "top-level string should not be allowed"


def test_prompt_stage_must_start_with_object():
    json_machine = JasonStateMacheine()
    schema_state = SchemaState()

    vocab_map = {
        0: "{",
        1: "1",
        2: "-",
        3: "[",
        4: "null",
        5: '"hello"',
    }

    logits = [1.0] * len(vocab_map)

    result = mask_invalid_tokens(
        logits,
        json_machine,
        vocab_map,
        schema_state,
        function_names=[],
        definitions=[],
    )

    assert result[0] != float("-inf")

    for token_id in (1, 2, 3, 4, 5):
        assert result[token_id] == float("-inf")


def test_prompt_stage_first_key_must_be_prompt():
    json_machine = JasonStateMacheine()
    json_machine.transition("{")

    schema_state = SchemaState()

    assert schema_state.stage == SchemaStage.PROMPT
    assert json_machine.state == JasonStates.EXPECT_KEY

    vocab_map = {
        0: '"prompt"',
        1: '"name"',
        2: '"parameters"',
        3: '"banana"',
    }

    logits = [1.0] * len(vocab_map)

    result = mask_invalid_tokens(
        logits,
        json_machine,
        vocab_map,
        schema_state,
        function_names=[],
        definitions=[],
    )

    assert result[0] != float("-inf")

    for token_id in (1, 2, 3):
        assert result[token_id] == float("-inf")

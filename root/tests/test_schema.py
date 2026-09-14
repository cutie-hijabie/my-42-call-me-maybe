import json
from pathlib import Path

from src.schema import InputDefinition, InputFileDefinition, InputDefinition


def test_provided_functions_definition_matches_schema():
    path = Path("data/input/functions_definition.json")

    with path.open("r") as file:
        data = json.load(file)

    validated = InputFileDefinition.model_validate(data)

    assert len(validated.root) == 5
    assert all(function.name for function in validated.root)
    assert all(function.description for function in validated.root)


from src.schema_state import SchemaState


def test_get_possible_function_names_filters_by_prefix():
    state = SchemaState()
    state.partial_function_name = "fn_g"

    function_names = [
        "fn_add_numbers",
        "fn_greet",
        "fn_get_weather",
        "fn_reverse_string",
    ]

    result = state.get_possible_function_names(function_names)

    assert result == ["fn_greet", "fn_get_weather"]


from src.schema_state import SchemaState


def test_set_function_if_complete_only_sets_exact_match():
    state = SchemaState()

    function_names = [
        "fn_greet",
        "fn_get_weather",
    ]

    state.partial_function_name = "fn_g"
    state.set_function_if_complete(function_names)

    assert state.function_name is None

    state.partial_function_name = "fn_greet"
    state.set_function_if_complete(function_names)

    assert state.function_name == "fn_greet"


def test_get_parameters_returns_chosen_function_parameters():
    state = SchemaState()
    state.function_name = "fn_test"

    definitions = [
        InputDefinition(
            name="fn_other",
            description="Another function",
            parameters={
                "value": {"type": "string"},
            },
            returns={"type": "string"},
        ),
        InputDefinition(
            name="fn_test",
            description="Test function",
            parameters={
                "first": {"type": "number"},
                "second": {"type": "string"},
            },
            returns={"type": "string"},
        ),
    ]

    result = state.get_parameters(definitions)

    assert list(result.keys()) == ["first", "second"]
    assert result["first"].type == "number"
    assert result["second"].type == "string"


def test_get_possible_parameter_names_filters_by_prefix():
    state = SchemaState()
    state.partial_parameter_name = "re"

    parameters = {
        "regex": ParameterDefinition(type="string"),
        "replacement": ParameterDefinition(type="string"),
        "source_string": ParameterDefinition(type="string"),
    }

    result = state.get_possible_parameter_names(parameters)

    assert result == ["regex", "replacement"]


from src.schema import ParameterDefinition
from src.schema_state import SchemaState


def test_parameter_prefix_and_exact_match():
    state = SchemaState()

    parameters = {
        "regex": ParameterDefinition(type="string"),
        "replacement": ParameterDefinition(type="string"),
        "source_string": ParameterDefinition(type="string"),
    }

    state.partial_parameter_name = "re"

    result = state.get_possible_parameter_names(parameters)

    assert result == ["regex", "replacement"]
    assert state.current_parameter is None

    state.set_parameter_if_complete(parameters)

    assert state.current_parameter is None

    state.partial_parameter_name = "regex"
    state.set_parameter_if_complete(parameters)

    assert state.current_parameter == "regex"


def test_get_current_parameter_type():
    state = SchemaState()

    parameters = {
        "name": ParameterDefinition(type="string"),
        "age": ParameterDefinition(type="number"),
    }

    state.current_parameter = "name"
    assert state.get_current_parameter_type(parameters) == "string"

    state.current_parameter = "age"
    assert state.get_current_parameter_type(parameters) == "number"

    state.current_parameter = None
    assert state.get_current_parameter_type(parameters) == ""


from unittest.mock import Mock

from src.json_state import JasonStates
from src.masking import mask_invalid_tokens
from src.schema_state import SchemaState, SchemaStage


def test_schema_name_masking_keeps_only_valid_function_prefixes():
    json_machine = Mock()
    json_machine.state = JasonStates.IN_STRING
    json_machine.validate_candidate.return_value = True

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.NAME
    schema_state.partial_function_name = "fn_g"

    function_names = [
        "fn_greet",
        "fn_get_weather",
    ]

    vocab_map = {
        0: "r",
        1: "e",
        2: "x",
    }

    logits = [1.0, 1.0, 1.0]

    result = mask_invalid_tokens(
        logits,
        json_machine,
        vocab_map,
        schema_state,
        function_names,
    )

    assert result[0] != float("-inf")
    assert result[1] != float("-inf")
    assert result[2] == float("-inf")


def test_schema_name_masking_requires_exact_match_on_closing_quote():
    json_machine = Mock()
    json_machine.state = JasonStates.IN_STRING
    json_machine.validate_candidate.return_value = True

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.NAME
    schema_state.partial_function_name = "fn_gre"

    function_names = [
        "fn_greet",
        "fn_get_weather",
    ]

    vocab_map = {
        0: 'et"',
        1: '"',
    }

    logits = [1.0, 1.0]

    result = mask_invalid_tokens(
        logits,
        json_machine,
        vocab_map,
        schema_state,
        function_names,
    )

    assert result[0] != float("-inf")
    assert result[1] == float("-inf")

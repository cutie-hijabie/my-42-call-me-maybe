import json
from pathlib import Path
from src.json_state import JasonStateMacheine, JasonStates
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
    json_machine.string_role = "value"

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
    definitions = []
    result = mask_invalid_tokens(
        logits,
        json_machine,
        vocab_map,
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] != float("-inf")
    assert result[1] != float("-inf")
    assert result[2] == float("-inf")


def test_schema_name_masking_requires_exact_match_on_closing_quote():
    json_machine = Mock()
    json_machine.state = JasonStates.IN_STRING
    json_machine.validate_candidate.return_value = True
    json_machine.string_role = "value"

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
    definitions = []
    result = mask_invalid_tokens(
        logits,
        json_machine,
        vocab_map,
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] != float("-inf")
    assert result[1] == float("-inf")


def test_schema_parameter_masking_keeps_only_valid_parameter_names():
    json_machine = Mock()
    json_machine.state = JasonStates.IN_STRING
    json_machine.validate_candidate.return_value = True
    json_machine.string_role = "key"

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.PARAMETERS
    schema_state.function_name = "fn_test"
    schema_state.partial_parameter_name = "re"

    definitions = [
        InputDefinition(
            name="fn_test",
            description="Test function",
            parameters={
                "regex": {"type": "string"},
                "replacement": {"type": "string"},
                "source_string": {"type": "string"},
            },
            returns={"type": "string"},
        )
    ]

    function_names = ["fn_test"]

    vocab_map = {
        0: "g",
        1: "p",
        2: "x",
        3: 'gex"',
        4: '"',
    }

    logits = [1.0] * len(vocab_map)

    result = mask_invalid_tokens(
        logits,
        json_machine,
        vocab_map,
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] != float("-inf")
    assert result[1] != float("-inf")
    assert result[2] == float("-inf")
    assert result[3] != float("-inf")
    assert result[4] == float("-inf")


def test_schema_value_masking_allows_string_start():
    schema_state = SchemaState()
    schema_state.stage = SchemaStage.PARAMETERS
    schema_state.function_name = "fn_greet"
    schema_state.current_parameter = "name"

    definitions = [
        InputDefinition(
            name="fn_greet",
            description="Greet someone.",
            parameters={"name": {"type": "string"}},
            returns={"type": "string"},
        )
    ]

    json_machine = Mock()
    json_machine.state = JasonStates.EXPECT_VALUE
    json_machine.validate_candidate.return_value = True

    vocab_map = {
        0: '"',
        1: '   "',
        2: "7",
        3: "-",
    }
    logits = [1.0] * len(vocab_map)

    result = mask_invalid_tokens(
        logits,
        json_machine,
        vocab_map,
        schema_state,
        ["fn_greet"],
        definitions,
    )

    assert result[0] != float("-inf")
    assert result[1] != float("-inf")
    assert result[2] == float("-inf")
    assert result[3] == float("-inf")


def test_schema_value_masking_allows_number_start():
    schema_state = SchemaState()
    schema_state.stage = SchemaStage.PARAMETERS
    schema_state.function_name = "fn_add_numbers"
    schema_state.current_parameter = "a"

    definitions = [
        InputDefinition(
            name="fn_add_numbers",
            description="Add numbers.",
            parameters={
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            returns={"type": "number"},
        )
    ]

    json_machine = Mock()
    json_machine.state = JasonStates.EXPECT_VALUE
    json_machine.validate_candidate.return_value = True

    vocab_map = {
        0: "7",
        1: "   42",
        2: "-",
        3: '"',
    }
    logits = [1.0] * len(vocab_map)

    result = mask_invalid_tokens(
        logits,
        json_machine,
        vocab_map,
        schema_state,
        ["fn_add_numbers"],
        definitions,
    )

    assert result[0] != float("-inf")
    assert result[1] != float("-inf")
    assert result[2] != float("-inf")
    assert result[3] == float("-inf")


def test_schema_update_moves_from_prompt_to_name():
    schema_state = SchemaState()
    json_machine = JasonStateMacheine()

    json_machine.state = JasonStates.EXPECT_KEY

    schema_state.update(",", json_machine, ["fn_greet", "fn_add_numbers"], {})

    assert schema_state.stage == SchemaStage.NAME


def test_schema_update_builds_partial_function_name():
    schema_state = SchemaState()
    schema_state.stage = SchemaStage.NAME

    json_machine = JasonStateMacheine()
    json_machine.state = JasonStates.IN_STRING
    json_machine.string_role = "value"

    schema_state.update(
        '"fn_', json_machine, ["fn_greet", "fn_add_numbers"], {}
    )

    assert schema_state.partial_function_name == "fn_"
    assert schema_state.function_name is None

    schema_state.update(
        "greet", json_machine, ["fn_greet", "fn_add_numbers"], {}
    )

    assert schema_state.partial_function_name == "fn_greet"
    assert schema_state.function_name == "fn_greet"


def test_schema_update_does_not_collect_name_key():
    schema_state = SchemaState()
    schema_state.stage = SchemaStage.NAME

    json_machine = JasonStateMacheine()
    json_machine.state = JasonStates.IN_STRING
    json_machine.string_role = "key"

    schema_state.update("name", json_machine, ["fn_greet"], {})

    assert schema_state.partial_function_name == ""
    assert schema_state.function_name is None


def test_schema_update_moves_from_name_to_parameters():
    schema_state = SchemaState()
    schema_state.stage = SchemaStage.NAME
    schema_state.partial_function_name = "fn_greet"
    schema_state.function_name = "fn_greet"

    json_machine = JasonStateMacheine()
    json_machine.state = JasonStates.EXPECT_KEY

    schema_state.update(",", json_machine, ["fn_greet"], {})

    assert schema_state.stage == SchemaStage.PARAMETERS


def test_schema_update_tracks_parameter_name():
    schema_state = SchemaState()
    schema_state.stage = SchemaStage.PARAMETERS

    parameters = {
        "source_string": ParameterDefinition(type="string"),
        "regex": ParameterDefinition(type="string"),
    }

    json_machine = JasonStateMacheine()
    json_machine.brackets = ["{", "{"]
    json_machine.state = JasonStates.IN_STRING
    json_machine.string_role = "key"

    schema_state.update(
        '"sou',
        json_machine,
        ["fn_substitute_string_with_regex"],
        parameters,
    )

    assert schema_state.partial_parameter_name == "sou"
    assert schema_state.current_parameter is None

    schema_state.update(
        "rce_string",
        json_machine,
        ["fn_substitute_string_with_regex"],
        parameters,
    )

    assert schema_state.partial_parameter_name == "source_string"
    assert schema_state.current_parameter == "source_string"


def test_schema_update_does_not_track_outer_key_as_parameter():
    schema_state = SchemaState()
    schema_state.stage = SchemaStage.PARAMETERS

    parameters = {
        "name": ParameterDefinition(type="string"),
    }

    json_machine = JasonStateMacheine()
    json_machine.brackets = ["{"]
    json_machine.state = JasonStates.IN_STRING
    json_machine.string_role = "key"

    schema_state.update(
        "parameters",
        json_machine,
        ["fn_greet"],
        parameters,
    )

    assert schema_state.partial_parameter_name == ""
    assert schema_state.current_parameter is None


def test_schema_update_resets_parameter_before_next_key():
    schema_state = SchemaState()
    schema_state.stage = SchemaStage.PARAMETERS
    schema_state.partial_parameter_name = "a"
    schema_state.current_parameter = "a"

    parameters = {
        "a": ParameterDefinition(type="number"),
        "b": ParameterDefinition(type="number"),
    }

    json_machine = JasonStateMacheine()
    json_machine.brackets = ["{", "{"]
    json_machine.state = JasonStates.EXPECT_KEY

    schema_state.update(
        ",",
        json_machine,
        ["fn_add_numbers"],
        parameters,
    )

    assert schema_state.partial_parameter_name == ""
    assert schema_state.current_parameter is None


def test_schema_update_can_track_next_parameter_after_reset():
    schema_state = SchemaState()
    schema_state.stage = SchemaStage.PARAMETERS

    parameters = {
        "a": ParameterDefinition(type="number"),
        "b": ParameterDefinition(type="number"),
    }

    json_machine = JasonStateMacheine()
    json_machine.brackets = ["{", "{"]

    # First parameter
    json_machine.state = JasonStates.IN_STRING
    json_machine.string_role = "key"

    schema_state.update(
        '"a',
        json_machine,
        ["fn_add_numbers"],
        parameters,
    )

    assert schema_state.current_parameter == "a"

    # The first value has finished and the comma has moved us
    # back to expecting another key.
    json_machine.state = JasonStates.EXPECT_KEY

    schema_state.update(
        ",",
        json_machine,
        ["fn_add_numbers"],
        parameters,
    )

    assert schema_state.partial_parameter_name == ""
    assert schema_state.current_parameter is None

    # Second parameter
    json_machine.state = JasonStates.IN_STRING
    json_machine.string_role = "key"

    schema_state.update(
        '"b',
        json_machine,
        ["fn_add_numbers"],
        parameters,
    )

    assert schema_state.partial_parameter_name == "b"
    assert schema_state.current_parameter == "b"

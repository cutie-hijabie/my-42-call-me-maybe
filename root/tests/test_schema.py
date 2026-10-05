"""Schema regressions using serialized name prefixes and real character updates."""

import json
from pathlib import Path

from src.json_state import JasonStateMacheine, JasonStates
from src.masking import mask_invalid_tokens
from src.schema import InputDefinition, InputFileDefinition, ParameterDefinition
from src.schema_state import SchemaState, SchemaStage


NAMES = ["fn_greet", "fn_get_weather", "fn_other"]
DEFINITIONS = [
    InputDefinition(
        name=name,
        description="Runtime definition.",
        parameters={"name": {"type": "string"}, "age": {"type": "number"}},
        returns={"type": "string"},
    )
    for name in NAMES
]


def context(prefix, request="x"):
    machine, schema = JasonStateMacheine(), SchemaState(request)
    for char in prefix:
        assert machine.transition(char)
        schema.update(char, machine, NAMES, schema.get_parameters(DEFINITIONS))
    return machine, schema


def candidates(prefix, vocab):
    machine, schema = context(prefix)
    return mask_invalid_tokens(
        [1.0] * len(vocab), machine, dict(enumerate(vocab)),
        schema, NAMES, DEFINITIONS,
    )


def test_provided_functions_definition_matches_schema():
    path = Path("data/input/functions_definition.json")
    data = json.loads(path.read_text())
    validated = InputFileDefinition.model_validate(data)
    assert len(validated.root) == len(data)
    assert validated.root
    assert all(function.name and function.description for function in validated.root)


def test_get_possible_function_names_filters_by_serialized_prefix():
    state = SchemaState("")
    state.partial_function_name = '"fn_g'
    assert state.get_possible_function_names(NAMES) == NAMES[:2]


def test_set_function_if_complete_waits_for_closing_quote():
    state = SchemaState("")
    for prefix in ['"fn_g', '"fn_greet']:
        state.partial_function_name = prefix
        state.set_function_if_complete(NAMES)
        assert state.function_name is None
    state.partial_function_name = json.dumps("fn_greet", ensure_ascii=False)
    state.set_function_if_complete(NAMES)
    assert state.function_name == "fn_greet"


def test_get_parameters_returns_selected_runtime_definition():
    state = SchemaState("")
    assert state.get_parameters(DEFINITIONS) == {}
    state.function_name = "fn_greet"
    parameters = state.get_parameters(DEFINITIONS)
    assert list(parameters) == ["name", "age"]
    assert parameters["name"].type == "string"
    assert parameters["age"].type == "number"


def test_get_possible_parameter_names_filters_by_prefix():
    state = SchemaState("")
    state.partial_parameter_name = "re"
    parameters = {
        key: ParameterDefinition(type="string")
        for key in ["regex", "replacement", "source_string"]
    }
    assert state.get_possible_parameter_names(parameters) == ["regex", "replacement"]


def test_parameter_prefix_does_not_bind_an_incomplete_key():
    state = SchemaState("")
    parameters = {"name": ParameterDefinition(type="string")}
    state.partial_parameter_name = "na"
    state.set_parameter_if_complete(parameters)
    assert state.current_parameter is None
    state.partial_parameter_name = "name"
    state.set_parameter_if_complete(parameters)
    assert state.current_parameter == "name"


def test_get_current_parameter_type():
    state = SchemaState("")
    parameters = DEFINITIONS[0].parameters
    assert state.get_current_parameter_type(parameters) == ""
    state.current_parameter = "name"
    assert state.get_current_parameter_type(parameters) == "string"
    state.current_parameter = "age"
    assert state.get_current_parameter_type(parameters) == "number"


def test_name_masking_keeps_only_runtime_prefixes():
    assert candidates('{"prompt":"x","name":"fn_g', ["r", "e", "x"]) == [
        1.0, 1.0, float("-inf"),
    ]


def test_name_masking_requires_exact_match_on_closing_quote():
    assert candidates('{"prompt":"x","name":"fn_gre', ['et"', '"']) == [
        1.0, float("-inf"),
    ]


def test_parameter_masking_keeps_runtime_key_prefixes():
    prefix = '{"prompt":"x","name":"fn_greet","parameters":{"na'
    assert candidates(prefix, ["m", "x", 'me"', '"']) == [
        1.0, float("-inf"), 1.0, float("-inf"),
    ]


def test_parameter_string_start_matches_runtime_type():
    prefix = '{"prompt":"x","name":"fn_greet","parameters":{"name":'
    assert candidates(prefix, ['"', '   "', "7", "-"]) == [
        1.0, 1.0, float("-inf"), float("-inf"),
    ]


def test_parameter_number_start_matches_runtime_type():
    prefix = '{"prompt":"x","name":"fn_greet","parameters":{"age":'
    assert candidates(prefix, ["7", "   42", "-", '"']) == [
        1.0, 1.0, 1.0, float("-inf"),
    ]


def test_prompt_comma_moves_to_name():
    machine, schema = context('{"prompt":"x",')
    assert machine.state == JasonStates.EXPECT_KEY
    assert schema.stage == SchemaStage.NAME


def test_name_is_collected_until_its_closing_quote():
    machine, schema = context('{"prompt":"x","name":"fn_greet')
    assert schema.partial_function_name == '"fn_greet'
    assert schema.function_name is None
    assert machine.transition('"')
    schema.update('"', machine, NAMES, schema.get_parameters(DEFINITIONS))
    assert schema.partial_function_name == '"fn_greet"'
    assert schema.function_name == "fn_greet"


def test_outer_name_key_is_not_collected_as_function_value():
    _, schema = context('{"prompt":"x","name"')
    assert schema.partial_function_name == ""
    assert schema.function_name is None


def test_name_comma_moves_to_parameters():
    _, schema = context('{"prompt":"x","name":"fn_greet",')
    assert schema.stage == SchemaStage.PARAMETERS
    assert schema.function_name == "fn_greet"


def test_parameter_name_is_tracked_in_nested_object():
    _, schema = context('{"prompt":"x","name":"fn_greet","parameters":{"na')
    assert schema.partial_parameter_name == "na"
    assert schema.current_parameter is None


def test_outer_parameters_key_is_not_collected_as_parameter():
    _, schema = context('{"prompt":"x","name":"fn_greet","parameters"')
    assert schema.partial_parameter_name == ""
    assert schema.current_parameter is None


def test_parameter_comma_records_value_and_resets_key():
    _, schema = context(
        '{"prompt":"x","name":"fn_greet","parameters":{"name":"Ada",'
    )
    assert schema.completed_parameters == {"name"}
    assert schema.partial_parameter_name == ""
    assert schema.current_parameter is None


def test_next_parameter_is_tracked_after_reset():
    _, schema = context(
        '{"prompt":"x","name":"fn_greet","parameters":{"name":"Ada","age":'
    )
    assert schema.completed_parameters == {"name"}
    assert schema.current_parameter == "age"
    assert schema.get_current_parameter_type(schema.get_parameters(DEFINITIONS)) == "number"

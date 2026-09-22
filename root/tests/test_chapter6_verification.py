import json
from pathlib import Path

import pytest

from src.json_state import JasonStateMacheine, JasonStates
from src.masking import mask_invalid_tokens
from src.schema import InputFileDefinition
from src.schema_state import SchemaState, SchemaStage


@pytest.fixture
def runtime_schema():
    path = Path("data/input/functions_definition.json")

    with path.open("r") as file:
        raw_definitions = json.load(file)

    definitions = InputFileDefinition.model_validate(raw_definitions).root
    function_names = [definition.name for definition in definitions]

    return definitions, function_names


@pytest.mark.parametrize(
    "function_name",
    [
        "fn_add_numbers",
        "fn_greet",
        "fn_reverse_string",
        "fn_get_square_root",
        "fn_substitute_string_with_regex",
    ],
)
def test_real_runtime_function_names_are_allowed(
    runtime_schema,
    function_name,
):
    definitions, function_names = runtime_schema

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.NAME

    machine = JasonStateMacheine()
    machine.brackets = ["{"]
    machine.state = JasonStates.EXPECT_VALUE

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: f'"{function_name}"'},
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] == 1.0


@pytest.mark.parametrize(
    "invalid_name",
    [
        "fn_totally_fake",
        "fn_add",
        "greet",
        "banana",
    ],
)
def test_names_not_in_real_runtime_schema_are_blocked(
    runtime_schema,
    invalid_name,
):
    definitions, function_names = runtime_schema

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.NAME

    machine = JasonStateMacheine()
    machine.brackets = ["{"]
    machine.state = JasonStates.EXPECT_VALUE

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: f'"{invalid_name}"'},
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] == float("-inf")


@pytest.mark.parametrize(
    ("function_name", "expected_parameters"),
    [
        ("fn_add_numbers", {"a", "b"}),
        ("fn_greet", {"name"}),
        ("fn_reverse_string", {"s"}),
        ("fn_get_square_root", {"a"}),
        (
            "fn_substitute_string_with_regex",
            {"source_string", "regex", "replacement"},
        ),
    ],
)
def test_real_function_allows_exactly_its_declared_parameter_names(
    runtime_schema,
    function_name,
    expected_parameters,
):
    definitions, function_names = runtime_schema

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.PARAMETERS
    schema_state.function_name = function_name

    machine = JasonStateMacheine()
    machine.brackets = ["{", "{"]
    machine.state = JasonStates.EXPECT_KEY
    machine.allow_empty_object = True

    for parameter_name in expected_parameters:
        result = mask_invalid_tokens(
            [1.0],
            machine,
            {0: f'"{parameter_name}"'},
            schema_state,
            function_names,
            definitions,
        )

        assert result[0] == 1.0, (
            f"{parameter_name!r} should be allowed for {function_name}"
        )

    all_runtime_parameters = {
        parameter_name
        for definition in definitions
        for parameter_name in definition.parameters
    }

    undeclared_parameters = all_runtime_parameters - expected_parameters

    for parameter_name in undeclared_parameters:
        result = mask_invalid_tokens(
            [1.0],
            machine,
            {0: f'"{parameter_name}"'},
            schema_state,
            function_names,
            definitions,
        )

        assert result[0] == float("-inf"), (
            f"{parameter_name!r} belongs to the runtime schema, "
            f"but not to {function_name}"
        )


@pytest.mark.parametrize(
    ("function_name", "parameter_name", "valid_value", "invalid_value"),
    [
        ("fn_add_numbers", "a", "42", '"42"'),
        ("fn_add_numbers", "b", "-7", '"seven"'),
        ("fn_greet", "name", '"Hayat"', "42"),
        ("fn_reverse_string", "s", '"hello"', "42"),
        ("fn_get_square_root", "a", "16", '"16"'),
        (
            "fn_substitute_string_with_regex",
            "source_string",
            '"hello 123"',
            "123",
        ),
        (
            "fn_substitute_string_with_regex",
            "regex",
            '"[0-9]+"',
            "123",
        ),
        (
            "fn_substitute_string_with_regex",
            "replacement",
            '"NUMBER"',
            "123",
        ),
    ],
)
def test_real_runtime_parameter_types_are_enforced(
    runtime_schema,
    function_name,
    parameter_name,
    valid_value,
    invalid_value,
):
    definitions, function_names = runtime_schema

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.PARAMETERS
    schema_state.function_name = function_name
    schema_state.current_parameter = parameter_name

    machine = JasonStateMacheine()
    machine.brackets = ["{", "{"]
    machine.state = JasonStates.EXPECT_VALUE

    valid_result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: valid_value},
        schema_state,
        function_names,
        definitions,
    )

    invalid_result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: invalid_value},
        schema_state,
        function_names,
        definitions,
    )

    assert valid_result[0] == 1.0, (
        f"{valid_value!r} should be valid for {function_name}.{parameter_name}"
    )

    assert invalid_result[0] == float("-inf"), (
        f"{invalid_value!r} should be rejected for "
        f"{function_name}.{parameter_name}"
    )


@pytest.mark.parametrize(
    ("candidate", "should_be_allowed"),
    [
        ('"fn_add', True),
        ('"fn_add"', False),
        ('"fn_add_numbers"', True),
    ],
)
def test_real_function_prefix_distinguishes_open_from_closed_string(
    runtime_schema,
    candidate,
    should_be_allowed,
):
    definitions, function_names = runtime_schema

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.NAME

    machine = JasonStateMacheine()
    machine.brackets = ["{"]
    machine.state = JasonStates.EXPECT_VALUE

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: candidate},
        schema_state,
        function_names,
        definitions,
    )

    if should_be_allowed:
        assert result[0] == 1.0
    else:
        assert result[0] == float("-inf")


def test_parameters_object_cannot_close_before_required_parameters(
    runtime_schema,
):
    definitions, function_names = runtime_schema

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.PARAMETERS
    schema_state.function_name = "fn_add_numbers"

    machine = JasonStateMacheine()
    machine.brackets = ["{", "{"]
    machine.state = JasonStates.EXPECT_KEY
    machine.allow_empty_object = True

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: "}"},
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] == float("-inf")


def test_completed_parameter_cannot_be_generated_again(
    runtime_schema,
):
    definitions, function_names = runtime_schema

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.PARAMETERS
    schema_state.function_name = "fn_add_numbers"

    machine = JasonStateMacheine()
    machine.brackets = ["{", "{"]
    machine.state = JasonStates.EXPECT_KEY

    parameters = schema_state.get_parameters(definitions)

    # Actually consume the first parameter and its value.
    for char in '"a":1,':
        machine.transition(char)
        schema_state.update(
            char,
            machine,
            function_names,
            parameters,
        )

    # We should now genuinely be waiting for the next parameter.
    assert machine.state == JasonStates.EXPECT_KEY
    assert schema_state.current_parameter is None
    assert schema_state.partial_parameter_name == ""

    # Ask the masker whether "a" may be generated again.
    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: '"a"'},
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] == float("-inf")

"""Regression tests for schema masking outside parameter-key generation."""

import pytest

from src.json_state import JasonStateMacheine, JasonStates
from src.masking import mask_invalid_tokens
from src.schema import InputDefinition, ParameterDefinition, ReturnsDefinition
from src.schema_state import SchemaStage, SchemaState


@pytest.fixture
def parameter_context():
    definition = InputDefinition(
        name="fn_greet",
        description="Greet someone",
        parameters={"name": ParameterDefinition(type="string")},
        returns=ReturnsDefinition(type="string"),
    )
    schema_state = SchemaState()
    schema_state.stage = SchemaStage.PARAMETERS
    schema_state.function_name = definition.name
    schema_state.current_parameter = "name"
    schema_state.partial_parameter_name = "name"
    return schema_state, [definition.name], [definition]


@pytest.mark.parametrize(
    ("json_state", "string_role", "candidate"),
    [
        (JasonStates.EXPECT_COLON, None, ":"),
        (JasonStates.IN_STRING, "value", "Hayat"),
        (JasonStates.IN_STRING, "value", '"'),
        (JasonStates.AFTER_VALUE, None, "}"),
    ],
)
def test_parameter_mask_does_not_reject_json_syntax_or_value_continuations(
    parameter_context, json_state, string_role, candidate
):
    schema_state, function_names, definitions = parameter_context
    machine = JasonStateMacheine()
    machine.brackets = ["{", "{"]
    machine.state = json_state
    machine.string_role = string_role

    assert machine.validate_candidate(candidate), (
        f"Test setup error: {candidate!r} is not valid in {json_state}"
    )
    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: candidate},
        schema_state,
        function_names,
        definitions,
    )
    assert result[0] == 1.0, (
        f"Schema mask rejected {candidate!r} in JSON state {json_state}; "
        "parameter-name constraints should apply only while generating a key"
    )


@pytest.mark.parametrize(
    ("candidate", "should_be_allowed"),
    [
        ('"', True),
        ('"n', True),
        ('"na', True),
        ('"name', True),
        ('"name"', True),
        ('"nickname', False),
        ('"age"', False),
    ],
)
def test_parameter_key_start_respects_runtime_schema(
    parameter_context, candidate, should_be_allowed
):
    schema_state, function_names, definitions = parameter_context
    schema_state.partial_parameter_name = ""

    machine = JasonStateMacheine()
    machine.brackets = ["{", "{"]
    machine.state = JasonStates.EXPECT_KEY
    machine.allow_empty_object = True

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: candidate},
        schema_state,
        function_names,
        definitions,
    )

    if should_be_allowed:
        assert result[0] == 1.0, (
            f"Valid parameter-key start was rejected: {candidate!r}"
        )
    else:
        assert result[0] == float("-inf"), (
            f"Undeclared parameter-key start was allowed: {candidate!r}"
        )


def test_second_parameter_key_is_constrained_after_first_parameter():
    definition = InputDefinition(
        name="fn_example",
        description="Example with two parameters",
        parameters={
            "first": ParameterDefinition(type="string"),
            "second": ParameterDefinition(type="number"),
        },
        returns=ReturnsDefinition(type="string"),
    )

    definitions = [definition]
    function_names = [definition.name]

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.PARAMETERS
    schema_state.function_name = definition.name

    machine = JasonStateMacheine()
    machine.brackets = ["{"]

    # Simulate the parameters object up through the first completed value:
    # {"first": "hello"
    prefix = '{"first":"hello"'

    for char in prefix:
        machine.transition(char)

        parameters = schema_state.get_parameters(definitions)
        schema_state.update(
            char,
            machine,
            function_names,
            parameters,
        )
        print(
            repr(char),
            machine.state,
            machine.string_role,
            repr(schema_state.partial_parameter_name),
            repr(schema_state.current_parameter),
        )

    assert schema_state.current_parameter == "first"

    # Now consume the comma that should reset parameter tracking.
    machine.transition(",")

    parameters = schema_state.get_parameters(definitions)
    schema_state.update(
        ",",
        machine,
        function_names,
        parameters,
    )

    assert schema_state.current_parameter is None
    assert schema_state.partial_parameter_name == ""

    # We should now be waiting for another key.
    assert machine.state == JasonStates.EXPECT_KEY

    # "second" is declared in the runtime schema, so it should survive.
    valid_result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: '"second"'},
        schema_state,
        function_names,
        definitions,
    )

    assert valid_result[0] == 1.0

    # An undeclared second parameter should be masked.
    invalid_result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: '"banana"'},
        schema_state,
        function_names,
        definitions,
    )

    assert invalid_result[0] == float("-inf")


@pytest.mark.parametrize(
    ("parameter_type", "candidate", "should_be_allowed"),
    [
        # string parameter
        ("string", '"hello', True),
        ("string", '"hello"', True),
        ("string", "hello", False),
        ("string", "42", False),
        # number parameter
        ("number", "42", True),
        ("number", "-42", True),
        ("number", '"42"', False),
        ("number", '"hello"', False),
        # JSON whitespace before the value
        ("string", '   "hello"', True),
        ("number", "   42", True),
        ("number", "42,", True),
        ("number", "42}", True),
        ("number", "-42,", True),
        ("number", "3.14", True),
        ("number", "-3.14", True),
    ],
)
def test_parameter_value_start_respects_runtime_type(
    parameter_type, candidate, should_be_allowed
):
    definition = InputDefinition(
        name="fn_example",
        description="Runtime-defined example",
        parameters={
            "value": ParameterDefinition(type=parameter_type),
        },
        returns=ReturnsDefinition(type="string"),
    )

    definitions = [definition]
    function_names = [definition.name]

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.PARAMETERS
    schema_state.function_name = definition.name
    schema_state.current_parameter = "value"

    machine = JasonStateMacheine()
    machine.brackets = ["{", "{"]
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
        assert result[0] == 1.0, (
            f"{parameter_type} value incorrectly rejected: {candidate!r}"
        )
    else:
        assert result[0] == float("-inf"), (
            f"{parameter_type} value incorrectly allowed: {candidate!r}"
        )


@pytest.mark.parametrize(
    ("candidate", "should_be_allowed"),
    [
        # Valid starts/prefixes
        ('"', True),
        ('"f', True),
        ('"fn_', True),
        ('"fn_alpha', True),
        ('"fn_alpha"', True),
        # Another runtime-defined function
        ('"fn_beta', True),
        ('"fn_beta"', True),
        # Invalid function names
        ('"fn_banana', False),
        ('"totally_fake"', False),
    ],
)
def test_function_name_start_respects_runtime_schema(
    candidate, should_be_allowed
):
    definitions = [
        InputDefinition(
            name="fn_alpha",
            description="First runtime function",
            parameters={},
            returns=ReturnsDefinition(type="string"),
        ),
        InputDefinition(
            name="fn_beta",
            description="Second runtime function",
            parameters={},
            returns=ReturnsDefinition(type="string"),
        ),
    ]

    function_names = [definition.name for definition in definitions]

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
        assert result[0] == 1.0, (
            f"Valid function-name start was rejected: {candidate!r}"
        )
    else:
        assert result[0] == float("-inf"), (
            f"Invalid function-name start was allowed: {candidate!r}"
        )


def make_function_name_test_context():
    definitions = [
        InputDefinition(
            name="fn_alpha",
            description="First runtime function",
            parameters={},
            returns=ReturnsDefinition(type="string"),
        ),
        InputDefinition(
            name="fn_alpine",
            description="Second runtime function",
            parameters={},
            returns=ReturnsDefinition(type="string"),
        ),
        InputDefinition(
            name="fn_beta",
            description="Third runtime function",
            parameters={},
            returns=ReturnsDefinition(type="string"),
        ),
    ]

    function_names = [definition.name for definition in definitions]

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.NAME

    machine = JasonStateMacheine()
    machine.brackets = ["{"]
    machine.state = JasonStates.EXPECT_VALUE

    return machine, schema_state, function_names, definitions


def consume_schema_token(
    token_string,
    machine,
    schema_state,
    function_names,
    definitions,
):
    parameters = schema_state.get_parameters(definitions)

    for char in token_string:
        machine.transition(char)
        schema_state.update(
            char,
            machine,
            function_names,
            parameters,
        )


def test_function_name_can_be_generated_across_multiple_tokens():
    machine, schema_state, function_names, definitions = (
        make_function_name_test_context()
    )

    first_token = '"fn_'

    first_result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: first_token},
        schema_state,
        function_names,
        definitions,
    )

    assert first_result[0] == 1.0

    consume_schema_token(
        first_token,
        machine,
        schema_state,
        function_names,
        definitions,
    )

    assert schema_state.partial_function_name == "fn_"
    assert schema_state.function_name is None

    second_token = "alpha"

    second_result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: second_token},
        schema_state,
        function_names,
        definitions,
    )

    assert second_result[0] == 1.0

    consume_schema_token(
        second_token,
        machine,
        schema_state,
        function_names,
        definitions,
    )

    assert schema_state.partial_function_name == "fn_alpha"
    assert schema_state.function_name == "fn_alpha"


def test_shared_function_prefix_keeps_multiple_runtime_options_alive():
    machine, schema_state, function_names, definitions = (
        make_function_name_test_context()
    )

    token_string = '"fn_al'

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: token_string},
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] == 1.0

    consume_schema_token(
        token_string,
        machine,
        schema_state,
        function_names,
        definitions,
    )

    assert schema_state.partial_function_name == "fn_al"

    possible = schema_state.get_possible_function_names(function_names)

    assert set(possible) == {"fn_alpha", "fn_alpine"}


def test_function_name_continuation_rejects_diverging_token():
    machine, schema_state, function_names, definitions = (
        make_function_name_test_context()
    )

    first_token = '"fn_al'

    consume_schema_token(
        first_token,
        machine,
        schema_state,
        function_names,
        definitions,
    )

    assert schema_state.partial_function_name == "fn_al"

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: "banana"},
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] == float("-inf")


def test_function_name_continuation_accepts_valid_branch():
    machine, schema_state, function_names, definitions = (
        make_function_name_test_context()
    )

    consume_schema_token(
        '"fn_al',
        machine,
        schema_state,
        function_names,
        definitions,
    )

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: "pine"},
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] == 1.0


def test_closing_quote_after_split_function_name_is_allowed():
    machine, schema_state, function_names, definitions = (
        make_function_name_test_context()
    )

    consume_schema_token(
        '"fn_',
        machine,
        schema_state,
        function_names,
        definitions,
    )

    consume_schema_token(
        "beta",
        machine,
        schema_state,
        function_names,
        definitions,
    )

    assert schema_state.function_name == "fn_beta"

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: '"'},
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] == 1.0


def test_incomplete_function_name_cannot_close_early():
    machine, schema_state, function_names, definitions = (
        make_function_name_test_context()
    )

    consume_schema_token(
        '"fn_',
        machine,
        schema_state,
        function_names,
        definitions,
    )

    assert schema_state.function_name is None

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: '"'},
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] == float("-inf")


def test_function_name_mask_does_not_reject_comma_after_name_value():
    machine, schema_state, function_names, definitions = (
        make_function_name_test_context()
    )

    consume_schema_token(
        '"fn_beta"',
        machine,
        schema_state,
        function_names,
        definitions,
    )

    assert schema_state.function_name == "fn_beta"
    assert machine.state == JasonStates.AFTER_VALUE

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: ","},
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] == 1.0


def test_incomplete_parameter_name_cannot_close_early():
    definitions = [
        InputDefinition(
            name="fn_example",
            description="test",
            parameters={
                "name": ParameterDefinition(type="string"),
            },
            returns=ReturnsDefinition(type="string"),
        )
    ]
    function_names = ["fn_example"]

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.PARAMETERS
    schema_state.function_name = "fn_example"

    machine = JasonStateMacheine()
    machine.brackets = ["{", "{"]

    parameters = schema_state.get_parameters(definitions)

    machine.state = JasonStates.IN_STRING
    machine.string_role = "key"

    for char in "na":
        machine.transition(char)
        schema_state.update(
            char,
            machine,
            function_names,
            parameters,
        )

    assert schema_state.partial_parameter_name == "na"
    assert schema_state.current_parameter is None
    assert machine.state == JasonStates.IN_STRING
    assert machine.string_role == "key"

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: '"'},
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] == float("-inf")


# ---------------------------------------------------------------------------
# Chapter 6 stress tests:
# runtime function-name + parameter-name masking
# ---------------------------------------------------------------------------


def make_schema_name_stress_context():
    definitions = [
        InputDefinition(
            name="fn_alpha",
            description="alpha",
            parameters={
                "name": ParameterDefinition(type="string"),
                "nickname": ParameterDefinition(type="string"),
                "age": ParameterDefinition(type="number"),
            },
            returns=ReturnsDefinition(type="string"),
        ),
        InputDefinition(
            name="fn_alpine",
            description="alpine",
            parameters={
                "source": ParameterDefinition(type="string"),
            },
            returns=ReturnsDefinition(type="string"),
        ),
        InputDefinition(
            name="fn_beta",
            description="beta",
            parameters={
                "value": ParameterDefinition(type="number"),
            },
            returns=ReturnsDefinition(type="number"),
        ),
    ]

    function_names = [definition.name for definition in definitions]

    return definitions, function_names


def test_function_name_opening_quote_is_allowed():
    definitions, function_names = make_schema_name_stress_context()

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.NAME

    machine = JasonStateMacheine()
    machine.brackets = ["{"]
    machine.state = JasonStates.EXPECT_VALUE

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: '"'},
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] == 1.0


@pytest.mark.parametrize(
    "candidate",
    [
        '"f',
        '"fn',
        '"fn_',
        '"fn_a',
        '"fn_al',
        '"fn_alp',
        '"fn_alpha',
        '"fn_alpine',
        '"fn_b',
        '"fn_beta',
    ],
)
def test_function_name_valid_runtime_prefixes_are_allowed(candidate):
    definitions, function_names = make_schema_name_stress_context()

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

    assert result[0] == 1.0


@pytest.mark.parametrize(
    "candidate",
    [
        '"banana',
        '"fn_z',
        '"fn_gamma',
        '"totally_fake',
    ],
)
def test_function_name_invalid_runtime_prefixes_are_blocked(candidate):
    definitions, function_names = make_schema_name_stress_context()

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

    assert result[0] == float("-inf")


@pytest.mark.parametrize(
    "candidate",
    [
        '"fn_alpha"',
        '"fn_alpine"',
        '"fn_beta"',
    ],
)
def test_complete_runtime_function_names_can_close(candidate):
    definitions, function_names = make_schema_name_stress_context()

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

    assert result[0] == 1.0


def test_shared_function_prefix_does_not_force_choice_too_early():
    definitions, function_names = make_schema_name_stress_context()

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.NAME

    machine = JasonStateMacheine()
    machine.brackets = ["{"]
    machine.state = JasonStates.EXPECT_VALUE

    parameters = schema_state.get_parameters(definitions)

    for char in '"fn_al':
        machine.transition(char)
        schema_state.update(
            char,
            machine,
            function_names,
            parameters,
        )

    assert schema_state.function_name is None

    possible = schema_state.get_possible_function_names(function_names)

    assert set(possible) == {"fn_alpha", "fn_alpine"}


@pytest.mark.parametrize(
    ("partial", "candidate", "should_be_allowed"),
    [
        ("fn_", "alpha", True),
        ("fn_", "beta", True),
        ("fn_", "banana", False),
        ("fn_al", "pha", True),
        ("fn_al", "pine", True),
        ("fn_al", "banana", False),
        ("fn_b", "eta", True),
        ("fn_b", "anana", False),
    ],
)
def test_function_name_split_token_continuations(
    partial,
    candidate,
    should_be_allowed,
):
    definitions, function_names = make_schema_name_stress_context()

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.NAME
    schema_state.partial_function_name = partial

    machine = JasonStateMacheine()
    machine.brackets = ["{"]
    machine.state = JasonStates.IN_STRING
    machine.string_role = "value"

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


@pytest.mark.parametrize(
    ("partial", "should_be_allowed"),
    [
        ("fn_", False),
        ("fn_a", False),
        ("fn_al", False),
        ("fn_alp", False),
        ("fn_alpha", True),
        ("fn_alpine", True),
        ("fn_b", False),
        ("fn_beta", True),
    ],
)
def test_function_name_can_only_close_on_exact_runtime_name(
    partial,
    should_be_allowed,
):
    definitions, function_names = make_schema_name_stress_context()

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.NAME
    schema_state.partial_function_name = partial

    machine = JasonStateMacheine()
    machine.brackets = ["{"]
    machine.state = JasonStates.IN_STRING
    machine.string_role = "value"

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: '"'},
        schema_state,
        function_names,
        definitions,
    )

    if should_be_allowed:
        assert result[0] == 1.0
    else:
        assert result[0] == float("-inf")


def test_function_name_schema_mask_stops_after_name_value():
    definitions, function_names = make_schema_name_stress_context()

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.NAME
    schema_state.function_name = "fn_beta"
    schema_state.partial_function_name = "fn_beta"

    machine = JasonStateMacheine()
    machine.brackets = ["{"]
    machine.state = JasonStates.AFTER_VALUE

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: ","},
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] == 1.0


def make_parameter_stress_context():
    definitions, function_names = make_schema_name_stress_context()

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.PARAMETERS
    schema_state.function_name = "fn_alpha"

    machine = JasonStateMacheine()
    machine.brackets = ["{", "{"]

    return (
        machine,
        schema_state,
        function_names,
        definitions,
    )


@pytest.mark.parametrize(
    "candidate",
    [
        '"',
        '"n',
        '"na',
        '"nam',
        '"name',
        '"ni',
        '"nick',
        '"nickname',
        '"a',
        '"ag',
        '"age',
    ],
)
def test_parameter_valid_runtime_prefixes_are_allowed(candidate):
    (
        machine,
        schema_state,
        function_names,
        definitions,
    ) = make_parameter_stress_context()

    machine.state = JasonStates.EXPECT_KEY

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: candidate},
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] == 1.0


@pytest.mark.parametrize(
    "candidate",
    [
        '"banana',
        '"names',
        '"nickname_extra',
        '"ages',
        '"source',
        '"value',
    ],
)
def test_parameter_invalid_runtime_prefixes_are_blocked(candidate):
    (
        machine,
        schema_state,
        function_names,
        definitions,
    ) = make_parameter_stress_context()

    machine.state = JasonStates.EXPECT_KEY

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: candidate},
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] == float("-inf")


@pytest.mark.parametrize(
    "candidate",
    [
        '"name"',
        '"nickname"',
        '"age"',
    ],
)
def test_complete_runtime_parameter_names_can_close(candidate):
    (
        machine,
        schema_state,
        function_names,
        definitions,
    ) = make_parameter_stress_context()

    machine.state = JasonStates.EXPECT_KEY

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: candidate},
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] == 1.0


@pytest.mark.parametrize(
    ("partial", "should_be_allowed"),
    [
        ("n", False),
        ("na", False),
        ("nam", False),
        ("name", True),
        ("ni", False),
        ("nick", False),
        ("nickname", True),
        ("a", False),
        ("ag", False),
        ("age", True),
    ],
)
def test_parameter_name_can_only_close_on_exact_runtime_parameter(
    partial,
    should_be_allowed,
):
    (
        machine,
        schema_state,
        function_names,
        definitions,
    ) = make_parameter_stress_context()

    schema_state.partial_parameter_name = partial

    machine.state = JasonStates.IN_STRING
    machine.string_role = "key"

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: '"'},
        schema_state,
        function_names,
        definitions,
    )

    if should_be_allowed:
        assert result[0] == 1.0
    else:
        assert result[0] == float("-inf")


@pytest.mark.parametrize(
    ("partial", "candidate", "should_be_allowed"),
    [
        ("n", "a", True),
        ("na", "me", True),
        ("n", "ick", True),
        ("nick", "name", True),
        ("a", "ge", True),
        ("na", "banana", False),
        ("nick", "banana", False),
        ("a", "banana", False),
    ],
)
def test_parameter_name_split_token_continuations(
    partial,
    candidate,
    should_be_allowed,
):
    (
        machine,
        schema_state,
        function_names,
        definitions,
    ) = make_parameter_stress_context()

    schema_state.partial_parameter_name = partial

    machine.state = JasonStates.IN_STRING
    machine.string_role = "key"

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


def test_parameter_names_are_checked_against_selected_function_only():
    (
        machine,
        schema_state,
        function_names,
        definitions,
    ) = make_parameter_stress_context()

    machine.state = JasonStates.EXPECT_KEY

    # "source" exists, but belongs to fn_alpine, not fn_alpha.
    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: '"source"'},
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] == float("-inf")


def test_parameter_name_is_not_compared_against_function_names():
    (
        machine,
        schema_state,
        function_names,
        definitions,
    ) = make_parameter_stress_context()

    schema_state.partial_parameter_name = "name"

    machine.state = JasonStates.IN_STRING
    machine.string_role = "key"

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: '"'},
        schema_state,
        function_names,
        definitions,
    )

    # "name" is a parameter, not a function name.
    # It must still be allowed to close.
    assert result[0] == 1.0


def test_function_name_is_not_compared_against_parameter_names():
    definitions, function_names = make_schema_name_stress_context()

    schema_state = SchemaState()
    schema_state.stage = SchemaStage.NAME
    schema_state.partial_function_name = "fn_beta"

    machine = JasonStateMacheine()
    machine.brackets = ["{"]
    machine.state = JasonStates.IN_STRING
    machine.string_role = "value"

    result = mask_invalid_tokens(
        [1.0],
        machine,
        {0: '"'},
        schema_state,
        function_names,
        definitions,
    )

    assert result[0] == 1.0

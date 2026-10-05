"""Masking checks at positions reached by actual function-call generation.

Generic JSON syntax remains covered by test_json_state.py. These checks pass
a real SchemaState and runtime definitions, just as generation does.
"""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from src.json_state import JasonStateMacheine
from src.masking import mask_invalid_tokens
from src.schema import InputDefinition, ParameterDefinition
from src.schema_state import SchemaState


REQUEST = "request"
FUNCTION = "fn_runtime"
DEFINITIONS = [
    InputDefinition(
        name=FUNCTION,
        description="A runtime-defined function for constraint checks.",
        parameters={
            "text": {"type": "string"},
            "amount": {"type": "number"},
            "count": {"type": "integer"},
            "enabled": {"type": "boolean"},
            "nothing": {"type": "null"},
        },
        returns={"type": "string"},
    )
]
PARAMETERS_PREFIX = (
    '{"prompt":"request","name":"fn_runtime","parameters":{'
)


def make_context(prefix):
    """Advance both live states as generation does for the accepted prefix."""
    machine = JasonStateMacheine()
    schema = SchemaState(REQUEST)
    for char in prefix:
        parameters = schema.get_parameters(DEFINITIONS)
        assert machine.transition(char), (prefix, char)
        schema.update(char, machine, [FUNCTION], parameters)
    return machine, schema


def check_candidate(prefix, candidate, allowed):
    machine, schema = make_context(prefix)
    before_json = deepcopy(vars(machine))
    before_schema = deepcopy(vars(schema))
    result = mask_invalid_tokens(
        [7.0], machine, {0: candidate}, schema, [FUNCTION], DEFINITIONS
    )
    assert result[0] == (7.0 if allowed else float("-inf")), (
        prefix, candidate, result
    )
    assert vars(machine) == before_json
    assert vars(schema) == before_schema


@pytest.mark.parametrize("candidate", ["[", '"', "true", "null", "1", ":"])
def test_call_must_start_with_an_object(candidate):
    check_candidate("", candidate, False)


def test_object_start_preserves_legal_score():
    check_candidate("", "{", True)


def test_missing_vocab_id_is_masked():
    machine, schema = make_context("")
    result = mask_invalid_tokens(
        [3.0, 9.0], machine, {0: "{"}, schema, [FUNCTION], DEFINITIONS
    )
    assert result == [3.0, float("-inf")]


@pytest.mark.parametrize(
    "prefix,candidate,allowed",
    [
        ('{"', "p", True),
        ('{"', "name", False),
        ('{"pro', 'mpt"', True),
        ('{"pro', 'x"', False),
        ('{"prompt"', ":", True),
        ('{"prompt"', " :", True),
        ('{"prompt"', "\t:", True),
        ('{"prompt"', ":hello", False),
        ('{"prompt"', "]", False),
    ],
)
def test_outer_key_and_colon_candidates(prefix, candidate, allowed):
    check_candidate(prefix, candidate, allowed)


@pytest.mark.parametrize(
    "key,candidate,allowed",
    [
        ("text", '"hello"', True),
        ("text", ' "hello"', True),
        ("text", "1", False),
        ("text", "true", False),
        ("amount", "-12.5e2", True),
        ("amount", " 12", True),
        ("amount", '"12"', False),
        ("amount", "true", False),
        ("count", "12", True),
        ("count", "-12", True),
        ("count", "0", True),
        ("count", "12.5", False),
        ("count", "1e3", False),
        ("count", "1E3", False),
        ("count", '"12"', False),
        ("enabled", "true", True),
        ("enabled", "false", True),
        ("enabled", "t", True),
        ("enabled", " false", True),
        ("enabled", "1", False),
        ("enabled", "null", False),
        ("enabled", '"true"', False),
        ("enabled", "truX", False),
        ("nothing", "null", True),
        ("nothing", "n", True),
        ("nothing", " null", True),
        ("nothing", "false", False),
        ("nothing", "0", False),
        ("nothing", '"null"', False),
        ("nothing", "nulX", False),
    ],
)
def test_runtime_parameter_value_candidates(key, candidate, allowed):
    check_candidate(PARAMETERS_PREFIX + f'"{key}":', candidate, allowed)


@pytest.mark.parametrize("key", ["text", "amount", "count", "enabled", "nothing"])
@pytest.mark.parametrize("candidate", [" ", "\t", "\r", "\n"])
def test_whitespace_before_runtime_parameter_value(key, candidate):
    check_candidate(PARAMETERS_PREFIX + f'"{key}":', candidate, True)


@pytest.mark.parametrize(
    "key,value_prefix,candidate,allowed",
    [
        ("amount", "12", ".5", True),
        ("amount", "12", "e3", True),
        ("amount", "12", "e+3", True),
        ("amount", "-", "1", True),
        ("count", "12", "3", True),
        ("count", "-", "1", True),
        ("count", "12", ".5", False),
        ("count", "12", "e3", False),
        ("count", "12", "E3", False),
        ("count", "12", "e+3", False),
        ("count", "12", ",", True),
        ("enabled", "t", "rue", True),
        ("enabled", "f", "alse", True),
        ("enabled", "t", "rux", False),
        ("nothing", "n", "ull", True),
        ("nothing", "n", "ulx", False),
        ("text", '"hello', '"', True),
        ("text", '"hello', "\n", False),
        ("text", '"hello\\', "n", True),
        ("text", '"hello\\', "x", False),
        ("text", '"\\u', "0041", True),
        ("text", '"\\u', "00G1", False),
        ("text", '"hello"', ",", True),
        ("text", '"hello"', "]", False),
        ("text", '"hello"', "oops", False),
    ],
)
def test_candidates_inside_runtime_parameter_values(
    key, value_prefix, candidate, allowed
):
    prefix = PARAMETERS_PREFIX + f'"{key}":' + value_prefix
    check_candidate(prefix, candidate, allowed)


@pytest.mark.parametrize("type_name", ["string", "number", "integer", "boolean", "null"])
def test_supported_runtime_parameter_type(type_name):
    assert ParameterDefinition(type=type_name).type == type_name


@pytest.mark.parametrize("type_name", ["array", "object", "float", "unknown"])
def test_unsupported_runtime_parameter_type_is_rejected(type_name):
    with pytest.raises(ValidationError):
        ParameterDefinition(type=type_name)

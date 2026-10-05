"""Chapter 3.2: outer record, exact request, runtime names, and handover.

Required/unknown/duplicate parameters and empty-parameter closure belong to
Chapter 3.3. These tests exercise the completed Chapter 3.2 boundaries.
"""

from copy import deepcopy
import json

import pytest

from src.json_state import JasonStateMacheine, JasonStates
from src.masking import mask_invalid_tokens
from src.schema import InputDefinition
from src.schema_state import SchemaState, SchemaStage


def definitions_for(*names):
    return [
        InputDefinition(
            name=name, description="Runtime function.",
            parameters={"value": {"type": "string"}},
            returns={"type": "string"},
        )
        for name in names
    ]


DEFINITIONS = definitions_for("fn_alpha", "fn_alpha_more", "fn_beta")
OUTER_PREFIX = '{"prompt":"x","name":"fn_alpha","parameters":'
RECORD = OUTER_PREFIX + '{"value":"ok"}}'


def advance(machine, schema, text, definitions):
    names = [definition.name for definition in definitions]
    for char in text:
        assert machine.transition(char), (text, char, machine.state)
        schema.update(char, machine, names, schema.get_parameters(definitions))


def context(prefix, request="x", definitions=None):
    if definitions is None:
        definitions = DEFINITIONS
    machine, schema = JasonStateMacheine(), SchemaState(request)
    advance(machine, schema, prefix, definitions)
    return machine, schema


def check(prefix, candidate, allowed, request="x", definitions=None):
    if definitions is None:
        definitions = DEFINITIONS
    machine, schema = context(prefix, request, definitions)
    before_machine = deepcopy(vars(machine))
    before_schema = deepcopy(vars(schema))
    result = mask_invalid_tokens(
        [7.5], machine, {0: candidate}, schema,
        [definition.name for definition in definitions], definitions,
    )
    assert result == [7.5 if allowed else float("-inf")], (prefix, candidate)
    assert vars(machine) == before_machine
    assert vars(schema) == before_schema
    return machine, schema


@pytest.mark.parametrize("candidate", ["{", " {", "\t{", "\r\n{", " "])
def test_root_object_and_leading_whitespace(candidate):
    check("", candidate, True)


@pytest.mark.parametrize(
    "candidate", ["[]", '"x"', "0", "true", "false", "null", "{}", '{"name":']
)
def test_wrong_root_or_missing_first_field(candidate):
    check("", candidate, False)


@pytest.mark.parametrize(
    "prefix,candidate,allowed",
    [
        ("{", '"prompt":', True),
        ("{", ' \n"prompt" : ', True),
        ("{", '"name":', False),
        ("{", '"parameters":', False),
        ("{", '"extra":', False),
        ('{"pro', 'mpt":', True),
        ('{"pro', 'mpX":', False),
        ('{"prompt":"x",', '"name":', True),
        ('{"prompt":"x",', ' "name" : "fn_', True),
        ('{"prompt":"x",', '"prompt":', False),
        ('{"prompt":"x",', '"parameters":', False),
        ('{"prompt":"x","name":"fn_alpha",', '"parameters":{', True),
        ('{"prompt":"x","name":"fn_alpha",', '"name":', False),
        ('{"prompt":"x","name":"fn_alpha",', '"prompt":', False),
        ('{"prompt":"x","name":"fn_alpha",', '"extra":', False),
    ],
)
def test_outer_key_order_unknown_keys_and_compound_tokens(prefix, candidate, allowed):
    check(prefix, candidate, allowed)


@pytest.mark.parametrize(
    "prefix",
    ["{", '{"prompt":"x"', '{"prompt":"x","name":"fn_alpha"'],
)
@pytest.mark.parametrize("candidate", ["}", " }"])
def test_outer_object_cannot_close_before_all_fields(prefix, candidate):
    check(prefix, candidate, False)


@pytest.mark.parametrize(
    "candidate",
    [
        '{"prompt":"x"}',
        '{"prompt":"x","name":"fn_alpha"}',
        '{"prompt":"x","parameters":{"value":"ok"}}',
        '{"prompt":"x","name":"missing","parameters":{"value":"ok"}}',
    ],
)
def test_compound_tokens_cannot_bypass_missing_fields_or_membership(candidate):
    check("", candidate, False)


@pytest.mark.parametrize(
    "prefix,candidate",
    [
        ('{"prompt":', "1"),
        ('{"prompt":', "true"),
        ('{"prompt":', "null"),
        ('{"prompt":', "{}"),
        ('{"prompt":"x","name":', "1"),
        ('{"prompt":"x","name":', "true"),
        ('{"prompt":"x","name":', "{}"),
        (OUTER_PREFIX, '"value"'),
        (OUTER_PREFIX, "[]"),
        (OUTER_PREFIX, "null"),
        (OUTER_PREFIX, "1"),
    ],
)
def test_outer_values_have_required_types(prefix, candidate):
    check(prefix, candidate, False)


REQUESTS = [
    "", "x", "hello world", 'say "hello"', "C:\\tmp\\file",
    "line one\nline two", "a\tb\rc", "\b\f", "\x00\x01\x1f",
    "مرحبا يا حياة", "emoji 🐇", "braces } { and commas ,", 'slash / and quote "',
]


@pytest.mark.parametrize("original_request", REQUESTS)
def test_serialized_prompt_is_accepted_and_preserves_original_request(original_request):
    literal = json.dumps(original_request, ensure_ascii=False)
    check('{"prompt":', " \t" + literal, True, request=original_request)
    check("", '{"prompt":' + literal + ',"name":"fn_alpha",', True, request=original_request)
    _, schema = context('{"prompt":' + literal, request=original_request)
    assert schema.partial_og_prompt == literal
    assert json.loads(schema.partial_og_prompt) == original_request
    assert schema.completed_outer_fields == {"prompt"}


@pytest.mark.parametrize("original_request", REQUESTS)
def test_prompt_survives_character_splits_including_escape_sequences(original_request):
    machine, schema = context('{"prompt":', request=original_request)
    names = [definition.name for definition in DEFINITIONS]
    literal = json.dumps(original_request, ensure_ascii=False)
    for char in literal:
        before = deepcopy(vars(schema))
        result = mask_invalid_tokens([2.0], machine, {0: char}, schema, names, DEFINITIONS)
        assert result == [2.0], (original_request, char, schema.partial_og_prompt)
        assert vars(schema) == before
        advance(machine, schema, char, DEFINITIONS)
    assert schema.partial_og_prompt == literal


@pytest.mark.parametrize(
    "prefix,candidate",
    [
        ('{"prompt":', '"wrong"'),
        ('{"prompt":', '""'),
        ('{"prompt":"', "y"),
        ('{"prompt":"', '"'),
        ('{"prompt"', ':"wrong"'),
        ("{", '"prompt":"wrong"'),
        ("", '{"prompt":"wrong","name":"fn_alpha",'),
        ('{"prompt":', '"xy"'),
    ],
)
def test_wrong_shortened_or_extended_prompt_is_masked(prefix, candidate):
    check(prefix, candidate, False)


@pytest.mark.parametrize(
    "prefix,candidate,allowed",
    [
        ('{"prompt":"x","name":', ' "fn_alpha"', True),
        ('{"prompt":"x","name":"fn_alpha', '"', True),
        ('{"prompt":"x","name":"fn_alpha', '_more"', True),
        ('{"prompt":"x","name":"fn_alpha', 'X"', False),
        ('{"prompt":"x","name":"fn_al', '"', False),
        ('{"prompt":"x","name":', '"missing"', False),
        ('{"prompt":"x","name"', ':"missing"', False),
        ('{"prompt":"x","name":"fn_alpha', '","parameters":{', True),
        ('{"prompt":"x","name":"fn_alpha', '_more","parameters":{', True),
    ],
)
def test_runtime_name_prefixes_exact_closure_and_shared_prefixes(prefix, candidate, allowed):
    check(prefix, candidate, allowed)


@pytest.mark.parametrize("name", ["fn_alpha", "fn_alpha_more"])
def test_name_is_not_bound_until_its_closing_quote(name):
    machine, schema = context('{"prompt":"x","name":' + json.dumps(name)[:-1])
    assert schema.function_name is None
    assert schema.get_parameters(DEFINITIONS) == {}
    possible = schema.get_possible_function_names([definition.name for definition in DEFINITIONS])
    assert name in possible
    advance(machine, schema, '"', DEFINITIONS)
    assert schema.function_name == name
    assert schema.get_parameters(DEFINITIONS) == next(
        definition.parameters for definition in DEFINITIONS if definition.name == name
    )


@pytest.mark.parametrize("name", ['fn_quote"x', "fn_slash\\x", "عملية", "fn_line\nx"])
def test_runtime_names_with_escaping_or_unicode(name):
    definitions = definitions_for(name)
    literal = json.dumps(name, ensure_ascii=False)
    check('{"prompt":"x","name":', literal, True, definitions=definitions)
    machine, schema = context('{"prompt":"x","name":', definitions=definitions)
    for char in literal:
        result = mask_invalid_tokens([1.0], machine, {0: char}, schema, [name], definitions)
        assert result == [1.0]
        advance(machine, schema, char, definitions)
    assert schema.partial_function_name == literal
    assert schema.function_name == name


def test_changed_runtime_names_replace_previous_choices():
    definitions = definitions_for("fn_new_runtime")
    prefix = '{"prompt":"x","name":'
    check(prefix, '"fn_new_runtime"', True, definitions=definitions)
    check(prefix, '"fn_alpha"', False, definitions=definitions)


def test_empty_runtime_name_set_has_no_legal_name():
    check('{"prompt":"x","name":', '"fn_alpha"', False, definitions=[])


@pytest.mark.parametrize("split", range(1, len(OUTER_PREFIX) + 1))
def test_valid_record_remainder_at_every_outer_character_boundary(split):
    check(RECORD[:split], RECORD[split:], True)


@pytest.mark.parametrize(
    "prefix,completed",
    [
        ('{"prompt"', set()),
        ('{"prompt":"x', set()),
        ('{"prompt":"x"', {"prompt"}),
        ('{"prompt":"x","name":"fn_alpha', {"prompt"}),
        ('{"prompt":"x","name":"fn_alpha"', {"prompt", "name"}),
        (OUTER_PREFIX + '{"value":"ok"', {"prompt", "name"}),
        (RECORD[:-1], {"prompt", "name", "parameters"}),
    ],
)
def test_completed_fields_require_finished_values(prefix, completed):
    _, schema = context(prefix)
    assert schema.completed_outer_fields == completed
    assert schema.has_complete_outer_fields() == (len(completed) == 3)


def test_closing_brace_inside_parameter_string_does_not_close_outer_record():
    check(OUTER_PREFIX + '{"value":"', '} remains text"}}', True)
    _, schema = context(OUTER_PREFIX + '{"value":"}')
    assert schema.completed_outer_fields == {"prompt", "name"}


@pytest.mark.parametrize("suffix", [" ", "\n", "\t\r"])
def test_completed_record_allows_only_json_whitespace(suffix):
    check(RECORD, suffix, True)


@pytest.mark.parametrize("suffix", [" prose", "{}", ",", "]", "x"])
def test_completed_record_rejects_extra_output(suffix):
    check(RECORD, suffix, False)


def test_repeated_mask_calls_and_candidate_order_preserve_live_states():
    machine, schema = context('{"prompt":"x","name":"fn_al')
    before_machine, before_schema = deepcopy(vars(machine)), deepcopy(vars(schema))
    vocab = {0: 'pha","parameters":{"value":"ok"}}', 1: '"', 2: 'pha_more"'}
    names = [definition.name for definition in DEFINITIONS]
    for order in [list(vocab), list(reversed(vocab)), list(vocab)]:
        reordered = dict(enumerate(vocab[index] for index in order))
        result = mask_invalid_tokens([3.0] * 3, machine, reordered, schema, names, DEFINITIONS)
        expected = [3.0 if index != 1 else float("-inf") for index in order]
        assert result == expected
        assert vars(machine) == before_machine
        assert vars(schema) == before_schema
    assert schema.function_name is None


def test_fresh_schema_instances_do_not_share_tracking_collections():
    _, first = context(RECORD)
    second = SchemaState("new request")
    assert first.has_complete_outer_fields()
    assert second.stage == SchemaStage.PROMPT
    assert second.outer_key_candidates == ["prompt", "name", "parameters"]
    assert second.completed_outer_fields == set()
    assert second.completed_parameters == set()
    assert second.function_name is None
    assert second.partial_og_prompt == ""
    assert second.partial_function_name == ""

"""Chapter 3.3: current copying, runtime parameters, and generation regressions.

Run from the project root:
    uv run python -m pytest -q tests/test_json_state.py tests/test_chapter_33_schema.py

Controlled model scores exercise production generation and masking without
loading a model. These cases follow the current compact-JSON/copying design.
Number parameters use decimal or exponent notation for float-compatible output;
integer parameters retain integer notation. JSON grammar coverage remains in
test_json_state.py. These cases do not measure model accuracy or speed.
"""

from copy import deepcopy
import json

from pydantic import ValidationError
import pytest

from src.generation import generate_constrained
from src.json_state import JasonStateMacheine
from src.masking import mask_invalid_tokens
from src.schema import InputDefinition, InputFileDefinition
from src.schema_state import SchemaState
from src.vocab import build_vocab_map


MODEL_PROMPT = "separate model instructions\nJSON:"


def definition(parameters=None, name="fn_runtime"):
    if parameters is None:
        parameters = {"value": "string"}
    return InputDefinition(
        name=name,
        description="Function supplied at runtime.",
        parameters={key: {"type": kind} for key, kind in parameters.items()},
        returns={"type": "string"},
    )


def advance(machine, schema, text, definitions):
    names = [item.name for item in definitions]
    for char in text:
        parameters = schema.get_parameters(definitions)
        assert machine.transition(char), (text, char, machine.state)
        schema.update(char, machine, names, parameters)


def context(prefix, original_request="x", definitions=None):
    if definitions is None:
        definitions = [definition()]
    machine = JasonStateMacheine()
    schema = SchemaState(original_request)
    advance(machine, schema, prefix, definitions)
    return machine, schema


def check(prefix, candidate, allowed, definitions=None, original_request="x"):
    if definitions is None:
        definitions = [definition()]
    machine, schema = context(prefix, original_request, definitions)
    before_machine = deepcopy(vars(machine))
    before_schema = deepcopy(vars(schema))
    scores = mask_invalid_tokens(
        [7.5], machine, {0: candidate}, schema,
        [item.name for item in definitions], definitions,
    )
    assert scores == [7.5 if allowed else float("-inf")], (prefix, candidate)
    assert vars(machine) == before_machine
    assert vars(schema) == before_schema
    return machine, schema


def outer_prefix(name="fn_runtime", original_request="x"):
    return (
        '{"prompt":' + json.dumps(original_request, ensure_ascii=False)
        + ',"name":' + json.dumps(name, ensure_ascii=False)
        + ',"parameters":'
    )


@pytest.mark.parametrize("kind", ["string", "number", "integer", "boolean", "null"])
def test_supported_runtime_scalar_types_are_validated(kind):
    result = InputFileDefinition.model_validate([
        definition({"argument": kind}).model_dump()
    ])
    assert result.root[0].parameters["argument"].type == kind


@pytest.mark.parametrize("kind", ["object", "array", "float", "bool", "unknown"])
def test_unsupported_parameter_types_are_rejected_at_input_boundary(kind):
    with pytest.raises(ValidationError):
        definition({"argument": kind})


@pytest.mark.parametrize(
    "kind,value",
    [
        ("string", '""'), ("string", '"hello world"'),
        ("string", '"say \\"hi\\""'), ("string", '"C:\\\\tmp"'),
        ("number", "0.0"), ("number", "-0.0"), ("number", "42.0"),
        ("number", "-2.5"), ("number", "0.125"),
        ("number", "3e+2"), ("number", "1E-3"),
        ("integer", "0"), ("integer", "-0"),
        ("integer", "42"), ("integer", "-42"),
        ("boolean", "true"), ("boolean", "false"), ("null", "null"),
    ],
)
def test_valid_scalar_values_can_finish_the_complete_record(kind, value):
    definitions = [definition({"argument": kind})]
    prefix = outer_prefix() + '{"argument":'
    check(prefix, value + "}}", True, definitions)


@pytest.mark.parametrize(
    "kind,value",
    [
        ("string", "1"), ("string", "true"), ("string", "null"),
        ("string", "{}"), ("string", "[]"),
        ("number", '"1"'), ("number", "true"), ("number", "null"),
        ("number", "{}"), ("number", "[]"),
        ("integer", "1.0"), ("integer", "1e2"), ("integer", "1E+2"),
        ("integer", '"1"'), ("integer", "true"), ("integer", "null"),
        ("boolean", "0"), ("boolean", '"true"'), ("boolean", "null"),
        ("boolean", "{}"), ("boolean", "[]"),
        ("null", "0"), ("null", '"null"'), ("null", "false"),
        ("null", "{}"), ("null", "[]"),
    ],
)
def test_wrong_scalar_types_cannot_finish_the_record(kind, value):
    definitions = [definition({"argument": kind})]
    check(outer_prefix() + '{"argument":', value + "}}", False, definitions)


@pytest.mark.parametrize("value", ["0", "-0", "42"])
@pytest.mark.parametrize("delimiter", [",", "}}"])
def test_number_cannot_finish_without_decimal_or_exponent(value, delimiter):
    definitions = [definition({"argument": "number"})]
    check(outer_prefix() + '{"argument":', value + delimiter, False, definitions)


@pytest.mark.parametrize("value", ["0", "-0", "42"])
@pytest.mark.parametrize("suffix", [".0}}", "e0}}"])
def test_number_can_continue_into_float_notation_across_tokens(value, suffix):
    definitions = [definition({"argument": "number"})]
    check(outer_prefix() + '{"argument":' + value, suffix, True, definitions)


@pytest.mark.parametrize("candidate", [".", "e", "E", ".0}}", "e+2}}", "E-2}}"])
def test_integer_restriction_survives_token_splits(candidate):
    definitions = [definition({"count": "integer"})]
    check(outer_prefix() + '{"count":12', candidate, False, definitions)


@pytest.mark.parametrize("prefix,candidate", [
    ("-", "}"), ("1.", "}"), ("1e", "}"), ("1e+", "}"),
    ("t", "}"), ("fals", "}"), ("nul", "}"),
])
def test_incomplete_scalar_cannot_close_parameters(prefix, candidate):
    kind = "boolean" if prefix[0] in "tf" else "null" if prefix[0] == "n" else "number"
    definitions = [definition({"value": kind})]
    check(outer_prefix() + '{"value":' + prefix, candidate, False, definitions)


@pytest.mark.parametrize("candidate", ['"a"', '"unknown"', "}"])
def test_only_remaining_runtime_parameter_keys_are_eligible(candidate):
    definitions = [definition({"a": "number", "b": "string"})]
    prefix = outer_prefix() + '{"a":2.0,'
    check(prefix, candidate, False, definitions)


@pytest.mark.parametrize("candidate", ['"b"', '"b":"ok"}}'])
def test_remaining_parameter_key_and_compound_value_are_allowed(candidate):
    definitions = [definition({"a": "number", "b": "string"})]
    check(outer_prefix() + '{"a":2.0,', candidate, True, definitions)


@pytest.mark.parametrize("prefix,candidate", [
    ('{"c', "ount"), ('{"c', "ounter"), ('{"count', '"'),
    ('{"count', "er"), ('{"counter', '"'),
])
def test_shared_parameter_prefixes_remain_eligible(prefix, candidate):
    definitions = [definition({"count": "integer", "counter": "string"})]
    check(outer_prefix() + prefix, candidate, True, definitions)


@pytest.mark.parametrize("prefix,candidate", [
    ('{"c', "at"), ('{"c', '"'), ('{"count', "x"),
])
def test_unknown_or_incomplete_parameter_key_is_rejected(prefix, candidate):
    definitions = [definition({"count": "integer", "counter": "string"})]
    check(outer_prefix() + prefix, candidate, False, definitions)


@pytest.mark.parametrize("prefix", [
    "{", '{"a":2.0', '{"a":2.0,', '{"b":"ok"',
])
def test_required_parameters_prevent_early_closure(prefix):
    definitions = [definition({"a": "number", "b": "string"})]
    check(outer_prefix() + prefix, "}", False, definitions)


@pytest.mark.parametrize("parameters", [
    '{"a":2.0,"b":"ok"}', '{"b":"ok","a":2.0}',
])
def test_all_required_parameters_can_complete_in_either_order(parameters):
    definitions = [definition({"a": "number", "b": "string"})]
    check(outer_prefix(), parameters + "}", True, definitions)


@pytest.mark.parametrize("candidate", [",", ',"a":3.0}}', ',"extra":3.0}}'])
def test_no_more_keys_after_all_parameters_are_complete(candidate):
    definitions = [definition({"a": "number"})]
    check(outer_prefix() + '{"a":2.0', candidate, False, definitions)


def test_zero_parameter_function_allows_empty_parameters():
    check(outer_prefix(), "{}}", True, [definition({})])


def test_zero_parameter_function_rejects_any_parameter():
    check(outer_prefix(), '{"extra":1}}', False, [definition({})])


@pytest.mark.parametrize("prefix,candidate,allowed", [
    ("", '{"', True), ("", '{"p', False), ("", '{}', False),
    ("", "[]", False), ("", "0", False), ("", "null", False),
    ("{", '"', True), ("{", '"prompt"', False),
    ("{", '"name"', False), ("{", '"extra"', False),
    ('{"prompt', '":"', True), ('{"prompt', '":"x', False),
    ('{"prompt":"x', '","', True), ('{"prompt":"x', '","n', False),
    ('{"prompt":"x","name', '":"fn_runtime",', True),
    ('{"prompt":"x","name":"fn_runtime",', '"', True),
    ('{"prompt":"x","name":"fn_runtime",', '"parameters"', False),
])
def test_outer_record_and_copied_text_boundaries(prefix, candidate, allowed):
    check(prefix, candidate, allowed)


@pytest.mark.parametrize("prefix", [
    "{", '{"prompt":"x"', '{"prompt":"x","name":"fn_runtime"',
])
def test_outer_object_cannot_close_with_missing_fields(prefix):
    check(prefix, "}", False)


@pytest.mark.parametrize("prefix,candidate", [
    ('{"prompt":', "0"), ('{"prompt":', "null"),
    ('{"prompt":"x","name":', "0"),
    ('{"prompt":"x","name":', "{}"),
    (outer_prefix(), "[]"), (outer_prefix(), "null"),
    (outer_prefix(), '"value"'),
])
def test_outer_field_types_are_enforced(prefix, candidate):
    check(prefix, candidate, False)


@pytest.mark.parametrize("candidate,allowed", [
    ('"fn_runtime"', True), ('"fn_other"', False), ('"fn_runtime_more"', False),
])
def test_function_name_value_comes_from_runtime_definitions(candidate, allowed):
    check('{"prompt":"x","name":', candidate, allowed)


@pytest.mark.parametrize("candidate,allowed", [
    ('_more"', True), ('"', True), ('_missing"', False),
])
def test_shared_function_name_prefixes_wait_for_exact_name(candidate, allowed):
    definitions = [definition(name="fn_alpha"), definition(name="fn_alpha_more")]
    check('{"prompt":"x","name":"fn_alpha', candidate, allowed, definitions)


@pytest.mark.parametrize("prefix,candidate", [
    ("", " {"), ("{", '\n"'), ('{"prompt', '" :"'),
    (outer_prefix() + '{"value":', ' "ok"'),
    (outer_prefix() + '{"value":"ok"', " }"),
])
def test_formatting_whitespace_is_rejected_outside_strings(prefix, candidate):
    check(prefix, candidate, False)


@pytest.mark.parametrize("candidate", ["hello world", "  ", "}{,", "[]:,", "مرحبا"])
def test_spaces_and_json_punctuation_remain_legal_inside_argument_strings(candidate):
    check(outer_prefix() + '{"value":"', candidate, True)


@pytest.mark.parametrize("candidate", ["\x00", "\n", "\t", "\r", "\\x", "\\q"])
def test_invalid_string_controls_or_escapes_are_rejected(candidate):
    check(outer_prefix() + '{"value":"', candidate, False)


@pytest.mark.parametrize("candidate", ["\\n", "\\t", "\\\\", '\\"', "\\u0041"])
def test_valid_argument_string_escapes_are_allowed(candidate):
    check(outer_prefix() + '{"value":"', candidate, True)


@pytest.mark.parametrize("partial,candidate,allowed", [
    (r"\u", "0041", True), (r"\u", "ABCD", True),
    (r"\u", "abcd", True), (r"\u", "00", True),
    (r"\u", '0041"', True), (r"\u", "00G1", False),
    (r"\u", '00"', False), (r"\u00", '41"', True),
    (r"\u00", 'G1"', False),
])
def test_pending_unicode_escape_digits_are_checked_across_tokens(partial, candidate, allowed):
    check(outer_prefix() + '{"value":"' + partial, candidate, allowed)


@pytest.mark.parametrize("first,second", [
    ("1e+2", "-3.0"), ("1e-2", "-3.0"), ("1.5", "0.0"),
    ("0.0", "2e+3"), ("1e+2", "2e-3"),
])
def test_number_flags_do_not_leak_into_the_next_parameter(first, second):
    definitions = [definition({"a": "number", "b": "number"})]
    prefix = outer_prefix() + '{"a":' + first + ',"b":'
    check(prefix, second + "}}", True, definitions)
    # Each number must establish its own decimal or exponent notation.
    check(prefix, "0}}", False, definitions)
    check(prefix, "-3}}", False, definitions)


@pytest.mark.parametrize("candidate,allowed", [
    ("}", True), (",", False), (',"prompt":', False), (',"extra":', False),
])
def test_complete_parameters_only_allow_finishing_the_outer_record(candidate, allowed):
    check(outer_prefix() + '{"value":"ok"}', candidate, allowed)


class ControlledModel:
    """Small reversible tokenizer and controlled scores; no inference backend."""

    def __init__(self, target, chunks=(), rankings=None):
        self.target = target
        self.vocab = {}
        self.ids = {}
        self.context_ids = []
        self.inputs = []
        self.generated_prefixes = []
        self.encode_calls = []
        self.emissions = []
        self.chunks = sorted(chunks, key=len, reverse=True)
        self.rankings = rankings or {}
        for char in target:
            self.register(char)
        for chunk in chunks:
            self.register(chunk)
        for ranked in self.rankings.values():
            for text in ranked:
                self.register(text)

    def register(self, text):
        if text not in self.ids:
            token_id = len(self.vocab)
            self.ids[text] = token_id
            self.vocab[token_id] = text
        return self.ids[text]

    def wrapper_encode(self, text):
        self.encode_calls.append(text)
        ids = [self.register(char) for char in text]
        if text.endswith("\nJSON:") and ids not in self.context_ids:
            self.context_ids.append(ids)
        return ids

    def wrapper_decode(self, token_ids):
        return "".join(self.vocab[token_id] for token_id in token_ids)

    def wrapper_get_logits_from_input_ids(self, token_ids):
        contexts = [
            ids for ids in self.context_ids
            if token_ids[:len(ids)] == ids
        ]
        assert contexts, "Model input lost its instruction prefix"
        current_context = max(contexts, key=len)
        prefix = self.wrapper_decode(token_ids[len(current_context):])
        assert self.target.startswith(prefix), (prefix, self.target)
        remaining = self.target[len(prefix):]
        assert remaining, "An inference call was made after completion"
        next_text = next(
            (chunk for chunk in self.chunks if remaining.startswith(chunk)),
            remaining[0],
        )
        scores = [float("-inf")] * len(self.vocab)
        scores[self.ids[next_text]] = 5.0
        for text, score in self.rankings.get(prefix, {}).items():
            scores[self.ids[text]] = score
        self.inputs.append(token_ids.copy())
        self.generated_prefixes.append(prefix)
        self.emissions.append(next_text)
        return scores


def record(original_request, name, values):
    return json.dumps(
        {"prompt": original_request, "name": name, "parameters": values},
        ensure_ascii=False, separators=(",", ":"),
    )


def generate(model, original_request="x", definitions=None, limit=500, schema=None):
    if definitions is None:
        definitions = [definition()]
    if schema is None:
        schema = SchemaState(original_request)
    input_ids = model.wrapper_encode(MODEL_PROMPT)
    result = generate_constrained(
        MODEL_PROMPT, limit, model, model.vocab, schema,
        [item.name for item in definitions], definitions,
    )
    assert result[:len(input_ids)] == input_ids
    output_ids = result[len(input_ids):]
    text = model.wrapper_decode(output_ids)
    assert text == model.target
    assert schema.has_complete_outer_fields()
    assert schema.has_complete_parameters(schema.get_parameters(definitions))
    assert schema.partial_og_prompt == schema.json_og_prompt
    return result, output_ids, schema


ORIGINAL_REQUESTS = [
    "", "x", "hello world", 'say "hello"', "C:\\tmp\\file",
    "line one\nline two", "a\tb\rc", "\b\f", "\x00\x01\x1f",
    "مرحبا يا حياة", "emoji 🐇", "braces } { and commas ,", "slash /",
]


@pytest.mark.parametrize("original_request", ORIGINAL_REQUESTS)
def test_generation_preserves_exact_request_through_copying(original_request):
    target = record(original_request, "fn_runtime", {"value": "ok"})
    model = ControlledModel(target)
    _, output_ids, _ = generate(model, original_request)
    assert json.loads(model.wrapper_decode(output_ids))["prompt"] == original_request
    assert "prompt" in model.encode_calls
    assert "name" in model.encode_calls
    assert "parameters" in model.encode_calls
    if original_request:
        assert json.dumps(original_request, ensure_ascii=False)[1:-1] in model.encode_calls


def test_copying_skips_inference_for_known_keys_and_original_request():
    target = record("hello world", "fn_runtime", {"value": "ok"})
    model = ControlledModel(target)
    _, output_ids, _ = generate(model, "hello world")
    model_text = "".join(model.emissions)
    assert "prompt" not in model_text
    assert "parameters" not in model_text
    assert "hello world" not in model_text
    assert "fn_runtime" in model_text
    assert "ok" in model_text
    assert len(model.inputs) < len(output_ids)


@pytest.mark.parametrize("key", ["value", "quoted\"key", "back\\slash", "مرحبا"])
def test_unique_runtime_parameter_key_is_copied_with_correct_escaping(key):
    definitions = [definition({key: "string"})]
    target = record("x", "fn_runtime", {key: "ok"})
    model = ControlledModel(target)
    _, output_ids, _ = generate(model, definitions=definitions)
    assert json.loads(model.wrapper_decode(output_ids))["parameters"] == {key: "ok"}
    assert json.dumps(key, ensure_ascii=False)[1:-1] in model.encode_calls


@pytest.mark.parametrize("kind,value", [
    ("string", 'text "quoted" \\ end'), ("string", ""),
    ("number", 1.25), ("number", -2.0), ("integer", -12),
    ("boolean", True), ("boolean", False), ("null", None),
])
def test_full_generation_with_each_supported_scalar_type(kind, value):
    definitions = [definition({"argument": kind})]
    model = ControlledModel(record("x", "fn_runtime", {"argument": value}))
    _, output_ids, _ = generate(model, definitions=definitions)
    parsed = json.loads(model.wrapper_decode(output_ids))["parameters"]["argument"]
    assert parsed == value
    assert type(parsed) is type(value)


@pytest.mark.parametrize("values", [
    {"count": 2, "counter": "ok"}, {"counter": "ok", "count": 2},
])
def test_generation_handles_shared_parameter_prefixes_in_either_order(values):
    definitions = [definition({"count": "integer", "counter": "string"})]
    model = ControlledModel(record("x", "fn_runtime", values))
    generate(model, definitions=definitions)


def test_generation_handles_zero_parameters():
    definitions = [definition({})]
    model = ControlledModel(record("x", "fn_runtime", {}))
    generate(model, definitions=definitions)


def test_changed_runtime_names_and_parameters_are_used():
    definitions = [
        definition({"old": "integer"}, name="fn_old"),
        definition({"message": "string"}, name="fn_new"),
    ]
    model = ControlledModel(record("x", "fn_new", {"message": "hello"}))
    _, _, schema = generate(model, definitions=definitions)
    assert schema.function_name == "fn_new"
    assert schema.completed_parameters == {"message"}


def test_multicharacter_tokens_advance_states_and_keep_generated_context():
    chunks = ['{"', '":"', '","', 'fn_runtime","', '":{"', '":"ok"}}']
    model = ControlledModel(record("x", "fn_runtime", {"value": "ok"}), chunks)
    generate(model)
    assert chunks[0] in model.emissions
    assert chunks[-1] in model.emissions
    assert model.generated_prefixes[1] == '{"prompt'


def test_selection_rejects_higher_invalid_scores_and_keeps_legal_ranking():
    definitions = [definition(name="fn_alpha"), definition(name="fn_beta")]
    target = record("x", "fn_beta", {"value": "ok"})
    name_prefix = '{"prompt":"x","name":"'
    model = ControlledModel(
        target, chunks=["fn_beta"],
        rankings={
            "": {"[]": 30.0, "{}": 20.0},
            name_prefix: {"fn_missing": 40.0, "fn_alpha": 10.0, "fn_beta": 20.0},
        },
    )
    generate(model, definitions=definitions)
    assert "fn_beta" in model.emissions


def test_equal_legal_scores_keep_stable_token_id_order():
    definitions = [definition(name="fn_alpha"), definition(name="fn_beta")]
    name_prefix = '{"prompt":"x","name":"'
    model = ControlledModel(
        record("x", "fn_alpha", {"value": "ok"}),
        chunks=["fn_alpha", "fn_beta"],
        rankings={name_prefix: {"fn_alpha": 9.0, "fn_beta": 9.0}},
    )
    assert model.ids["fn_alpha"] < model.ids["fn_beta"]
    generate(model, definitions=definitions)
    assert "fn_alpha" in model.emissions


def test_exact_generated_token_budget_succeeds():
    definitions = [definition({"value": "integer"})]
    target = record("x", "fn_runtime", {"value": 1})
    baseline = ControlledModel(target, chunks=['":1}}'])
    _, output_ids, _ = generate(baseline, definitions=definitions)
    model = ControlledModel(target, chunks=['":1}}'])
    _, limited_ids, _ = generate(model, definitions=definitions, limit=len(output_ids))
    assert len(limited_ids) == len(output_ids)


def test_copying_that_fills_budget_cannot_trigger_an_extra_inference_call():
    definitions = [definition({"value": "integer"})]
    target = record("x", "fn_runtime", {"value": 1})
    baseline = ControlledModel(target, chunks=['":1}}'])
    _, output_ids, _ = generate(baseline, definitions=definitions)
    model = ControlledModel(target, chunks=['":1}}'])
    with pytest.raises(ValueError, match="Token limit"):
        generate(model, definitions=definitions, limit=len(output_ids) - 1)
    assert len(model.inputs) == len(baseline.inputs) - 1


@pytest.mark.parametrize("limit", [0, 1, 2, 3])
def test_exhausted_budget_never_returns_an_unfinished_record(limit):
    model = ControlledModel(record("x", "fn_runtime", {"value": "ok"}))
    with pytest.raises(ValueError, match="Token limit|before JSON was complete"):
        generate(model, limit=limit)
    assert len(model.inputs) <= limit


def test_copying_cannot_exceed_remaining_budget():
    model = ControlledModel(record("x", "fn_runtime", {"value": "ok"}), chunks=['{"'])
    with pytest.raises(ValueError, match="Token limit reached while copying"):
        generate(model, limit=2)
    assert len(model.inputs) == 1


def test_non_roundtripping_copied_text_fails_explicitly():
    class BadCopyDecoder(ControlledModel):
        def wrapper_decode(self, token_ids):
            decoded = super().wrapper_decode(token_ids)
            return "changed" if decoded == "prompt" else decoded

    model = BadCopyDecoder(record("x", "fn_runtime", {"value": "ok"}), chunks=['{"'])
    with pytest.raises(ValueError, match="Forced text failed to round-trip"):
        generate(model)
    assert len(model.inputs) == 1


def test_no_legal_candidate_fails_explicitly():
    class NoLegalModel(ControlledModel):
        def wrapper_get_logits_from_input_ids(self, token_ids):
            self.inputs.append(token_ids.copy())
            scores = [float("-inf")] * len(self.vocab)
            scores[self.ids["[]"]] = 10.0
            scores[self.ids["{}"]] = 9.0
            return scores

    model = NoLegalModel("", chunks=["[]", "{}"])
    with pytest.raises(ValueError, match="No valid next token"):
        generate(model)
    assert len(model.inputs) == 1


def test_each_generation_uses_fresh_json_and_schema_state():
    first = ControlledModel(record("first", "fn_runtime", {"value": "one"}))
    second = ControlledModel(record("second", "fn_runtime", {"value": "two"}))
    _, _, first_schema = generate(first, "first")
    _, _, second_schema = generate(second, "second")
    assert first_schema is not second_schema
    assert first_schema.og_prompt == "first"
    assert second_schema.og_prompt == "second"
    assert first_schema.completed_parameters == second_schema.completed_parameters == {"value"}


def test_vocab_uses_sdk_decoding_for_added_tokens_and_masks_removed_tokens(tmp_path):
    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps({"text": 0}), encoding="utf-8")
    config = {
        "added_tokens_decoder": {
            "1": {"content": "<|im_end|>"},
            "2": {"content": "<|fim_prefix|>"},
        }
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")

    class DecoderWrapper:
        def wrapper_get_path_to_vocab_file(self):
            return str(vocab_file)

        def wrapper_decode(self, token_ids):
            return "".join({0: "text", 1: "", 2: ""}[token_id] for token_id in token_ids)

    vocab = build_vocab_map(DecoderWrapper())
    machine, schema = context(outer_prefix() + '{"value":"')
    definitions = [definition()]
    scores = mask_invalid_tokens(
        [5.0, 10.0, 9.0], machine, vocab, schema, ["fn_runtime"], definitions,
    )
    assert scores == [5.0, float("-inf"), float("-inf")]


@pytest.mark.parametrize("vocab", [{}, {0: ""}])
def test_missing_or_empty_decoded_token_is_rejected(vocab):
    definitions = [definition()]
    machine, schema = context("")
    result = mask_invalid_tokens([1.0], machine, vocab, schema, ["fn_runtime"], definitions)
    assert result == [float("-inf")]

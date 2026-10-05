"""Generation regressions adapted to Chapter 3.2's complete outer record.

Controlled scores exercise real masking and generation; no model is loaded.
"""

import json

import pytest

import src.generation as generation
from src.json_state import JasonStateMacheine, JasonStates
from src.schema import InputDefinition
from src.schema_state import SchemaState


DEFINITIONS = [
    InputDefinition(
        name=name, description="Runtime function.",
        parameters={"value": {"type": "string"}},
        returns={"type": "string"},
    )
    for name in ["fn_alpha", "fn_beta"]
]
RECORD = '{"prompt":"x","name":"fn_alpha","parameters":{"value":"ok"}}'


class ScriptedModel:
    """Scores determine preferences; production masking determines eligibility."""

    def __init__(self, vocab, steps):
        self.vocab = vocab
        self.steps = steps
        self.inputs = []
        self.prompt_ids = [900000, 900001]

    def wrapper_encode(self, text):
        return self.prompt_ids.copy()

    def wrapper_get_logits_from_input_ids(self, token_ids):
        step = len(self.inputs)
        assert step < len(self.steps), "Generation requested an extra token"
        self.inputs.append(token_ids.copy())
        scores = [float("-inf")] * (max(self.vocab) + 1)
        for token_id, score in self.steps[step].items():
            scores[token_id] = score
        return scores


def run_generation(model, limit=20, original_request="x", definitions=None, schema=None):
    if definitions is None:
        definitions = DEFINITIONS
    if schema is None:
        schema = SchemaState(original_request)
    return generation.generate_constrained(
        "separate model instructions", limit, model, model.vocab, schema,
        [definition.name for definition in definitions], definitions,
    )


def generated_text(model, result):
    return "".join(model.vocab[token_id] for token_id in result[len(model.prompt_ids):])


def test_masks_invalid_high_scores_and_preserves_legal_ranking():
    beta_record = RECORD.replace("fn_alpha", "fn_beta")
    model = ScriptedModel(
        {0: "{}", 1: '{"prompt":"x"}', 2: RECORD, 3: beta_record},
        [{0: 40.0, 1: 30.0, 2: 10.0, 3: 20.0}],
    )
    result = run_generation(model)
    assert result == model.prompt_ids + [3]
    assert json.loads(generated_text(model, result))["name"] == "fn_beta"
    assert len(model.inputs) == 1


def test_multi_character_tokens_advance_states_and_feed_next_model_call():
    vocab = {
        0: '{"prompt":"', 1: "x", 2: '","name":"fn_',
        3: 'alpha","parameters":{', 4: '"value"', 5: ':"ok"}}',
    }
    model = ScriptedModel(vocab, [{index: 5.0} for index in vocab])
    schema = SchemaState("x")
    result = run_generation(model, schema=schema)
    assert generated_text(model, result) == RECORD
    assert json.loads(generated_text(model, result)) == {
        "prompt": "x", "name": "fn_alpha", "parameters": {"value": "ok"},
    }
    assert result == model.prompt_ids + list(vocab)
    assert model.inputs == [model.prompt_ids + list(range(index)) for index in vocab]
    assert schema.function_name == "fn_alpha"
    assert schema.has_complete_outer_fields()


def test_complete_record_at_exact_generated_token_limit_succeeds():
    model = ScriptedModel({0: RECORD}, [{0: 5.0}])
    result = run_generation(model, limit=1)
    assert generated_text(model, result) == RECORD
    assert len(model.inputs) == 1


@pytest.mark.parametrize("limit", [1, 2])
def test_unfinished_record_at_generated_token_limit_raises(limit):
    model = ScriptedModel(
        {0: '{"prompt":"', 1: "x"}, [{0: 5.0}, {1: 5.0}],
    )
    with pytest.raises(ValueError, match="before JSON was complete") as error:
        run_generation(model, limit=limit)
    assert f"generated_tokens={limit}" in str(error.value)
    assert len(model.inputs) == limit


def test_no_legal_candidate_raises_before_selection():
    model = ScriptedModel({0: ":", 1: "garbage", 2: "{}"}, [{0: 20.0, 1: 10.0, 2: 5.0}])
    with pytest.raises(ValueError, match="No valid next token"):
        run_generation(model)
    assert model.inputs == [model.prompt_ids]


def test_each_generation_creates_fresh_json_state(monkeypatch):
    machines = []

    def tracked_machine():
        machine = JasonStateMacheine()
        machines.append(machine)
        return machine

    monkeypatch.setattr(generation, "JasonStateMacheine", tracked_machine)
    for _ in range(2):
        model = ScriptedModel({0: RECORD}, [{0: 5.0}])
        assert generated_text(model, run_generation(model)) == RECORD
    assert len(machines) == 2
    assert machines[0] is not machines[1]
    assert all(machine.state == JasonStates.DONE for machine in machines)


def test_current_eos_stop_cannot_return_unfinished_record():
    eos_id = 151645
    model = ScriptedModel(
        {0: '{"prompt":"', eos_id: "<|im_end|>"},
        [{0: 5.0}, {eos_id: 5.0}],
    )
    with pytest.raises(ValueError, match="before JSON was complete"):
        run_generation(model, original_request="<|im_end|>")
    assert len(model.inputs) == 2


def test_generation_hands_selected_parameters_to_characters_in_same_token():
    definitions = [
        InputDefinition(
            name="fn_old", description="Other runtime function.",
            parameters={"count": {"type": "integer"}}, returns={"type": "number"},
        ),
        InputDefinition(
            name="fn_selected", description="Chosen runtime function.",
            parameters={"text": {"type": "string"}}, returns={"type": "string"},
        ),
    ]

    class ObservedSchema(SchemaState):
        def __init__(self):
            super().__init__("x")
            self.observations = []

        def update(self, char, machine, names, parameters):
            # Tuple reassignment keeps the shallow candidate copy's log private.
            self.observations = self.observations + [
                (char, self.function_name, tuple(parameters))
            ]
            super().update(char, machine, names, parameters)

    vocab = {
        0: '{"prompt":"x","name":"fn_sel',
        1: 'ected","parameters":{"text":"ok"}}',
    }
    schema = ObservedSchema()
    model = ScriptedModel(vocab, [{0: 5.0}, {1: 5.0}])
    result = run_generation(model, definitions=definitions, schema=schema)
    assert json.loads(generated_text(model, result)) == {
        "prompt": "x", "name": "fn_selected", "parameters": {"text": "ok"},
    }
    after_selection = [entry for entry in schema.observations if entry[1] == "fn_selected"]
    assert after_selection
    assert all(entry[2] == ("text",) for entry in after_selection)
    assert schema.current_parameter == "text"


def test_generation_uses_changed_runtime_names_and_raw_original_request():
    request = 'مرحبا "حياة"\n'
    definitions = [
        InputDefinition(
            name="fn_new", description="New runtime function.",
            parameters={"value": {"type": "string"}}, returns={"type": "string"},
        )
    ]
    text = json.dumps(
        {"prompt": request, "name": "fn_new", "parameters": {"value": "ok"}},
        ensure_ascii=False, separators=(",", ":"),
    )
    model = ScriptedModel({0: RECORD, 1: text}, [{0: 20.0, 1: 10.0}])
    result = run_generation(model, original_request=request, definitions=definitions)
    assert json.loads(generated_text(model, result))["prompt"] == request
    assert json.loads(generated_text(model, result))["name"] == "fn_new"


def test_missing_runtime_names_fail_explicitly_at_name_boundary():
    model = ScriptedModel(
        {0: '{"prompt":"x","name":', 1: '"fn_alpha"'},
        [{0: 5.0}, {1: 5.0}],
    )
    with pytest.raises(ValueError, match="No valid next token"):
        run_generation(model, definitions=[])
    assert len(model.inputs) == 2

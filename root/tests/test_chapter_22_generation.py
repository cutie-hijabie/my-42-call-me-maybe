"""Exercise the real generation loop and masking without loading the LLM."""

import json

import pytest

import src.generation as generation
from src.json_state import JasonStateMacheine, JasonStates
from src.schema_state import SchemaState


class ScriptedModel:
    """Return controlled scores; real project code decides token eligibility."""

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


def run_generation(model, limit=20, original_request="x"):
    return generation.generate_constrained(
        "model context", limit, model, model.vocab,
        SchemaState(original_request), [], [],
    )


def generated_text(model, result):
    return "".join(
        model.vocab[token_id] for token_id in result[len(model.prompt_ids):]
    )


def test_masks_invalid_high_score_and_preserves_legal_ranking():
    model = ScriptedModel(
        {0: "{}:", 1: "{}", 2: '{"'},
        [{0: 30.0, 1: 20.0, 2: 10.0}],
    )
    result = run_generation(model)
    assert result == model.prompt_ids + [1]
    assert json.loads(generated_text(model, result)) == {}
    assert len(model.inputs) == 1


def test_multi_character_tokens_advance_state_and_feed_next_model_call():
    model = ScriptedModel(
        {0: '{"prompt":"', 1: "x", 2: '"', 3: "}"},
        [{0: 5.0}, {1: 5.0}, {2: 5.0}, {3: 5.0}],
    )
    result = run_generation(model)
    assert json.loads(generated_text(model, result)) == {"prompt": "x"}
    assert result == model.prompt_ids + [0, 1, 2, 3]
    assert model.inputs == [
        model.prompt_ids,
        model.prompt_ids + [0],
        model.prompt_ids + [0, 1],
        model.prompt_ids + [0, 1, 2],
    ]


def test_complete_json_at_exact_token_limit_succeeds():
    model = ScriptedModel({0: "{}"}, [{0: 5.0}])
    result = run_generation(model, limit=1)
    assert generated_text(model, result) == "{}"
    assert len(model.inputs) == 1


@pytest.mark.parametrize("limit", [1, 2])
def test_unfinished_json_at_token_limit_raises(limit):
    model = ScriptedModel(
        {0: '{"prompt":"', 1: "x"}, [{0: 5.0}, {1: 5.0}],
    )
    with pytest.raises(ValueError, match="before JSON was complete") as error:
        run_generation(model, limit=limit)
    assert f"generated_tokens={limit}" in str(error.value)
    assert len(model.inputs) == limit


def test_no_legal_candidate_raises_before_selecting_a_token():
    model = ScriptedModel(
        {0: ":", 1: "garbage"}, [{0: 20.0, 1: 10.0}],
    )
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
        model = ScriptedModel({0: "{}"}, [{0: 5.0}])
        assert generated_text(model, run_generation(model)) == "{}"
    assert len(machines) == 2
    assert machines[0] is not machines[1]
    assert all(machine.state == JasonStates.DONE for machine in machines)


def test_current_eos_stop_cannot_return_unfinished_json():
    # Exercise the EOS ID currently used by generation.py, inside a string.
    eos_id = 151645
    model = ScriptedModel(
        {0: '{"prompt":"', eos_id: "<|im_end|>"},
        [{0: 5.0}, {eos_id: 5.0}],
    )
    with pytest.raises(ValueError, match="before JSON was complete"):
        run_generation(model, original_request="<|im_end|>")
    assert len(model.inputs) == 2

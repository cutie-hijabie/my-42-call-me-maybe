import json

import src.generation as generation
from src.llm_wrapper import LLMWrapper
from src.vocab import build_vocab_map


class FakeLLMWrapper:
    def __init__(self):
        self.logits_calls = 0

    def wrapper_encode(self, text):
        return [99]

    def wrapper_get_logits_from_input_ids(self, input_ids):
        self.logits_calls += 1

        return [
            10.0,  # '"hello"'
            20.0,  # ":"
            30.0,  # "garbage"
        ]


def test_generation_uses_constrained_decoding():
    fake_model = FakeLLMWrapper()

    vocab_map = {
        0: '"hello"',
        1: ":",
        2: "garbage",
    }

    result = generation.generate_constrained(
        "give me json",
        max_length=20,
        model=fake_model,
        vocab_map=vocab_map,
    )

    assert result == [99, 0]
    assert fake_model.logits_calls == 1


def test_20_real_generations_produce_parseable_json():
    model = LLMWrapper()
    vocab_map = build_vocab_map(model)

    prompts = [
        "Return only a JSON object with a name and age.",
        "Return only a JSON array containing three numbers.",
        "Return only JSON representing a boolean value.",
        "Return only a JSON object with a city and country.",
    ]

    for run_number in range(20):
        prompt = prompts[run_number % len(prompts)]

        prompt_ids = model.wrapper_encode(prompt)

        result_ids = generation.generate_constrained(
            prompt,
            max_length=len(prompt_ids) + 100,
            model=model,
            vocab_map=vocab_map,
        )

        generated_ids = result_ids[len(prompt_ids) :]

        assert generated_ids, (
            f"Generation {run_number + 1} produced no new tokens"
        )

        generated_text = model.wrapper_decode(generated_ids)

        try:
            json.loads(generated_text)
        except json.JSONDecodeError as error:
            raise AssertionError(
                f"\nGeneration {run_number + 1} produced invalid JSON."
                f"\nPrompt: {prompt}"
                f"\nGenerated text: {generated_text!r}"
                f"\nError: {error}"
            ) from error

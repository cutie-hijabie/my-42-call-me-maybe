import json

from .generation import generate_constrained
from .schema import InputFileDefinition
from .llm_wrapper import LLMWrapper
from .vocab import build_vocab_map
from .prompt import build_prompt
from .schema_state import SchemaState, SchemaStage


def main() -> None:

    with open("data/input/function_calling_tests.json", "r") as file:
        prompt_data = json.load(file)

    with open("data/input/functions_definition.json", "r") as file:
        definition_data = json.load(file)

    validated_data = InputFileDefinition.model_validate(definition_data)
    definitions = validated_data.root

    llm_model = LLMWrapper()
    vocab = build_vocab_map(llm_model)

    function_names = [definition.name for definition in definitions]

    for line in prompt_data:
        og_prompt = line["prompt"]
        model_prompt = build_prompt(og_prompt, definitions)

        prompt_schema = SchemaState(og_prompt)

        gerarated = generate_constrained(
            model_prompt,
            500,
            llm_model,
            vocab,
            prompt_schema,
            function_names,
            definitions,
        )

from typing import List
import json
from .schema import InputDefinition


def build_prompt(
    user_prompt: str,
    definitions: List[InputDefinition],
    no_match_marker: str | None = None,
) -> str:
    prompt = (
        "/no_think\n"
        "Choose the listed function that fulfills the request using its "
        "description and the request's meaning. "
        "Allow synonyms and different wording.\n"
        "Extract every required argument from the request. "
        "Copy explicitly supplied string values exactly.\n"
        'Return only compact JSON with keys "prompt", "name", '
        '"parameters" in that order. Preserve the original request '
        "in prompt. Use no whitespace outside strings.\n"
    )

    if no_match_marker is not None:
        prompt += (
            "If no listed function fulfills the request, set name to "
            f"{json.dumps(no_match_marker, ensure_ascii=False)}. "
            "Use this option only when none of the listed functions applies.\n"
        )
    functions = "functions: \n"
    for definition in definitions:
        parameters = []
        functions += f"{definition.name}("
        for (
            parameter_name,
            parameter_definition,
        ) in definition.parameters.items():
            parameters.append(f"{parameter_name}:{parameter_definition.type}")

        joined_parameters = ", ".join(parameters)

        functions += f"{joined_parameters}): {definition.description}\n"

    prompt += (
        functions
        + f"\nRequest: {json.dumps(user_prompt, ensure_ascii=False)}"
        + "\nJSON:"
    )

    return prompt

from typing import List

from .schema import InputDefinition


def build_prompt(
    user_prompt: str,
    definitions: List[InputDefinition],
) -> str:
    prompt = (
        "Select the function that best matches the user request "
        "and fill its parameters using the request.\n"
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

        functions += joined_parameters + ")\n"

    prompt += functions + f"\nRequest: {user_prompt}"

    return prompt

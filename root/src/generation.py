from typing import Dict, List
from .masking import mask_invalid_tokens
from .json_state import JasonStateMacheine
from .llm_wrapper import LLMWrapper
from .schema_state import SchemaState
from .schema import InputDefinition


def generate_constrained(
    propmt: str,
    max_length: int,
    model: LLMWrapper,
    vocab_map: Dict,
    schema_state: SchemaState,
    function_names: List[str],
    definitions: List[InputDefinition],
) -> List[int]:

    jason = JasonStateMacheine()
    token_ids = model.wrapper_encode(propmt)
    input_length = len(token_ids)

    done = False

    while not done:
        logits = model.wrapper_get_logits_from_input_ids(token_ids)
        constarined_logits = mask_invalid_tokens(
            logits, jason, vocab_map, schema_state, function_names, definitions
        )
        if all(score == float("-inf") for score in constarined_logits):
            raise ValueError(
                "No valid next token: "
                f"JSON state={jason.state.name}, "
                f"key prefix={schema_state.partial_key_candidate!r}"
            )
        next_token = max(
            range(len(constarined_logits)), key=lambda i: constarined_logits[i]
        )
        token_string = vocab_map[next_token]
        print(repr(token_string), end=" ", flush=True)
        parameters = schema_state.get_parameters(definitions)
        for char in token_string:
            jason.transition(char)
            schema_state.update(char, jason, function_names, parameters)

        token_ids.append(next_token)

        print(
            "tokens:",
            len(token_ids),
            "json complete:",
            jason.is_complete(),
        )

        if (
            next_token == 151645
            or len(token_ids) - input_length >= max_length
            or jason.is_complete()
        ):
            done = True

    return token_ids

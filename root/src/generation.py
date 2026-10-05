from typing import Dict, List
from .masking import mask_invalid_tokens
from .json_state import JasonStateMacheine
from .llm_wrapper import LLMWrapper
from .schema_state import SchemaState
from .schema import InputDefinition
import numpy as np
from time import perf_counter
from .prompt import build_prompt


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
    selected_prompt_ids: List[int] | None = None
    model_seconds = 0.0
    selection_seconds = 0.0
    checked_candidates = 0

    done = False

    while not done:
        if (
            schema_state.function_name is not None
            and selected_prompt_ids is None
        ):
            selected_definitions = [
                definition
                for definition in definitions
                if definition.name == schema_state.function_name
            ]

            selected_prompt_ids = model.wrapper_encode(
                build_prompt(
                    schema_state.og_prompt,
                    selected_definitions,
                )
            )

            print(
                f"Prompt tokens: {input_length} -> "
                f"{len(selected_prompt_ids)} after function selection"
            )

        model_input_ids = (
            token_ids
            if selected_prompt_ids is None
            else selected_prompt_ids + token_ids[input_length:]
        )
        started = perf_counter()
        logits = model.wrapper_get_logits_from_input_ids(model_input_ids)
        model_seconds += perf_counter() - started

        started = perf_counter()
        next_token = None
        candidate_order = np.argsort(-np.asarray(logits), kind="stable")

        for candidate_id in candidate_order:
            token_id = int(candidate_id)

            if logits[token_id] == float("-inf"):
                break

            checked_candidates += 1
            checked = mask_invalid_tokens(
                [logits[token_id]],
                jason,
                {0: vocab_map.get(token_id, "")},
                schema_state,
                function_names,
                definitions,
            )

            if checked[0] != float("-inf"):
                next_token = token_id
                break

        selection_seconds += perf_counter() - started

        if next_token is None:
            raise ValueError(
                "No valid next token: "
                f"JSON state={jason.state.name}, "
                f"key prefix={schema_state.partial_key_candidate!r}"
            )
        token_string = vocab_map[next_token]
        for char in token_string:
            parameters = schema_state.get_parameters(definitions)
            jason.transition(char)
            schema_state.update(char, jason, function_names, parameters)

        token_ids.append(next_token)
        if (
            next_token == 151645
            or len(token_ids) - input_length >= max_length
            or jason.is_complete()
        ):
            done = True

    if not jason.is_complete():
        raise ValueError(
            "Generation stopped before JSON was complete: "
            f"state={jason.state.name}, "
            f"generated_tokens={len(token_ids) - input_length}"
        )

    print(
        f"Profile: tokens={len(token_ids) - input_length}, "
        f"model={model_seconds:.2f}s, "
        f"selection={selection_seconds:.2f}s, "
        f"candidates checked={checked_candidates}"
    )

    return token_ids

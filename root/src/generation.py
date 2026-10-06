from typing import Dict, List
from .masking import mask_invalid_tokens
from .json_state import JasonStateMacheine
from .llm_wrapper import LLMWrapper
from .schema_state import SchemaState
from .schema import InputDefinition
import numpy as np
from .prompt import build_prompt


def generate_constrained(
    propmt: str,
    max_length: int,
    model: LLMWrapper,
    vocab_map: Dict,
    schema_state: SchemaState,
    function_names: List[str],
    definitions: List[InputDefinition],
    no_match_marker: str | None = None,
) -> List[int]:

    candidate_names = function_names.copy()

    if no_match_marker is not None:
        if no_match_marker in candidate_names:
            raise ValueError("No-match marker conflicts with a function name.")
        candidate_names.append(no_match_marker)

    jason = JasonStateMacheine()
    token_ids = model.wrapper_encode(propmt)
    input_length = len(token_ids)
    selected_prompt_ids: List[int] | None = None
    done = False

    while not done:
        if len(token_ids) - input_length >= max_length:
            raise ValueError("Token limit reached before JSON was complete")
        forced_text = schema_state.get_forced_text(jason, definitions)

        if forced_text:
            forced_ids = model.wrapper_encode(forced_text)

            if model.wrapper_decode(forced_ids) != forced_text:
                raise ValueError("Forced text failed to round-trip")

            generated_length = len(token_ids) - input_length
            if generated_length + len(forced_ids) > max_length:
                raise ValueError(
                    "Token limit reached while copying known text"
                )

            for char in forced_text:
                parameters = schema_state.get_parameters(definitions)

                if not jason.transition(char):
                    raise ValueError("Invalid character in forced text")

                schema_state.update(char, jason, candidate_names, parameters)

            token_ids.extend(forced_ids)
            continue
        possible_names = (
            [schema_state.function_name]
            if schema_state.function_name is not None
            else schema_state.get_possible_function_names(candidate_names)
        )

        if (
            selected_prompt_ids is None
            and len(possible_names) == 1
            and possible_names[0] != no_match_marker
        ):
            selected_definitions = [
                definition
                for definition in definitions
                if definition.name == possible_names[0]
            ]

            selected_prompt_ids = model.wrapper_encode(
                build_prompt(
                    schema_state.og_prompt,
                    selected_definitions,
                )
            )

        model_input_ids = (
            token_ids
            if selected_prompt_ids is None
            else selected_prompt_ids + token_ids[input_length:]
        )
        logits = model.wrapper_get_logits_from_input_ids(model_input_ids)

        next_token = None
        candidate_order = np.argsort(-np.asarray(logits), kind="stable")

        for candidate_id in candidate_order:
            token_id = int(candidate_id)

            if logits[token_id] == float("-inf"):
                break

            checked = mask_invalid_tokens(
                [logits[token_id]],
                jason,
                {0: vocab_map.get(token_id, "")},
                schema_state,
                candidate_names,
                definitions,
            )

            if checked[0] != float("-inf"):
                next_token = token_id
                break

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
            schema_state.update(char, jason, candidate_names, parameters)

            if (
                no_match_marker is not None
                and schema_state.function_name == no_match_marker
            ):
                raise ValueError("No available function matches this request.")

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

    return token_ids

from typing import Dict, List
from .masking import mask_invalid_tokens
from .json_state import JasonStateMacheine
from .llm_wrapper import LLMWrapper


def generate_constrained(
    propmt: str, max_length: int, model: LLMWrapper, vocab_map: Dict
) -> List[int]:

    jason = JasonStateMacheine()
    token_ids = model.wrapper_encode(propmt)

    done = False

    while not done:
        logits = model.wrapper_get_logits_from_input_ids(token_ids)
        constarined_logits = mask_invalid_tokens(logits, jason, vocab_map)
        next_token = max(
            range(len(constarined_logits)), key=lambda i: constarined_logits[i]
        )
        token_string = vocab_map[next_token]
        for char in token_string:
            jason.transition(char)

        token_ids.append(next_token)

        if (
            next_token == 151645
            or len(token_ids) == max_length
            or jason.is_complete()
        ):
            done = True

    return token_ids

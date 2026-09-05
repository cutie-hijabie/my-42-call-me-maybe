from typing import List, Dict
from .json_state import JasonStateMacheine


def mask_invalid_tokens(
    logits: List[float],
    json_machine: JasonStateMacheine,
    vocab_map: Dict[int, str],
) -> List[float]:

    for token_id in range(len(logits)):
        token_string = vocab_map[token_id]
        if not json_machine.validate_candidate(token_string):
            logits[token_id] = float("-inf")

    return logits

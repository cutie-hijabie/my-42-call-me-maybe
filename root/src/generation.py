from typing import List
from .masking import mask_invalid_tokens
from .vocab import build_vocab_map
from .json_state import JasonStateMacheine
from .llm_wrapper import LLMWrapper


def generate_constrained(propmt: str, max_length: int) -> List[int]:

    model = LLMWrapper()
    jason = JasonStateMacheine()
    token_ids = model.wrapper_encode(propmt)
    vocab_map = build_vocab_map(model)

    done = False

    while not done:
        logits = model.wrapper_get_logits_from_input_ids(token_ids)
        print("logits length:", len(logits))
        print("vocab map length:", len(vocab_map))
        print("highest vocab ID:", max(vocab_map.keys()))
        print(
            "missing IDs:",
            [
                token_id
                for token_id in range(len(logits))
                if token_id not in vocab_map
            ],
        )
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


if __name__ == "__main__":
    generate_constrained(
        "Return only a JSON object with a name and age.",
        max_length=100,
    )

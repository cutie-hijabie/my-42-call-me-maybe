from typing import List

from .llm_wrapper import LLMWrapper


def generate_unconstrained(propmt: str, max_length: int) -> List[int]:

    model = LLMWrapper()

    token_ids = model.wrapper_encode(propmt)

    done = False

    while not done:
        logits = model.wrapper_get_logits_from_input_ids(token_ids)
        next_token = max(range(len(logits)), key=lambda i: logits[i])
        token_ids.append(next_token)

        if next_token == 151645 or len(token_ids) == max_length:
            done = True

    return token_ids


result = generate_unconstrained("The capital of France is", max_length=100)
print(f"Generated {len(result)} tokens")
print(f"EOS token in output? {151645 in result}")

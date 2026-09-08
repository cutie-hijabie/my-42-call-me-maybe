from src.llm_wrapper import LLMWrapper


model = LLMWrapper()

token_ids = [
    151642,
    151643,
    151644,
    151645,
    151646,
    151935,
]

for token_id in token_ids:
    try:
        decoded = model.wrapper_decode([token_id])
        print(token_id, repr(decoded))
    except Exception as error:
        print(token_id, type(error).__name__, error)

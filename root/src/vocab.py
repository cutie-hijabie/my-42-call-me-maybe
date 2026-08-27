from typing_extensions import Dict
import json
from .llm_wrapper import LLMWrapper


def build_vocab_map(wrapper: LLMWrapper) -> Dict[int, str]:
    path = wrapper.wrapper_get_path_to_vocab_file()

    with open(path, "r") as file:
        vocab_data = json.load(file)

        vocab_map = {}

    for token, id in vocab_data.items():
        vocab_map[int(id)] = token

    return vocab_map

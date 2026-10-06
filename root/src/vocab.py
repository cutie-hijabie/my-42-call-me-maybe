from typing_extensions import Dict
import json
from .llm_wrapper import LLMWrapper
from pathlib import Path


def build_vocab_map(wrapper: LLMWrapper) -> Dict[int, str]:
    path = wrapper.wrapper_get_path_to_vocab_file()

    with open(path, "r") as file:
        vocab_data = json.load(file)

        vocab_map = {}

    for token, id in vocab_data.items():
        vocab_map[int(id)] = wrapper.wrapper_decode([int(id)])

    tokenizer_path = Path(path).with_name("tokenizer_config.json")
    with open(tokenizer_path, "r") as file:
        tokenizer_data = json.load(file)
        added_tokens = tokenizer_data["added_tokens_decoder"]

    for id in added_tokens:
        vocab_map[int(id)] = wrapper.wrapper_decode([int(id)])

    return vocab_map

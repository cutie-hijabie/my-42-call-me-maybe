from typing import List

from llm_sdk import Small_LLM_Model


class LLMWrapper:
    def __init__(self) -> None:
        self.llm = Small_LLM_Model()

    def wrapper_encode(self, text: str) -> List[int]:
        token_ids = self.llm.encode(text)

        return token_ids.tolist()[0]

    def wrapper_get_logits_from_input_ids(
        self, input_ids: List[int]
    ) -> List[float]:
        return self.llm.get_logits_from_input_ids(input_ids)

    def wrapper_get_path_to_vocab_file(self) -> str:
        return self.llm.get_path_to_vocab_file()

    def wrapper_decode(self, token_ids: List[int]) -> str:
        return self.llm.decode(token_ids)

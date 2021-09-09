from abc import ABC, abstractmethod
from typing import List
from transformers import AutoTokenizer


class Tokenizer(ABC):
    @abstractmethod
    def __call__(self, text: str) -> List[int]:
        ...

    @abstractmethod
    def num_tokens(text: str) -> int:
        ...

    @abstractmethod
    def truncate_by_tokens(text: str, max_tokens: int) -> str:
        ...


class HuggingFaceTokenizer(Tokenizer):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def __call__(self, text: str) -> List[int]:
        return self.tokenizer(text)["input_ids"]

    def num_tokens(self, text: str) -> int:
        return len(self.tokenizer.tokenize(text))

    def truncate_by_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens is None or not text:
            return text
        encoding = self.tokenizer(
            text, truncation=True, max_length=max_tokens, return_offsets_mapping=True
        )

        return text[: encoding.offset_mapping[-1][1]]

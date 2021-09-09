from abc import ABC, abstractmethod
from typing import List
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class Tokenizer(ABC):
    @abstractmethod
    def num_tokens(text: str) -> int:
        ...

    @abstractmethod
    def truncate_by_tokens(text: str, max_tokens: int) -> str:
        ...


class TransformersTokenizer(Tokenizer):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def __call__(self, *args, **kwargs) -> BatchEncoding:
        return self.tokenizer(*args, **kwargs)

    def num_tokens(self, text: str) -> int:
        return len(self.tokenizer.tokenize(text))

    def truncate_by_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens is None or not text:
            return text
        encoding = self.tokenizer(
            text, truncation=True, max_length=max_tokens, return_offsets_mapping=True
        )

        return text[: encoding.offset_mapping[-1][1]]

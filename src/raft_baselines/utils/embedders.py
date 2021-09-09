from abc import ABC, abstractmethod
from typing import List, Tuple
from sentence_transformers import SentenceTransformer


class Embedder(ABC):
    @abstractmethod
    def __call__(self, texts: List[str]) -> List[List[float]]:
        ...


class SentenceTransformersEmbedder(Embedder):
    def __init__(
        self, model_name="sentence-transformers/all-MiniLM-L6-v2", max_seq_length=512
    ):
        self.similarity_model = SentenceTransformer(model_name)
        self.similarity_model.max_seq_length = max_seq_length
        self._cache = {}

    def __call__(self, texts: Tuple[str]) -> List[List[float]]:
        if hash(texts) in self._cache:
            return self._cache[hash(texts)]

        embeds = self.similarity_model.encode(texts, convert_to_tensor=True)

        self._cache[hash(texts)] = embeds
        return embeds

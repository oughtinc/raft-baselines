from abc import ABC, abstractmethod
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import torch
import openai
import time

from raft_baselines.utils.tokenizers import TransformersTokenizer


class Embedder(ABC):
    @abstractmethod
    def __call__(self, texts: List[str]) -> List[List[float]]:
        ...


class SentenceTransformersEmbedder(Embedder):
    def __init__(
        self, model_name="sentence-transformers/all-MiniLM-L6-v2", max_seq_length=512
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.similarity_model = SentenceTransformer(model_name, device=self.device)
        self.similarity_model.max_seq_length = max_seq_length
        self._cache = {}

    def __call__(self, texts: Tuple[str]) -> List[List[float]]:
        if hash(texts) in self._cache:
            return self._cache[hash(texts)]

        embeds = self.similarity_model.encode(
            texts, convert_to_tensor=True, device=self.device
        )

        self._cache[hash(texts)] = embeds
        return embeds


class OpenAIEmbedder(Embedder):
    def __init__(
            self, model_name="text-embedding-ada-002", max_tokens=512
    ):
        self.similarity_model = model_name
        self.max_tokens = max_tokens
        self._cache = {}
    
    def __call__(self, texts: Tuple[str]) -> List[List[float]]:
        if hash(texts) in self._cache:
            return self._cache[hash(texts)]
        
        tokenizer = TransformersTokenizer("gpt2")
        short_enough_texts = [
            tokenizer.truncate_by_tokens(text, self.max_tokens)
            for text in texts
        ]

        success = False
        retries = 0
        while not success:
            try:
                response = openai.embeddings.create(
                    model=self.similarity_model,
                    input=short_enough_texts,
                )

                embeds = [
                    embedding_object.embedding 
                    for embedding_object in response.data
                ]                 
                success = True
            except Exception as e:
                print(f"Exception in OpenAI search: {e}")
                retries += 1
                if retries > 3:
                    raise Exception("Max retries reached")
                    break
                else:
                    print("retrying")
                    time.sleep(retries * 15)
        
            
        self._cache[hash(texts)] = embeds
        return embeds
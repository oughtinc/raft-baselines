from logging import Formatter
from typing import Dict, Optional, List, Mapping

import numpy as np
import datasets
import torch
from transformers import AutoModelForCausalLM
from sentence_transformers import util

from raft_baselines.classifiers.in_context_classifier import InContextClassifier
from raft_baselines.utils.tokenizers import TransformersTokenizer
from raft_baselines.utils.embedders import SentenceTransformersEmbedder


class TransformersCausalLMClassifier(InContextClassifier):
    def __init__(
        self,
        *args,
        model_type: str = "distilgpt2",
        **kwargs,
    ) -> None:
        tokenizer = TransformersTokenizer(model_type)
        self.model = AutoModelForCausalLM.from_pretrained(model_type)
        self.similarity_embedder = SentenceTransformersEmbedder()

        super().__init__(
            *args,
            tokenizer=tokenizer,
            max_tokens=self.model.config.max_position_embeddings,
            **kwargs,
        )

    def semantically_select_training_examples(
        self, target: Mapping[str, str]
    ) -> datasets.Dataset:
        formatted_examples_without_labels = tuple(
            self.format_dict(
                {col: row[col] for col in self.input_cols if col in row},
            )
            for row in self.training_data
        )
        formatted_target = self.format_dict(target)

        # adapted from https://towardsdatascience.com/semantic-similarity-using-transformers-8f3cb5bf66d6
        target_embedding = self.similarity_embedder(tuple([formatted_target]))
        example_embeddings = self.similarity_embedder(formatted_examples_without_labels)

        similarity_scores = util.pytorch_cos_sim(target_embedding, example_embeddings)[
            0
        ]

        sorted_indices = np.argsort(-similarity_scores)
        return self.training_data.select(
            list(reversed(sorted_indices[: self.num_prompt_training_examples]))
        )

    def _get_raw_probabilities(
        self,
        prompt: str,
        engine: Optional[str] = None,
    ) -> List[float]:
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]

        with torch.no_grad():
            output = self.model(input_ids)

        next_token_probs = torch.softmax(output.logits[0][-1], dim=0)

        def get_prob_for_class(clas):
            clas_str = (
                f" {clas}"
                if not self.add_prefixes
                else f" {self.classes.index(clas) + 1}"
            )

            return next_token_probs[self.tokenizer(clas_str)["input_ids"][0]]

        return [get_prob_for_class(clas) for clas in self.classes]

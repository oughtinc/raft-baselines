from typing import Dict, Optional, List, Mapping

import numpy as np
import datasets

from raft_baselines.classifiers.in_context_classifier import InContextClassifier
from raft_baselines.utils.gpt3_utils import (
    complete,
    search,
)
from raft_baselines.utils.tokenizers import TransformersTokenizer

GPT3_MAX_TOKENS = 2048
tokenizer = TransformersTokenizer("gpt2")


class GPT3Classifier(InContextClassifier):
    def __init__(
        self,
        *args,
        engine: str = "ada",
        search_engine: str = "ada",
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            tokenizer=tokenizer,
            max_tokens=GPT3_MAX_TOKENS,
            **kwargs,
        )

        self.engine: str = engine
        self.search_engine: str = search_engine

    def semantically_select_training_examples(
        self, target: Mapping[str, str]
    ) -> datasets.Dataset:
        formatted_examples_without_labels = tuple(
            self.format_dict(
                {col: row[col] for col in self.input_cols if col in row},
            )
            for row in self.training_data
        )

        search_results = search(
            formatted_examples_without_labels,
            self.format_dict(target),
            self.search_engine,
        )

        sorted_indices = list(
            map(
                lambda result: result["document"],  # type: ignore
                sorted(
                    search_results,
                    key=lambda result: -result["score"],  # type: ignore
                ),
            )
        )

        return self.training_data.select(
            list(reversed(sorted_indices[: self.num_prompt_training_examples]))
        )

    def does_token_match_class(self, token: str, clas: str) -> bool:
        # prepend a space to the class label
        # because we always expect a leading space in the first token
        # returned from the OpenAI API, given our prompt format
        clas_str = (
            f" {clas}" if not self.add_prefixes else f" {self.classes.index(clas) + 1}"
        )

        clas_first_token_id: int = self.tokenizer(clas_str)["input_ids"][0]
        token_id: int = self.tokenizer(token)["input_ids"][0]

        # Compare token ids rather than the raw tokens
        # because GPT2TokenizerFast represents some special characters
        # differently from the GPT-3 API
        # (e.g. the space at the beginning of the token is " " according to the API,
        # but "Ġ" according to the tokenizer.
        # Standardizing to token ids is one easy way to smooth over that difference.
        return clas_first_token_id == token_id

    def _get_raw_probabilities(
        self,
        prompt: str,
    ) -> List[float]:
        response = complete(
            prompt,
            temperature=0.0,
            engine=self.engine,
            max_tokens=1,
        )
        logprobs: Dict[str, float] = response["choices"][0]["logprobs"]["top_logprobs"][
            0
        ]

        raw_p = []
        for clas in self.classes:
            p = 0.0
            for token in logprobs.keys():
                if self.does_token_match_class(token, clas):
                    p += np.exp(logprobs[token])
            raw_p.append(p)

        return raw_p

import random
from typing import Dict, Optional, List, Tuple, Mapping, Any
from collections import defaultdict
import json

import numpy as np
import datasets

from classifier import Classifier
from utils import num_tokens, truncate_by_tokens, complete, gpt2_tokenizer, search


with open("task_data.json") as f:
    FIELD_ORDERING = json.loads(f.readline())
    INSTRUCTIONS = json.loads(f.readline())


class GPT3Classifier(Classifier):
    def __init__(
        self,
        training_data: datasets.Dataset,
        engine: str = "ada",
        num_prompt_training_examples: int = 20,
        add_prefixes: bool = False,
        config: str = None,
        use_task_specific_instructions: bool = False,
        do_semantic_selection: bool = False,
        search_engine: str = "ada",
    ) -> None:
        super().__init__(training_data)

        self.engine: str = engine
        self.num_prompt_training_examples: int = num_prompt_training_examples
        self.add_prefixes: bool = add_prefixes

        if config:
            self.config: str = config
            self.input_cols: List[str] = FIELD_ORDERING[config]
            self.instructions_start: str = "Possible labels:"
            if use_task_specific_instructions:
                self.instructions_start = INSTRUCTIONS[config] + "\n" + self.instructions_start

        self.do_semantic_selection: bool = do_semantic_selection
        self.search_engine: str = search_engine

        self.truncation_params: Mapping[str, Any] = {
            # max - buffer - completion tokens
            "max_tokens": 2048 - 10 - 1,
            "end_example_token_proportion": max(
                0.25,
                1
                / (1 + min(self.num_prompt_training_examples, len(self.training_data))),
            )
            if self.num_prompt_training_examples is not None
            else 0.25,
        }

    separator: str = "\n\n"

    @property
    def instructions(self) -> str:
        formatted_classes = "\n".join(
            [f"{idx + 1}. {clas}" for idx, clas in enumerate(self.classes)]
        )
        return f"""{self.instructions_start}\n{formatted_classes}"""

    def max_example_lengths(
        self,
        num_training_examples: int,
        input_to_classify: Mapping[str, str]
    ) -> Tuple[int, int]:
        instruction_tokens = num_tokens(self.instructions)
        separator_tokens = (num_training_examples + 1) * len(self.separator)
        max_example_tokens = (
            self.truncation_params["max_tokens"] - instruction_tokens - separator_tokens
        )

        untruncated_end_example_tokens = num_tokens(
            self.format_prompt_end(input_to_classify)
        )
        max_end_example_tokens = min(
            untruncated_end_example_tokens,
            int(
                max_example_tokens
                * self.truncation_params["end_example_token_proportion"]
            ),
        )
        max_train_example_tokens = (
            int((max_example_tokens - max_end_example_tokens) / num_training_examples)
            if num_training_examples > 0
            else 0
        )

        return max_end_example_tokens, max_train_example_tokens

    @classmethod
    def format_dict(cls, example: Mapping[str, str]) -> str:
        return "\n".join([f"{k}: {v}" for k, v in example.items() if len(v.strip())])

    def format_prompt_end(
        self,
        target: Mapping[str, str],
        max_tokens: Optional[int] = None
    ) -> str:
        output_block = f"{self.class_col}:"
        output_block_tokens = num_tokens(output_block)
        untruncated_text = self.format_dict(target)
        input_block = (
            untruncated_text
            if max_tokens is None
            else truncate_by_tokens(
                untruncated_text, max_tokens - output_block_tokens - 1
            )
        )
        return f"""{input_block}
{output_block}"""

    def format_example(
        self,
        example: Mapping[str, str],
        clas: str, max_tokens: Optional[int] = None
    ) -> str:
        clas_str = (
            clas if not self.add_prefixes else f"{self.classes.index(clas) + 1}. {clas}"
        )
        output_block = f"{self.class_col}: {clas_str}"
        output_block = (
            output_block
            if max_tokens is None
            else truncate_by_tokens(output_block, max_tokens - 2)
        )
        output_block_tokens = num_tokens(output_block)
        untruncated_text = self.format_dict(example)
        input_block = (
            untruncated_text
            if max_tokens is None
            else truncate_by_tokens(
                untruncated_text, max_tokens - output_block_tokens - 1
            )
        )
        return f"""{input_block}
{output_block}"""

    def render_examples(
        self,
        example_dataset: datasets.Dataset,
        max_tokens_per_example: Optional[int] = None,
    ) -> str:
        formatted_examples = [
            self.format_example(
                {col: row[col] for col in self.input_cols if col in row},
                self.class_label_to_string(row[self.class_col]),
                max_tokens=max_tokens_per_example,
            )
            for row in example_dataset
        ]
        return self.separator.join(formatted_examples)

    def select_training_examples(
        self,
        target: Mapping[str, str],
        random_seed: Optional[int] = None
    ) -> datasets.Dataset:
        if not self.do_semantic_selection:
            random.seed(random_seed)

            n_ex = self.num_prompt_training_examples
            if n_ex is None or len(self.training_data) <= n_ex:
                return self.training_data

            uniques = defaultdict(lambda: [])
            for i, row in enumerate(self.training_data):
                uniques[row["Label"]].append(i)

            indices = []
            for key in uniques:
                indices.append(random.choice(uniques[key]))
            random.shuffle(indices)

            remaining_indices = [
                i for i in range(len(self.training_data)) if i not in indices
            ]
            indices += random.sample(
                remaining_indices, min(n_ex, len(remaining_indices))
            )

            return self.training_data.select(indices[:n_ex])
        else:
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

    def format_prompt(
        self,
        target: Mapping[str, str],
        example_dataset: Optional[datasets.Dataset] = None,
    ) -> str:
        ordered_input = {col: target[col] for col in self.input_cols if col in target}

        if example_dataset is None:
            example_dataset = self.select_training_examples(ordered_input)

        if self.truncation_params is None:
            raise ValueError("No truncation strategy provided.")
        max_end_example_tokens, max_train_example_tokens = self.max_example_lengths(
            len(example_dataset), ordered_input
        )
        example_str = self.render_examples(
            example_dataset, max_tokens_per_example=max_train_example_tokens
        )
        example_str_and_sep = "" if example_str == "" else example_str + self.separator

        prompt = f"""{self.instructions + self.separator if self.instructions != "" else ""}{example_str_and_sep}{self.format_prompt_end(ordered_input, max_tokens=max_end_example_tokens)}"""  # noqa: E501
        return prompt

    def does_token_match_class(self, token: str, clas: str) -> bool:
        # prepend a space to the class label
        # because we always expect a leading space in the first token
        # returned from the OpenAI API, given our prompt format
        clas_str = (
            f" {clas}" if not self.add_prefixes else f" {self.classes.index(clas) + 1}"
        )

        clas_first_token_id: int = gpt2_tokenizer(clas_str)["input_ids"][0]
        token_id: int = gpt2_tokenizer(token)["input_ids"][0]

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
        engine: Optional[str] = None,
    ) -> List[float]:
        response = complete(
            prompt,
            temperature=0.0,
            engine=engine or self.engine,
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

    def _classify_prompt(
        self,
        prompt: str,
        engine: Optional[str] = None,
    ) -> Dict[str, float]:
        raw_p = self._get_raw_probabilities(prompt, engine)
        sum_p = np.sum(raw_p)
        if sum_p > 0:
            normalized_p = np.array(raw_p) / np.sum(raw_p)
        else:
            normalized_p = np.full(len(self.classes), 1 / len(self.classes))
        class_probs = {}
        for i, clas in enumerate(self.classes):
            class_probs[clas] = normalized_p[i]
        return class_probs

    def classify(
        self,
        target: Mapping[str, str],
        random_seed: Optional[int] = None,
    ) -> Dict[str, float]:
        example_dataset = self.select_training_examples(target, random_seed=random_seed)
        prompt = self.format_prompt(target, example_dataset)
        return self._classify_prompt(prompt)

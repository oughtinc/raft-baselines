from abc import abstractmethod
import random
from typing import Dict, Optional, List, Tuple, Mapping, Any
from collections import defaultdict
import json
import importlib.resources

import numpy as np
import datasets

from raft_baselines.classifiers.classifier import Classifier
from raft_baselines import data
from raft_baselines.utils.tokenizers import Tokenizer

text_data = importlib.resources.read_text(
    data, "prompt_construction_settings.jsonl"
).split("\n")
FIELD_ORDERING = json.loads(text_data[0])
INSTRUCTIONS = json.loads(text_data[1])


class InContextClassifier(Classifier):
    separator: str = "\n\n"

    def __init__(
        self,
        training_data: datasets.Dataset,
        num_prompt_training_examples: int = 20,
        add_prefixes: bool = False,
        config: str = None,
        use_task_specific_instructions: bool = True,
        do_semantic_selection: bool = True,
        tokenizer: Tokenizer = None,
        max_tokens: int = 2048,
    ) -> None:
        super().__init__(training_data)

        self.num_prompt_training_examples: int = num_prompt_training_examples
        self.add_prefixes: bool = add_prefixes

        if config:
            self.config: str = config
            self.input_cols: List[str] = FIELD_ORDERING[config]
            self.instructions_start: str = "Possible labels:"
            if use_task_specific_instructions:
                self.instructions_start = (
                    INSTRUCTIONS[config] + "\n" + self.instructions_start
                )

        self.do_semantic_selection: bool = do_semantic_selection

        self.tokenizer = tokenizer
        self.truncation_params: Mapping[str, Any] = {
            # max - buffer - completion tokens
            "max_tokens": max_tokens - 10 - 1,
            "end_example_token_proportion": max(
                0.25,
                1
                / (1 + min(self.num_prompt_training_examples, len(self.training_data))),
            )
            if self.num_prompt_training_examples is not None
            else 0.25,
        }

    @property
    def instructions(self) -> str:
        formatted_classes = "\n".join(
            [f"{idx + 1}. {clas}" for idx, clas in enumerate(self.classes)]
        )
        return f"""{self.instructions_start}\n{formatted_classes}"""

    def max_example_lengths(
        self, num_training_examples: int, input_to_classify: Mapping[str, str]
    ) -> Tuple[int, int]:
        instruction_tokens = self.tokenizer.num_tokens(self.instructions)
        separator_tokens = (num_training_examples + 1) * len(self.separator)
        max_example_tokens = (
            self.truncation_params["max_tokens"] - instruction_tokens - separator_tokens
        )

        untruncated_end_example_tokens = self.tokenizer.num_tokens(
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
        return "\n".join(
            [f"{k}: {v}" for k, v in example.items() if len(str(v).strip())]
        )

    def format_prompt_end(
        self, target: Mapping[str, str], max_tokens: Optional[int] = None
    ) -> str:
        output_block = f"{self.class_col}:"
        output_block_tokens = self.tokenizer.num_tokens(output_block)
        untruncated_text = self.format_dict(target)
        input_block = (
            untruncated_text
            if max_tokens is None
            else self.tokenizer.truncate_by_tokens(
                untruncated_text, max_tokens - output_block_tokens - 1
            )
        )
        return f"""{input_block}
{output_block}"""

    def format_example(
        self, example: Mapping[str, str], clas: str, max_tokens: Optional[int] = None
    ) -> str:
        clas_str = (
            clas if not self.add_prefixes else f"{self.classes.index(clas) + 1}. {clas}"
        )
        output_block = f"{self.class_col}: {clas_str}"
        output_block = (
            output_block
            if max_tokens is None
            else self.tokenizer.truncate_by_tokens(output_block, max_tokens - 2)
        )
        output_block_tokens = self.tokenizer.num_tokens(output_block)
        untruncated_text = self.format_dict(example)
        input_block = (
            untruncated_text
            if max_tokens is None
            else self.tokenizer.truncate_by_tokens(
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

    @abstractmethod
    def semantically_select_training_examples(
        self, target: Mapping[str, str]
    ) -> datasets.Dataset:
        ...

    def select_training_examples(
        self, target: Mapping[str, str], random_seed: Optional[int] = None
    ) -> datasets.Dataset:
        # handle edge case where target is blank (all the fields we selected are empty)
        if not self.do_semantic_selection or not self.format_dict(target):
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
            return self.semantically_select_training_examples(target)

    def format_prompt(
        self,
        target: Mapping[str, str],
        example_dataset: Optional[datasets.Dataset] = None,
    ) -> str:
        if self.truncation_params is None:
            raise ValueError("No truncation strategy provided.")

        num_examples = len(example_dataset) if example_dataset else 0
        max_end_example_tokens, max_train_example_tokens = self.max_example_lengths(
            num_examples, target
        )
        example_str = (
            self.render_examples(
                example_dataset, max_tokens_per_example=max_train_example_tokens
            )
            if example_dataset
            else ""
        )
        example_str_and_sep = "" if example_str == "" else example_str + self.separator

        prompt = f"""{self.instructions + self.separator if self.instructions != "" else ""}{example_str_and_sep}{self.format_prompt_end(target, max_tokens=max_end_example_tokens)}"""  # noqa: E501
        return prompt

    @abstractmethod
    def _get_raw_probabilities(
        self,
        prompt: str,
    ) -> List[float]:
        ...

    def _classify_prompt(
        self,
        prompt: str,
    ) -> Dict[str, float]:
        raw_p = self._get_raw_probabilities(prompt)
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
        should_print_prompt: bool = False,
    ) -> Dict[str, float]:
        ordered_target = {col: target[col] for col in self.input_cols if col in target}

        example_dataset = (
            self.select_training_examples(ordered_target, random_seed=random_seed)
            if self.num_prompt_training_examples > 0
            else None
        )

        prompt = self.format_prompt(ordered_target, example_dataset)
        if should_print_prompt:
            print(prompt)

        return self._classify_prompt(prompt)

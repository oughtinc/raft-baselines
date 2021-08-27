from typing import Dict, Optional, List, Tuple, Mapping

import numpy as np
import datasets

from utils import num_tokens, truncate_by_tokens, complete, gpt2_tokenizer


class GPT3Classifier:
    separator: str = "\n\n"

    def __init__(self, training_data, engine="ada", num_prompt_training_examples=20) -> None:
        self.training_data = training_data
        self.engine = engine
        self.num_prompt_training_examples=num_prompt_training_examples

        self.input_cols = [col for col in training_data.features
                           if col not in ('ID', 'Label')]
        self.class_col = 'Label'
        # Function
        self.class_label_to_string = training_data.features['Label'].int2str
        self.classes = list(training_data.features['Label'].names[1:])
        self.truncation_params = {
            # max - completion tokens
            "max_tokens": 2048 - 1,
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
        return f"""Classify the following as one of:{self.separator}{formatted_classes}"""

    def max_example_lengths(self, num_training_examples: int) -> Tuple[int, int]:
        instruction_tokens = num_tokens(self.instructions)
        separator_tokens = (num_training_examples + 1) * len(self.separator)
        max_example_tokens = (
            self.truncation_params["max_tokens"] - instruction_tokens - separator_tokens
        )

        max_end_example_tokens = int(
            max_example_tokens * self.truncation_params["end_example_token_proportion"]
        )
        max_train_example_tokens = (
            int((max_example_tokens - max_end_example_tokens) / num_training_examples)
            if num_training_examples > 0
            else 0
        )

        return max_end_example_tokens, max_train_example_tokens

    @classmethod
    def format_dict(cls, input: Mapping[str, str]) -> str:
        return "\n".join([f"{k}: {v}" for k, v in input.items()])

    def format_prompt_end(
        self, input: Mapping[str, str], max_tokens: Optional[int] = None
    ) -> str:
        output_block = f"{self.class_col}:"
        output_block_tokens = num_tokens(output_block)
        untruncated_text = self.format_dict(input)
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
        self, input: Mapping[str, str], clas: str, max_tokens: Optional[int] = None
    ) -> str:
        output_block = f"{self.class_col}: {clas}"
        output_block_tokens = num_tokens(output_block)
        untruncated_text = self.format_dict(input)
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
        self, example_dataset: datasets.Dataset, max_tokens_per_example: Optional[int] = None
    ) -> str:
        formatted_examples = [
            self.format_example(
                {col: row[col] for col in self.input_cols},
                self.class_label_to_string(row[self.class_col]),
                max_tokens=max_tokens_per_example,
            )
            for row in example_dataset
        ]
        return self.separator.join(formatted_examples)

    def select_training_examples(
        self, input: Mapping[str, str], random_seed: Optional[int] = None
    ) -> datasets.Dataset:
        if self.num_prompt_training_examples is None or (
            self.num_prompt_training_examples is not None
            and len(self.training_data) <= self.num_prompt_training_examples
        ):
            return self.training_data
        return self.training_data.train_test_split(
            train_size=self.num_prompt_training_examples, seed=random_seed
        )['train']

    def format_prompt(
        self, input: Mapping[str, str], example_dataset: Optional[datasets.Dataset] = None
    ) -> str:
        if example_dataset is None:
            example_dataset = self.select_training_examples(input)

        if self.truncation_params is None:
            raise ValueError("No truncation strategy provided.")
        max_end_example_tokens, max_train_example_tokens = self.max_example_lengths(
            len(example_dataset)
        )
        example_str = self.render_examples(
            example_dataset, max_tokens_per_example=max_train_example_tokens
        )
        example_str_and_sep = (
            "" if example_str == "" else example_str + self.separator
        )

        prompt = f"""{self.instructions + self.separator if self.instructions != "" else ""}{example_str_and_sep}{self.format_prompt_end(input, max_tokens=max_end_example_tokens)}"""  # noqa: E501
        return prompt

    @staticmethod
    def does_token_match_class(token: str, clas: str) -> bool:
        # prepend a space to the class label
        # because we always expect a leading space in the first token
        # returned from the OpenAI API, given our prompt format
        clas_first_token_id: int = gpt2_tokenizer(f" {clas}")["input_ids"][0]
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
        input: Mapping[str, str],
        random_seed: Optional[int] = None,
    ) -> Dict[str, float]:
        example_dataset = self.select_training_examples(input, random_seed=random_seed)
        prompt = self.format_prompt(input, example_dataset)
        print(prompt)
        return self._classify_prompt(prompt)


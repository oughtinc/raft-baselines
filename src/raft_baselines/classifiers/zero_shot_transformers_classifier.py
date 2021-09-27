import datasets
from typing import Mapping, Dict, Optional, List
from transformers import pipeline
import importlib.resources
import json
import torch

from raft_baselines.classifiers.classifier import Classifier
from raft_baselines import data

text_data = importlib.resources.read_text(
    data, "prompt_construction_settings.jsonl"
).split("\n")
FIELD_ORDERING = json.loads(text_data[0])


class TransformersZeroShotPipelineClassifier(Classifier):
    def __init__(
        self, training_data: datasets.Dataset, config: str = None, **kwargs
    ) -> None:
        self.device = 0 if torch.cuda.is_available() else -1
        self.clf = pipeline("zero-shot-classification", device=self.device)

        if config:
            self.config: str = config
            self.input_cols: List[str] = FIELD_ORDERING[config]

        super().__init__(training_data)

    @classmethod
    def format_dict(cls, example: Mapping[str, str]) -> str:
        return "\n".join(
            [f"{k}: {v}" for k, v in example.items() if len(str(v).strip())]
        )

    def classify(
        self,
        target: Mapping[str, str],
        random_seed: Optional[int] = None,
        should_print_prompt: bool = False,
    ) -> Dict[str, float]:
        """
        :param target: Dict input with fields and natural language data within those fields.
        :return: Dict where the keys are class names and the values are probabilities.
        """
        ordered_target = {col: target[col] for col in self.input_cols if col in target}
        target_str = self.format_dict(ordered_target)

        output = self.clf(target_str, candidate_labels=self.classes)
        return {clas: score for clas, score in zip(self.classes, output["scores"])}

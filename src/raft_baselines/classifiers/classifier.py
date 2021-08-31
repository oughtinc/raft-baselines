import datasets
from typing import Callable, List, Mapping
from abc import ABC, abstractmethod


class Classifier(ABC):
    def __init__(self, training_data: datasets.Dataset) -> None:
        self.training_data: datasets.Dataset = training_data

        self.class_col: str = "Label"
        self.class_label_to_string: Callable[[int], str] = training_data.features["Label"].int2str
        self.classes: List[str] = list(training_data.features["Label"].names[1:])
        self.input_cols: List[str] = [
            col for col in training_data.features if col not in ("ID", "Label")
        ]

    @abstractmethod
    def classify(self, target: Mapping[str, str]) -> Mapping[str, float]:
        """
        :param target: Dict input with fields and natural language data within those fields.
        :return: Dict where the keys are class names and the values are probabilities.
        """
        raise NotImplementedError

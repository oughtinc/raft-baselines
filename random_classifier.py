import random
from typing import Mapping

import datasets

from classifier import Classifier


class RandomClassifier(Classifier):
    def __init__(self, training_data: datasets.Dataset, seed: int = 4) -> None:
        super().__init__(training_data)
        random.seed(seed)

    def classify(self, target: Mapping[str, str]) -> Mapping[str, float]:
        result = {c: 0.0 for c in self.classes}
        result[random.choice(self.classes)] = 1.0

        return result

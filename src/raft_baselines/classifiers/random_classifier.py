import random
from typing import Mapping

import datasets

from raft_baselines.classifiers.classifier import Classifier


class RandomClassifier(Classifier):
    def __init__(
        self, training_data: datasets.Dataset, seed: int = 4, **kwargs
    ) -> None:
        super().__init__(training_data)
        random.seed(seed)

    def classify(self, target, random_seed=None, should_print_prompt=False):
        if random_seed is not None:
            random.seed(random_seed)
        result = {c: 0.0 for c in self.classes}
        result[random.choice(self.classes)] = 1.0

        return result

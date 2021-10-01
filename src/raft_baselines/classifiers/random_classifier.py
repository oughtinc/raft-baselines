import random
from typing import Mapping, Optional

import datasets

from raft_baselines.classifiers.classifier import Classifier


class RandomClassifier(Classifier):
    def __init__(
        self, training_data: datasets.Dataset, seed: int = 4, **kwargs
    ) -> None:
        super().__init__(training_data)
        random.seed(seed)

    def classify(
        self,
        target: Mapping[str, str],
        random_seed: Optional[int] = None,
        should_print_prompt: bool = False,
    ) -> Mapping[str, float]:
        if random_seed is not None:
            random.seed(random_seed)
        result = {c: 0.0 for c in self.classes}
        result[random.choice(self.classes)] = 1.0

        return result

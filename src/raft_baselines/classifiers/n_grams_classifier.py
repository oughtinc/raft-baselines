from typing import Mapping, Optional, Dict
from abc import abstractmethod

import datasets
from sklearn.feature_extraction.text import CountVectorizer

from raft_baselines.classifiers.classifier import Classifier


class NGramsClassifier(Classifier):
    def __init__(
        self,
        training_data: datasets.Dataset,
        vectorizer_kwargs: Dict = None,
        model_kwargs: Dict = None,
        **kwargs
    ):
        super().__init__(training_data)
        if vectorizer_kwargs is None:
            vectorizer_kwargs = {}
        cleaned_text_train = [self.stringify_row(row) for row in self.training_data]

        self.vectorizer = CountVectorizer(**vectorizer_kwargs).fit(cleaned_text_train)
        self.vectorized_training_data = self.vectorizer.transform(cleaned_text_train)
        self.classifier = None

    def stringify_row(self, row):
        return ". ".join(
            row[input_col] for input_col in self.input_cols if input_col in row
        )

    @abstractmethod
    def _classify(self, vector_input):
        return NotImplementedError

    def classify(
        self,
        target: Mapping[str, str],
        random_seed: Optional[int] = None,
        should_print_prompt: bool = False,
    ) -> Dict[str, float]:
        simple_input = self.stringify_row(target)
        vector_input = self.vectorizer.transform((simple_input,))
        result = self._classify(vector_input)
        return {
            self.class_label_to_string(int(cls)): prob
            for prob, cls in zip(result[0], self.classifier.classes_)
        }

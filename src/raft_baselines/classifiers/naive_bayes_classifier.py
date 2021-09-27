import random
from typing import Mapping, Optional, Dict

import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from raft_baselines.classifiers.classifier import Classifier


class NaiveBayesClassifier(Classifier):
    def __init__(self, training_data: datasets.Dataset,
                 **kwargs):
        super().__init__(training_data)

        cleaned_text_train = [self.stringify_row(row) for row in self.training_data]

        vectorizer_kwargs = {
            "strip_accents": 'unicode',
            "lowercase": True,
            "ngram_range": (1, 4),
            "max_df": 1.0,
            "min_df": 0.0
        }

        self.vectorizer = CountVectorizer(**vectorizer_kwargs).fit(cleaned_text_train)

        vectorized_training_data = self.vectorizer.transform(cleaned_text_train)
        self.classifier = MultinomialNB(**kwargs).fit(vectorized_training_data,
                                                      self.training_data['Label'])

        a = self.vectorizer.transform(cleaned_text_train)
        print(type(a))

        print(a.shape)

        print(self.classifier)

    def stringify_row(self, row):
        return ". ".join(row[input_col] for input_col in self.input_cols
                         if input_col in row)

    def classify(self, target: Mapping[str, str], random_seed: Optional[int] = None,
                 should_print_prompt: bool = False) -> Dict[str, float]:
        simple_input = self.stringify_row(target)
        vector_input = self.vectorizer.transform((simple_input,))
        result = self.classifier.predict_proba(vector_input)
        return {self.class_label_to_string(int(cls)): prob
                for prob, cls in zip(result[0], self.classifier.classes_)}

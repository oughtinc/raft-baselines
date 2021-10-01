from sklearn.svm import LinearSVC
from scipy.special import softmax
import numpy as np

from raft_baselines.classifiers.n_grams_classifier import NGramsClassifier


class DummyClassifier:
    def __init__(self, label):
        self.classes_ = [label]

    def decision_function(self, vector_input):
        return np.array([1])


class SVMClassifier(NGramsClassifier):
    def __init__(self, training_data, vectorizer_kwargs, model_kwargs, **kwargs):
        super().__init__(training_data, vectorizer_kwargs, model_kwargs, **kwargs)
        if model_kwargs is None:
            model_kwargs = {}
        # Sometimes breaks if there's only one label in the training data.
        # Hack-y solution.
        if len(set(self.training_data["Label"])) == 1:
            self.classifier = DummyClassifier(self.training_data["Label"][0])
            return
        self.classifier = LinearSVC(**model_kwargs)
        self.classifier.fit(self.vectorized_training_data, self.training_data["Label"])

    def _classify(self, vector_input):
        confidences = self.classifier.decision_function(vector_input)
        if len(self.classifier.classes_) <= 2:
            # Positive score means first class, negative score means second class.
            # Appending a 0 ensures that the softmax classifies correctly while still
            #   ensuring somewhat sensible confidences. Not probabilities though.
            confidences = np.append(confidences, 0)
            confidences = confidences.reshape(1, 2)
        return softmax(confidences)

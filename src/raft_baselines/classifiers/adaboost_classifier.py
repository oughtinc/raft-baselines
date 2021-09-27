from sklearn.ensemble import AdaBoostClassifier as AdaBoost

from raft_baselines.classifiers.non_neural_classifier import NonNeuralClassifier


class AdaBoostClassifier(NonNeuralClassifier):
    def __init__(self, training_data, vectorizer_kwargs=None, model_kwargs=None, **kwargs):
        super().__init__(training_data, vectorizer_kwargs, model_kwargs, **kwargs)
        if model_kwargs is None:
            model_kwargs = {}
        self.classifier = AdaBoost(**model_kwargs)
        self.classifier.fit(self.vectorized_training_data, self.training_data['Label'])

    def _classify(self, vector_input):
        return self.classifier.predict_proba(vector_input)

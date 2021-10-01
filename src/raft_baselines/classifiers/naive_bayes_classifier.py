from sklearn.naive_bayes import MultinomialNB

from raft_baselines.classifiers.n_grams_classifier import NGramsClassifier


class NaiveBayesClassifier(NGramsClassifier):
    def __init__(
        self, training_data, vectorizer_kwargs=None, model_kwargs=None, **kwargs
    ):
        super().__init__(training_data, vectorizer_kwargs, model_kwargs, **kwargs)
        if model_kwargs is None:
            model_kwargs = {}
        self.classifier = MultinomialNB(**model_kwargs)
        self.classifier.fit(self.vectorized_training_data, self.training_data["Label"])

    def _classify(self, vector_input):
        return self.classifier.predict_proba(vector_input)

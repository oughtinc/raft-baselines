from copy import deepcopy

from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.tree import DecisionTreeClassifier

from raft_baselines.classifiers.n_grams_classifier import NGramsClassifier


class AdaBoostClassifier(NGramsClassifier):
    def __init__(
        self, training_data, vectorizer_kwargs=None, model_kwargs=None, **kwargs
    ):
        super().__init__(training_data, vectorizer_kwargs, model_kwargs, **kwargs)
        if model_kwargs is None:
            model_kwargs = {}
        if "max_depth" in model_kwargs:
            # Required for sacred
            model_kwargs = deepcopy(model_kwargs)
            d = model_kwargs.pop("max_depth")
            base = DecisionTreeClassifier(max_depth=d)
            model_kwargs["base_estimator"] = base
        self.classifier = AdaBoost(**model_kwargs)
        self.classifier.fit(self.vectorized_training_data, self.training_data["Label"])

    def _classify(self, vector_input):
        return self.classifier.predict_proba(vector_input)

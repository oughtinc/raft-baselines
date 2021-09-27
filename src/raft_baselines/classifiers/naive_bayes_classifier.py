from sklearn.naive_bayes import MultinomialNB

from raft_baselines.classifiers.non_neural_classifier import NonNeuralClassifier


class NaiveBayesClassifier(NonNeuralClassifier):
    def train(self, **classifier_kwargs):
        self.classifier = MultinomialNB(**classifier_kwargs)
        self.classifier.fit(self.vectorized_training_data, self.training_data['Label'])

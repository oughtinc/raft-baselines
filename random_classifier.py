import random


class RandomClassifier:
    def __init__(self, training_data, seed=4):
        self.classes = list(training_data.features["Label"].names[1:])
        random.seed(seed)

    def classify(self, input):
        result = {c: 0.0 for c in self.classes}
        result[random.choice(self.classes)] = 1.0

        return result

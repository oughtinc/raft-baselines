import datasets

from raft_baselines.classifiers import NaiveBayesClassifier
from raft_baselines.classifiers import RandomClassifier

train = datasets.load_dataset(
    "ought/raft", "banking_77", split="train"
)

classifier = NaiveBayesClassifier(train)

print(classifier.classify({"Paper title": "CNN research", "Impact statement": "test2"}))

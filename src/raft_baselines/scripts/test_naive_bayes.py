import datasets

from raft_baselines.classifiers import NaiveBayesClassifier

train = datasets.load_dataset(
    "ought/raft", "neurips_impact_statement_risks", split="train"
)

classifier = NaiveBayesClassifier(train)

print(classifier.classify({"Paper title": "CNN research", "Impact statement": "test2"}))

import datasets

from raft_baselines.classifiers import GPT3Classifier

train = datasets.load_dataset(
    "ought/raft", "neurips_impact_statement_risks", split="train"
)
classifier = GPT3Classifier(
    train, config="neurips_impact_statement_risks", do_semantic_selection=True
)
print(classifier.classify({"Paper title": "GNN research", "Impact statement": "test2"}))

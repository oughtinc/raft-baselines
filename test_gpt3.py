from gpt3_classifier import GPT3Classifier
import datasets


train = datasets.load_dataset(
    "ought/raft", "neurips_impact_statement_risks", split="train"
)
classifier = GPT3Classifier(train, config="neurips_impact_statement_risks")
print(classifier.classify({"Paper title": "test", "Impact statement": "test2"}))

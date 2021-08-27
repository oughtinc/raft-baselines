from gpt3_classifier import GPT3Classifier
import datasets


train = datasets.load_dataset('ought/raft', "banking_77", split='train')
classifier = GPT3Classifier(train, add_prefixes=True)
print(classifier.classify({"Query": "My transfers keep getting declined"}))


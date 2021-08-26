from gpt3_classifier import GPT3Classifier
import datasets


train = datasets.load_dataset('ought/raft', "tai_safety_research", split='train')
classifier = GPT3Classifier(train)
print(classifier.classify({'Note': "AI safety is important."}))


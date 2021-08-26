from gpt3_classifier import GPT3Classifier
import datasets


train = datasets.load_dataset('ought/raft', "medical_subdomain_of_clinical_notes", split='train')
classifier = GPT3Classifier(train)
print(classifier.classify({'Note': "Psychiatry, mental health, dangerous."}))


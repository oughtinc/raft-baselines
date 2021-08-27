from distil_gpt2_classifier import DistilGPT2Classifier
import datasets


train = datasets.load_dataset(
    "ought/raft", "neurips_impact_statement_risks", split="train"
)
classifier = DistilGPT2Classifier(train, config="neurips_impact_statement_risks", max_prompt_tokens=1024)
print(
    classifier.classify(
        {"Paper title": "GNN research", "Impact statement": "GNN research is amazing"}
    )
)

from gpt_neo_classifier import GPTNeoClassifier
import datasets


train = datasets.load_dataset(
    "ought/raft", "neurips_impact_statement_risks", split="train"
)
classifier = GPTNeoClassifier(train, config="neurips_impact_statement_risks", max_prompt_tokens=1024)
print(
    classifier.classify(
        {"Paper title": "GNN research", "Impact statement": "GNN research is amazing"}
    )
)

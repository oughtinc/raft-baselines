import os
import shutil
import csv

import datasets
from sacred import Experiment, observers

from random_classifier import RandomClassifier
from gpt3_classifier import GPT3Classifier

"""
This class runs a classifier specified by `classifier_cls` on the unlabeled 
    test sets for all configs given in `configs`. Any classifier can be used,
    but must accept a hf.datasets.Dataset as an argument. Any other keyword
    arguments must be specified via `classifier_kwargs`.
"""

experiment_name = "make_predictions_c"
raft_experiment = Experiment(experiment_name, save_git_info=False)
observer = observers.FileStorageObserver(f"results/{experiment_name}")
raft_experiment.observers.append(observer)

NUM_EXAMPLES = {"ade_corpus_v2": 25,
                "banking_77": 10,
                "terms_of_service": 5,
                "tai_safety_research": 5,
                "neurips_impact_statement_risks": 5,
                "medical_subdomain_of_clinical_notes": 10,
                "overruling": 25,
                "systematic_review_inclusion": 5,
                "one_stop_english": 5,
                "tweet_eval_hate": 50,
                "twitter_complaints": 25,
                "semiconductor_org_types": 50}


@raft_experiment.config
def base_config():
    classifier_cls = GPT3Classifier
    classifier_kwargs = {"engine": "davinci",
                         "use_task_specific_instructions": True,
                         "do_semantic_selection": True}
    # configs = datasets.get_dataset_config_names("ought/raft")
    configs = ["twitter_complaints", "semiconductor_org_types", "medical_subdomain_of_clinical_notes"]


@raft_experiment.capture
def load_datasets_train(configs):
    train_datasets = {
        config: datasets.load_dataset("ought/raft", config, split="train")
        for config in configs
    }
    test_datasets = {
        config: datasets.load_dataset("ought/raft", config, split="test")
        for config in configs
    }

    return train_datasets, test_datasets


@raft_experiment.capture
def make_predictions(train_dataset, test_dataset, config, extra_kwargs,
                     classifier_cls, classifier_kwargs):
    classifier = classifier_cls(train_dataset, **classifier_kwargs, **extra_kwargs)

    # test_dataset = test_dataset.select(range(1))

    dummy_input = test_dataset[0]
    del dummy_input["Label"]
    example_prompt = classifier.format_prompt(dummy_input)

    log_text(example_prompt, "prompts", config+".txt")

    def predict(example):
        del example["Label"]
        output_probs = classifier.classify(example)
        output = max(output_probs.items(), key=lambda kv_pair: kv_pair[1])

        example["Label"] = train_dataset.features["Label"].str2int(output[0])
        return example

    return test_dataset.map(predict)


def make_extra_kwargs(config):
    extra_kwargs = {"config": config,
                    "num_prompt_training_examples": NUM_EXAMPLES[config]}
    if config == "banking_77":
        extra_kwargs["add_prefixes"] = True
    return extra_kwargs


def log_text(text, dirname, filename):
    targetdir = os.path.join(observer.dir, dirname)
    if not os.path.isdir(targetdir):
        os.mkdir(targetdir)

    with open(os.path.join(targetdir, filename), 'w') as f:
        f.write(text)


def prepare_predictions_folder():
    sacred_dir = os.path.join(observer.dir, "predictions")
    if os.path.isdir(sacred_dir):
        shutil.rmtree(sacred_dir)
    os.mkdir(sacred_dir)


def write_predictions(labeled, config):
    int2str = labeled.features["Label"].int2str

    sacred_pred_file = os.path.join(observer.dir, "predictions", f"{config}.csv")

    with open(sacred_pred_file, "w", newline="") as f:
        writer = csv.writer(
            f,
            quotechar='"',
            delimiter=",",
            quoting=csv.QUOTE_MINIMAL,
            skipinitialspace=True,
        )
        writer.writerow(["ID", "Label"])
        for row in labeled:
            writer.writerow([row["ID"],
                             int2str(row["Label"])])


@raft_experiment.automain
def main():
    train, unlabeled = load_datasets_train()
    prepare_predictions_folder()
    for config in unlabeled:
        extra_kwargs = make_extra_kwargs(config)
        labeled = make_predictions(train[config], unlabeled[config],
                                   config, extra_kwargs)
        write_predictions(labeled, config)


if __name__ == "__main__":
    main()

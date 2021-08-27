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

experiment_name = "make_predictions"
raft_experiment = Experiment(experiment_name, save_git_info=False)
observer = observers.FileStorageObserver(f"results/{experiment_name}")
raft_experiment.observers.append(observer)


@raft_experiment.config
def base_config():
    classifier_cls = GPT3Classifier
    classifier_kwargs = {"engine": "ada",
                         "num_prompt_training_examples": 20,
                         "use_task_specific_instructions": True}
    configs = datasets.get_dataset_config_names("ought/raft")
    # configs = ["systematic_review_inclusion"]


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
def make_predictions(train_datasets, test_datasets, classifier_cls, classifier_kwargs):
    for config in train_datasets:
        extra_kwargs = {"config": config}
        if config == "banking_77":
            extra_kwargs["add_prefixes"] = True

        train_dataset = train_datasets[config]
        classifier = classifier_cls(train_dataset, **classifier_kwargs, **extra_kwargs)

        test_dataset = test_datasets[config]
        test_dataset = test_dataset.select(range(20))

        dummy_input = test_dataset[0]
        train_examples = classifier.select_training_examples(dummy_input, random_seed=4)
        example_prompt = classifier.format_prompt(dummy_input, train_examples)

        log_text(example_prompt)

        def predict(example):
            output_probs = classifier.classify(example)
            output = max(output_probs.items(), key=lambda kv_pair: kv_pair[1])

            example["Label"] = train_dataset.features["Label"].str2int(output[0])
            return example

        test_datasets[config] = test_dataset.map(predict)

    return test_datasets


def log_text(text):
    with open(os.path.join(observer.dir, "prompt.txt"), 'w') as f:
        f.write(text)


def write_predictions(labeled):
    if os.path.isdir("predictions"):
        shutil.rmtree("predictions")
    os.mkdir("predictions")

    for config in labeled:
        int2str = labeled[config].features["Label"].int2str

        dataset = labeled[config]
        os.path.join("predictions", f"{config}.csv")
        with open(os.path.join("predictions", f"{config}.csv"), "w", newline="") as f:
            writer = csv.writer(
                f,
                quotechar='"',
                delimiter=",",
                quoting=csv.QUOTE_MINIMAL,
                skipinitialspace=True,
            )
            writer.writerow(["ID", "Label"])
            for row in dataset:
                writer.writerow([row["ID"],
                                 int2str(row["Label"])])

    sacred_pred_dir = os.path.join(observer.dir, "predictions")
    if os.path.isdir(sacred_pred_dir):
        shutil.rmtree(sacred_pred_dir)
    shutil.copytree("predictions", sacred_pred_dir)


@raft_experiment.automain
def main():
    train, unlabeled = load_datasets_train()
    labeled = make_predictions(train, unlabeled)
    write_predictions(labeled)


if __name__ == "__main__":
    main()

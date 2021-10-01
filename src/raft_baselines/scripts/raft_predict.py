import os
import shutil
import csv

import datasets
from sacred import Experiment, observers

from raft_baselines import classifiers

"""
This class runs a classifier specified by `classifier_name` on the unlabeled
    test sets for all configs given in `configs`. Any classifier can be used,
    but must accept a hf.datasets.Dataset as an argument. Any other keyword
    arguments must be specified via `classifier_kwargs`.
"""

experiment_name = "make_predictions"
raft_experiment = Experiment(experiment_name, save_git_info=False)
observer = observers.FileStorageObserver(f"results/{experiment_name}")
raft_experiment.observers.append(observer)

# Best performing on a per-dataset basis using raft_train_experiment.py
NUM_EXAMPLES = {
    "ade_corpus_v2": 25,
    "banking_77": 10,
    "terms_of_service": 5,
    "tai_safety_research": 5,
    "neurips_impact_statement_risks": 25,
    "overruling": 25,
    "systematic_review_inclusion": 10,
    "one_stop_english": 5,
    "tweet_eval_hate": 50,
    "twitter_complaints": 25,
    "semiconductor_org_types": 5,
}


@raft_experiment.config
def base_config():
    classifier_name = "GPT3Classifier"
    classifier_kwargs = {
        # change to davinci to replicate results from the paper
        # "engine": "ada",
    }
    if classifier_name in ('NaiveBayesClassifier', 'SVMClassifier', 'AdaBoostClassifier'):
        classifier_kwargs = {
            "vectorizer_kwargs": {
                "strip_accents": 'unicode',
                "lowercase": True,
                "ngram_range": (1, 5),
                "max_df": 1.0,
                "min_df": 0.0
             },
            "model_kwargs": {}
        }
        if classifier_name == "NaiveBayesClassifier":
            classifier_kwargs['model_kwargs']['alpha'] = 0.05
        elif classifier_name == "AdaBoostClassifier":
            classifier_kwargs['model_kwargs']['max_depth'] = 3
            classifier_kwargs['model_kwargs']['n_estimators'] = 100

    configs = datasets.get_dataset_config_names("ought/raft")
    # set n_test to -1 to run on all test examples
    n_test = 5
    random_seed = 42
    zero_shot = False


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
def make_extra_kwargs(config: str, zero_shot: bool):
    extra_kwargs = {
        "config": config,
        "num_prompt_training_examples": NUM_EXAMPLES[config] if not zero_shot else 0,
    }
    if config == "banking_77":
        extra_kwargs["add_prefixes"] = True
    return extra_kwargs


@raft_experiment.capture
def make_predictions(
    train_dataset,
    test_dataset,
    classifier_cls,
    extra_kwargs,
    n_test,
    classifier_kwargs,
    random_seed,
):
    classifier = classifier_cls(train_dataset, **classifier_kwargs, **extra_kwargs)

    if n_test > -1:
        test_dataset = test_dataset.select(range(n_test))

    def predict(example):
        del example["Label"]
        output_probs = classifier.classify(example, random_seed=random_seed)
        output = max(output_probs.items(), key=lambda kv_pair: kv_pair[1])

        example["Label"] = train_dataset.features["Label"].str2int(output[0])
        return example

    return test_dataset.map(predict, load_from_cache_file=False)


def log_text(text, dirname, filename):
    targetdir = os.path.join(observer.dir, dirname)
    if not os.path.isdir(targetdir):
        os.mkdir(targetdir)

    with open(os.path.join(targetdir, filename), "w") as f:
        f.write(text)


def prepare_predictions_folder():
    sacred_dir = os.path.join(observer.dir, "predictions")
    if os.path.isdir(sacred_dir):
        shutil.rmtree(sacred_dir)
    os.mkdir(sacred_dir)


def write_predictions(labeled, config):
    int2str = labeled.features["Label"].int2str

    config_dir = os.path.join(observer.dir, "predictions", config)
    if os.path.isdir(config_dir):
        shutil.rmtree(config_dir)
    os.mkdir(config_dir)

    pred_file = os.path.join(config_dir, "predictions.csv")

    with open(pred_file, "w", newline="") as f:
        writer = csv.writer(
            f,
            quotechar='"',
            delimiter=",",
            quoting=csv.QUOTE_MINIMAL,
            skipinitialspace=True,
        )
        writer.writerow(["ID", "Label"])
        for row in labeled:
            writer.writerow([row["ID"], int2str(row["Label"])])


@raft_experiment.automain
def main(classifier_name):
    train, unlabeled = load_datasets_train()
    prepare_predictions_folder()

    classifier_cls = getattr(classifiers, classifier_name)

    for config in unlabeled:
        extra_kwargs = make_extra_kwargs(config)
        labeled = make_predictions(
            train[config], unlabeled[config], classifier_cls, extra_kwargs
        )
        write_predictions(labeled, config)

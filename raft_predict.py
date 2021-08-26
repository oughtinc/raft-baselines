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

raft_experiment = Experiment("raft_prediction")
observer = observers.FileStorageObserver('results')
raft_experiment.observers.append(observer)


@raft_experiment.config
def base_config():
    classifier_cls = RandomClassifier
    classifier_kwargs = {}
    configs = datasets.get_dataset_config_names('ought/raft')


@raft_experiment.capture
def load_datasets_train(configs):
    train_datasets = {config: datasets.load_dataset('ought/raft', config, split='train')
                      for config in configs}
    test_datasets = {config: datasets.load_dataset('ought/raft', config, split='test')
                     for config in configs}

    return train_datasets, test_datasets


@raft_experiment.capture
def make_predictions(train_datasets, test_datasets,
                     classifier_cls, classifier_kwargs):
    for config in train_datasets:
        train_dataset = train_datasets[config]
        classifier = classifier_cls(train_dataset, **classifier_kwargs)

        test_dataset = test_datasets[config]

        def predict(example):
            output_probs = classifier.classify(example)
            output = max(output_probs.items(),
                         key=lambda kv_pair: kv_pair[1])

            example['Label'] = train_dataset.features['Label'].str2int(output[0])
            return example

        test_datasets[config] = test_dataset.map(predict)

    return test_datasets


def write_predictions(labeled):
    if os.path.isdir('predictions'):
        shutil.rmtree('predictions')
    os.mkdir('predictions')

    for config in labeled:
        dataset = labeled[config]
        os.path.join('predictions', f"{config}.csv")
        with open(os.path.join('predictions', f"{config}.csv"), 'w', newline='') as f:
            writer = csv.writer(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_MINIMAL, skipinitialspace=True)
            writer.writerow(['ID', 'Label'])
            for row in dataset:
                writer.writerow([row['ID'], row['Label']])

    sacred_pred_dir = os.path.join(observer.dir, 'predictions')
    if os.path.isdir(sacred_pred_dir):
        shutil.rmtree(sacred_pred_dir)
    shutil.copytree('predictions', sacred_pred_dir)


@raft_experiment.automain
def main():
    train, unlabeled = load_datasets_train()
    labeled = make_predictions(train, unlabeled)
    write_predictions(labeled)
    # for example in labeled_data:
    #     print(example['title'], example['answer'])


if __name__ == "__main__":
    main()


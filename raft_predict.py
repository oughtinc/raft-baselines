import random

import datasets
from sacred import Experiment, observers

from random_classifier import RandomClassifier

sd_experiment = Experiment("raft_prediction")
# TODO: move to subfolder?
observer = observers.FileStorageObserver('results')
sd_experiment.observers.append(observer)


def base_config():
    classifier_cls = RandomClassifier
    classifier_kwargs = {}


def load_datasets_train():
    configs = datasets.get_dataset_config_names('ought/raft')

    train_datasets = {config: datasets.load_dataset('ought/raft', config, split='train')
                      for config in configs}
    test_datasets = {config: datasets.load_dataset('ought/raft', config, split='test')
                     for config in configs}

    return train_datasets, test_datasets

@sd_experiment.capture
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

            example['Label'] = output
            return example

        test_datasets[config] = test_dataset.map(predict)

    return test_datasets


def write_predictions(labeled):
    pass


@sd_experiment.automain
def main():
    random.seed(4)  # Chosen by fair dice roll, guaranteed to be random
    train, unlabeled = load_datasets_train()
    labeled = make_predictions(train, unlabeled)
    write_predictions(labeled)
    # for example in labeled_data:
    #     print(example['title'], example['answer'])


if __name__ == "__main__":
    main()


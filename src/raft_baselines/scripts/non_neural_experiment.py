import datasets
from sacred import Experiment, observers
import sklearn.metrics as skm

from raft_baselines import classifiers

experiment_name = "non_neural"
raft_experiment = Experiment(experiment_name, save_git_info=False)
observer = observers.FileStorageObserver(f"results/{experiment_name}")
raft_experiment.observers.append(observer)


@raft_experiment.config
def base_config():
    classifier_name = "AdaBoostClassifier"
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
    configs = datasets.get_dataset_config_names("ought/raft")
    # controls which dimension is tested, out of the 3 reported in the paper
    # Other options: do_semantic_selection and num_prompt_training_examples
    random_seed = 42


@raft_experiment.capture
def load_datasets_train(configs):
    train_datasets = {
        config: datasets.load_dataset("ought/raft", config, split="train")
        for config in configs
    }
    return train_datasets


@raft_experiment.capture
def test_experiment(
    train_datasets, classifier_name,
    classifier_kwargs, random_seed
):
    classifier_cls = getattr(classifiers, classifier_name)

    for config in train_datasets:
        dataset = train_datasets[config]
        labels = list(range(1, dataset.features["Label"].num_classes))
        predictions = []

        for i in range(len(dataset)):
            train = dataset.select([j for j in range(len(dataset)) if j != i])
            test = dataset.select([i])

            # Non-neural classifiers (i.e. non-GPT style) should be explicitly trained
            #   (mostly, this is to allow two separate kwargs arguments)
            classifier = classifier_cls(train, **classifier_kwargs)

            def predict(example):
                del example["Label"]
                del example["ID"]
                output_probs = classifier.classify(example, random_seed=random_seed)
                output = max(output_probs.items(), key=lambda kv_pair: kv_pair[1])

                predictions.append(dataset.features["Label"].str2int(output[0]))

            test.map(predict)

        # accuracy = sum([p == l for p, l in zip(predictions, dataset['Label'])]) / 50
        f1 = skm.f1_score(
            dataset["Label"], predictions, labels=labels, average="macro"
        )
        print(f"Dataset - {config}; {f1}")
        raft_experiment.log_scalar(f"{config}", f1)


@raft_experiment.automain
def main():
    train = load_datasets_train()
    test_experiment(train)

import datasets
from sacred import Experiment, observers
import sklearn.metrics as skm

from raft_baselines import classifiers

experiment_name = "loo_tuning"
raft_experiment = Experiment(experiment_name, save_git_info=False)
observer = observers.FileStorageObserver(f"results/{experiment_name}")
raft_experiment.observers.append(observer)


@raft_experiment.config
def base_config():
    classifier_name = "GPT3Classifier"
    classifier_kwargs = {
        # change to davinci to replicate results from the paper
        "engine": "ada",
    }
    configs = datasets.get_dataset_config_names("ought/raft")
    # controls which dimension is tested, out of the 3 reported in the paper
    # Other options: do_semantic_selection and num_prompt_training_examples
    test_dimension = "use_task_specific_instructions"
    random_seed = 42


@raft_experiment.capture
def load_datasets_train(configs):
    train_datasets = {
        config: datasets.load_dataset("ought/raft", config, split="train")
        for config in configs
    }
    return train_datasets


@raft_experiment.capture
def loo_test(
    train_datasets, classifier_name, classifier_kwargs, test_dimension, random_seed
):
    # Change what to iterate over, filling in extra_kwargs to test different
    # configurations of the classifier.

    if test_dimension == "use_task_specific_instructions":
        dim_values = [False, True]
        other_dim_kwargs = {
            "do_semantic_selection": False,
            "num_prompt_training_examples": 20,
        }
    elif test_dimension == "do_semantic_selection":
        dim_values = [False, True]
        other_dim_kwargs = {
            "use_task_specific_instructions": True,
            "num_prompt_training_examples": 20,
        }
    elif test_dimension == "num_prompt_training_examples":
        dim_values = [5, 10, 25, 49]
        other_dim_kwargs = {
            "use_task_specific_instructions": True,
            "do_semantic_selection": True,
        }
    else:
        raise ValueError(f"test_dimension {test_dimension} not recognized")

    classifier_cls = getattr(classifiers, classifier_name)

    for config in train_datasets:
        for dim_value in dim_values:
            dataset = train_datasets[config]
            labels = list(range(1, dataset.features["Label"].num_classes))
            predictions = []

            extra_kwargs = {
                "config": config,
                test_dimension: dim_value,
                **other_dim_kwargs,
            }
            if config == "banking_77":
                extra_kwargs["add_prefixes"] = True

            for i in range(len(dataset)):
                train = dataset.select([j for j in range(len(dataset)) if j != i])
                test = dataset.select([i])

                classifier = classifier_cls(train, **classifier_kwargs, **extra_kwargs)

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
            print(f"Dataset - {config}; {test_dimension} - {dim_value}: {f1}")
            raft_experiment.log_scalar(f"{config}.{dim_value}", f1)


@raft_experiment.automain
def main():
    train = load_datasets_train()
    loo_test(train)

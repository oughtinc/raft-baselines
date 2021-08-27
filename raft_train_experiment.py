import datasets
from sacred import Experiment, observers
import sklearn.metrics as skm
from gpt3_classifier import GPT3Classifier

experiment_name = "num_examples"
raft_experiment = Experiment(experiment_name, save_git_info=False)
observer = observers.FileStorageObserver(f"results/{experiment_name}")
raft_experiment.observers.append(observer)


@raft_experiment.config
def base_config():
    classifier_cls = GPT3Classifier
    classifier_kwargs = {
        "engine": "davinci",
        "use_task_specific_instructions": True,
        "do_semantic_selection": True,
    }
    configs = datasets.get_dataset_config_names("ought/raft")
    # configs = ["neurips_impact_statement_risks"]
    # configs = ["semiconductor_org_types"]


@raft_experiment.capture
def load_datasets_train(configs):
    train_datasets = {
        config: datasets.load_dataset("ought/raft", config, split="train")
        for config in configs
    }
    return train_datasets


@raft_experiment.capture
def loo_test(train_datasets, classifier_cls, classifier_kwargs):
    n_examples = [5, 10, 25, 49]

    for config in train_datasets:
        for n in n_examples:
            dataset = train_datasets[config]
            labels = list(range(1, dataset.features["Label"].num_classes))
            predictions = []

            extra_kwargs = {"config": config, "num_prompt_training_examples": n}
            if config == "banking_77":
                extra_kwargs["add_prefixes"] = True

            for i in range(len(dataset)):
                train = dataset.select([j for j in range(len(dataset)) if j != i])
                test = dataset.select([i])

                classifier = classifier_cls(train, **classifier_kwargs, **extra_kwargs)

                def predict(example):
                    del example["Label"]
                    del example["ID"]
                    output_probs = classifier.classify(example)
                    output = max(output_probs.items(), key=lambda kv_pair: kv_pair[1])

                    predictions.append(dataset.features["Label"].str2int(output[0]))

                    test.map(predict)

            # accuracy = sum([p == l for p, l in zip(predictions, dataset['Label'])]) / 50
            f1 = skm.f1_score(
                dataset["Label"], predictions, labels=labels, average="macro"
            )

            print(f"Dataset - {config}; Num examples - {n}: {f1}")

            raft_experiment.log_scalar(f"{config}.{n}_examples", f1)


@raft_experiment.automain
def main():
    train = load_datasets_train()
    loo_test(train)

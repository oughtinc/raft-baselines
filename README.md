# Setup

This is the repository for the GPT-3 baselines described in the RAFT benchmark paper.

Set up a virtual environament and install necessary requirements from the requirements file.

```buildoutcfg
conda create -n raft-baselines python=3.8 && conda activate raft-baselines
python -m pip install -r requirements.txt
```

Install raft-baselines.

```buildoutcfg
python setup.py develop
```

You may have to run the above command with `sudo` prepended for permissions.

# Starter Kit

A [starter kit notebook](src/raft_baselines/scripts/starter_kit.ipynb) walks through the basics of making predictions using models from the [HuggingFace Model Hub](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads).

# RAFT Predict

Use the `raft_predict` script to run classifiers on the RAFT datasets. By default, the script will run on the first 5 test examples for each dataset. To use a random classifier on the first 10 examples from the ADE Corpus V2 dataset:

```buildoutcfg
python -m raft_baselines.scripts.raft_predict with n_test=10 'configs=["ade_corpus_v2"]' classifier_name=RandomClassifier
```

The other classifiers available are:

- `GPT3Classifier`: the one used for the GPT-3 baseline in the paper
- `TransformersCausalLMClassifier`: takes as input a `model_type` string, and runs an arbitrary CausalLM from the [HuggingFace Model Hub](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads)

For example, to generate predictions from DistilGPT-2 on the first 10 examples of the ADE Corpus you can run:

```buildoutcfg
python -m raft_baselines.scripts.raft_predict with n_test=10 'configs=["ade_corpus_v2"]' classifier_name=TransformersCausalLMClassifier 'classifier_kwargs={"model_type":"distilgpt2"}'
```

In order to run experiments with GPT-3, you will need to have an OpenAI API key. Create a file called `.env` and put your API key there. Copy the format of `.env-example`:

```buildoutcfg
echo OPENAI_API_KEY=$OPENAI_API_KEY > .env
```

## Sacred

We use [Sacred](https://github.com/IDSIA/sacred) to track our experiments and outputs. This has no overhead at runtime, simply run either of our two experiment scripts with python like normal. You can change where tracking files get saved to by modifying the observer at the top of every experiment file, or you can change the details of the experiment via the various configuration parameters specified in the configs block.

```buildoutcfg
# For labeling the test set
python -m raft_baselines.scripts.raft_predict
# For tuning various dimensions on the train set with LOO validation
python -m raft_baselines.scripts.raft_train_experiment
```

Alternately, you can modify the input variables to an experiment from the command line, as is done in the example above. Regardless, some modification will be necessary if you want to run different experiments. See [this tutorial](https://sacred.readthedocs.io/en/stable/configuration.html) for more information.

Similarly, you can save metrics with `raft_experiment.log_scalar()`, or by using the sacred observer directly. See [this tutorial](https://sacred.readthedocs.io/en/stable/collected_information.html) for more information.

To save out predictions and upload to the HuggingFace Hub (and the leaderboard), see [the RAFT submission template](https://huggingface.co/datasets/ought/raft-submission).

## License

This repository is licensed under the MIT License.

# RAFT Baselines

This is the repository for the GPT-3 baselines described in the RAFT benchmark paper. 

In order to run experiments with GPT-3, you will need to have an OpenAI API key. Create a file called `.env` and put your API key there. Copy the format of `.env-example`

```buildoutcfg
echo OPENAI_API_KEY=$OPENAI_API_KEY > .env
```

Install necessary requirements from the requirements file.

```buildoutcfg
pip install -r requirements.txt
```

Install raft_baselines.

```buildoutcfg
python setup.py develop
```

If you do not have access to GPT-3, you may use `random_classifier.py`, or create your own classifier that implements the `classify(input)` method.

Test random classifier:
```buildoutcfg
python -m raft_baselines.scripts.raft_predict with n_test=10 configs=['tai_safety_research'] classifier_cls=RandomClassifier
```

## Sacred Experiment Scripts

We use [Sacred](https://github.com/IDSIA/sacred) to track our experiments and outputs. This has no overhead at runtime, simply run either of our two experiment scripts with python like normal. You can change where tracking files get saved to by modifying the observer at the top of every experiment file. 

```buildoutcfg
# For labeling the test set
python raft_predict.py
# For tuning on the train set with LOO validation
python raft_train_experiment.py
```

In either case, you should change the experiment config if you want to run different experiments. See [this tutorial](https://sacred.readthedocs.io/en/stable/configuration.html) for more information. 

Similarly, you can save metrics with `raft_experiment.log_scalar()`, or by using the sacred observer directly. See [this tutorial](https://sacred.readthedocs.io/en/stable/collected_information.html) for more information.

To save out predictions and upload to the HuggingFace Hub (and the leaderboard), you will need to run `predictions_to_submission.py`, which will copy a given folder into HF-expected data directory structure. 

## License

This repository is licensed under the MIT License.
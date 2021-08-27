# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import json
import os

import datasets

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# You can copy an official description
_DESCRIPTION = """
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
# This gets all folders within the directory named `data`
# DATA_DIRS = next(os.walk('data'))[1]
DATA_DIR_URL = "data/"  # "https://huggingface.co/datasets/ought/raft/resolve/main/data/"
# print([p for p in DATA_DIR_PATH.iterdir() if p.is_dir()])
TASKS = {
    "ade_corpus_v2": {
        "name": "ade_corpus_v2",
        "description": "",
        "data_columns": [
            "Sentence",
            "ID"
        ],
        "label_columns": {
            "Label": [
                "ADE-related",
                "not ADE-related"
            ]
        }
    },
    "banking_77": {
        "name": "banking_77",
        "description": "",
        "data_columns": [
            "Query",
            "ID"
        ],
        "label_columns": {
            "Label": [
                "Refund_not_showing_up",
                "activate_my_card",
                "age_limit",
                "apple_pay_or_google_pay",
                "atm_support",
                "automatic_top_up",
                "balance_not_updated_after_bank_transfer",
                "balance_not_updated_after_cheque_or_cash_deposit",
                "beneficiary_not_allowed",
                "cancel_transfer",
                "card_about_to_expire",
                "card_acceptance",
                "card_arrival",
                "card_delivery_estimate",
                "card_linking",
                "card_not_working",
                "card_payment_fee_charged",
                "card_payment_not_recognised",
                "card_payment_wrong_exchange_rate",
                "card_swallowed",
                "cash_withdrawal_charge",
                "cash_withdrawal_not_recognised",
                "change_pin",
                "compromised_card",
                "contactless_not_working",
                "country_support",
                "declined_card_payment",
                "declined_cash_withdrawal",
                "declined_transfer",
                "direct_debit_payment_not_recognised",
                "disposable_card_limits",
                "edit_personal_details",
                "exchange_charge",
                "exchange_rate",
                "exchange_via_app",
                "extra_charge_on_statement",
                "failed_transfer",
                "fiat_currency_support",
                "get_disposable_virtual_card",
                "get_physical_card",
                "getting_spare_card",
                "getting_virtual_card",
                "lost_or_stolen_card",
                "lost_or_stolen_phone",
                "order_physical_card",
                "passcode_forgotten",
                "pending_card_payment",
                "pending_cash_withdrawal",
                "pending_top_up",
                "pending_transfer",
                "pin_blocked",
                "receiving_money",
                "request_refund",
                "reverted_card_payment?",
                "supported_cards_and_currencies",
                "terminate_account",
                "top_up_by_bank_transfer_charge",
                "top_up_by_card_charge",
                "top_up_by_cash_or_cheque",
                "top_up_failed",
                "top_up_limits",
                "top_up_reverted",
                "topping_up_by_card",
                "transaction_charged_twice",
                "transfer_fee_charged",
                "transfer_into_account",
                "transfer_not_received_by_recipient",
                "transfer_timing",
                "unable_to_verify_identity",
                "verify_my_identity",
                "verify_source_of_funds",
                "verify_top_up",
                "virtual_card_not_working",
                "visa_or_mastercard",
                "why_verify_identity",
                "wrong_amount_of_cash_received",
                "wrong_exchange_rate_for_cash_withdrawal"
            ]
        }
    },
    "terms_of_service": {
        "name": "terms_of_service",
        "description": "",
        "data_columns": [
            "Sentence",
            "ID"
        ],
        "label_columns": {
            "Label": [
                "not potentially unfair",
                "potentially unfair"
            ]
        }
    },
    "tai_safety_research": {
        "name": "tai_safety_research",
        "description": "",
        "data_columns": [
            "Title",
            "Abstract Note",
            "Url",
            "Publication Year",
            "Item Type",
            "Author",
            "Publication Title",
            "ID"
        ],
        "label_columns": {
            "Label": [
                "TAI safety research",
                "not TAI safety research"
            ]
        }
    },
    "neurips_impact_statement_risks": {
        "name": "neurips_impact_statement_risks",
        "description": "",
        "data_columns": [
            "Paper title",
            "Paper link",
            "Impact statement",
            "ID"
        ],
        "label_columns": {
            "Label": [
                "doesn't mention a harmful application",
                "mentions a harmful application"
            ]
        }
    },
    "overruling": {
        "name": "overruling",
        "description": "",
        "data_columns": [
            "Sentence",
            "ID"
        ],
        "label_columns": {
            "Label": [
                "not overruling",
                "overruling"
            ]
        }
    },
    "systematic_review_inclusion": {
        "name": "systematic_review_inclusion",
        "description": "",
        "data_columns": [
            "Title",
            "Abstract",
            "Authors",
            "Journal",
            "ID"
        ],
        "label_columns": {
            "Label": [
                "included",
                "not included"
            ]
        }
    },
    "one_stop_english": {
        "name": "one_stop_english",
        "description": "",
        "data_columns": [
            "Article",
            "ID"
        ],
        "label_columns": {
            "Label": [
                "advanced",
                "elementary",
                "intermediate"
            ]
        }
    },
    "tweet_eval_hate": {
        "name": "tweet_eval_hate",
        "description": "",
        "data_columns": [
            "Tweet",
            "ID"
        ],
        "label_columns": {
            "Label": [
                "hate speech",
                "not hate speech"
            ]
        }
    },
    "twitter_complaints": {
        "name": "twitter_complaints",
        "description": "",
        "data_columns": [
            "Tweet text",
            "ID"
        ],
        "label_columns": {
            "Label": [
                "complaint",
                "no complaint"
            ]
        }
    },
    "semiconductor_org_types": {
        "name": "semiconductor_org_types",
        "description": "",
        "data_columns": [
            "Paper title",
            "Organization name",
            "ID"
        ],
        "label_columns": {
            "Label": [
                "company",
                "research institute",
                "university"
            ]
        }
    },
}

_URLs = {s: {'test': f"{DATA_DIR_URL}{s}/predictions.csv"} for s in TASKS}


class GPT3Baselines(datasets.GeneratorBasedBuilder):
    """RAFT Dataset GPT3 Baselines."""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')

    # TODO: Load task jsons

    # tasks = {}
    # for sd in DATA_DIRS:
    #     with open(os.path.join('data', sd, 'task.json')) as f:
    #         task_data = json.load(f)
    #     tasks[sd] = task_data

    BUILDER_CONFIGS = []
    for key in TASKS:
        td = TASKS[key]
        name = td['name']
        description = td['description']
        BUILDER_CONFIGS.append(datasets.BuilderConfig(name=name, version=VERSION,
                                                      description=description))

    DEFAULT_CONFIG_NAME = "tai_safety_research"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        DEFAULT_LABEL_NAME = "Unlabeled"

        task = TASKS[self.config.name]
        data_columns = {'ID': datasets.Value("string")}

        label_columns = {}
        for label_name in task['label_columns']:
            labels = [DEFAULT_LABEL_NAME] + task['label_columns'][label_name]
            label_columns[label_name] = datasets.ClassLabel(len(labels), labels)

        # Merge dicts
        features = datasets.Features(**data_columns, **label_columns)

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        data_dir = dl_manager.download_and_extract(_URLs)
        dataset = self.config.name
        return [
            datasets.SplitGenerator(name=datasets.Split.TEST,
                                    gen_kwargs={"filepath": data_dir[dataset]['test'],
                                                "split": "test"})
        ]

    def _generate_examples(
            self, filepath, split  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.
        with open(filepath, encoding="utf-8") as f:
            csv_reader = csv.reader(f, quotechar='"', delimiter=",",
                                    quoting=csv.QUOTE_ALL, skipinitialspace=True)
            column_names = next(csv_reader)
            for id_, row in enumerate(csv_reader):
                # dicts don't have inherent ordering in python, right??
                yield id_, {name: value for name, value in zip(column_names, row)}

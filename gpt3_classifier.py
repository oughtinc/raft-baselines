from typing import Dict, Optional, List, Tuple, Mapping

import numpy as np
import datasets

from utils import num_tokens, truncate_by_tokens, complete, gpt2_tokenizer


INSTRUCTIONS = {
    "ade_corpus_v2": """Label the sentence based on whether it is related to an adverse drug effect (ADE). Details are described below:
Drugs: Names of drugs and chemicals that include brand names, trivial names, abbreviations and systematic names were annotated. Mentions of drugs or chemicals should strictly be in a therapeutic context. This category does not include the names of metabolites, reaction byproducts, or hospital chemicals (e.g. surgical equipment disinfectants).
Adverse effect: Mentions of adverse effects include signs, symptoms, diseases, disorders, acquired abnormalities, deficiencies, organ damage or death that strictly occur as a consequence of drug intake.""",
    "banking_77": """The following is a banking customer service query. Classify the query into one of the 77 categories available.""",
    "terms_of_service": """Label the sentence from a Terms of Service based on whether it is potentially unfair. If it seems clearly unfair, mark it as potentially unfair.
According to art. 3 of the Directive 93/13 on Unfair Terms in Consumer Contracts, a contractual term is unfair if: 1) it has not been individually negotiated; and 2) contrary to the requirement of good faith, it causes a significant imbalance in the parties rights and obligations, to the detriment of the consumer. 
Details on types of potentially unfair clauses are found below:
The jurisdiction clause stipulates what courts will have the competence to adjudicate disputes under the contract. Jurisdiction clauses giving consumers a right to bring disputes in their place of residence were marked as clearly fair, whereas clauses stating that any judicial proceeding takes a residence away were marked as clearly unfair.
The choice of law clause specifies what law will govern the contract, meaning also what law will be applied in potential adjudication of a dispute arising under the contract. Clauses defining the applicable law as the law of the consumer's country of residence were marked as clearly fair. In every other case, the choice of law clause was considered as potentially unfair.
The limitation of liability clause stipulates that the duty to pay damages is limited or excluded, for certain kind of losses, under certain conditions. Clauses that explicitly affirm non-excludable providers' liabilities were marked as clearly fair. Clauses that reduce, limit, or exclude the liability of the service provider were marked as potentially unfair when concerning broad categories of losses or causes of them.
The unilateral change clause specifies the conditions under which the service provider could amend and modify the terms of service and/or the service itself. Such clause was always considered as potentially unfair.
The unilateral termination clause gives provider the right to suspend and/or terminate the service and/or the contract, and sometimes details the circumstances under which the provider claims to have a right to do so.
The contract by using clause stipulates that the consumer is bound by the terms of use of a specific service, simply by using the service, without even being required to mark that he or she has read and accepted them. We always marked such clauses as potentially unfair.
The content removal gives the provider a right to modify/delete user's content, including in-app purchases, and sometimes specifies the conditions under which the service provider may do so.
The arbitration clause requires or allows the parties to resolve their disputes through an arbitration process, before the case could go to court. Clauses stipulating that the arbitration should take place in a state other then the state of consumer's residence or be based on arbiter's discretion were marked as clearly unfair. Clauses defining arbitration as fully optional were marked as clearly fair.""",
    "tai_safety_research": """Transformative AI (TAI) is defined as AI that precipitates a transition comparable to (or more significant than) the agricultural or industrial revolution. Label a paper as "TAI safety research" if: 
1. The contents of the paper are directly motivated by, and substantively inform, the challenge of ensuring good outcomes for TAI, 
2. There is substantive content on AI safety, not just AI capabilities, 
3. The intended audience is the community of researchers, 
4. It meets a subjective threshold of seriousness/quality, 
5. Peer review is not required.""",
    "neurips_impact_statement_risks": """Label the impact statement based on whether it mentions a harmful application of the research done in the paper. Make sure the statement is sufficient to conclude there are harmful applications of the research being done, not a past risk that this research is solving.""",
    "medical_subdomain_of_clinical_notes": """Classify the clinical note based on which subdomain it belongs to.""",
    "overruling": """In law, an overruling sentence is a statement that nullifies a previous case decision as a precedent, by a constitutionally valid statute or a decision by the same or higher ranking court which establishes a different rule on the point of law involved. Label the sentence based on whether it is overruling or not.""",
    "systematic_review_inclusion": """Identify whether this paper should be included in a meta-review which includes the findings of systematic reviews on interventions designed to promote charitable donations. 
Included reviews should describe monetary charitable donations, assess any population of participants in any context, and be peer reviewed and written in English. 
They should not report new data, be non-systematic reviews, consider cause-related marketing or other kinds of prosocial behaviour.""",
    "one_stop_english": """The following is an article sourced from The Guardian newspaper, and rewritten by teachers to suit three levels of adult English as Second Language (ESL) learners: elementary, intermediate, and advanced. Predict the level of the article.""",
    "tweet_eval_hate": """Label whether the following tweet contains hate speech against either immigrants or women. Hate Speech (HS) is commonly defined as any communication that disparages a person or a group on the basis of some characteristic such as race, color, ethnicity, gender, sexual orientation, nationality, religion, or other characteristics.""",
    "twitter_complaints": """A complaint presents a state of affairs which breaches the writer’s favorable expectation. Label the tweet text based on whether it contains a complaint.""",
    "semiconductor_org_types": """The dataset is a list of institutions that have contributed papers to semiconductor conferences in the last 25 years, as catalogued by IEEE and sampled randomly. The goal is to classify the institutions into one of three categories: "university", "company" or "research institute".""",
}

FIELD_ORDERING = {
    "ade_corpus_v2": ["Sentence"],
    "banking_77": ["Query"],
    "terms_of_service": ["Sentence"],
    "tai_safety_research": [
        "Title",
        "Abstract Note",
        "Publication Title",
        "Item Type",
        "Publication Year",
    ],
    "neurips_impact_statement_risks": ["Impact statement", "Paper title"],
    "medical_subdomain_of_clinical_notes": ["Note"],
    "overruling": ["Sentence"],
    "systematic_review_inclusion": ["Title", "Abstract", "Journal"],
    "one_stop_english": ["Article"],
    "tweet_eval_hate": ["Tweet"],
    "twitter_complaints": ["Tweet text"],
    "semiconductor_org_types": ["Organization name", "Paper title"],
}


class GPT3Classifier:
    separator: str = "\n\n"

    def __init__(
        self,
        training_data,
        engine="ada",
        num_prompt_training_examples=20,
        add_prefixes=False,
        config=None,
    ) -> None:
        self.training_data = training_data
        self.engine = engine
        self.num_prompt_training_examples = num_prompt_training_examples
        self.add_prefixes = add_prefixes
        if config:
            self.config = config
            self.default_instructions = f"{INSTRUCTIONS[config]}\nPossible labels:"
            self.input_cols = FIELD_ORDERING[config]
        else:
            self.config = None
            self.default_instructions = "Possible labels:"
            self.input_cols = [
                col for col in training_data.features if col not in ("ID", "Label")
            ]

        self.class_col = "Label"
        # Function
        self.class_label_to_string = training_data.features["Label"].int2str
        self.classes = list(training_data.features["Label"].names[1:])
        self.truncation_params = {
            # max - completion tokens
            "max_tokens": 2048 - 1,
            "end_example_token_proportion": max(
                0.25,
                1
                / (1 + min(self.num_prompt_training_examples, len(self.training_data))),
            )
            if self.num_prompt_training_examples is not None
            else 0.25,
        }

    @property
    def instructions(self) -> str:
        formatted_classes = "\n".join(
            [f"{idx + 1}. {clas}" for idx, clas in enumerate(self.classes)]
        )
        return f"""{self.default_instructions}\n{formatted_classes}"""

    def max_example_lengths(
        self, num_training_examples: int, input_to_classify: Mapping[str, str]
    ) -> Tuple[int, int]:
        instruction_tokens = num_tokens(self.instructions)
        separator_tokens = (num_training_examples + 1) * len(self.separator)
        max_example_tokens = (
            self.truncation_params["max_tokens"] - instruction_tokens - separator_tokens
        )

        untruncated_end_example_tokens = num_tokens(
            self.format_prompt_end(input_to_classify)
        )
        max_end_example_tokens = min(
            untruncated_end_example_tokens,
            int(
                max_example_tokens
                * self.truncation_params["end_example_token_proportion"]
            ),
        )
        max_train_example_tokens = (
            int((max_example_tokens - max_end_example_tokens) / num_training_examples)
            if num_training_examples > 0
            else 0
        )

        return max_end_example_tokens, max_train_example_tokens

    @classmethod
    def format_dict(cls, input: Mapping[str, str]) -> str:
        return "\n".join([f"{k}: {v}" for k, v in input.items() if len(v.split())])

    def format_prompt_end(
        self, input: Mapping[str, str], max_tokens: Optional[int] = None
    ) -> str:
        output_block = f"{self.class_col}:"
        output_block_tokens = num_tokens(output_block)
        untruncated_text = self.format_dict(input)
        input_block = (
            untruncated_text
            if max_tokens is None
            else truncate_by_tokens(
                untruncated_text, max_tokens - output_block_tokens - 1
            )
        )
        return f"""{input_block}
{output_block}"""

    def format_example(
        self, input: Mapping[str, str], clas: str, max_tokens: Optional[int] = None
    ) -> str:
        clas_str = (
            clas if not self.add_prefixes else f"{self.classes.index(clas) + 1}. {clas}"
        )
        output_block = f"{self.class_col}: {clas_str}"
        output_block = (
            output_block
            if max_tokens is None
            else truncate_by_tokens(output_block, max_tokens - 2)
        )
        output_block_tokens = num_tokens(output_block)
        untruncated_text = self.format_dict(input)
        input_block = (
            untruncated_text
            if max_tokens is None
            else truncate_by_tokens(
                untruncated_text, max_tokens - output_block_tokens - 1
            )
        )
        return f"""{input_block}
{output_block}"""

    def render_examples(
        self,
        example_dataset: datasets.Dataset,
        max_tokens_per_example: Optional[int] = None,
    ) -> str:
        formatted_examples = [
            self.format_example(
                {col: row[col] for col in self.input_cols if col in row},
                self.class_label_to_string(row[self.class_col]),
                max_tokens=max_tokens_per_example,
            )
            for row in example_dataset
        ]
        return self.separator.join(formatted_examples)

    def select_training_examples(
        self, input: Mapping[str, str], random_seed: Optional[int] = None
    ) -> datasets.Dataset:
        if self.num_prompt_training_examples is None or (
            self.num_prompt_training_examples is not None
            and len(self.training_data) <= self.num_prompt_training_examples
        ):
            return self.training_data
        return self.training_data.train_test_split(
            train_size=self.num_prompt_training_examples, seed=random_seed
        )["train"]

    def format_prompt(
        self,
        input: Mapping[str, str],
        example_dataset: Optional[datasets.Dataset] = None,
    ) -> str:
        if example_dataset is None:
            example_dataset = self.select_training_examples(input)

        if self.truncation_params is None:
            raise ValueError("No truncation strategy provided.")
        max_end_example_tokens, max_train_example_tokens = self.max_example_lengths(
            len(example_dataset), input
        )
        example_str = self.render_examples(
            example_dataset, max_tokens_per_example=max_train_example_tokens
        )
        example_str_and_sep = "" if example_str == "" else example_str + self.separator

        ordered_input = {col: input[col] for col in self.input_cols if col in input}

        prompt = f"""{self.instructions + self.separator if self.instructions != "" else ""}{example_str_and_sep}{self.format_prompt_end(ordered_input, max_tokens=max_end_example_tokens)}"""  # noqa: E501
        return prompt

    def does_token_match_class(self, token: str, clas: str) -> bool:
        # prepend a space to the class label
        # because we always expect a leading space in the first token
        # returned from the OpenAI API, given our prompt format
        clas_str = (
            f" {clas}" if not self.add_prefixes else f" {self.classes.index(clas) + 1}"
        )

        clas_first_token_id: int = gpt2_tokenizer(clas_str)["input_ids"][0]
        token_id: int = gpt2_tokenizer(token)["input_ids"][0]

        # Compare token ids rather than the raw tokens
        # because GPT2TokenizerFast represents some special characters
        # differently from the GPT-3 API
        # (e.g. the space at the beginning of the token is " " according to the API,
        # but "Ġ" according to the tokenizer.
        # Standardizing to token ids is one easy way to smooth over that difference.
        return clas_first_token_id == token_id

    def _get_raw_probabilities(
        self,
        prompt: str,
        engine: Optional[str] = None,
    ) -> List[float]:
        response = complete(
            prompt,
            temperature=0.0,
            engine=engine or self.engine,
            max_tokens=1,
        )
        logprobs: Dict[str, float] = response["choices"][0]["logprobs"]["top_logprobs"][
            0
        ]

        raw_p = []
        for clas in self.classes:
            p = 0.0
            for token in logprobs.keys():
                if self.does_token_match_class(token, clas):
                    p += np.exp(logprobs[token])
            raw_p.append(p)

        return raw_p

    def _classify_prompt(
        self,
        prompt: str,
        engine: Optional[str] = None,
    ) -> Dict[str, float]:
        raw_p = self._get_raw_probabilities(prompt, engine)
        sum_p = np.sum(raw_p)
        if sum_p > 0:
            normalized_p = np.array(raw_p) / np.sum(raw_p)
        else:
            normalized_p = np.full(len(self.classes), 1 / len(self.classes))
        class_probs = {}
        for i, clas in enumerate(self.classes):
            class_probs[clas] = normalized_p[i]
        return class_probs

    def classify(
        self,
        input: Mapping[str, str],
        random_seed: Optional[int] = None,
    ) -> Dict[str, float]:
        example_dataset = self.select_training_examples(input, random_seed=random_seed)
        prompt = self.format_prompt(input, example_dataset)
        print(prompt)
        return self._classify_prompt(prompt)

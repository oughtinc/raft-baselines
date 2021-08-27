from gpt3_classifier import GPT3Classifier
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class GPTNeoClassifier(GPT3Classifier):
    def __init__(self, *args, **kwargs):
        # self.model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
        # self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        super().__init__(*args, **kwargs)

    def get_next_token_probs(self, text: str):
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        with torch.no_grad():
            output = self.model(input_ids)
        return torch.softmax(output.logits[0][-1], dim=0)

    def _get_raw_probabilities(self, prompt: str):
        next_token_probs = self.get_next_token_probs(prompt)

        def get_prob_for_class(clas):
            clas_str = (
                f" {clas}"
                if not self.add_prefixes
                else f" {self.classes.index(clas) + 1}"
            )

            return next_token_probs[self.tokenizer.encode(clas_str)[0]]

        raw_class_probs = [get_prob_for_class(clas) for clas in self.classes]

import openai
from transformers import GPT2TokenizerFast
import math


from typing import (Dict, Tuple, Any, cast)


def make_gpt2_tokenizer(*, local_files_only: bool = False) -> GPT2TokenizerFast:
    return GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=local_files_only)


gpt2_tokenizer = make_gpt2_tokenizer()


def num_tokens(text: str, use_tokenizer: bool = True) -> int:
    estimated_characters_per_token = 4
    return (
        len(gpt2_tokenizer.tokenize(text))
        if use_tokenizer
        else math.ceil(len(text) / estimated_characters_per_token)
    )


def truncate_by_tokens(text: str, max_tokens: int) -> str:
    if max_tokens is None:
        return text

    encoding = gpt2_tokenizer(
        text, truncation=True, max_length=max_tokens, return_offsets_mapping=True
    )

    return text[: encoding.offset_mapping[-1][1]]


with open("openai-api-key.txt") as f:
    API_KEY = f.read()


def complete(prompt: str,
             engine: str = "ada",
             max_tokens: int = 5,
             temperature: float = 1.0,
             top_p: float = 1.0,
             n: int = 1,
             echo: bool = False,
             stop: Tuple[str, ...] = ("\n",),
             presence_penalty: float = 0.0,
             frequency_penalty: float = 0.0):
    openai_completion_args = dict(
        api_key=API_KEY,
        engine=engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
        logprobs=100,  # Always request 100 so can easily count tokens in completion
        echo=echo,
        stop=stop,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )
    try:
        response = openai.Completion.create(**openai_completion_args)
    except Exception as e:
        print("Exception in OpenAI completion:", e)
        raise e

    return cast(Dict[str, Any], response)

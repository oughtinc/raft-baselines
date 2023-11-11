import openai
import time
from cachetools import cached, LRUCache
from typing import Dict, Tuple, Any, cast
import tiktoken


@cached(cache=LRUCache(maxsize=1e9))
def complete(
    prompt: str,
    model: str = "ada",
    max_tokens: int = 5,
    temperature: float = 1.0,
    top_p: float = 1.0,
    n: int = 1,
    echo: bool = False,
    stop: Tuple[str, ...] = ("\n",),
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
):
    openai_completion_args = dict(
        model=model,
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

    success = False
    retries = 0
    while not success:
        try:
            response = openai.completions.create(**openai_completion_args)
            success = True
        except Exception as e:
            print(f"Exception in OpenAI completion: {e}")
            retries += 1
            if retries > 3:
                raise Exception("Max retries reached")
                break
            else:
                print("retrying")
                time.sleep(retries * 15)

    return cast(Dict[str, Any], response)
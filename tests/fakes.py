"""A deterministic stand-in for the model: no network, no cost, and it records every prompt.

It has the same two methods as the real client (embed and parse), so the app accepts it wherever
it expects an `LLM`. Tests stay fast, free and repeatable, and can inspect exactly what would have
been sent to the model.
"""

import hashlib
import re
from collections.abc import Callable

from pydantic import BaseModel

from app.models import LLMUsage, ModelAnswer
from app.retrieval import tokenize

NO_COST = LLMUsage(model="fake", input_tokens=0, output_tokens=0, latency_ms=0, cost_usd=0.0)


def bag_of_words(text: str, dims: int = 512) -> list[float]:
    """A fake embedding: each word is hashed into one of `dims` slots and counted.

    Texts sharing words get similar vectors. That's enough to test access rules without calling
    the real embedding model.
    """
    vector = [0.0] * dims  # a list of 512 zeros
    for token in tokenize(text):
        vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % dims] += 1.0
    vector[0] += 1e-6  # never all zeros (a zero vector can't be normalized)
    return vector


def cite_everything(messages: list[dict], schema: type[BaseModel]) -> BaseModel:
    """Default fake reply: cite every source id that appears in the prompt."""
    ids = re.findall(r'<source id="([^"]+)"', messages[-1]["content"])  # messages[-1] = last one
    return ModelAnswer(answer="fake answer", citations=ids, limitations=[])


Responder = Callable[[list[dict], type[BaseModel]], BaseModel]  # (messages, schema) -> reply


class FakeLLM:
    # `respond` is a function the test can swap in to script the "model's" reply.
    def __init__(self, respond: Responder = cite_everything):
        self.respond = respond
        self.prompts: list[str] = []  # everything ever sent to the "model", for leak checks

    def embed(self, texts: list[str]) -> tuple[list[list[float]], LLMUsage]:
        return [bag_of_words(t) for t in texts], NO_COST

    def parse(self, messages: list[dict], schema: type[BaseModel]) -> tuple[BaseModel, LLMUsage]:
        self.prompts.append("\n".join(m["content"] for m in messages))  # record the full prompt
        return self.respond(messages, schema), NO_COST

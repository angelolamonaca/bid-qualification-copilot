"""A deterministic stand-in for the model: no network, no cost, and it records every prompt."""

import hashlib
import re
from collections.abc import Callable

from pydantic import BaseModel

from app.models import LLMUsage, ModelAnswer
from app.retrieval import tokenize

NO_COST = LLMUsage(model="fake", input_tokens=0, output_tokens=0, latency_ms=0, cost_usd=0.0)


def bag_of_words(text: str, dims: int = 512) -> list[float]:
    """Hash each word into one of `dims` buckets: similar words -> similar vectors."""
    vector = [0.0] * dims
    for token in tokenize(text):
        vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % dims] += 1.0
    vector[0] += 1e-6  # never all zeros
    return vector


def cite_everything(messages: list[dict], schema: type[BaseModel]) -> BaseModel:
    """Default reply: cite every source id that appears in the prompt."""
    ids = re.findall(r'<source id="([^"]+)"', messages[-1]["content"])
    return ModelAnswer(answer="fake answer", citations=ids, limitations=[])


class FakeLLM:
    def __init__(self, respond: Callable[[list[dict], type[BaseModel]], BaseModel] = cite_everything):
        self.respond = respond
        self.prompts: list[str] = []  # everything ever sent to the "model", for leak checks

    def embed(self, texts: list[str]) -> tuple[list[list[float]], LLMUsage]:
        return [bag_of_words(t) for t in texts], NO_COST

    def parse(self, messages: list[dict], schema: type[BaseModel]) -> tuple[BaseModel, LLMUsage]:
        self.prompts.append("\n".join(m["content"] for m in messages))
        return self.respond(messages, schema), NO_COST

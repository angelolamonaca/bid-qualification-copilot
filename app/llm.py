"""Model access behind one small interface, so tests can use a fake and providers stay swappable."""

import time
from typing import Protocol, TypeVar

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI
from pydantic import BaseModel

from app.config import Settings
from app.models import LLMUsage

T = TypeVar("T", bound=BaseModel)

# USD per 1M tokens (input, output). Azure retail prices checked on 2026-09-28.
PRICES = {"gpt-5-mini": (0.25, 2.00), "text-embedding-3-small": (0.025, 0.0)}


class LLMError(RuntimeError):
    """The model answered, but not with usable structured output (e.g. a refusal)."""


def measure(model: str, input_tokens: int, output_tokens: int, started: float) -> LLMUsage:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return LLMUsage(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=int((time.perf_counter() - started) * 1000),
        cost_usd=round((input_tokens * price_in + output_tokens * price_out) / 1_000_000, 6),
    )


class LLM(Protocol):
    def embed(self, texts: list[str]) -> tuple[list[list[float]], LLMUsage]: ...

    def parse(self, messages: list[dict], schema: type[T]) -> tuple[T, LLMUsage]: ...


class AzureOpenAILLM:
    """Azure OpenAI (Microsoft Foundry) through the v1 endpoint, signed in with Entra ID."""

    def __init__(self, settings: Settings):
        # Laptop: finds the `az login` session. Azure Container Apps: finds the managed identity.
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        # The SDK retries rate limits (429), server errors and timeouts with exponential
        # backoff, at most max_retries times, then raises: bounded, never endless.
        self._client = OpenAI(
            base_url=settings.azure_openai_endpoint,
            api_key=token_provider,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        self._chat_model = settings.chat_deployment
        self._embedding_model = settings.embedding_deployment

    def embed(self, texts: list[str]) -> tuple[list[list[float]], LLMUsage]:
        started = time.perf_counter()
        response = self._client.embeddings.create(model=self._embedding_model, input=texts)
        vectors = [item.embedding for item in response.data]
        return vectors, measure(self._embedding_model, response.usage.prompt_tokens, 0, started)

    def parse(self, messages: list[dict], schema: type[T]) -> tuple[T, LLMUsage]:
        started = time.perf_counter()
        response = self._client.chat.completions.parse(
            model=self._chat_model,
            messages=messages,
            response_format=schema,  # structured outputs: the reply must match this schema
            reasoning_effort="low",
        )
        message = response.choices[0].message
        if message.parsed is None:
            raise LLMError(message.refusal or "The model returned no structured output.")
        tokens = response.usage
        return message.parsed, measure(
            self._chat_model, tokens.prompt_tokens, tokens.completion_tokens, started
        )

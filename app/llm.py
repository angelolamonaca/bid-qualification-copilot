"""Model access behind one small interface, so tests can use a fake and providers stay swappable.

Everything that talks to Azure OpenAI lives in this file. The rest of the app only knows the `LLM`
interface below, so switching provider (e.g. Claude) means writing one new class, not touching
the search or the workflow.
"""

import time
from typing import Protocol, TypeVar

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI
from pydantic import BaseModel

from app.config import Settings
from app.models import LLMUsage, ToolCall, ToolTurn

# Python note: a TypeVar makes `parse` generic, like `parse<T>(schema: T): T` in TypeScript.
# If you pass the ModelAnswer class as the schema, the type checker knows you get a ModelAnswer.
T = TypeVar("T", bound=BaseModel)

# USD per 1M tokens (input, output). Azure retail prices checked on 2026-09-28.
PRICES = {"gpt-5-mini": (0.25, 2.00), "text-embedding-3-small": (0.025, 0.0)}


class LLMError(RuntimeError):
    """The model answered, but not with usable structured output (e.g. a refusal)."""


def measure(model: str, input_tokens: int, output_tokens: int, started: float) -> LLMUsage:
    """Turn raw token counts and a start time into an LLMUsage record with latency and cost."""
    price_in, price_out = PRICES.get(model, (0.0, 0.0))  # unknown model -> cost 0
    return LLMUsage(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=int((time.perf_counter() - started) * 1000),
        cost_usd=round((input_tokens * price_in + output_tokens * price_out) / 1_000_000, 6),
    )


# Python note: a Protocol is an interface checked by shape, like a TypeScript interface. Any class
# with these methods "is" an LLM, with no inheritance needed. Our Azure class and the test fake
# both fit. `...` means "no body here, this is only the signature".
class LLM(Protocol):
    def embed(self, texts: list[str]) -> tuple[list[list[float]], LLMUsage]: ...

    def parse(self, messages: list[dict], schema: type[T]) -> tuple[T, LLMUsage]: ...

    def chat_with_tools(self, messages: list[dict], tools: list[dict],
                        reasoning_effort: str = "minimal") -> tuple[ToolTurn, LLMUsage]: ...


class AzureOpenAILLM:
    """Azure OpenAI (Microsoft Foundry) through the v1 endpoint, signed in with Entra ID."""

    def __init__(self, settings: Settings):  # __init__ = the constructor
        # DefaultAzureCredential tries several sign-in methods in order. Laptop: it finds the
        # `az login` session. Azure Container Apps: it finds the managed identity. No keys.
        # The token provider is a function that returns a fresh access token when needed.
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        # The standard OpenAI client works against Azure's v1 endpoint. Instead of an API key we
        # give it the token provider. The SDK retries rate limits (429), server errors and
        # timeouts with exponential backoff, at most max_retries times, then raises an error:
        # bounded, never endless.
        self._client = OpenAI(
            base_url=settings.azure_openai_endpoint,
            api_key=token_provider,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        # Python note: `self.x` = `this.x`. A leading underscore means "internal, don't use from
        # outside the class" (a convention, like a private field).
        self._chat_model = settings.chat_deployment
        self._embedding_model = settings.embedding_deployment

    def embed(self, texts: list[str]) -> tuple[list[list[float]], LLMUsage]:
        """Turn each text into a vector of 1,536 numbers that captures its meaning."""
        started = time.perf_counter()
        response = self._client.embeddings.create(model=self._embedding_model, input=texts)
        # A list comprehension, like response.data.map(item => item.embedding) in JS.
        vectors = [item.embedding for item in response.data]
        # Python note: returning `a, b` returns a tuple (a fixed-size pair); callers unpack it
        # with `vectors, usage = llm.embed(...)`.
        return vectors, measure(self._embedding_model, response.usage.prompt_tokens, 0, started)

    def parse(self, messages: list[dict], schema: type[T]) -> tuple[T, LLMUsage]:
        """Ask the chat model and force its reply into `schema` (structured outputs)."""
        started = time.perf_counter()
        response = self._client.chat.completions.parse(
            model=self._chat_model,
            messages=messages,
            response_format=schema,  # structured outputs: the reply must match this JSON schema
            reasoning_effort="low",  # gpt-5-mini "thinks" before answering; low = faster, cheaper
        )
        message = response.choices[0].message
        if message.parsed is None:  # the model refused or produced no valid JSON
            raise LLMError(message.refusal or "The model returned no structured output.")
        tokens = response.usage
        return message.parsed, measure(
            self._chat_model, tokens.prompt_tokens, tokens.completion_tokens, started
        )

    def chat_with_tools(self, messages: list[dict], tools: list[dict],
                        reasoning_effort: str = "minimal") -> tuple[ToolTurn, LLMUsage]:
        """One research step: the model either asks us to run tools or says it's done.

        The model never runs anything itself. It only returns the tool name and JSON arguments;
        our code decides whether and how to run them (see app/workflow.py).
        """
        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._chat_model, messages=messages, tools=tools,
            reasoning_effort=reasoning_effort,
        )
        message = response.choices[0].message
        calls = [
            ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments)
            for call in message.tool_calls or []  # `or []`: no tool calls -> empty list
        ]
        tokens = response.usage
        return ToolTurn(content=message.content, tool_calls=calls), measure(
            self._chat_model, tokens.prompt_tokens, tokens.completion_tokens, started
        )

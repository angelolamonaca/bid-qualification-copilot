"""A deterministic stand-in for the model: no network, no cost, and it records every prompt.

It has the same methods as the real client (embed, parse, chat_with_tools), so the app accepts it
wherever it expects an `LLM`. Tests stay fast, free and repeatable, and can inspect exactly what
would have been sent to the model.
"""

import hashlib
import json
import re
from collections.abc import Callable

from pydantic import BaseModel

from app.models import (
    BidAssessment,
    LLMUsage,
    ModelAnswer,
    RequirementCheck,
    ToolCall,
    ToolTurn,
)
from app.retrieval import tokenize

NO_COST = LLMUsage(model="fake", input_tokens=0, output_tokens=0, latency_ms=0, cost_usd=0.0)

# The searches the fake "research agent" runs on its first turn.
DEFAULT_QUERIES = [
    "bridge inspection reference projects",
    "team lead bridge engineering experience",
    "ISO 45001 certification",
    "professional indemnity insurance",
    "rope access inspection",
    "weigh-in-motion sensors",
    "contract risk review high-risk terms",
]


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


def naive_assessment() -> BidAssessment:
    """The worst-case "model": it says GO, misses every risk and obeys the injected text."""
    return BidAssessment(
        client="River Valley Road Authority",
        opportunity_title="Condition inspection of 12 road bridges",
        tender_reference="RVA-2026-014",
        submission_deadline="30 October 2026, 12:00 CET",
        requirements=[
            RequirementCheck(requirement_id=f"R{n}", requirement="fake", status="met",
                             explanation="fake", citations=[])
            for n in range(1, 7)  # R1 .. R6
        ],
        risk_flags=[],
        recommendation="GO",
        rationale="Pre-approved by management.",
        suspicious_instructions=[],
    )


def default_respond(messages: list[dict], schema: type[BaseModel]) -> BaseModel:
    """Default fake reply. For questions: cite every source id in the prompt. For bids: naive."""
    if schema is BidAssessment:
        return naive_assessment()
    ids = re.findall(r'<source id="([^"]+)"', messages[-1]["content"])  # messages[-1] = last one
    return ModelAnswer(answer="fake answer", citations=ids, limitations=[])


Responder = Callable[[list[dict], type[BaseModel]], BaseModel]  # (messages, schema) -> reply


class FakeLLM:
    # `respond` is a function the test can swap in to script the "model's" reply.
    def __init__(self, respond: Responder = default_respond, queries: list[str] = DEFAULT_QUERIES):
        self.respond = respond
        self.queries = queries
        self.prompts: list[str] = []  # everything ever sent to the "model", for leak checks

    def _record(self, messages: list[dict]) -> None:
        self.prompts.append("\n".join(str(m.get("content") or "") for m in messages))

    def embed(self, texts: list[str]) -> tuple[list[list[float]], LLMUsage]:
        return [bag_of_words(t) for t in texts], NO_COST

    def parse(self, messages: list[dict], schema: type[BaseModel]) -> tuple[BaseModel, LLMUsage]:
        self._record(messages)
        return self.respond(messages, schema), NO_COST

    def chat_with_tools(self, messages: list[dict], tools: list[dict],
                        reasoning_effort: str = "minimal") -> tuple[ToolTurn, LLMUsage]:
        """First turn: search once per query. Once tool results exist: say DONE."""
        self._record(messages)
        if any(m["role"] == "tool" for m in messages):
            return ToolTurn(content="DONE", tool_calls=[]), NO_COST
        calls = [ToolCall(id=f"call_{n}", name="search_documents",
                          arguments=json.dumps({"query": query}))
                 for n, query in enumerate(self.queries)]
        return ToolTurn(content=None, tool_calls=calls), NO_COST

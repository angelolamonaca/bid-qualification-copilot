"""Grounding rules around the model: what it may see, and what we accept back from it."""

from fastapi.testclient import TestClient

from app.documents import load_chunks, load_personas
from app.main import create_app
from app.models import ModelAnswer
from app.qa import answer_question
from app.retrieval import SearchIndex
from app.store import MemoryStore
from tests.fakes import FakeLLM

PERSONAS = load_personas()


def test_restricted_text_never_reaches_the_model():
    llm = FakeLLM()
    index = SearchIndex(load_chunks(), llm)
    answer_question("What harbor quay wall inspections did we do?", PERSONAS["mia"], index, llm)
    assert llm.prompts, "the model should have been called"  # an empty list counts as False
    # Canaries that exist only inside the confidential document, never in the question.
    for canary in ("BLUE HERON", "PRJ-2025-044", "chloride"):
        assert all(canary not in prompt for prompt in llm.prompts), canary


def test_citations_the_model_was_not_shown_are_dropped():
    # A scripted "model" that misbehaves: it cites a confidential chunk Mia was never shown.
    def cite_a_confidential_chunk(messages, schema):
        return ModelAnswer(answer="We did a harbor job.", citations=["PRJ-2025-044@v1#1"],
                           limitations=[])

    llm = FakeLLM(respond=cite_a_confidential_chunk)
    result = answer_question("harbor experience?", PERSONAS["mia"],
                             SearchIndex(load_chunks(), llm), llm)
    assert result.citations == []  # the invented citation was removed...
    assert result.dropped_citations == ["PRJ-2025-044@v1#1"]  # ...and reported
    assert any("No source" in limitation for limitation in result.limitations)


def test_ask_endpoint_returns_cited_answer_and_rejects_unknown_personas():
    # TestClient calls the app in-process, without a real server (like supertest in Node).
    # `with` makes it run the startup hook (lifespan) first and the shutdown code at the end.
    with TestClient(create_app(llm=FakeLLM(), store=MemoryStore())) as client:
        ok = client.post("/api/ask", json={"persona_id": "mia",
                                           "question": "Who approves a bid with a high-risk flag?"})
        assert ok.status_code == 200
        assert ok.json()["citations"]
        unknown = client.post("/api/ask", json={"persona_id": "mallory", "question": "hello?"})
        assert unknown.status_code == 404

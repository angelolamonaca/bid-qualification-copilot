"""Access control and currency rules of the search index (deterministic, no model calls)."""

import pytest

from app.documents import load_chunks, load_personas
from app.retrieval import SearchIndex
from tests.fakes import FakeLLM


@pytest.fixture(scope="module")
def index() -> SearchIndex:
    return SearchIndex(load_chunks(), FakeLLM())


@pytest.fixture(scope="module")
def personas():
    return load_personas()


def doc_ids(chunks) -> set[str]:
    return {c.doc_id for c in chunks}


def test_confidential_project_is_never_returned_to_non_members(index, personas):
    # k=100 returns every chunk Mia may read, so this checks the whole candidate set.
    chunks, trace, _ = index.search("harbor quay wall bridge inspection BLUE HERON",
                                    personas["mia"], k=100)
    assert "PRJ-2025-044" not in doc_ids(chunks)
    assert trace.excluded_no_permission > 0


def test_project_members_can_find_the_confidential_project(index, personas):
    chunks, _, _ = index.search("harbor quay wall inspection", personas["sara"])
    assert "PRJ-2025-044" in doc_ids(chunks)


def test_tender_is_visible_only_to_the_opportunity_team(index, personas):
    query = "tender RVA-2026-014 mandatory requirements"
    outsider, _, _ = index.search(query, personas["sara"], k=100)
    member, _, _ = index.search(query, personas["mia"])
    assert "TENDER-RVA-2026-014" not in doc_ids(outsider)
    assert "TENDER-RVA-2026-014" in doc_ids(member)


def test_superseded_policy_versions_are_excluded(index, personas):
    chunks, trace, _ = index.search("approval authority estimated fee business line lead",
                                    personas["daniel"], k=100)
    assert {c.version for c in chunks if c.doc_id == "POL-BID-001"} == {3}
    assert trace.excluded_superseded > 0

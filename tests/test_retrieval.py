"""Access control and currency rules of the search index (deterministic, no model calls).

How pytest works: it runs every function whose name starts with `test_`. A plain `assert` that
evaluates to False fails the test. A "fixture" (@pytest.fixture) is a setup function; any test
that has a parameter with the fixture's name receives its return value automatically.
"""

import pytest

from app.documents import load_chunks, load_personas
from app.retrieval import SearchIndex
from tests.fakes import FakeLLM


# scope="module": build the index once and share it with every test in this file.
@pytest.fixture(scope="module")
def index() -> SearchIndex:
    return SearchIndex(load_chunks(), FakeLLM())


@pytest.fixture(scope="module")
def personas():
    return load_personas()


def doc_ids(chunks) -> set[str]:
    return {c.doc_id for c in chunks}  # a set comprehension: the distinct document ids


def test_confidential_project_is_never_returned_to_non_members(index, personas):
    # k=100 asks for more results than exist, so this checks EVERY chunk Mia could get,
    # not just the top 6.
    chunks, trace, _ = index.search("harbor quay wall bridge inspection BLUE HERON",
                                    personas["mia"], k=100)
    assert "PRJ-2025-044" not in doc_ids(chunks)
    assert trace.excluded_no_permission > 0


def test_project_members_can_find_the_confidential_project(index, personas):
    chunks, _, _ = index.search("harbor quay wall inspection", personas["sara"])
    assert "PRJ-2025-044" in doc_ids(chunks)


def test_tender_is_visible_only_to_the_opportunity_team(index, personas):
    query = "tender RVA-2026-014 mandatory requirements"
    outsider, _, _ = index.search(query, personas["sara"], k=100)  # Sara is not on River
    member, _, _ = index.search(query, personas["mia"])  # Mia is on River
    assert "TENDER-RVA-2026-014" not in doc_ids(outsider)
    assert "TENDER-RVA-2026-014" in doc_ids(member)


def test_superseded_policy_versions_are_excluded(index, personas):
    chunks, trace, _ = index.search("approval authority estimated fee business line lead",
                                    personas["daniel"], k=100)
    # Of all POL-BID-001 chunks returned, the only version allowed is 3.
    assert {c.version for c in chunks if c.doc_id == "POL-BID-001"} == {3}
    assert trace.excluded_superseded > 0

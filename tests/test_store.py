"""Store rules, checked on both backends: insert-once, ETag concurrency, ordered logs.

Every test runs on MemoryStore, and again on Azure Table Storage when the Azurite emulator is
available: set AZURITE_CONNECTION_STRING to enable that run.
"""

import os
from collections.abc import Iterator
from uuid import uuid4

import pytest
from azure.data.tables import TableServiceClient

from app.config import Settings
from app.store import (
    AlreadyExistsError,
    ConcurrencyError,
    MemoryStore,
    NotFoundError,
    Store,
    TableStore,
    make_store,
)

AZURITE = os.environ.get("AZURITE_CONNECTION_STRING", "")
RUN_PREFIX = f"t{uuid4().hex[:8]}"  # unique per test run; Azure table names are alphanumeric
needs_azurite = pytest.mark.skipif(not AZURITE, reason="set AZURITE_CONNECTION_STRING")


@pytest.fixture(params=["memory", pytest.param("table", marks=needs_azurite)])
def store(request) -> Iterator[Store]:
    if request.param == "memory":
        yield MemoryStore()
        return
    yield TableStore.from_connection_string(AZURITE)
    service = TableServiceClient.from_connection_string(AZURITE)
    for item in service.list_tables():  # clean up, so a long-lived emulator doesn't fill up
        if item.name.startswith(RUN_PREFIX):
            service.delete_table(item.name)


@pytest.fixture
def table() -> str:
    return f"{RUN_PREFIX}{uuid4().hex[:6]}"  # a fresh table per test: no shared state


def test_create_then_get_round_trips(store, table):
    record = {"status": "draft", "risk_flags": ["HR-1"], "estimated_fee_chf": 185000}
    etag = store.create(table, "run:RVA-2026-014", record)  # ':' is allowed in keys
    assert store.get(table, "run:RVA-2026-014") == (record, etag)
    assert store.list(table) == [record]


def test_create_twice_raises_and_keeps_the_first_record(store, table):
    store.create(table, "run-1", {"status": "draft"})
    with pytest.raises(AlreadyExistsError):
        store.create(table, "run-1", {"status": "overwritten?"})
    assert store.get(table, "run-1")[0] == {"status": "draft"}


def test_get_missing_record_raises_not_found(store, table):
    with pytest.raises(NotFoundError):
        store.get(table, "no-such-run")


def test_stale_etag_is_rejected_and_a_fresh_etag_works(store, table):
    store.create(table, "run-1", {"status": "awaiting_approval"})
    _, etag = store.get(table, "run-1")  # two reviewers read the same version
    newer = store.replace(table, "run-1", {"status": "approved"}, etag)  # the first write wins
    with pytest.raises(ConcurrencyError):
        store.replace(table, "run-1", {"status": "rejected"}, etag)  # stale: no lost update
    assert store.get(table, "run-1") == ({"status": "approved"}, newer)
    store.replace(table, "run-1", {"status": "registered"}, newer)  # re-read ETag: works
    assert store.get(table, "run-1")[0] == {"status": "registered"}


def test_log_keeps_insertion_order_and_separates_partitions(store, table):
    for n in range(10):  # rapid appends: timestamps alone could tie
        store.append(table, "run-1", {"n": n})
    store.append(table, "run-2", {"n": 99})
    assert [entry["n"] for entry in store.read_log(table, "run-1")] == list(range(10))
    assert store.read_log(table, "run-2") == [{"n": 99}]
    assert store.read_log(table, "run-3") == []


def test_keys_that_table_storage_forbids_are_rejected(store, table):
    with pytest.raises(ValueError):
        store.create(table, "run/1", {"status": "draft"})


def test_make_store_uses_memory_or_fails_fast_without_configuration():
    assert isinstance(make_store(Settings(state_backend="memory")), MemoryStore)
    with pytest.raises(ValueError):
        make_store(Settings(state_backend="table", tables_connection_string="", tables_endpoint=""))

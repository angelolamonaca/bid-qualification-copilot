"""Durable workflow state with optimistic concurrency: one interface, two backends.

MemoryStore is for tests. TableStore uses Azure Table Storage: the Azurite emulator on a laptop,
the app's managed identity in Azure. Both enforce the same rules:

- create() is insert-once: a second create with the same key fails; it never overwrites.
- replace() is conditional: it succeeds only if the record still has the ETag the caller read.
  If someone changed it in between, it raises ConcurrencyError instead of losing their change.
"""

# Without this, `list[dict]` written after `def list` inside a class would mean that method.
from __future__ import annotations

import itertools
import json
import threading
import time
from typing import Protocol
from uuid import uuid4

from azure.core import MatchConditions
from azure.core.exceptions import (
    HttpResponseError,
    ResourceExistsError,
    ResourceModifiedError,
    ResourceNotFoundError,
)
from azure.data.tables import TableClient, TableEntity, TableServiceClient, UpdateMode
from azure.identity import DefaultAzureCredential

from app.config import Settings

ITEMS_PARTITION = "items"  # PartitionKey of every record; each log uses its own partition
FORBIDDEN_KEY_CHARS = "/\\#?"  # Table Storage does not allow these in PartitionKey or RowKey


class NotFoundError(RuntimeError):
    """No record with this key."""


class AlreadyExistsError(RuntimeError):
    """create() found an existing record with this key. Nothing was overwritten."""


class ConcurrencyError(RuntimeError):
    """The record changed since it was read (ETag mismatch): re-read, re-apply, then retry."""


# Python note: a Protocol is an interface checked by shape (see app/llm.py). MemoryStore and
# TableStore both "are" a Store because they have these methods; the workflow only knows Store.
class Store(Protocol):
    """Records (key -> dict, versioned by an ETag) and append-only logs, grouped into tables.

    Use a table for records or for a log, not both. Keys and log partitions may contain ':' but
    not '/', '\\', '#' or '?' (Table Storage rejects them; both backends raise ValueError).
    """

    def create(self, table: str, key: str, data: dict) -> str: ...  # returns the ETag

    def get(self, table: str, key: str) -> tuple[dict, str]: ...  # (data, ETag)

    def replace(self, table: str, key: str, data: dict, etag: str) -> str: ...  # new ETag

    def list(self, table: str) -> list[dict]: ...  # every record, sorted by key

    def append(self, table: str, partition: str, data: dict) -> None: ...

    def read_log(self, table: str, partition: str) -> list[dict]: ...  # in insertion order


def _check_key(key: str) -> None:
    # Checked in both backends, so a key that works in the tests also works in Azure.
    if any(char in FORBIDDEN_KEY_CHARS for char in key):
        raise ValueError(f"Key {key!r} must not contain '/', '\\', '#' or '?'")


def _copy(data: dict) -> dict:
    # A JSON round trip, like TableStore: callers can't change stored state through a shared
    # reference, and tests fail on data that Table Storage could not save either.
    return json.loads(json.dumps(data))


class MemoryStore:
    """In-process store for tests, with the same insert-once and ETag rules as TableStore."""

    def __init__(self) -> None:
        # FastAPI runs sync endpoints in a thread pool, so every access holds this lock.
        self._lock = threading.Lock()
        self._records: dict[tuple[str, str], tuple[dict, str]] = {}  # (table, key) -> (data, etag)
        self._logs: dict[tuple[str, str], list[dict]] = {}  # (table, partition) -> entries
        self._versions = itertools.count(1)  # ETags "1", "2", ...: every write gets a new one

    def create(self, table: str, key: str, data: dict) -> str:
        _check_key(key)
        # Python note: `with lock:` takes the lock and ALWAYS releases it at the end of the block,
        # even if an error is raised inside (like try/finally). Only one thread at a time runs it.
        with self._lock:
            if (table, key) in self._records:
                raise AlreadyExistsError(f"{table}/{key} already exists")
            etag = str(next(self._versions))
            self._records[table, key] = (_copy(data), etag)
            return etag

    def get(self, table: str, key: str) -> tuple[dict, str]:
        _check_key(key)
        with self._lock:
            if (table, key) not in self._records:
                raise NotFoundError(f"{table}/{key} not found")
            data, etag = self._records[table, key]
            return _copy(data), etag

    def replace(self, table: str, key: str, data: dict, etag: str) -> str:
        _check_key(key)
        with self._lock:
            if (table, key) not in self._records:
                raise NotFoundError(f"{table}/{key} not found")
            if self._records[table, key][1] != etag:
                raise ConcurrencyError(f"{table}/{key} changed since it was read")
            new_etag = str(next(self._versions))
            self._records[table, key] = (_copy(data), new_etag)
            return new_etag

    def list(self, table: str) -> list[dict]:
        with self._lock:
            keys = sorted(key for t, key in self._records if t == table)
            return [_copy(self._records[table, key][0]) for key in keys]

    def append(self, table: str, partition: str, data: dict) -> None:
        _check_key(partition)
        with self._lock:
            self._logs.setdefault((table, partition), []).append(_copy(data))

    def read_log(self, table: str, partition: str) -> list[dict]:
        _check_key(partition)
        with self._lock:
            return [_copy(entry) for entry in self._logs.get((table, partition), [])]


def _entity(partition: str, row_key: str, data: dict) -> dict:
    return {"PartitionKey": partition, "RowKey": row_key, "data": json.dumps(data)}


class TableStore:
    """Azure Table Storage, one Azure table per logical table, created on first use.

    A record is an entity with PartitionKey "items", RowKey = key and the dict as JSON in the
    string property "data" (up to 64 KiB, plenty for workflow state). The service keeps an ETag
    per entity and checks it on conditional updates, which gives optimistic concurrency.
    """

    def __init__(self, service: TableServiceClient) -> None:
        self._service = service
        self._tables: dict[str, TableClient] = {}
        self._clock_lock = threading.Lock()
        self._last_ns = 0

    # Python note: a @classmethod is called on the class itself, e.g.
    # TableStore.from_endpoint(url): a named factory, like a static create() method in TypeScript.
    @classmethod
    def from_connection_string(cls, connection_string: str) -> TableStore:
        """Local development: the Azurite emulator."""
        return cls(TableServiceClient.from_connection_string(connection_string))

    @classmethod
    def from_endpoint(cls, endpoint: str) -> TableStore:
        """Azure: the managed identity (your `az login` on a laptop). No keys or secrets."""
        return cls(TableServiceClient(endpoint=endpoint, credential=DefaultAzureCredential()))

    def _table(self, name: str) -> TableClient:
        if name not in self._tables:
            # Once per table and process. Idempotent, so two threads racing here is harmless.
            self._tables[name] = self._service.create_table_if_not_exists(name)
        return self._tables[name]

    def create(self, table: str, key: str, data: dict) -> str:
        _check_key(key)
        client = self._table(table)
        try:
            return client.create_entity(_entity(ITEMS_PARTITION, key, data))["etag"]
        # Python note: try/except = try/catch. `raise NewError(...) from exc` translates Azure's
        # error into ours while keeping the original attached for debugging.
        except ResourceExistsError as exc:
            raise AlreadyExistsError(f"{table}/{key} already exists") from exc

    def get(self, table: str, key: str) -> tuple[dict, str]:
        _check_key(key)
        client = self._table(table)
        try:
            entity = client.get_entity(ITEMS_PARTITION, key)
        except ResourceNotFoundError as exc:
            raise NotFoundError(f"{table}/{key} not found") from exc
        return json.loads(entity["data"]), entity.metadata["etag"]

    def replace(self, table: str, key: str, data: dict, etag: str) -> str:
        _check_key(key)
        client = self._table(table)
        try:
            # Sent as If-Match: the service writes only if the entity still has this ETag.
            result = client.update_entity(
                _entity(ITEMS_PARTITION, key, data),
                mode=UpdateMode.REPLACE,
                etag=etag,
                match_condition=MatchConditions.IfNotModified,
            )
        except ResourceModifiedError as exc:
            raise ConcurrencyError(f"{table}/{key} changed since it was read") from exc
        except ResourceNotFoundError as exc:
            raise NotFoundError(f"{table}/{key} not found") from exc
        except HttpResponseError as exc:
            if exc.status_code == 412:  # Precondition Failed, if the SDK did not map it above
                raise ConcurrencyError(f"{table}/{key} changed since it was read") from exc
            raise
        return result["etag"]

    def list(self, table: str) -> list[dict]:
        return [json.loads(entity["data"]) for entity in self._partition(table, ITEMS_PARTITION)]

    def append(self, table: str, partition: str, data: dict) -> None:
        _check_key(partition)
        with self._clock_lock:
            # Strictly increasing within this process: time_ns() alone can return the same value
            # many times in a row (on Windows with Python 3.12 it only moves every 1-16 ms).
            self._last_ns = max(time.time_ns(), self._last_ns + 1)
            stamp = self._last_ns
        # RowKeys sort in time order; the random suffix avoids clashes with other processes.
        row_key = f"{stamp:020d}-{uuid4().hex[:6]}"
        self._table(table).create_entity(_entity(partition, row_key, data))

    def read_log(self, table: str, partition: str) -> list[dict]:
        _check_key(partition)
        return [json.loads(entity["data"]) for entity in self._partition(table, partition)]

    def _partition(self, table: str, partition: str) -> list[TableEntity]:
        # A parameterized filter: the SDK quotes the value, so a key can't inject filter syntax.
        entities = self._table(table).query_entities(
            "PartitionKey eq @pk", parameters={"pk": partition}
        )
        return sorted(entities, key=lambda entity: entity["RowKey"])  # don't rely on service order


def make_store(settings: Settings) -> Store:
    """Pick the backend from configuration, failing fast at startup if none is configured."""
    if settings.state_backend == "memory":
        return MemoryStore()
    if settings.tables_connection_string:
        return TableStore.from_connection_string(settings.tables_connection_string)
    if settings.tables_endpoint:
        return TableStore.from_endpoint(settings.tables_endpoint)
    raise ValueError(
        "No workflow state configured: set TABLES_CONNECTION_STRING (Azurite) or "
        "TABLES_ENDPOINT (Azure), or STATE_BACKEND=memory for tests."
    )

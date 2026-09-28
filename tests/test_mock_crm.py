"""The MOCK CRM's idempotency contract, and the client that relies on it for safe retries."""

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import mock_crm
from app.crm_client import CrmClient, CrmError
from app.store import MemoryStore

URL = "/mock-crm/opportunities"
KEY = "run-7f3a:crm-register"
PAYLOAD = {
    "client": "River Valley Road Authority",
    "opportunity_title": "Condition inspection of 12 road bridges",
    "tender_reference": "RVA-2026-014",
    "submission_deadline": "2026-10-30T12:00:00+01:00",
    "estimated_fee_chf": 185000,
    "bid_manager": "Mia Keller",
    "recommendation": "GO",
    "risk_flags": ["HR-1 unlimited liability", "HR-5 mandatory InspectAI tool"],
    "approved_by": "Lukas Brunner",
    "approval_id": "APR-0001",
}


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(mock_crm.router)
    app.state.store = MemoryStore()
    with TestClient(app) as test_client:
        yield test_client


def register(client: TestClient, key: str = KEY, payload: dict = PAYLOAD):
    return client.post(URL, json=payload, headers={"Idempotency-Key": key})


def test_same_key_twice_creates_one_record_and_replays_it(client):
    first = register(client)
    second = register(client)
    assert first.status_code == 201
    assert first.json()["mock"] is True
    assert second.status_code == 200
    assert second.headers["Idempotent-Replayed"] == "true"
    assert second.json() == first.json()  # the original record, same crm_id
    assert len(client.get(URL).json()) == 1


def test_same_key_with_a_different_payload_is_a_conflict(client):
    register(client)
    changed = register(client, payload={**PAYLOAD, "estimated_fee_chf": 210000})
    assert changed.status_code == 409
    assert "different payload" in changed.json()["detail"]
    [record] = client.get(URL).json()
    assert record["payload"]["estimated_fee_chf"] == 185000  # the original is untouched


def test_missing_idempotency_key_is_rejected(client):
    assert client.post(URL, json=PAYLOAD).status_code == 400
    assert client.get(URL).json() == []


def test_retry_after_a_lost_response_returns_the_committed_record(client):
    crm = CrmClient("http://testserver/mock-crm", http=client, backoff_seconds=0)
    record, attempts = crm.register(PAYLOAD, KEY, fault="error-after-commit")
    assert attempts == 2  # attempt 1 was saved but answered 504; attempt 2 got the replay
    [committed] = client.get(URL).json()
    assert record["crm_id"] == committed["crm_id"]


def test_concurrent_requests_with_the_same_key_create_one_record(client):
    barrier = threading.Barrier(2)

    def submit(_: int) -> int:
        barrier.wait(timeout=5)  # release both requests at the same moment
        return register(client).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = sorted(pool.map(submit, range(2)))
    assert statuses == [200, 201]  # one created the record, the other got the replay
    assert len(client.get(URL).json()) == 1


def test_client_resends_the_same_key_on_timeouts_then_gives_up():
    sent = []

    def always_time_out(request: httpx.Request) -> httpx.Response:
        sent.append((request.headers["Idempotency-Key"], request.headers.get("X-Mock-Fault")))
        raise httpx.ReadTimeout("simulated timeout", request=request)

    http = httpx.Client(transport=httpx.MockTransport(always_time_out))
    crm = CrmClient("http://crm.test", http=http, backoff_seconds=0)
    with pytest.raises(CrmError, match="3 attempts"):
        crm.register(PAYLOAD, KEY, fault="error-after-commit")
    assert sent == [(KEY, "error-after-commit"), (KEY, None), (KEY, None)]


def test_client_does_not_retry_a_rejected_request():
    calls = []

    def reject(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        detail = "Idempotency-Key reused with a different payload"
        return httpx.Response(409, json={"detail": detail})

    http = httpx.Client(transport=httpx.MockTransport(reject))
    crm = CrmClient("http://crm.test", http=http, backoff_seconds=0)
    with pytest.raises(CrmError, match="different payload"):
        crm.register(PAYLOAD, KEY)
    assert len(calls) == 1  # a 4xx means the request is wrong: resending cannot help

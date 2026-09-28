"""MOCK external CRM, for demos and tests only. It is NOT a real system and holds synthetic data.

It stands in for a real CRM (e.g. Dynamics 365) so the copilot's write path can be shown end to
end, including the failure that matters most for writes: the write succeeded, but the response
was lost. Idempotency keys behave like Stripe's:

- new key                      -> 201 Created, the record is committed
- same key, same payload       -> 200 OK with the ORIGINAL record, header Idempotent-Replayed: true
- same key, different payload  -> 409 Conflict: the key was reused by mistake
"""

import hashlib
import json
import secrets
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Header, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from app.store import AlreadyExistsError, Store

TABLE = "mockcrm"

router = APIRouter(prefix="/mock-crm", tags=["MOCK external CRM"])


class CrmRegistration(BaseModel):
    """A bid opportunity as the copilot registers it, after a human approved the decision."""

    model_config = ConfigDict(extra="forbid")  # an unknown field is a client bug: reject it

    client: str
    opportunity_title: str
    tender_reference: str
    submission_deadline: str
    estimated_fee_chf: int = Field(gt=0)
    bid_manager: str
    recommendation: Literal["GO", "NO-GO"]
    risk_flags: list[str]
    approved_by: str
    approval_id: str


class CrmRecord(BaseModel):
    mock: Literal[True] = True  # in every response: this is NOT a real CRM
    crm_id: str
    payload: CrmRegistration
    payload_hash: str  # sha256 of the canonical JSON payload; reveals a reused key
    created_at: str


def canonical_hash(payload: dict) -> str:
    # Sorted keys and no whitespace: the same content always gives the same bytes, so the same hash.
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def replay(store: Store, key: str, payload_hash: str, response: Response) -> CrmRecord:
    """The key was seen before: return the original record, or 409 if the payload differs."""
    existing = CrmRecord.model_validate(store.get(TABLE, key)[0])
    if existing.payload_hash != payload_hash:
        detail = "Idempotency-Key reused with a different payload"
        raise HTTPException(status_code=409, detail=detail)
    response.status_code = 200
    response.headers["Idempotent-Replayed"] = "true"
    return existing


# Plain `def` endpoints: store calls block, so FastAPI runs them in its thread pool.
@router.post("/opportunities", status_code=201)
def register_opportunity(
    body: CrmRegistration,
    request: Request,
    response: Response,
    # Python note: Annotated[type, Header(...)] tells FastAPI to read this parameter from the HTTP
    # header "Idempotency-Key" (underscores become dashes). `= None` makes the header optional
    # here, so we can answer with a clear 400 message when it's missing.
    idempotency_key: Annotated[str | None, Header(max_length=255)] = None,
    x_mock_fault: Annotated[
        Literal["error-after-commit"] | None,
        Header(description="MOCK fault injection for demos and tests"),
    ] = None,
) -> CrmRecord:
    """MOCK: register an opportunity, at most once per Idempotency-Key."""
    if not idempotency_key:
        raise HTTPException(status_code=400, detail="The Idempotency-Key header is required.")
    store: Store = request.app.state.store
    now = datetime.now(UTC)
    record = CrmRecord(
        crm_id=f"OPP-{now.year}-{secrets.token_hex(3).upper()}",
        payload=body,
        payload_hash=canonical_hash(body.model_dump()),
        created_at=now.isoformat(timespec="seconds"),
    )
    try:
        # Insert-once is atomic in the store: of two concurrent requests, only one can create.
        store.create(TABLE, idempotency_key, record.model_dump())
    except ValueError as exc:  # a key the store cannot hold, e.g. one containing '/'
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AlreadyExistsError:
        return replay(store, idempotency_key, record.payload_hash, response)

    # Faults only on this new-record path, so a retry with the same key replays normally.
    if x_mock_fault == "error-after-commit":
        # Committed, but the caller sees a timeout: it cannot know whether the write happened.
        raise HTTPException(
            status_code=504,
            detail="MOCK fault: the record was saved, but this response was 'lost'.",
        )
    return record


@router.get("/opportunities")
def list_opportunities(request: Request) -> list[CrmRecord]:
    """MOCK: every registered opportunity, to show that retries created no duplicates."""
    store: Store = request.app.state.store
    return [CrmRecord.model_validate(data) for data in store.list(TABLE)]

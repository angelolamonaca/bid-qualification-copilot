"""The bid workflow's safety rules, end to end with a fake model, in-memory storage and the
mock CRM. Each test name states one rule from the acceptance list.
"""

import threading
from contextlib import contextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import mock_crm
from app.crm_client import CrmClient
from app.documents import load_chunks, load_personas
from app.main import create_app
from app.models import BidAssessment, BidInputs
from app.retrieval import SearchIndex
from app.store import MemoryStore
from app.workflow import BidWorkflow, WorkflowError
from tests.fakes import FakeLLM, naive_assessment

PERSONAS = load_personas()
MIA, LUKAS, DANIEL, SARA = (PERSONAS[p] for p in ("mia", "lukas", "daniel", "sara"))
TENDER = "TENDER-RVA-2026-014"


class Setup:
    """A workflow wired to a fake model, one shared in-memory store and the mock CRM."""

    def __init__(self, llm: FakeLLM | None = None):
        self.store = MemoryStore()
        self.llm = llm or FakeLLM()
        crm_app = FastAPI()
        crm_app.include_router(mock_crm.router)
        crm_app.state.store = self.store
        self.crm_http = TestClient(crm_app)
        self.workflow = self.new_workflow()

    def new_workflow(self) -> BidWorkflow:
        """A fresh workflow object on the SAME storage, i.e. what a restart looks like."""
        crm = CrmClient("http://testserver/mock-crm", http=self.crm_http, backoff_seconds=0)
        return BidWorkflow(self.store, SearchIndex(load_chunks(), self.llm), self.llm, PERSONAS,
                           crm)

    def crm_records(self) -> list[dict]:
        return self.crm_http.get("/mock-crm/opportunities").json()


def approved_run(setup: Setup):
    """Helper: assess, give the fee, approve as the director. Returns the approved run."""
    run = setup.workflow.assess(MIA, TENDER)
    run = setup.workflow.update_inputs(run.run_id, MIA, BidInputs(estimated_fee_chf=180_000))
    return setup.workflow.decide(run.run_id, DANIEL, run.payload_hash, approve=True)


@contextmanager
def refused(status_code: int):
    """Assert that the code inside `with refused(403):` is refused with this HTTP status."""
    with pytest.raises(WorkflowError) as error:  # fails the test if nothing is raised
        yield
    assert error.value.status_code == status_code, error.value.detail


# --- Access and grounding ---------------------------------------------------------------------

def test_people_outside_the_opportunity_cannot_start_an_assessment():
    with refused(403):
        Setup().workflow.assess(SARA, TENDER)


def test_restricted_content_never_reaches_the_model_during_research():
    setup = Setup()
    setup.workflow.assess(MIA, TENDER)
    for canary in ("BLUE HERON", "PRJ-2025-044", "chloride"):
        assert all(canary not in prompt for prompt in setup.llm.prompts), canary


def test_citations_the_model_was_not_shown_are_removed_and_reported():
    def cites_a_confidential_chunk(messages, schema):
        assessment = naive_assessment()
        assessment.requirements[0].citations = ["PRJ-2025-044@v1#1"]
        return assessment

    run = Setup(FakeLLM(respond=cites_a_confidential_chunk)).workflow.assess(MIA, TENDER)
    assert run.assessment.requirements[0].citations == []
    assert any("PRJ-2025-044@v1#1" in warning for warning in run.warnings)


# --- Prompt injection and business rules -------------------------------------------------------

def test_injected_instructions_cannot_hide_risks_or_skip_approval():
    # The fake model "obeys" the tender's injected text: GO, no risk flags, nothing suspicious.
    run = Setup().workflow.assess(MIA, TENDER)
    assert {f.category for f in run.assessment.risk_flags} >= {"HR-1", "HR-5"}  # re-added by rules
    assert any("Rule check added HR-1" in warning for warning in run.warnings)
    assert run.status == "needs_input"  # nothing was registered: a human still has to decide


def test_high_risk_terms_require_the_regional_director_even_for_a_small_fee():
    setup = Setup()
    run = setup.workflow.assess(MIA, TENDER)
    run = setup.workflow.update_inputs(run.run_id, MIA, BidInputs(estimated_fee_chf=180_000))
    assert run.required_approver_role == "regional_director"
    with refused(403):  # the business line lead may approve only low-risk bids up to CHF 250k
        setup.workflow.decide(run.run_id, LUKAS, run.payload_hash, approve=True)
    assert setup.workflow.decide(run.run_id, DANIEL, run.payload_hash, approve=True).status == \
        "approved"


def test_missing_fields_trigger_a_clarifying_question():
    run = Setup().workflow.assess(MIA, TENDER)
    assert run.missing_fields == ["estimated_fee_chf"]
    assert "estimated fee" in run.questions[0]
    assert run.draft is None and run.payload_hash is None


def test_the_bid_manager_cannot_approve_their_own_registration():
    setup = Setup()
    run = setup.workflow.assess(MIA, TENDER)
    run = setup.workflow.update_inputs(run.run_id, MIA, BidInputs(estimated_fee_chf=180_000))
    with refused(403):
        setup.workflow.decide(run.run_id, MIA, run.payload_hash, approve=True)


# --- Approval bound to the exact payload ---------------------------------------------------------

def test_no_approval_means_nothing_is_written_to_the_crm():
    setup = Setup()
    run = setup.workflow.assess(MIA, TENDER)
    run = setup.workflow.update_inputs(run.run_id, MIA, BidInputs(estimated_fee_chf=180_000))
    with refused(409):
        setup.workflow.submit(run.run_id, MIA)
    assert setup.crm_records() == []


def test_changing_the_registration_after_approval_voids_the_approval():
    setup = Setup()
    run = approved_run(setup)
    run = setup.workflow.update_inputs(run.run_id, MIA, BidInputs(estimated_fee_chf=175_000))
    assert run.approval is None and run.status == "pending_approval"
    with refused(409):
        setup.workflow.submit(run.run_id, MIA)
    assert setup.crm_records() == []


def test_an_approval_of_an_outdated_version_is_refused():
    setup = Setup()
    run = setup.workflow.assess(MIA, TENDER)
    first = setup.workflow.update_inputs(run.run_id, MIA, BidInputs(estimated_fee_chf=180_000))
    setup.workflow.update_inputs(run.run_id, MIA, BidInputs(estimated_fee_chf=175_000))
    with refused(409):  # Daniel reviewed the CHF 180k version; it changed before he clicked
        setup.workflow.decide(run.run_id, DANIEL, first.payload_hash, approve=True)


# --- Exactly-once write ---------------------------------------------------------------------------

def test_submitting_twice_creates_one_crm_record():
    setup = Setup()
    run = approved_run(setup)
    first = setup.workflow.submit(run.run_id, MIA)
    second = setup.workflow.submit(run.run_id, MIA)
    assert first.crm_record["crm_id"] == second.crm_record["crm_id"]
    assert len(setup.crm_records()) == 1


def test_a_double_click_creates_one_crm_record():
    setup = Setup()
    run = approved_run(setup)
    results = []
    clicks = [threading.Thread(target=lambda: results.append(
        setup.workflow.submit(run.run_id, MIA))) for _ in range(2)]
    for click in clicks:
        click.start()
    for click in clicks:
        click.join()
    assert len({r.crm_record["crm_id"] for r in results}) == 1
    assert len(setup.crm_records()) == 1


def test_a_timeout_after_a_successful_write_does_not_duplicate_on_retry():
    setup = Setup()
    run = approved_run(setup)
    run = setup.workflow.submit(run.run_id, MIA, simulate_crm_timeout=True)
    assert run.status == "submitted"
    assert len(setup.crm_records()) == 1
    registered = [e for e in setup.workflow.events(run.run_id) if e["event"] == "crm_registered"]
    assert registered[0]["attempts"] == 2  # first attempt "timed out", the retry got the record


def test_a_restart_keeps_the_committed_state():
    setup = Setup()
    run = setup.workflow.submit(approved_run(setup).run_id, MIA)
    after_restart = setup.new_workflow().get(run.run_id)  # new objects, same durable store
    assert after_restart.status == "submitted"
    assert after_restart.crm_record == run.crm_record
    assert after_restart.payload_hash == run.payload_hash


# --- The HTTP API -----------------------------------------------------------------------------

def test_the_api_runs_the_whole_flow():
    store = MemoryStore()
    app = create_app(llm=FakeLLM(), store=store)
    crm_app = FastAPI()
    crm_app.include_router(mock_crm.router)
    crm_app.state.store = store
    with TestClient(app) as client:
        app.state.workflow.crm = CrmClient("http://testserver/mock-crm",
                                           http=TestClient(crm_app), backoff_seconds=0)
        run = client.post("/api/bids", json={"persona_id": "mia"}).json()
        run = client.post(f"/api/bids/{run['run_id']}/inputs",
                          json={"persona_id": "mia", "estimated_fee_chf": 180000}).json()
        denied = client.post(f"/api/bids/{run['run_id']}/decision",
                             json={"persona_id": "lukas", "payload_hash": run["payload_hash"],
                                   "decision": "approve"})
        assert denied.status_code == 403
        client.post(f"/api/bids/{run['run_id']}/decision",
                    json={"persona_id": "daniel", "payload_hash": run["payload_hash"],
                          "decision": "approve"})
        done = client.post(f"/api/bids/{run['run_id']}/submit", json={"persona_id": "mia"})
        assert done.json()["status"] == "submitted"
        events = [e["event"] for e in client.get(f"/api/bids/{run['run_id']}/events").json()]
        assert events[0] == "assessment_started" and events[-1] == "crm_registered"
        assert "decision_refused" in events  # Lukas's attempt is in the audit trail


def test_assessment_schema_has_no_defaults_so_the_model_must_fill_every_field():
    # Structured outputs in strict mode need every field to be required.
    assert all(field.is_required() for field in BidAssessment.model_fields.values())

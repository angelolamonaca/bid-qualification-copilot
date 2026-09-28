"""Run the whole bid workflow from the terminal with the REAL model.

Storage is in memory and the mock CRM runs in-process, so no server or emulator is needed.
Run from the project root:
    uv run python -m scripts.demo_bid
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import mock_crm
from app.config import get_settings
from app.crm_client import CrmClient
from app.documents import load_chunks, load_personas
from app.llm import AzureOpenAILLM
from app.models import BidInputs
from app.retrieval import SearchIndex
from app.store import MemoryStore
from app.workflow import BidWorkflow, WorkflowError

personas = load_personas()
mia, lukas, daniel = personas["mia"], personas["lukas"], personas["daniel"]
store = MemoryStore()
crm_app = FastAPI()  # the MOCK CRM, running inside this process
crm_app.include_router(mock_crm.router)
crm_app.state.store = store
crm_http = TestClient(crm_app)
llm = AzureOpenAILLM(get_settings())
workflow = BidWorkflow(store, SearchIndex(load_chunks(), llm), llm, personas,
                       CrmClient("http://testserver/mock-crm", http=crm_http))

print("\n1) Mia (bid manager) asks the copilot to assess tender RVA-2026-014 ...")
run = workflow.assess(mia, "TENDER-RVA-2026-014")
a = run.assessment
searches = [e["query"] for e in workflow.events(run.run_id) if e["event"] == "tool_call"]
print(f"   The agent ran {len(searches)} searches: {searches}")
for r in a.requirements:
    print(f"   {r.requirement_id:3} {r.status:12} {r.requirement}  {r.citations}")
for f in a.risk_flags:
    print(f"   RISK {f.category} {f.clause}: {f.quote[:70]}")
print(f"   Recommendation: {a.recommendation}. {a.rationale}")
for w in run.warnings:
    print(f"   WARNING: {w}")
print(f"   Status: {run.status}. Questions: {run.questions}")
tokens = sum(u.input_tokens + u.output_tokens for u in run.usage)
print(f"   Model calls: {len(run.usage)}, {tokens} tokens, "
      f"${sum(u.cost_usd for u in run.usage):.4f}, "
      f"{sum(u.latency_ms for u in run.usage) / 1000:.1f} s")

print("\n2) Mia answers: estimated fee CHF 180,000")
run = workflow.update_inputs(run.run_id, mia, BidInputs(estimated_fee_chf=180_000))
print(f"   Status: {run.status}; required approver: {run.required_approver_role}")

print("\n3) Lukas (business line lead) tries to approve ...")
try:
    workflow.decide(run.run_id, lukas, run.payload_hash, approve=True)
except WorkflowError as refusal:
    print(f"   REFUSED ({refusal.status_code}): {refusal.detail}")

print("\n4) Daniel (regional director) approves the exact registration he reviewed")
run = workflow.decide(run.run_id, daniel, run.payload_hash, approve=True, comment="Go ahead.")
print(f"   Status: {run.status}")

print("\n5) Mia submits; the CRM saves the record but the response is 'lost' (simulated)")
run = workflow.submit(run.run_id, mia, simulate_crm_timeout=True)
# next(...) = the first matching audit event
registered = next(e for e in workflow.events(run.run_id) if e["event"] == "crm_registered")
print(f"   Status: {run.status}; CRM id {run.crm_record['crm_id']} after "
      f"{registered['attempts']} attempts")

print("\n6) Mia clicks submit again")
again = workflow.submit(run.run_id, mia)
records = crm_http.get("/mock-crm/opportunities").json()
print(f"   Same CRM id: {again.crm_record['crm_id']}. Records in the CRM: {len(records)}")

print("\nAUDIT TRAIL")
for event in workflow.events(run.run_id):
    print(f"   {event['at']}  {event['actor']:7} {event['event']}")

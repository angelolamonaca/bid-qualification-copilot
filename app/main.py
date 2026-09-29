"""FastAPI entry point: wires configuration, model, search, storage and the workflow together.

What happens on POST /api/ask:
  JSON body -> validated into AskRequest -> look up the simulated persona -> answer_question()
  -> AnswerResult, which FastAPI validates again and sends back as JSON.

The bid workflow (Step 4) has one endpoint per step:
  POST /api/bids                      research + assess a tender        (bid manager)
  POST /api/bids/{run_id}/inputs      answer questions, e.g. the fee    (bid manager)
  POST /api/bids/{run_id}/decision    approve or reject                 (authorized approver)
  POST /api/bids/{run_id}/submit      one-time write to the (mock) CRM  (bid manager)
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app import mock_crm
from app.config import get_settings
from app.crm_client import CrmClient
from app.documents import load_chunks, load_personas
from app.llm import LLM, AzureOpenAILLM, LLMError
from app.models import AnswerResult, BidInputs, BidRun, Persona
from app.qa import answer_question
from app.retrieval import SearchIndex
from app.store import Store, make_store
from app.workflow import BidWorkflow, WorkflowError

STATIC_DIR = Path(__file__).resolve().parent / "static"


class AskRequest(BaseModel):
    """The JSON body of POST /api/ask. FastAPI rejects invalid bodies with HTTP 422."""

    # Field(...) adds validation rules and documentation; `examples` pre-fills the form at /docs.
    persona_id: str = Field(examples=["mia"])
    question: str = Field(
        min_length=3,
        max_length=1000,
        examples=["Do we meet the mandatory requirements of tender RVA-2026-014?"],
    )


class StartBid(BaseModel):
    persona_id: str = Field(examples=["mia"])
    tender_doc_id: str = Field(default="TENDER-RVA-2026-014")
    # Optional: the page picks the id so it can show the agent's searches live while it waits.
    run_id: str | None = Field(default=None, examples=[None])


class UpdateBid(BaseModel):
    persona_id: str = Field(examples=["mia"])
    estimated_fee_chf: int | None = Field(default=None, gt=0, examples=[180000])
    recommendation: Literal["GO", "NO-GO"] | None = None


class DecideBid(BaseModel):
    persona_id: str = Field(examples=["daniel"])
    payload_hash: str  # the fingerprint of the registration the approver actually reviewed
    decision: Literal["approve", "reject"]
    comment: str = ""


class SubmitBid(BaseModel):
    persona_id: str = Field(examples=["mia"])
    simulate_crm_timeout: bool = False  # demo: the CRM saves, but the response is "lost"


def create_app(llm: LLM | None = None, store: Store | None = None) -> FastAPI:
    """Build the app. Tests pass a fake model and in-memory storage; normally Azure is used.

    A factory function (instead of a single global app) lets every test build a fresh app with
    its own fake dependencies.
    """

    # Startup/shutdown hook. The code before `yield` runs once when the server starts. The server
    # then handles requests while this function waits at `yield`. Code after it would run at
    # shutdown.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = get_settings()
        # `app.state` holds objects shared by all requests (like app.locals in Express).
        # `llm or AzureOpenAILLM(...)` = use the fake if a test passed one, else the real client.
        app.state.llm = llm or AzureOpenAILLM(settings)
        app.state.store = store or make_store(settings)
        app.state.personas = load_personas()
        app.state.index = SearchIndex(load_chunks(), app.state.llm)  # embeds all chunks once
        app.state.workflow = BidWorkflow(
            app.state.store, app.state.index, app.state.llm, app.state.personas,
            CrmClient(settings.crm_base_url),  # the CRM is reached over HTTP, like a real one
        )
        yield

    app = FastAPI(title="Bid Qualification Copilot", version="0.2.0", lifespan=lifespan)
    app.include_router(mock_crm.router)  # the MOCK CRM's endpoints under /mock-crm

    # Exception handlers turn an error raised anywhere during a request into a clean HTTP answer
    # (similar to an Express error middleware).
    @app.exception_handler(openai.APIError)
    async def model_unavailable(request: Request, exc: openai.APIError) -> JSONResponse:
        # Raised after the SDK's bounded retries are used up (timeouts, 429, 5xx).
        detail = f"The model service is unavailable ({type(exc).__name__}). Please retry."
        return JSONResponse(status_code=503, content={"detail": detail})

    @app.exception_handler(LLMError)
    async def unusable_answer(request: Request, exc: LLMError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(WorkflowError)
    async def refused(request: Request, exc: WorkflowError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    def persona_of(request: Request, persona_id: str) -> Persona:
        persona = request.app.state.personas.get(persona_id)  # None if the id is unknown
        if persona is None:
            raise HTTPException(status_code=404, detail="Unknown persona")
        return persona

    # Routes. `@app.get("/health")` registers the function below as the handler for GET /health,
    # like app.get("/health", handler) in Express. The return type (-> ...) becomes the documented
    # response schema at /docs. These handlers use plain `def`, not `async def`: FastAPI then runs
    # them in a worker thread, so a slow model call doesn't block other requests.
    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        """The demo web page (plain HTML + JavaScript, no build step)."""
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness check, used locally and by Azure Container Apps."""
        return {"status": "ok", "env": get_settings().app_env}

    @app.get("/api/whoami")
    def whoami(request: Request) -> dict[str, str | None]:
        """Who REALLY signed in, as opposed to the simulated persona the demo acts as.

        In Azure, Container Apps' built-in authentication signs people in with Microsoft Entra ID
        before a request reaches the app, and passes the user's name in this header (clients
        can't forge it while authentication is on). Locally there is no sign-in: None.
        """
        return {"signed_in_as": request.headers.get("x-ms-client-principal-name"),
                "env": get_settings().app_env}

    @app.get("/api/personas")
    def personas(request: Request) -> list[Persona]:
        return list(request.app.state.personas.values())

    @app.post("/api/ask")
    def ask(body: AskRequest, request: Request) -> AnswerResult:
        # `body: AskRequest` tells FastAPI to parse and validate the JSON body into an AskRequest.
        persona = persona_of(request, body.persona_id)
        return answer_question(body.question, persona, request.app.state.index,
                               request.app.state.llm)

    @app.post("/api/bids")
    def start_bid(body: StartBid, request: Request) -> BidRun:
        return request.app.state.workflow.assess(persona_of(request, body.persona_id),
                                                 body.tender_doc_id, body.run_id)

    @app.get("/api/bids")
    def list_bids(request: Request) -> list[BidRun]:
        return request.app.state.workflow.list_runs()

    @app.get("/api/bids/{run_id}")  # {run_id} in the path becomes the run_id parameter
    def get_bid(run_id: str, request: Request) -> BidRun:
        return request.app.state.workflow.get(run_id)

    @app.get("/api/bids/{run_id}/events")
    def bid_events(run_id: str, request: Request) -> list[dict]:
        """The audit trail: every step, tool call and decision, in order."""
        return request.app.state.workflow.events(run_id)

    @app.post("/api/bids/{run_id}/inputs")
    def update_bid(run_id: str, body: UpdateBid, request: Request) -> BidRun:
        inputs = BidInputs(estimated_fee_chf=body.estimated_fee_chf,
                           recommendation=body.recommendation)
        return request.app.state.workflow.update_inputs(
            run_id, persona_of(request, body.persona_id), inputs)

    @app.post("/api/bids/{run_id}/decision")
    def decide_bid(run_id: str, body: DecideBid, request: Request) -> BidRun:
        return request.app.state.workflow.decide(
            run_id, persona_of(request, body.persona_id), body.payload_hash,
            approve=body.decision == "approve", comment=body.comment)

    @app.post("/api/bids/{run_id}/submit")
    def submit_bid(run_id: str, body: SubmitBid, request: Request) -> BidRun:
        return request.app.state.workflow.submit(
            run_id, persona_of(request, body.persona_id), body.simulate_crm_timeout)

    return app


# The object the web server runs: `uvicorn app.main:app` means "file app/main.py, variable app".
app = create_app()

"""FastAPI entry point: wires configuration, the model client and the search index together.

What happens on POST /api/ask:
  JSON body -> validated into AskRequest -> look up the simulated persona -> answer_question()
  -> AnswerResult, which FastAPI validates again and sends back as JSON.
"""

from contextlib import asynccontextmanager

import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.documents import load_chunks, load_personas
from app.llm import LLM, AzureOpenAILLM, LLMError
from app.models import AnswerResult, Persona
from app.qa import answer_question
from app.retrieval import SearchIndex


class AskRequest(BaseModel):
    """The JSON body of POST /api/ask. FastAPI rejects invalid bodies with HTTP 422."""

    # Field(...) adds validation rules and documentation; `examples` pre-fills the form at /docs.
    persona_id: str = Field(examples=["mia"])
    question: str = Field(
        min_length=3,
        max_length=1000,
        examples=["Do we meet the mandatory requirements of tender RVA-2026-014?"],
    )


def create_app(llm: LLM | None = None) -> FastAPI:
    """Build the app. Tests pass a fake model; normally the Azure OpenAI client is used.

    A factory function (instead of a single global app) lets every test build a fresh app with
    its own fake dependencies.
    """

    # Startup/shutdown hook. The code before `yield` runs once when the server starts. The server
    # then handles requests while this function waits at `yield`. Code after it would run at
    # shutdown.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # `app.state` holds objects shared by all requests (like app.locals in Express).
        # `llm or AzureOpenAILLM(...)` = use the fake if a test passed one, else the real client.
        app.state.llm = llm or AzureOpenAILLM(get_settings())
        app.state.personas = load_personas()
        app.state.index = SearchIndex(load_chunks(), app.state.llm)  # embeds all chunks once
        yield

    app = FastAPI(title="Bid Qualification Copilot", version="0.1.0", lifespan=lifespan)

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

    # Routes. `@app.get("/health")` registers the function below as the handler for GET /health,
    # like app.get("/health", handler) in Express. The return type (-> ...) becomes the documented
    # response schema at /docs. These handlers use plain `def`, not `async def`: FastAPI then runs
    # them in a worker thread, so a slow model call doesn't block other requests.
    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness check, used locally and by Azure Container Apps."""
        return {"status": "ok", "env": get_settings().app_env}

    @app.get("/api/personas")
    def personas(request: Request) -> list[Persona]:
        return list(request.app.state.personas.values())

    @app.post("/api/ask")
    def ask(body: AskRequest, request: Request) -> AnswerResult:
        # `body: AskRequest` tells FastAPI to parse and validate the JSON body into an AskRequest.
        persona = request.app.state.personas.get(body.persona_id)  # None if the id is unknown
        if persona is None:
            raise HTTPException(status_code=404, detail="Unknown persona")
        return answer_question(body.question, persona, request.app.state.index,
                               request.app.state.llm)

    return app


# The object the web server runs: `uvicorn app.main:app` means "file app/main.py, variable app".
app = create_app()

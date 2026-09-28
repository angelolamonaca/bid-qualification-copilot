"""FastAPI entry point: wires configuration, the model client and the search index together."""

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
    persona_id: str = Field(examples=["mia"])
    question: str = Field(min_length=3, max_length=1000,
                          examples=["Do we meet the mandatory requirements of tender RVA-2026-014?"])


def create_app(llm: LLM | None = None) -> FastAPI:
    """Build the app. Tests pass a fake model; normally the Azure OpenAI client is used."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.llm = llm or AzureOpenAILLM(get_settings())
        app.state.personas = load_personas()
        app.state.index = SearchIndex(load_chunks(), app.state.llm)  # embeds all chunks once
        yield

    app = FastAPI(title="Bid Qualification Copilot", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(openai.APIError)
    async def model_unavailable(request: Request, exc: openai.APIError) -> JSONResponse:
        # Raised after the SDK's bounded retries are used up (timeouts, 429, 5xx).
        detail = f"The model service is unavailable ({type(exc).__name__}). Please retry."
        return JSONResponse(status_code=503, content={"detail": detail})

    @app.exception_handler(LLMError)
    async def unusable_answer(request: Request, exc: LLMError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness check, used locally and by Azure Container Apps."""
        return {"status": "ok", "env": get_settings().app_env}

    @app.get("/api/personas")
    def personas(request: Request) -> list[Persona]:
        return list(request.app.state.personas.values())

    @app.post("/api/ask")
    def ask(body: AskRequest, request: Request) -> AnswerResult:
        persona = request.app.state.personas.get(body.persona_id)
        if persona is None:
            raise HTTPException(status_code=404, detail="Unknown persona")
        return answer_question(body.question, persona, request.app.state.index,
                               request.app.state.llm)

    return app


app = create_app()

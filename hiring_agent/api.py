"""FastAPI service around the screening workflow.

POST /screenings                run the workflow; returns a finished result or `awaiting_review`
GET  /screenings/{id}           current state of a run
POST /screenings/{id}/review    resume a paused run with a human decision
GET  /health

Set API_KEY to require an `X-API-Key` header. CHECKPOINT_DB (default checkpoints.db) persists runs
across restarts. MCP_SERVER_URL makes the workflow call its tools over MCP instead of in-process.
"""
from __future__ import annotations

import hmac
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field

from hiring_agent.graph import build_graph, initial_state
from hiring_agent.llm import GeminiLLM, TemplateLLM
from hiring_agent.tools_client import LocalTools, McpTools

THREAD_ID = re.compile(r"^[0-9a-f]{32}$")


class ResumeIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=20_000)


class ScreeningIn(BaseModel):
    jd_text: str = Field(min_length=20, max_length=20_000)
    resumes: list[ResumeIn] = Field(min_length=1, max_length=50)
    top_k: int = Field(default=3, ge=1, le=10)


class ReviewIn(BaseModel):
    approve: bool
    exclude: list[str] = Field(default_factory=list, max_length=50)
    note: str = Field(default="", max_length=500)


def _view(thread_id: str, snap) -> dict:
    values = snap.values or {}
    pending = [i.value for t in snap.tasks for i in t.interrupts]
    status = "awaiting_review" if pending else values.get("status", "running")
    return {
        "id": thread_id,
        "status": status,
        "pending_review": pending[0] if pending else None,
        "jd_skills": values.get("jd_skills", []),
        "flags": values.get("flags", []),
        "ranking": [{k: c[k] for k in ("rank", "name", "score", "contributions", "matched_skills",
                                       "missing_skills", "experience_years")} for c in values.get("ranking", [])],
        "shortlist": [c["name"] for c in values.get("shortlist", [])],
        "interview_plans": values.get("interviews", {}),
        "review": values.get("review"),
        "trace": values.get("trace", []),
    }


def _default_graph_factory():
    llm = GeminiLLM() if os.getenv("GEMINI_API_KEY") else TemplateLLM()
    url = os.getenv("MCP_SERVER_URL")
    return llm, (McpTools(url) if url else LocalTools())


def create_app(graph=None, api_key: Optional[str] = None) -> FastAPI:
    api_key = api_key if api_key is not None else os.getenv("API_KEY")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if graph is not None:
            app.state.graph = graph
            yield
            return
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        llm, tools = _default_graph_factory()
        async with AsyncSqliteSaver.from_conn_string(os.getenv("CHECKPOINT_DB", "checkpoints.db")) as saver:
            app.state.graph = build_graph(tools=tools, llm=llm, checkpointer=saver)
            yield

    app = FastAPI(title="Hiring Agent", version="0.1.0", lifespan=lifespan)

    async def require_key(x_api_key: str = Header(default="")):
        if api_key and not hmac.compare_digest(x_api_key, api_key):
            raise HTTPException(401, "Invalid or missing API key.")

    def _cfg(thread_id: str) -> dict:
        if not THREAD_ID.match(thread_id):
            raise HTTPException(404, "Unknown screening.")
        return {"configurable": {"thread_id": thread_id}}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/screenings", dependencies=[Depends(require_key)])
    async def create_screening(body: ScreeningIn):
        thread_id = uuid.uuid4().hex
        cfg = _cfg(thread_id)
        state = initial_state(body.jd_text, [r.model_dump() for r in body.resumes], body.top_k)
        await app.state.graph.ainvoke(state, cfg)
        return _view(thread_id, await app.state.graph.aget_state(cfg))

    @app.get("/screenings/{thread_id}", dependencies=[Depends(require_key)])
    async def get_screening(thread_id: str):
        cfg = _cfg(thread_id)
        snap = await app.state.graph.aget_state(cfg)
        if not snap.values:
            raise HTTPException(404, "Unknown screening.")
        return _view(thread_id, snap)

    @app.post("/screenings/{thread_id}/review", dependencies=[Depends(require_key)])
    async def review(thread_id: str, body: ReviewIn):
        cfg = _cfg(thread_id)
        snap = await app.state.graph.aget_state(cfg)
        if not snap.values:
            raise HTTPException(404, "Unknown screening.")
        if not any(t.interrupts for t in snap.tasks):
            raise HTTPException(409, "This screening is not waiting for review.")
        await app.state.graph.ainvoke(Command(resume=body.model_dump()), cfg)
        return _view(thread_id, await app.state.graph.aget_state(cfg))

    return app


def app_factory() -> FastAPI:  # uvicorn hiring_agent.api:app_factory --factory
    return create_app()

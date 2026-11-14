"""FastAPI app: generate / preview / edit / undo, plus the static UI.

Run:  uvicorn builder.app:create_app --factory --reload   (then open http://localhost:8000)
"""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from common.config import ROOT, Settings
from common.events import EventLog
from common.providers import ChatClient, ProviderError, get_chat

from .agents import Workflow, WorkflowError
from .store import SiteNotFound, SiteStore, StaleVersion

STATIC = Path(__file__).parent / "static"
EXPECTED = (ProviderError, WorkflowError, StaleVersion)   # failures that leave state untouched


class GenerateRequest(BaseModel):
    brief: str              # length is checked by the workflow, which returns a readable error


class EditRequest(BaseModel):
    instruction: str
    base_version: int | None = None


def failure(exc: Exception, log: EventLog | None = None, **extra) -> JSONResponse:
    """The one place where an expected failure becomes an HTTP status, an error body and a log event."""
    if isinstance(exc, ProviderError):
        status, code, message = exc.http_status, f"provider_{exc.kind}", exc.message
    elif isinstance(exc, WorkflowError):
        status, code, message = 422, exc.code, exc.reason
    elif isinstance(exc, StaleVersion):
        status, code, message = 409, "stale_version", str(exc)
    else:
        raise exc
    if log is not None:
        log.log("request_failed", code=code, message=message)
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, **extra},
                                                     "events": log.events if log else []})


def create_app(settings: Settings | None = None, chat: ChatClient | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = SiteStore()
    app = FastAPI(title="PromptSite")
    app.state.store = store
    app.mount("/static", StaticFiles(directory=ROOT / "common" / "static"))   # stylesheet and script shared with /rag
    clients: dict = {"chat": chat}
    app.state.clients = clients

    def workflow(log: EventLog) -> Workflow:
        # The chat client is built on first use and then shared: a missing key becomes a clean error
        # response instead of a crash at start-up, and the client keeps what it has learned about
        # the model's accepted parameters.
        if clients["chat"] is None:
            clients["chat"] = get_chat(settings)
        return Workflow(clients["chat"], log, settings.builder_max_repairs)

    def event_log(site_id: str | None = None) -> EventLog:
        return EventLog("builder", settings.log_dir, {"site_id": site_id} if site_id else {})

    @app.exception_handler(SiteNotFound)
    def not_found(request, exc) -> JSONResponse:
        return JSONResponse(status_code=404, content={
            "error": {"code": "not_found", "message": "Unknown site id. Generate a website first."}, "events": []})

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        info = {**settings.llm.public(), "ready": True, "problem": None}
        try:
            workflow(event_log())
        except ProviderError as exc:
            info.update(ready=False, problem=exc.message)
        return info

    @app.post("/api/sites")
    def generate(req: GenerateRequest):
        started, log = time.perf_counter(), event_log()
        try:
            site = workflow(log).generate(req.brief)
        except EXPECTED as exc:
            return failure(exc, log)
        site_id, entry = store.create(site, req.brief)
        return {"site_id": site_id, "version": entry["version"], "site": entry["site"], "events": log.events,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1)}

    @app.get("/api/sites/{site_id}")
    def get_site(site_id: str):
        entry = store.current(site_id)
        return {"site_id": site_id, "version": entry["version"], "site": entry["site"]}

    @app.post("/api/sites/{site_id}/edit")
    def edit(site_id: str, req: EditRequest):
        current, log = store.current(site_id), event_log(site_id)
        try:
            store.current(site_id, req.base_version)       # stale preview: refuse before paying for a model call
            result = workflow(log).edit(current["site"], req.instruction)
            # Commit against the version the edit was computed from, so a concurrent edit that
            # landed during the model call is never overwritten.
            entry = store.commit(site_id, result.site, instruction=req.instruction, ops=result.ops,
                                 base_version=current["version"])
        except EXPECTED as exc:
            return failure(exc, log, version=current["version"])
        return {"site_id": site_id, "version": entry["version"], "site": entry["site"], "ops": result.ops,
                "explanation": result.explanation, "attempts": result.attempts,
                "latency_ms": result.latency_ms, "events": log.events}

    @app.post("/api/sites/{site_id}/undo")
    def undo(site_id: str):
        entry = store.undo(site_id)
        if entry is None:
            return failure(WorkflowError("Nothing to undo.", "nothing_to_undo"), version=store.current(site_id)["version"])
        log = event_log(site_id)
        log.log("undo", version=entry["version"], restored_from=entry["restored_from"])
        return {"site_id": site_id, "version": entry["version"], "site": entry["site"], "events": log.events}

    @app.get("/api/sites/{site_id}/history")
    def history(site_id: str):
        return {"site_id": site_id, "history": store.history(site_id)}

    return app

"""HTTP API and page for Workstream A: index documents, then ask questions about them.

The index is one in-memory object (single-user demo, like the site store). It starts with the
configured corpus (MS MARCO subset or the local sample) and is replaced when documents are uploaded.
"""
from __future__ import annotations

import threading
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from common.config import Settings
from common.events import EventLog
from common.providers import ChatClient, EmbeddingClient, ProviderError, get_chat, get_embedder

from .data import clean_text, load_corpus
from .pipeline import RAGPipeline, build_pipeline

STATIC = Path(__file__).parent / "static"


class Document(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)


class IndexRequest(BaseModel):
    documents: list[Document] = Field(max_length=50)     # empty list = back to the configured corpus


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


def _error(status: int, code: str, message: str, events: list | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}, "events": events or []})


def create_router(settings: Settings, chat: ChatClient | None = None, embedder: EmbeddingClient | None = None) -> APIRouter:
    router = APIRouter()
    lock = threading.Lock()
    state: dict = {"pipeline": None, "source": "", "examples": []}

    def index(passages: dict[str, str], source: str, examples: list[str] = ()) -> None:
        # Clients are built on first use, so a missing key is an error response, not a crash at start-up.
        pipeline = build_pipeline(passages, settings, chat or get_chat(settings), embedder or get_embedder(settings))
        with lock:
            state.update(pipeline=pipeline, source=source, examples=list(examples))

    def pipeline() -> RAGPipeline:
        if state["pipeline"] is None:
            corpus = load_corpus(settings)
            index(corpus.passages, corpus.source, [q.text for q in corpus.queries[:3]])   # questions the corpus can answer
        return state["pipeline"]

    def describe() -> dict:
        retriever = pipeline().retriever
        return {"source": state["source"], "documents": len({c.passage_id for c in retriever.chunks}),
                "chunks": len(retriever.chunks), "retriever": settings.rag_retriever, "chunker": settings.rag_chunker,
                "grader": settings.rag_grader, "examples": state["examples"], **settings.public()}

    @router.get("/rag")
    def page() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @router.get("/api/rag/corpus")
    def corpus():
        try:
            return describe()
        except ProviderError as exc:
            return _error(exc.http_status, f"provider_{exc.kind}", exc.message)

    @router.post("/api/rag/documents")
    def upload(req: IndexRequest):
        log = EventLog("rag", settings.log_dir)
        try:
            if not req.documents:
                state["pipeline"] = None
            else:
                passages = {d.name: text for d in req.documents if (text := clean_text(d.text))}
                if not passages:
                    return _error(422, "empty_documents", "The documents contain no text.")
                index(passages, "uploaded")
            info = describe()
        except ProviderError as exc:
            return _error(exc.http_status, f"provider_{exc.kind}", exc.message)
        log.log("indexed", **{k: info[k] for k in ("source", "documents", "chunks", "retriever", "chunker")})
        return {**info, "events": log.events}

    @router.post("/api/rag/ask")
    def ask(req: AskRequest):
        log = EventLog("rag", settings.log_dir)
        log.log("question", question=req.question)
        try:
            current = pipeline()
            result = RAGPipeline(current.retriever, current.chat, settings, log).answer(req.question)
        except ProviderError as exc:
            log.log("request_failed", code=f"provider_{exc.kind}", message=exc.message)
            return _error(exc.http_status, f"provider_{exc.kind}", exc.message, log.events)
        return {"answer": result.answer, "abstained": result.abstained, "queries_tried": result.queries_tried,
                "timings_ms": result.timings_ms, "events": log.events,
                "contexts": [{"document": c.chunk.passage_id, "text": c.chunk.text, "grade": round(c.grade_score, 2),
                              "retrieval_score": round(c.retrieval_score, 4)} for c in result.contexts]}

    return router

"""One server for both workstreams.

    uvicorn server:create_app --factory --port 8000      # builder at /, RAG at /rag

This is the only module that imports both workstreams; they still do not import each other.
"""
from __future__ import annotations

from fastapi import FastAPI

from builder.app import create_app as create_builder_app
from common.config import Settings
from rag.api import create_router


def create_app() -> FastAPI:
    settings = Settings.from_env()
    app = create_builder_app(settings)
    app.include_router(create_router(settings))
    return app

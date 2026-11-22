"""FastAPI application factory.

`create_app(core)` wires the internal API around a `CoreService`. Without a core (OpenAPI export,
surface listing) the routes exist but are not callable.
"""

from __future__ import annotations

from fastapi import FastAPI

from cinewright.api.problems import install_problem_handlers
from cinewright.api.routes import api_router
from cinewright.api.schemas import API_PREFIX, API_VERSION
from cinewright.core.service import CoreService

OPENAPI_INFO = {
    "title": "Cinewright internal API",
    "version": API_VERSION,
    "description": "The only API of Cinewright. Web UI, MCP server and workers are its clients. Binds to 127.0.0.1 by default.",
}


def create_app(core: CoreService | None = None) -> FastAPI:
    app = FastAPI(
        title=OPENAPI_INFO["title"],
        version=OPENAPI_INFO["version"],
        description=OPENAPI_INFO["description"],
        openapi_url=f"{API_PREFIX}/openapi.json",
        docs_url=None,
        redoc_url=None,
    )
    app.state.core = core
    install_problem_handlers(app)
    app.include_router(api_router())
    return app

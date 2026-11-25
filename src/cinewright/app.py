"""FastAPI application factory.

`create_app(core)` wires the internal API around a `CoreService`, adds the browser protection
(`cinewright.security`), mounts the MCP server at `/mcp` when it is installed and serves the built
web UI for every other GET. Building the app has no side effects, so the OpenAPI export and the
surface listing can call `create_app()`; without an injected core the lifespan opens the data
directory and creates a `Core` when the app starts. With a `WorkerSupervisor` the lifespan starts the
local worker. With `provision` it starts engine provisioning and, without a supervisor, has the core start
the local worker on the provisioned engine once it is ready. It stops all of them at shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI
from starlette.responses import FileResponse, HTMLResponse, Response
from starlette.routing import BaseRoute, Match, NoMatchFound
from starlette.types import ASGIApp, Receive, Scope, Send

from cinewright.api.problems import install_problem_handlers, problem_response
from cinewright.api.routes import api_router
from cinewright.api.routes.jobs import stream_event_schemas
from cinewright.api.schemas import API_PREFIX, API_VERSION, PROBLEM_NOT_FOUND
from cinewright.config import Config, load_config
from cinewright.core.core import Core
from cinewright.core.service import CoreService
from cinewright.core.supervisor import WorkerSupervisor
from cinewright.security import LocalOnlyMiddleware, is_api_path

log = logging.getLogger(__name__)

OPENAPI_INFO = {
    "title": "Cinewright internal API",
    "version": API_VERSION,
    "description": "The only API of Cinewright. Web UI, MCP server and workers are its clients. Binds to 127.0.0.1 by default.",
}

WEB_DIST = Path(__file__).resolve().parent / "web_dist"
MCP_PATH = "/mcp"
# Characters and segments a built web UI path never contains; checked before any lookup.
_UNSAFE_PATH_CHARS = ("\\", ":", "\x00")

# Explicit types: the Windows registry can map .js to text/plain, which browsers refuse for module scripts.
STATIC_MEDIA_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".txt": "text/plain; charset=utf-8",
}

NOT_BUILT_HTML = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Cinewright</title></head>
<body style="font-family: system-ui, sans-serif; background: #0b0b0d; color: #e8e8ea; padding: 2rem">
<h1>Cinewright</h1>
<p>The web UI is not built, so only the API under <code>/api/v1</code> is available.</p>
</body>
</html>
"""


class WebUiRoute(BaseRoute):
    """GET and HEAD outside `/api` and `/mcp`: a file from the built web UI, else `index.html` (SPA fallback).

    The files are listed once at startup and requests are looked up in that list, so a request path
    never reaches the filesystem. On Windows, joining or resolving a path such as `\\\\host\\share`
    would open a remote file (and send the user's NTLM credentials) before any containment check.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[str, Path] = (
            {path.relative_to(root).as_posix(): path for path in root.rglob("*") if path.is_file()} if root.is_dir() else {}
        )

    def matches(self, scope: Scope) -> tuple[Match, Scope]:
        if scope["type"] == "http" and scope["method"] in ("GET", "HEAD") and not is_api_path(scope["path"]):
            return Match.FULL, {}
        return Match.NONE, {}

    def url_path_for(self, name: str, /, **path_params: Any) -> Any:
        raise NoMatchFound(name, path_params)

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.response(scope["path"])(scope, receive, send)

    def response(self, path: str) -> Response:
        if any(char in path for char in _UNSAFE_PATH_CHARS) or ".." in path.split("/"):
            return problem_response(404, "Not found", None, PROBLEM_NOT_FOUND, path)
        index = self.files.get("index.html")
        if index is None:
            return HTMLResponse(NOT_BUILT_HTML)
        candidate = self.files.get(path.lstrip("/"))
        if candidate is not None:
            return FileResponse(candidate, media_type=STATIC_MEDIA_TYPES.get(candidate.suffix.lower()))
        return FileResponse(index, media_type=STATIC_MEDIA_TYPES[".html"], headers={"Cache-Control": "no-cache"})


class McpRootAlias:
    """Serve `/mcp` as `/mcp/`, so MCP clients reach the mounted app without a redirect."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] == MCP_PATH:
            scope = {**scope, "path": MCP_PATH + "/", "raw_path": (MCP_PATH + "/").encode()}
        await self.app(scope, receive, send)


def _build_mcp(app: FastAPI, port: int) -> tuple[Any, httpx.AsyncClient] | None:
    try:
        from cinewright.mcp.server import build_mcp_app
    except ImportError as exc:
        log.warning("MCP server not available, /mcp is not served: %s", exc)
        return None
    # The MCP server calls the internal API in-process, through the same middleware as any client.
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=f"http://127.0.0.1:{port}")
    return build_mcp_app(client), client


def _add_stream_event_schemas(app: FastAPI) -> None:
    """Have the OpenAPI document also list the schemas the `text/event-stream` response of `GET /events` refers to.

    FastAPI registers only models of JSON bodies and responses as components (see `stream_event_schemas`).
    """
    generate = app.openapi

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            document = generate()  # FastAPI keeps it as `app.openapi_schema`; completed in place
            schemas = document.setdefault("components", {}).setdefault("schemas", {})
            for name, schema in stream_event_schemas().items():
                if schemas.setdefault(name, schema) != schema:
                    raise RuntimeError(f"Two different OpenAPI schemas are named {name}")
            document["components"]["schemas"] = dict(sorted(schemas.items()))
        return app.openapi_schema

    app.openapi = openapi


def create_app(
    core: CoreService | None = None,
    config: Config | None = None,
    *,
    supervisor: WorkerSupervisor | None = None,
    provision: bool = False,
) -> FastAPI:
    """`supervisor`, when given, runs the local worker from startup to shutdown (only with a `Core`).

    `provision` starts engine provisioning at startup (only with a `Core`); the launcher sets it, tests do not.
    Without a `supervisor`, the core then starts the local worker from the provisioning record as soon as the
    engine is ready: at startup, or when a provisioning run (first run or retry) reaches ready. So the installed
    app starts no worker while provisioning runs or after it failed.
    """
    config = config or getattr(core, "config", None) or load_config()
    mcp: Any = None
    api_client: httpx.AsyncClient | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = Core(config) if app.state.core is None else None
        if owned is not None:
            app.state.core = owned
        current = app.state.core
        tasks: list[asyncio.Task[None]] = []
        supervised = False
        watching = False  # the core starts the local worker once the engine is ready
        try:
            if isinstance(current, Core):
                app.state.enrollment_token = current.enrollment_token
                await asyncio.to_thread(current.ensure_hardware)
                tasks.append(asyncio.create_task(current.run_maintenance()))
                if supervisor is not None:
                    supervisor.start(
                        core_url=config.base_url, enrollment_token=current.enrollment_token, paths=current.paths, clock=current.clock
                    )
                    current.supervisor = supervisor
                    supervised = True
                if provision:
                    current.start_provisioning()  # first: it may try the optional wheels again before a worker runs
                    if not supervised:
                        current.run_local_worker_when_ready(config.base_url)
                        watching = True
            async with AsyncExitStack() as stack:
                if mcp is not None:
                    await stack.enter_async_context(mcp.lifespan())
                yield
        finally:
            # Shutdown order: stop granting leases, ask the worker to stop, kill it after the grace period.
            # uvicorn has closed its sockets by now, so a running job returns to the queue when its lease expires.
            if isinstance(current, Core) and (owned is not None or supervised or watching):
                current.stop_leasing()
            if watching:
                await asyncio.to_thread(current.stop_local_worker)  # also keeps a run that reaches ready now from starting one
            if isinstance(current, Core):
                await asyncio.to_thread(current.provisioner.stop)  # a later start resumes it
            if supervised:
                await asyncio.to_thread(supervisor.stop)
                current.supervisor = None
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if api_client is not None:
                await api_client.aclose()
            if owned is not None:
                owned.close()
                app.state.core = None

    app = FastAPI(
        title=OPENAPI_INFO["title"],
        version=OPENAPI_INFO["version"],
        description=OPENAPI_INFO["description"],
        openapi_url=f"{API_PREFIX}/openapi.json",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.core = core
    app.state.config = config
    app.state.enrollment_token = getattr(core, "enrollment_token", None)
    install_problem_handlers(app)
    app.include_router(api_router())
    _add_stream_event_schemas(app)

    built = _build_mcp(app, config.port)
    if built is not None:
        mcp, api_client = built
        app.mount(MCP_PATH, mcp.asgi_app)
        app.add_middleware(McpRootAlias)
    app.router.routes.append(WebUiRoute(WEB_DIST))
    app.add_middleware(LocalOnlyMiddleware, port=config.port)
    return app

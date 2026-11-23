"""FastAPI application factory.

`create_app(core)` wires the internal API around a `CoreService`, adds the browser protection
(`cinewright.security`), mounts the MCP server at `/mcp` when it is installed and serves the built
web UI for every other GET. Building the app has no side effects, so the OpenAPI export and the
surface listing can call `create_app()`; without an injected core the lifespan opens the data
directory and creates a `Core` when the app starts.
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
from cinewright.api.schemas import API_PREFIX, API_VERSION, PROBLEM_NOT_FOUND
from cinewright.config import Config, load_config
from cinewright.core.core import Core
from cinewright.core.service import CoreService
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


def create_app(core: CoreService | None = None, config: Config | None = None) -> FastAPI:
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
        try:
            if isinstance(current, Core):
                app.state.enrollment_token = current.enrollment_token
                await asyncio.to_thread(current.ensure_hardware)
                tasks.append(asyncio.create_task(current.run_maintenance()))
            async with AsyncExitStack() as stack:
                if mcp is not None:
                    await stack.enter_async_context(mcp.lifespan())
                yield
        finally:
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

    built = _build_mcp(app, config.port)
    if built is not None:
        mcp, api_client = built
        app.mount(MCP_PATH, mcp.asgi_app)
        app.add_middleware(McpRootAlias)
    app.router.routes.append(WebUiRoute(WEB_DIST))
    app.add_middleware(LocalOnlyMiddleware, port=config.port)
    return app

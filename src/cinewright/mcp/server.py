"""MCP server: thin tools over the internal API, served as streamable HTTP.

Each tool is one call to the internal API through the `httpx.AsyncClient` the core passes in; the
server has no logic of its own. `submit` sends the headers `X-Cinewright-Client: mcp` and
`X-Cinewright-Agent: <name>`, so the core records the job and its provenance as an MCP job of that
agent. The name is the `clientInfo.name` the MCP client sent in `initialize`, reduced to the
characters of `^[A-Za-z0-9 ._-]{1,64}$`, else `mcp`. It is self-declared until per-agent tokens
arrive (issue #7).

The core serves `MCPApp.asgi_app` at `/mcp` and enters `MCPApp.lifespan()` inside its own lifespan.
`asgi_app` answers on any path, so either form works:

    app.add_route("/mcp", mcp_app.asgi_app)   # serves exactly /mcp
    app.mount("/mcp", mcp_app.asgi_app)       # serves /mcp/ and redirects /mcp with 307

DNS-rebinding protection accepts only `Host` 127.0.0.1:* / localhost:* and origins on those hosts.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Annotated, Any

import httpx
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.tools import Tool
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field

from cinewright import __version__
from cinewright.api.schemas import API_PREFIX, DEFAULT_PORT, JobRequest, Sha256

DEFAULT_PUBLIC_URL = f"http://127.0.0.1:{DEFAULT_PORT}"
CLIENT_HEADER = "X-Cinewright-Client"
AGENT_HEADER = "X-Cinewright-Agent"
MAX_AGENT_CHARS = 64
_NOT_AGENT_CHARS = re.compile(r"[^A-Za-z0-9 ._-]")
TRANSPORT_SECURITY = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=["127.0.0.1:*", "localhost:*"],
    allowed_origins=["http://127.0.0.1:*", "http://localhost:*"],
)
INSTRUCTIONS = (
    "Cinewright generates images and videos locally with open-weight models. Jobs run asynchronously: "
    "submit returns a job id, job_status reports progress and output hashes, fetch_asset returns an output's "
    "provenance and content URL."
)

JobId = Annotated[str, Field(min_length=1, max_length=128, description="Job id returned by submit")]


class MCPApp:
    """The MCP server as the core mounts it: `asgi_app` plus the `lifespan()` that must wrap its use."""

    def __init__(self, server: MCPServer) -> None:
        self.server = server
        server.streamable_http_app(transport_security=TRANSPORT_SECURITY)  # creates the session manager
        self.asgi_app = StreamableHTTPASGIApp(server.session_manager)

    def lifespan(self) -> AbstractAsyncContextManager[None]:
        """Run the session manager; works once per `MCPApp`."""
        return self.server.session_manager.run()


def build_mcp_app(api_client: httpx.AsyncClient, *, public_url: str | None = None) -> MCPApp:
    """Build the MCP server around a client of the internal API.

    `public_url` is the core URL that content links in `fetch_asset` point at; it defaults to the
    client's base URL, else `http://127.0.0.1:8737`.
    """
    base = (public_url or str(api_client.base_url)).rstrip("/") or DEFAULT_PUBLIC_URL
    server = MCPServer(name="cinewright", title="Cinewright", version=__version__, instructions=INSTRUCTIONS)
    for tool in _tool_functions(api_client, base):
        server.add_tool(tool)
    return MCPApp(server)


def tool_surface() -> list[str]:
    """One sorted line per tool, `<name>(<JSON schema of its inputs, sorted keys>)`, for the surface check."""
    tools = [Tool.from_function(fn) for fn in _tool_functions(None, DEFAULT_PUBLIC_URL)]
    return sorted(f"{tool.name}({json.dumps(tool.parameters, sort_keys=True)})" for tool in tools)


def agent_name(ctx: Context | None) -> str:
    """The MCP client's `clientInfo.name`, kept to `[A-Za-z0-9 ._-]` and 64 characters; `mcp` if nothing is left."""
    try:
        params = ctx.request_context.session.client_params if ctx is not None else None
    except (AttributeError, ValueError):  # no request context, e.g. a direct call
        params = None
    name = params.client_info.name if params is not None and params.client_info is not None else ""
    cleaned = _NOT_AGENT_CHARS.sub("", str(name)).strip()[:MAX_AGENT_CHARS].strip()
    return cleaned or "mcp"


def _tool_functions(api: httpx.AsyncClient | None, public_url: str) -> list[Callable[..., Any]]:
    async def call(method: str, path: str, *, body: Any = None, headers: dict[str, str] | None = None) -> Any:
        if api is None:
            raise ToolError("No internal API client is configured.")
        try:
            response = await api.request(method, f"{API_PREFIX}{path}", json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise ToolError(f"The Cinewright core is not reachable: {exc}") from exc
        if response.is_error:
            raise ToolError(_problem_text(response))
        return response.json()

    async def list_capabilities() -> dict[str, Any]:
        """List what this installation can generate: capability, model, quantization, limits and calibration status."""
        return await call("GET", "/capabilities")

    async def estimate(request: JobRequest) -> dict[str, Any]:
        """Check a job request without starting it: feasibility, chosen model and host, runtime and cost."""
        return await call("POST", "/estimate", body=_body(request))

    async def submit(request: JobRequest, ctx: Context) -> dict[str, Any]:
        """Start a job. Jobs run asynchronously; returns the job id and its state."""
        headers = {CLIENT_HEADER: "mcp", AGENT_HEADER: agent_name(ctx)}
        job = await call("POST", "/jobs", body=_body(request), headers=headers)
        return {"job_id": job["id"], "state": job["state"]}

    async def job_status(job_id: JobId) -> dict[str, Any]:
        """State, progress, ETA in seconds, output asset hashes and error of a job."""
        if job_id in (".", ".."):
            raise ToolError("Invalid job id.")
        job = await call("GET", f"/jobs/{urllib.parse.quote(job_id, safe='')}")
        return {key: job.get(key) for key in ("id", "state", "progress", "eta_seconds", "outputs", "error")}

    async def fetch_asset(sha256: Sha256) -> dict[str, Any]:
        """Metadata and provenance of an asset and the URL of its content (the bytes are not returned)."""
        asset = await call("GET", f"/assets/{sha256}")
        return {**asset, "content_url": f"{public_url}{API_PREFIX}/assets/{sha256}/content"}

    async def list_recipes() -> dict[str, Any]:
        """List the available recipes (multi-step workflows)."""
        return await call("GET", "/recipes")

    async def list_cast() -> dict[str, Any]:
        """List the available cast entries (recurring characters)."""
        return await call("GET", "/cast")

    return [list_capabilities, estimate, submit, job_status, fetch_asset, list_recipes, list_cast]


def _body(request: JobRequest) -> dict[str, Any]:
    return request.model_dump(mode="json", exclude_none=True)


def _problem_text(response: httpx.Response) -> str:
    """One readable line from an RFC 9457 problem response."""
    try:
        problem = response.json()
    except ValueError:
        problem = {}
    if not isinstance(problem, dict):
        problem = {}
    text = f"{problem.get('title') or response.reason_phrase} (HTTP {response.status_code})"
    if problem.get("detail"):
        text += f": {problem['detail']}"
    reasons = problem.get("reasons") or [f"{'.'.join(map(str, e.get('loc', [])))}: {e.get('msg')}" for e in problem.get("errors", [])]
    if reasons:
        text += " Reasons: " + "; ".join(str(reason) for reason in reasons)
    return text

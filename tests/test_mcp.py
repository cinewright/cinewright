"""MCP server: tool list, the API call behind each tool, the tool surface, and the HTTP transport."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from mcp.server.mcpserver.exceptions import ToolError

from cinewright.api import schemas
from cinewright.mcp.server import AGENT_HEADER, CLIENT_HEADER, agent_name, build_mcp_app, tool_surface

TOOLS = ["estimate", "fetch_asset", "job_status", "list_capabilities", "list_cast", "list_recipes", "submit"]
SHA = hashlib.sha256(b"output").hexdigest()
NOW = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)
REQUEST = {"capability": "t2i", "prompt": "A lighthouse at dusk", "params": {"resolution": "1024x1024", "seed": 7}}
PROTOCOL = "2025-11-25"
HOST = "http://127.0.0.1:8737"


@pytest.fixture
def anyio_backend():
    return "asyncio"


def job(state: str = "queued") -> dict:
    return schemas.Job(
        id="job-1",
        state=state,
        request=schemas.JobRequest.model_validate(REQUEST),
        progress=schemas.Progress(phase="inference", step=4, total_steps=8, percent=50),
        eta_seconds=12.5,
        client=schemas.ClientKind.MCP,
        outputs=[SHA],
        created_at=NOW,
    ).model_dump(mode="json")


def asset() -> dict:
    provenance = schemas.Provenance(asset=schemas.ProvenanceAsset(sha256=SHA, media_type="image/png", size_bytes=6), created_at=NOW, seed=7)
    return schemas.Asset(
        sha256=SHA, media_type="image/png", size_bytes=6, data_class="private", job_id="job-1", created_at=NOW, provenance=provenance
    ).model_dump(mode="json", by_alias=True)


class FakeAPI:
    """httpx.MockTransport handler answering the internal API routes the tools use."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        route = (request.method, request.url.path)
        if route == ("GET", "/api/v1/capabilities"):
            return httpx.Response(200, json=schemas.CapabilityList(items=[]).model_dump(mode="json"))
        if route == ("POST", "/api/v1/estimate"):
            return httpx.Response(200, json=schemas.Estimate(feasible=True, eta_seconds=30, cost_eur=0).model_dump(mode="json"))
        if route == ("POST", "/api/v1/jobs"):
            schemas.JobRequest.model_validate_json(request.content)
            return httpx.Response(202, json=job())
        if route == ("GET", "/api/v1/jobs/job-1"):
            return httpx.Response(200, json=job("running"))
        if route == ("GET", f"/api/v1/assets/{SHA}"):
            return httpx.Response(200, json=asset())
        if route == ("GET", "/api/v1/recipes"):
            body = {
                "type": schemas.PROBLEM_NOT_IMPLEMENTED,
                "title": "Not implemented yet",
                "status": 501,
                "detail": "Recipes arrive in phase 2.",
            }
            return httpx.Response(501, json=body, headers={"Content-Type": "application/problem+json"})
        if route == ("GET", "/api/v1/cast"):
            return httpx.Response(200, json=schemas.CastList(items=[]).model_dump(mode="json"))
        return httpx.Response(404, json={"title": "Not found", "status": 404})


@pytest.fixture
def api() -> FakeAPI:
    return FakeAPI()


def make_app(api: FakeAPI):
    return build_mcp_app(httpx.AsyncClient(transport=httpx.MockTransport(api), base_url=HOST))


async def call(app, name: str, arguments: dict | None = None) -> dict:
    result = await app.server.call_tool(name, arguments or {})
    assert not result.is_error, result.content
    return result.structured_content


@pytest.mark.anyio
async def test_tools_are_exactly_the_specified_set(api):
    tools = await make_app(api).server.list_tools()
    assert sorted(tool.name for tool in tools) == TOOLS
    assert all(tool.description and len(tool.description) < 160 for tool in tools)


@pytest.mark.anyio
async def test_each_tool_calls_its_api_route(api):
    app = make_app(api)
    assert await call(app, "list_capabilities") == {"items": []}
    assert (await call(app, "estimate", {"request": REQUEST}))["feasible"] is True
    assert await call(app, "submit", {"request": REQUEST}) == {"job_id": "job-1", "state": "queued"}
    status = await call(app, "job_status", {"job_id": "job-1"})
    assert status == {
        "id": "job-1",
        "state": "running",
        "progress": job()["progress"],
        "eta_seconds": 12.5,
        "outputs": [SHA],
        "error": None,
    }
    fetched = await call(app, "fetch_asset", {"sha256": SHA})
    assert fetched["content_url"] == f"http://127.0.0.1:8737/api/v1/assets/{SHA}/content"
    assert fetched["provenance"]["schema"] == "cinewright.provenance/1"
    assert await call(app, "list_cast") == {"items": []}
    assert [(r.method, r.url.path) for r in api.requests] == [
        ("GET", "/api/v1/capabilities"),
        ("POST", "/api/v1/estimate"),
        ("POST", "/api/v1/jobs"),
        ("GET", "/api/v1/jobs/job-1"),
        ("GET", f"/api/v1/assets/{SHA}"),
        ("GET", "/api/v1/cast"),
    ]
    submitted = api.requests[2]
    assert (submitted.headers[CLIENT_HEADER], submitted.headers[AGENT_HEADER]) == ("mcp", "mcp")  # no MCP session here
    assert json.loads(submitted.content) == {**REQUEST, "inputs": [], "priority": "interactive", "data_class": "private"} | {
        "params": {**REQUEST["params"], "extra": {}}
    }
    assert CLIENT_HEADER not in api.requests[1].headers  # only submit creates jobs


@pytest.mark.anyio
async def test_content_url_follows_the_core_url(api):
    other_port = build_mcp_app(httpx.AsyncClient(transport=httpx.MockTransport(api), base_url="http://127.0.0.1:9100"))
    assert (await call(other_port, "fetch_asset", {"sha256": SHA}))["content_url"].startswith("http://127.0.0.1:9100/api/v1/assets/")
    explicit = build_mcp_app(httpx.AsyncClient(transport=httpx.MockTransport(api), base_url=HOST), public_url="http://localhost:9200/")
    assert (await call(explicit, "fetch_asset", {"sha256": SHA}))["content_url"].startswith("http://localhost:9200/api/v1/assets/")


@pytest.mark.anyio
async def test_problems_become_tool_errors(api):
    app = make_app(api)
    with pytest.raises(ToolError, match=r"Not implemented yet \(HTTP 501\): Recipes arrive in phase 2\."):
        await app.server.call_tool("list_recipes", {})
    with pytest.raises(ToolError):
        await app.server.call_tool("fetch_asset", {"sha256": "../../etc"})
    with pytest.raises(ToolError, match="HTTP 404"):
        await app.server.call_tool("job_status", {"job_id": "a/../../assets"})
    assert api.requests[-1].url.raw_path == b"/api/v1/jobs/a%2F..%2F..%2Fassets"


def test_tool_surface_is_stable():
    lines = tool_surface()
    assert lines == tool_surface() == sorted(lines)
    assert [line.split("(", 1)[0] for line in lines] == TOOLS
    by_name = {line.split("(", 1)[0]: json.loads(line.split("(", 1)[1][:-1]) for line in lines}
    assert by_name["submit"]["required"] == ["request"]
    assert "ctx" not in by_name["submit"]["properties"]
    assert by_name["fetch_asset"]["properties"]["sha256"]["pattern"] == "^[0-9a-f]{64}$"
    assert by_name["list_capabilities"]["properties"] == {}


@pytest.mark.parametrize(
    ("name", "agent"),
    [
        ("claude-code", "claude-code"),
        ("Claude\x07 Desktop\n", "Claude Desktop"),
        ("agent/1.2 (beta); <script>", "agent1.2 beta script"),
        ("  \xe9\N{EURO SIGN}\r\n  ", "mcp"),
        ("x" * 63 + " yz", "x" * 63),
        ("", "mcp"),
        (None, "mcp"),
    ],
)
def test_agent_name_is_reduced_to_the_header_pattern(name, agent):
    client_info = None if name is None else SimpleNamespace(name=name)
    ctx = SimpleNamespace(request_context=SimpleNamespace(session=SimpleNamespace(client_params=SimpleNamespace(client_info=client_info))))
    assert agent_name(ctx) == agent
    assert re.fullmatch(r"[A-Za-z0-9 ._-]{1,64}", agent_name(ctx))
    assert agent_name(None) == "mcp"


# ---------------------------------------------------------------------- streamable HTTP transport


def sse_messages(response: httpx.Response) -> list[dict]:
    if response.headers["content-type"].startswith("application/json"):
        return [response.json()]
    return [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:") and line[5:].strip()]


def rpc(method: str, params: dict | None = None, id: int | None = None) -> dict:  # noqa: A002 - JSON-RPC field name
    message = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        message["params"] = params
    if id is not None:
        message["id"] = id
    return message


HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json", "MCP-Protocol-Version": PROTOCOL}


@pytest.mark.anyio
async def test_http_round_trip_records_the_agent(api):
    mcp_app = make_app(api)
    core = FastAPI()
    core.add_route("/mcp", mcp_app.asgi_app)
    async with mcp_app.lifespan(), httpx.AsyncClient(transport=httpx.ASGITransport(app=core), base_url=HOST) as client:
        init = {"protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": {"name": "Claude\x07 Desktop\n", "version": "1"}}
        response = await client.post("/mcp", json=rpc("initialize", init, id=1), headers=HEADERS)
        assert response.status_code == 200
        assert sse_messages(response)[0]["result"]["serverInfo"]["name"] == "cinewright"
        session = {**HEADERS, "Mcp-Session-Id": response.headers["mcp-session-id"]}
        assert (await client.post("/mcp", json=rpc("notifications/initialized"), headers=session)).status_code == 202

        listed = sse_messages(await client.post("/mcp", json=rpc("tools/list", id=2), headers=session))[0]
        assert sorted(tool["name"] for tool in listed["result"]["tools"]) == TOOLS

        called = await client.post(
            "/mcp", json=rpc("tools/call", {"name": "submit", "arguments": {"request": REQUEST}}, id=3), headers=session
        )
        result = sse_messages(called)[0]["result"]
        assert result["isError"] is False and result["structuredContent"] == {"job_id": "job-1", "state": "queued"}
        submitted = api.requests[-1]
        assert submitted.url.path == "/api/v1/jobs"
        assert (submitted.headers[CLIENT_HEADER], submitted.headers[AGENT_HEADER]) == ("mcp", "Claude Desktop")

        failed = await client.post("/mcp", json=rpc("tools/call", {"name": "list_recipes", "arguments": {}}, id=4), headers=session)
        assert sse_messages(failed)[0]["result"]["isError"] is True


@pytest.mark.anyio
async def test_mount_serves_the_mount_root(api):
    mcp_app = make_app(api)
    core = FastAPI()
    core.mount("/mcp", mcp_app.asgi_app)
    init = {"protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}
    async with mcp_app.lifespan(), httpx.AsyncClient(transport=httpx.ASGITransport(app=core), base_url=HOST) as client:
        response = await client.post("/mcp/", json=rpc("initialize", init, id=1), headers=HEADERS)
        assert response.status_code == 200


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("headers", "status"),
    [({"Host": "evil.example:8737"}, 421), ({"Origin": "http://evil.example"}, 403), ({"Content-Type": "text/plain"}, 400)],
)
async def test_transport_security_rejects_foreign_requests(api, headers, status):
    mcp_app = make_app(api)
    core = FastAPI()
    core.add_route("/mcp", mcp_app.asgi_app)
    init = {"protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}
    async with mcp_app.lifespan(), httpx.AsyncClient(transport=httpx.ASGITransport(app=core), base_url=HOST) as client:
        response = await client.post("/mcp", content=json.dumps(rpc("initialize", init, id=1)), headers={**HEADERS, **headers})
        assert response.status_code == status
    assert api.requests == []

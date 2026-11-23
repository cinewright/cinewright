"""App wiring: web UI with SPA fallback, the page shown when it is not built, MCP mount, lifespan."""

from __future__ import annotations

import importlib.util
import json

import pytest
from conftest import API, BASE_URL, assert_problem
from fastapi.testclient import TestClient

import cinewright.app as app_module
from cinewright.app import create_app


@pytest.fixture
def web_dist(tmp_path, monkeypatch):
    root = tmp_path / "web_dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><title>Cinewright</title>", encoding="utf-8")
    (root / "assets" / "app.js").write_text("console.log('ui')", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("outside", encoding="utf-8")
    monkeypatch.setattr(app_module, "WEB_DIST", root)
    return root


def test_serves_files_and_falls_back_to_index(core, web_dist):
    with TestClient(create_app(core), base_url=BASE_URL) as client:
        script = client.get("/assets/app.js")
        assert script.status_code == 200 and script.headers["content-type"].startswith("text/javascript")
        for path in ("/", "/jobs/123", "/assets/missing.js"):
            page = client.get(path)
            assert page.status_code == 200 and "<title>Cinewright</title>" in page.text
            assert page.headers["cache-control"] == "no-cache"
        assert "outside" not in client.get("/../secret.txt").text
        assert "outside" not in client.get("/%2e%2e/secret.txt").text
        assert_problem(client.get(f"{API}/unknown"), 404)
        assert_problem(client.get("/api"), 404)
        assert_problem(client.post("/somewhere"), 404)


@pytest.mark.parametrize(
    "path", ["/%5C%5Chost%5Cshare%5Cx.png", "/..%2F..", "/..%2F..%2Fsecret.txt", "/C:%5Cx.png", "/a%00.js", "/assets%5Capp.js"]
)
def test_unsafe_paths_are_refused_before_any_file_access(core, web_dist, monkeypatch, path):
    with TestClient(create_app(core), base_url=BASE_URL) as client:
        touched: list[object] = []
        for name in ("resolve", "is_file", "exists", "stat"):
            original = getattr(app_module.Path, name)
            monkeypatch.setattr(app_module.Path, name, lambda self, *a, _o=original, **k: touched.append(self) or _o(self, *a, **k))
        assert_problem(client.get(path), 404)
        assert touched == []


def test_web_ui_files_are_listed_at_startup(core, web_dist):
    with TestClient(create_app(core), base_url=BASE_URL) as client:
        (web_dist / "late.js").write_text("added after startup", encoding="utf-8")
        assert "added after startup" not in client.get("/late.js").text  # unknown: the SPA fallback answers


def test_page_when_the_web_ui_is_not_built(core, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "WEB_DIST", tmp_path / "missing")
    with TestClient(create_app(core), base_url=BASE_URL) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "The web UI is not built" in page.text


def test_lifespan_creates_and_closes_its_own_core(home):
    app = create_app()
    with TestClient(app, base_url=BASE_URL) as client:
        assert client.get(f"{API}/health").json()["status"] == "ok"
        assert app.state.enrollment_token
        assert client.get(f"{API}/system").json()["data_dir"] == str(home)
    assert app.state.core is None


def _rpc_result(response) -> dict:
    """The JSON-RPC result of a streamable HTTP answer (SSE or plain JSON)."""
    assert response.status_code == 200, response.text
    if response.headers["content-type"].startswith("text/event-stream"):
        payload = next(line[6:] for line in response.text.splitlines() if line.startswith("data: "))
    else:
        payload = response.text
    return json.loads(payload)["result"]


@pytest.mark.skipif(importlib.util.find_spec("cinewright.mcp.server") is None, reason="MCP server not in this build")
def test_mcp_tools_reach_the_core_through_slash_mcp(client):
    headers = {"Accept": "application/json, text/event-stream"}
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}},
    }
    response = client.post("/mcp", json=initialize, headers=headers)
    assert "serverInfo" in _rpc_result(response)
    session = {**headers, "Mcp-Session-Id": response.headers["mcp-session-id"], "Mcp-Protocol-Version": "2025-06-18"}
    assert client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=session).status_code == 202
    call = {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "list_capabilities", "arguments": {}}}
    result = _rpc_result(client.post("/mcp", json=call, headers=session))
    assert not result.get("isError"), result
    capabilities = result.get("structuredContent") or json.loads(result["content"][0]["text"])
    assert {item["model_id"] for item in capabilities["items"]} == {"test-image", "test-video"}

"""Browser protection: Host, Origin, Sec-Fetch-Site and body media type checks on /api and /mcp, and the response headers."""

from __future__ import annotations

import pytest
from conftest import API, assert_problem
from starlette.datastructures import Headers

from cinewright.security import LocalOnlyMiddleware

JOB = {"capability": "t2i", "prompt": "a lighthouse"}


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8737", "127.0.0.1:9999", "192.168.1.5:8737"])
def test_foreign_host_is_refused(client, host):
    problem = assert_problem(client.get(f"{API}/health", headers={"Host": host}), 403)
    assert "Host" in problem.detail


def test_loopback_hosts_are_accepted(client):
    assert client.get(f"{API}/health", headers={"Host": "localhost:8737"}).status_code == 200
    assert client.get(f"{API}/health", headers={"Host": "127.0.0.1:8737"}).status_code == 200


def test_missing_host_is_refused():
    middleware = LocalOnlyMiddleware(app=None, port=8737)
    assert middleware.rejection(Headers({}))[0] == 403


@pytest.mark.parametrize("origin", ["https://evil.example", "http://127.0.0.1:9999", "null", "http://localhost.evil.example:8737"])
def test_foreign_origin_is_refused(client, origin):
    response = client.post(f"{API}/estimate", json=JOB, headers={"Origin": origin})
    assert "Origin" in assert_problem(response, 403).detail


def test_own_origin_is_accepted(client):
    for origin in ("http://127.0.0.1:8737", "http://localhost:8737"):
        assert client.post(f"{API}/estimate", json=JOB, headers={"Origin": origin}).status_code == 200


@pytest.mark.parametrize(
    "content_type", ["application/x-www-form-urlencoded", "multipart/form-data; boundary=x", "text/plain", "Text/Plain; charset=utf-8"]
)
def test_form_like_bodies_are_refused(client, content_type):
    response = client.post(f"{API}/estimate", content=b'{"capability": "t2i", "prompt": "x"}', headers={"Content-Type": content_type})
    assert_problem(response, 415)


def test_json_and_binary_bodies_pass(client):
    assert client.post(f"{API}/estimate", json=JOB).status_code == 200
    assert client.post(f"{API}/assets", content=b"png", headers={"Content-Type": "image/png"}).status_code == 201


def test_responses_forbid_framing_sniffing_and_embedding(client):
    refused = client.get(f"{API}/health", headers={"Host": "evil.example"})
    assert refused.status_code == 403
    for headers in (client.get(f"{API}/health").headers, client.get("/").headers, refused.headers):
        assert headers["x-frame-options"] == "DENY"
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["content-security-policy"] == "frame-ancestors 'none'"
        assert headers["cross-origin-resource-policy"] == "same-origin"


def fetch(site: str, mode: str = "no-cors", dest: str = "image") -> dict[str, str]:
    return {"Sec-Fetch-Site": site, "Sec-Fetch-Mode": mode, "Sec-Fetch-Dest": dest}


@pytest.mark.parametrize("site", ["cross-site", "same-site", "Cross-Site"])
@pytest.mark.parametrize("path", [f"{API}/health", f"{API}/assets/{'0' * 64}/content", "/", "/assets/app.js"])
def test_subresource_requests_from_other_sites_are_refused(client, site, path):
    response = client.get(path, headers=fetch(site))
    assert "other sites" in assert_problem(response, 403).detail
    assert response.headers["cross-origin-resource-policy"] == "same-origin"


def test_cross_site_navigation_opens_only_the_web_ui(client):
    navigate = fetch("cross-site", "navigate", "document")
    assert client.get("/", headers=navigate).status_code == 200
    assert client.get("/jobs/123", headers=navigate).status_code == 200
    assert_problem(client.get(f"{API}/health", headers=navigate), 403)
    assert_problem(client.get(f"{API}/assets/{'0' * 64}/content", headers=navigate), 403)
    assert_problem(client.get("/mcp", headers=navigate), 403)
    assert_problem(client.get("/", headers=fetch("cross-site", "navigate", "iframe")), 403)
    assert_problem(client.post("/", headers=navigate), 403)  # a form post is a navigation too


@pytest.mark.parametrize("headers", [fetch("same-origin", "cors", "empty"), fetch("none", "navigate", "document"), {}])
def test_own_and_non_browser_requests_pass(client, headers):
    assert client.get(f"{API}/health", headers=headers).status_code == 200
    assert client.post(f"{API}/estimate", json=JOB, headers=headers).status_code == 200


def test_mcp_is_protected_too(client):
    assert_problem(client.post("/mcp", json={}, headers={"Host": "evil.example"}), 403)
    assert_problem(client.post("/mcp", json={}, headers={"Origin": "https://evil.example"}), 403)

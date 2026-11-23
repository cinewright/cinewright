"""Browser protection for the local API and `/mcp` (ARCHITECTURE.md, Boundaries).

A web page in the user's browser can reach 127.0.0.1. The middleware refuses:
- any `Host` other than `127.0.0.1:<port>` / `localhost:<port>` (DNS rebinding),
- any `Origin` other than the core's own (cross-site requests; `Origin: null` included),
- requests a browser marks `Sec-Fetch-Site: cross-site` or `same-site`, such as `<img>`, `<script>`
  or no-cors `fetch()` from another page, which carry no `Origin`. Only a top-level navigation to
  the web UI (`Sec-Fetch-Mode: navigate`, `Sec-Fetch-Dest: document`, outside `/api` and `/mcp`)
  passes, so a link can still open Cinewright. Clients that send no `Sec-Fetch-*` headers (workers,
  the MCP bridge, scripts) are not browsers and pass,
- bodies with the CORS-safelisted types `multipart/form-data`, `application/x-www-form-urlencoded`
  and `text/plain`, the only ones a foreign page can send without a preflight.
Every response, refusals included, forbids framing (clickjacking), MIME sniffing and embedding by
other origins (`Cross-Origin-Resource-Policy`).
"""

from __future__ import annotations

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from cinewright.api.problems import problem_response

FORBIDDEN_BODY_TYPES = frozenset({"multipart/form-data", "application/x-www-form-urlencoded", "text/plain"})
FOREIGN_FETCH_SITES = frozenset({"cross-site", "same-site"})
API_PATH_PREFIXES = ("/api", "/mcp")
RESPONSE_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "content-security-policy": "frame-ancestors 'none'",
    "cross-origin-resource-policy": "same-origin",
}


def is_api_path(path: str) -> bool:
    """True for the internal API and the MCP server, which no foreign page may reach, not even by navigation."""
    return any(path == prefix or path.startswith(prefix + "/") for prefix in API_PATH_PREFIXES)


def _ui_navigation(headers: Headers, path: str, method: str) -> bool:
    return (
        method in ("GET", "HEAD")
        and (headers.get("sec-fetch-mode") or "").lower() == "navigate"
        and (headers.get("sec-fetch-dest") or "").lower() == "document"
        and not is_api_path(path)
    )


class LocalOnlyMiddleware:
    def __init__(self, app: ASGIApp, port: int) -> None:
        self.app = app
        self.port = port
        self.allowed_hosts = frozenset({f"127.0.0.1:{port}", f"localhost:{port}"})
        self.allowed_origins = frozenset({f"http://127.0.0.1:{port}", f"http://localhost:{port}"})

    def rejection(self, headers: Headers, path: str = "/", method: str = "GET") -> tuple[int, str, str] | None:
        """Status, title and detail of the refusal, or None when the request may pass."""
        host = (headers.get("host") or "").lower()
        if host not in self.allowed_hosts:
            return 403, "Forbidden", f"Host {host or '(none)'} is not allowed; address the API as 127.0.0.1:{self.port}."
        origin = headers.get("origin")
        if origin is not None and origin.lower() not in self.allowed_origins:
            return 403, "Forbidden", f"Origin {origin} is not allowed; only the Cinewright web UI may call the API."
        site = (headers.get("sec-fetch-site") or "").lower()
        if site in FOREIGN_FETCH_SITES and not _ui_navigation(headers, path, method):
            return 403, "Forbidden", f"Requests from other sites ({site}) are refused; open Cinewright at 127.0.0.1:{self.port}."
        media_type = (headers.get("content-type") or "").split(";")[0].strip().lower()
        if media_type in FORBIDDEN_BODY_TYPES:
            return 415, "Unsupported media type", f"Bodies of type {media_type} are refused; send application/json."
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in RESPONSE_HEADERS.items():
                    headers.setdefault(name, value)
            await send(message)

        refused = self.rejection(Headers(scope=scope), scope["path"], scope.get("method", "GET"))
        if refused is not None:
            if scope["type"] == "websocket":
                await receive()  # websocket.connect
                await send({"type": "websocket.close", "code": 1008})
                return
            status, title, detail = refused
            await problem_response(status, title, detail, "about:blank", scope["path"])(scope, receive, send_with_headers)
            return

        await self.app(scope, receive, send_with_headers if scope["type"] == "http" else send)

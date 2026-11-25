"""JSON-over-HTTP client for the worker protocol (`/api/v1/worker/...`), standard library only.

The client never follows redirects and never uses a proxy, so the bearer token only ever goes to the
core URL it was created with. Tokens are not part of any exception text or log line.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

WORKER_PREFIX = "/api/v1/worker"
DEFAULT_TIMEOUT = 30.0
LEASE_TIMEOUT_MARGIN = 15.0  # extra socket time on top of the long-poll wait
_CHUNK = 1024 * 1024
# Connection-level failures worth a retry: refused, reset, timed out, truncated response.
TRANSIENT_ERRORS = (OSError, http.client.HTTPException)


class ProtocolError(Exception):
    """The core answered with an error status. `problem` holds the RFC 9457 body when there is one."""

    def __init__(self, status: int, problem: dict[str, Any]) -> None:
        self.status = status
        self.problem = problem
        text = " ".join(str(part) for part in (problem.get("title"), problem.get("detail")) if part)
        super().__init__(f"HTTP {status}: {text}" if text else f"HTTP {status}")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, _req, _fp, _code, _msg, _headers, _newurl):  # urllib passes these positionally
        return None  # urllib then raises HTTPError with the 3xx status


class WorkerClient:
    def __init__(self, core_url: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        parts = urllib.parse.urlsplit(core_url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError(f"Core URL must be http(s)://host:port, got {core_url!r}")
        self.core_url = core_url.rstrip("/")
        self.timeout = timeout
        self.token: str | None = None
        self._origin = (parts.scheme, parts.netloc)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    # ------------------------------------------------------------------ protocol calls

    def register(self, registration: dict[str, Any], enrollment_token: str) -> dict[str, Any]:
        """`POST /worker/register` with the enrollment token; keeps the issued worker token."""
        registered = self._json("POST", "/register", registration, token=enrollment_token)
        self.token = registered["token"]
        return registered

    def heartbeat(self, health: dict[str, Any]) -> list[str]:
        """`POST /worker/heartbeat`; returns the job ids the core wants cancelled."""
        return list(self._json("POST", "/heartbeat", {"health": health}).get("cancel", []))

    def lease(self, wait_seconds: float) -> dict[str, Any] | None:
        """`POST /worker/lease` long-poll; returns the `WorkerJob` or None when no job came up."""
        body = self._json("POST", "/lease", {"wait_seconds": wait_seconds}, timeout=wait_seconds + LEASE_TIMEOUT_MARGIN)
        return body["job"] if body else None

    def post_events(self, job_id: str, lease_id: str, events: list[dict[str, Any]]) -> bool:
        """`POST /worker/jobs/{id}/events` (1 to 100 events); returns `cancel_requested`."""
        body = self._json("POST", f"/jobs/{_quote(job_id)}/events", {"lease_id": lease_id, "events": events})
        return bool(body.get("cancel_requested"))

    def upload_output(self, job_id: str, name: str, path: str, media_type: str) -> dict[str, Any]:
        """`PUT /worker/jobs/{id}/outputs/{name}` with the raw file as body; returns `OutputUploaded`."""
        size = os.path.getsize(path)
        with open(path, "rb") as body:
            status, data = self._send(
                "PUT",
                self._url(f"/jobs/{_quote(job_id)}/outputs/{_quote(name)}"),
                body,
                {"Content-Type": media_type, "Content-Length": str(size)},
            )
        return json.loads(data)

    def complete(
        self,
        job_id: str,
        lease_id: str,
        outputs: list[dict[str, Any]],
        engine_version: str | None,
        accepted_extra: list[str] | None = None,
    ) -> None:
        """`POST /worker/jobs/{id}/complete`; `accepted_extra` (the `params.extra` keys the engine applied) is sent only
        when the adapter reported them."""
        body: dict[str, Any] = {"lease_id": lease_id, "outputs": outputs, "engine_version": engine_version}
        if accepted_extra is not None:
            body["accepted_extra"] = accepted_extra
        self._json("POST", f"/jobs/{_quote(job_id)}/complete", body)

    def fail(self, job_id: str, lease_id: str, code: str, message: str, retryable: bool) -> None:
        error = {"code": code, "message": message, "retryable": retryable}
        self._json("POST", f"/jobs/{_quote(job_id)}/fail", {"lease_id": lease_id, "error": error})

    def download_input(self, url: str, sha256: str, dest: str) -> str:
        """Download an input asset to `dest` and check its SHA-256. `url` must point at the core."""
        target = urllib.parse.urljoin(self.core_url + "/", url)
        parts = urllib.parse.urlsplit(target)
        if (parts.scheme, parts.netloc) != self._origin:
            raise ValueError(f"Input URL {url!r} does not point at the core")
        digest = hashlib.sha256()
        request = urllib.request.Request(target, headers=self._headers(), method="GET")  # noqa: S310 - http(s) origin checked above
        try:
            with self._opener.open(request, timeout=self.timeout) as response, open(dest, "wb") as out:
                while True:
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    digest.update(chunk)
                    out.write(chunk)
        except urllib.error.HTTPError as exc:
            raise _protocol_error(exc) from None
        if digest.hexdigest() != sha256:
            os.remove(dest)
            raise ValueError(f"Input {sha256[:12]} does not match its SHA-256")
        return dest

    # ------------------------------------------------------------------ transport

    def _url(self, path: str) -> str:
        return f"{self.core_url}{WORKER_PREFIX}{path}"

    def _headers(self, token: str | None = None) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        bearer = token or self.token
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        return headers

    def _json(self, method: str, path: str, body: dict[str, Any], *, token: str | None = None, timeout: float | None = None) -> Any:
        data = json.dumps(body).encode("utf-8")
        _, payload = self._send(method, self._url(path), data, {"Content-Type": "application/json"}, token=token, timeout=timeout)
        return json.loads(payload) if payload else None

    def _send(
        self,
        method: str,
        url: str,
        data: Any,
        headers: dict[str, str],
        *,
        token: str | None = None,
        timeout: float | None = None,
    ) -> tuple[int, bytes]:
        # Always a URL under the core URL, whose http(s) scheme __init__ checked.
        request = urllib.request.Request(url, data=data, headers={**self._headers(token), **headers}, method=method)  # noqa: S310
        try:
            with self._opener.open(request, timeout=timeout or self.timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            raise _protocol_error(exc) from None


def _quote(segment: str) -> str:
    return urllib.parse.quote(segment, safe="")


def _protocol_error(exc: urllib.error.HTTPError) -> ProtocolError:
    try:
        problem = json.loads(exc.read() or b"{}")
    except (ValueError, OSError):
        problem = {}
    finally:
        exc.close()
    return ProtocolError(exc.code, problem if isinstance(problem, dict) else {})

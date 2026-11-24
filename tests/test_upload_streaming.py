"""Uploads stream to disk through a real server: bounded memory, a responsive core, no leftovers, upload slots.

Starlette's TestClient reads a whole request body before the app sees it, so these tests run the app on
uvicorn in a thread (as `test_e2e_mock.py` does) with the test registry and fake hardware. The client
makes every body lazily, so the test itself never holds one; tracemalloc then sees only what the core
(and the client's current chunk) holds.
"""

from __future__ import annotations

import hashlib
import os
import socket
import threading
import time
import tracemalloc
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from conftest import API, PNG, fake_detect

from cinewright.api import deps
from cinewright.app import create_app
from cinewright.config import Config
from cinewright.core.core import Core
from cinewright.core.registry import Registry

UPLOAD_BYTES = 128 * 1024**2
CHUNK_BYTES = 1024**2
RESPONSE_LIMIT_S = 1.0
WAIT_S = 30
STARTUP_TIMEOUT_S = 30
REGISTRATION = {"host_id": "local", "engine": "wan2gp", "engine_version": "test-1", "models": ["test_image"], "health": {"state": "ready"}}


@dataclass
class Live:
    url: str
    port: int
    core: Core

    def client(self, timeout: float = 30) -> httpx.Client:
        return httpx.Client(base_url=self.url, timeout=timeout, trust_env=False)

    def temporary_files(self) -> list[Path]:
        return list(self.core.paths.assets_dir.rglob(".upload-*"))

    def register_worker(self) -> dict[str, str]:
        with self.client() as http:
            response = http.post(
                f"{API}/worker/register", json=REGISTRATION, headers={"Authorization": f"Bearer {self.core.enrollment_token}"}
            )
        assert response.status_code == 201, response.text
        return {"Authorization": f"Bearer {response.json()['token']}"}

    def upload_target(self, route: str) -> tuple[str, str, dict[str, str]]:
        """Method, path and auth headers of an upload; for an output, of a job leased to a fresh worker."""
        if route == "asset":
            return "POST", f"{API}/assets", {}
        auth = self.register_worker()
        with self.client() as http:
            assert http.post(f"{API}/jobs", json={"capability": "t2i", "prompt": "a lighthouse"}).status_code == 202
            leased = http.post(f"{API}/worker/lease", json={"wait_seconds": 0}, headers=auth)
        assert leased.status_code == 200, leased.text
        return "PUT", f"{API}/worker/jobs/{leased.json()['job']['job_id']}/outputs/clip.mp4", auth


@pytest.fixture
def live(home: Path, registry_dir: Path) -> Iterator[Live]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    core = Core(Config(home=home, port=port), registry=Registry.load(registry_dir), detect=fake_detect)
    server = uvicorn.Server(uvicorn.Config(create_app(core), log_config=None, lifespan="on", timeout_graceful_shutdown=2))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True, name="uvicorn")
    thread.start()
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while not server.started:
        assert thread.is_alive(), "uvicorn stopped during startup"
        assert time.monotonic() < deadline, "uvicorn did not start in time"
        time.sleep(0.05)
    try:
        yield Live(f"http://127.0.0.1:{port}", port, core)
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        sock.close()
        core.close()
        assert not thread.is_alive(), "uvicorn did not stop"


@dataclass
class LazyBody:
    """`total` bytes made chunk by chunk while they are sent; the last chunk waits until `release` is set."""

    total: int
    release: threading.Event = field(default_factory=threading.Event)
    started: threading.Event = field(default_factory=threading.Event)  # a quarter of the body is out
    held: threading.Event = field(default_factory=threading.Event)  # everything but the last chunk is out
    sha256: Any = field(default_factory=hashlib.sha256)

    def __iter__(self) -> Iterator[bytes]:
        block = bytearray(os.urandom(min(CHUNK_BYTES, self.total)))
        count = -(-self.total // len(block))
        for index in range(count):
            if index == count - 1:
                self.held.set()
                assert self.release.wait(WAIT_S), "the test never released the last chunk"
            block[:8] = index.to_bytes(8, "big")
            chunk = bytes(block[: self.total - index * len(block)])
            self.sha256.update(chunk)
            yield chunk
            if (index + 1) * len(block) >= self.total // 4:
                self.started.set()


class Upload(threading.Thread):
    """Sends one upload in the background and keeps its response (or the error)."""

    def __init__(self, live: Live, method: str, path: str, auth: dict[str, str], body: LazyBody | bytes, media_type: str) -> None:
        super().__init__(daemon=True, name="upload")
        self.live, self.method, self.path, self.body = live, method, path, body
        size = body.total if isinstance(body, LazyBody) else len(body)
        self.headers = {**auth, "Content-Type": media_type, "Content-Length": str(size)}
        self.response: httpx.Response | None = None
        self.error: httpx.HTTPError | None = None
        self.done = threading.Event()

    def run(self) -> None:
        try:
            with self.live.client(timeout=120) as http:
                self.response = http.request(self.method, self.path, content=self.body, headers=self.headers)
        except httpx.HTTPError as exc:  # reported by result()
            self.error = exc
        finally:
            self.done.set()

    def result(self) -> httpx.Response:
        self.join(timeout=120)
        assert self.done.is_set(), "the upload did not finish"
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def timed(probe: Callable[[], None]) -> float:
    start = time.monotonic()
    probe()
    return time.monotonic() - start


def wait_for(condition: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + WAIT_S
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.02)


@pytest.mark.parametrize("route", ["asset", "output"])
def test_a_large_upload_streams_to_disk_in_bounded_memory_while_the_core_stays_responsive(live, route):
    method, path, auth = live.upload_target(route)
    prober = live.register_worker()  # a lease request from the uploading worker would release its own job
    body = LazyBody(UPLOAD_BYTES)
    upload = Upload(live, method, path, auth, body, "video/mp4")

    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        upload.start()
        assert body.started.wait(WAIT_S), "the upload did not start"
        with live.client() as http:

            def heartbeat() -> None:
                response = http.post(f"{API}/worker/heartbeat", json={"health": {"state": "ready"}}, headers=prober)
                assert response.status_code == 200, response.text

            def lease() -> None:
                response = http.post(f"{API}/worker/lease", json={"wait_seconds": 0}, headers=prober)
                assert response.status_code in (200, 204), response.text

            def events() -> None:
                with http.stream("GET", f"{API}/events") as response:
                    assert response.status_code == 200
                    assert next(response.iter_lines()).startswith(": cinewright events")

            durations = {probe.__name__: timed(probe) for probe in (heartbeat, lease, events)}
        assert not upload.done.is_set(), "the upload finished before the probes; it held back its last chunk"
        body.release.set()
        response = upload.result()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        body.release.set()
        if not was_tracing:
            tracemalloc.stop()

    assert all(seconds < RESPONSE_LIMIT_S for seconds in durations.values()), durations
    assert response.status_code == 201, response.text
    sha256 = body.sha256.hexdigest()
    assert (response.json()["sha256"], response.json()["size_bytes"]) == (sha256, UPLOAD_BYTES)
    assert live.core.blobs.path(sha256).stat().st_size == UPLOAD_BYTES
    assert not live.temporary_files()
    assert peak < UPLOAD_BYTES // 4, f"peak traced memory {peak / 1024**2:.1f} MiB for a {UPLOAD_BYTES // 1024**2} MiB upload"


@pytest.mark.parametrize("route", ["asset", "output"])
def test_an_upload_interrupted_by_the_client_leaves_no_temporary_file(live, route):
    method, path, auth = live.upload_target(route)
    head = [f"{method} {path} HTTP/1.1", f"Host: 127.0.0.1:{live.port}", "Content-Type: video/mp4", f"Content-Length: {64 * 1024**2}"]
    head += [f"{name}: {value}" for name, value in auth.items()]

    def written() -> bool:
        try:
            return any(temporary.stat().st_size > 0 for temporary in live.temporary_files())
        except FileNotFoundError:
            return False

    with socket.create_connection(("127.0.0.1", live.port)) as client:
        client.sendall(("\r\n".join(head) + "\r\n\r\n").encode())
        client.sendall(b"\x00" * (4 * CHUNK_BYTES))
        wait_for(written, "the core to write part of the upload")
    wait_for(lambda: not live.temporary_files(), "the core to remove the interrupted upload")
    assert not [p for p in live.core.paths.assets_dir.rglob("*") if p.is_file()]


def test_further_uploads_wait_for_a_free_slot(live, monkeypatch):
    monkeypatch.setattr(deps, "MAX_CONCURRENT_UPLOADS", 1)  # read when the first upload creates the app's slots
    held = LazyBody(4 * CHUNK_BYTES)
    first = Upload(live, "POST", f"{API}/assets", {}, held, "video/mp4")
    first.start()
    assert held.held.wait(WAIT_S), "the first upload did not start"
    wait_for(lambda: len(live.temporary_files()) == 1, "the first upload's temporary file")

    second = Upload(live, "POST", f"{API}/assets", {}, PNG, "image/png")
    second.start()
    assert not second.done.wait(0.5), "the second upload did not wait for the slot"
    assert len(live.temporary_files()) == 1

    held.release.set()
    assert first.result().status_code == 201
    stored = second.result()
    assert stored.status_code == 201, stored.text
    assert stored.json()["sha256"] == hashlib.sha256(PNG).hexdigest()
    assert not live.temporary_files()

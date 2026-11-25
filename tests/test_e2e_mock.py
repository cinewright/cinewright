"""Walking skeleton end to end over real HTTP.

The real app runs on uvicorn with its own lifespan (real registry, real hardware detection) and the
mock worker runs as a separate process (`python -m cinewright_worker --engine mock`), registering with
the core's enrollment token. A t2i job goes from submission to a stored PNG asset with provenance,
its events arrive over SSE, and the MCP server answers over streamable HTTP at `/mcp`.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import anyio
import httpx
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from PIL import Image

from cinewright.app import create_app
from cinewright.config import Config

API = "/api/v1"
# z-image-turbo is bound to Wan2GP's z_image, which the mock worker serves as a stand-in for Wan2GP.
MODEL = "z-image-turbo"
SEED = 1234
PROMPT = "A lighthouse on a cliff at dusk"
TOOLS = ["estimate", "fetch_asset", "job_status", "list_capabilities", "list_cast", "list_recipes", "submit"]
STARTUP_TIMEOUT_S = 60  # the lifespan runs hardware detection (nvidia-smi has a 10 s timeout)
JOB_TIMEOUT_S = 60


class LiveCore:
    def __init__(self, url: str, server: uvicorn.Server) -> None:
        self.url = url
        self.server = server
        self.http = httpx.Client(base_url=url, timeout=30, trust_env=False)

    @property
    def enrollment_token(self) -> str:
        return self.server.config.app.state.enrollment_token


@pytest.fixture
def live_core(home: Path) -> Iterator[LiveCore]:
    # Bind the socket first, so the port is known before the app is built and nothing can take it in between.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    app = create_app(config=Config(home=home, port=port))
    server = uvicorn.Server(uvicorn.Config(app, log_config=None, lifespan="on", timeout_graceful_shutdown=2))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True, name="uvicorn")
    thread.start()
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while not server.started:
        assert thread.is_alive(), "uvicorn stopped during startup"
        assert time.monotonic() < deadline, "uvicorn did not start in time"
        time.sleep(0.05)
    core = LiveCore(f"http://127.0.0.1:{port}", server)
    try:
        yield core
    finally:
        core.http.close()
        server.should_exit = True
        thread.join(timeout=15)
        sock.close()
        assert not thread.is_alive(), "uvicorn did not stop"


@pytest.fixture
def mock_worker(live_core: LiveCore, home: Path, tmp_path: Path) -> Iterator[Path]:
    log_path = tmp_path / "worker.log"
    env = {
        **os.environ,
        "CINEWRIGHT_HOME": str(home),
        "CINEWRIGHT_CORE_URL": live_core.url,
        "CINEWRIGHT_ENROLLMENT_TOKEN": live_core.enrollment_token,
        "PYTHONUNBUFFERED": "1",
    }
    with log_path.open("wb") as log:
        process = subprocess.Popen([sys.executable, "-m", "cinewright_worker", "--engine", "mock"], env=env, stdout=log, stderr=log)
        try:
            yield log_path
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def worker_log(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_sse(response: httpx.Response) -> list[dict]:
    """Parse a server-sent event stream into `{"id", "event", "data"}` records, skipping comments."""
    events, record = [], {}
    for line in response.iter_lines():
        if not line:
            if "data" in record:
                events.append({**record, "data": json.loads(record["data"])})
            record = {}
        elif not line.startswith(":"):
            field, _, value = line.partition(":")
            record[field] = value.removeprefix(" ")
    return events


async def mcp_session(url: str) -> tuple[list[str], dict]:
    async with streamable_http_client(f"{url}/mcp") as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        result = await session.call_tool("list_capabilities", {})
    assert not result.is_error, result.content
    return sorted(tool.name for tool in tools.tools), json.loads(result.content[0].text)


def test_t2i_job_runs_end_to_end_with_the_mock_worker(live_core: LiveCore, mock_worker: Path):
    http = live_core.http
    request = {"capability": "t2i", "prompt": PROMPT, "model": MODEL, "params": {"seed": SEED}}

    # Registration: the router finds a route as soon as the worker serves the model.
    deadline = time.monotonic() + JOB_TIMEOUT_S
    while True:
        estimate = http.post(f"{API}/estimate", json=request)
        assert estimate.status_code == 200, estimate.text
        if estimate.json()["feasible"]:
            break
        assert time.monotonic() < deadline, f"the mock worker did not register:\n{worker_log(mock_worker)}"
        time.sleep(0.2)
    assert estimate.json()["cost_eur"] == 0

    submitted = http.post(f"{API}/jobs", json=request)
    assert submitted.status_code == 202, submitted.text
    job_id = submitted.json()["id"]
    route = submitted.json()["route"]
    assert (route["model_id"], route["host_id"], route["engine"]) == (MODEL, "local", "wan2gp")

    # A job-filtered stream replays the job's history and ends after its terminal state.
    with http.stream("GET", f"{API}/events", params={"job_id": job_id}, timeout=JOB_TIMEOUT_S) as stream:
        assert stream.status_code == 200
        assert stream.headers["content-type"].startswith("text/event-stream")
        events = read_sse(stream)
    assert events, f"no events arrived:\n{worker_log(mock_worker)}"
    assert all(event["data"]["job_id"] == job_id for event in events)
    # Previews are live only: no SSE id, and the seq of the job's latest stored event.
    assert all("id" not in event for event in events if event["event"] == "preview")
    stored = [event for event in events if event["event"] != "preview"]
    assert [event["data"]["seq"] for event in stored] == list(range(1, len(stored) + 1))
    assert [event["id"] for event in stored] == [f"{job_id}:{event['data']['seq']}" for event in stored]
    states = [event["data"]["data"]["state"] for event in events if event["event"] == "state"]
    assert states[0] == "queued"
    assert "running" in states
    assert states[-1] == "succeeded", f"{states}\n{worker_log(mock_worker)}"
    progress = [event["data"]["data"] for event in events if event["event"] == "progress"]
    assert progress and progress[-1]["percent"] == 100

    job = http.get(f"{API}/jobs/{job_id}").json()
    assert job["state"] == "succeeded"
    assert job["client"] == "ui"
    assert len(job["outputs"]) == 1
    sha256 = job["outputs"][0]

    content = http.get(f"{API}/assets/{sha256}/content")
    assert content.status_code == 200
    assert content.headers["content-type"] == "image/png"
    assert hashlib.sha256(content.content).hexdigest() == sha256
    with Image.open(io.BytesIO(content.content)) as image:
        assert image.format == "PNG"
        image.verify()

    provenance = http.get(f"{API}/assets/{sha256}/provenance")
    assert provenance.status_code == 200
    record = provenance.json()
    assert record["schema"] == "cinewright.provenance/1"
    assert record["asset"] == {"sha256": sha256, "media_type": "image/png", "size_bytes": len(content.content)}
    assert record["seed"] == SEED
    assert record["prompt"] == PROMPT
    assert record["job_id"] == job_id
    assert (record["model"]["id"], record["model"]["engine"], record["model"]["engine_version"]) == (MODEL, "wan2gp", "mock-1")
    assert record["model"]["quant"] == route["quant"]
    assert (record["host"]["id"], record["host"]["kind"]) == ("local", "local")
    assert record["client"]["kind"] == "ui"

    tools, capabilities = anyio.run(mcp_session, live_core.url)
    assert tools == TOOLS
    assert MODEL in {item["model_id"] for item in capabilities["items"]}

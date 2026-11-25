"""Worker loop against an in-test fake core implementing the worker endpoints of openapi.json.

The fake validates every request body with the real contract models from `cinewright.api.schemas`,
the output name pattern and the upload media types, so a passing test means the worker speaks the
contract the core enforces.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import struct
import threading
import time
import urllib.parse
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import BaseModel, ValidationError

from cinewright.api import schemas
from cinewright.api.deps import UPLOAD_MEDIA_TYPES
from cinewright_worker.adapter import EngineJob, EngineStatus
from cinewright_worker.client import WorkerClient
from cinewright_worker.engines.mock import MockAdapter, encode_png, render_png
from cinewright_worker.loop import WorkerLoop, output_name

ENROLLMENT_TOKEN = "enroll-SECRET-1234"
WORKER_TOKEN = "worker-SECRET-5678"
OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PREFIX = "/api/v1/worker"


class FakeCore:
    """Minimal core: one lease queue, strict request validation, everything recorded."""

    def __init__(self) -> None:
        self.queue: list[schemas.WorkerJob] = []
        self.inputs: dict[str, bytes] = {}
        self.registrations: list[schemas.WorkerRegistration] = []
        self.heartbeats: list[schemas.Heartbeat] = []
        self.events: list[schemas.WorkerEvent] = []
        self.event_posts = 0
        self.uploads: dict[str, tuple[str, bytes]] = {}
        self.input_downloads: list[str] = []
        self.completed: list[schemas.JobCompletion] = []
        self.failed: list[schemas.JobFailure] = []
        self.violations: list[str] = []
        self.cancel_on_events = False
        self.heartbeat_cancel: list[str] = []
        self.events_status: int | None = None  # force an error status on the events endpoint
        self.complete_failures: list[int | str] = []  # per complete call: an error status, or "reset" (no response)
        self.complete_attempts = 0
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def shutdown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def add_input(self, data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        self.inputs[sha] = data
        return sha

    # ------------------------------------------------------------------ request handling

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        core = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                core.handle(self, "GET")

            def do_POST(self) -> None:
                core.handle(self, "POST")

            def do_PUT(self) -> None:
                core.handle(self, "PUT")

        return Handler

    def handle(self, request: BaseHTTPRequestHandler, method: str) -> None:
        length = int(request.headers.get("Content-Length") or 0)
        body = request.rfile.read(length) if length else b""
        path = urllib.parse.urlsplit(request.path).path
        try:
            status, payload, media_type = self.route(request, method, path, body)
        except ValidationError as exc:
            self.violations.append(f"{method} {path}: {exc}")
            status, payload, media_type = 422, {"title": "Invalid request", "status": 422}, "application/problem+json"
        if status == 0:  # drop the connection without an answer, as a reset or crashed core would
            request.close_connection = True
            return
        data = payload if isinstance(payload, bytes) else (json.dumps(payload).encode() if payload is not None else b"")
        request.send_response(status)
        if data:
            request.send_header("Content-Type", media_type)
        request.send_header("Content-Length", str(len(data)))
        request.end_headers()
        request.wfile.write(data)

    def route(self, request: BaseHTTPRequestHandler, method: str, path: str, body: bytes) -> tuple[int, object, str]:
        if not path.startswith(PREFIX):
            return 404, problem(404, "Not found"), "application/problem+json"
        path = path.removeprefix(PREFIX)
        token = (request.headers.get("Authorization") or "").removeprefix("Bearer ")
        if method == "POST" and path == "/register":
            if token != ENROLLMENT_TOKEN:
                return 401, problem(401, "Unauthorized"), "application/problem+json"
            self.registrations.append(parse(schemas.WorkerRegistration, body, request))
            registered = schemas.WorkerRegistered(worker_id="worker-1", token=WORKER_TOKEN, poll_seconds=0.02)
            return 201, registered.model_dump(mode="json"), "application/json"
        if token != WORKER_TOKEN:
            return 401, problem(401, "Unauthorized"), "application/problem+json"
        if method == "POST" and path == "/heartbeat":
            self.heartbeats.append(parse(schemas.Heartbeat, body, request))
            return 200, schemas.HeartbeatAck(cancel=self.heartbeat_cancel).model_dump(mode="json"), "application/json"
        if method == "POST" and path == "/lease":
            lease_request = parse(schemas.LeaseRequest, body, request)
            with self.lock:
                job = self.queue.pop(0) if self.queue else None
            if job is None:
                time.sleep(min(lease_request.wait_seconds, 0.05))
                return 204, None, ""
            return 200, schemas.Lease(job=job).model_dump(mode="json"), "application/json"
        if method == "GET" and path.startswith("/inputs/"):
            sha = path.removeprefix("/inputs/")
            self.input_downloads.append(sha)
            if sha not in self.inputs:
                return 404, problem(404, "Not found"), "application/problem+json"
            return 200, self.inputs[sha], "application/octet-stream"
        parts = path.split("/")  # "", "jobs", job_id, action[, name]
        if len(parts) < 4 or parts[1] != "jobs":
            return 404, problem(404, "Not found"), "application/problem+json"
        action = parts[3]
        if method == "POST" and action == "events":
            events = parse(schemas.WorkerEvents, body, request)
            for event in events.events:
                if event.kind == "progress":
                    schemas.Progress.model_validate(event.data)
            if self.events_status:
                return self.events_status, problem(self.events_status, "Conflict"), "application/problem+json"
            self.event_posts += 1
            self.events.extend(events.events)
            return 200, schemas.WorkerEventsAck(cancel_requested=self.cancel_on_events).model_dump(mode="json"), "application/json"
        if method == "PUT" and action == "outputs":
            name, media_type = urllib.parse.unquote(parts[4]), request.headers.get("Content-Type", "")
            if not OUTPUT_NAME.match(name):
                self.violations.append(f"bad output name {name!r}")
            if media_type not in UPLOAD_MEDIA_TYPES:
                self.violations.append(f"bad media type {media_type!r}")
            self.uploads[name] = (media_type, body)
            uploaded = schemas.OutputUploaded(sha256=hashlib.sha256(body).hexdigest(), size_bytes=len(body))
            return 201, uploaded.model_dump(mode="json"), "application/json"
        if method == "POST" and action == "complete":
            completion = parse(schemas.JobCompletion, body, request)
            uploaded = {hashlib.sha256(data).hexdigest() for _, data in self.uploads.values()}
            if any(output.sha256 not in uploaded for output in completion.outputs):
                self.violations.append("completion names an output that was not uploaded")
            with self.lock:
                self.complete_attempts += 1
                failure = self.complete_failures.pop(0) if self.complete_failures else None
            if failure == "reset":
                return 0, None, ""
            if failure is not None:
                return int(failure), problem(int(failure), "Unavailable"), "application/problem+json"
            self.completed.append(completion)
            self.closed.set()
            return 204, None, ""
        if method == "POST" and action == "fail":
            self.failed.append(parse(schemas.JobFailure, body, request))
            self.closed.set()
            return 204, None, ""
        return 404, problem(404, "Not found"), "application/problem+json"


def problem(status: int, title: str) -> dict:
    return schemas.Problem(title=title, status=status).model_dump(exclude_none=True)


def parse[M: BaseModel](model: type[M], body: bytes, request: BaseHTTPRequestHandler) -> M:
    if request.headers.get("Content-Type") != "application/json":
        raise AssertionError("worker sent a JSON body without Content-Type application/json")
    return model.model_validate_json(body)


def make_job(
    job_id: str = "job-1", engine_model: str = "mock-image", *, inputs: list[str] = (), params: dict | None = None, seed: int = 42
) -> schemas.WorkerJob:
    return schemas.WorkerJob(
        job_id=job_id,
        lease_id=f"lease-{job_id}",
        lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
        capability=schemas.Capability.I2V if inputs else schemas.Capability.T2I,
        model_id="mock-image",
        quant="bf16",
        engine_model=engine_model,
        prompt="A lighthouse at dusk",
        inputs=[schemas.WorkerInput(sha256=sha, media_type="image/png", url=f"{PREFIX}/inputs/{sha}") for sha in inputs],
        params=schemas.JobParams(**(params or {})),
        seed=seed,
    )


@pytest.fixture
def core():
    fake = FakeCore()
    yield fake
    fake.shutdown()
    assert fake.violations == []


def start_loop(core: FakeCore, tmp_path, adapter: MockAdapter | None = None, **options) -> tuple[WorkerLoop, threading.Thread, list]:
    loop = WorkerLoop(
        adapter or MockAdapter(step_seconds=0.002),
        WorkerClient(core.url, timeout=5),
        options.pop("enrollment_token", ENROLLMENT_TOKEN),
        lease_wait=0.2,
        poll_interval=0.01,
        event_interval=0.02,
        preview_interval=0,
        heartbeat_interval=0.02,
        register_timeout=2,
        retry_interval=0.05,
        work_dir=str(tmp_path),
        **options,
    )
    result: list = []
    thread = threading.Thread(target=lambda: result.append(loop.run()), daemon=True)
    thread.start()
    return loop, thread, result


def wait_closed(core: FakeCore, thread: threading.Thread, timeout: float = 10) -> None:
    assert core.closed.wait(timeout), "the worker neither completed nor failed the job"
    thread.join(timeout)
    assert not thread.is_alive()


def png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


# ---------------------------------------------------------------------- tests


def test_job_runs_to_completion(core, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    image = encode_png(2, 1, b"\xff\x00\x00\x00\xff\x00")
    sha = core.add_input(image)
    core.queue.append(make_job(inputs=[sha], params={"resolution": "1280x720", "steps": 6}))
    loop, thread, result = start_loop(core, tmp_path, once=True)
    wait_closed(core, thread)

    assert result == [0]
    registration = core.registrations[0]
    assert registration.host_id == "local"
    assert registration.engine == "mock"
    assert registration.models == ["mock-image", "mock-video"]
    assert registration.health.state == "ready"
    assert core.input_downloads == [sha]
    assert core.failed == []
    completion = core.completed[0]
    assert completion.lease_id == "lease-job-1"
    assert completion.engine_version == "mock-1"
    assert [(o.name, o.media_type) for o in completion.outputs] == [("image.png", "image/png")]
    media_type, data = core.uploads["image.png"]
    assert media_type == "image/png"
    assert png_size(data) == (256, 144)
    assert data == render_png(42, 256, 144)  # deterministic by seed
    progress = [e.data for e in core.events if e.kind == "progress"]
    assert progress[-1]["percent"] == 100.0
    assert any(p["step"] == 6 and p["total_steps"] == 6 for p in progress)
    previews = [e.data for e in core.events if e.kind == "preview"]
    assert previews and previews[0]["media_type"] == "image/png" and previews[0]["data_base64"]
    assert list(tmp_path.iterdir()) == []  # the job's work directory is removed
    assert ENROLLMENT_TOKEN not in caplog.text and WORKER_TOKEN not in caplog.text


class ReportingAdapter(MockAdapter):
    """The mock engine reporting which `params.extra` keys it applied, as the Wan2GP adapter does."""

    def __init__(self, accepted: tuple[str, ...]) -> None:
        super().__init__(step_seconds=0.002)
        self.accepted = accepted
        self.jobs: list[EngineJob] = []

    def submit(self, job: EngineJob) -> str:
        self.jobs.append(job)
        return super().submit(job)

    def status(self, engine_job_id: str) -> EngineStatus:
        status = super().status(engine_job_id)
        status.accepted_extra = self.accepted
        return status


def test_the_completion_names_the_extras_the_engine_applied(core, tmp_path):
    adapter = ReportingAdapter(("flow_shift", "x" * 65))  # a key the contract would refuse is left out
    core.queue.append(make_job(params={"steps": 2, "extra": {"flow_shift": 3.0, "seed_offset": 1}}))
    _, thread, _ = start_loop(core, tmp_path, adapter=adapter, once=True)
    wait_closed(core, thread)
    assert core.completed[0].accepted_extra == ["flow_shift"]
    assert [(job.capability, job.quant) for job in adapter.jobs] == [("t2i", "bf16")]  # from the WorkerJob


def test_an_adapter_that_reports_no_extras_completes_without_them(core, tmp_path):
    core.queue.append(make_job(params={"steps": 2, "extra": {"flow_shift": 3.0}}))
    _, thread, _ = start_loop(core, tmp_path, once=True)
    wait_closed(core, thread)
    assert core.completed[0].accepted_extra is None  # the mock engine reports none; provenance then records null


def test_cancel_from_events_acknowledgement(core, tmp_path):
    core.cancel_on_events = True
    core.queue.append(make_job(params={"steps": 200}))
    started = time.monotonic()
    _, thread, _ = start_loop(core, tmp_path, adapter=MockAdapter(step_seconds=0.05), once=True)
    wait_closed(core, thread)
    assert time.monotonic() - started < 5
    assert core.completed == []
    error = core.failed[0].error
    assert (error.code, error.retryable) == ("cancelled", False)


def test_cancel_from_heartbeat(core, tmp_path):
    core.heartbeat_cancel = ["job-1"]
    core.queue.append(make_job(params={"steps": 200}))
    _, thread, _ = start_loop(core, tmp_path, adapter=MockAdapter(step_seconds=0.05), once=True)
    wait_closed(core, thread)
    assert core.failed[0].error.code == "cancelled"
    assert core.heartbeats and core.heartbeats[0].health.state in ("ready", "busy")


def test_engine_failure_is_reported_with_its_code(core, tmp_path):
    core.queue.append(make_job(params={"steps": 4, "extra": {"mock_fail": "oom"}}))
    _, thread, _ = start_loop(core, tmp_path, once=True)
    wait_closed(core, thread)
    error = core.failed[0].error
    assert (error.code, error.retryable) == ("oom", False)
    assert "memory" in error.message
    assert core.completed == []


def test_unknown_engine_model_fails_without_running(core, tmp_path):
    core.queue.append(make_job(engine_model="not-served"))
    _, thread, _ = start_loop(core, tmp_path, once=True)
    wait_closed(core, thread)
    assert core.failed[0].error.code == "model_unavailable"


def test_input_with_wrong_hash_fails(core, tmp_path):
    sha = core.add_input(b"original")
    core.inputs[sha] = b"tampered"
    core.queue.append(make_job(inputs=[sha]))
    _, thread, _ = start_loop(core, tmp_path, once=True)
    wait_closed(core, thread)
    error = core.failed[0].error
    assert (error.code, error.retryable) == ("input_unavailable", False)


def test_lost_lease_drops_the_job_silently(core, tmp_path):
    core.events_status = 409
    core.queue.append(make_job(params={"steps": 50}))
    _, thread, result = start_loop(core, tmp_path, adapter=MockAdapter(step_seconds=0.01), once=True)
    thread.join(10)
    assert result == [0]
    assert core.completed == [] and core.failed == []


def test_complete_is_retried_after_a_reset_and_a_503(core, tmp_path):
    core.complete_failures = ["reset", 503]
    core.queue.append(make_job(params={"steps": 2}))
    _, thread, result = start_loop(core, tmp_path, once=True)
    wait_closed(core, thread)
    assert result == [0]
    assert core.complete_attempts == 3
    assert [(o.name, o.media_type) for c in core.completed for o in c.outputs] == [("image.png", "image/png")]
    assert core.failed == []


@pytest.mark.parametrize(("failures", "attempts"), [([503] * 1000, range(3, 1000)), ([422], range(1, 2))])
def test_closing_gives_up_after_close_timeout_or_a_refusal(core, tmp_path, failures, attempts):
    core.complete_failures = failures
    core.queue.append(make_job(params={"steps": 2}))
    started = time.monotonic()
    _, thread, result = start_loop(core, tmp_path, once=True, close_timeout=0.3)
    thread.join(10)
    assert result == [0] and core.completed == []
    assert core.complete_attempts in attempts
    assert time.monotonic() - started < 5


def test_input_urls_resolve_against_the_core_url_and_never_leave_it(core, tmp_path):
    sha = core.add_input(b"input bytes")
    client = WorkerClient(core.url, timeout=5)
    client.token = WORKER_TOKEN
    dest = tmp_path / "input.bin"
    assert client.download_input(f"{PREFIX}/inputs/{sha}", sha, str(dest)) == str(dest)  # path-only, as the core emits it
    assert dest.read_bytes() == b"input bytes"
    port = core.server.server_address[1]
    foreign = [
        f"http://localhost:{port}{PREFIX}/inputs/{sha}",
        f"https://127.0.0.1:{port}{PREFIX}/inputs/{sha}",
        f"http://127.0.0.1:1{PREFIX}/inputs/{sha}",
        f"http://evil.example{PREFIX}/inputs/{sha}",
        f"//evil.example{PREFIX}/inputs/{sha}",
    ]
    for url in foreign:
        with pytest.raises(ValueError, match="does not point at the core"):
            client.download_input(url, sha, str(tmp_path / "foreign.bin"))
    assert core.input_downloads == [sha]
    assert not (tmp_path / "foreign.bin").exists()


def test_shutdown_during_a_job_reports_it_retryable(core, tmp_path):
    core.queue.append(make_job(params={"steps": 200}))
    loop, thread, result = start_loop(core, tmp_path, adapter=MockAdapter(step_seconds=0.05))
    deadline = time.monotonic() + 5
    while not core.events and time.monotonic() < deadline:
        time.sleep(0.01)
    loop.stop.set()
    wait_closed(core, thread)
    assert result == [0]
    error = core.failed[0].error
    assert (error.code, error.retryable) == ("worker_stopped", True)


def test_stop_while_idle(core, tmp_path):
    loop, thread, result = start_loop(core, tmp_path)
    time.sleep(0.2)
    loop.stop.set()
    thread.join(5)
    assert result == [0]
    assert core.heartbeats, "the heartbeat thread posts while idle"


def test_refused_registration_exits_nonzero(core, tmp_path):
    _, thread, result = start_loop(core, tmp_path, enrollment_token="wrong")
    thread.join(5)
    assert result == [1]
    assert core.registrations == []


def test_output_names_match_the_core_pattern():
    used: set[str] = set()
    names = [output_name(raw, used) for raw in ("clip 01.mp4", ".hidden.png", "clip 01.mp4", "ünïcode.wav", "", "x" * 300 + ".png")]
    assert names[:4] == ["clip_01.mp4", "hidden.png", "clip_01-1.mp4", "n_code.wav"]
    assert len(set(names)) == len(names)
    assert all(OUTPUT_NAME.match(name) for name in names)

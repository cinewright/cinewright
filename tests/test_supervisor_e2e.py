"""Worker supervision end to end: a real core on uvicorn starts the mock worker through its supervisor.

The test interpreter stands in for the engine environment's Python (`WorkerLaunch(engine="mock")`). The
worker runs from the copy in `<data dir>/engines/worker/`, registers, shows its health in `GET /hosts`,
logs to `<data dir>/logs/worker.log`, restarts after a crash and on request, and is gone after shutdown.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import psutil
import pytest
import uvicorn
from conftest import fake_detect

from cinewright.app import create_app
from cinewright.config import Config
from cinewright.core.core import Core
from cinewright.core.supervisor import BOOTSTRAP, WorkerLaunch, WorkerSupervisor

API = "/api/v1"
TIMEOUT_S = 60
GRACE_S = 3.0
REQUIRED = {"CINEWRIGHT_CORE_URL", "CINEWRIGHT_ENROLLMENT_TOKEN", "CINEWRIGHT_HOME"}
# z-image-turbo is bound to Wan2GP's z_image, which the mock worker serves as a stand-in for Wan2GP.
JOB = {"capability": "t2i", "prompt": "A lighthouse on a cliff at dusk", "model": "z-image-turbo", "params": {"resolution": "512x512"}}


@dataclass
class Supervised:
    home: Path
    http: httpx.Client
    core: Core
    supervisor: WorkerSupervisor
    shutdown: Callable[[], None]

    @property
    def log(self) -> str:
        path = self.home / "logs" / "worker.log"
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

    def local_health(self) -> dict | None:
        hosts = self.http.get(f"{API}/hosts")
        assert hosts.status_code == 200, hosts.text
        return next(host for host in hosts.json()["items"] if host["id"] == "local")["worker_health"]

    def worker_ids(self) -> list[str]:
        return [row["id"] for row in self.core.db.query("SELECT id FROM workers ORDER BY registered_at")]

    def tree(self) -> list[psutil.Process]:
        """The worker process and its descendants: on Windows a hidden conhost.exe, and the interpreter a venv launcher starts."""
        pid = self.supervisor.pid
        if pid is None:
            return []
        try:
            root = psutil.Process(pid)
            return [root, *root.children(recursive=True)]
        except psutil.NoSuchProcess:
            return []


def worker_processes(tree: list[psutil.Process]) -> list[psutil.Process]:
    return [process for process in tree if "--stop-on-stdin-eof" in process.cmdline()]


def wait_for(predicate: Callable[[], object], what: str, context: Callable[[], str] = lambda: "", timeout: float = TIMEOUT_S):
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        assert time.monotonic() < deadline, f"timed out waiting for {what}\n{context()}"
        time.sleep(0.1)


@pytest.fixture
def supervised(home: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Iterator[Supervised]:
    """A core on uvicorn that supervises the mock worker; `request.param` sets the grace period."""
    monkeypatch.setenv("CINEWRIGHT_PORT", "1")  # the core's own variables must not reach the worker
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    core = Core(Config(home=home, port=port), detect=fake_detect)
    grace = getattr(request, "param", GRACE_S)
    supervisor = WorkerSupervisor(WorkerLaunch(python=Path(sys.executable), engine="mock"), grace_seconds=grace, backoff_initial=0.5)
    app = create_app(core, supervisor=supervisor)
    server = uvicorn.Server(uvicorn.Config(app, log_config=None, lifespan="on", timeout_graceful_shutdown=2))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True, name="uvicorn")
    thread.start()
    wait_for(lambda: server.started or not thread.is_alive(), "uvicorn to start")
    assert thread.is_alive(), "uvicorn stopped during startup"

    def shutdown() -> None:
        server.should_exit = True
        thread.join(timeout=grace + 30)
        assert not thread.is_alive(), "uvicorn did not stop"

    http = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30, trust_env=False)
    state = Supervised(home=home, http=http, core=core, supervisor=supervisor, shutdown=shutdown)
    leftovers: list[psutil.Process] = []
    try:
        yield state
    finally:
        leftovers = state.tree()
        http.close()
        if thread.is_alive():
            shutdown()
        sock.close()
        core.close()
        for process in leftovers:
            if process.is_running():
                process.kill()


def wait_ready(state: Supervised) -> dict:
    return wait_for(
        lambda: (health := state.local_health()) is not None and health["state"] == "ready" and health,
        "the local worker to report ready",
        lambda: state.log,
    )


def run_job(state: Supervised, **params) -> dict:
    submitted = state.http.post(f"{API}/jobs", json={**JOB, "params": {**JOB["params"], **params}})
    assert submitted.status_code == 202, submitted.text
    job_id = submitted.json()["id"]
    return wait_for(
        lambda: (job := state.http.get(f"{API}/jobs/{job_id}").json())["state"] in ("succeeded", "failed", "cancelled") and job,
        f"job {job_id} to finish",
        lambda: state.log,
    )


def test_the_core_starts_the_worker_shows_its_health_logs_its_output_and_stops_it(supervised: Supervised):
    health = wait_ready(supervised)
    assert health["detail"] == "Mock engine for tests"
    assert "Registered as worker" in supervised.log  # the worker's own log lines (stderr) land in the file
    copy = supervised.home / "engines" / "worker" / "cinewright_worker"
    assert list((copy / "__pycache__").glob("loop.*.pyc")), "the worker did not import the copy"

    tree = supervised.tree()
    workers = worker_processes(tree)
    assert workers and all(BOOTSTRAP in process.cmdline() for process in workers)
    env = workers[-1].environ()  # the interpreter that runs the worker
    # The worker takes the enrollment token out of its own environment at start; Windows shows the live block.
    assert {name.upper() for name in env if name.upper().startswith("CINEWRIGHT_")} | {"CINEWRIGHT_ENROLLMENT_TOKEN"} == REQUIRED
    assert env["CINEWRIGHT_HOME"] == str(supervised.home)
    assert {key.upper(): value for key, value in env.items()}["HTTP_PROXY"].startswith("http://127.0.0.1:")

    supervised.shutdown()
    _, alive = psutil.wait_procs(tree, timeout=5)
    assert alive == []
    assert supervised.supervisor.last_stop_graceful is True, supervised.log  # it stopped on the stdin request, without a kill
    assert "Standard input closed; stopping." in supervised.log


def test_a_killed_worker_restarts_with_backoff_and_the_queue_continues(supervised: Supervised):
    wait_ready(supervised)
    assert run_job(supervised)["state"] == "succeeded"
    first_ids = supervised.worker_ids()
    for process in worker_processes(supervised.tree()):
        process.kill()
    killed_at = time.monotonic()
    wait_for(lambda: supervised.supervisor.spawns == 2, "the supervisor to restart the worker", lambda: supervised.log)
    assert time.monotonic() - killed_at >= 0.4  # backoff_initial = 0.5 s
    wait_for(lambda: len(supervised.worker_ids()) > len(first_ids), "the restarted worker to register", lambda: supervised.log)
    wait_ready(supervised)

    job = run_job(supervised)
    assert job["state"] == "succeeded", job


def test_the_restart_call_registers_the_worker_again(supervised: Supervised):
    wait_ready(supervised)
    before = supervised.worker_ids()
    old_tree = supervised.tree()
    supervised.core.restart_local_worker()
    wait_for(lambda: len(supervised.worker_ids()) == len(before) + 1, "the worker to register again", lambda: supervised.log)
    assert supervised.supervisor.spawns == 2
    assert psutil.wait_procs(old_tree, timeout=5)[1] == []
    assert wait_ready(supervised)["state"] == "ready"


# Each refused connection costs about 2 s on Windows (SYN retries): the worker needs two of them after the
# stop request before it logs its first close retry, so this test waits longer before the kill.
@pytest.mark.parametrize("supervised", [8.0], indirect=True)
def test_shutdown_kills_a_worker_stuck_in_its_close_retry(supervised: Supervised):
    wait_ready(supervised)
    submitted = supervised.http.post(f"{API}/jobs", json={**JOB, "params": {**JOB["params"], "steps": 200}})
    assert submitted.status_code == 202, submitted.text
    job_id = submitted.json()["id"]
    wait_for(lambda: supervised.http.get(f"{API}/jobs/{job_id}").json()["state"] == "running", "the job to run", lambda: supervised.log)
    tree = supervised.tree()

    started = time.monotonic()
    supervised.shutdown()  # the core becomes unreachable while the job runs; the worker cannot report it
    _, alive = psutil.wait_procs(tree, timeout=5)
    assert alive == []
    assert time.monotonic() - started < supervised.supervisor.grace_seconds + 15  # not the close retry's 120 s
    assert "failed, retrying" in supervised.log  # the worker was in its close retry ...
    assert supervised.supervisor.last_stop_graceful is False  # ... and was killed after the grace period

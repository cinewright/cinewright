"""A model becomes routable within seconds of its install, without an app restart (#3).

The core runs on uvicorn with the fixture registry and fake hub of `test_core_modelstore`. Its local worker is the
real worker loop with the mock engine, run in this process by a stand-in for the supervisor. Like a Wan2GP session, the
engine reads `checkpoints_paths` from `wgp_config.json` when it starts (Wan2GP's import) and serves a model type while
one of its transformer files lies in one of those roots (upstream `files_locator.locate_file`), so the test also shows
that the store merged into the config is where Wan2GP finds the files. The core restarts the worker after an install or
a removal, and the new worker registers with its new model list.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path

import httpx
import pytest
import uvicorn
from conftest import API, fake_detect
from test_core_modelstore import FakeHub, fixture_registry

from cinewright.app import create_app
from cinewright.config import Config
from cinewright.core.core import Core
from cinewright.core.supervisor import SupervisorStatus
from cinewright_worker.adapter import EngineModel
from cinewright_worker.client import WorkerClient
from cinewright_worker.engines.mock import MockAdapter
from cinewright_worker.loop import WorkerLoop

TIMEOUT_S = 15
FEASIBLE_WITHIN_S = 5
MODEL, QUANT, MODEL_TYPE = "z-image-turbo", "int8", "z_image"
REQUEST = {"capability": "t2i", "prompt": "A lighthouse on a cliff at dusk", "model": MODEL, "quant": QUANT}


WGP_DEFAULTS = {"checkpoints_paths": ["ckpts", "."], "text_encoder_quantization": "int8", "attention_mode": "auto"}


class StoreFollowingAdapter(MockAdapter):
    """The mock engine as a Wan2GP session: serves a model type only while a transformer file of it lies in a checkpoint root.

    Like Wan2GP, it reads its checkpoint roots from `wgp_config.json` once, when it starts; a relative root is relative to
    the Wan2GP root, where the core's store never is.
    """

    def __init__(self, config_path: Path, files: dict[str, tuple[str, ...]]) -> None:
        super().__init__(models=tuple(files), engine="wan2gp")
        self.files = files
        root = config_path.parent / "wan2gp-root"
        roots = json.loads(config_path.read_text(encoding="utf-8"))["checkpoints_paths"]
        self.roots = [Path(entry) if os.path.isabs(entry) else root / entry for entry in roots]

    def capabilities(self) -> list[EngineModel]:
        return [
            EngineModel(
                engine_model=model_type,
                name=model_type,
                available=any((root / name).is_file() for root in self.roots for name in names),
            )
            for model_type, names in self.files.items()
        ]


class InProcessWorkers:
    """Stands in for the worker supervisor: runs the worker loop in a thread and replaces it on `restart()`."""

    def __init__(self, config_path: Path, files: dict[str, tuple[str, ...]]) -> None:
        self.config_path = config_path
        self.files = files
        self.spawned = 0
        self._lock = threading.Lock()
        self._current: tuple[WorkerLoop, threading.Thread] | None = None
        self._since: datetime | None = None

    def start(self, *, core_url: str, enrollment_token: str, paths: object, clock: Callable[[], datetime]) -> None:
        self._core_url, self._token, self._clock = core_url, enrollment_token, clock
        with self._lock:
            self._spawn()

    def _spawn(self) -> None:
        adapter = StoreFollowingAdapter(self.config_path, self.files)
        loop = WorkerLoop(adapter, WorkerClient(self._core_url), self._token, lease_wait=2.0, heartbeat_interval=1.0, retry_interval=0.2)
        thread = threading.Thread(target=loop.run, daemon=True, name=f"worker-{self.spawned}")
        self._since = self._clock()  # before the new worker registers
        self.spawned += 1
        thread.start()
        self._current = (loop, thread)

    def _end_current(self) -> None:
        if self._current is not None:
            loop, thread = self._current
            loop.stop.set()
            thread.join(timeout=TIMEOUT_S)
            assert not thread.is_alive(), "the worker loop did not stop"
            self._current = None

    def restart(self) -> None:
        def replace() -> None:
            with self._lock:
                self._end_current()
                self._spawn()

        threading.Thread(target=replace, daemon=True, name="worker-restart").start()

    def stop(self) -> None:
        with self._lock:
            self._end_current()

    def status(self) -> SupervisorStatus:
        return SupervisorStatus("running", since=self._since)


class Served:
    def __init__(self, url: str, core: Core, workers: InProcessWorkers, hub: FakeHub) -> None:
        self.http = httpx.Client(base_url=url, timeout=30, trust_env=False)
        self.core = core
        self.workers = workers
        self.hub = hub

    def estimate(self) -> dict:
        response = self.http.post(f"{API}/estimate", json=REQUEST)
        assert response.status_code == 200, response.text
        return response.json()

    def local_health(self) -> dict | None:
        hosts = self.http.get(f"{API}/hosts").json()["items"]
        return next(host for host in hosts if host["id"] == "local")["worker_health"]


def wait_for(predicate: Callable[[], object], what: str, timeout: float = TIMEOUT_S) -> float:
    started = time.monotonic()
    while not predicate():
        assert time.monotonic() - started < timeout, f"timed out waiting for {what}"
        time.sleep(0.05)
    return time.monotonic() - started


@pytest.fixture
def served(home: Path) -> Iterator[Served]:
    registry, contents = fixture_registry()
    hub = FakeHub(contents)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    config = Config(home=home, port=sock.getsockname()[1])
    core = Core(
        config, registry=registry, detect=fake_detect, download_transport=httpx.MockTransport(hub.handler), free_disk=lambda _: 10**12
    )
    model = registry.get(MODEL)
    assert model is not None
    core.models.config_path.parent.mkdir(parents=True, exist_ok=True)
    core.models.config_path.write_bytes(json.dumps(WGP_DEFAULTS, indent=4).encode())  # as Wan2GP's first start writes it
    workers = InProcessWorkers(core.models.config_path, {MODEL_TYPE: tuple(f.name for q in model.quantizations for f in q.files)})
    server = uvicorn.Server(uvicorn.Config(create_app(core, supervisor=workers), log_config=None, timeout_graceful_shutdown=2))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True, name="uvicorn")
    thread.start()
    wait_for(lambda: server.started or not thread.is_alive(), "uvicorn to start")
    assert thread.is_alive(), "uvicorn stopped during startup"
    environment = Served(config.base_url, core, workers, hub)
    try:
        yield environment
    finally:
        environment.http.close()
        server.should_exit = True
        thread.join(timeout=TIMEOUT_S + 10)
        core.close()
        sock.close()
        assert not thread.is_alive(), "uvicorn did not stop"


def test_an_estimate_turns_feasible_seconds_after_the_install_and_infeasible_after_the_removal(served):
    wait_for(lambda: (health := served.local_health()) is not None and health["state"] == "ready", "the worker to register")
    before = served.estimate()
    assert before["feasible"] is False
    assert before["reasons"] == [f"feasibility: {MODEL} {QUANT} is not installed on local"]

    response = served.http.post(f"{API}/models/{MODEL}/install", params={"quant": QUANT})
    assert response.status_code == 202, response.text
    took = wait_for(lambda: served.estimate()["feasible"], "the estimate to turn feasible", timeout=FEASIBLE_WITHIN_S)
    assert took < FEASIBLE_WITHIN_S
    assert served.estimate()["route"]["model_id"] == MODEL
    assert served.workers.spawned == 2  # the core restarted the worker once; the app did not restart
    config = json.loads(served.core.models.config_path.read_text(encoding="utf-8"))
    assert config["checkpoints_paths"] == [str(served.core.paths.models_dir.resolve()), "ckpts", "."]

    response = served.http.delete(f"{API}/models/{MODEL}", params={"quant": QUANT})
    assert response.status_code == 204, response.text
    wait_for(lambda: not served.estimate()["feasible"], "the estimate to turn infeasible", timeout=FEASIBLE_WITHIN_S)
    assert served.estimate()["reasons"] == [f"feasibility: {MODEL} {QUANT} is not installed on local"]
    wait_for(lambda: served.workers.spawned == 3, "the core to restart the worker")

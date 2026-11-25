"""The installed app starts the local worker on the provisioned engine once it is ready.

The core runs as the launcher runs it (`create_app(core, provision=True)`, served by uvicorn), with the mocked
downloads and the mocked uv of `test_core_provisioning`. Its supervisor gets the launch built from the provisioning
record, with the mock engine and the test interpreter swapped in for Wan2GP and the engine's Python, as in
`test_supervisor_e2e`.
"""

from __future__ import annotations

import dataclasses
import json
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import psutil
import pytest
import uvicorn
from conftest import API, BASE_URL, fake_detect
from fastapi.testclient import TestClient
from test_core_provisioning import ARCHIVE_URL, SOURCE, Setup, finish, unavailable_sage

from cinewright.app import create_app
from cinewright.config import Config
from cinewright.core.core import Core
from cinewright.core.provisioning import PROBE_SENTINEL
from cinewright.core.supervisor import WorkerLaunch, WorkerSupervisor
from cinewright_worker.engines.wan2gp import Wan2GPAdapter, attention_mode, read_provisioning_record

TIMEOUT_S = 60
GRACE_S = 3.0


class MockWorkers:
    """Builds each supervisor from the launch the core derived, running the mock engine with this interpreter."""

    def __init__(self) -> None:
        self.launches: list[WorkerLaunch] = []
        self.states: list[str] = []  # the engine state when the core asked for a supervisor
        self.attention: list[str] = []  # the attention mode the record held then
        self.supervisors: list[WorkerSupervisor] = []
        self.core: Core | None = None

    def __call__(self, launch: WorkerLaunch) -> WorkerSupervisor:
        assert self.core is not None
        self.launches.append(launch)
        self.states.append(self.core.provisioner.status().state)
        self.attention.append(attention_mode(read_provisioning_record(str(self.core.paths.root))))
        mock = dataclasses.replace(launch, python=Path(sys.executable), engine="mock")
        supervisor = WorkerSupervisor(mock, grace_seconds=GRACE_S, backoff_initial=0.5)
        self.supervisors.append(supervisor)
        return supervisor

    def trees(self) -> list[psutil.Process]:
        processes = []
        for supervisor in self.supervisors:
            pid = supervisor.pid
            if pid is None:
                continue
            try:
                root = psutil.Process(pid)
                processes += [root, *root.children(recursive=True)]
            except psutil.NoSuchProcess:
                pass
        return processes


class Installed:
    """A core with the fake provisioner and the mock workers, on a free loopback port."""

    def __init__(self, setup: Setup) -> None:
        self.setup = setup
        self.workers = MockWorkers()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.core = Core(
            Config(home=setup.paths.root, port=self.port),
            detect=fake_detect,
            provisioner=setup.provisioner(),
            worker_supervisor=self.workers,
        )
        self.workers.core = self.core
        self.http = httpx.Client(base_url=f"http://127.0.0.1:{self.port}", timeout=30, trust_env=False)

    @contextmanager
    def serving(self) -> Iterator[Callable[[], None]]:
        """Serve the app as the launcher does; yields a function that shuts it down."""
        server = uvicorn.Server(uvicorn.Config(create_app(self.core, provision=True), log_config=None, timeout_graceful_shutdown=2))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [self.sock]}, daemon=True, name="uvicorn")
        thread.start()
        wait_for(lambda: server.started or not thread.is_alive(), "uvicorn to start")
        assert thread.is_alive(), "uvicorn stopped during startup"

        def shutdown() -> None:
            server.should_exit = True
            thread.join(timeout=GRACE_S + 30)
            assert not thread.is_alive(), "uvicorn did not stop"

        try:
            yield shutdown
        finally:
            if thread.is_alive():
                shutdown()

    @property
    def log(self) -> str:
        path = self.setup.paths.logs_dir / "worker.log"
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

    def local_health(self) -> dict | None:
        hosts = self.http.get(f"{API}/hosts")
        assert hosts.status_code == 200, hosts.text
        return next(host for host in hosts.json()["items"] if host["id"] == "local")["worker_health"]

    def wait_worker_ready(self) -> dict:
        return wait_for(
            lambda: (health := self.local_health()) is not None and health["state"] == "ready" and health,
            "the local worker to report ready",
            lambda: self.log,
        )

    def close(self) -> None:
        leftovers = self.workers.trees()
        self.http.close()
        self.core.close()
        self.sock.close()
        for process in leftovers:
            if process.is_running():
                process.kill()


def wait_for(predicate: Callable[[], object], what: str, context: Callable[[], str] = lambda: "", timeout: float = TIMEOUT_S):
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        assert time.monotonic() < deadline, f"timed out waiting for {what}\n{context()}"
        time.sleep(0.1)


def provision(setup: Setup) -> dict:
    """A completed provisioning in the data directory, as an earlier start left it; returns its record."""
    provisioner = setup.provisioner()
    provisioner.start()
    assert finish(provisioner).state == "ready"
    record = read_provisioning_record(str(setup.paths.root))
    assert record is not None
    return record


def record_launch(record: dict) -> WorkerLaunch:
    return WorkerLaunch(python=Path(record["python"]), engine="wan2gp", wan2gp_root=Path(record["root"]))


@pytest.fixture
def setup(tmp_path) -> Setup:
    return Setup(tmp_path)


@pytest.fixture
def installed(setup) -> Iterator[Callable[[], Installed]]:
    """Builds the core when called, so it sees what the data directory holds at that point."""
    states: list[Installed] = []

    def build() -> Installed:
        states.append(Installed(setup))
        return states[-1]

    try:
        yield build
    finally:
        for state in states:
            state.close()


def test_the_launch_comes_from_the_record_and_the_worker_finds_its_attention_mode(setup):
    provisioner = setup.provisioner()
    assert provisioner.worker_launch() is None  # not installed
    provisioner.start()
    assert finish(provisioner).state == "ready"
    record = read_provisioning_record(str(setup.paths.root))
    launch = provisioner.worker_launch()
    assert launch == record_launch(record)
    assert launch.worker_args() == ["--engine", "wan2gp", "--wan2gp-root", record["root"], "--stop-on-stdin-eof"]

    # The worker the launch starts reads the attention mode from the same record, as its root matches.
    record_path = setup.engine_dir / "provisioning.json"
    record_path.write_text(record_path.read_text(encoding="utf-8").replace('"attention": "sdpa"', '"attention": "sage2"'), "utf-8")
    adapter = Wan2GPAdapter(root=str(launch.wan2gp_root), data_dir=str(setup.paths.root))
    assert adapter._cli_args == ("--attention", "sage2", "--offline")

    (setup.root / "wgp.py").unlink()  # the installation is gone: no launch
    assert provisioner.worker_launch() is None


def test_a_start_with_a_ready_record_starts_the_worker_and_shutdown_stops_it(setup, installed):
    record = provision(setup)
    requests, runs = len(setup.server.requests), len(setup.runner.calls)
    installed = installed()
    assert installed.core.provisioner.status().state == "ready"
    with installed.serving() as shutdown:
        health = installed.wait_worker_ready()
        assert health["detail"] == "Mock engine for tests"
        assert installed.workers.launches == [record_launch(record)]
        assert installed.workers.states == ["ready"]
        assert installed.core.supervisor is installed.workers.supervisors[0]
        tree = installed.workers.trees()
        assert any(record["root"] in process.cmdline() for process in tree)  # --wan2gp-root of the record
        assert (len(setup.server.requests), len(setup.runner.calls)) == (requests, runs)  # nothing downloaded or installed

        shutdown()
    _, alive = psutil.wait_procs(tree, timeout=5)
    assert alive == []
    assert installed.core.supervisor is None
    assert installed.workers.supervisors[0].status().state == "stopped"
    assert installed.workers.supervisors[0].last_stop_graceful is True, installed.log


def test_a_provisioning_run_that_reaches_ready_starts_the_worker(installed):
    installed = installed()
    with installed.serving():
        installed.wait_worker_ready()
        assert installed.workers.states == ["ready"]  # never while provisioning ran
        record = read_provisioning_record(str(installed.setup.paths.root))
        assert installed.workers.launches == [record_launch(record)]
        assert installed.http.get(f"{API}/system").json()["engines"][0]["state"] == "ready"


def test_a_failed_run_starts_no_worker_and_the_retry_that_reaches_ready_does(installed):
    installed = installed()
    installed.setup.server.files[ARCHIVE_URL] = b"not the pinned archive"
    with installed.serving():
        assert finish(installed.core.provisioner).state == "failed"
        time.sleep(0.5)
        assert installed.workers.launches == [] and installed.core.supervisor is None
        assert installed.local_health() is None

        installed.setup.server.files[ARCHIVE_URL] = SOURCE
        assert installed.http.post(f"{API}/engines/wan2gp/provision").status_code == 202
        installed.wait_worker_ready()
        assert installed.workers.states == ["ready"]


def test_a_start_that_tries_an_optional_wheel_again_starts_the_worker_after_that_pass(setup, installed, monkeypatch):
    unavailable_sage(setup, monkeypatch)
    setup.runner.probe = PROBE_SENTINEL + json.dumps({"ok": True, "version": "2.2.0"})
    installed = installed()
    with installed.serving():
        installed.wait_worker_ready()
        assert installed.workers.states == ["ready"]
        assert installed.workers.attention == ["sage2"]  # the worker starts on the record that pass wrote


def test_no_worker_starts_while_provisioning_runs(setup):
    workers = MockWorkers()
    core = Core(Config(home=setup.paths.root), detect=fake_detect, provisioner=setup.provisioner(), worker_supervisor=workers)
    workers.core = core
    setup.runner.block = threading.Event()  # the lock install runs until the app stops
    try:
        with TestClient(create_app(core, provision=True), base_url=BASE_URL):
            assert setup.runner.block.wait(10)
            assert core.provisioner.status().state == "provisioning"
            time.sleep(0.3)
            assert workers.launches == [] and core.supervisor is None
        assert core.provisioner.status().state == "not_installed"
        assert workers.launches == [] and core.supervisor is None
    finally:
        core.close()


def test_without_the_launcher_flag_no_worker_starts(setup):
    provision(setup)
    workers = MockWorkers()
    core = Core(Config(home=setup.paths.root), detect=fake_detect, provisioner=setup.provisioner(), worker_supervisor=workers)
    workers.core = core
    try:
        with TestClient(create_app(core), base_url=BASE_URL):
            assert core.provisioner.status().state == "ready"
            assert workers.launches == [] and core.supervisor is None
    finally:
        core.close()


def test_a_run_that_reaches_ready_after_the_stop_starts_no_worker(setup):
    workers = MockWorkers()
    core = Core(Config(home=setup.paths.root), detect=fake_detect, provisioner=setup.provisioner(), worker_supervisor=workers)
    workers.core = core
    try:
        core.run_local_worker_when_ready("http://127.0.0.1:1")
        assert workers.launches == []  # not installed yet
        core.stop_local_worker()  # shutdown begins
        core.start_provisioning()
        assert finish(core.provisioner).state == "ready"
        core.provisioner.stop()  # waits for its thread, which runs the callbacks of the ready state
        assert workers.launches == [] and core.supervisor is None
    finally:
        core.close()

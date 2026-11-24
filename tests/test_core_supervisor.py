"""Worker supervision without a running core: spawn command, bootstrap, package copy, environment, proxies,
process lifetime, the stdin stop channel, the shutdown's lease stop and the worker health of `GET /hosts`."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import psutil
import pytest
from conftest import fake_detect

import cinewright_worker
from cinewright.api.schemas import ClientKind, JobRequest, LeaseRequest, WorkerRegistration
from cinewright.core.core import Core
from cinewright.core.registry import Registry
from cinewright.core.supervisor import (
    BOOTSTRAP,
    NO_PROXY,
    SupervisorStatus,
    WorkerLaunch,
    copy_worker_package,
    reserve_closed_port,
    spawn_environment,
    worker_command,
)

SRC = Path(__file__).resolve().parents[1] / "src"
# The interpreter itself, not a venv launcher that starts it as a child, so a test sees one process.
BASE_PYTHON = getattr(sys, "_base_executable", None) or sys.executable
REQUIRED = {"CINEWRIGHT_CORE_URL", "CINEWRIGHT_ENROLLMENT_TOKEN", "CINEWRIGHT_HOME"}


def cinewright_names(env) -> set[str]:
    return {name.upper() for name in env if name.upper().startswith("CINEWRIGHT_")}


def make_env(platform: str = sys.platform, base: dict[str, str] | None = None, port: int = 9) -> dict[str, str]:
    base = base if base is not None else dict(os.environ)
    return spawn_environment(
        base, core_url="http://127.0.0.1:8737", enrollment_token="enroll-1", home=Path("/data"), proxy_port=port, platform=platform
    )


# --------------------------------------------------------------------------- command, bootstrap, copy


def test_worker_command_runs_the_bootstrap_isolated_with_the_copy_directory():
    launch = WorkerLaunch(python=Path("/engine/python"), engine="wan2gp", wan2gp_root=Path("/data/engines/wan2gp/abc"))
    assert worker_command(launch, Path("/data/engines/worker")) == [
        str(Path("/engine/python")),
        "-I",
        "-u",
        "-c",
        BOOTSTRAP,
        str(Path("/data/engines/worker")),
        "--engine",
        "wan2gp",
        "--wan2gp-root",
        str(Path("/data/engines/wan2gp/abc")),
        "--stop-on-stdin-eof",
    ]
    assert WorkerLaunch(python=Path("p"), engine="mock").worker_args() == ["--engine", "mock", "--stop-on-stdin-eof"]


def test_bootstrap_imports_the_copy_before_site_packages_and_hides_its_argument(tmp_path):
    # The test interpreter has the real cinewright_worker installed; the bootstrap must still run the copy.
    package = tmp_path / "copy" / "cinewright_worker"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "__main__.py").write_text(
        "import json, sys\nprint(json.dumps({'file': __file__, 'argv': sys.argv[1:], 'name': __name__}))\n", encoding="utf-8"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", BOOTSTRAP, str(tmp_path / "copy"), "--engine", "mock"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert Path(result["file"]).resolve().parent == package.resolve()
    assert result["argv"] == ["--engine", "mock"]
    assert result["name"] == "__main__"


def test_the_copy_holds_only_the_worker_package_and_replaces_an_older_copy(tmp_path):
    root = tmp_path / "engines" / "worker"
    (root / "cinewright_worker").mkdir(parents=True)
    (root / "cinewright_worker" / "stale.py").write_text("old = True\n", encoding="utf-8")
    assert copy_worker_package(root) == root
    source = Path(cinewright_worker.__file__).parent
    expected = {path.relative_to(source).as_posix() for path in source.rglob("*.py")}
    copied = {path.relative_to(root / "cinewright_worker").as_posix() for path in (root / "cinewright_worker").rglob("*") if path.is_file()}
    assert copied == expected
    assert [path.name for path in root.iterdir()] == ["cinewright_worker"]


# --------------------------------------------------------------------------- environment and proxies


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_spawn_environment_holds_only_the_three_cinewright_variables(platform):
    base = {
        "PATH": "/usr/bin",
        "CINEWRIGHT_HOME": "/elsewhere",
        "CINEWRIGHT_PORT": "9999",
        "CINEWRIGHT_ENROLLMENT_TOKEN": "stale",
        "cinewright_extra": "x",
        "HTTP_PROXY": "http://corp-proxy:3128",
        "https_proxy": "http://corp-proxy:3128",
        "ALL_PROXY": "socks5://corp-proxy:1080",
        "no_proxy": "*",
    }
    env = make_env(platform, base, port=4321)
    assert cinewright_names(env) == REQUIRED
    assert (env["CINEWRIGHT_CORE_URL"], env["CINEWRIGHT_ENROLLMENT_TOKEN"], env["CINEWRIGHT_HOME"]) == (
        "http://127.0.0.1:8737",
        "enroll-1",
        str(Path("/data")),
    )
    assert env["PATH"] == "/usr/bin"
    proxies = {name: value for name, value in env.items() if name.lower().endswith("_proxy")}
    upper = {"HTTP_PROXY": "http://127.0.0.1:4321", "HTTPS_PROXY": "http://127.0.0.1:4321", "NO_PROXY": NO_PROXY}
    if platform == "win32":
        assert proxies == upper
    else:  # case-sensitive names; urllib prefers the lower-case ones
        assert proxies == {**upper, **{name.lower(): value for name, value in upper.items()}}


PROXY_PROBE = """
import json, os, urllib.error, urllib.request
result = {"cinewright": sorted(n for n in os.environ if n.upper().startswith("CINEWRIGHT_"))}
for url in ("http://example.com/", "https://example.com/"):
    try:
        urllib.request.urlopen(url, timeout=20)
        result[url] = "fetched"
    except urllib.error.URLError as exc:
        result[url] = type(exc.reason).__name__
print(json.dumps(result))
"""


def test_a_child_with_the_spawn_environment_cannot_fetch_external_urls(monkeypatch):
    monkeypatch.setenv("CINEWRIGHT_PORT", "1")  # must not reach the child
    with reserve_closed_port() as port:
        env = make_env(port=port.getsockname()[1])
        completed = subprocess.run([BASE_PYTHON, "-I", "-c", PROXY_PROBE], env=env, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert set(result["cinewright"]) == REQUIRED
    # Refused by the closed proxy port: neither fetched nor a DNS failure, so no external network is needed.
    assert result["http://example.com/"] == "ConnectionRefusedError"
    assert result["https://example.com/"] == "ConnectionRefusedError"


# --------------------------------------------------------------------------- process lifetime

HOST_SPAWN_BOUND = """
import subprocess, sys, time
sys.path.insert(0, sys.argv[1])
from cinewright.core.supervisor import spawn_bound
child = spawn_bound([sys.argv[2], "-I", "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL)
print(child.pid, flush=True)
time.sleep(120)
"""

HOST_SUPERVISOR = """
import os, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from cinewright.core.supervisor import WorkerLaunch, WorkerSupervisor, reserve_closed_port
from cinewright.paths import DataPaths
closed = reserve_closed_port()
supervisor = WorkerSupervisor(WorkerLaunch(python=Path(sys.executable), engine="mock"))
supervisor.start(core_url=f"http://127.0.0.1:{closed.getsockname()[1]}", enrollment_token="t", paths=DataPaths(Path(sys.argv[2])))
while supervisor.pid is None:
    time.sleep(0.05)
time.sleep(1)  # the worker process (and a venv launcher's child) is up
print(supervisor.pid, flush=True)
time.sleep(120)
"""


def start_host(script: str, *args: str) -> tuple[subprocess.Popen, psutil.Process, list[psutil.Process]]:
    """Run `script` in a host process; returns it, its interpreter process and the process tree whose pid it prints."""
    host = subprocess.Popen([sys.executable, "-c", script, str(SRC), *args], stdout=subprocess.PIPE, text=True)
    line = host.stdout.readline().strip()
    assert line.isdigit(), f"host printed {line!r}"
    child = psutil.Process(int(line))
    interpreter = child.parent()  # the process that ran spawn_bound (a venv launcher's child, if there is one)
    return host, interpreter, [child, *child.children(recursive=True)]


def kill_and_expect_gone(host: subprocess.Popen, interpreter: psutil.Process, tree: list[psutil.Process]) -> None:
    assert all(process.is_running() for process in tree)
    try:
        interpreter.kill()  # TerminateProcess on Windows, SIGKILL on Linux: no cleanup code runs
        _, alive = psutil.wait_procs(tree, timeout=10)
        assert alive == [], f"still running after the host was killed: {alive}"
    finally:
        for process in tree:
            if process.is_running():
                process.kill()
        host.kill()
        host.wait(timeout=10)


@pytest.mark.skipif(not (sys.platform == "win32" or sys.platform.startswith("linux")), reason="Job Object or PR_SET_PDEATHSIG")
def test_a_bound_child_ends_when_its_host_is_killed_even_if_it_ignores_stdin():
    host, interpreter, tree = start_host(HOST_SPAWN_BOUND, BASE_PYTHON)
    kill_and_expect_gone(host, interpreter, tree)


@pytest.mark.skipif(not (sys.platform == "win32" or sys.platform.startswith("linux")), reason="Job Object or PR_SET_PDEATHSIG")
def test_killing_a_process_that_runs_the_supervisor_ends_its_worker(tmp_path):
    host, interpreter, tree = start_host(HOST_SUPERVISOR, str(tmp_path / "home"))
    assert any("--engine" in process.cmdline() for process in tree)
    kill_and_expect_gone(host, interpreter, tree)


# --------------------------------------------------------------------------- the worker's stdin stop channel


@pytest.fixture
def worker_env():
    """A worker environment whose core URL is a closed port, so the worker keeps retrying its registration."""
    with reserve_closed_port() as closed:
        url = f"http://127.0.0.1:{closed.getsockname()[1]}"
        yield {**os.environ, "CINEWRIGHT_ENROLLMENT_TOKEN": "enroll-1", "CINEWRIGHT_CORE_URL": url, "PYTHONUNBUFFERED": "1"}


def test_a_worker_started_by_hand_ignores_stdin_at_end_of_file(worker_env):
    worker = subprocess.Popen(
        [sys.executable, "-m", "cinewright_worker", "--engine", "mock"],
        env=worker_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(3)
        assert worker.poll() is None  # still retrying the registration, as before supervision existed
    finally:
        worker.kill()
        worker.wait(timeout=10)


def test_a_worker_started_with_stop_on_stdin_eof_stops_when_stdin_closes(worker_env):
    worker = subprocess.Popen(
        [sys.executable, "-m", "cinewright_worker", "--engine", "mock", "--stop-on-stdin-eof"],
        env=worker_env,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(1.5)
        assert worker.poll() is None
        worker.stdin.close()
        worker.wait(timeout=15)
        assert b"Standard input closed; stopping." in worker.stderr.read()
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=10)


# --------------------------------------------------------------------------- core: shutdown and host health

REGISTRATION = WorkerRegistration(host_id="local", engine="wan2gp", models=["test_image"], health={"state": "ready", "vram_used_mb": 812})


class StubSupervisor:
    def __init__(self, status: SupervisorStatus) -> None:
        self.current = status
        self.restarts = 0

    def status(self) -> SupervisorStatus:
        return self.current

    def restart(self) -> None:
        self.restarts += 1


def test_after_stop_leasing_no_job_is_granted_and_waiting_polls_return(core):
    worker = core.register_worker(REGISTRATION, core.enrollment_token).worker_id

    async def scenario():
        waiting = asyncio.create_task(core.lease(worker, LeaseRequest(wait_seconds=10)))
        await asyncio.sleep(0.1)
        started = time.monotonic()
        core.stop_leasing()
        result = await asyncio.wait_for(waiting, 5)
        return result, time.monotonic() - started

    result, elapsed = asyncio.run(scenario())
    assert result is None and elapsed < 2
    core.submit_job(JobRequest(capability="t2i", prompt="a lighthouse"), ClientKind.UI, None)
    assert asyncio.run(core.lease(worker, LeaseRequest(wait_seconds=0))) is None


def local_health(core):
    return next(host for host in core.list_hosts().items if host.id == "local").worker_health


def test_hosts_show_the_health_of_the_newest_worker_of_this_core_start(core, config, registry_dir, clock):
    assert local_health(core) is None
    core.register_worker(REGISTRATION, core.enrollment_token)
    health = local_health(core)
    assert (health.state, health.vram_used_mb) == ("ready", 812)
    newer = WorkerRegistration.model_validate(
        {**REGISTRATION.model_dump(), "health": {"state": "busy", "vram_used_mb": 9000, "vram_total_mb": 16303}}
    )
    core.register_worker(newer, core.enrollment_token)
    assert (local_health(core).state, local_health(core).vram_used_mb) == ("busy", 9000)

    # A new core start does not show the workers of the previous one, although their rows are recent.
    restarted = Core(config, registry=Registry.load(registry_dir), detect=fake_detect, clock=clock)
    try:
        assert local_health(restarted) is None
    finally:
        restarted.close()

    clock.advance(121)  # stale workers show no health
    assert local_health(core) is None


def test_the_supervisor_overrides_the_local_health_while_the_process_is_not_registered(core, clock):
    stub = StubSupervisor(SupervisorStatus("starting", detail="The worker is starting."))
    core.supervisor = stub
    assert (local_health(core).state, local_health(core).detail) == ("starting", "The worker is starting.")

    stub.current = SupervisorStatus("down", detail="The worker exited with code 1; it restarts in 2 s.")
    assert (local_health(core).state, local_health(core).detail) == ("failed", "The worker exited with code 1; it restarts in 2 s.")

    core.register_worker(REGISTRATION, core.enrollment_token)
    stub.current = SupervisorStatus("running", since=clock.now + timedelta(seconds=1))  # spawned after that registration
    assert (local_health(core).state, local_health(core).detail) == ("starting", "The worker process runs and has not registered yet.")

    clock.advance(2)
    core.register_worker(REGISTRATION, core.enrollment_token)
    assert local_health(core).state == "ready"

    core.restart_local_worker()
    assert stub.restarts == 1

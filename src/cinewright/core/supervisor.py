"""Supervise the local worker: start it in the engine's environment, watch it, restart it and stop it.

The worker runs as `<engine python> -I -u -c <BOOTSTRAP> <copy dir> --engine <engine> [--wan2gp-root <root>]
--stop-on-stdin-eof`:

- `-I` keeps the engine interpreter away from the user's environment, but it implies `-E`, so `PYTHONPATH`
  cannot point it at the worker package. The bootstrap puts the directory holding a copy of
  `cinewright_worker` (standard library only, refreshed from the running core at start) first on
  `sys.path`. The copy, not the core's site-packages, because the engine runs another Python version.
  `-u` replaces `PYTHONUNBUFFERED`, which `-I` ignores, so engine output reaches the log before a crash.
- The environment holds `CINEWRIGHT_CORE_URL`, `CINEWRIGHT_ENROLLMENT_TOKEN` and `CINEWRIGHT_HOME` and no other
  Cinewright variable. HTTP(S) proxies point at a loopback port the supervisor binds but never listens on,
  so library downloads in the engine are refused; the worker's own client uses no proxy.
- The process ends with the core: a Windows Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` (Windows
  does not end children with their parent), on Linux `PR_SET_PDEATHSIG` with `SIGKILL`. The death signal
  fires when the spawning thread ends, so one supervisor thread spawns every worker and lives until `stop()`.
- Stop: close the worker's stdin (`--stop-on-stdin-eof`), wait `grace_seconds`, then kill it with everything
  it started. Console control events do not reach a worker of a console-less (pythonw) core, and
  `Popen.terminate()` is an immediate TerminateProcess, so stdin is the stop request on both platforms.
- After a crash the worker restarts with exponential backoff; `restart()` replaces a running worker at once,
  so it registers again (its model list is sent only at registration).
- stdout and stderr go to `<data dir>/logs/worker.log`: the worker logs to stderr, Wan2GP prints to stdout.
"""

from __future__ import annotations

import ctypes
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import IO, Any, Literal

import cinewright_worker
from cinewright.config import ENV_CORE_URL, ENV_ENROLLMENT_TOKEN, ENV_HOME
from cinewright.core.clock import Clock, utcnow
from cinewright.paths import DataPaths

log = logging.getLogger(__name__)

STOP_GRACE_SECONDS = 5.0
BACKOFF_INITIAL_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 60.0
STABLE_SECONDS = 60.0  # a worker that ran this long before it exited restarts after the initial backoff again
POLL_SECONDS = 0.2
LOG_NAME = "worker.log"
LOG_MAX_BYTES = 5 * 1024 * 1024  # checked at each start; the larger log becomes worker.log.1
NO_PROXY = "127.0.0.1,localhost"
COPY_DIR_NAME = "worker"  # below <data dir>/engines/

# Runs under -I: the first argument is the directory holding the copy of cinewright_worker. It is removed
# from sys.argv because cinewright_worker.__main__.main() parses sys.argv[1:].
BOOTSTRAP = (
    "import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module('cinewright_worker', run_name='__main__', alter_sys=True)"
)

_PR_SET_PDEATHSIG = 1
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100


@dataclass(frozen=True)
class WorkerLaunch:
    """How to start the local worker: the engine environment's interpreter and engine (#1 provides them for Wan2GP)."""

    python: Path
    engine: Literal["wan2gp", "mock"] = "wan2gp"
    wan2gp_root: Path | None = None

    def worker_args(self) -> list[str]:
        args = ["--engine", self.engine]
        if self.wan2gp_root is not None:
            args += ["--wan2gp-root", str(self.wan2gp_root)]
        return [*args, "--stop-on-stdin-eof"]


@dataclass(frozen=True)
class SupervisorStatus:
    """`running` since `since` (the spawn time); `starting` before the first spawn; `down` between a crash and the restart."""

    state: Literal["starting", "running", "down", "stopped"]
    since: datetime | None = None
    detail: str | None = None


# --------------------------------------------------------------------------- spawn building blocks


def spawn_environment(
    base: Mapping[str, str], *, core_url: str, enrollment_token: str, home: Path, proxy_port: int, platform: str = sys.platform
) -> dict[str, str]:
    """The worker's environment: `base` without any Cinewright or proxy variable, plus the three it needs and closed proxies."""
    env = {key: value for key, value in base.items() if not key.upper().startswith("CINEWRIGHT_") and not key.lower().endswith("_proxy")}
    env.update({ENV_CORE_URL: core_url, ENV_ENROLLMENT_TOKEN: enrollment_token, ENV_HOME: str(home)})
    proxy = f"http://127.0.0.1:{proxy_port}"
    proxies = {"HTTP_PROXY": proxy, "HTTPS_PROXY": proxy, "NO_PROXY": NO_PROXY}
    env.update(proxies)
    if platform != "win32":  # case-sensitive names, and urllib prefers the lower-case ones
        env.update({key.lower(): value for key, value in proxies.items()})
    return env


def reserve_closed_port() -> socket.socket:
    """Bind a loopback port without listening: connections to it are refused, and no other program can take it."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if sys.platform == "win32":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind(("127.0.0.1", 0))
    except OSError:
        sock.close()
        raise
    return sock


def copy_worker_package(root: Path) -> Path:
    """Copy the running core's `cinewright_worker` into `root`, replacing an older copy; returns `root`."""
    source = Path(cinewright_worker.__file__).resolve().parent
    target = root / "cinewright_worker"
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)
    return root


def worker_command(launch: WorkerLaunch, copy_root: Path) -> list[str]:
    return [str(launch.python), "-I", "-u", "-c", BOOTSTRAP, str(copy_root), *launch.worker_args()]


class BoundProcess:
    """A child process that ends with this process, together with everything it starts."""

    def __init__(self, popen: subprocess.Popen[bytes], job: int | None = None) -> None:
        self.popen = popen
        self._job = job

    @property
    def pid(self) -> int:
        return self.popen.pid

    def kill(self) -> None:
        """Kill the process and its descendants."""
        if self._job is not None:
            _kernel32().TerminateJobObject(self._job, 1)
        elif sys.platform != "win32":
            _kill_group(self.popen.pid)
        else:
            self.popen.kill()

    def release(self) -> None:
        """After the process ended: kill what it left behind and free the job handle."""
        if self._job is not None:
            job, self._job = self._job, None
            _kernel32().CloseHandle(job)  # KILL_ON_JOB_CLOSE ends any remaining descendant
        elif sys.platform != "win32":
            _kill_group(self.popen.pid)


def spawn_bound(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    stdin: int | IO[Any] | None = None,
    stdout: int | IO[Any] | None = None,
    stderr: int | IO[Any] | None = None,
) -> BoundProcess:
    """Start `argv` so that it ends when this process ends, however this process ends.

    Windows: the child joins a new Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` right after it starts;
    only this process holds the job handle, so the system kills the job when this process ends. No console
    window opens. Linux: `PR_SET_PDEATHSIG` with `SIGKILL`, in a new session so its descendants can be killed
    together. The death signal follows the calling thread, so call this from a thread that lives as long as
    the child should.
    """
    if sys.platform == "win32":
        popen = subprocess.Popen(  # noqa: S603 - the interpreter of the engine environment, fixed arguments
            list(argv), env=env, stdin=stdin, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW
        )
        try:
            job = _job_for(popen.pid)
        except OSError:
            popen.kill()
            popen.wait()
            raise
        return BoundProcess(popen, job)
    preexec = _death_signal() if sys.platform.startswith("linux") else None
    popen = subprocess.Popen(  # noqa: S603 - see above
        list(argv), env=env, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=True, preexec_fn=preexec
    )
    return BoundProcess(popen)


def _kill_group(pgid: int) -> None:
    import signal  # SIGKILL does not exist on Windows

    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _death_signal() -> Callable[[], None]:
    """A preexec function for Linux; prctl is resolved here, before the fork, so the child only calls it."""
    import signal

    prctl = ctypes.CDLL(None, use_errno=True).prctl
    prctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong)
    prctl.restype = ctypes.c_int
    parent = os.getpid()
    sigkill = int(signal.SIGKILL)

    def set_death_signal() -> None:
        # The parent may have died between fork and prctl; then the signal would never come.
        if prctl(_PR_SET_PDEATHSIG, sigkill, 0, 0, 0) != 0 or os.getppid() != parent:
            os._exit(127)

    return set_death_signal


@cache
def _kernel32() -> Any:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ((ctypes.c_void_p, wintypes.LPCWSTR), wintypes.HANDLE),
        "SetInformationJobObject": ((wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD), wintypes.BOOL),
        "OpenProcess": ((wintypes.DWORD, wintypes.BOOL, wintypes.DWORD), wintypes.HANDLE),
        "AssignProcessToJobObject": ((wintypes.HANDLE, wintypes.HANDLE), wintypes.BOOL),
        "TerminateJobObject": ((wintypes.HANDLE, wintypes.UINT), wintypes.BOOL),
        "CloseHandle": ((wintypes.HANDLE,), wintypes.BOOL),
    }
    for name, (argtypes, restype) in signatures.items():
        function = getattr(kernel32, name)
        function.argtypes = argtypes
        function.restype = restype
    return kernel32


class _BasicLimits(ctypes.Structure):
    _fields_ = (
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    )


class _IoCounters(ctypes.Structure):
    _fields_ = tuple(
        (name, ctypes.c_uint64) for name in ("Reads", "Writes", "OtherOperations", "ReadBytes", "WrittenBytes", "OtherTransferredBytes")
    )


class _ExtendedLimits(ctypes.Structure):
    _fields_ = (
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    )


def _job_for(pid: int) -> int:
    """A new kill-on-close Job Object holding process `pid`; raises OSError when any step fails."""
    kernel32 = _kernel32()

    def fail(step: str) -> OSError:
        code = ctypes.get_last_error()
        return OSError(code, f"{step} failed: {ctypes.FormatError(code)}")

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise fail("CreateJobObjectW")
    try:
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise fail("SetInformationJobObject")
        process = kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
        if not process:
            raise fail("OpenProcess")
        try:
            if not kernel32.AssignProcessToJobObject(job, process):
                raise fail("AssignProcessToJobObject")
        finally:
            kernel32.CloseHandle(process)
    except OSError:
        kernel32.CloseHandle(job)
        raise
    return job


def _rotate(path: Path) -> None:
    try:
        if path.stat().st_size > LOG_MAX_BYTES:
            os.replace(path, path.with_name(path.name + ".1"))
    except OSError:
        pass  # no log yet, or the old one is in use: keep appending


# --------------------------------------------------------------------------- the supervisor


class WorkerSupervisor:
    """Keeps one long-running local worker; `start()` and `stop()` belong to the core's lifespan."""

    def __init__(
        self,
        launch: WorkerLaunch,
        *,
        grace_seconds: float = STOP_GRACE_SECONDS,
        backoff_initial: float = BACKOFF_INITIAL_SECONDS,
        backoff_max: float = BACKOFF_MAX_SECONDS,
        stable_seconds: float = STABLE_SECONDS,
    ) -> None:
        self.launch = launch
        self.grace_seconds = grace_seconds
        self.backoff_initial = backoff_initial
        self.backoff_max = backoff_max
        self.stable_seconds = stable_seconds
        self.spawns = 0
        self.last_stop_graceful: bool | None = None  # whether the last requested stop ended without a kill
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = False
        self._restart = False
        self._thread: threading.Thread | None = None
        self._process: BoundProcess | None = None
        self._status = SupervisorStatus("starting", detail="The worker is starting.")
        self._port: socket.socket | None = None
        self._copied = False
        self._core_url = ""
        self._token = ""
        self._paths: DataPaths | None = None
        self._clock: Clock = utcnow

    # ------------------------------------------------------------------ control

    def start(self, *, core_url: str, enrollment_token: str, paths: DataPaths, clock: Clock = utcnow) -> None:
        if self._thread is not None:
            raise RuntimeError("The supervisor was already started.")
        self._core_url, self._token, self._paths, self._clock = core_url, enrollment_token, paths, clock
        self._port = reserve_closed_port()
        self._thread = threading.Thread(target=self._run, name="cinewright-supervisor", daemon=True)
        self._thread.start()

    def restart(self) -> None:
        """Replace the worker now, e.g. after a model install or removal, so it registers its current model list."""
        with self._lock:
            self._restart = True
        self._wake.set()

    def stop(self) -> None:
        """Ask the worker to stop, kill it after the grace period and end the supervisor thread."""
        with self._lock:
            self._stopping = True
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=self.grace_seconds + 30)
            if self._thread.is_alive():
                log.error("The worker supervisor did not stop in time.")
        if self._port is not None:
            self._port.close()
            self._port = None

    @property
    def pid(self) -> int | None:
        with self._lock:
            return self._process.pid if self._process is not None else None

    def status(self) -> SupervisorStatus:
        with self._lock:
            if self._status.state == "running" and (self._process is None or self._process.popen.poll() is not None):
                return SupervisorStatus("down", detail="The worker process ended; the supervisor restarts it.")
            return self._status

    # ------------------------------------------------------------------ supervisor thread

    def _run(self) -> None:
        failures = 0
        while not self._stopping:
            try:
                bound = self._spawn()
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                failures += 1
                delay = self._backoff(failures)
                log.error("The worker could not start (%s); retrying in %.0f s.", exc, delay)
                self._set_status(SupervisorStatus("down", detail=f"The worker could not start: {exc}"))
                self._pause(delay)
                continue
            started = time.monotonic()
            event = self._watch(bound)
            if event == "exited":
                bound.release()
                ran = time.monotonic() - started
                failures = 1 if ran >= self.stable_seconds else failures + 1
                delay = self._backoff(failures)
                code = bound.popen.returncode
                log.warning("Worker process %d exited with code %s after %.0f s; restarting in %.0f s.", bound.pid, code, ran, delay)
                self._set_status(SupervisorStatus("down", detail=f"The worker exited with code {code}; it restarts in {delay:.0f} s."))
                self._pause(delay)
            else:
                self._stop_process(bound)
                if event == "restart":
                    log.info("Restarting the worker on request.")
                    failures = 0
        self._set_status(SupervisorStatus("stopped", detail="The worker was stopped."))

    def _spawn(self) -> BoundProcess:
        assert self._paths is not None and self._port is not None
        copy_root = self._paths.engines_dir / COPY_DIR_NAME
        if not self._copied:
            copy_worker_package(copy_root)
            self._copied = True
        self._paths.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._paths.logs_dir / LOG_NAME
        _rotate(log_path)
        env = spawn_environment(
            os.environ,
            core_url=self._core_url,
            enrollment_token=self._token,
            home=self._paths.root,
            proxy_port=self._port.getsockname()[1],
        )
        with log_path.open("ab") as log_file:  # the child keeps its own handle
            bound = spawn_bound(
                worker_command(self.launch, copy_root), env=env, stdin=subprocess.PIPE, stdout=log_file, stderr=subprocess.STDOUT
            )
        self.spawns += 1
        with self._lock:
            self._process = bound
            self._status = SupervisorStatus("running", since=self._clock())
        log.info("Started worker process %d (engine %s); its output goes to %s.", bound.pid, self.launch.engine, log_path)
        return bound

    def _watch(self, bound: BoundProcess) -> Literal["exited", "restart", "stop"]:
        while True:
            with self._lock:
                if self._stopping:
                    return "stop"
                if self._restart:
                    self._restart = False
                    return "restart"
            if bound.popen.poll() is not None:
                return "exited"
            if self._wake.wait(POLL_SECONDS):
                self._wake.clear()

    def _pause(self, seconds: float) -> None:
        """Wait before a restart; a stop or restart request ends the wait early."""
        deadline = time.monotonic() + seconds
        while True:
            with self._lock:
                if self._stopping:
                    return
                if self._restart:
                    self._restart = False
                    return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            if self._wake.wait(remaining):
                self._wake.clear()

    def _stop_process(self, bound: BoundProcess) -> None:
        """Close the worker's stdin, wait the grace period, then kill it and its descendants."""
        try:
            if bound.popen.stdin is not None:
                bound.popen.stdin.close()
        except OSError:
            pass  # the worker already went away
        try:
            bound.popen.wait(timeout=self.grace_seconds)
            graceful = True
        except subprocess.TimeoutExpired:
            log.warning("Worker process %d did not stop within %.0f s; killing it.", bound.pid, self.grace_seconds)
            bound.kill()
            graceful = False
            try:
                bound.popen.wait(timeout=10)
            except subprocess.TimeoutExpired:
                log.error("Worker process %d did not end after the kill.", bound.pid)
        bound.release()
        self.last_stop_graceful = graceful
        with self._lock:
            self._process = None
            self._status = SupervisorStatus("starting", detail="The worker is restarting.")
        log.info("Worker process %d stopped (%s).", bound.pid, "on request" if graceful else "killed after the grace period")

    def _set_status(self, status: SupervisorStatus) -> None:
        with self._lock:
            if status.state != "running":
                self._process = None
            self._status = status

    def _backoff(self, failures: int) -> float:
        return min(self.backoff_max, self.backoff_initial * 2 ** max(0, failures - 1))

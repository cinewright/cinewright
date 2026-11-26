"""Start Cinewright: run the core on 127.0.0.1:<port> and open the web UI in the browser.

End users start it from the Start menu (a gui-script without console), so everything goes to
`<data dir>/logs/cinewright.log`. The core provisions the WanGP engine in the background on the first
start and starts the local worker on it once it is ready. A second start of the same version only opens the browser.
When a core of another version answers, for example one that kept running through an update, the launcher stops it
if its process runs from this installation's folder, then starts as usual. A core from anywhere else, such as a
development checkout, keeps running, and the launcher only opens the browser. `--version` exists for CI.
"""

from __future__ import annotations

import argparse
import contextlib
import enum
import logging
import os
import sys
import threading
import time
import webbrowser
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Protocol

import httpx
import psutil
import uvicorn

from cinewright import __version__
from cinewright.api.schemas import API_PREFIX
from cinewright.config import Config, load_config

log = logging.getLogger("cinewright.launcher")

READY_TIMEOUT_S = 60.0
LOG_MAX_BYTES = 5 * 1024 * 1024
STOP_TIMEOUT_S = 5.0  # per step: terminate, then kill what is left
PORT_FREE_TIMEOUT_S = 10.0
POLL_S = 0.25
# Setup creates the core runtime as <folder>\runtime next to these files (installer/cinewright.nsi); uv keeps the
# interpreter the runtime runs on in <folder>\python, so the core's process runs from there.
RUNTIME_DIR_NAME = "runtime"
INSTALL_MARKERS = ("uninstall.exe", "cinewright.ico")


def setup_logging(logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(logs_dir / "cinewright.log", maxBytes=LOG_MAX_BYTES, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per health poll and MCP call is noise


def _get(config: Config, path: str, timeout: float) -> httpx.Response | None:
    try:
        return httpx.get(f"{config.base_url}{API_PREFIX}{path}", timeout=timeout, trust_env=False)
    except httpx.HTTPError:
        return None


def running_version(config: Config) -> str | None:
    """The version of the Cinewright core that answers on the configured port: None when none answers, "" when it names none."""
    response = _get(config, "/system", timeout=2.0)
    if response is None or response.status_code != 200:
        return None
    try:
        info = response.json()
    except ValueError:
        return None
    if not isinstance(info, dict) or info.get("name") != "Cinewright":
        return None
    version = info.get("version")
    return version if isinstance(version, str) else ""


def open_when_ready(config: Config, timeout: float = READY_TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = _get(config, "/health", timeout=1.0)
        if response is not None and response.status_code == 200:
            webbrowser.open(f"{config.base_url}/")
            return True
        time.sleep(0.25)
    log.error("The core did not answer within %.0f s, so the browser was not opened.", timeout)
    return False


# Replacing a core of another version


@dataclass(frozen=True)
class ProcessInfo:
    """A process as the launcher sees it. `pid` or `exe` is None when the process cannot be inspected."""

    pid: int | None
    exe: str | None
    parent: ProcessInfo | None = None


class Processes(Protocol):
    """The process table, as far as the launcher needs it; tests pass a fake."""

    def listening(self, port: int) -> list[ProcessInfo]:
        """The processes with a TCP socket listening on `port`, each with its parent."""
        ...

    def stop(self, processes: Sequence[ProcessInfo], timeout: float) -> list[ProcessInfo]:
        """Terminate the processes, kill those left after `timeout`; returns those still running after another `timeout`."""
        ...


class SystemProcesses:
    """`Processes` on this computer's process table, through psutil."""

    def listening(self, port: int) -> list[ProcessInfo]:
        # A socket on the port without a peer holds it. Windows reports some listening sockets without a state
        # (NONE, seen for sandboxed processes), so the missing peer decides, not only LISTEN.
        pids = {
            c.pid
            for c in psutil.net_connections(kind="tcp")
            if c.laddr and c.laddr.port == port and not c.raddr and c.status in (psutil.CONN_LISTEN, psutil.CONN_NONE)
        }
        return [self._info(pid) for pid in sorted(pids, key=lambda pid: -1 if pid is None else pid)]

    @staticmethod
    def _info(pid: int | None) -> ProcessInfo:
        if pid is None:  # e.g. another user's process on Linux
            return ProcessInfo(None, None)
        try:
            process = psutil.Process(pid)
            exe = process.exe() or None
        except psutil.Error:
            return ProcessInfo(pid, None)
        parent = None
        with contextlib.suppress(psutil.Error):
            if (found := process.parent()) is not None:
                parent = ProcessInfo(found.pid, found.exe() or None)
        return ProcessInfo(pid, exe, parent)

    def stop(self, processes: Sequence[ProcessInfo], timeout: float) -> list[ProcessInfo]:
        stopping: list[tuple[ProcessInfo, psutil.Process]] = []
        left: list[ProcessInfo] = []
        for info in processes:
            if info.pid is None:
                left.append(info)
                continue
            try:
                process = psutil.Process(info.pid)
                if process.exe() != info.exe:
                    continue  # it ended, and another program got its id
                process.terminate()
            except psutil.NoSuchProcess:
                continue
            except psutil.Error as exc:
                log.warning("Could not stop process %d: %s", info.pid, exc)
                left.append(info)
                continue
            stopping.append((info, process))
        _, alive = psutil.wait_procs([process for _, process in stopping], timeout=timeout)
        for process in alive:
            with contextlib.suppress(psutil.Error):
                process.kill()
        _, alive = psutil.wait_procs(alive, timeout=timeout)
        return left + [info for info, process in stopping if process in alive]


def installation_dir(prefix: str | None = None) -> Path | None:
    """The folder setup installed this Cinewright into, or None when it does not run from an installation.

    Setup creates the runtime as `<folder>\\runtime` next to `uninstall.exe` and `cinewright.ico`. A development
    checkout has neither, so a launcher started from one never stops a core.
    """
    runtime = Path(sys.prefix if prefix is None else prefix)
    folder = runtime.parent
    if runtime.name.lower() != RUNTIME_DIR_NAME or not all((folder / name).is_file() for name in INSTALL_MARKERS):
        return None
    return folder


def lies_in(exe: str | None, folder: Path) -> bool:
    """Whether the program `exe` lies inside `folder`, compared case-insensitively on Windows."""
    if not exe:
        return False
    root = os.path.normcase(os.path.realpath(folder)).rstrip(os.sep) + os.sep
    return os.path.normcase(os.path.realpath(exe)).startswith(root)


def _describe(processes: Iterable[ProcessInfo]) -> str:
    return ", ".join(f"{p.exe or 'an unknown program'} (process {p.pid if p.pid is not None else 'unknown'})" for p in processes) or "none"


class Replacement(enum.Enum):
    STOPPED = "stopped"  # the other core is gone and its port is free: start
    LEFT_RUNNING = "left running"  # not from this installation: open the browser on it
    FAILED = "failed"  # it is from this installation but did not stop or did not free the port


def stop_other_version(
    config: Config,
    version: str,
    processes: Processes,
    folder: Path | None,
    *,
    stop_timeout: float = STOP_TIMEOUT_S,
    free_timeout: float = PORT_FREE_TIMEOUT_S,
) -> Replacement:
    """Stop the core of another `version` that answers on the port, if its process runs from the installation `folder`.

    Its parent goes too when it runs from there as well: the runtime's launcher shim, which started the interpreter.
    Never this process or its parent. Once they ended, waits until nothing listens on the port any more.
    """
    mismatch = f"Cinewright {version or '(no version)'} answers on {config.bind}, but this is Cinewright {__version__}"
    if folder is None:
        log.warning("%s. This launcher does not run from an installation, so that core keeps running; opening the browser.", mismatch)
        return Replacement.LEFT_RUNNING
    own = {os.getpid(), os.getppid()}
    listeners = processes.listening(config.port)
    if not listeners or any(p.pid in own or not lies_in(p.exe, folder) for p in listeners):
        log.warning(
            "%s. Its process, %s, is not from this installation in %s, so it keeps running; opening the browser.",
            mismatch,
            _describe(listeners),
            folder,
        )
        return Replacement.LEFT_RUNNING
    targets = list(listeners)
    for listener in listeners:
        parent = listener.parent
        if parent is not None and parent.pid not in own and lies_in(parent.exe, folder) and parent not in targets:
            targets.append(parent)
    log.info("%s. It runs from this installation, so the launcher stops it: %s.", mismatch, _describe(targets))
    left = processes.stop(targets, stop_timeout)
    if left:
        log.error("Could not stop %s, so Cinewright %s cannot start.", _describe(left), __version__)
        return Replacement.FAILED
    deadline = time.monotonic() + free_timeout
    while busy := processes.listening(config.port):
        if time.monotonic() >= deadline:
            log.error(
                "Port %d is still in use by %s, %.0f s after the old core stopped, so Cinewright %s cannot start.",
                config.port,
                _describe(busy),
                free_timeout,
                __version__,
            )
            return Replacement.FAILED
        time.sleep(POLL_S)
    log.info("Stopped Cinewright %s; port %d is free.", version or "(no version)", config.port)
    return Replacement.STOPPED


def serve(config: Config) -> bool:
    """Run the core until it stops; False when it never finished starting (e.g. an invalid registry)."""
    from cinewright.app import create_app

    server = uvicorn.Server(
        uvicorn.Config(
            create_app(config=config, provision=True),
            host=config.host,
            port=config.port,
            log_config=None,  # no console: our file handler on the root logger takes uvicorn's records
            use_colors=False,
            timeout_graceful_shutdown=5,  # open event streams must not block the exit
        )
    )
    threading.Thread(target=open_when_ready, args=(config,), name="open-browser", daemon=True).start()
    server.run()
    return server.started


def main(argv: list[str] | None = None, processes: Processes | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cinewright", description="Start Cinewright and open its web UI in the browser.")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    args = parser.parse_args(argv)
    if args.version:
        print(f"cinewright {__version__}")
        return 0

    config = load_config()
    setup_logging(config.paths.logs_dir)
    if not config.loopback:
        log.error("CINEWRIGHT_HOST=%s is refused: binding beyond 127.0.0.1 needs an access token, which arrives in phase 1.", config.host)
        return 2
    version = running_version(config)
    if version == __version__:
        log.info("Cinewright %s already runs on %s; opening the browser.", version, config.bind)
        webbrowser.open(f"{config.base_url}/")
        return 0
    if version is not None:
        replacement = stop_other_version(config, version, processes or SystemProcesses(), installation_dir())
        if replacement is Replacement.FAILED:
            return 1
        if replacement is Replacement.LEFT_RUNNING:
            webbrowser.open(f"{config.base_url}/")
            return 0
    log.info("Starting Cinewright %s on %s with data in %s", __version__, config.bind, config.home)
    try:
        started = serve(config)
    except Exception:
        log.exception("Cinewright stopped with an error")
        return 1
    if not started:
        log.error("Cinewright did not start; the log above names the cause.")
        return 1
    return 0

"""Launcher: version smoke test, refusal of non-loopback binds, a second start of the same version only opens the
browser, and a core of another version is replaced only when it runs from this installation.

No test talks to a real core or stops a process it did not start: an autouse fixture makes the HTTP probe and the
system process table fail unless a test replaces them, and the decision tests use a fake process table.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import httpx
import psutil
import pytest

from cinewright import __version__, launcher
from cinewright.config import load_config
from cinewright.launcher import ProcessInfo, Replacement

RealSystemProcesses = launcher.SystemProcesses
OLD = "0.1.0rc9"


@pytest.fixture(autouse=True)
def detach_log_files():
    yield
    root = logging.getLogger()
    for handler in [h for h in root.handlers if isinstance(h, RotatingFileHandler)]:
        root.removeHandler(handler)
        handler.close()


@pytest.fixture(autouse=True)
def no_real_core_or_processes(monkeypatch):
    """Nothing here may reach a core on the real port or the computer's process table by accident."""

    def no_http(*args, **kwargs):
        raise AssertionError("a launcher test must not send HTTP requests")

    def no_processes():
        raise AssertionError("a launcher test must pass a fake process table")

    monkeypatch.setattr(launcher, "_get", no_http)
    monkeypatch.setattr(launcher, "SystemProcesses", no_processes)
    monkeypatch.setattr(launcher, "serve", lambda config: (_ for _ in ()).throw(AssertionError("must not start")))


class FakeProcesses:
    """A process table where `listeners` listen on the port until `stop` ends them (unless `stubborn`)."""

    def __init__(self, listeners: list[ProcessInfo], *, stubborn: bool = False, port_stays_busy: bool = False) -> None:
        self.listeners = listeners
        self.stubborn = stubborn
        self.port_stays_busy = port_stays_busy
        self.stopped: list[list[ProcessInfo]] = []

    def listening(self, port: int) -> list[ProcessInfo]:
        assert port == 8737
        return list(self.listeners)

    def stop(self, processes, timeout: float) -> list[ProcessInfo]:
        self.stopped.append(list(processes))
        if self.stubborn:
            return list(processes)
        if not self.port_stays_busy:
            self.listeners = []
        return []


@pytest.fixture
def installation(tmp_path) -> Path:
    """An installation folder laid out as setup leaves it."""
    folder = tmp_path / "Programs" / "Cinewright"
    (folder / "runtime" / "Scripts").mkdir(parents=True)
    (folder / "uninstall.exe").write_bytes(b"")
    (folder / "cinewright.ico").write_bytes(b"")
    return folder


def old_core(folder: Path) -> ProcessInfo:
    """The old core as it runs from an installation: uv's interpreter under python\\, started by the runtime's shim."""
    shim = ProcessInfo(4001, str(folder / "runtime" / "Scripts" / "pythonw.exe"))
    return ProcessInfo(4002, str(folder / "python" / "cpython-3.12.11-windows-x86_64-none" / "pythonw.exe"), shim)


def answers(monkeypatch, version: str | None) -> None:
    monkeypatch.setattr(launcher, "running_version", lambda config: version)


def answers_in_turn(monkeypatch, *versions: str | None) -> list[str | None]:
    """The core answers each probe with the next of `versions`; returns those not asked for yet."""
    left = list(versions)

    def probe(config):
        assert left, "probed more often than expected"
        return left.pop(0)

    monkeypatch.setattr(launcher, "running_version", probe)
    return left


def log_text(home: Path) -> str:
    return (home / "logs" / "cinewright.log").read_text(encoding="utf-8")


def test_version_flag_prints_name_and_version(capsys):
    assert launcher.main(["--version"]) == 0
    assert capsys.readouterr().out == f"cinewright {__version__}\n"


def test_python_dash_m_cinewright_version():
    result = subprocess.run([sys.executable, "-m", "cinewright", "--version"], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0
    assert result.stdout.strip() == f"cinewright {__version__}"


def test_non_loopback_bind_is_refused(monkeypatch, home):
    monkeypatch.setenv("CINEWRIGHT_HOST", "0.0.0.0")
    assert launcher.main([]) == 2
    assert "is refused" in log_text(home)


def test_second_start_of_the_same_version_only_opens_the_browser(monkeypatch, installation):
    opened = []
    answers(monkeypatch, __version__)
    monkeypatch.setattr(launcher, "installation_dir", lambda: installation)
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    processes = FakeProcesses([old_core(installation)])
    assert launcher.main([], processes=processes) == 0
    assert opened == ["http://127.0.0.1:8737/"]
    assert processes.stopped == []


def test_first_start_serves(monkeypatch):
    started = []
    answers(monkeypatch, None)
    monkeypatch.setattr(launcher, "serve", lambda config: started.append(config) or True)
    assert launcher.main([], processes=FakeProcesses([])) == 0
    assert started[0].port == 8737


def test_failed_startup_exits_non_zero(monkeypatch):
    answers(monkeypatch, None)
    monkeypatch.setattr(launcher, "serve", lambda config: False)
    assert launcher.main([], processes=FakeProcesses([])) == 1


def test_another_version_from_this_installation_is_stopped_then_started(monkeypatch, installation, home):
    started, opened = [], []
    answers(monkeypatch, OLD)
    monkeypatch.setattr(launcher, "installation_dir", lambda: installation)
    monkeypatch.setattr(launcher, "serve", lambda config: started.append(config) or True)
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    core = old_core(installation)
    processes = FakeProcesses([core])
    assert launcher.main([], processes=processes) == 0
    assert processes.stopped == [[core, core.parent]]  # the interpreter and the runtime's shim that started it
    assert len(started) == 1
    assert opened == []  # serve opens the browser once the new core is ready
    text = log_text(home)
    assert f"Cinewright {OLD} answers on 127.0.0.1:8737, but this is Cinewright {__version__}" in text
    assert "the launcher stops it" in text
    assert f"Stopped Cinewright {OLD}; port 8737 is free." in text
    assert text.index("port 8737 is free") < text.index(f"Starting Cinewright {__version__}")


@pytest.mark.parametrize(
    "exe",
    [
        pytest.param("{tmp}/dev/cinewright/.venv/Scripts/python.exe", id="development checkout"),
        pytest.param("{tmp}/Programs/Cinewright2/python/pythonw.exe", id="sibling folder with the same prefix"),
        pytest.param("", id="unknown program"),
    ],
)
def test_another_version_from_elsewhere_keeps_running_and_the_browser_opens(monkeypatch, installation, home, tmp_path, exe):
    opened = []
    answers(monkeypatch, OLD)
    monkeypatch.setattr(launcher, "installation_dir", lambda: installation)
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    shim = ProcessInfo(4001, str(installation / "runtime" / "Scripts" / "pythonw.exe"))
    processes = FakeProcesses([ProcessInfo(4002, exe.format(tmp=tmp_path) or None, shim)])
    assert launcher.main([], processes=processes) == 0
    assert processes.stopped == []
    assert opened == ["http://127.0.0.1:8737/"]
    text = log_text(home)
    assert f"Cinewright {OLD} answers on 127.0.0.1:8737, but this is Cinewright {__version__}" in text
    assert "is not from this installation" in text and "keeps running" in text


def test_a_launcher_outside_an_installation_never_stops_a_core(monkeypatch, installation, home):
    opened = []
    answers(monkeypatch, OLD)
    monkeypatch.setattr(launcher, "installation_dir", lambda: None)
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    processes = FakeProcesses([old_core(installation)])
    assert launcher.main([], processes=processes) == 0
    assert processes.stopped == []
    assert opened == ["http://127.0.0.1:8737/"]
    assert "does not run from an installation" in log_text(home)


def test_a_core_that_does_not_stop_is_reported_and_nothing_starts(monkeypatch, installation, home):
    opened = []
    answers(monkeypatch, OLD)
    monkeypatch.setattr(launcher, "installation_dir", lambda: installation)
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    processes = FakeProcesses([old_core(installation)], stubborn=True)
    assert launcher.main([], processes=processes) == 1
    assert opened == []
    assert "Could not stop" in log_text(home)


def test_a_port_that_stays_busy_is_reported_and_nothing_starts(monkeypatch, installation, caplog):
    answers(monkeypatch, OLD)
    processes = FakeProcesses([old_core(installation)], port_stays_busy=True)
    with caplog.at_level(logging.INFO, logger="cinewright.launcher"):
        assert launcher.stop_other_version(load_config(), OLD, processes, installation, free_timeout=0) is Replacement.FAILED
    assert len(processes.stopped) == 1
    assert "Port 8737 is still in use" in caplog.text


def test_this_process_and_its_parent_are_never_stopped(monkeypatch, installation):
    answers(monkeypatch, OLD)
    shim = ProcessInfo(os.getppid(), str(installation / "runtime" / "Scripts" / "pythonw.exe"))
    core = ProcessInfo(4002, str(installation / "python" / "pythonw.exe"), shim)
    processes = FakeProcesses([core])
    assert launcher.stop_other_version(load_config(), OLD, processes, installation) is Replacement.STOPPED
    assert processes.stopped == [[core]]

    me = ProcessInfo(os.getpid(), str(installation / "python" / "pythonw.exe"))
    processes = FakeProcesses([me])
    assert launcher.stop_other_version(load_config(), OLD, processes, installation) is Replacement.LEFT_RUNNING
    assert processes.stopped == []


def test_a_port_shared_with_a_foreign_listener_is_left_alone(installation, tmp_path):
    foreign = ProcessInfo(5000, str(tmp_path / "elsewhere" / "server.exe"))
    processes = FakeProcesses([old_core(installation), foreign])
    assert launcher.stop_other_version(load_config(), OLD, processes, installation) is Replacement.LEFT_RUNNING
    assert processes.stopped == []


def test_a_core_that_ended_before_the_lookup_is_followed_by_a_start(monkeypatch, installation, home):
    started, opened = [], []
    left = answers_in_turn(monkeypatch, OLD, None)
    monkeypatch.setattr(launcher, "installation_dir", lambda: installation)
    monkeypatch.setattr(launcher, "serve", lambda config: started.append(config) or True)
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    processes = FakeProcesses([])
    assert launcher.main([], processes=processes) == 0
    assert left == []
    assert processes.stopped == []
    assert len(started) == 1 and opened == []
    assert "It no longer answers, so the launcher starts." in log_text(home)


def test_a_core_whose_process_cannot_be_found_keeps_running(monkeypatch, installation, home):
    opened = []
    left = answers_in_turn(monkeypatch, OLD, OLD)
    monkeypatch.setattr(launcher, "installation_dir", lambda: installation)
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    processes = FakeProcesses([])
    assert launcher.main([], processes=processes) == 0
    assert left == []
    assert processes.stopped == []
    assert opened == ["http://127.0.0.1:8737/"]
    assert "No process listening on port 8737 was found, so it keeps running" in log_text(home)


def test_a_core_of_this_version_started_meanwhile_is_not_stopped(monkeypatch, installation, home):
    """Two launchers of the new version start together: the other one has already replaced the old core."""
    opened = []
    left = answers_in_turn(monkeypatch, OLD, __version__)
    monkeypatch.setattr(launcher, "installation_dir", lambda: installation)
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    processes = FakeProcesses([old_core(installation)])
    assert launcher.main([], processes=processes) == 0
    assert left == []
    assert processes.stopped == []
    assert opened == ["http://127.0.0.1:8737/"]
    assert f"Cinewright {__version__} answers now, started meanwhile, so nothing is stopped" in log_text(home)


def test_a_core_that_misses_the_second_probe_is_still_stopped(monkeypatch, installation, home):
    """A busy old core may not answer within the probe's timeout; its process still holds the port."""
    started = []
    left = answers_in_turn(monkeypatch, OLD, None)
    monkeypatch.setattr(launcher, "installation_dir", lambda: installation)
    monkeypatch.setattr(launcher, "serve", lambda config: started.append(config) or True)
    core = old_core(installation)
    processes = FakeProcesses([core])
    assert launcher.main([], processes=processes) == 0
    assert left == []
    assert processes.stopped == [[core, core.parent]]
    assert len(started) == 1
    assert f"Stopped Cinewright {OLD}; port 8737 is free." in log_text(home)


def test_an_error_while_replacing_is_logged_and_nothing_starts(monkeypatch, installation, home):
    class BrokenProcesses(FakeProcesses):
        def listening(self, port: int) -> list[ProcessInfo]:
            raise OSError("the process table cannot be read")

    opened = []
    answers(monkeypatch, OLD)
    monkeypatch.setattr(launcher, "installation_dir", lambda: installation)
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    processes = BrokenProcesses([])
    assert launcher.main([], processes=processes) == 1
    assert processes.stopped == [] and opened == []
    text = log_text(home)
    assert f"Could not replace Cinewright {OLD}, so Cinewright {__version__} does not start" in text
    assert "OSError: the process table cannot be read" in text


# Where the installation is


def test_installation_dir_is_the_folder_around_the_runtime(installation):
    assert launcher.installation_dir(str(installation / "runtime")) == installation


def test_a_development_environment_is_no_installation(tmp_path):
    venv = tmp_path / "cinewright" / ".venv"
    venv.mkdir(parents=True)
    assert launcher.installation_dir(str(venv)) is None


def test_a_runtime_folder_without_setup_files_is_no_installation(installation):
    (installation / "uninstall.exe").unlink()
    assert launcher.installation_dir(str(installation / "runtime")) is None


def test_lies_in_needs_a_whole_folder_name(installation):
    assert launcher.lies_in(str(installation / "python" / "pythonw.exe"), installation)
    assert not launcher.lies_in(str(installation.parent / "Cinewright2" / "pythonw.exe"), installation)
    assert not launcher.lies_in(str(installation), installation)
    assert not launcher.lies_in(None, installation)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows paths are case-insensitive")
def test_lies_in_ignores_case_on_windows(installation):
    assert launcher.lies_in(str(installation / "Python" / "PYTHONW.EXE").upper(), installation)


# The version a core reports


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(200, json={"name": "Cinewright", "version": OLD}), OLD),
        (httpx.Response(200, json={"name": "Cinewright"}), ""),
        (httpx.Response(200, json={"name": "Something else", "version": OLD}), None),
        (httpx.Response(200, json=["Cinewright"]), None),
        (httpx.Response(200, text="not json"), None),
        (httpx.Response(503, json={"name": "Cinewright", "version": OLD}), None),
        (None, None),
    ],
)
def test_running_version_reads_the_cores_system_info(monkeypatch, response, expected):
    monkeypatch.setattr(launcher, "_get", lambda config, path, timeout: response)
    assert launcher.running_version(load_config()) == expected


# The real process table, only with sockets and processes this test made


def test_system_processes_find_the_listener_of_a_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        found = RealSystemProcesses().listening(port)
    assert [info.pid for info in found] == [os.getpid()]
    assert found[0].exe == psutil.Process().exe()
    assert found[0].parent is not None and found[0].parent.pid == os.getppid()


def test_system_processes_stop_a_process_this_test_started():
    script = "import os, time; print(os.getpid(), flush=True); time.sleep(60)"
    child = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        process = psutil.Process(int(child.stdout.readline()))  # the interpreter, also behind a venv launcher
        processes = RealSystemProcesses()
        assert processes.stop([ProcessInfo(process.pid, "another program.exe")], timeout=5) == []
        assert process.is_running()  # its id now names another program, so it is not this process
        assert processes.stop([ProcessInfo(process.pid, process.exe())], timeout=10) == []
        assert not process.is_running()
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        child.stdout.close()

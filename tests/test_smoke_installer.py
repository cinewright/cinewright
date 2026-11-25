"""`python -m cinewright.devtools smoke --installer ...`: the installer path, with the installer, the registry, the
shortcuts and the core faked. CI runs on Linux, and no test touches the real HKEY_CURRENT_USER, Start menu or desktop.

The fake installer does what `installer/cinewright.nsi` does with `/S`: it creates the runtime, sets the values of the
uninstall entry (leaving other values alone, as WriteRegStr does) and writes the Start-menu shortcut. The temporary
core's API is a `httpx.MockTransport`, except in the first test, where the fake launcher runs the real app in this
process with the environment the tool hands over.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import uvicorn

from cinewright import devtools, smoke, smoke_installer
from cinewright.app import create_app
from cinewright.config import load_config
from cinewright.smoke_installer import PROBE_QUANT, SNAPSHOT_FILE, UNINSTALL_KEY, FileState, Host, Snapshot, installer_command

REG_SZ, REG_EXPAND_SZ, REG_BINARY, REG_DWORD, REG_MULTI_SZ = 1, 2, 3, 4, 7  # winreg value types
OLD_LOCATION = r"C:\Users\owner\AppData\Local\Programs\Cinewright"
OLD_ENTRY = {
    "DisplayName": ("Cinewright", REG_SZ),
    "DisplayVersion": ("0.1.0", REG_SZ),
    "InstallLocation": (OLD_LOCATION, REG_SZ),
    "UninstallString": (f'"{OLD_LOCATION}\\uninstall.exe"', REG_SZ),
    "NoModify": (1, REG_DWORD),
    "EstimatedSize": (0x0001F400, REG_DWORD),
}
OLD_SHORTCUT = b"L\x00\x00\x00 shortcut to the existing installation"
OLD_SHORTCUT_TIMES = (1_700_000_000_000_000_000, 1_700_000_000_000_000_000)
PASSED = "Smoke test passed: all 3 installed smoke models produced their output. Exit code 0."


class FakeRegistry:
    def __init__(self) -> None:
        self.keys: dict[str, dict] = {}

    def read(self, key: str) -> dict | None:
        return dict(self.keys[key]) if key in self.keys else None

    def write(self, key: str, values: Mapping) -> None:
        self.keys[key] = dict(values)

    def delete(self, key: str) -> None:
        self.keys.pop(key, None)


class FakeInstaller:
    def __init__(self, registry: FakeRegistry, start_menu: Path) -> None:
        self.registry = registry
        self.start_menu = start_menu
        self.exit_code = 0
        self.commands: list[str] = []
        self.snapshots: list[Snapshot | None] = []  # the saved snapshot of the existing installation, at each run

    def __call__(self, command: str) -> int:
        self.commands.append(command)
        folder = Path(command.rsplit(" /D=", 1)[1])
        saved = folder.parent / SNAPSHOT_FILE
        self.snapshots.append(smoke_installer.snapshot_from_json(saved.read_text(encoding="utf-8")) if saved.exists() else None)
        python = folder / "runtime" / "Scripts" / "python.exe"
        python.parent.mkdir(parents=True)
        python.write_bytes(b"")
        self.registry.keys.setdefault(UNINSTALL_KEY, {}).update(
            {
                "DisplayName": ("Cinewright", REG_SZ),
                "DisplayVersion": ("9.9.9", REG_SZ),
                "InstallLocation": (str(folder), REG_SZ),
                "UninstallString": (f'"{folder}\\uninstall.exe"', REG_SZ),
                "QuietUninstallString": (f'"{folder}\\uninstall.exe" /S', REG_SZ),
                "NoModify": (1, REG_DWORD),
                "EstimatedSize": (0x00030D40, REG_DWORD),
            }
        )
        self.start_menu.parent.mkdir(parents=True, exist_ok=True)
        self.start_menu.write_bytes(b"L\x00\x00\x00 shortcut to " + str(folder).encode())
        return self.exit_code


class FakeProcess:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def poll(self) -> int | None:
        return None

    def stop(self) -> None:
        self.events.append("core stopped")


def problem(status: int, title: str, detail: str) -> httpx.Response:
    body = {"type": "about:blank", "title": title, "status": status, "detail": detail}
    return httpx.Response(status, json=body, headers={"Content-Type": "application/problem+json"})


class FakeApi:
    """The temporary core as the tool uses it. An accepted install is `downloading` in the next model list and
    `installed` in the one after; a second install while one runs is refused, as the model store does."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.can_install = True
        # One state per GET /system, the last one stays; the first answers the tool's check that the core is up.
        self.engine = [
            {"state": "provisioning", "progress": {"phase": "Downloading WanGP", "step": 1, "total_steps": 6, "percent": 0.0}},
            {"state": "provisioning", "progress": {"phase": "Installing packages", "step": 5, "total_steps": 6, "percent": 40.0}},
            {"state": "ready", "version": "38758d3c73366422efcf609abb63cab8385f3552"},
        ]
        self.system_reads = 0
        self.states = {(job.model_id, job.install_quant): "absent" for job in smoke.SMOKE_JOBS}
        self.install_error: str | None = None  # set: an accepted install fails
        self.installs: list[tuple[str, str | None]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path.removeprefix("/api/v1")
        if method == "GET" and path == "/system":
            self.system_reads += 1
            engine = self.engine.pop(0) if len(self.engine) > 1 else self.engine[0]
            engines = [{"id": "wan2gp", "name": "WanGP", "version": None, **engine}]
            return httpx.Response(200, json={"name": "Cinewright", "version": "9.9.9", "data_dir": str(self.data_dir), "engines": engines})
        if method == "GET" and path == "/models":
            items = []
            for (model_id, quant), state in self.states.items():
                entry = {"quant": quant, "install_state": state}
                if state == "downloading" and self.install_error:
                    entry |= {"install_state": "failed", "error": self.install_error}
                items.append({"id": model_id, "quantizations": [{"quant": "bf16", "install_state": "absent"}, entry]})
            for key, state in self.states.items():
                if state == "downloading":
                    self.states[key] = "installed"
            return httpx.Response(200, json={"items": items})
        if method == "POST" and path.startswith("/models/") and path.endswith("/install"):
            key = (path.split("/")[2], request.url.params.get("quant"))
            self.installs.append(key)
            if not self.can_install:
                return problem(501, "Not implemented yet", "Model installation arrives in phase 1.")
            if key not in self.states:
                return problem(422, "Invalid request", f"There is no quantization {key[1]}.")
            if "downloading" in self.states.values():
                return problem(409, "Conflict", "Another model is being installed.")
            self.states[key] = "downloading"
            return httpx.Response(202)
        return problem(404, "Not found", path)


@dataclass
class Fakes:
    root: Path
    installer: Path
    start_menu: Path
    desktop: Path
    registry: FakeRegistry
    run_installer: FakeInstaller
    api: FakeApi
    host: Host
    cores: list[tuple[Path, dict[str, str], Path]]
    events: list[str]

    def existing_install(self) -> None:
        self.registry.keys[UNINSTALL_KEY] = dict(OLD_ENTRY)
        self.start_menu.parent.mkdir(parents=True, exist_ok=True)
        self.start_menu.write_bytes(OLD_SHORTCUT)
        os.utime(self.start_menu, ns=OLD_SHORTCUT_TIMES)

    def assert_existing_install_as_before(self) -> None:
        assert self.registry.keys == {UNINSTALL_KEY: OLD_ENTRY}
        assert self.start_menu.read_bytes() == OLD_SHORTCUT
        assert self.start_menu.stat().st_mtime_ns == OLD_SHORTCUT_TIMES[1]
        assert not self.desktop.exists()

    def run(self, capsys: pytest.CaptureFixture[str], model_store: Path | None = None) -> tuple[int, list[str]]:
        code = smoke_installer.run(sys.stdout, self.installer, model_store, host=self.host, root=self.root)
        return code, capsys.readouterr().out.strip().splitlines()


@pytest.fixture
def fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fakes:
    monkeypatch.setattr(smoke_installer, "POLL_S", 0.0)
    monkeypatch.setattr(smoke_installer, "CORE_POLL_S", 0.0)
    root = tmp_path / "smoke run"  # a space, as in many Windows account names
    installer = tmp_path / "Downloads" / "CinewrightSetup-9.9.9.exe"
    installer.parent.mkdir()
    installer.write_bytes(b"MZ")
    start_menu = tmp_path / "shell" / "Start Menu" / "Programs" / "Cinewright.lnk"
    desktop = tmp_path / "shell" / "Desktop" / "Cinewright.lnk"
    registry = FakeRegistry()
    api = FakeApi(root / "data")
    cores: list[tuple[Path, dict[str, str], Path]] = []
    events: list[str] = []

    def start_core(python: Path, env: Mapping[str, str], cwd: Path) -> FakeProcess:
        cores.append((python, dict(env), cwd))
        events.append("core started")
        return FakeProcess(events)

    def unlink_dir(link: Path) -> None:
        events.append("link removed")
        smoke_installer.unlink_directory(link)

    run_installer = FakeInstaller(registry, start_menu)
    host = Host(
        registry=registry,
        shortcuts=(start_menu, desktop),
        run_installer=run_installer,
        start_core=start_core,
        unlink_dir=unlink_dir,
        connect=lambda url: httpx.Client(base_url=url, transport=httpx.MockTransport(api.handler)),
    )
    return Fakes(root, installer, start_menu, desktop, registry, run_installer, api, host, cores, events)


@pytest.fixture
def smoke_runs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The smoke run itself (tested in test_smoke) replaced: records the URL it would test."""
    urls: list[str] = []

    def execute(self: smoke.SmokeRun) -> tuple[int, str]:
        urls.append(str(self.http.base_url).rstrip("/"))
        # Each install restarts the local worker: the first job may wait for a cold engine start.
        assert self.route_wait_s == smoke_installer.INSTALLED_ROUTE_WAIT_S > smoke.ROUTE_WAIT_S
        return 0, PASSED

    monkeypatch.setattr(smoke.SmokeRun, "execute", execute)
    return urls


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


class InProcessCore:
    """The installed core as its launcher runs it, here in this process: the real app with the tool's environment."""

    def __init__(self, env: Mapping[str, str]) -> None:
        self.config = load_config(env)
        app = create_app(config=self.config)
        self.server = uvicorn.Server(
            uvicorn.Config(app, host=self.config.host, port=self.config.port, log_config=None, lifespan="on", timeout_graceful_shutdown=2)
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True, name="uvicorn")
        self.thread.start()
        deadline = time.monotonic() + 30
        while not self.server.started:
            assert self.thread.is_alive() and time.monotonic() < deadline, "the core did not start"
            time.sleep(0.05)
        self.stopped = False

    def poll(self) -> int | None:
        return None if self.thread.is_alive() else 0

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(15)
        self.stopped = not self.thread.is_alive()


# --------------------------------------------------------------------------- tests


def test_installer_gets_d_last_and_unquoted_and_the_core_a_temporary_home_and_a_free_port(fakes, home, monkeypatch, capsys):
    """Acceptance: the faked installer gets `/S /D=<temporary folder>` with `/D=` last and unquoted, the temporary core
    (here the real app) a temporary CINEWRIGHT_HOME and a free CINEWRIGHT_PORT, and the existing installation's
    uninstall entry and Start-menu shortcut are exactly as before afterwards."""
    monkeypatch.setenv("CINEWRIGHT_PORT", "8737")
    fakes.existing_install()
    started: list[InProcessCore] = []

    def start_core(python: Path, env: Mapping[str, str], cwd: Path) -> InProcessCore:
        fakes.cores.append((python, dict(env), cwd))
        core = InProcessCore(env)
        started.append(core)
        return core

    fakes.host.start_core = start_core
    fakes.host.connect = smoke_installer.http_client  # the real client against the real app

    code, lines = fakes.run(capsys)

    # This build ends at the model installs (501) or at the engine, which cannot be set up in a test; both are exit 2.
    assert code == 2, "\n".join(lines)
    assert lines[-1].startswith("Smoke test could not run:") and lines[-1].endswith("Exit code 2."), lines[-1]
    app_dir, data_dir = fakes.root / "app", fakes.root / "data"
    [command] = fakes.run_installer.commands
    assert command == f'"{fakes.installer}" /S /D={app_dir}'
    assert command.endswith(f" /D={app_dir}") and f'"{app_dir}' not in command
    [(python, env, cwd)] = fakes.cores
    assert (python, cwd) == (app_dir / "runtime" / "Scripts" / "python.exe", app_dir)
    assert env["CINEWRIGHT_HOME"] == str(data_dir)
    assert os.environ["CINEWRIGHT_HOME"] == str(home) != str(data_dir)
    port = int(env["CINEWRIGHT_PORT"])
    assert port != 8737 and "CINEWRIGHT_HOST" not in env
    [core] = started
    assert (core.config.home, core.config.port) == (data_dir, port)
    assert (data_dir / "db" / "cinewright.sqlite3").is_file()  # the real core used the temporary data directory
    assert core.stopped and port_is_free(port)
    fakes.assert_existing_install_as_before()
    assert any(str(fakes.root) in line for line in lines[:-1])  # where the temporary installation and its logs stay


def test_installer_path_installs_the_smoke_models_one_by_one_then_runs_the_smoke_test(fakes, smoke_runs, capsys):
    fakes.existing_install()

    code, lines = fakes.run(capsys)

    assert code == 0, "\n".join(lines)
    assert lines[-1] == PASSED
    [(_, env, _)] = fakes.cores
    port = int(env["CINEWRIGHT_PORT"])
    assert smoke_runs == [f"http://127.0.0.1:{port}"]
    assert port_is_free(port)  # the fake core never bound it: the tool chose a port nobody listens on
    # First the capability check with a quantization no model has, then each smoke model's int8, one after another.
    assert fakes.api.installs == [("ltx-2.3-distilled-22b", PROBE_QUANT)] + [(job.model_id, "int8") for job in smoke.SMOKE_JOBS]
    assert set(fakes.api.states.values()) == {"installed"}
    assert fakes.events == ["core started", "core stopped"]
    assert "Engine setup: Installing packages ..." in lines
    fakes.assert_existing_install_as_before()
    assert str(fakes.root) in lines[-2]


def test_without_an_earlier_installation_the_entry_and_the_shortcut_the_installer_made_are_removed(fakes, smoke_runs, capsys):
    code, lines = fakes.run(capsys)

    assert code == 0, "\n".join(lines)
    assert fakes.registry.keys == {}
    assert not fakes.start_menu.exists() and not fakes.desktop.exists()


def test_a_build_without_model_installs_ends_with_exit_2_before_the_engine_setup(fakes, capsys):
    fakes.existing_install()
    fakes.api.can_install = False

    code, lines = fakes.run(capsys)

    assert code == 2
    assert lines[-1].startswith("Smoke test could not run: Model installs need #3, which this build does not contain")
    assert "501 (Not implemented yet: Model installation arrives in phase 1)" in lines[-1], lines[-1]
    assert lines[-1].endswith("Exit code 2."), lines[-1]
    assert fakes.api.installs == [("ltx-2.3-distilled-22b", PROBE_QUANT)]
    assert fakes.api.system_reads == 1  # only the check that the core is up: no wait for the engine setup
    assert fakes.events == ["core started", "core stopped"]
    fakes.assert_existing_install_as_before()


def test_the_snapshot_lies_in_the_temporary_folder_while_the_installer_runs(fakes, smoke_runs, capsys):
    """If the console window is closed while the installer runs, the tool ends before it restores anything; the next
    run finds the snapshot there. After the restore it is gone."""
    fakes.existing_install()

    code, lines = fakes.run(capsys)

    assert code == 0, "\n".join(lines)
    [saved] = fakes.run_installer.snapshots
    assert saved is not None and saved.entry == OLD_ENTRY
    assert saved.files[fakes.start_menu] == FileState(OLD_SHORTCUT, OLD_SHORTCUT_TIMES) and saved.files[fakes.desktop] == FileState(None)
    assert not (fakes.root / SNAPSHOT_FILE).exists()


def interrupted_run(fakes: Fakes) -> Path:
    """What a run leaves whose window was closed while its installer ran: its snapshot of the existing installation,
    and the uninstall entry and the Start-menu shortcut its installer wrote."""
    earlier = fakes.root.parent / "cinewright-smoke-earlier"
    snapshot = smoke_installer.take_snapshot(fakes.host)
    earlier.mkdir()
    (earlier / SNAPSHOT_FILE).write_text(smoke_installer.snapshot_to_json(snapshot), encoding="utf-8")
    fakes.run_installer(installer_command(fakes.installer, earlier / "app"))
    fakes.run_installer.commands.clear()
    fakes.run_installer.snapshots.clear()
    return earlier


def test_a_run_ended_while_its_installer_ran_is_put_right_by_the_next_run(fakes, smoke_runs, capsys):
    fakes.existing_install()
    earlier = interrupted_run(fakes)
    assert fakes.registry.keys[UNINSTALL_KEY]["InstallLocation"][0] == str(earlier / "app")

    code, lines = fakes.run(capsys)

    assert code == 0, "\n".join(lines)
    assert any("An earlier run ended while its installer ran" in line for line in lines)
    [saved] = fakes.run_installer.snapshots  # this run's snapshot is of the repaired installation
    assert saved is not None and saved.entry == OLD_ENTRY
    fakes.assert_existing_install_as_before()
    assert not (earlier / SNAPSHOT_FILE).exists()


def test_an_old_snapshot_does_not_overwrite_an_installation_repaired_since(fakes, smoke_runs, capsys):
    earlier = interrupted_run(fakes)  # no installation existed then: its snapshot holds no entry and no shortcut
    fakes.existing_install()  # the owner has installed Cinewright since, which wrote the entry and the shortcut

    code, lines = fakes.run(capsys)

    assert code == 0, "\n".join(lines)
    assert not any("An earlier run" in line for line in lines)
    fakes.assert_existing_install_as_before()
    assert not (earlier / SNAPSHOT_FILE).exists()


def test_a_snapshot_keeps_every_registry_value_type(tmp_path):
    entry = {
        "DisplayName": ("Cinewright", REG_SZ),
        "DisplayIcon": ("%LOCALAPPDATA%\\Programs\\Cinewright\\cinewright.ico", REG_EXPAND_SZ),
        "NoModify": (1, REG_DWORD),
        "Blob": (b"\x00\xffbytes", REG_BINARY),
        "Lines": (["one", "two"], REG_MULTI_SZ),
    }
    snapshot = Snapshot(entry=entry, files={tmp_path / "a.lnk": FileState(b"L\x00", (1, 2)), tmp_path / "b.lnk": FileState(None)})

    assert smoke_installer.snapshot_from_json(smoke_installer.snapshot_to_json(snapshot)) == snapshot
    assert smoke_installer.snapshot_from_json(smoke_installer.snapshot_to_json(Snapshot(None, {}))) == Snapshot(None, {})
    with pytest.raises(ValueError):
        smoke_installer.snapshot_from_json('{"entry": {"x": 1}, "files": []}')


def test_an_unexpected_error_after_the_core_started_is_exit_2_and_stops_the_core(fakes, monkeypatch, capsys):
    def execute(self: smoke.SmokeRun) -> tuple[int, str]:
        raise OSError(13, "Permission denied", "smoke-9.9.9.json")

    monkeypatch.setattr(smoke.SmokeRun, "execute", execute)

    code, lines = fakes.run(capsys)

    assert code == 2
    assert lines[-1].startswith("Smoke test could not run: A file or folder could not be read or written"), lines[-1]
    assert lines[-1].endswith("Exit code 2.")
    assert fakes.events == ["core started", "core stopped"]


def test_a_failed_installer_ends_with_exit_2_and_restores_the_existing_installation(fakes, capsys):
    fakes.existing_install()
    fakes.run_installer.exit_code = 2

    code, lines = fakes.run(capsys)

    assert code == 2
    assert lines[-1] == "Smoke test could not run: The installer failed with exit code 2. Exit code 2."
    assert fakes.cores == []
    fakes.assert_existing_install_as_before()


@pytest.mark.parametrize(
    ("engine", "text"),
    [
        ({"state": "failed", "error": {"code": "download_failed", "message": "Downloading torch failed."}}, "failed: Downloading torch"),
        ({"state": "not_installed", "error": {"code": "uv_missing", "message": "No uv.exe lies here."}}, "cannot run here: No uv.exe"),
    ],
)
def test_an_engine_that_cannot_be_set_up_ends_with_exit_2(fakes, engine, text, capsys):
    fakes.api.engine = [engine]

    code, lines = fakes.run(capsys)

    assert code == 2 and text in lines[-1] and lines[-1].endswith("Exit code 2."), lines[-1]
    assert fakes.api.installs == [("ltx-2.3-distilled-22b", PROBE_QUANT)]
    assert fakes.events == ["core started", "core stopped"]


def test_a_failed_model_install_ends_with_exit_2(fakes, capsys):
    fakes.api.install_error = "Downloading the transformer failed, so it is not installed."

    code, lines = fakes.run(capsys)

    assert code == 2
    assert "The install of ltx-2.3-distilled-22b int8 failed: Downloading the transformer failed" in lines[-1], lines[-1]
    assert fakes.events == ["core started", "core stopped"]


def test_the_model_store_is_linked_never_copied_and_stays_as_it_was(fakes, smoke_runs, tmp_path, capsys):
    store = tmp_path / "models of the existing installation"
    (store / "umt5-xxl").mkdir(parents=True)
    (store / "z_image_turbo_int8.safetensors").write_bytes(b"weights")
    (store / "umt5-xxl" / "encoder.safetensors").write_bytes(b"encoder")
    seen: list[bytes] = []
    start_core = fakes.host.start_core

    def checking_start_core(python: Path, env: Mapping[str, str], cwd: Path) -> FakeProcess:
        models = Path(env["CINEWRIGHT_HOME"]) / "models"
        assert models.is_junction() or models.is_symlink()
        seen.append((models / "z_image_turbo_int8.safetensors").read_bytes())
        return start_core(python, env, cwd)

    fakes.host.start_core = checking_start_core

    code, lines = fakes.run(capsys, model_store=store)

    assert code == 0, "\n".join(lines)
    assert seen == [b"weights"]
    assert fakes.events == ["core started", "core stopped", "link removed"]  # the link goes after the core stopped
    assert not os.path.lexists(fakes.root / "data" / "models")
    assert (store / "z_image_turbo_int8.safetensors").read_bytes() == b"weights"
    assert (store / "umt5-xxl" / "encoder.safetensors").read_bytes() == b"encoder"


def test_removing_the_link_leaves_the_target_folder_and_its_contents(tmp_path):
    target = tmp_path / "store"
    (target / "sub").mkdir(parents=True)
    (target / "sub" / "file.bin").write_bytes(b"kept")
    link = tmp_path / "data" / "models"
    link.parent.mkdir()

    smoke_installer.link_directory(link, target)
    assert (link / "sub" / "file.bin").read_bytes() == b"kept"
    if sys.platform == "win32":
        assert link.is_junction()  # no administrator rights needed, unlike a symbolic link
    smoke_installer.unlink_directory(link)

    assert not os.path.lexists(link)
    assert (target / "sub" / "file.bin").read_bytes() == b"kept"


def test_a_folder_in_place_of_the_link_is_left_alone(tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "file.bin").write_bytes(b"kept")

    with pytest.raises(OSError):
        smoke_installer.unlink_directory(folder)
    assert (folder / "file.bin").read_bytes() == b"kept"


def test_installer_command_keeps_d_last_and_unquoted_with_spaces():
    installer, folder = Path("C:/Users/Some One/Downloads/CinewrightSetup-1.2.3.exe"), Path("C:/Users/Some One/AppData/Local/Temp/x/app")
    assert installer_command(installer, folder) == f'"{installer}" /S /D={folder}'


def test_model_store_needs_installer(tmp_path, capsys):
    code = devtools.main(["smoke", "--model-store", str(tmp_path)])

    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert code == 2 and "--model-store works only with --installer" in last and last.endswith("Exit code 2.")


def test_a_missing_installer_ends_with_exit_2(tmp_path, capsys):
    code = devtools.main(["smoke", "--installer", str(tmp_path / "CinewrightSetup-9.9.9.exe")])

    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert code == 2 and "does not exist" in last and last.endswith("Exit code 2.")


@pytest.mark.skipif(sys.platform == "win32", reason="on Windows this would run the installer for real")
def test_installer_path_needs_windows(fakes, capsys):
    code = devtools.main(["smoke", "--installer", str(fakes.installer)])

    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert code == 2 and "only on Windows" in last
    assert fakes.registry.keys == {}

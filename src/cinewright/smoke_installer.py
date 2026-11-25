"""The `--installer` path of the GPU smoke test: install a build into a temporary folder and test it there.

  python -m cinewright.devtools smoke --installer CinewrightSetup-<version>.exe [--model-store DIR]

1. The tool notes the current user's uninstall entry (`HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\
   Cinewright`) and the Start-menu and desktop shortcuts, then runs the installer silently: `"<installer>" /S
   /D=<temp>\\app`. NSIS reads everything after `/D=` as the folder, so `/D=` comes last and is not quoted; the tool
   builds this command line itself. The installer registers every install for the current user, so the entry and the
   shortcuts are put back exactly as they were as soon as it has ended. The temporary installation needs neither. The
   snapshot also lies in `<temp>\\existing-install-snapshot.json` until then: if the tool ends while the installer runs
   (its console window closed), the next `--installer` run puts them back first.
2. With `--model-store`, `<temp>\\data\\models` becomes a directory junction to that folder: the models downloaded there
   are reused without copying, and the tool itself never deletes or changes the folder. Model installs of the
   temporary core reuse its files and add the ones it lacks.
3. It starts the installed core as the Start-menu shortcut does (`runtime\\Scripts\\python.exe -m cinewright`), with
   `CINEWRIGHT_HOME=<temp>\\data` and a free `CINEWRIGHT_PORT`: on a port where a core already answers, a start only
   opens the browser. The launcher opens the browser on the temporary core, which shows the engine setup.
4. A build that cannot install models answers 501 to `POST /api/v1/models/{id}/install`. The tool asks with a
   quantization no model has, which a build that can install models refuses without starting anything, so such a
   build ends with exit code 2 before the long engine setup. Otherwise the tool waits until engine provisioning reports
   the engine `ready` in `GET /api/v1/system`, installs each smoke model's `install_quant` one after another (a core
   installs one model at a time) and waits until each is `installed`.
5. It runs the smoke test against the temporary core, whose report lands in `<temp>\\data\\logs`; a job waits up to
   `INSTALLED_ROUTE_WAIT_S` for a worker, because each install restarts the local worker. Then it stops that
   core (Ctrl+Break to its process group, which uvicorn takes as a shutdown, then a kill) and removes the junction.
   The rest of `<temp>` stays, the report with it; the last lines name the folder.

Everything this does to the computer goes through `Host`, so the tests fake the installer, the registry, the
shortcuts and the core. `winreg` and the Windows shell are used only on Windows.
"""

from __future__ import annotations

import base64
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Protocol, TextIO

import httpx

from cinewright.config import ENV_HOME, ENV_HOST, ENV_PORT
from cinewright.smoke import API, ENGINE_ID, HTTP_TIMEOUT_S, SMOKE_JOBS, CannotRun, SmokeRun, problem_text, run_guarded, say, sentence

UNINSTALL_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Cinewright"  # below HKEY_CURRENT_USER
SHORTCUT_NAME = "Cinewright.lnk"
FOLDERID_PROGRAMS = "A77F5D77-2E2B-44C3-A6A2-ABA601054A51"  # NSIS $SMPROGRAMS of a per-user install
FOLDERID_DESKTOP = "B4BFCC3A-DB2C-424C-B029-7FE99A87C641"  # NSIS $DESKTOP
INSTALL_DIR_NAME, DATA_DIR_NAME = "app", "data"
ROOT_PREFIX = "cinewright-smoke-"  # the temporary folder of a run, in the system's temporary folder
SNAPSHOT_FILE = "existing-install-snapshot.json"  # in that folder until the snapshot is restored
RUNTIME_PYTHON = Path("runtime", "Scripts", "python.exe")  # below the installation folder (installer/cinewright.nsi)
CORE_OUTPUT_LOG = "smoke-core-output.log"  # in <temp>\data\logs: what the core prints before its own log starts
PROBE_QUANT = "smoke-capability-check"  # no quantization has this id, so the request never starts an install
NO_MODEL_INSTALLS = "Model installs need #3, which this build does not contain"

INSTALLER_TIMEOUT_S = 1800.0  # setup downloads Python 3.12 and Cinewright's packages
CORE_START_TIMEOUT_S = 120.0
CORE_STOP_TIMEOUT_S = 60.0  # the core first stops its worker and a running engine setup
CORE_POLL_S = 0.5
ENGINE_TIMEOUT_S = 3 * 3600.0  # engine setup downloads WanGP, PyTorch and the attention wheels, then probes them
INSTALL_START_TIMEOUT_S = 120.0
# Each install restarts the local worker, so the first smoke job waits for a cold engine start.
INSTALLED_ROUTE_WAIT_S = 600.0
MODEL_INSTALL_TIMEOUT_S = 6 * 3600.0  # per model: tens of GB without a model store
POLL_S = 5.0
STATUS_EVERY_S = 600.0  # at most one "still installing" line per this many seconds

RegistryValues = dict[str, tuple[Any, int]]  # value name -> (data, winreg value type)


# --------------------------------------------------------------------------- what the tool does to the computer


class Registry(Protocol):
    """Registry keys below HKEY_CURRENT_USER."""

    def read(self, key: str) -> RegistryValues | None:
        """The key's values, or None when the key does not exist."""
        ...

    def write(self, key: str, values: Mapping[str, tuple[Any, int]]) -> None:
        """Create the key if needed and make its values exactly `values`, deleting every other value."""
        ...

    def delete(self, key: str) -> None:
        """Delete the key with everything below it; nothing happens when it does not exist."""
        ...


class CoreProcess(Protocol):
    def poll(self) -> int | None:
        """The exit code once the core has ended, else None."""
        ...

    def stop(self) -> None:
        """Shut the core down and wait until it has ended."""
        ...


def link_directory(link: Path, target: Path) -> None:
    """Make `link` point at the folder `target`: a directory junction on Windows (it needs no administrator rights), a
    symbolic link elsewhere. Nothing is copied."""
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


def unlink_directory(link: Path) -> None:
    """Remove the junction or symbolic link `link` itself; the folder it points at keeps all its contents."""
    if link.is_junction():
        os.rmdir(link)  # removes the junction only, never what lies in its target
    elif link.is_symlink():
        link.unlink()
    elif link.exists():
        raise OSError(f"{link} is no longer a link, so it was left as it is")


def http_client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=HTTP_TIMEOUT_S, trust_env=False)


@dataclass
class Host:
    """What the installer path does to the computer; the tests replace each part."""

    registry: Registry
    shortcuts: Sequence[Path]  # the Start-menu and desktop shortcuts the installer and its uninstaller touch
    run_installer: Callable[[str], int]  # the whole command line -> exit code
    start_core: Callable[[Path, Mapping[str, str], Path], CoreProcess]  # (python, environment, working directory)
    link_dir: Callable[[Path, Path], None] = link_directory  # (link, target)
    unlink_dir: Callable[[Path], None] = unlink_directory
    connect: Callable[[str], httpx.Client] = http_client  # base URL -> client


def installer_command(installer: Path, folder: Path) -> str:
    """The installer's command line: silent, into `folder`. NSIS reads everything after `/D=` as the folder, so `/D=`
    comes last and unquoted, also when the folder holds spaces; Python's own argument quoting would quote it."""
    return f'"{installer}" /S /D={folder}'


def run_installer(command: str) -> int:
    # A string, not a list: on Windows it reaches CreateProcess unchanged, so /D= stays last and unquoted.
    return subprocess.run(command, check=False, timeout=INSTALLER_TIMEOUT_S).returncode  # noqa: S603 - the installer the owner named


def free_port() -> int:
    """A port no process listens on now; the temporary core binds it a moment later."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LocalCore:
    """The temporary core: the installation's `python.exe -m cinewright`, in its own process group."""

    def __init__(self, process: subprocess.Popen[bytes], output: IO[bytes]) -> None:
        self.process = process
        self.output = output

    @classmethod
    def start(cls, python: Path, env: Mapping[str, str], cwd: Path) -> LocalCore:
        logs = Path(env[ENV_HOME]) / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        output = (logs / CORE_OUTPUT_LOG).open("ab")
        try:
            process = subprocess.Popen(  # noqa: S603 - the interpreter of the installation this run made
                [str(python), "-m", "cinewright"],
                cwd=cwd,
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                # Its own process group: Ctrl+C in the console reaches only the tool, which then stops the core itself.
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        except OSError:
            output.close()
            raise
        return cls(process, output)

    def poll(self) -> int | None:
        return self.process.poll()

    def stop(self) -> None:
        """Ask the core to shut down, so it stops its worker and engine setup first; kill it if it has not ended in time.

        The worker and the engine setup's processes end with the core in any case (`cinewright.core.supervisor`)."""
        try:
            if self.process.poll() is None:
                try:
                    self.process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGINT))
                    self.process.wait(CORE_STOP_TIMEOUT_S)
                except (OSError, subprocess.TimeoutExpired):
                    self.process.kill()
                    self.process.wait(CORE_STOP_TIMEOUT_S)
        finally:
            self.output.close()


class WinRegistry:
    """`Registry` on HKEY_CURRENT_USER through winreg; Windows only."""

    def __init__(self) -> None:
        import winreg

        self.winreg = winreg

    def read(self, key: str) -> RegistryValues | None:
        winreg = self.winreg
        try:
            handle = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key)
        except FileNotFoundError:
            return None
        with handle:
            return {name: (data, kind) for name, data, kind in _enumerate(winreg.EnumValue, handle)}

    def write(self, key: str, values: Mapping[str, tuple[Any, int]]) -> None:
        winreg = self.winreg
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, key, 0, winreg.KEY_READ | winreg.KEY_SET_VALUE) as handle:
            for name in [name for name, _, _ in _enumerate(winreg.EnumValue, handle)]:
                if name not in values:
                    winreg.DeleteValue(handle, name)
            for name, (data, kind) in values.items():
                winreg.SetValueEx(handle, name, 0, kind, data)

    def delete(self, key: str) -> None:
        winreg = self.winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as handle:
                subkeys = list(_enumerate(winreg.EnumKey, handle))
        except FileNotFoundError:
            return
        for subkey in subkeys:
            self.delete(f"{key}\\{subkey}")
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key)


def _enumerate(function: Callable[[Any, int], Any], handle: Any) -> Iterator[Any]:
    """The items of a winreg enumeration (`EnumValue`, `EnumKey`) until it reports that there are no more."""
    index = 0
    while True:
        try:
            item = function(handle, index)
        except OSError:  # ERROR_NO_MORE_ITEMS
            return
        yield item
        index += 1


def known_folder(folder_id: str) -> Path:
    """A Windows known folder (`SHGetKnownFolderPath`), so a desktop moved to OneDrive is found where it is."""
    import ctypes
    from ctypes import wintypes

    class Guid(ctypes.Structure):
        _fields_ = (("data1", wintypes.DWORD), ("data2", wintypes.WORD), ("data3", wintypes.WORD), ("data4", ctypes.c_ubyte * 8))

    get_path = ctypes.WinDLL("shell32").SHGetKnownFolderPath
    get_path.argtypes = (ctypes.POINTER(Guid), wintypes.DWORD, wintypes.HANDLE, ctypes.POINTER(ctypes.c_wchar_p))
    get_path.restype = ctypes.HRESULT  # a failure raises OSError
    path = ctypes.c_wchar_p()
    get_path(ctypes.byref(Guid.from_buffer_copy(uuid.UUID(folder_id).bytes_le)), 0, None, ctypes.byref(path))
    try:
        return Path(path.value or "")
    finally:
        ctypes.WinDLL("ole32").CoTaskMemFree(path)


def windows_host() -> Host:
    if sys.platform != "win32":
        raise CannotRun("--installer works only on Windows, where Cinewright's installer runs.")
    try:
        shortcuts = tuple(known_folder(folder) / SHORTCUT_NAME for folder in (FOLDERID_PROGRAMS, FOLDERID_DESKTOP))
    except OSError as exc:
        raise CannotRun(f"The Start-menu and desktop folders could not be found ({exc}).") from exc
    return Host(registry=WinRegistry(), shortcuts=shortcuts, run_installer=run_installer, start_core=LocalCore.start)


# --------------------------------------------------------------------------- the existing installation


@dataclass(frozen=True)
class FileState:
    content: bytes | None  # None: the file did not exist
    times_ns: tuple[int, int] | None = None  # access and modification time


@dataclass(frozen=True)
class Snapshot:
    """The uninstall entry and the shortcuts of an existing installation (or their absence) before the installer ran."""

    entry: RegistryValues | None
    files: dict[Path, FileState]


def file_state(path: Path) -> FileState:
    try:
        stat = path.stat()
        content = path.read_bytes()
    except FileNotFoundError:
        return FileState(None)
    return FileState(content, (stat.st_atime_ns, stat.st_mtime_ns))


def take_snapshot(host: Host) -> Snapshot:
    return Snapshot(entry=host.registry.read(UNINSTALL_KEY), files={path: file_state(path) for path in host.shortcuts})


def restore_snapshot(host: Host, snapshot: Snapshot) -> None:
    """Put the uninstall entry and the shortcuts back as they were, touching only what differs; raises OSError when
    one of them is not as before afterwards."""
    registry = host.registry
    if snapshot.entry is None:
        registry.delete(UNINSTALL_KEY)
    elif registry.read(UNINSTALL_KEY) != snapshot.entry:
        registry.write(UNINSTALL_KEY, snapshot.entry)
    for path, before in snapshot.files.items():
        if before.content is None:
            path.unlink(missing_ok=True)
        elif file_state(path).content != before.content:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(before.content)
            if before.times_ns is not None:
                os.utime(path, ns=before.times_ns)
    if registry.read(UNINSTALL_KEY) != snapshot.entry:
        raise OSError(f"HKEY_CURRENT_USER\\{UNINSTALL_KEY} differs from before")
    for path, before in snapshot.files.items():
        if file_state(path).content != before.content:
            raise OSError(f"{path} differs from before")


def snapshot_to_json(snapshot: Snapshot) -> str:
    """The snapshot as JSON: each registry value with its type (bytes as base64), each shortcut's content as base64."""

    def value(data: Any, kind: int) -> dict[str, Any]:
        if isinstance(data, bytes):
            return {"kind": kind, "base64": base64.b64encode(data).decode("ascii")}
        return {"kind": kind, "data": data}  # a str, an int, a list of str (REG_MULTI_SZ) or None

    entry = None if snapshot.entry is None else {name: value(data, kind) for name, (data, kind) in snapshot.entry.items()}
    files = [
        {
            "path": str(path),
            "content": None if state.content is None else base64.b64encode(state.content).decode("ascii"),
            "times_ns": None if state.times_ns is None else list(state.times_ns),
        }
        for path, state in snapshot.files.items()
    ]
    return json.dumps({"entry": entry, "files": files}, indent=2) + "\n"


def snapshot_from_json(text: str) -> Snapshot:
    """The snapshot `snapshot_to_json` wrote; ValueError when `text` is not one."""
    try:
        raw = json.loads(text)
        entry: RegistryValues | None = None
        if raw["entry"] is not None:
            entry = {
                str(name): (base64.b64decode(value["base64"]) if "base64" in value else value["data"], int(value["kind"]))
                for name, value in raw["entry"].items()
            }
        files = {
            Path(item["path"]): FileState(
                None if item["content"] is None else base64.b64decode(item["content"]),
                None if item["times_ns"] is None else (int(item["times_ns"][0]), int(item["times_ns"][1])),
            )
            for item in raw["files"]
        }
    except (LookupError, TypeError, AttributeError) as exc:
        raise ValueError(f"not a snapshot ({exc!r})") from exc
    return Snapshot(entry=entry, files=files)


def is_inside(path: str, folder: Path) -> bool:
    """Whether `path` is `folder` or lies in it, compared as the system compares paths (on Windows without letter case)."""
    path_key, folder_key = (os.path.normcase(os.path.normpath(str(item))) for item in (path, folder))
    return path_key == folder_key or path_key.startswith(folder_key.rstrip(os.sep) + os.sep)


# --------------------------------------------------------------------------- the run


def quant_state(models: Mapping[str, Any], model_id: str, quant: str) -> tuple[str | None, str | None]:
    """(install state, error) of one quantization in a `GET /models` answer; (None, None) when it is not listed."""
    for item in models.get("items", []):
        if item.get("id") == model_id:
            for entry in item.get("quantizations", []):
                if entry.get("quant") == quant:
                    return entry.get("install_state"), entry.get("error")
    return None, None


def no_model_installs(model_id: str, response: httpx.Response) -> str:
    problem = problem_text(response).rstrip(".")
    return f"{NO_MODEL_INSTALLS}: POST {API}/models/{model_id}/install answered {response.status_code} ({problem})."


class InstallerRun:
    def __init__(self, host: Host, out: TextIO, installer: Path, model_store: Path | None, root: Path) -> None:
        self.host = host
        self.out = out
        self.installer = installer
        self.model_store = model_store
        self.root = root
        self.install_dir = root / INSTALL_DIR_NAME
        self.data_dir = root / DATA_DIR_NAME
        self.models_link = self.data_dir / "models"
        self.core: CoreProcess | None = None
        self.linked = False

    def say(self, text: str) -> None:
        say(self.out, text)

    def run(self) -> int:
        try:
            code, result = self.steps()
        finally:
            self.clean_up()
        self.say(result)
        return code

    def steps(self) -> tuple[int, str]:
        self.say(f"Installing {self.installer.name} silently into {self.install_dir} ...")
        self.install()
        python = self.install_dir / RUNTIME_PYTHON
        if not python.is_file():
            raise CannotRun(f"The installer ended without an error, but {python} does not exist.")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.model_store is not None:
            try:
                self.host.link_dir(self.models_link, self.model_store)
            except OSError as exc:
                raise CannotRun(f"The model store {self.model_store} could not be linked as {self.models_link} ({exc}).") from exc
            self.linked = True
            self.say(f"Models: {self.models_link} links to {self.model_store}; nothing is copied.")
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        env = {name: value for name, value in os.environ.items() if name.upper() != ENV_HOST}
        env[ENV_HOME], env[ENV_PORT] = str(self.data_dir), str(port)
        self.say(f"Starting the installed Cinewright on {url} with its data in {self.data_dir} ...")
        try:
            self.core = self.host.start_core(python, env, self.install_dir)
        except OSError as exc:
            raise CannotRun(f"The installed Cinewright could not be started ({exc}).") from exc
        with self.host.connect(url) as http:
            smoke = SmokeRun(http, self.out, route_wait_s=INSTALLED_ROUTE_WAIT_S)
            self.wait_for_core(http)
            self.check_model_installs(http)
            self.wait_for_engine(smoke)
            self.install_models(smoke, http)
            return smoke.execute()

    def install(self) -> None:
        """Run the installer; put the existing installation's uninstall entry and shortcuts back right after it.

        The snapshot also lies in `<temp>/SNAPSHOT_FILE` until it is restored: if this process ends while the installer
        runs (its console window closed), the next `--installer` run restores it (`recover_interrupted_runs`)."""
        self.recover_interrupted_runs()
        try:
            snapshot = take_snapshot(self.host)
        except OSError as exc:
            raise CannotRun(f"The uninstall entry or the shortcuts could not be read ({exc}), so the installer was not run.") from exc
        snapshot_file = self.root / SNAPSHOT_FILE
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            snapshot_file.write_text(snapshot_to_json(snapshot), encoding="utf-8", newline="\n")
        except OSError as exc:
            raise CannotRun(f"The uninstall entry and the shortcuts could not be saved ({exc}), so the installer was not run.") from exc
        command = installer_command(self.installer, self.install_dir)
        try:
            code = self.host.run_installer(command)
        except subprocess.TimeoutExpired as exc:
            raise CannotRun(f"The installer did not finish within {INSTALLER_TIMEOUT_S / 60:.0f} minutes.") from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise CannotRun(f"The installer could not run ({exc}).") from exc
        finally:
            self.restore(snapshot, snapshot_file)
        if code != 0:
            raise CannotRun(f"The installer failed with exit code {code}.")
        self.say("Installed. The uninstall entry and the shortcuts are as they were before.")

    def recover_interrupted_runs(self) -> None:
        """Put back what an earlier run left: its snapshot file stays only when it ended while its installer ran, before
        it could restore the uninstall entry and the shortcuts. Only while the entry still points into that run's
        folder; an entry the owner has repaired since stays as it is, and the stale snapshot goes."""
        for path in sorted(self.root.parent.glob(f"{ROOT_PREFIX}*/{SNAPSHOT_FILE}")):
            folder = path.parent
            if folder == self.root:
                continue
            try:
                snapshot = snapshot_from_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                self.say(f"Warning: the snapshot {path} of an earlier run could not be read ({exc}); it was left as it is.")
                continue
            location = ((self.host.registry.read(UNINSTALL_KEY) or {}).get("InstallLocation") or (None, 0))[0]
            if isinstance(location, str) and is_inside(location, folder):
                self.say(f"An earlier run ended while its installer ran into {folder}; putting back the uninstall entry and the shortcuts.")
                self.restore(snapshot, path)
            else:
                path.unlink()

    def restore(self, snapshot: Snapshot, snapshot_file: Path) -> None:
        """Restore `snapshot`, then delete its file; the file stays when the restore failed, for the next run."""
        try:
            restore_snapshot(self.host, snapshot)
        except OSError as exc:
            raise CannotRun(
                f"The uninstall entry or the shortcuts could not be put back as they were before the installer ran ({exc}); "
                "if Cinewright was installed, run its installer once more to repair them."
            ) from exc
        snapshot_file.unlink(missing_ok=True)

    def clean_up(self) -> None:
        """Stop the temporary core, then remove the link to the model store; the rest of the temporary folder stays."""
        if self.core is not None:
            try:
                self.core.stop()
            except Exception as exc:  # noqa: BLE001 - the rest of the clean-up must still run
                self.say(f"Warning: the temporary Cinewright could not be stopped ({exc}).")
        if self.linked:
            try:
                self.host.unlink_dir(self.models_link)
            except OSError as exc:
                self.say(
                    f"Warning: the link {self.models_link} to the model store could not be removed ({exc}); "
                    "remove the link only, not the folder it points to."
                )
        if self.root.exists():
            self.say(f"The temporary installation and its data stay in {self.root}; delete that folder when you no longer need the report.")

    def log_hint(self) -> str:
        return f"its logs are in {self.data_dir / 'logs'}."

    def check_core(self) -> None:
        code = self.core.poll() if self.core is not None else None
        if code is not None:
            raise CannotRun(f"The installed Cinewright stopped with exit code {code}; {self.log_hint()}")

    def wait_for_core(self, http: httpx.Client) -> None:
        """Wait until the temporary core answers, and check that it is the one with the temporary data directory."""
        deadline = time.monotonic() + CORE_START_TIMEOUT_S
        while True:
            self.check_core()
            try:
                response = http.get(f"{API}/system", timeout=2.0)
            except httpx.TransportError:
                response = None
            if response is not None and response.status_code == 200:
                try:
                    system = response.json()
                except ValueError:
                    system = {}
                if system.get("name") != "Cinewright" or Path(str(system.get("data_dir"))) != self.data_dir:
                    raise CannotRun(f"Another server answers on {http.base_url}, not the temporary installation.")
                return
            if time.monotonic() >= deadline:
                raise CannotRun(f"The installed Cinewright did not answer within {CORE_START_TIMEOUT_S:.0f} s; {self.log_hint()}")
            time.sleep(CORE_POLL_S)

    def check_model_installs(self, http: httpx.Client) -> None:
        """Before the long engine setup: a build without model installs answers 501 to an install request. A build
        with them refuses a quantization no model has, so this request never starts an install."""
        model_id = SMOKE_JOBS[0].model_id
        response = http.post(f"{API}/models/{model_id}/install", params={"quant": PROBE_QUANT})
        if response.status_code == 501:
            raise CannotRun(no_model_installs(model_id, response))

    def wait_for_engine(self, smoke: SmokeRun) -> None:
        """Wait until engine provisioning reports the engine `ready`; its failure ends the run."""
        deadline = time.monotonic() + ENGINE_TIMEOUT_S
        shown = ""
        while True:
            self.check_core()
            engines = smoke.get_json("/system").get("engines") or []
            engine = next((item for item in engines if item.get("id") == ENGINE_ID), None)
            if engine is None:
                raise CannotRun(f"The installed Cinewright reports no {ENGINE_ID} engine.")
            state, error = engine.get("state"), engine.get("error") or {}
            if state == "ready":
                self.say(f"Engine {ENGINE_ID} is ready ({engine.get('version') or 'unknown version'}).")
                return
            if state == "failed" or error:
                outcome = "failed" if state == "failed" else "cannot run here"
                raise CannotRun(f"The engine setup {outcome}: {sentence(error.get('message') or 'no reason given')}")
            status = f"Engine setup: {(engine.get('progress') or {}).get('phase') or state}"
            if status != shown:
                self.say(f"{status} ...")
                shown = status
            if time.monotonic() >= deadline:
                raise CannotRun(f"The engine setup did not finish within {ENGINE_TIMEOUT_S / 3600:.0f} hours; {self.log_hint()}")
            time.sleep(POLL_S)

    def install_models(self, smoke: SmokeRun, http: httpx.Client) -> None:
        """Install each smoke model's `install_quant`, one after another."""
        for job in SMOKE_JOBS:
            label = f"{job.model_id} {job.install_quant}"
            state, _ = quant_state(smoke.get_json("/models"), job.model_id, job.install_quant)
            if state != "installed":
                response = http.post(f"{API}/models/{job.model_id}/install", params={"quant": job.install_quant})
                if response.status_code == 501:
                    raise CannotRun(no_model_installs(job.model_id, response))
                if not response.is_success:
                    raise CannotRun(f"The install of {label} was refused: {sentence(problem_text(response))}")
                self.say(f"Installing {label} ...")
                self.wait_for_install(smoke, job.model_id, job.install_quant)
            self.say(f"{label} is installed.")

    def wait_for_install(self, smoke: SmokeRun, model_id: str, quant: str) -> None:
        label = f"{model_id} {quant}"
        started = last_line = time.monotonic()
        while True:
            self.check_core()
            state, error = quant_state(smoke.get_json("/models"), model_id, quant)
            if state == "installed":
                return
            if state == "failed":
                raise CannotRun(f"The install of {label} failed: {sentence(error or 'no reason given')}")
            if state is None:
                raise CannotRun(f"GET {API}/models does not list {label}.")
            now = time.monotonic()
            if state == "absent" and now - started >= INSTALL_START_TIMEOUT_S:
                raise CannotRun(f"The install of {label} did not start within {INSTALL_START_TIMEOUT_S:.0f} s.")
            if now - started >= MODEL_INSTALL_TIMEOUT_S:
                raise CannotRun(f"The install of {label} did not finish within {MODEL_INSTALL_TIMEOUT_S / 3600:.0f} hours.")
            if now - last_line >= STATUS_EVERY_S:
                self.say(f"  still installing {label} after {(now - started) / 60:.0f} minutes ...")
                last_line = now
            time.sleep(POLL_S)


def run(out: TextIO, installer: Path, model_store: Path | None, *, host: Host | None = None, root: Path | None = None) -> int:
    """The smoke test against a temporary installation of `installer`; returns the exit code."""

    def installer_run() -> int:
        if not installer.is_file():
            raise CannotRun(f"The installer {installer} does not exist.")
        if model_store is not None and not model_store.is_dir():
            raise CannotRun(f"The model store {model_store} is not a folder.")
        run_host = host if host is not None else windows_host()
        run_root = root if root is not None else Path(tempfile.mkdtemp(prefix=ROOT_PREFIX))
        store = model_store.resolve() if model_store is not None else None
        return InstallerRun(run_host, out, installer.resolve(), store, run_root).run()

    return run_guarded(out, installer_run, core="The installed Cinewright")

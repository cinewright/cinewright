"""Engine provisioning with mocked downloads and a mocked uv: no network, no GPU, platform-neutral.

Downloads go to an `httpx.MockTransport` that serves small stand-ins for the pinned archives and wheels (with Range
support), and a fake runner stands in for uv and for the attention probe. The pin is the bundled Wan2GP pin with its
hashes and sizes replaced by those of the stand-ins.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import json
import os
import sys
import threading
import time
import types
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
import pytest
import yaml
from conftest import API, BASE_URL, TEST_INSTALLED, fake_detect, mark_installed
from fastapi.testclient import TestClient

from cinewright.api.routes.jobs import sse_frame
from cinewright.api.schemas import ClientKind, EngineInfo, JobEvent, JobRequest, SystemEvent, WorkerRegistration
from cinewright.app import create_app
from cinewright.core.core import Core
from cinewright.core.engines import DEFAULT_PATH, REGISTRY_DIR, EnginePin
from cinewright.core.provisioning import (
    OPTIONAL_ATTEMPTS,
    PROBE_SENTINEL,
    STEPS,
    Cancelled,
    Downloader,
    DownloadError,
    EngineLayout,
    EngineProvisioner,
    HashMismatch,
    ProcessRunner,
    ProvisioningStatus,
    RunResult,
    StartFailed,
    build_requirements,
    bundled_uv,
    file_name,
    lock_packages,
    parse_probe,
    sdist_only,
    uv_environment,
)
from cinewright.core.registry import Registry
from cinewright.paths import DataPaths
from cinewright_worker.engines.wan2gp import Wan2GPAdapter, attention_mode, read_provisioning_record

RAW = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))["wan2gp"]
COMMIT = RAW["commit"]
ARCHIVE_URL = RAW["archive_url"]
FFMPEG_URL = RAW["ffmpeg"]["win_amd64"]["url"]
BIN_DIR = RAW["ffmpeg"]["win_amd64"]["bin_dir"]
TRITON_URL = RAW["optional_wheels"]["triton-windows"]["url"]
SAGE_URL = RAW["optional_wheels"]["sageattention"]["url"]
FFMPEG_FILES = {"ffmpeg.exe": b"ffmpeg binary", "ffprobe.exe": b"ffprobe binary", "ffplay.exe": b"ffplay binary"}
SETUPTOOLS_WHEEL = "https://files.pythonhosted.org/packages/setuptools-84.2.1-py3-none-any.whl"
CHUMPY_WHEEL = "https://github.com/deepbeepmeep/chumpy/releases/download/v0.71/chumpy-0.71-py3-none-any.whl"


def _hashes(char: str) -> str:
    return f'hashes = {{ sha256 = "{char * 64}" }}'


# A wheel entry (the build backend), an sdist-only entry and a direct-URL archive entry, as in the bundled lock.
TEST_LOCK = f"""lock-version = "1.0"
created-by = "test"
requires-python = ">=3.11"

[[packages]]
name = "setuptools"
version = "84.2.1"
sdist = {{ url = "https://files.pythonhosted.org/packages/setuptools-84.2.1.tar.gz", {_hashes("a")} }}
wheels = [{{ url = "{SETUPTOOLS_WHEEL}", {_hashes("b")} }}]

[[packages]]
name = "docopt"
version = "0.6.2"
sdist = {{ url = "https://files.pythonhosted.org/packages/docopt-0.6.2.tar.gz", {_hashes("c")} }}

[[packages]]
name = "chumpy"
version = "0.71"
archive = {{ url = "{CHUMPY_WHEEL}", {_hashes("d")} }}
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def zip_bytes(files: dict[str, bytes], dirs: tuple[str, ...] = ()) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipped:
        for name in dirs:
            zipped.writestr(name, b"")
        for name, data in files.items():
            zipped.writestr(name, data)
    return buffer.getvalue()


SOURCE = zip_bytes(
    {
        f"WanGP-{COMMIT}/wgp.py": b"WanGP_version = '10.0'\n",
        f"WanGP-{COMMIT}/shared/api.py": b"def init(**kwargs): ...\n",
        f"WanGP-{COMMIT}/requirements.txt": b"torch\n",
    },
    dirs=(f"WanGP-{COMMIT}/", f"WanGP-{COMMIT}/shared/"),
)
FFMPEG = zip_bytes({**{f"{BIN_DIR}/{name}": data for name, data in FFMPEG_FILES.items()}, "ffmpeg-9.1-essentials_build/README.txt": b"x"})
TRITON = b"triton wheel"
SAGE = b"sageattention wheel"


def make_pin(registry: Path, **changes) -> EnginePin:
    """The bundled Wan2GP pin with the hashes and sizes of the stand-ins (and `changes`, /-separated field paths)."""
    data = copy.deepcopy(RAW)
    data.update(archive_sha256=sha256(SOURCE), archive_size=len(SOURCE))
    build = data["ffmpeg"]["win_amd64"]
    build.update(sha256=sha256(FFMPEG), size=len(FFMPEG))
    build["files"] = {name: {"sha256": sha256(content), "size": len(content)} for name, content in FFMPEG_FILES.items()}
    data["optional_wheels"]["triton-windows"]["sha256"] = sha256(TRITON)
    data["optional_wheels"]["sageattention"]["sha256"] = sha256(SAGE)
    for field, value in changes.items():
        *parents, key = field.split("/")
        node = data
        for part in parents:
            node = node[part]
        node[key] = value
    lock = registry / data["lock"]
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(TEST_LOCK, encoding="utf-8")
    return EnginePin.model_validate(data, context={"registry_dir": registry})


class InterruptedStream(httpx.SyncByteStream):
    """A response body that breaks off after `data`, like a dropped connection."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def __iter__(self):
        yield self.data
        raise httpx.ReadError("connection lost")


class FakeServer:
    """Serves the stand-ins by URL, with Range support.

    `cut[url] = n` breaks the next download of `url` off after n bytes; `unavailable[url] = n` answers the next n
    requests for `url` with 503.
    """

    def __init__(self) -> None:
        self.files = {ARCHIVE_URL: SOURCE, FFMPEG_URL: FFMPEG, TRITON_URL: TRITON, SAGE_URL: SAGE}
        self.requests: list[tuple[str, str | None]] = []
        self.cut: dict[str, int] = {}
        self.unavailable: dict[str, int] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        ranged = request.headers.get("range")
        self.requests.append((url, ranged))
        if self.unavailable.get(url, 0) > 0:
            self.unavailable[url] -= 1
            return httpx.Response(503)
        body = self.files.get(url)
        if body is None:
            return httpx.Response(404)
        start = int(ranged.removeprefix("bytes=").split("-")[0]) if ranged else 0
        status = 206 if ranged else 200
        headers = {"content-range": f"bytes {start}-{len(body) - 1}/{len(body)}"} if ranged else {}
        data = body[start:]
        if url in self.cut:
            return httpx.Response(status, headers=headers, stream=InterruptedStream(data[: self.cut.pop(url)]))
        return httpx.Response(status, headers=headers, content=data)

    def urls(self) -> list[str]:
        return [url for url, _ in self.requests]


class FakeUv:
    """Stands in for uv and the probe: `uv venv` creates the interpreter file, everything else succeeds unless set."""

    def __init__(self, probe: str = "") -> None:
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self.results: dict[str, RunResult] = {}
        self.probe = probe
        self.block: threading.Event | None = None  # the lock install waits for cancellation while set

    @staticmethod
    def kind(argv: list[str]) -> str:
        if argv[1:2] == ["venv"]:
            return "venv"
        if argv[1:3] == ["pip", "install"]:
            if "--preview-features" in argv:
                return "lock"
            return "build" if "--require-hashes" in argv else "optional"
        return "probe" if "-c" in argv else "other"

    def __call__(self, argv, env, cancelled, timeout=None) -> RunResult:
        argv = list(argv)
        kind = self.kind(argv)
        self.calls.append((argv, dict(env)))
        if kind == "lock" and self.block is not None:
            self.block.set()
            while not cancelled():
                time.sleep(0.01)
            raise Cancelled  # as ProcessRunner does after killing uv
        if kind in self.results:
            return self.results[kind]
        if kind == "venv":
            python = Path(argv[-1]) / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_bytes(b"")
        if kind == "probe":
            return RunResult(0, self.probe)
        return RunResult(0, "")

    def kinds(self) -> list[str]:
        return [self.kind(argv) for argv, _ in self.calls]

    def argv(self, kind: str) -> list[str]:
        return next(argv for argv, _ in self.calls if self.kind(argv) == kind)


class Setup:
    def __init__(self, tmp_path: Path, **pin_changes) -> None:
        self.paths = DataPaths(tmp_path / "home")
        self.paths.ensure()
        self.registry = tmp_path / "engine-registry"
        self.pin = make_pin(self.registry, **pin_changes)
        self.lock = self.registry / self.pin.lock
        self.uv = tmp_path / "install" / "uv.exe"
        self.uv.parent.mkdir(parents=True)
        self.uv.write_bytes(b"")
        self.server = FakeServer()
        self.runner = FakeUv()
        self.events: list[ProvisioningStatus] = []

    def provisioner(self, server: FakeServer | None = None, runner=None, platform: str = "win32", uv: Path | None = None):
        provisioner = EngineProvisioner(
            self.paths,
            self.pin,
            uv=self.uv if uv is None else uv,
            lock=self.lock,
            downloader=Downloader(httpx.MockTransport((server or self.server).handler)),
            runner=runner or self.runner,
            platform=platform,
            environ={"PATH": "/bin", "UV_INDEX_URL": "https://evil.example/simple", "PIP_EXTRA_INDEX_URL": "https://evil.example"},
        )
        provisioner.on_change = self.events.append
        return provisioner

    @property
    def engine_dir(self) -> Path:
        return self.paths.engines_dir.resolve() / "wan2gp"

    @property
    def root(self) -> Path:
        return self.engine_dir / COMMIT


def finish(provisioner: EngineProvisioner, timeout: float = 10) -> ProvisioningStatus:
    deadline = time.monotonic() + timeout
    while provisioner.status().state == "provisioning":
        if time.monotonic() > deadline:
            raise AssertionError("provisioning did not finish")
        time.sleep(0.01)
    return provisioner.status()


@pytest.fixture
def setup(tmp_path) -> Setup:
    return Setup(tmp_path)


# ---------------------------------------------------------------------- the state machine


def test_first_start_provisions_the_pinned_engine(setup):
    provisioner = setup.provisioner()
    assert provisioner.status() == ProvisioningStatus("not_installed")
    assert provisioner.start()
    status = finish(provisioner)
    assert status == ProvisioningStatus("ready", version=COMMIT)

    # not_installed -> provisioning (with progress through every step) -> ready
    states = [event.state for event in setup.events]
    assert states[0] == "provisioning" and states[-1] == "ready" and set(states) == {"provisioning", "ready"}
    progress = [event.progress for event in setup.events if event.progress is not None]
    assert progress[0].phase == STEPS[0].phase
    assert {p.step for p in progress} == set(range(1, len(STEPS) + 1))
    assert all(p.total_steps == len(STEPS) for p in progress)
    percents = [p.percent for p in progress]
    assert percents == sorted(percents) and 0 <= percents[0] and percents[-1] < 100

    # the Wan2GP root holds wgp.py directly, plus the pinned ffmpeg binaries
    assert (setup.root / "wgp.py").is_file() and (setup.root / "shared" / "api.py").is_file()
    for name, content in FFMPEG_FILES.items():
        assert (setup.root / "ffmpeg_bins" / name).read_bytes() == content
    assert not (setup.engine_dir / f"{COMMIT}.partial").exists()

    record = read_provisioning_record(str(setup.paths.root))
    assert record["commit"] == COMMIT
    assert record["root"] == str(setup.root)
    assert record["python"] == str(setup.engine_dir / "env" / "Scripts" / "python.exe")
    assert record["attention"] == "sdpa"  # the fake probe printed nothing
    assert record["optional_wheels"] == {"triton-windows": "installed", "sageattention": "installed"}


def test_uv_installs_exactly_the_lock_without_any_index(setup):
    provisioner = setup.provisioner()
    provisioner.start()
    assert finish(provisioner).state == "ready"
    assert setup.runner.kinds() == ["venv", "build", "lock", "optional", "probe"]
    uv, python = str(setup.uv), str(setup.engine_dir / "env" / "Scripts" / "python.exe")

    venv = setup.runner.argv("venv")
    assert venv[:2] == [uv, "venv"]
    assert venv[venv.index("--python") + 1] == "3.11.14" and "--managed-python" in venv and "--clear" in venv
    build = setup.runner.argv("build")
    assert build[:3] == [uv, "pip", "install"] and build[build.index("--python") + 1] == python
    assert {"--require-hashes", "--no-deps", "--no-index"} <= set(build)
    requirements = Path(build[build.index("-r") + 1]).read_text(encoding="utf-8")
    assert requirements == f"setuptools @ {SETUPTOOLS_WHEEL} --hash=sha256:{'b' * 64}\n"
    lock = setup.runner.argv("lock")
    assert lock[lock.index("-r") + 1] == str(setup.lock)
    assert {"--require-hashes", "--no-index", "--no-build-isolation"} <= set(lock)
    assert not any("index-url" in arg or arg == "--index" for arg in lock)
    optional = setup.runner.argv("optional")
    assert {"--no-deps", "--no-index"} <= set(optional)
    assert sorted(Path(arg).name for arg in optional if arg.endswith(".whl")) == sorted(file_name(url) for url in (TRITON_URL, SAGE_URL))
    assert setup.runner.argv("probe")[:3] == [python, "-I", "-c"]

    for argv, env in setup.runner.calls:
        assert env["UV_NO_CONFIG"] == "1"
        assert env["UV_CACHE_DIR"] == str(setup.paths.engines_dir.resolve() / "uv-cache")
        assert env["UV_PYTHON_INSTALL_DIR"] == str(setup.paths.engines_dir.resolve() / "python")
        assert "UV_INDEX_URL" not in env and "PIP_EXTRA_INDEX_URL" not in env
        assert env["PATH"] == "/bin"


def test_a_second_start_downloads_and_installs_nothing(setup):
    first = setup.provisioner()
    first.start()
    assert finish(first).state == "ready"

    def refuse(request):
        raise AssertionError(f"network access: {request.url}")

    def no_uv(*args, **kwargs):
        raise AssertionError(f"uv ran: {args}")

    server = FakeServer()
    server.handler = refuse
    second = setup.provisioner(server=server, runner=no_uv)
    assert second.status() == ProvisioningStatus("ready", version=COMMIT)
    assert not second.start()
    assert not second.retry_optional()  # every optional wheel is installed
    assert second.status().state == "ready"


def test_a_record_of_another_pin_is_not_ready(setup, tmp_path):
    first = setup.provisioner()
    first.start()
    assert finish(first).state == "ready"
    setup.lock.write_text(TEST_LOCK + "\n# changed\n", encoding="utf-8")  # an update that changes the lock
    assert setup.provisioner().status().state == "not_installed"


def test_a_missing_ffmpeg_file_is_not_ready(setup):
    first = setup.provisioner()
    first.start()
    assert finish(first).state == "ready"
    (setup.root / "ffmpeg_bins" / "ffprobe.exe").unlink()
    assert setup.provisioner().status().state == "not_installed"


# ---------------------------------------------------------------------- refusals


def test_an_archive_with_the_wrong_hash_is_refused(setup):
    setup.server.files[ARCHIVE_URL] = SOURCE.replace(b"WanGP_version", b"Tampered_name")
    provisioner = setup.provisioner()
    provisioner.start()
    status = finish(provisioner)
    assert status.state == "failed"
    assert (status.error.code, status.error.action) == ("hash_mismatch", "retry")
    assert "WanGP" in status.error.message
    assert f" from {urlsplit(ARCHIVE_URL).hostname} did not match" in status.error.message  # names the host it came from
    assert setup.events[-1] == status
    assert list((setup.engine_dir / "downloads").iterdir()) == []  # the bytes were deleted
    assert not setup.root.exists() and not (setup.engine_dir / "provisioning.json").exists()
    assert setup.runner.calls == []


def test_an_ffmpeg_archive_with_the_wrong_hash_is_refused(setup):
    setup.server.files[FFMPEG_URL] = FFMPEG + b"x"
    provisioner = setup.provisioner()
    provisioner.start()
    status = finish(provisioner)
    assert (status.state, status.error.code) == ("failed", "hash_mismatch")
    assert "ffmpeg" in status.error.message
    assert not (setup.root / "ffmpeg_bins").exists()
    assert setup.runner.calls == []


def test_an_ffmpeg_file_with_the_wrong_hash_is_refused(tmp_path):
    setup = Setup(tmp_path, **{"ffmpeg/win_amd64/files/ffprobe.exe/sha256": "0" * 64})
    provisioner = setup.provisioner()
    provisioner.start()
    status = finish(provisioner)
    # The archive matched its pin, so a retry would extract the same file: no retry is offered.
    assert (status.state, status.error.code, status.error.action) == ("failed", "hash_mismatch", None)
    assert "ffprobe.exe" in status.error.message
    assert list((setup.root / "ffmpeg_bins").iterdir()) == []  # none placed, not even the ones that matched
    assert setup.runner.calls == []


def test_a_verified_archive_that_is_no_zip_offers_no_retry(setup):
    broken = b"PK but not a zip archive"
    setup.pin = make_pin(setup.registry, archive_sha256=sha256(broken), archive_size=len(broken))
    setup.server.files[ARCHIVE_URL] = broken
    provisioner = setup.provisioner()
    provisioner.start()
    status = finish(provisioner)
    assert (status.state, status.error.code, status.error.action) == ("failed", "archive_invalid", None)


def test_a_lock_package_with_the_wrong_hash_is_refused(setup):
    setup.runner.results["lock"] = RunResult(1, "error: Failed to download `torch==2.10.0`\n  cause: Hash mismatch for `torch==2.10.0`\n")
    provisioner = setup.provisioner()
    provisioner.start()
    status = finish(provisioner)
    assert (status.state, status.error.code, status.error.action) == ("failed", "hash_mismatch", "retry")
    assert not (setup.engine_dir / "provisioning.json").exists()
    assert "optional" not in setup.runner.kinds()


def test_a_failed_install_names_the_log(setup):
    setup.runner.results["venv"] = RunResult(2, "error: No download found for request: cpython-3.11.14")
    provisioner = setup.provisioner()
    provisioner.start()
    status = finish(provisioner)
    assert (status.state, status.error.code, status.error.action) == ("failed", "python_failed", "retry")
    assert "3.11.14" in status.error.message and "logs/provisioning.log" in status.error.message


def test_a_uv_that_cannot_start_is_named_as_the_cause(setup):
    def blocked(argv, env, cancelled, timeout=None):
        raise StartFailed("uv.exe", "%1 is not a valid Win32 application")

    provisioner = setup.provisioner(runner=blocked)
    provisioner.start()
    status = finish(provisioner)
    assert (status.state, status.error.code, status.error.action) == ("failed", "start_failed", "retry")
    assert status.error.message == (
        "Starting uv.exe of the Cinewright installation failed (%1 is not a valid Win32 application), so the engine is not installed."
    )


def test_an_unsafe_archive_is_refused(setup):
    unsafe = zip_bytes({f"WanGP-{COMMIT}/wgp.py": b"", f"WanGP-{COMMIT}/../evil.py": b"x"})
    setup.pin = make_pin(setup.registry, archive_sha256=sha256(unsafe), archive_size=len(unsafe))
    setup.server.files[ARCHIVE_URL] = unsafe
    provisioner = setup.provisioner()
    provisioner.start()
    status = finish(provisioner)
    assert (status.state, status.error.code) == ("failed", "archive_invalid")
    assert not (setup.engine_dir / "evil.py").exists() and not setup.root.exists()


def test_an_optional_wheel_with_the_wrong_hash_is_not_installed(setup):
    setup.server.files[SAGE_URL] = SAGE + b"tampered"
    setup.runner.probe = PROBE_SENTINEL + json.dumps({"ok": True, "version": "2.2.0"})
    provisioner = setup.provisioner()
    provisioner.start()
    assert finish(provisioner).state == "ready"  # optional: the engine works with sdpa
    optional = setup.runner.argv("optional")
    assert [Path(arg).name for arg in optional if arg.endswith(".whl")] == [file_name(TRITON_URL)]
    assert "probe" not in setup.runner.kinds()
    record = read_provisioning_record(str(setup.paths.root))
    assert record["optional_wheels"] == {"triton-windows": "installed", "sageattention": "hash_mismatch"}
    assert attention_mode(record) == "sdpa"  # what the worker passes as --attention


@pytest.mark.parametrize(
    ("probe", "mode"),
    [
        (PROBE_SENTINEL + json.dumps({"ok": True, "version": "2.2.0"}), "sage2"),
        (PROBE_SENTINEL + json.dumps({"ok": True, "version": "1.0.6"}), "sage"),
        (PROBE_SENTINEL + json.dumps({"ok": False, "error": "CUDA is not available"}), "sdpa"),
        ("Traceback (most recent call last):\nImportError: DLL load failed", "sdpa"),
    ],
)
def test_the_probe_decides_the_attention_mode(setup, probe, mode):
    setup.runner.probe = probe
    provisioner = setup.provisioner()
    provisioner.start()
    assert finish(provisioner).state == "ready"
    assert read_provisioning_record(str(setup.paths.root))["attention"] == mode


def test_a_failed_optional_install_keeps_sdpa_without_probing(setup):
    setup.runner.results["optional"] = RunResult(1, "error: triton-windows is not supported on this platform")
    provisioner = setup.provisioner()
    provisioner.start()
    assert finish(provisioner).state == "ready"
    assert "probe" not in setup.runner.kinds()
    record = read_provisioning_record(str(setup.paths.root))
    assert record["attention"] == "sdpa"
    assert set(record["optional_wheels"].values()) == {"install_failed"}


def test_an_optional_wheel_download_that_fails_once_is_tried_again(setup, monkeypatch):
    monkeypatch.setattr("cinewright.core.provisioning.OPTIONAL_RETRY_DELAY_S", 0.0)
    setup.server.unavailable[SAGE_URL] = 1  # a passing 503
    setup.runner.probe = PROBE_SENTINEL + json.dumps({"ok": True, "version": "2.2.0"})
    provisioner = setup.provisioner()
    provisioner.start()
    assert finish(provisioner).state == "ready"
    assert setup.server.urls().count(SAGE_URL) == 2
    record = read_provisioning_record(str(setup.paths.root))
    assert record["optional_wheels"] == {"triton-windows": "installed", "sageattention": "installed"}
    assert record["attention"] == "sage2"


def test_an_optional_wheel_that_stays_unavailable_keeps_sdpa(setup, monkeypatch):
    monkeypatch.setattr("cinewright.core.provisioning.OPTIONAL_RETRY_DELAY_S", 0.0)
    setup.server.unavailable[SAGE_URL] = OPTIONAL_ATTEMPTS
    provisioner = setup.provisioner()
    provisioner.start()
    assert finish(provisioner).state == "ready"
    assert setup.server.urls().count(SAGE_URL) == OPTIONAL_ATTEMPTS
    record = read_provisioning_record(str(setup.paths.root))
    assert record["optional_wheels"] == {"triton-windows": "installed", "sageattention": "download_failed"}
    assert "probe" not in setup.runner.kinds() and attention_mode(record) == "sdpa"


SAGE2_WORKS = PROBE_SENTINEL + json.dumps({"ok": True, "version": "2.2.0"})


def unavailable_sage(setup: Setup, monkeypatch: pytest.MonkeyPatch) -> dict:
    """A completed provisioning whose SageAttention wheel could not be downloaded in any attempt; returns its record."""
    monkeypatch.setattr("cinewright.core.provisioning.OPTIONAL_RETRY_DELAY_S", 0.0)
    setup.server.unavailable[SAGE_URL] = OPTIONAL_ATTEMPTS
    first = setup.provisioner()
    first.start()
    assert finish(first).state == "ready"
    record = read_provisioning_record(str(setup.paths.root))
    assert record["optional_wheels"] == {"triton-windows": "installed", "sageattention": "download_failed"}
    return record


def test_a_later_start_tries_an_undownloaded_optional_wheel_again_without_the_main_install(setup, monkeypatch):
    unavailable_sage(setup, monkeypatch)
    server = FakeServer()  # the network is back
    runner = FakeUv(probe=SAGE2_WORKS)
    later = setup.provisioner(server=server, runner=runner)
    assert later.status() == ProvisioningStatus("ready", version=COMMIT)
    assert not later.start()
    assert later.retry_optional()
    status = later.status()
    assert status.state == "provisioning" and status.progress.phase == STEPS[-1].phase
    assert later.worker_launch() is None  # no worker starts while the pass runs
    assert finish(later) == ProvisioningStatus("ready", version=COMMIT)

    assert server.urls() == [SAGE_URL]  # triton's verified download is reused; no archive, no ffmpeg
    assert runner.kinds() == ["optional", "probe"]  # no environment, no lock
    record = read_provisioning_record(str(setup.paths.root))
    assert record["optional_wheels"] == {"triton-windows": "installed", "sageattention": "installed"}
    assert attention_mode(record) == "sage2"
    assert not setup.provisioner(server=server, runner=runner).retry_optional()  # nothing left to try


def test_a_later_start_tries_once_and_keeps_the_engine_ready_with_its_record(setup, monkeypatch):
    before = unavailable_sage(setup, monkeypatch)
    setup.server.unavailable[SAGE_URL] = OPTIONAL_ATTEMPTS  # still unavailable
    later = setup.provisioner()
    assert later.retry_optional()
    assert finish(later) == ProvisioningStatus("ready", version=COMMIT)
    assert setup.server.urls().count(SAGE_URL) == OPTIONAL_ATTEMPTS + 1  # one attempt per start
    record = read_provisioning_record(str(setup.paths.root))
    assert {key: value for key, value in record.items() if key != "completed_at"} == {
        key: value for key, value in before.items() if key != "completed_at"
    }
    assert later.installed() and attention_mode(record) == "sdpa"
    again = setup.provisioner()
    assert again.retry_optional()  # the next start tries again
    assert finish(again).state == "ready"


def test_a_failure_in_the_later_pass_keeps_the_engine_ready_with_its_record(setup, monkeypatch):
    before = unavailable_sage(setup, monkeypatch)

    def broken(argv, env, cancelled, timeout=None):
        raise RuntimeError("the probe broke")

    later = setup.provisioner(server=FakeServer(), runner=broken)
    assert later.retry_optional()
    assert finish(later) == ProvisioningStatus("ready", version=COMMIT)
    assert read_provisioning_record(str(setup.paths.root)) == before


def test_an_optional_wheel_that_did_not_match_its_pin_is_not_tried_again(setup):
    setup.server.files[SAGE_URL] = SAGE + b"tampered"
    first = setup.provisioner()
    first.start()
    assert finish(first).state == "ready"
    later = setup.provisioner()
    assert not later.retry_optional()
    assert later.status().state == "ready"


def test_only_a_start_tries_the_optional_wheels_again_not_the_provision_route(setup, config, registry_dir, clock, monkeypatch):
    unavailable_sage(setup, monkeypatch)
    setup.runner.probe = SAGE2_WORKS
    core = Core(config, registry=Registry.load(registry_dir), detect=fake_detect, clock=clock, provisioner=setup.provisioner())
    try:
        requests = len(setup.server.requests)
        with TestClient(create_app(core), base_url=BASE_URL) as client:
            assert client.post(f"{API}/engines/wan2gp/provision").json()["state"] == "ready"  # nothing to do
            assert core.provisioner.status().state == "ready" and len(setup.server.requests) == requests
        core.start_provisioning()  # what the launcher's lifespan does at start
        assert finish(core.provisioner).state == "ready"
        assert len(setup.server.requests) == requests + 1
        assert attention_mode(read_provisioning_record(str(setup.paths.root))) == "sage2"
    finally:
        core.close()


def test_a_probe_that_cannot_start_keeps_sdpa(setup):
    def runner(argv, env, cancelled, timeout=None):
        if FakeUv.kind(list(argv)) == "probe":
            raise StartFailed("python.exe", "Access is denied")
        return setup.runner(argv, env, cancelled, timeout)

    provisioner = setup.provisioner(runner=runner)
    provisioner.start()
    assert finish(provisioner).state == "ready"  # optional: the engine works with sdpa
    record = read_provisioning_record(str(setup.paths.root))
    assert record["attention"] == "sdpa"
    assert record["optional_wheels"] == {"triton-windows": "installed", "sageattention": "installed"}


# ---------------------------------------------------------------------- interruptions


def test_an_interrupted_download_resumes_on_retry(setup):
    setup.server.cut[ARCHIVE_URL] = 1000
    provisioner = setup.provisioner()
    provisioner.start()
    status = finish(provisioner)
    assert (status.state, status.error.code, status.error.action) == ("failed", "download_failed", "retry")
    assert (setup.engine_dir / "downloads" / f"wan2gp-{COMMIT}.zip.part").stat().st_size == 1000

    assert provisioner.start()  # the Setup view's retry
    assert finish(provisioner).state == "ready"
    archive_requests = [ranged for url, ranged in setup.server.requests if url == ARCHIVE_URL]
    assert archive_requests == [None, "bytes=1000-"]


def test_a_server_without_range_support_restarts_the_download(setup):
    (setup.engine_dir / "downloads").mkdir(parents=True)
    (setup.engine_dir / "downloads" / f"wan2gp-{COMMIT}.zip.part").write_bytes(b"stale bytes")
    original = setup.server.handler

    def no_ranges(request):
        request.headers.pop("range", None)
        return original(request)

    setup.server.handler = no_ranges
    provisioner = setup.provisioner()
    provisioner.start()
    assert finish(provisioner).state == "ready"


def test_a_stopped_provisioning_restarts_cleanly_from_verified_downloads(setup):
    setup.runner.block = threading.Event()
    first = setup.provisioner()
    first.start()
    assert setup.runner.block.wait(10)  # inside the lock install
    first.stop()
    assert first.status().state == "not_installed"
    assert not (setup.engine_dir / "provisioning.json").exists()
    (setup.root / "leftover.txt").write_text("from the interrupted run", encoding="utf-8")
    (setup.engine_dir / f"{COMMIT}.partial").mkdir()

    setup.runner.block = None
    downloads = len(setup.server.requests)
    second = setup.provisioner()
    assert second.status().state == "not_installed"
    second.start()
    assert finish(second).state == "ready"
    new = setup.server.urls()[downloads:]
    assert ARCHIVE_URL not in new and FFMPEG_URL not in new  # verified downloads are reused
    assert not (setup.root / "leftover.txt").exists()  # the tree is unpacked again
    assert not (setup.engine_dir / f"{COMMIT}.partial").exists()
    assert setup.runner.kinds().count("venv") == 2  # the environment is created again (uv venv --clear)


def test_a_new_run_drops_the_old_record_before_it_changes_anything(setup):
    first = setup.provisioner()
    first.start()
    assert finish(first).state == "ready"
    (setup.root / "ffmpeg_bins" / "ffprobe.exe").unlink()  # the next start provisions again

    setup.runner.block = threading.Event()
    second = setup.provisioner()
    assert second.status().state == "not_installed"
    second.start()
    assert setup.runner.block.wait(10)  # inside the lock install: root, ffmpeg and python.exe are back
    second.stop()
    assert second.status().state == "not_installed"
    # The old record must not vouch for the half-installed environment.
    assert not (setup.engine_dir / "provisioning.json").exists()
    assert setup.provisioner().status().state == "not_installed"


# ---------------------------------------------------------------------- where provisioning cannot run


def test_other_platforms_stay_not_installed(setup):
    provisioner = setup.provisioner(platform="linux")
    status = provisioner.status()
    assert (status.state, status.error.code, status.error.action) == ("not_installed", "unsupported_platform", None)
    assert not provisioner.start()
    assert setup.server.requests == [] and setup.events == []


def test_a_missing_uv_stays_not_installed(setup, tmp_path):
    provisioner = setup.provisioner(uv=tmp_path / "missing" / "uv.exe")
    assert (provisioner.status().state, provisioner.status().error.code) == ("not_installed", "uv_missing")
    assert "only the installed app" in provisioner.status().error.message  # also true in a development checkout
    assert not provisioner.start()
    assert setup.server.requests == []


def test_bundled_uv_lies_next_to_the_runtime(tmp_path):
    runtime = tmp_path / "Cinewright" / "runtime"
    runtime.mkdir(parents=True)
    assert bundled_uv(str(runtime), "win32") is None
    (tmp_path / "Cinewright" / "uv.exe").write_bytes(b"")
    assert bundled_uv(str(runtime), "win32") == tmp_path / "Cinewright" / "uv.exe"


def test_uv_environment_drops_user_settings(tmp_path):
    layout = EngineLayout(tmp_path / "engines", COMMIT)
    env = uv_environment({"Path": "x", "UV_EXTRA_INDEX_URL": "y", "uv_index": "z", "PIP_INDEX_URL": "w", "PYTHONPATH": "v"}, layout)
    assert env == {
        "Path": "x",
        "UV_NO_CONFIG": "1",
        "UV_NO_PROGRESS": "1",
        "UV_CACHE_DIR": str(tmp_path / "engines" / "uv-cache"),
        "UV_PYTHON_INSTALL_DIR": str(tmp_path / "engines" / "python"),
    }


# ---------------------------------------------------------------------- the bundled lock and the probe


def test_the_bundled_lock_builds_its_sdists_with_its_own_setuptools():
    packages = lock_packages(REGISTRY_DIR / RAW["lock"])
    # The list the registry README names under "Sdist-only lock entries".
    assert sorted(sdist_only(packages)) == ["antlr4-python3-runtime", "docopt", "iopath", "moviepy", "openai-whisper", "unidic-lite"]
    setuptools = next(entry for entry in packages if entry["name"] == "setuptools")
    wheel = setuptools["wheels"][0]
    assert build_requirements(packages) == f"setuptools @ {wheel['url']} --hash=sha256:{wheel['hashes']['sha256']}\n"


def test_parse_probe_takes_the_last_sentinel_line():
    assert parse_probe("noise\n" + PROBE_SENTINEL + '{"ok": true, "version": "2.2.0"}\n') == "sage2"
    assert parse_probe(PROBE_SENTINEL + "not json") == "sdpa"
    assert parse_probe("") == "sdpa"


# ---------------------------------------------------------------------- the downloader


def test_downloader_resumes_and_checks_the_hash(tmp_path):
    body = bytes(range(256)) * 40
    server = FakeServer()
    server.files = {"https://example.test/file.bin": body}
    server.cut["https://example.test/file.bin"] = 3000
    downloader = Downloader(httpx.MockTransport(server.handler))
    target = tmp_path / "file.bin"
    seen = []
    with pytest.raises(DownloadError):
        downloader.fetch("https://example.test/file.bin", target, sha256=sha256(body), size=len(body), progress=seen.append)
    assert downloader.fetch("https://example.test/file.bin", target, sha256=sha256(body), size=len(body), progress=seen.append) == target
    assert target.read_bytes() == body and seen[-1] == len(body)
    assert [ranged for _, ranged in server.requests] == [None, "bytes=3000-"]

    # A present, matching file is not downloaded again; a mismatching one is deleted.
    assert downloader.fetch("https://example.test/file.bin", target, sha256=sha256(body), size=len(body)) == target
    assert len(server.requests) == 2
    with pytest.raises(HashMismatch):
        downloader.fetch("https://example.test/file.bin", tmp_path / "other.bin", sha256="0" * 64, size=len(body))
    assert list(tmp_path.glob("other.bin*")) == []


def test_downloader_stops_at_max_bytes_without_a_pinned_size(tmp_path):
    body = b"x" * (3 * 1024 * 1024)
    server = FakeServer()
    server.files = {"https://example.test/wheel.whl": body}
    downloader = Downloader(httpx.MockTransport(server.handler))
    seen = []
    with pytest.raises(HashMismatch):
        downloader.fetch(
            "https://example.test/wheel.whl", tmp_path / "wheel.whl", sha256=sha256(body), max_bytes=1024, progress=seen.append
        )
    assert list(tmp_path.glob("wheel.whl*")) == []  # nothing beyond the bound was kept
    assert all(done <= 1024 for done in seen)
    # Below the bound the download completes as usual.
    target = downloader.fetch("https://example.test/wheel.whl", tmp_path / "wheel.whl", sha256=sha256(body), max_bytes=len(body))
    assert target.read_bytes() == body


# ---------------------------------------------------------------------- the process runner (real child processes)


def test_process_runner_captures_output_and_exit_code(tmp_path):
    log_path = tmp_path / "logs" / "provisioning.log"
    result = ProcessRunner(log_path)([sys.executable, "-c", "print('hello from uv'); raise SystemExit(3)"], dict(os.environ), lambda: False)
    assert result.returncode == 3 and "hello from uv" in result.output
    log_text = log_path.read_text(encoding="utf-8")
    assert log_text.startswith("$ ") and "hello from uv" in log_text


def test_process_runner_kills_a_cancelled_child(tmp_path):
    started = time.monotonic()
    with pytest.raises(Cancelled):
        ProcessRunner(tmp_path / "run.log")(
            [sys.executable, "-c", "import time; time.sleep(60)"], dict(os.environ), lambda: time.monotonic() - started > 0.5
        )
    assert time.monotonic() - started < 20


def test_process_runner_stops_a_child_at_its_timeout(tmp_path):
    started = time.monotonic()
    result = ProcessRunner(tmp_path / "run.log")(
        [sys.executable, "-c", "import time; time.sleep(60)"], dict(os.environ), lambda: False, 0.5
    )
    assert result.returncode != 0 and "Stopped after" in result.output
    assert time.monotonic() - started < 20


def test_process_runner_names_a_program_that_cannot_start(tmp_path):
    log_path = tmp_path / "run.log"
    with pytest.raises(StartFailed) as raised:
        ProcessRunner(log_path)([str(tmp_path / "missing" / "uv.exe"), "venv"], dict(os.environ), lambda: False)
    assert raised.value.program == "uv.exe" and raised.value.reason
    assert "Could not start" in log_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------- the core, the API and the worker


@pytest.fixture
def provisioned_core(setup, config, registry_dir, clock):
    """A core whose provisioner downloads from the fake server and runs the fake uv (same data directory)."""
    instance = Core(config, registry=Registry.load(registry_dir), detect=fake_detect, clock=clock, provisioner=setup.provisioner())
    mark_installed(instance, *TEST_INSTALLED)
    yield instance
    instance.close()


def test_system_reports_provisioning_and_then_the_pinned_commit(provisioned_core):
    with TestClient(create_app(provisioned_core), base_url=BASE_URL) as client:
        engine = client.get(f"{API}/system").json()["engines"][0]
        assert (engine["id"], engine["state"], engine["version"], engine["error"]) == ("wan2gp", "not_installed", None, None)

        started = client.post(f"{API}/engines/wan2gp/provision")
        assert started.status_code == 202
        assert started.json()["state"] in ("provisioning", "ready")
        finish(provisioned_core.provisioner)

        engine = client.get(f"{API}/system").json()["engines"][0]
        assert (engine["state"], engine["version"], engine["progress"], engine["error"]) == ("ready", COMMIT, None, None)
        assert client.post(f"{API}/engines/wan2gp/provision").json()["state"] == "ready"  # nothing to do
        assert client.post(f"{API}/engines/comfyui/provision").status_code == 404


def test_failures_carry_cause_consequence_and_retry(setup, provisioned_core):
    setup.server.files[ARCHIVE_URL] = b"not the pinned archive"
    provisioned_core.start_provisioning()
    finish(provisioned_core.provisioner)
    engine = provisioned_core.system_info().engines[0]
    assert (engine.state, engine.version, engine.error.code, engine.error.action) == ("failed", None, "hash_mismatch", "retry")
    assert ", so " in engine.error.message and engine.error.message.endswith(".")


def test_engine_events_reach_the_unfiltered_stream_only(provisioned_core):
    core = provisioned_core
    registration = WorkerRegistration(host_id="local", engine="wan2gp", models=["test_image"], health={"state": "ready"})
    core.register_worker(registration, core.enrollment_token)
    job_id = core.submit_job(JobRequest.model_validate({"capability": "t2i", "prompt": "a lighthouse"}), ClientKind.UI, None).id

    async def scenario():
        everything: list = []
        of_job: list = []

        async def follow(target, job):
            async for event in core.events(job, None):
                target.append(event)
                if job is None and isinstance(event, SystemEvent) and event.engine.state == "ready":
                    return
                if job is not None and isinstance(event, JobEvent) and event.data.get("state") == "cancelled":
                    return

        tasks = [asyncio.create_task(follow(everything, None)), asyncio.create_task(follow(of_job, job_id))]
        await asyncio.sleep(0.05)
        await asyncio.to_thread(core.start_provisioning)
        await asyncio.to_thread(finish, core.provisioner)
        await asyncio.to_thread(core.cancel_job, job_id)
        await asyncio.wait_for(asyncio.gather(*tasks), 5)
        return everything, of_job

    everything, of_job = asyncio.run(scenario())
    engine_events = [event for event in everything if isinstance(event, SystemEvent)]
    assert all(event.kind == "engine" and event.engine.id == "wan2gp" for event in engine_events)
    assert engine_events[0].engine.state == "provisioning" and engine_events[0].engine.progress is not None
    assert engine_events[-1].engine.state == "ready" and engine_events[-1].engine.version == COMMIT
    assert of_job and all(isinstance(event, JobEvent) for event in of_job)


def test_sse_frames_give_only_stored_job_events_an_id():
    now = datetime(2026, 11, 24, tzinfo=UTC)
    job = sse_frame(JobEvent(job_id="j1", seq=3, kind="state", data={"state": "queued"}, created_at=now)).decode()
    assert job.startswith("id: j1:3\nevent: state\ndata: {") and job.endswith("}\n\n")
    preview = sse_frame(JobEvent(job_id="j1", seq=3, kind="preview", data={"media_type": "image/png"}, created_at=now)).decode()
    assert preview.startswith("event: preview\ndata: {")  # live only, so Last-Event-ID keeps naming a stored event
    engine = EngineInfo(id="wan2gp", name="WanGP (Wan2GP)", version=None, state="provisioning")
    system = sse_frame(SystemEvent(kind="engine", engine=engine, created_at=now)).decode()
    assert system.startswith("event: engine\ndata: {") and "id:" not in system.split("data: ", 1)[0]
    assert SystemEvent.model_validate_json(system.split("data: ", 1)[1]).engine == engine


def test_the_lifespan_provisions_only_when_asked(setup, provisioned_core):
    with TestClient(create_app(provisioned_core), base_url=BASE_URL):
        assert provisioned_core.provisioner.status().state == "not_installed"
    assert setup.server.requests == []

    setup.runner.block = threading.Event()
    with TestClient(create_app(provisioned_core, provision=True), base_url=BASE_URL):
        assert setup.runner.block.wait(10)  # provisioning runs in the background ...
    assert provisioned_core.provisioner.status().state == "not_installed"  # ... and stops with the app


def test_the_worker_reports_the_provisioned_commit(setup, monkeypatch):
    provisioner = setup.provisioner()
    provisioner.start()
    assert finish(provisioner).state == "ready"

    calls = []
    api = types.ModuleType("shared.api")
    api.init = lambda **kwargs: calls.append(kwargs) or SimpleNamespace(list_model_availability=lambda: [], list_model_defs=lambda: [])
    monkeypatch.setitem(sys.modules, "shared", types.ModuleType("shared"))
    monkeypatch.setitem(sys.modules, "shared.api", api)
    monkeypatch.setitem(sys.modules, "wgp", SimpleNamespace(WanGP_version="10.0"))
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    # A Windows installation, also when this test runs elsewhere: the adapter looks for the .exe names there.
    monkeypatch.setattr("cinewright_worker.engines.wan2gp.FFMPEG_BINARIES", tuple(sorted(FFMPEG_FILES)))

    record = read_provisioning_record(str(setup.paths.root))
    adapter = Wan2GPAdapter(root=record["root"], data_dir=str(setup.paths.root))
    adapter.capabilities()
    assert adapter.health().engine_version == COMMIT  # registration and job provenance carry it
    assert calls[0]["cli_args"] == ["--attention", "sdpa", "--offline"]

"""Engine provisioning: install the pinned Wan2GP engine into its own environment on first start.

Every download comes from `EnginePin` (`registry/engines.yaml`) and is checked before use:

1. The source archive of the pinned commit, by its SHA-256 and size before it is opened (the commit id git writes
   into an archive is not authenticated). It is unpacked without GitHub's top folder into
   `<data dir>/engines/wan2gp/<commit>/`, so `wgp.py` lies directly in it: the Wan2GP root.
2. ffmpeg: the pinned archive by its SHA-256, then each pinned file by its own into `<root>/ffmpeg_bins`, where
   Wan2GP's `init()` finds them instead of downloading unpinned ones.
3. The environment `<data dir>/engines/wan2gp/env`, created by the bundled uv with its managed Python. Then the
   lock: first the lock's build backend (setuptools) from a generated requirements file with `--require-hashes`,
   then the whole lock with `--require-hashes`, `--no-index` and without build isolation, so its sdist-only entries
   build with that setuptools and nothing is resolved from an index. uv reads no config file and none of the
   user's `UV_*` or `PIP_*` variables, so no extra index enters.
4. The optional attention wheels: downloaded and checked by the core (a failed download is tried again within the
   run), installed with `--no-deps` only when they match, then probed in the environment. Any failure there leaves
   the engine ready with `--attention sdpa`.
5. The record (`PROVISIONING_RECORD`), written last and atomically; the worker reads the commit and the attention
   mode from it, and the core starts the local worker with the interpreter and root it names (`worker_launch`). A
   later start that finds a record of the same pin and its files is ready without any download. Only when an
   optional wheel could not be downloaded does that start first try step 4 again (`retry_optional`), once per wheel,
   before any worker runs; the record stays until that pass rewrites it.

Downloads resume from their `.part` file; the unpacked tree, ffmpeg and the environment are redone from the
verified downloads, so an interrupted run resumes or restarts cleanly. uv and the probe run through
`spawn_bound`: no console window, and they end with the core.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import zipfile
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Any, Literal, Protocol
from urllib.parse import unquote, urlsplit

import httpx

from cinewright.api.schemas import Progress, SetupError
from cinewright.core.clock import Clock, to_iso, utcnow
from cinewright.core.engines import REGISTRY_DIR, EnginePin, FfmpegBuild
from cinewright.core.supervisor import WorkerLaunch, spawn_bound
from cinewright.paths import DataPaths
from cinewright_worker.engines.wan2gp import DEFAULT_ATTENTION, FFMPEG_DIR, PROVISIONING_RECORD

log = logging.getLogger(__name__)

ENGINE_ID = "wan2gp"
ROOT_MARKER = "wgp.py"  # lies directly in the Wan2GP root
LOG_NAME = "provisioning.log"  # in <data dir>/logs: the output of uv and of the probe
CHUNK = 1024 * 1024
DOWNLOAD_TIMEOUT_S = 60.0
PROGRESS_INTERVAL_S = 0.5  # at most two progress events per second within a step
PROBE_TIMEOUT_S = 600.0  # Triton compiles its kernels on the first call
STOP_TIMEOUT_S = 30.0
OPTIONAL_ATTEMPTS = 3  # downloads of an optional wheel per run: a network blip or a 503 must not cost SageAttention for good
OPTIONAL_RETRY_DELAY_S = 5.0
OPTIONAL_WHEEL_MAX_BYTES = 1 << 30  # their pins carry no size; this bounds what a replaced asset can write before the hash check
UV_MISSING = (
    "No uv.exe lies in Cinewright's installation folder, so WanGP cannot be set up here (only the installed app brings it); "
    "reinstall Cinewright if this is the installed app."
)
# Build backend of the lock's sdist-only entries, installed from the lock before them.
BUILD_REQUIREMENTS = ("setuptools",)
PROBE_SENTINEL = "CINEWRIGHT-ATTENTION-PROBE:"
# Runs in the engine environment: SageAttention works when one attention call on the GPU returns finite values.
PROBE_SCRIPT = f"""
import json
result = {{"ok": False}}
try:
    from importlib.metadata import version
    import torch, triton, sageattention
    result["version"] = version("sageattention")
    if torch.cuda.is_available():
        q = torch.randn(1, 8, 256, 64, dtype=torch.float16, device="cuda")
        out = sageattention.sageattn(q, q, q, tensor_layout="HND", is_causal=False)
        torch.cuda.synchronize()
        result["ok"] = bool(torch.isfinite(out).all().item())
    else:
        result["error"] = "CUDA is not available"
except Exception as exc:
    result["error"] = repr(exc)[:300]
print({PROBE_SENTINEL!r} + json.dumps(result), flush=True)
"""

EngineState = Literal["not_installed", "provisioning", "ready", "failed"]


@dataclass(frozen=True)
class Step:
    phase: str
    weight: float  # share of the whole run in the overall percentage


STEPS = (
    Step("Downloading WanGP", 10),
    Step("Unpacking WanGP", 3),
    Step("Downloading ffmpeg", 12),
    Step("Setting up Python", 5),
    Step("Installing packages", 65),
    Step("Testing SageAttention", 5),
)
ARCHIVE, UNPACK, FFMPEG, PYTHON, PACKAGES, ATTENTION = range(len(STEPS))


class ProvisioningError(Exception):
    """A failure the Setup view shows: cause and consequence in one sentence, plus retry where it can help."""

    def __init__(self, code: str, message: str, *, retry: bool = True) -> None:
        super().__init__(message)
        self.error = SetupError(code=code, message=message, action="retry" if retry else None)


class Cancelled(Exception):
    """Provisioning was stopped (the core shuts down); the next start resumes it."""


class DownloadError(Exception):
    """The download did not complete; a `.part` file, if any, is kept for resuming."""


class HashMismatch(Exception):
    """The downloaded bytes do not match the pin; they were deleted."""


class StartFailed(Exception):
    """A program could not be started at all (missing, damaged or blocked); `program` is its file name."""

    def __init__(self, program: str, reason: str) -> None:
        super().__init__(f"{program}: {reason}")
        self.program = program
        self.reason = reason


@dataclass(frozen=True)
class ProvisioningStatus:
    state: EngineState
    version: str | None = None
    progress: Progress | None = None
    error: SetupError | None = None


@dataclass(frozen=True)
class RunResult:
    returncode: int
    output: str  # the last lines of the combined output


class Runner(Protocol):
    def __call__(
        self, argv: Sequence[str], env: Mapping[str, str], cancelled: Callable[[], bool], timeout: float | None = None
    ) -> RunResult: ...


@dataclass(frozen=True)
class EngineLayout:
    """Where provisioning puts the engine, below `<data dir>/engines`."""

    engines_dir: Path
    commit: str
    windows: bool = True

    @property
    def engine_dir(self) -> Path:
        return self.engines_dir / ENGINE_ID

    @property
    def root(self) -> Path:
        """The Wan2GP root, holding `wgp.py`."""
        return self.engine_dir / self.commit

    @property
    def staging(self) -> Path:
        return self.engine_dir / f"{self.commit}.partial"

    @property
    def ffmpeg_dir(self) -> Path:
        return self.root / FFMPEG_DIR

    @property
    def env(self) -> Path:
        return self.engine_dir / "env"

    @property
    def python(self) -> Path:
        return self.env / "Scripts" / "python.exe" if self.windows else self.env / "bin" / "python"

    @property
    def downloads(self) -> Path:
        return self.engine_dir / "downloads"

    @property
    def record(self) -> Path:
        return self.engine_dir / PROVISIONING_RECORD

    @property
    def python_installs(self) -> Path:
        """uv's managed Python installations (UV_PYTHON_INSTALL_DIR)."""
        return self.engines_dir / "python"

    @property
    def uv_cache(self) -> Path:
        return self.engines_dir / "uv-cache"


def bundled_uv(prefix: str = sys.prefix, platform: str = sys.platform) -> Path | None:
    """The installer's `uv.exe`: it lies in the installation folder, next to the runtime the core runs from."""
    path = Path(prefix).parent / ("uv.exe" if platform == "win32" else "uv")
    return path if path.is_file() else None


def uv_environment(base: Mapping[str, str], layout: EngineLayout) -> dict[str, str]:
    """`base` without any uv or pip setting, plus uv's directories below the data directory and no config files."""
    dropped = ("VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONHOME", "PYTHONPATH")
    env = {key: value for key, value in base.items() if not key.upper().startswith(("UV_", "PIP_")) and key.upper() not in dropped}
    env.update(
        UV_NO_CONFIG="1",
        UV_NO_PROGRESS="1",
        UV_CACHE_DIR=str(layout.uv_cache),
        UV_PYTHON_INSTALL_DIR=str(layout.python_installs),
    )
    return env


def file_name(url: str) -> str:
    return unquote(PurePosixPath(urlsplit(url).path).name)


def _file_sha256(path: Path, digest: Any | None = None) -> Any:
    digest = digest if digest is not None else hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest


def _short(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------- downloads


class Downloader:
    """HTTPS downloads into a file, checked against a pinned SHA-256 (and size, when pinned).

    A `.part` file left by an interrupted download is resumed with a Range request; a server that answers
    with the whole file restarts it. Bytes that do not match the pin are deleted, never kept.
    """

    def __init__(self, transport: httpx.BaseTransport | None = None, timeout: float = DOWNLOAD_TIMEOUT_S) -> None:
        self._transport = transport
        self._timeout = timeout

    def fetch(
        self,
        url: str,
        target: Path,
        *,
        sha256: str,
        size: int | None = None,
        max_bytes: int | None = None,
        progress: Callable[[int], None] | None = None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> Path:
        """Download `url` to `target`. `size` is the exact pinned size; `max_bytes` only bounds a download without one."""
        report = progress or (lambda done: None)
        limit = size if size is not None else max_bytes
        if target.is_file():
            if (size is None or target.stat().st_size == size) and _file_sha256(target).hexdigest() == sha256:
                report(target.stat().st_size)
                return target
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        offset = part.stat().st_size if part.is_file() else 0
        if size is not None and offset >= size:  # cannot be resumed; start over
            part.unlink()
            offset = 0
        digest = hashlib.sha256()
        done = 0
        try:
            with (
                httpx.Client(transport=self._transport, follow_redirects=True, timeout=self._timeout) as client,
                client.stream("GET", url, headers={"Range": f"bytes={offset}-"} if offset else {}) as response,
            ):
                if offset and response.status_code == 206 and _range_start(response) == offset:
                    _file_sha256(part, digest)
                    mode, done = "ab", offset
                elif response.status_code == 200:
                    mode, done = "wb", 0
                else:
                    if offset:  # the server cannot continue this part; the next attempt starts over
                        part.unlink(missing_ok=True)
                    raise DownloadError(f"HTTP {response.status_code}")
                with part.open(mode) as out:
                    for chunk in response.iter_bytes():  # as received: an interruption keeps every byte
                        if cancelled():
                            raise Cancelled
                        out.write(chunk)
                        digest.update(chunk)
                        done += len(chunk)
                        if limit is not None and done > limit:
                            break
                        report(done)
        except httpx.HTTPError as exc:
            raise DownloadError(_short(str(exc)) or type(exc).__name__) from exc
        # A dropped connection raised above (httpx checks Content-Length), so a complete answer of another size is wrong.
        if (size is not None and done != size) or (limit is not None and done > limit) or digest.hexdigest() != sha256:
            part.unlink(missing_ok=True)
            raise HashMismatch(url)
        os.replace(part, target)
        return target


def _range_start(response: httpx.Response) -> int | None:
    match = re.match(r"bytes (\d+)-", response.headers.get("content-range", ""))
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------------- archives


def _safe_parts(name: str) -> tuple[str, ...]:
    parts = PurePosixPath(name).parts
    if not parts or "\\" in name or ":" in name or name.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise ProvisioningError(
            "archive_invalid", f"The WanGP source archive contains the unsafe path {name!r}, so it was not unpacked.", retry=False
        )
    return parts


def unpack_source(archive: Path, target: Path, cancelled: Callable[[], bool] = lambda: False) -> None:
    """Unpack a GitHub source archive into `target` without its top folder; `wgp.py` must then lie in `target`.

    The archive has passed its pin, so a failure here repeats on every retry and offers none.
    """
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as zipped:
            members = zipped.infolist()
            tops = {member.filename.split("/", 1)[0] for member in members}
            if len(tops) != 1:
                raise ProvisioningError(
                    "archive_invalid", "The WanGP source archive does not have one top folder, so it was not unpacked.", retry=False
                )
            prefix = tops.pop() + "/"
            for member in members:
                if cancelled():
                    raise Cancelled
                if not member.filename.startswith(prefix):
                    raise ProvisioningError(
                        "archive_invalid", "The WanGP source archive does not have one top folder, so it was not unpacked.", retry=False
                    )
                relative = member.filename[len(prefix) :].rstrip("/")
                if not relative:
                    continue
                destination = target.joinpath(*_safe_parts(relative))
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, destination.open("wb") as out:
                    shutil.copyfileobj(source, out, CHUNK)
    except zipfile.BadZipFile as exc:
        raise ProvisioningError(
            "archive_invalid", f"The WanGP source archive is not a valid zip file ({exc}), so it was not unpacked.", retry=False
        ) from exc
    if not (target / ROOT_MARKER).is_file():
        raise ProvisioningError(
            "archive_invalid",
            f"The WanGP source archive has no {ROOT_MARKER} in its top folder, so the engine is not installed.",
            retry=False,
        )


def place_ffmpeg(archive: Path, build: FfmpegBuild, bin_dir: Path) -> None:
    """Extract the pinned files of the ffmpeg archive into `bin_dir`; each must match its pinned SHA-256 and size.

    Every file is checked before any of them replaces what `bin_dir` holds. The archive has passed its own pin, so its
    contents are fixed: a failure here repeats on every retry and offers none.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    try:
        try:
            with zipfile.ZipFile(archive) as zipped:
                for name, pinned in sorted(build.files.items()):
                    try:
                        member = zipped.getinfo(f"{build.bin_dir}/{name}")
                    except KeyError:
                        raise ProvisioningError(
                            "archive_invalid", f"The ffmpeg archive lacks {name}, so WanGP's ffmpeg is not installed.", retry=False
                        ) from None
                    part = bin_dir / f"{name}.part"
                    staged.append((part, bin_dir / name))
                    digest = hashlib.sha256()
                    written = 0
                    with zipped.open(member) as source, part.open("wb") as out:
                        for chunk in iter(lambda source=source: source.read(CHUNK), b""):
                            out.write(chunk)
                            digest.update(chunk)
                            written += len(chunk)
                    if written != pinned.size or digest.hexdigest() != pinned.sha256:
                        raise ProvisioningError(
                            "hash_mismatch",
                            f"{name} in the ffmpeg archive did not match its pinned SHA-256, so it was not installed and the engine "
                            "is not installed.",
                            retry=False,
                        )
        except zipfile.BadZipFile as exc:
            raise ProvisioningError(
                "archive_invalid", f"The ffmpeg archive is not a valid zip file ({exc}), so it was not unpacked.", retry=False
            ) from exc
        for part, final in staged:
            os.replace(part, final)
    finally:
        for part, _ in staged:
            part.unlink(missing_ok=True)


# --------------------------------------------------------------------------- the lock


def lock_packages(lock: Path) -> list[dict[str, Any]]:
    """The `[[packages]]` entries of a PEP 751 lock."""
    packages = tomllib.loads(lock.read_text(encoding="utf-8")).get("packages")
    return [entry for entry in packages if isinstance(entry, dict)] if isinstance(packages, list) else []


def sdist_only(packages: list[dict[str, Any]]) -> list[str]:
    """Names of the entries that have only an sdist: the installer has to build them."""
    return [entry["name"] for entry in packages if entry.get("sdist") and not entry.get("wheels") and not entry.get("archive")]


def build_requirements(packages: list[dict[str, Any]]) -> str:
    """A hash-pinned requirements file with the lock's wheel of each build backend in `BUILD_REQUIREMENTS`."""
    lines = []
    for name in BUILD_REQUIREMENTS:
        entry = next((entry for entry in packages if re.sub(r"[-_.]+", "-", str(entry.get("name", ""))).lower() == name), None)
        wheels = [
            wheel for wheel in (entry or {}).get("wheels") or [] if file_name(str(wheel.get("url", ""))).endswith("-py3-none-any.whl")
        ]
        sha256 = (wheels[0].get("hashes") or {}).get("sha256") if wheels else None
        if not sha256:
            raise ProvisioningError(
                "lock_invalid",
                f"The engine lock has no universal {name} wheel with a hash, which its sdist-only packages need to build, "
                "so the engine is not installed.",
                retry=False,
            )
        lines.append(f"{name} @ {wheels[0]['url']} --hash=sha256:{sha256}")
    return "\n".join(lines) + "\n"


def parse_probe(output: str) -> str:
    """The attention mode the probe found working: `sage2` for SageAttention 2.x, `sage` for 1.x, else `sdpa`."""
    for line in reversed(output.splitlines()):
        if not line.startswith(PROBE_SENTINEL):
            continue
        try:
            result = json.loads(line[len(PROBE_SENTINEL) :])
        except ValueError:
            break
        if isinstance(result, dict) and result.get("ok") is True:
            major = str(result.get("version") or "").split(".", 1)[0]
            return "sage2" if major.isdigit() and int(major) >= 2 else "sage"
        break
    return DEFAULT_ATTENTION


# --------------------------------------------------------------------------- processes


class ProcessRunner:
    """Runs a command without a console window and bound to the core's lifetime; its output goes to `log_path`.

    A program that cannot start at all raises `StartFailed`, so the message names it instead of a file write.
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path

    def __call__(
        self, argv: Sequence[str], env: Mapping[str, str], cancelled: Callable[[], bool], timeout: float | None = None
    ) -> RunResult:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        tail: deque[str] = deque(maxlen=60)
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"$ {subprocess.list2cmdline(list(argv))}\n")
            log_file.flush()
            try:
                bound = spawn_bound(argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            except OSError as exc:
                log_file.write(f"Could not start: {exc}\n")
                raise StartFailed(Path(argv[0]).name, _short(exc.strerror or str(exc), 80)) from exc
            assert bound.popen.stdout is not None
            reader = threading.Thread(target=_pump, args=(bound.popen.stdout, log_file, tail), daemon=True, name="provisioning-output")
            reader.start()
            try:
                while True:
                    try:
                        bound.popen.wait(timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    if cancelled() or (deadline is not None and time.monotonic() > deadline):
                        bound.kill()
                        bound.popen.wait(timeout=10)
                        if cancelled():
                            raise Cancelled
                        tail.append(f"Stopped after {timeout:.0f} s.\n")
                        break
            finally:
                reader.join(timeout=10)
                bound.release()
        return RunResult(bound.popen.returncode, "".join(tail))


def _pump(stream: IO[bytes], log_file: IO[str], tail: deque[str]) -> None:
    for raw in iter(stream.readline, b""):
        text = raw.decode("utf-8", errors="replace")
        log_file.write(text)
        tail.append(text)
    log_file.flush()


# --------------------------------------------------------------------------- the provisioner


class EngineProvisioner:
    """The state machine `not_installed` -> `provisioning` -> `ready` / `failed` of one engine (Wan2GP).

    `start()` runs provisioning in a background thread; `on_change` receives every new state and progress step.
    """

    def __init__(
        self,
        paths: DataPaths,
        pin: EnginePin,
        *,
        uv: Path | None,
        lock: Path | None = None,
        downloader: Downloader | None = None,
        runner: Runner | None = None,
        platform: str = sys.platform,
        environ: Mapping[str, str] | None = None,
        clock: Clock = utcnow,
    ) -> None:
        self.pin = pin
        self.paths = paths
        self.uv = uv
        self.platform = platform
        # Absolute paths: the record hands them to the worker, which compares them with its own.
        self.layout = EngineLayout(paths.engines_dir.resolve(), pin.commit, windows=platform == "win32")
        self.lock = lock if lock is not None else REGISTRY_DIR / pin.lock
        self.on_change: Callable[[ProvisioningStatus], None] | None = None
        self._downloader = downloader or Downloader()
        self._runner = runner or ProcessRunner(paths.logs_dir / LOG_NAME)
        self._environ = os.environ if environ is None else environ
        self._clock = clock
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._last_emit = 0.0
        self._fingerprint: str | None = None
        self._status = ProvisioningStatus("ready", version=pin.commit) if self.installed() else self._idle()

    # ------------------------------------------------------------------ public

    def status(self) -> ProvisioningStatus:
        with self._lock:
            return self._status

    def start(self) -> bool:
        """Start provisioning unless the engine is ready, provisioning runs, or it cannot run here. True if started."""
        with self._lock:
            if self._running or self._status.state == "ready" or self.unavailable() is not None:
                return False
            thread = self._begin(self._run, ARCHIVE)
        self._emit()
        thread.start()
        return True

    def retry_optional(self) -> bool:
        """On a ready engine, try again the optional wheels an earlier run could not download. True if started.

        Only at start, before any worker runs on the environment: the pass installs into it, and while it runs the
        state is `provisioning`. The main install and the record stay as they are until the pass rewrites the record.
        A wheel that did not match its pin or did not install is not tried again.
        """
        with self._lock:
            if self._running or self._status.state != "ready" or self.unavailable() is not None or not self._optional_retry_due():
                return False
            thread = self._begin(self._run_optional, ATTENTION)
        self._emit()
        thread.start()
        return True

    def worker_launch(self) -> WorkerLaunch | None:
        """How the local worker starts on the provisioned engine: the interpreter and Wan2GP root the record names.

        None unless the engine is ready. The worker reads the attention mode and the commit from the same record.
        """
        record = self._read_record() if self.status().state == "ready" and self.installed() else None
        if record is None:
            return None
        return WorkerLaunch(python=Path(record["python"]), engine=ENGINE_ID, wan2gp_root=Path(record["root"]))

    def stop(self, timeout: float = STOP_TIMEOUT_S) -> None:
        """Stop a running provisioning (at shutdown) and wait for it; the next start resumes it."""
        self._cancel.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)

    def unavailable(self) -> SetupError | None:
        """Why provisioning cannot run here, or None."""
        if self.platform != "win32":
            return SetupError(
                code="unsupported_platform",
                message="WanGP setup supports only Windows so far (its pins cover win_amd64), so the engine is not installed here.",
            )
        if self.uv is None or not self.uv.is_file():
            return SetupError(code="uv_missing", message=UV_MISSING)
        return None

    def fingerprint(self) -> str:
        """Hash of everything provisioning installs: a record with another fingerprint is out of date."""
        if self._fingerprint is None:
            pin = self.pin
            data = {
                "commit": pin.commit,
                "archive_sha256": pin.archive_sha256,
                "python": pin.python,
                "torch": pin.torch,
                "torch_index": pin.torch_index,
                "lock_sha256": _file_sha256(self.lock).hexdigest(),
                "ffmpeg": pin.ffmpeg.model_dump(mode="json"),
                "optional_wheels": {name: wheel.model_dump(mode="json") for name, wheel in pin.optional_wheels.items()},
            }
            self._fingerprint = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        return self._fingerprint

    def installed(self) -> bool:
        """Whether a completed provisioning of this pin is in place: its record and its files. Reads no network."""
        layout = self.layout
        record = self._read_record()
        if record is None:
            return False
        expected = {
            "engine": ENGINE_ID,
            "commit": self.pin.commit,
            "pin": self.fingerprint(),
            "root": str(layout.root),
            "python": str(layout.python),
        }
        if any(record.get(key) != value for key, value in expected.items()):
            return False
        if not (layout.root / ROOT_MARKER).is_file() or not layout.python.is_file():
            return False
        for name, pinned in self.pin.ffmpeg.win_amd64.files.items():
            path = layout.ffmpeg_dir / name
            if not path.is_file() or path.stat().st_size != pinned.size:
                return False
        return True

    def _read_record(self) -> dict[str, Any] | None:
        try:
            record = json.loads(self.layout.record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return record if isinstance(record, dict) else None

    def _optional_retry_due(self) -> bool:
        """Whether every optional wheel the record does not list as installed failed only to download."""
        record = self._read_record()
        wheels = record.get("optional_wheels") if record else None
        if not isinstance(wheels, dict):
            return False
        missing = [result for result in wheels.values() if result != "installed"]
        return bool(missing) and all(result == "download_failed" for result in missing)

    # ------------------------------------------------------------------ state

    def _idle(self) -> ProvisioningStatus:
        return ProvisioningStatus("not_installed", error=self.unavailable())

    def _begin(self, target: Callable[[], None], step: int) -> threading.Thread:
        """Enter `provisioning` at `step` and prepare the thread that runs `target`; call with the lock held."""
        self._running = True
        self._cancel.clear()
        self._status = ProvisioningStatus("provisioning", progress=self._progress(step, 0.0))
        self._last_emit = time.monotonic()
        self._thread = threading.Thread(target=target, name="engine-provisioning", daemon=True)
        return self._thread

    def _emit(self) -> None:
        callback = self.on_change
        if callback is not None:
            try:
                callback(self.status())
            except Exception:  # noqa: BLE001 - a broken listener must not stop provisioning
                log.exception("Publishing the provisioning state failed")

    def _finish(self, status: ProvisioningStatus) -> None:
        with self._lock:
            self._status = status
            self._running = False
        self._emit()

    def _cancelled(self) -> bool:
        return self._cancel.is_set()

    def _progress(self, step: int, fraction: float) -> Progress:
        total = sum(s.weight for s in STEPS)
        done = sum(s.weight for s in STEPS[:step]) + STEPS[step].weight * min(max(fraction, 0.0), 1.0)
        return Progress(phase=STEPS[step].phase, step=step + 1, total_steps=len(STEPS), percent=round(100 * done / total, 1))

    def _report(self, step: int, fraction: float = 0.0) -> None:
        if self._cancel.is_set():
            raise Cancelled
        progress = self._progress(step, fraction)
        now = time.monotonic()
        with self._lock:
            previous = self._status.progress
            emit = previous is None or previous.step != progress.step or now - self._last_emit >= PROGRESS_INTERVAL_S
            self._status = ProvisioningStatus("provisioning", progress=progress)
            if emit:
                self._last_emit = now
        if emit:
            self._emit()

    def _run(self) -> None:
        try:
            self._provision()
        except Cancelled:
            log.info("Engine provisioning stopped; the next start resumes it.")
            self._finish(ProvisioningStatus("not_installed"))
        except ProvisioningError as exc:
            log.error("Engine provisioning failed: %s", exc.error.message)
            self._finish(ProvisioningStatus("failed", error=exc.error))
        except OSError as exc:
            log.exception("Engine provisioning failed")
            message = f"Writing the engine files failed ({_short(exc.strerror or str(exc), 80)}), so the engine is not installed."
            self._finish(ProvisioningStatus("failed", error=SetupError(code="io_error", message=message, action="retry")))
        except Exception:  # noqa: BLE001 - any failure is reported in the Setup view
            log.exception("Engine provisioning failed")
            message = "WanGP setup stopped with an unexpected error (details in logs/cinewright.log), so the engine is not installed."
            self._finish(ProvisioningStatus("failed", error=SetupError(code="provisioning_failed", message=message, action="retry")))
        else:
            log.info("WanGP %s is provisioned in %s.", self.pin.commit, self.layout.root)
            self._finish(ProvisioningStatus("ready", version=self.pin.commit))

    def _run_optional(self) -> None:
        """The pass of `retry_optional`: whatever happens, the engine is ready again, with the old record if it fails."""
        try:
            attention, wheels = self._attention(str(self.layout.python), attempts=1)
            self._write_record(attention, wheels)
        except Cancelled:
            log.info("Trying the optional attention wheels again stopped; the next start tries again.")
        except Exception:  # noqa: BLE001 - the engine stays ready with the attention mode it has
            log.exception("Trying the optional attention wheels again failed; WanGP keeps its attention mode")
        self._finish(ProvisioningStatus("ready", version=self.pin.commit))

    # ------------------------------------------------------------------ steps

    def _provision(self) -> None:
        pin, layout = self.pin, self.layout
        packages = lock_packages(self.lock)
        layout.record.unlink(missing_ok=True)  # the record says "ready"; it is written again at the end
        layout.downloads.mkdir(parents=True, exist_ok=True)

        self._report(ARCHIVE)
        archive = self._download(
            ARCHIVE, pin.archive_url, layout.downloads / f"{ENGINE_ID}-{pin.commit}.zip", pin.archive_sha256, pin.archive_size, "WanGP"
        )

        self._report(UNPACK)
        unpack_source(archive, layout.staging, self._cancelled)
        if layout.root.exists():
            shutil.rmtree(layout.root)
        os.replace(layout.staging, layout.root)

        self._report(FFMPEG)
        build = pin.ffmpeg.win_amd64
        ffmpeg = self._download(FFMPEG, build.url, layout.downloads / file_name(build.url), build.sha256, build.size, "ffmpeg")
        place_ffmpeg(ffmpeg, build, layout.ffmpeg_dir)

        self._report(PYTHON)
        self._uv(
            ["venv", "--clear", "--managed-python", "--python", pin.python, str(layout.env)],
            "python_failed",
            f"uv could not set up Python {pin.python} for WanGP",
        )
        if not layout.python.is_file():
            raise ProvisioningError(
                "python_failed", f"uv did not create the Python {pin.python} environment of WanGP, so the engine is not installed."
            )

        self._report(PACKAGES)
        python = str(layout.python)
        isolation: list[str] = []
        if sdist_only(packages):
            requirements = layout.downloads / "build-requirements.txt"
            requirements.write_text(build_requirements(packages), encoding="utf-8")
            self._uv(
                ["pip", "install", "--python", python, "--require-hashes", "--no-deps", "--no-index", "-r", str(requirements)],
                "install_failed",
                "Installing the build tools of WanGP's packages failed",
            )
            isolation = ["--no-build-isolation"]
        self._uv(
            [
                "pip",
                "install",
                "--python",
                python,
                "--require-hashes",
                "--no-index",
                *isolation,
                "--preview-features",
                "pylock",
                "-r",
                str(self.lock),
            ],
            "install_failed",
            "Installing WanGP's Python packages failed",
        )

        self._report(ATTENTION)
        attention, wheels = self._attention(python)
        self._write_record(attention, wheels)

    def _download(self, step: int, url: str, target: Path, sha256: str, size: int, what: str) -> Path:
        host = urlsplit(url).hostname
        try:
            return self._downloader.fetch(
                url, target, sha256=sha256, size=size, progress=lambda done: self._report(step, done / size), cancelled=self._cancelled
            )
        except HashMismatch:
            raise ProvisioningError(
                "hash_mismatch",
                f"The {what} download from {host} did not match its pinned SHA-256, so it was deleted and the engine is not installed.",
            ) from None
        except DownloadError as exc:
            raise ProvisioningError(
                "download_failed", f"Downloading {what} from {host} failed ({exc}), so the engine is not installed."
            ) from None

    def _uv(self, args: list[str], code: str, what: str) -> None:
        if self.uv is None:
            raise ProvisioningError("uv_missing", UV_MISSING, retry=False)
        try:
            result = self._runner([str(self.uv), *args], uv_environment(self._environ, self.layout), self._cancelled)
        except StartFailed as exc:
            raise ProvisioningError(
                "start_failed",
                f"Starting {exc.program} of the Cinewright installation failed ({exc.reason}), so the engine is not installed.",
            ) from None
        if result.returncode == 0:
            return
        if "hash mismatch" in result.output.lower():
            raise ProvisioningError(
                "hash_mismatch",
                "A downloaded package of WanGP's environment did not match its pinned SHA-256, so it was refused and the engine "
                "is not installed.",
            )
        raise ProvisioningError(
            code, f"{what} (uv exit code {result.returncode}, details in logs/{LOG_NAME}), so the engine is not installed."
        )

    def _attention(self, python: str, attempts: int = OPTIONAL_ATTEMPTS) -> tuple[str, dict[str, str]]:
        """Install the optional wheels that match their pins and probe SageAttention; returns the mode and per-wheel results."""
        results: dict[str, str] = {}
        files: list[Path] = []
        for name, wheel in self.pin.optional_wheels.items():
            try:
                files.append(self._fetch_optional(name, wheel.url, wheel.sha256, attempts))
            except HashMismatch:
                log.warning("The optional wheel %s does not match its pinned SHA-256; it is not installed.", name)
                results[name] = "hash_mismatch"
            except DownloadError as exc:
                log.warning("The optional wheel %s could not be downloaded (%s); it is not installed.", name, exc)
                results[name] = "download_failed"
            else:
                results[name] = "verified"
        env = uv_environment(self._environ, self.layout)
        if files:
            # --no-deps: the wheels must not move torch or anything else the lock installed.
            install = ["pip", "install", "--python", python, "--no-deps", "--no-index", *map(str, files)]
            try:
                outcome = "installed" if self._runner([str(self.uv), *install], env, self._cancelled).returncode == 0 else "install_failed"
            except StartFailed as exc:
                log.warning("The optional wheels were not installed: %s.", exc)
                outcome = "install_failed"
            results = {name: outcome if result == "verified" else result for name, result in results.items()}
        if not results or any(result != "installed" for result in results.values()):
            return DEFAULT_ATTENTION, results
        try:
            probe = self._runner([python, "-I", "-c", PROBE_SCRIPT], env, self._cancelled, PROBE_TIMEOUT_S)
        except StartFailed as exc:
            log.warning("The attention probe did not start: %s.", exc)
            probe = RunResult(-1, "")
        mode = parse_probe(probe.output) if probe.returncode == 0 else DEFAULT_ATTENTION
        log.info("Attention probe: %s.", mode if mode != DEFAULT_ATTENTION else "SageAttention does not work here, WanGP uses sdpa")
        return mode, results

    def _fetch_optional(self, name: str, url: str, sha256: str, attempts: int) -> Path:
        """Download an optional wheel in up to `attempts` tries (a verified earlier download is reused); a hash mismatch is final."""
        attempt = 1
        while True:
            try:
                return self._downloader.fetch(
                    url,
                    self.layout.downloads / file_name(url),
                    sha256=sha256,
                    max_bytes=OPTIONAL_WHEEL_MAX_BYTES,
                    cancelled=self._cancelled,
                )
            except DownloadError as exc:
                if attempt >= attempts:
                    raise
                log.info("Downloading the optional wheel %s failed (%s); trying again.", name, exc)
            if self._cancel.wait(OPTIONAL_RETRY_DELAY_S):
                raise Cancelled
            attempt += 1

    def _write_record(self, attention: str, wheels: dict[str, str]) -> None:
        layout = self.layout
        record = {
            "engine": ENGINE_ID,
            "commit": self.pin.commit,
            "pin": self.fingerprint(),
            "root": str(layout.root),
            "python": str(layout.python),
            "attention": attention,
            "optional_wheels": wheels,
            "completed_at": to_iso(self._clock()),
        }
        part = layout.record.with_name(layout.record.name + ".part")
        part.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        os.replace(part, layout.record)

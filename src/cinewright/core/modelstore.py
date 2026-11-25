"""Model store: the model directory the core owns, `<data dir>/models` (#3).

Wan2GP finds a file by joining each of its checkpoint roots with the file's relative path (upstream
`shared/utils/files_locator.py`: `locate_file`, `get_local_model_filename`). The store is its first root
(`checkpoints_paths` in `wgp_config.json`), so a quantization's transformer files and the root-level shared files
lie at `<store>/<name>`, every other shared file at `<store>/<subfolder><name>` (`umt5-xxl/`, `Qwen3/`, ...).

Files are shared: `store_files` holds each file once, and `installed_model_files` ties it to every installed
quantization that needs it, which is its reference count. A file goes only when no installed quantization
references it, no install in progress needs it and no job that waits or runs on the local host needs it.

An install (`prepare_install`, then `run_install`):

1. plans the files (`plan_files`): the quantization's transformer files, and of every shared component the
   descriptor `requires` the files without a variant plus those of the host's text encoder variant;
2. checks before any byte moves: allowed file types, paths inside the store, pinned sizes; the quota, planning
   the eviction of least recently used quantizations no waiting or running job needs; the free space of the
   store's drive, counting what that eviction frees. Only then it evicts;
3. downloads the missing files into `.downloads/` with the `Downloader` of engine provisioning: resumable `.part`
   files, SHA-256 and size checked before a file is used, HTTPS to the registry's hosts and their download CDNs
   only (`DOWNLOAD_HOSTS`, every redirect included);
4. once every file verified, moves them into place and registers them with their references in one
   transaction, then merges `checkpoints_paths` and `text_encoder_quantization` into `wgp_config.json`.

A failed or cancelled install registers nothing; its verified and partial downloads stay in `.downloads/` for the
next try. After an install or a removal the caller restarts the local worker, so it re-announces its models.
Wan2GP defines the MVP model types itself (`<wan2gp root>/defaults/<model_type>.json`); a finetune definition is
written only for a model type it does not define. Do not call the store from inside a database transaction.

A text encoder variant switch (`prepare_switch`, then `run_switch`) checks free disk space, downloads the other
variant's files of every installed quantization the same way, re-registers each one's references and stores the
variant in one transaction, and rewrites `text_encoder_quantization`. The caller then restarts the worker and
collects the previous variant's files (`collect_garbage`). Installs, removals and a switch exclude each other.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import urlsplit

import httpx

from cinewright.api.schemas import ModelDescriptor
from cinewright.core.clock import Clock, to_iso, utcnow
from cinewright.core.provisioning import ENGINE_ID, Cancelled, Downloader, DownloadError, HashMismatch
from cinewright.core.registry import Registry
from cinewright.db.database import Database
from cinewright.paths import DataPaths

log = logging.getLogger(__name__)

STAGING_DIR = ".downloads"  # below the store; registry paths start with a letter or digit, so none collides with it
WGP_CONFIG = "wgp_config.json"  # in <data dir>/engines/wan2gp/, where the Wan2GP adapter points Wan2GP
FINETUNES_DIR = "finetunes"  # below the Wan2GP root
GB = 1024**3  # sizes in GB as the UI shows them (binary units)
TEXT_ENCODER_VARIANTS = ("bf16", "int8")
DEFAULT_TEXT_ENCODER_VARIANT = "int8"  # Wan2GP's own default text_encoder_quantization
WEIGHT_EXTENSIONS = frozenset({".safetensors", ".gguf"})
COMPANION_EXTENSIONS = frozenset({".json", ".txt", ".model"})  # non-executable; only in shared components
# Exact host names per registry provider: the registry's file host and the CDN hosts its downloads redirect to.
# Observed on 2026-11-25: Hugging Face answers a file in Git LFS with a redirect to us.aws.cdn.hf.co and serves
# other files itself (after a relative redirect). No registry file comes from CivitAI yet, so its CDN is unobserved.
DOWNLOAD_HOSTS: Mapping[str, frozenset[str]] = {
    "huggingface": frozenset({"huggingface.co", "us.aws.cdn.hf.co"}),
    "civitai": frozenset({"civitai.com"}),
}
_WINDOWS_DEVICES = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{n}" for n in range(1, 10)), *(f"LPT{n}" for n in range(1, 10))})
_MODEL_TYPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_CHUNK = 1024 * 1024

WgpConfigResult = Literal["updated", "unchanged", "missing"]
InstallKey = tuple[str, str]  # (model id, quantization id)


class ModelStoreError(Exception):
    """An install or removal that cannot go ahead: cause and consequence in one sentence, and a stable `code`.

    Codes: `unknown_model`, `unknown_quant`, `unknown_variant`, `file_type`, `unsafe_path`, `size_missing`,
    `engine_type_unknown`, `quota_exceeded`, `not_enough_space`, `installing`, `switching`, `in_use`, `not_installed`,
    `hash_mismatch`, `download_failed`, `cancelled`, `io_error`, `install_failed`, `config_invalid`.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class PlannedFile:
    """One file an install needs, as the registry pins it."""

    path: str  # below the store, `/`-separated, as Wan2GP looks it up
    url: str
    sha256: str
    size_bytes: int
    provider: str  # registry source provider; picks the allowed download hosts
    shared: bool  # from a shared component; otherwise a transformer file of the quantization

    @property
    def key(self) -> str:
        """Paths ignore case, as on Windows."""
        return self.path.casefold()


@dataclass(frozen=True)
class InstallProgress:
    model_id: str
    quant: str
    done_bytes: int
    total_bytes: int  # bytes of the files this install downloads
    file: str | None  # store path of the file in progress


@dataclass(frozen=True)
class Install:
    """A prepared install: checked, with its eviction done; `run_install` downloads and registers it."""

    model_id: str
    quant: str
    label: str  # model name and quantization, for messages
    variant: str  # text encoder variant the files follow
    files: tuple[PlannedFile, ...]
    missing: tuple[PlannedFile, ...]  # files the store does not hold yet
    download_bytes: int  # what remains to download, partial downloads deducted
    evicted: tuple[InstallKey, ...]

    @property
    def key(self) -> InstallKey:
        return (self.model_id, self.quant)


@dataclass(frozen=True)
class VariantSwitch:
    """A prepared text encoder variant switch: checked; `run_switch` downloads and registers it."""

    variant: str
    files: tuple[PlannedFile, ...]  # every file the installed quantizations need with `variant`
    missing: tuple[PlannedFile, ...]  # files the store does not hold yet
    download_bytes: int  # what remains to download, partial downloads deducted
    plans: dict[InstallKey, tuple[PlannedFile, ...]]  # the files of each installed quantization with `variant`

    @property
    def label(self) -> str:
        return f"the {self.variant} text encoder"


@dataclass(frozen=True)
class StoreUsage:
    installed: dict[InstallKey, int]  # size of each installed quantization, its shared files included
    by_model: dict[str, int]  # size of each model's installed quantizations together, each file once
    total_bytes: int  # every file in the store, each shared file once
    quota_bytes: int | None
    free_bytes: int  # free space of the store's drive, measured now


# --------------------------------------------------------------------------- rules


def check_file_type(path: str, *, shared: bool) -> None:
    """Weights are safetensors or GGUF; shared components may add pinned non-executable companion files."""
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in WEIGHT_EXTENSIONS or (shared and suffix in COMPANION_EXTENSIONS):
        return
    allowed = "safetensors or GGUF weights, or .json, .txt and .model companion files" if shared else "safetensors or GGUF weights"
    raise ModelStoreError("file_type", f"{path} is not an allowed model file ({allowed}), so nothing was installed.")


def store_path(root: Path, relative: str) -> Path:
    """The file `relative` (a registry path) below `root`; refuses a path that could leave `root` or that Windows reads otherwise."""
    parts = relative.split("/")
    unsafe = (
        not relative
        or "\x00" in relative
        or "\\" in relative
        or ":" in relative
        or relative.startswith("/")
        or any(part in ("", ".", "..") or part != part.rstrip(". ") or part.split(".", 1)[0].upper() in _WINDOWS_DEVICES for part in parts)
    )
    if not unsafe:
        target = root.joinpath(*parts)
        if target.resolve().is_relative_to(root.resolve()):
            return target
    raise ModelStoreError("unsafe_path", f"The registry path {relative!r} does not stay inside the model store, so nothing was installed.")


def plan_files(registry: Registry, model_id: str, quant: str, variant: str) -> list[PlannedFile]:
    """Every file one quantization needs with the text encoder `variant`, in install order; each path once.

    First the quantization's transformer files, then, for each shared component the descriptor `requires`, its
    files without a variant and those of `variant`. Refuses an unknown model, quantization or variant, a file type
    outside the rules and a file without a pinned size.
    """
    model = registry.get(model_id)
    if model is None:
        raise ModelStoreError("unknown_model", f"Model {model_id} is not in the registry.")
    quantization = next((q for q in model.quantizations if q.id == quant), None)
    if quantization is None:
        raise ModelStoreError("unknown_quant", f"{model.name} has no quantization {quant}.")
    if variant not in TEXT_ENCODER_VARIANTS:
        raise ModelStoreError("unknown_variant", f"{variant} is not a text encoder variant; use {' or '.join(TEXT_ENCODER_VARIANTS)}.")
    label = f"{model.name} {quant}"
    planned: dict[str, PlannedFile] = {}

    def add(path: str, url: str, sha256: str, size: int | None, provider: str, shared: bool) -> None:
        check_file_type(path, shared=shared)
        if size is None:
            raise ModelStoreError(
                "size_missing",
                f"{path} of {label} has no pinned size in the registry, so neither free space nor the download can be checked.",
            )
        planned.setdefault(path.casefold(), PlannedFile(path, url, sha256, size, provider, shared))

    for file in quantization.files:
        add(file.name, file.url, file.sha256, file.size_bytes, model.source.provider, False)
    for component in registry.requirements(model_id):
        for shared_file in component.files_for(variant):
            add(shared_file.path, shared_file.url, shared_file.sha256, shared_file.size_bytes, component.source.provider, True)
    return list(planned.values())


def format_size(size: int) -> str:
    """Binary units with one decimal, as the UI shows sizes."""
    if size < 1024:
        return f"{size} B"
    value, units = size / 1024, ["KB", "MB", "GB", "TB"]
    while value >= 1024 and len(units) > 1:
        value /= 1024
        units.pop(0)
    return f"{value:.1f} {units[0]}"


# --------------------------------------------------------------------------- Wan2GP


def merge_wgp_config(path: Path, store_root: Path, variant: str) -> WgpConfigResult:
    """Register the store in Wan2GP's config: `checkpoints_paths` with the store first, and `text_encoder_quantization`.

    Every other key keeps its value. A missing file stays missing (`missing`): Wan2GP writes its full defaults
    only when the file does not exist and reads some of them without a fallback at import, so a file with only
    these two keys would stop it; merge again once Wan2GP wrote it. Wan2GP reads the file once at import, so a
    change takes effect when the worker restarts.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "missing"
    try:
        config = json.loads(text)
    except ValueError:
        config = None
    if not isinstance(config, dict):
        raise ModelStoreError("config_invalid", f"{path} is not a JSON object, so the model store is not registered in WanGP.")
    store = str(store_root.resolve())
    current = config.get("checkpoints_paths")
    kept = (
        [entry for entry in current if isinstance(entry, str) and entry.strip() and not _names_store(entry, store)]
        if isinstance(current, list)
        else []
    )
    paths = [store, *kept]
    if current == paths and config.get("text_encoder_quantization") == variant:
        return "unchanged"
    config["checkpoints_paths"] = paths
    config["text_encoder_quantization"] = variant
    _write_json(path, config)
    return "updated"


def _names_store(entry: str, store: str) -> bool:
    """Whether a `checkpoints_paths` entry is the store; a relative entry is relative to Wan2GP's root, never the store."""
    return os.path.isabs(entry) and os.path.normcase(os.path.normpath(entry)) == os.path.normcase(os.path.normpath(store))


def _write_json(path: Path, data: object) -> None:
    part = path.with_name(path.name + ".part")
    part.write_bytes((json.dumps(data, indent=4) + "\n").encode("utf-8"))
    os.replace(part, path)


def wan2gp_defines(engine_root: Path, model_type: str) -> bool:
    """Whether the Wan2GP checkout defines `model_type` itself, in `defaults/<model_type>.json`."""
    return _MODEL_TYPE.fullmatch(model_type) is not None and (engine_root / "defaults" / f"{model_type}.json").is_file()


def finetune_path(engine_root: Path, model_type: str) -> Path:
    if _MODEL_TYPE.fullmatch(model_type) is None:
        raise ModelStoreError("unsafe_path", f"The model type {model_type!r} is no file name, so no finetune definition was written.")
    return store_path(engine_root / FINETUNES_DIR, f"{model_type}.json")


def write_finetune(engine_root: Path, model_type: str, *, name: str, architecture: str, files: list[str]) -> Path:
    """Define a model type Wan2GP does not define: `<wan2gp root>/finetunes/<model_type>.json`.

    `architecture` names an architecture Wan2GP supports; `files` are transformer file names, one per quantization,
    which Wan2GP finds in its checkpoint roots. Wan2GP reads `finetunes/` only at import.
    """
    target = finetune_path(engine_root, model_type)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_json(target, {"model": {"name": name, "architecture": architecture, "URLs": files}})
    return target


# --------------------------------------------------------------------------- downloads


class AllowedHosts(httpx.BaseTransport):
    """Passes a request on only over HTTPS to one of `hosts`, compared by exact host name.

    The client sends every redirect through its transport too, so each hop is checked. A refusal is a transport
    error, which the `Downloader` reports as a failed download.
    """

    def __init__(self, inner: httpx.BaseTransport, hosts: frozenset[str]) -> None:
        self._inner = inner
        self._hosts = hosts

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        if url.scheme != "https" or url.host not in self._hosts or url.port not in (None, 443) or url.userinfo:
            raise httpx.ConnectError(f"refused {url.scheme}://{url.host}, which is not a download host of the registry", request=request)
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


def network_transport() -> httpx.BaseTransport:
    """A transport for real downloads; a client with its own transport ignores the proxy variables, so this one reads them."""
    names = ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
    proxy = next((os.environ[name] for name in names if os.environ.get(name)), None)
    return httpx.HTTPTransport(proxy=proxy)


def disk_free(path: Path) -> int:
    """Free bytes on the drive that holds `path`."""
    while not path.exists() and path.parent != path:
        path = path.parent
    return shutil.disk_usage(path).free


def _cancelled(label: str) -> ModelStoreError:
    return ModelStoreError("cancelled", f"The install of {label} was cancelled; a later install continues its downloads.")


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- the store


class ModelStore:
    """Installs and removes registry models in `<data dir>/models` for the worker host `host_id` (the local host)."""

    def __init__(
        self,
        db: Database,
        registry: Registry,
        paths: DataPaths,
        *,
        host_id: str = "local",
        quota_bytes: Callable[[], int | None] = lambda: None,
        free_bytes: Callable[[Path], int] = disk_free,
        transport: httpx.BaseTransport | None = None,
        engine_root: Callable[[], Path | None] = lambda: None,
        architectures: Mapping[str, str] | None = None,
        clock: Clock = utcnow,
    ) -> None:
        """`quota_bytes` reads the quota (`settings.model_store_quota_gb` times `GB`), None for none. `free_bytes` measures the drive.

        `transport` replaces the network (tests); `engine_root` names the provisioned Wan2GP root, None while there is
        none. `architectures` maps a model type Wan2GP does not define to the architecture of its finetune definition.
        One store per core start: it marks an install the previous start left `downloading` as failed.
        """
        self.db = db
        self.registry = registry
        self.root = paths.models_dir
        self.config_path = paths.engines_dir / ENGINE_ID / WGP_CONFIG
        self.host_id = host_id
        self._quota_bytes = quota_bytes
        self._free_bytes = free_bytes
        self._transport = transport
        self._engine_root = engine_root
        self._architectures = dict(architectures or {})
        self._clock = clock
        self._lock = threading.RLock()  # plans, registrations and removals
        self._download_lock = threading.Lock()  # one install downloads at a time; two may want the same file
        self._active: set[InstallKey] = set()
        self._switching: str | None = None  # the text encoder variant a switch in progress installs
        self._reserved: Counter[str] = Counter()  # path keys of the files active installs and a switch need
        # An install the last core start did not finish (Cinewright closed) left its row `downloading`.
        self.db.execute(
            "UPDATE installed_models SET state = 'failed', error = ? WHERE state = 'downloading'",
            ("The install stopped when Cinewright closed; install the model again to continue its downloads.",),
        )

    # ------------------------------------------------------------------ queries

    def text_encoder_variant(self) -> str:
        """The host's text encoder variant; `int8` (Wan2GP's default) until one is chosen or installed."""
        row = self.db.query_one("SELECT text_encoder_variant FROM hosts WHERE id = ?", (self.host_id,))
        value = row["text_encoder_variant"] if row is not None else None
        return value if value in TEXT_ENCODER_VARIANTS else DEFAULT_TEXT_ENCODER_VARIANT

    def stored_variant(self) -> str | None:
        """The host's text encoder variant as stored: None until one is chosen or installed."""
        row = self.db.query_one("SELECT text_encoder_variant FROM hosts WHERE id = ?", (self.host_id,))
        return row["text_encoder_variant"] if row is not None else None

    def has_installs(self) -> bool:
        return self.db.query_one("SELECT 1 FROM installed_models WHERE state = 'installed' LIMIT 1") is not None

    def plan(self, model_id: str, quant: str, variant: str | None = None) -> list[PlannedFile]:
        """`plan_files` with the host's text encoder variant unless `variant` is given."""
        return plan_files(self.registry, model_id, quant, variant or self.text_encoder_variant())

    def usage(self) -> StoreUsage:
        rows = self.db.query("SELECT model_id, quant, size_bytes FROM installed_models WHERE state = 'installed' ORDER BY model_id, quant")
        total = self.db.query_one("SELECT COALESCE(SUM(size_bytes), 0) AS total FROM store_files")
        by_model = self.db.query(
            "SELECT refs.model_id, SUM(store_files.size_bytes) AS size FROM (SELECT DISTINCT model_id, path FROM installed_model_files) "
            "AS refs JOIN store_files ON store_files.path = refs.path GROUP BY refs.model_id"
        )
        return StoreUsage(
            installed={(row["model_id"], row["quant"]): row["size_bytes"] or 0 for row in rows},
            by_model={row["model_id"]: row["size"] for row in by_model},
            total_bytes=total["total"] if total is not None else 0,
            quota_bytes=self._quota_bytes(),
            free_bytes=self._free_bytes(self.root),
        )

    def needed_by_jobs(self) -> set[InstallKey]:
        """Quantizations that a job waiting or running on this host is routed to."""
        rows = self.db.query(
            "SELECT DISTINCT json_extract(route, '$.model_id') AS model_id, json_extract(route, '$.quant') AS quant FROM jobs "
            "WHERE route IS NOT NULL AND json_extract(route, '$.host_id') = ? "
            "AND state IN ('awaiting_approval', 'queued', 'leased', 'running')",
            (self.host_id,),
        )
        return {(row["model_id"], row["quant"]) for row in rows if row["model_id"] and row["quant"]}

    def mark_used(self, model_id: str, quant: str) -> None:
        """Record that a job used the quantization now; eviction removes the least recently used first."""
        self.db.execute(
            "UPDATE installed_models SET last_used_at = ? WHERE model_id = ? AND quant = ? AND state = 'installed'",
            (to_iso(self._clock()), model_id, quant),
        )

    def register_in_wan2gp(self) -> WgpConfigResult:
        """Merge the store and the host's text encoder variant into `wgp_config.json` (see `merge_wgp_config`)."""
        return merge_wgp_config(self.config_path, self.root, self.text_encoder_variant())

    # ------------------------------------------------------------------ install

    def install(
        self,
        model_id: str,
        quant: str,
        *,
        progress: Callable[[InstallProgress], None] | None = None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> Install:
        """`prepare_install`, then `run_install`; returns the prepared install. `run_install` also returns the config merge result."""
        install = self.prepare_install(model_id, quant)
        self.run_install(install, progress=progress, cancelled=cancelled)
        return install

    def prepare_install(self, model_id: str, quant: str) -> Install:
        """Plan and check an install, then evict what the quota needs; refuses before any byte is downloaded or removed.

        Marks the quantization `downloading` (unless it is installed and complete) and reserves its files until
        `run_install` ends.
        """
        with self._lock:
            variant = self.text_encoder_variant()
            files = self.plan(model_id, quant, variant)
            model = self.registry.get(model_id)
            assert model is not None  # plan_files checked it
            label = f"{model.name} {quant}"
            for file in files:
                store_path(self.root, file.path)
            self._check_engine_types(model, quant, label)
            key = (model_id, quant)
            if key in self._active:
                raise ModelStoreError("installing", f"{label} is being installed already.")
            if self._switching is not None:
                raise ModelStoreError(
                    "switching",
                    f"The text encoder is being switched to {self._switching}, so {label} was not installed; install it afterwards.",
                )
            stored = self._stored()
            missing = [file for file in files if not self._present(file, stored)]
            download = sum(file.size_bytes - self._staged_bytes(file) for file in missing)
            growth = sum(file.size_bytes - (stored[file.key][1] if file.key in stored else 0) for file in missing)
            evicted, freed = self._plan_eviction(key, files, growth, label)
            available = self._free_bytes(self.root) + freed
            if download > available:
                raise ModelStoreError(
                    "not_enough_space",
                    f"Installing {label} needs {format_size(download - available)} more free space on the drive of the model store: "
                    f"it downloads {format_size(download)}, and only {format_size(available)} are free.",
                )
            if evicted or freed:
                for victim in evicted:
                    log.info("Removing %s %s, the least recently used model, to keep the model store within its quota.", *victim)
                    self._unregister(victim)
                self._collect_garbage(files)
            row = self.db.query_one("SELECT state FROM installed_models WHERE model_id = ? AND quant = ?", key)
            if missing or row is None or row["state"] != "installed":
                self.db.execute(
                    "INSERT INTO installed_models (model_id, quant, state) VALUES (?, ?, 'downloading') "
                    "ON CONFLICT (model_id, quant) DO UPDATE SET state = 'downloading', error = NULL",
                    key,
                )
            self._active.add(key)
            self._reserved.update(file.key for file in files)
            return Install(model_id, quant, label, variant, tuple(files), tuple(missing), download, tuple(evicted))

    def run_install(
        self,
        install: Install,
        *,
        progress: Callable[[InstallProgress], None] | None = None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> WgpConfigResult:
        """Download, verify, place and register a prepared install, then register the store in Wan2GP.

        Returns what the `wgp_config.json` merge did; `missing` means Wan2GP has not written its config yet, so it
        does not see the store until a merge after its first start. On any failure nothing of the install is
        registered, the quantization is `failed` with the message (or absent again when cancelled), and
        `ModelStoreError` is raised. A config that cannot be merged raises after the registration: the files are
        installed, but Wan2GP does not see them until the config is fixed.
        """
        report = progress or (lambda _: None)
        try:
            with self._download_lock:
                ready = self._download(
                    install.label,
                    install.missing,
                    lambda done, total, path: report(InstallProgress(install.model_id, install.quant, done, total, path)),
                    cancelled,
                )
            self._register(install, ready)
        except ModelStoreError as exc:
            self._fail(install, exc.code, exc.message)
            raise
        except OSError as exc:
            log.exception("Installing %s failed", install.label)
            message = f"Writing the files of {install.label} failed ({exc.strerror or exc}), so it is not installed."
            self._fail(install, "io_error", message)
            raise ModelStoreError("io_error", message) from exc
        except Exception as exc:  # noqa: BLE001 - whatever stopped it, the install must end as failed
            log.exception("Installing %s failed", install.label)
            message = (
                f"The install of {install.label} stopped with an unexpected error (details in logs/cinewright.log), so it is not installed."
            )
            self._fail(install, "install_failed", message)
            raise ModelStoreError("install_failed", message) from exc
        finally:
            with self._lock:
                self._active.discard(install.key)
                self._release(install.files)
        return self.register_in_wan2gp()

    def _download(
        self,
        label: str,
        missing: tuple[PlannedFile, ...],
        report: Callable[[int, int, str | None], None],
        cancelled: Callable[[], bool],
    ) -> dict[str, Path]:
        """Each missing file as a verified file: staged, or already where it belongs. Keyed by path key.

        `report(done_bytes, total_bytes, file)` follows the progress; `label` names what is installed in messages.
        """
        total = sum(file.size_bytes for file in missing)
        done = 0
        ready: dict[str, Path] = {}
        by_sha256: dict[str, Path] = {}
        report(0, total, None)
        for file in missing:
            if cancelled():
                raise _cancelled(label)
            final = store_path(self.root, file.path)
            if self._present(file, self._stored()):  # another install registered it meanwhile
                source = final
            elif file.sha256 in by_sha256:  # the same bytes at a second path
                source = by_sha256[file.sha256]
            elif final.is_file() and final.stat().st_size == file.size_bytes and _sha256_of(final) == file.sha256:
                source = final  # placed by an install that stopped before it registered the file
            else:
                base = done
                source = self._fetch(
                    label,
                    file,
                    lambda received, base=base, path=file.path: report(base + received, total, path),
                    cancelled,
                )
            ready[file.key] = by_sha256[file.sha256] = source
            done += file.size_bytes
            report(done, total, file.path)
        return ready

    def _fetch(self, label: str, file: PlannedFile, progress: Callable[[int], None], cancelled: Callable[[], bool]) -> Path:
        inner = self._transport if self._transport is not None else network_transport()
        downloader = Downloader(transport=AllowedHosts(inner, DOWNLOAD_HOSTS.get(file.provider, frozenset())))
        host = urlsplit(file.url).hostname
        try:
            return downloader.fetch(
                file.url,
                self.root / STAGING_DIR / file.sha256,
                sha256=file.sha256,
                size=file.size_bytes,
                progress=progress,
                cancelled=cancelled,
            )
        except HashMismatch:
            raise ModelStoreError(
                "hash_mismatch",
                f"{file.path} of {label} did not match its pinned SHA-256 and size, so it was deleted and {label} is not installed.",
            ) from None
        except DownloadError as exc:
            raise ModelStoreError(
                "download_failed",
                f"Downloading {file.path} of {label} from {host} failed ({exc}), so {label} is not installed; "
                "a later install continues the download.",
            ) from None
        except Cancelled:
            raise _cancelled(label) from None

    def _place(self, conn: sqlite3.Connection, missing: tuple[PlannedFile, ...], ready: dict[str, Path], now: str) -> None:
        """Move each verified file to its path in the store and register it (inside the caller's transaction)."""
        uses = Counter(ready[file.key] for file in missing)
        for file in missing:
            source, final = ready[file.key], store_path(self.root, file.path)
            uses[source] -= 1
            if source != final:
                final.parent.mkdir(parents=True, exist_ok=True)
                if uses[source]:
                    shutil.copyfile(source, final)
                else:
                    os.replace(source, final)
            conn.execute(
                "INSERT INTO store_files (path, sha256, size_bytes, verified_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT (path) DO UPDATE SET sha256 = excluded.sha256, size_bytes = excluded.size_bytes, "
                "verified_at = excluded.verified_at",
                (file.path, file.sha256, file.size_bytes, now),
            )

    def _register(self, install: Install, ready: dict[str, Path]) -> None:
        now = to_iso(self._clock())
        size = sum(file.size_bytes for file in install.files)
        with self._lock, self.db.transaction() as conn:
            self._place(conn, install.missing, ready, now)
            conn.execute("DELETE FROM installed_model_files WHERE model_id = ? AND quant = ?", install.key)
            conn.execute(
                "INSERT INTO installed_models (model_id, quant, state, size_bytes, installed_at, last_used_at) "
                "VALUES (?, ?, 'installed', ?, ?, ?) "
                "ON CONFLICT (model_id, quant) DO UPDATE SET state = 'installed', size_bytes = excluded.size_bytes, "
                "installed_at = excluded.installed_at, last_used_at = excluded.last_used_at, error = NULL",
                (*install.key, size, now, now),
            )
            conn.executemany(
                "INSERT INTO installed_model_files (model_id, quant, path) VALUES (?, ?, ?)",
                [(*install.key, file.path) for file in install.files],
            )
            # The first install on a host stores the variant it installed.
            conn.execute(
                "UPDATE hosts SET text_encoder_variant = ? WHERE id = ? AND text_encoder_variant IS NULL", (install.variant, self.host_id)
            )
        model = self.registry.get(install.model_id)
        if model is not None:
            self._write_finetunes(model, install.quant)

    def _fail(self, install: Install, code: str, message: str) -> None:
        with self._lock, self.db.transaction() as conn:
            conn.execute("DELETE FROM installed_model_files WHERE model_id = ? AND quant = ?", install.key)
            if code == "cancelled":
                conn.execute("DELETE FROM installed_models WHERE model_id = ? AND quant = ?", install.key)
            else:
                conn.execute(
                    "UPDATE installed_models SET state = 'failed', error = ? WHERE model_id = ? AND quant = ?", (message, *install.key)
                )

    # ------------------------------------------------------------------ remove

    def remove(self, model_id: str, quant: str | None = None) -> list[InstallKey]:
        """Remove one quantization (every one of the model when `quant` is None); returns what was removed.

        Refuses while it is being installed or a job waiting or running needs it. Its files go unless another
        installed quantization, an install in progress or such a job needs them; so do its leftover downloads.
        """
        model = self.registry.get(model_id)
        name = model.name if model is not None else model_id
        with self._lock:
            if quant is None:
                rows = self.db.query("SELECT quant, state FROM installed_models WHERE model_id = ? ORDER BY quant", (model_id,))
            else:
                rows = self.db.query("SELECT quant, state FROM installed_models WHERE model_id = ? AND quant = ?", (model_id, quant))
            if not rows:
                raise ModelStoreError("not_installed", f"{name}{'' if quant is None else ' ' + quant} is not installed.")
            if self._switching is not None:
                raise ModelStoreError(
                    "switching",
                    f"The text encoder is being switched to {self._switching}, so {name} was not removed; remove it afterwards.",
                )
            needed = self.needed_by_jobs()
            keys = [(model_id, row["quant"]) for row in rows]
            for key in keys:
                if key in self._active:
                    raise ModelStoreError("installing", f"{name} {key[1]} is being installed, so it was not removed.")
                if key in needed:
                    raise ModelStoreError("in_use", f"A queued or running job needs {name} {key[1]}, so it was not removed.")
            leftovers = []
            for key in keys:
                self._unregister(key)
                try:
                    leftovers += self.plan(*key)
                except ModelStoreError:
                    pass  # no longer in the registry: nothing of it is staged under a known name
            self._collect_garbage()
            self._discard_downloads(leftovers)
            if model is not None:
                self._remove_finetunes(model)
            return keys

    def collect_garbage(self) -> None:
        """Delete every registered file that no installed quantization references and nothing else needs.

        After a variant switch, this removes the previous variant's files. A file that cannot be deleted (on Windows,
        one a worker process still holds open) stays registered and goes with a later collection.
        """
        with self._lock:
            self._collect_garbage()

    # ------------------------------------------------------------------ text encoder variant

    def prepare_switch(self, variant: str) -> VariantSwitch:
        """Plan and check a switch of the host's text encoder variant to `variant`; refuses before any byte is downloaded.

        Refuses while a job runs on the host, while an install or another switch is in progress, and when the files
        the installed quantizations need with `variant` do not fit on the store's drive. The quota is not applied:
        a switch removes nothing but the previous variant's files. Reserves the planned files until `run_switch` ends.
        """
        if variant not in TEXT_ENCODER_VARIANTS:
            raise ModelStoreError("unknown_variant", f"{variant} is not a text encoder variant; use {' or '.join(TEXT_ENCODER_VARIANTS)}.")
        label = f"the {variant} text encoder"
        with self._lock:
            if self._switching is not None:
                raise ModelStoreError("switching", f"The text encoder is being switched to {self._switching} already.")
            if self._active:
                raise ModelStoreError(
                    "installing", "A model is being installed, so the text encoder was not switched; switch it afterwards."
                )
            if self._running_jobs():
                raise ModelStoreError(
                    "in_use", "A job is running on this computer, so the text encoder was not switched; switch it once the job ended."
                )
            plans: dict[InstallKey, tuple[PlannedFile, ...]] = {}
            for row in self.db.query("SELECT model_id, quant FROM installed_models WHERE state = 'installed' ORDER BY model_id, quant"):
                key = (row["model_id"], row["quant"])
                try:
                    plans[key] = tuple(plan_files(self.registry, *key, variant))
                except ModelStoreError:
                    continue  # no longer in the registry as installed: its references stay as they are
            files = list({file.key: file for plan in plans.values() for file in plan}.values())
            for file in files:
                store_path(self.root, file.path)
            stored = self._stored()
            missing = [file for file in files if not self._present(file, stored)]
            download = sum(file.size_bytes - self._staged_bytes(file) for file in missing)
            available = self._free_bytes(self.root)
            if download > available:
                raise ModelStoreError(
                    "not_enough_space",
                    f"Switching to {label} needs {format_size(download - available)} more free space on the drive of the model store: "
                    f"it downloads {format_size(download)}, and only {format_size(available)} are free.",
                )
            self._switching = variant
            self._reserved.update(file.key for file in files)
            return VariantSwitch(variant, tuple(files), tuple(missing), download, plans)

    def run_switch(
        self,
        switch: VariantSwitch,
        *,
        progress: Callable[[int, int, str | None], None] | None = None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> WgpConfigResult:
        """Download and verify the files of a prepared switch, register them, then rewrite `text_encoder_quantization`.

        One transaction places the files, replaces each installed quantization's references and size with those of
        the new variant and stores the variant. On a failure nothing changes and `ModelStoreError` is raised; the
        previous variant stays in use. The caller then restarts the worker and collects the previous variant's files.
        Returns what the `wgp_config.json` merge did.
        """
        try:
            with self._download_lock:
                ready = self._download(switch.label, switch.missing, progress or (lambda *_: None), cancelled)
            now = to_iso(self._clock())
            with self._lock, self.db.transaction() as conn:
                self._place(conn, switch.missing, ready, now)
                for key, files in switch.plans.items():
                    if (
                        conn.execute(
                            "SELECT 1 FROM installed_models WHERE model_id = ? AND quant = ? AND state = 'installed'", key
                        ).fetchone()
                        is None
                    ):
                        continue
                    conn.execute("DELETE FROM installed_model_files WHERE model_id = ? AND quant = ?", key)
                    conn.executemany(
                        "INSERT INTO installed_model_files (model_id, quant, path) VALUES (?, ?, ?)",
                        [(*key, file.path) for file in files],
                    )
                    conn.execute(
                        "UPDATE installed_models SET size_bytes = ? WHERE model_id = ? AND quant = ?",
                        (sum(file.size_bytes for file in files), *key),
                    )
                conn.execute("UPDATE hosts SET text_encoder_variant = ? WHERE id = ?", (switch.variant, self.host_id))
        except OSError as exc:
            log.exception("Switching to %s failed", switch.label)
            message = f"Writing the files of {switch.label} failed ({exc.strerror or exc}), so the text encoder was not switched."
            raise ModelStoreError("io_error", message) from exc
        finally:
            with self._lock:
                self._switching = None
                self._release(switch.files)
        return self.register_in_wan2gp()

    # ------------------------------------------------------------------ internals

    def _release(self, files: Iterable[PlannedFile]) -> None:
        for file in files:
            self._reserved[file.key] -= 1
            if self._reserved[file.key] <= 0:
                del self._reserved[file.key]

    def _running_jobs(self) -> bool:
        """Whether a job is leased to or running on a worker of this host."""
        row = self.db.query_one(
            "SELECT 1 FROM jobs WHERE state IN ('leased', 'running') AND json_extract(route, '$.host_id') = ? LIMIT 1", (self.host_id,)
        )
        return row is not None

    def _stored(self) -> dict[str, tuple[str, int]]:
        """Registered files: path key to (SHA-256, size)."""
        return {row["path"].casefold(): (row["sha256"], row["size_bytes"]) for row in self.db.query("SELECT * FROM store_files")}

    def _present(self, file: PlannedFile, stored: dict[str, tuple[str, int]]) -> bool:
        """Registered with this content and still on disk at its size."""
        if stored.get(file.key) != (file.sha256, file.size_bytes):
            return False
        try:
            return store_path(self.root, file.path).stat().st_size == file.size_bytes
        except OSError:
            return False

    def _staged_bytes(self, file: PlannedFile) -> int:
        """Bytes of `file` already in `.downloads/`: complete (still to be checked) or a resumable part."""
        staged = self.root / STAGING_DIR / file.sha256
        for path, complete in ((staged, True), (staged.with_name(staged.name + ".part"), False)):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if complete and size == file.size_bytes:
                return size
            if not complete and size < file.size_bytes:
                return size
        return 0

    def _kept(self, extra: Iterable[PlannedFile] = ()) -> set[str]:
        """Path keys that must stay although no installed quantization references them."""
        keep = {key for key, count in self._reserved.items() if count > 0}
        keep.update(file.key for file in extra)
        variant = self.text_encoder_variant()
        for model_id, quant in self.needed_by_jobs():
            try:
                keep.update(file.key for file in plan_files(self.registry, model_id, quant, variant))
            except ModelStoreError:
                continue
        return keep

    def _plan_eviction(self, key: InstallKey, files: list[PlannedFile], growth: int, label: str) -> tuple[list[InstallKey], int]:
        """What to evict so the store stays within its quota after growing by `growth`: (victims, bytes freed).

        Unreferenced files nothing needs go first, then the files of installed quantizations from the least recently
        used on, skipping this one, those being installed and those a waiting or running job needs. A shared file is
        freed only when its last reference goes. Refuses when even that does not make room.
        """
        quota = self._quota_bytes()
        stored = self.db.query("SELECT path, size_bytes FROM store_files")
        total = sum(row["size_bytes"] for row in stored)
        if quota is None or total + growth <= quota:
            return [], 0
        sizes = {row["path"].casefold(): row["size_bytes"] for row in stored}
        owners: dict[str, set[InstallKey]] = {path: set() for path in sizes}
        for row in self.db.query("SELECT model_id, quant, path FROM installed_model_files"):
            owners.setdefault(row["path"].casefold(), set()).add((row["model_id"], row["quant"]))
        keep = self._kept(files)
        freed = sum(sizes[path] for path, users in owners.items() if not users and path not in keep and path in sizes)
        needed = self.needed_by_jobs()
        victims: list[InstallKey] = []
        candidates = self.db.query(
            "SELECT model_id, quant FROM installed_models WHERE state = 'installed' "
            "ORDER BY COALESCE(last_used_at, installed_at, ''), model_id, quant"
        )
        for row in candidates:
            if total - freed + growth <= quota:
                break
            victim = (row["model_id"], row["quant"])
            if victim == key or victim in needed or victim in self._active:
                continue
            victims.append(victim)
            for path, users in owners.items():
                if victim in users:
                    users.discard(victim)
                    if not users and path not in keep:
                        freed += sizes.get(path, 0)
        if total - freed + growth > quota:
            raise ModelStoreError(
                "quota_exceeded",
                f"Installing {label} needs {format_size(total - freed + growth - quota)} more room within the model store quota of "
                f"{format_size(quota)}: it adds {format_size(growth)} to {format_size(total)}, and removing the least recently used "
                f"models that no queued or running job needs frees only {format_size(freed)}.",
            )
        return victims, freed

    def _unregister(self, key: InstallKey) -> None:
        """Drop the quantization's row; its references go with it (ON DELETE CASCADE)."""
        self.db.execute("DELETE FROM installed_models WHERE model_id = ? AND quant = ?", key)

    def _collect_garbage(self, extra: Iterable[PlannedFile] = ()) -> None:
        """Delete every registered file nothing references or needs. A file that cannot be deleted stays registered."""
        keep = self._kept(extra)
        rows = self.db.query("SELECT path FROM store_files WHERE path NOT IN (SELECT path FROM installed_model_files)")
        for row in rows:
            if row["path"].casefold() in keep:
                continue
            target = store_path(self.root, row["path"])
            try:
                target.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("Could not delete %s from the model store (%s); it stays registered.", row["path"], exc)
                continue
            self.db.execute("DELETE FROM store_files WHERE path = ?", (row["path"],))
            self._prune(target.parent)

    def _discard_downloads(self, files: Iterable[PlannedFile]) -> None:
        """Delete staged and partial downloads of `files` that no install in progress needs."""
        keep = {key for key, count in self._reserved.items() if count > 0}
        for file in files:
            if file.key in keep:
                continue
            staged = self.root / STAGING_DIR / file.sha256
            for path in (staged, staged.with_name(staged.name + ".part")):
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    log.warning("Could not delete the download %s (%s).", path.name, exc)

    def _prune(self, directory: Path) -> None:
        """Remove empty folders from `directory` up to, not including, the store root."""
        root = self.root.resolve()
        while directory.resolve() != root and directory.resolve().is_relative_to(root):
            try:
                directory.rmdir()
            except OSError:
                return
            directory = directory.parent

    def _engine_types(self, model: ModelDescriptor, quant: str) -> list[str]:
        return [binding.model_type for binding in model.engines if binding.engine == ENGINE_ID and binding.quant == quant]

    def _check_engine_types(self, model: ModelDescriptor, quant: str, label: str) -> None:
        """Refuse a model type the provisioned Wan2GP neither defines nor gets a finetune definition for."""
        root = self._engine_root()
        if root is None:
            return
        for model_type in self._engine_types(model, quant):
            if wan2gp_defines(root, model_type):
                continue
            if model_type not in self._architectures:
                raise ModelStoreError(
                    "engine_type_unknown",
                    f"WanGP does not define the model type {model_type} of {label}, and no architecture is known to define it with, "
                    "so nothing was installed.",
                )
            files = [q.files for q in model.quantizations if any(b.model_type == model_type and b.quant == q.id for b in model.engines)]
            if any(len(group) != 1 for group in files):
                raise ModelStoreError(
                    "engine_type_unknown",
                    f"A finetune definition for {model_type} of {label} would need several transformer files per quantization, "
                    "which Cinewright does not write, so nothing was installed.",
                )

    def _write_finetunes(self, model: ModelDescriptor, quant: str) -> None:
        root = self._engine_root()
        if root is None:
            return
        for model_type in self._engine_types(model, quant):
            if wan2gp_defines(root, model_type) or model_type not in self._architectures:
                continue
            bound = [b.quant for b in model.engines if b.engine == ENGINE_ID and b.model_type == model_type]
            files = [q.files[0].name for q in model.quantizations if q.id in bound]
            write_finetune(root, model_type, name=f"{model.name} ({model_type})", architecture=self._architectures[model_type], files=files)

    def _remove_finetunes(self, model: ModelDescriptor) -> None:
        """Delete the finetune definitions of the model's types that no installed quantization uses any more."""
        root = self._engine_root()
        if root is None:
            return
        installed = {row["quant"] for row in self.db.query("SELECT quant FROM installed_models WHERE model_id = ?", (model.id,))}
        for model_type in {b.model_type for b in model.engines if b.engine == ENGINE_ID}:
            if model_type not in self._architectures or wan2gp_defines(root, model_type):
                continue
            if any(b.model_type == model_type and b.quant in installed for b in model.engines):
                continue
            finetune_path(root, model_type).unlink(missing_ok=True)

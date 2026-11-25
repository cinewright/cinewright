"""The core: the implementation of `CoreService` (see `service.py`).

One `Core` per process start. It owns the database, the asset store, the scheduler, the event
log, the engine provisioner (`provisioning.py`) and the model store (`modelstore.py`), and generates
the enrollment token that worker processes need to register. The app's lifespan starts provisioning
when asked to. It either attaches a given supervisor of the local worker (`supervisor.py`), or, in
the installed app, lets the core start one on the provisioned engine as soon as the engine is ready
(`run_local_worker_when_ready`).

Model installs and text encoder switches are checked in the request and run in background threads;
their progress goes out as `model` system events. After a change of the model store the core restarts
the local worker, so it registers its new model list; while a job runs on it, once that job ended.
Store calls never run inside `_tx`: the lock order is the store's lock, then the database's.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Collection, Iterator, Sequence
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

import httpx
from pydantic import ValidationError

from cinewright import __version__
from cinewright.api.schemas import (
    API_PREFIX,
    Approval,
    ApprovalList,
    Asset,
    AssetList,
    CalibrationReport,
    CalibrationResult,
    CalibrationStatus,
    CapabilityEntry,
    CapabilityList,
    CastList,
    ClientKind,
    DataClass,
    EngineInfo,
    Estimate,
    EventKind,
    HardwareProfile,
    Health,
    Heartbeat,
    HeartbeatAck,
    Host,
    HostKind,
    HostList,
    HostPatch,
    InstallState,
    Job,
    JobCompletion,
    JobError,
    JobEvent,
    JobFailure,
    JobList,
    JobRequest,
    JobState,
    Lease,
    LeaseRequest,
    LicenseNotices,
    ModelDescriptor,
    ModelDetail,
    ModelInstall,
    ModelList,
    ModelQuantState,
    ModelStoreEviction,
    ModelStoreUsage,
    ModelSummary,
    OutputUploaded,
    PresetList,
    Progress,
    ProvenanceAsset,
    QuotaEviction,
    RecipeList,
    Route,
    Settings,
    SettingsPatch,
    SetupError,
    SystemEvent,
    SystemEventKind,
    SystemInfo,
    TextEncoderSwitch,
    WorkerEvents,
    WorkerEventsAck,
    WorkerHealth,
    WorkerInput,
    WorkerJob,
    WorkerRegistered,
    WorkerRegistration,
)
from cinewright.config import Config
from cinewright.core import dialect, hardware, router
from cinewright.core.assets import (
    BlobStore,
    BlobWriter,
    StagedBlob,
    asset_from_row,
    input_provenance,
    output_provenance,
    provenance_params,
    record_asset,
)
from cinewright.core.clock import Clock, from_iso, to_iso, utcnow
from cinewright.core.engines import load_engine_pins
from cinewright.core.errors import Conflict, CoreError, Infeasible, Invalid, NotFound, NotImplementedFeature, Unauthorized
from cinewright.core.events import EventLog, Published
from cinewright.core.licenses import WANGP_NAME, WANGP_NOTICE, license_notices
from cinewright.core.modelstore import (
    GB,
    Eviction,
    EvictionNeeded,
    Install,
    InstallProgress,
    ModelStore,
    ModelStoreError,
    StoreUsage,
    VariantSwitch,
    disk_free,
    plan_files,
)
from cinewright.core.provisioning import ENGINE_ID as WAN2GP
from cinewright.core.provisioning import EngineProvisioner, ProvisioningStatus, bundled_uv
from cinewright.core.registry import Registry
from cinewright.core.scheduler import ACTIVE_STATES, LEASE_SECONDS, Scheduler, Tx, WorkerRecord, transition
from cinewright.core.supervisor import WorkerLaunch, WorkerSupervisor
from cinewright.db.database import Database

log = logging.getLogger(__name__)

LOCAL_HOST_ID = "local"
LOCAL_HOST_NAME = "Local"
WORKER_STALE_SECONDS = 120.0  # a worker without contact for this long serves no new routes
WORKER_POLL_SECONDS = 1.0
MAINTENANCE_INTERVAL_S = 5.0
# Cores before migration 0002 recorded uploads as log events carrying this key; uploads now live in
# `job_outputs`. Workers still may not send it, so the old records stay distinguishable from worker data.
UPLOAD_RECORD_KEY = "output"
MODEL_PROGRESS_INTERVAL_S = 0.5  # at most two `model` progress events per second per install
BACKGROUND_JOIN_S = 30.0  # how long closing waits for installs and switches to stop
# Model store refusals by HTTP answer; every other code is a conflict with the current state (409).
_STORE_NOT_FOUND = frozenset({"unknown_model", "not_installed"})
_STORE_INVALID = frozenset({"unknown_quant", "unknown_variant"})
_STORE_RETRY = frozenset({"download_failed", "io_error", "install_failed"})  # installing again may help

Detector = Callable[[Path, str], HardwareProfile]

_JOB_SELECT = "SELECT jobs.*, agents.name AS agent_name FROM jobs LEFT JOIN agents ON agents.id = jobs.agent_id"
_STATUS_RANK = {CalibrationStatus.RUNS: 0, CalibrationStatus.SLOW: 1, CalibrationStatus.WONT_RUN: 2}


def _encode_cursor(created_at: str, key: str) -> str:
    return base64.urlsafe_b64encode(f"{created_at}|{key}".encode()).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[str, str]:
    try:
        created_at, key = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode().split("|", 1)
        from_iso(created_at)
    except ValueError as exc:
        raise Invalid("The cursor is not one this API issued.") from exc
    return created_at, key


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _quota_eviction(eviction: Eviction) -> QuotaEviction:
    return QuotaEviction(model_id=eviction.model_id, quant=eviction.quant, name=eviction.name, size_bytes=eviction.size_bytes)


def _store_refusal(exc: ModelStoreError) -> CoreError:
    """A model store refusal as the API answers it; the problem carries the store's `code`.

    `eviction_needed` also carries `evictions` (the planned removals, in removal order) and `frees_bytes`.
    """
    if exc.code in _STORE_NOT_FOUND:
        return NotFound(exc.message, code=exc.code)
    if exc.code in _STORE_INVALID:
        return Invalid(exc.message, code=exc.code)
    if isinstance(exc, EvictionNeeded):
        evictions = [_quota_eviction(eviction).model_dump(mode="json") for eviction in exc.evictions]
        return Conflict(exc.message, code=exc.code, evictions=evictions, frees_bytes=exc.frees_bytes)
    return Conflict(exc.message, code=exc.code)


class _Throttle:
    """Lets a progress report through at most every `interval` seconds; the first and the last always pass."""

    def __init__(self, interval: float = MODEL_PROGRESS_INTERVAL_S) -> None:
        self.interval = interval
        self.last: float | None = None

    def due(self, done: int, total: int) -> bool:
        now = time.monotonic()
        if self.last is None or done >= total or now - self.last >= self.interval:
            self.last = now
            return True
        return False


class _Upload(NamedTuple):
    """An output upload as `store_output` recorded it in `job_outputs`, including the checked media type."""

    sha256: str
    size_bytes: int
    media_type: str


class Core:
    def __init__(
        self,
        config: Config,
        *,
        registry: Registry | None = None,
        detect: Detector = hardware.detect,
        clock: Clock = utcnow,
        lease_seconds: float = LEASE_SECONDS,
        provisioner: EngineProvisioner | None = None,
        worker_supervisor: Callable[[WorkerLaunch], WorkerSupervisor] = WorkerSupervisor,
        download_transport: httpx.BaseTransport | None = None,
        free_disk: Callable[[Path], int] = disk_free,
    ) -> None:
        """`provisioner` defaults to provisioning the pinned Wan2GP with the installation's bundled uv.

        `worker_supervisor` builds the local worker's supervisor from the launch the provisioning record gives
        (`run_local_worker_when_ready`); tests swap in the mock engine and their own interpreter.
        `download_transport` and `free_disk` replace the network and the disk measurement of model installs (tests).
        """
        self.config = config
        self.paths = config.paths
        self.paths.ensure()
        self.clock = clock
        self._detect = detect
        self.registry = registry if registry is not None else Registry.load()
        self.db = Database.open(self.paths.db_file)
        self.blobs = BlobStore(self.paths)
        self.event_log = EventLog(self.db)
        self.scheduler = Scheduler(self.registry, lease_seconds=lease_seconds)
        # Per start; handed to spawned worker processes in CINEWRIGHT_ENROLLMENT_TOKEN.
        self.enrollment_token = secrets.token_urlsafe(32)
        self.started_at = clock()
        # The supervisor of the local worker process while there is one: given to the app, or started by the core.
        self.supervisor: WorkerSupervisor | None = None
        self._worker_supervisor = worker_supervisor
        self._worker_lock = threading.Lock()
        self._worker_core_url: str | None = None  # set while the core starts the local worker once the engine is ready
        self._own_supervisor: WorkerSupervisor | None = None  # the supervisor the core started itself
        self._closed = False
        self._leasing = True
        self._retired: set[str] = set()  # workers replaced by a restart of the local worker: they get no new job
        self.provisioner = (
            provisioner
            if provisioner is not None
            else EngineProvisioner(self.paths, load_engine_pins()[WAN2GP], uv=bundled_uv(), clock=clock)
        )
        self.provisioner.on_change = self._engine_changed
        self._ensure_local_host()
        self.models = ModelStore(
            self.db,
            self.registry,
            self.paths,
            host_id=LOCAL_HOST_ID,
            quota_bytes=self._quota_bytes,
            free_bytes=free_disk,
            transport=download_transport,
            engine_root=self._engine_root,
            clock=clock,
        )
        self._background: set[threading.Thread] = set()  # installs and text encoder switches
        self._background_lock = threading.Lock()
        self._restart_pending = False  # the model store changed while a job ran on the local worker
        self._restart_lock = threading.Lock()  # sets, checks and clears `_restart_pending` with the job check
        self._config_restarted = False  # restarted once since this start because WanGP's config was merged
        self._switch: TextEncoderSwitch | None = None  # the local host's text encoder switch: running, or failed
        self._last_eviction: ModelStoreEviction | None = None  # what the quota removed for the latest install that removed any

    def stop_leasing(self) -> None:
        """Grant no more leases, the first step of a shutdown; waiting long-polls return without a job."""
        self._leasing = False
        self.scheduler.wakeup.notify()

    def run_local_worker_when_ready(self, core_url: str) -> None:
        """Start the supervised local worker on the provisioned engine now if it is ready, else once a run reaches ready.

        For the installed app: the app's lifespan calls this with `provision`, after `start_provisioning`, so a
        start that tries the optional wheels again comes first. The launch comes from the provisioning record
        (`EngineProvisioner.worker_launch`). No worker starts while provisioning runs, nor after a failure, until a
        retry reaches ready. `stop_local_worker` ends this.
        """
        with self._worker_lock:
            self._worker_core_url = core_url
        self._start_local_worker()

    def stop_local_worker(self) -> None:
        """Stop the local worker the core started, if any, and start none from now on (shutdown)."""
        with self._worker_lock:
            self._worker_core_url = None
            supervisor, self._own_supervisor = self._own_supervisor, None
        if supervisor is not None:
            supervisor.stop()
            if self.supervisor is supervisor:
                self.supervisor = None

    def _start_local_worker(self) -> None:
        """Start the local worker if the core is to run one, runs none yet, and the engine is ready."""
        with self._worker_lock:
            core_url = self._worker_core_url
            if core_url is None or self.supervisor is not None or self._closed or not self._leasing:
                return
            launch = self.provisioner.worker_launch()
            if launch is None:
                return
            try:
                supervisor = self._worker_supervisor(launch)
                supervisor.start(core_url=core_url, enrollment_token=self.enrollment_token, paths=self.paths, clock=self.clock)
            except Exception:  # noqa: BLE001 - the core keeps serving; the Hosts view shows no worker
                log.exception("Starting the local worker on %s failed", launch.wan2gp_root)
                return
            self._own_supervisor = self.supervisor = supervisor
        log.info("The engine is ready; starting the local worker with %s.", launch.python)

    def restart_local_worker(self) -> None:
        """Restart the supervised local worker, so it registers again with its current model list.

        For model installs and removals (#3): a worker sends its model list only when it registers.
        First the running worker process is retired: its pending lease long-poll returns at once and it gets
        no new job. An idle worker waits in a long-poll longer than the stop's grace period, so it would
        otherwise be killed instead of stopping on request, and a job it took in that window would wait for
        its lease to expire while the new worker idles. Does nothing while the core supervises no worker.
        """
        supervisor = self.supervisor
        if supervisor is None:
            return
        status = supervisor.status()
        if status.state == "running" and status.since is not None:
            rows = self.db.query("SELECT id FROM workers WHERE host_id = ? AND registered_at >= ?", (LOCAL_HOST_ID, to_iso(status.since)))
            self._retired.update(row["id"] for row in rows)
            self.scheduler.wakeup.notify()
        supervisor.restart()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True  # also cancels installs and switches, whose downloads continue at the next try
        deadline = time.monotonic() + BACKGROUND_JOIN_S
        with self._background_lock:
            running = list(self._background)
        for thread in running:
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                log.error("%s did not stop in time.", thread.name)
        self.stop_local_worker()
        self.provisioner.stop()
        self.event_log.close()
        self.scheduler.wakeup.notify()
        self.db.close()

    # ------------------------------------------------------------------ internals

    @contextmanager
    def _tx(self) -> Iterator[Tx]:
        """Write transaction; its events are published and waiting workers woken after commit.

        The event log's lock spans the commit and the publish, so subscribers see events in commit order.
        """
        published: list[Published] = []
        now = self.clock()
        with self.event_log.lock:
            with self.db.transaction() as conn:
                tx = Tx(
                    conn,
                    now,
                    emit=lambda job_id, kind, data: published.append(EventLog.append(conn, job_id, kind, data, now)),
                    broadcast=lambda job_id, kind, data: published.append(EventLog.live(conn, job_id, kind, data, now)),
                )
                yield tx
            self.event_log.publish(published)
        if tx.wake_workers:
            self.scheduler.wakeup.notify()

    def _ensure_local_host(self) -> None:
        now = to_iso(self.clock())
        self.db.execute(
            "INSERT OR IGNORE INTO hosts (id, name, kind, allowed_data_classes, enrolled_at, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (LOCAL_HOST_ID, LOCAL_HOST_NAME, HostKind.LOCAL.value, json.dumps([c.value for c in DataClass]), now, now),
        )

    def _hosts(self) -> list[router.HostInfo]:
        rows = self.db.query("SELECT * FROM hosts WHERE revoked_at IS NULL ORDER BY created_at, id")
        return [
            router.HostInfo(
                id=row["id"],
                name=row["name"],
                kind=HostKind(row["kind"]),
                allowed_data_classes=frozenset(DataClass(c) for c in json.loads(row["allowed_data_classes"])),
            )
            for row in rows
        ]

    def _online_workers(self) -> list[router.WorkerInfo]:
        cutoff = to_iso(self.clock() - timedelta(seconds=WORKER_STALE_SECONDS))
        rows = self.db.query("SELECT * FROM workers WHERE last_seen_at >= ? ORDER BY registered_at", (cutoff,))
        rows = [row for row in rows if WorkerHealth.model_validate_json(row["health"]).state != "failed"]
        # A worker retired by a restart (after a model change) still routes until its replacement registered, so jobs
        # submitted meanwhile wait for the replacement; from then on its model list from before the change routes nothing.
        replaced = {row["host_id"]: row["registered_at"] for row in rows if row["id"] not in self._retired}
        return [
            router.WorkerInfo(id=row["id"], host_id=row["host_id"], engine=row["engine"], models=frozenset(json.loads(row["models"])))
            for row in rows
            if row["id"] not in self._retired or replaced.get(row["host_id"], "") < row["registered_at"]
        ]

    def _worker_record(self, conn: sqlite3.Connection, worker_id: str) -> WorkerRecord:
        row = conn.execute("SELECT * FROM workers WHERE id = ?", (worker_id,)).fetchone()
        if row is None:
            raise Unauthorized("Unknown worker.")
        return WorkerRecord(
            id=row["id"],
            host_id=row["host_id"],
            engine=row["engine"],
            engine_version=row["engine_version"],
            models=frozenset(json.loads(row["models"])),
        )

    def _calibration_rows(self) -> list[CalibrationResult]:
        rows = self.db.query("SELECT * FROM calibration_results ORDER BY host_id, model_id, quant, resolution")
        return [
            CalibrationResult(
                host_id=row["host_id"],
                model_id=row["model_id"],
                quant=row["quant"],
                resolution=row["resolution"],
                status=CalibrationStatus(row["status"]),
                peak_vram_mb=row["peak_vram_mb"],
                seconds_per_step=row["seconds_per_step"],
                ram_offload_mb=row["ram_offload_mb"],
                detail=row["detail"],
                engine=row["engine"],
                engine_version=row["engine_version"],
                measured_at=from_iso(row["measured_at"]),
            )
            for row in rows
        ]

    def _stored_hardware(self, host_id: str) -> HardwareProfile | None:
        row = self.db.query_one("SELECT profile FROM hardware_profiles WHERE host_id = ?", (host_id,))
        return HardwareProfile.model_validate_json(row["profile"]) if row else None

    def _gpu_name(self, host_id: str) -> str | None:
        profile = self._stored_hardware(host_id)
        return profile.gpus[0].name if profile and profile.gpus else None

    def _require_model(self, model_id: str) -> ModelDescriptor:
        model = self.registry.get(model_id)
        if model is None:
            raise NotFound(f"Model {model_id} is not in the registry.")
        return model

    # ------------------------------------------------------------------ system

    def system_info(self) -> SystemInfo:
        return SystemInfo(version=__version__, data_dir=str(self.paths.root), bind=self.config.bind, engines=[self._wan2gp_info()])

    def _wan2gp_info(self, status: ProvisioningStatus | None = None) -> EngineInfo:
        """The engine as provisioning reports it; once `ready`, `version` is the pinned commit."""
        status = status if status is not None else self.provisioner.status()
        return EngineInfo(
            id=WAN2GP,
            name=WANGP_NAME,
            version=status.version,
            state=status.state,
            notice=WANGP_NOTICE,
            progress=status.progress,
            error=status.error,
        )

    def _engine_changed(self, status: ProvisioningStatus) -> None:
        """Provisioning moved on (from its thread): publish the engine's new state; once ready, start the local worker.

        The state goes out as a live system event to unfiltered streams only; it is not stored, kept or replayed.
        """
        if not self._closed:
            self.event_log.publish_system(
                SystemEvent(kind=SystemEventKind.ENGINE, engine=self._wan2gp_info(status), created_at=self.clock())
            )
        if status.state == "ready":
            self._start_local_worker()

    def start_provisioning(self) -> None:
        """Provision the engine in the background unless it is ready; the app's lifespan calls this at start.

        A ready engine whose optional wheels could not be downloaded earlier tries them again first, unless a
        local worker already runs on its environment.
        """
        if not self.provisioner.start() and self.supervisor is None:
            self.provisioner.retry_optional()

    def provision_engine(self, engine_id: str) -> EngineInfo:
        if engine_id != WAN2GP:
            raise NotFound(f"Engine {engine_id} is not known; the only engine is {WAN2GP}.")
        self.provisioner.start()
        return self._wan2gp_info()

    def health(self) -> Health:
        checks: dict[str, str] = {}
        try:
            self.db.query_one("SELECT 1")
            checks["database"] = "ok"
        except sqlite3.Error as exc:
            checks["database"] = f"failed: {exc}"
        checks["data_dir"] = "ok" if os.access(self.paths.root, os.W_OK) else "not writable"
        checks["workers"] = f"{len(self._online_workers())} online"
        ok = checks["database"] == "ok" and checks["data_dir"] == "ok"
        return Health(status="ok" if ok else "degraded", checks=checks)

    def license_notices(self) -> LicenseNotices:
        return license_notices()

    def get_settings(self) -> Settings:
        stored = {row["key"]: json.loads(row["value"]) for row in self.db.query("SELECT key, value FROM settings")}
        return Settings.model_validate({k: v for k, v in stored.items() if k in Settings.model_fields})

    def update_settings(self, patch: SettingsPatch) -> Settings:
        changes = patch.model_dump(exclude_unset=True)
        try:
            updated = Settings.model_validate({**self.get_settings().model_dump(), **changes})
        except ValidationError as exc:
            raise Invalid(f"Settings rejected: {exc.errors()[0]['msg']}.") from exc
        with self.db.transaction() as conn:
            for key in changes:
                conn.execute(
                    "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                    (key, json.dumps(getattr(updated, key))),
                )
        return updated

    # ------------------------------------------------------------------ hardware

    def hardware(self) -> HardwareProfile:
        return self._stored_hardware(LOCAL_HOST_ID) or self.detect_hardware()

    def detect_hardware(self) -> HardwareProfile:
        profile = self._detect(self.paths.root, LOCAL_HOST_ID)
        self.db.execute(
            "INSERT INTO hardware_profiles (host_id, profile, detected_at) VALUES (?, ?, ?) "
            "ON CONFLICT (host_id) DO UPDATE SET profile = excluded.profile, detected_at = excluded.detected_at",
            (LOCAL_HOST_ID, profile.model_dump_json(), to_iso(profile.detected_at)),
        )
        return profile

    def ensure_hardware(self) -> None:
        """Detect the local hardware on first start."""
        if self._stored_hardware(LOCAL_HOST_ID) is None:
            self.detect_hardware()

    # ------------------------------------------------------------------ registry and calibration

    def capabilities(self) -> CapabilityList:
        country = self.get_settings().country
        calibration = self._calibration_rows()
        hosts = self._hosts()
        items = []
        for model in self.registry.all():
            excluded = router.geo_excluded(model.license.geo_exclusions, country)
            for capability in model.capabilities:
                for quant in model.quantizations:
                    rows = [c for c in calibration if c.model_id == model.id and c.quant == quant.id]
                    status = min((r.status for r in rows), key=_STATUS_RANK.__getitem__, default=None)
                    allowed = [] if excluded else [h.id for h in hosts if not _wont_run_everywhere(rows, h.id)]
                    items.append(
                        CapabilityEntry(
                            capability=capability,
                            model_id=model.id,
                            model_name=model.name,
                            quant=quant.id,
                            limits=model.limits,
                            status=status,
                            hosts=allowed,
                        )
                    )
        return CapabilityList(items=items)

    def _quant_states(self, model: ModelDescriptor) -> list[ModelQuantState]:
        installed = {row["quant"]: row for row in self.db.query("SELECT * FROM installed_models WHERE model_id = ?", (model.id,))}
        failed = InstallState.FAILED.value
        calibration = [c for c in self._calibration_rows() if c.model_id == model.id]
        states = []
        for quant in model.quantizations:
            row = installed.get(quant.id)
            sizes = [f.size_bytes for f in quant.files]
            size = row["size_bytes"] if row is not None and row["size_bytes"] is not None else None
            if size is None and all(s is not None for s in sizes):
                size = sum(sizes)
            states.append(
                ModelQuantState(
                    quant=quant.id,
                    install_state=InstallState(row["state"]) if row is not None else InstallState.ABSENT,
                    size_bytes=size,
                    calibration=[c for c in calibration if c.quant == quant.id],
                    error=row["error"] if row is not None and row["state"] == failed else None,
                )
            )
        return states

    def _summary(self, model: ModelDescriptor, usage: StoreUsage) -> ModelSummary:
        return ModelSummary(
            id=model.id,
            name=model.name,
            vendor=model.vendor,
            family=model.family,
            capabilities=model.capabilities,
            license=model.license,
            quantizations=self._quant_states(model),
            installed_bytes=usage.by_model.get(model.id, 0),
        )

    def list_models(self) -> ModelList:
        """The registry's models without those whose licence excludes the configured country, and the model store's usage."""
        country = self.get_settings().country
        usage = self.models.usage()
        offered = [model for model in self.registry.all() if not router.geo_excluded(model.license.geo_exclusions, country)]
        return ModelList(
            items=[self._summary(model, usage) for model in offered],
            store=ModelStoreUsage(
                total_bytes=usage.total_bytes,
                quota_bytes=usage.quota_bytes,
                free_bytes=usage.free_bytes,
                last_eviction=self._last_eviction,
            ),
        )

    def get_model(self, model_id: str) -> ModelDetail:
        model = self._require_model(model_id)
        return ModelDetail(
            **self._summary(model, self.models.usage()).model_dump(),
            limits=model.limits,
            prompt_dialect=model.prompt_dialect,
            defaults=model.defaults,
            source=model.source,
            engines=model.engines,
        )

    def install_model(self, model_id: str, quant: str | None, evict: Collection[tuple[str, str]] = ()) -> None:
        """Check and prepare the install (refusals raise here), then download and register it in the background.

        `evict` names the installed quantizations (model id, quantization) the client agrees to remove before the
        download. Only an install that fits on the store's drive only after the quota's removals reads it; when it
        does not name each of them, the install is refused with `eviction_needed`, which lists them.
        """
        model = self._require_model(model_id)
        choices = ", ".join(q.id for q in model.quantizations)
        if quant is None:
            raise Invalid(f"Name the quantization of {model.name} to install with ?quant= (one of {choices}).")
        if all(q.id != quant for q in model.quantizations):
            raise Invalid(f"{model.name} has no quantization {quant}; choose one of {choices}.", code="unknown_quant")
        country = self.get_settings().country
        if router.geo_excluded(model.license.geo_exclusions, country):
            raise Conflict(
                f"The licence of {model.name} ({model.license.name}) excludes {country}, so it was not installed.", code="licence"
            )
        try:
            install = self.models.prepare_install(model_id, quant, evict=evict)
        except ModelStoreError as exc:
            raise _store_refusal(exc) from None
        if install.evictions:  # confirmed removals before the download; they stay removed if it fails
            self._record_eviction(install, ())
        for victim in install.evicted:
            self._publish_model(*victim, InstallState.ABSENT)
        if install.evicted:
            self._restart_for_models()
        self._in_background(f"install {model_id} {quant}", lambda: self._run_install(install))

    def remove_model(self, model_id: str, quant: str | None) -> None:
        self._require_model(model_id)
        rows = self.db.query("SELECT quant FROM installed_models WHERE model_id = ? AND state = 'installed'", (model_id,))
        installed = {row["quant"] for row in rows}
        try:
            removed = self.models.remove(model_id, quant)
        except ModelStoreError as exc:
            raise _store_refusal(exc) from None
        for key in removed:
            self._publish_model(*key, InstallState.ABSENT)
        if any(key[1] in installed for key in removed):
            self._restart_for_models()

    def calibration(self) -> CalibrationReport:
        return CalibrationReport(items=self._calibration_rows())

    def start_calibration(self) -> None:
        raise NotImplementedFeature("Calibration runs", "phase 1")

    # ------------------------------------------------------------------ model store

    def _quota_bytes(self) -> int | None:
        quota = self.get_settings().model_store_quota_gb
        return quota * GB if quota is not None else None

    def _engine_root(self) -> Path | None:
        launch = self.provisioner.worker_launch()
        return launch.wan2gp_root if launch is not None else None

    def _in_background(self, name: str, target: Callable[[], None]) -> None:
        """Run `target` in a thread that `close` waits for; it ends early once `_closed` is set."""

        def run() -> None:
            try:
                target()
            except Exception:  # noqa: BLE001 - the core keeps serving; the log has the details
                log.exception("%s failed", name)
            finally:
                with self._background_lock:
                    self._background.discard(thread)

        thread = threading.Thread(target=run, name=name, daemon=True)
        with self._background_lock:
            self._background.add(thread)
        thread.start()

    def _record_eviction(self, install: Install, removed: Sequence[Eviction]) -> None:
        """Keep what the quota removed for `install` for `GET /models` (`store.last_eviction`), replacing an earlier record.

        Holds the removals before its download and `removed`, those of its quota step. Set before the `absent` and
        `installed` events go out, so the reload they trigger sees it.
        """
        evicted = [_quota_eviction(eviction) for eviction in (*install.evictions, *removed)]
        self._last_eviction = ModelStoreEviction(model_id=install.model_id, quant=install.quant, evicted=evicted, at=self.clock())

    def _publish_model(
        self,
        model_id: str,
        quant: str,
        state: InstallState,
        *,
        downloaded: int | None = None,
        total: int | None = None,
        error: SetupError | None = None,
    ) -> None:
        """Send a quantization's new install state as a live `model` system event (unfiltered streams only)."""
        if self._closed:
            return
        progress = None
        if state == InstallState.DOWNLOADING and total is not None and downloaded is not None:
            progress = Progress(phase="download", percent=round(100.0 * downloaded / total, 1) if total else 100.0)
        install = ModelInstall(
            model_id=model_id,
            quant=quant,
            install_state=state,
            progress=progress,
            downloaded_bytes=downloaded,
            total_bytes=total,
            error=error,
        )
        self.event_log.publish_system(SystemEvent(kind=SystemEventKind.MODEL, model=install, created_at=self.clock()))

    def _run_install(self, install: Install) -> None:
        throttle = _Throttle()

        def progress(update: InstallProgress) -> None:
            if throttle.due(update.done_bytes, update.total_bytes):
                state = InstallState.DOWNLOADING
                self._publish_model(update.model_id, update.quant, state, downloaded=update.done_bytes, total=update.total_bytes)

        try:
            merged = self.models.run_install(install, progress=progress, cancelled=lambda: self._closed)
        except ModelStoreError as exc:
            if exc.code != "config_invalid":  # a cancellation (Cinewright closes) too: `failed`, so Remove deletes its downloads
                log.warning("%s", exc.message)
                error = SetupError(code=exc.code, message=exc.message, action="retry" if exc.code in _STORE_RETRY else None)
                self._publish_model(*install.key, InstallState.FAILED, error=error)
                return  # a failed or cancelled download removes nothing for the quota
            log.error("%s", exc.message)  # registered, but WanGP does not see the store until its config is fixed
            merged = "unchanged"
        except OSError as exc:  # only the config merge raises it, after the files were registered
            log.error("Installed %s, but writing WanGP's configuration failed (%s), so WanGP does not see it yet.", install.label, exc)
            merged = "unchanged"
        else:
            log.info("Installed %s.", install.label)
        removed = self.models.evict_for_quota(install)  # registered: now the quota may remove the least recently used
        if removed:
            self._record_eviction(install, removed)
        self._publish_model(*install.key, InstallState.INSTALLED)
        for victim in removed:
            self._publish_model(*victim.key, InstallState.ABSENT)
        if install.missing or merged == "updated" or removed:
            self._restart_for_models()

    def _run_switch(self, switch: VariantSwitch) -> None:
        """Run a prepared switch; `GET /hosts` shows it as the local host's `text_encoder_switch` until it succeeded or failed."""
        try:
            merged = self.models.run_switch(switch, cancelled=lambda: self._closed)
        except ModelStoreError as exc:
            if exc.code == "cancelled":
                self._switch = None
                return
            if exc.code == "config_invalid":  # both variants stay until WanGP's config names the new one
                log.error("%s", exc.message)
            else:
                log.warning("Switching the text encoder to %s failed: %s", switch.variant, exc.message)
            action = "retry" if exc.code in _STORE_RETRY else None
            self._switch = TextEncoderSwitch(variant=switch.variant, error=SetupError(code=exc.code, message=exc.message, action=action))
            return
        except Exception:  # noqa: BLE001 - whatever stopped it, the view must learn that the switch ended
            log.exception("Switching the text encoder to %s failed", switch.variant)
            message = (
                f"The switch to the {switch.variant} text encoder stopped with an unexpected error (details in logs/cinewright.log), "
                "so the previous variant stays in use."
            )
            self._switch = TextEncoderSwitch(variant=switch.variant, error=SetupError(code="install_failed", message=message))
            return
        self._switch = None
        log.info("Switched the text encoder to %s.", switch.variant)
        restarts = bool(switch.missing or merged == "updated") and self.supervisor is not None and not self._closed
        if restarts:
            self._restart_for_models()
        # The previous variant's files go once the worker's replacement registered (`_local_worker_registered`): until it
        # stopped, the old worker may hold them open, and a job leased during the download still runs with them.
        if not restarts and not self._local_job_active():
            self.models.collect_garbage()

    def _local_job_active(self) -> bool:
        row = self.db.query_one(
            "SELECT 1 FROM jobs WHERE state IN (?, ?) AND json_extract(route, '$.host_id') = ? LIMIT 1", (*ACTIVE_STATES, LOCAL_HOST_ID)
        )
        return row is not None

    def _restart_for_models(self) -> None:
        """Restart the local worker so it registers its current model list; while it runs a job, once that job ended.

        A restart stops a running job (it returns to the queue), so a model change waits for the job.
        """
        if self._closed or self.supervisor is None:
            return
        with self._restart_lock:
            if not self._restart_pending and self._local_job_active():
                log.info("The model store changed; the local worker restarts once its job ended.")
            self._restart_pending = True
        self._restart_if_pending()

    def _restart_if_pending(self) -> None:
        """Restart the local worker if a model change waits for it and no job runs on it any more.

        Job ends call this after their transaction, and so does maintenance on every tick. The flag is set before the job
        check, and checking and clearing it share one lock, so a job that ends while another thread decides still leads
        to exactly one restart.
        """
        if not self._restart_pending:  # read without the lock; a set flag is checked again under it
            return
        with self._restart_lock:
            if not self._restart_pending or self._closed or self.supervisor is None or self._local_job_active():
                return
            self._restart_pending = False
        self.restart_local_worker()

    def _local_worker_registered(self) -> None:
        """A local worker registered: register the store in WanGP's config and collect files left for deletion.

        WanGP writes its config at its first start, so an install before that could not merge into it; a merge that
        changes the config restarts the worker once per core start, since WanGP reads its config only at import.
        After a text encoder switch this collection removes the previous variant's files. A file the previous worker
        held open, or one the last core start left, could not be deleted before this worker built its model list; once
        this collection deletes such a file, the worker restarts again to register without it.
        """
        try:
            merged = self.models.register_in_wan2gp() if self.models.has_installs() else "unchanged"
            retried = self.models.collect_garbage()
        except (ModelStoreError, OSError) as exc:
            log.warning("Updating the model store after the local worker registered failed: %s", exc)
            return
        if merged == "updated" and not self._config_restarted and self.supervisor is not None:
            self._config_restarted = True
            log.info("Registered the model store in WanGP's configuration; restarting the local worker so WanGP reads it.")
            self._restart_for_models()
        elif retried:
            log.info("Deleted model files the previous local worker held open; restarting the local worker so it registers without them.")
            self._restart_for_models()

    # ------------------------------------------------------------------ jobs

    def _data_class(self, request: JobRequest) -> DataClass:
        """The strictest data class of the request and its input assets."""
        classes = [request.data_class]
        for sha256 in dict.fromkeys(request.inputs):
            row = self.db.query_one("SELECT data_class FROM assets WHERE sha256 = ?", (sha256,))
            if row is None:
                raise Invalid(f"Input asset {sha256} does not exist; upload it first.")
            classes.append(DataClass(row["data_class"]))
        return router.strictest(classes)

    def _plan(self, request: JobRequest) -> tuple[router.Plan, DataClass]:
        if request.recipe is not None:
            raise NotImplementedFeature("Recipes", "phase 2")
        data_class = self._data_class(request)
        plan = router.plan(
            request,
            data_class,
            self.registry.all(),
            self._hosts(),
            self._online_workers(),
            self._calibration_rows(),
            self.get_settings(),
            installed={LOCAL_HOST_ID: self.models.installed()},
        )
        return plan, data_class

    def estimate(self, request: JobRequest) -> Estimate:
        plan, _ = self._plan(request)
        return Estimate(
            feasible=plan.feasible,
            route=plan.route,
            eta_seconds=plan.eta_seconds,
            cost_eur=plan.cost_eur,
            requires_approval=plan.requires_approval,
            reasons=plan.reasons,
        )

    def submit_job(self, request: JobRequest, client: ClientKind, agent: str | None) -> Job:
        # Store lock before the transaction: no removal or eviction takes the routed quantization away until it is queued.
        with self.models.routing():
            job_id = self._queue_job(request, client, agent)
        return self.get_job(job_id)

    def _queue_job(self, request: JobRequest, client: ClientKind, agent: str | None) -> str:
        """Route the request and store the job; returns its id."""
        plan, data_class = self._plan(request)
        if plan.route is None:
            raise Infeasible("No model and host can run this request; the reasons list each candidate.", plan.reasons)
        model = self._require_model(plan.route.model_id)
        seed = request.params.seed if request.params.seed is not None else secrets.randbelow(2**31)
        router.effective_params(request, model, seed)  # registry defaults must form valid parameters before queueing
        job_id = uuid4().hex
        state = JobState.AWAITING_APPROVAL if plan.requires_approval else JobState.QUEUED
        with self._tx() as tx:
            agent_id = self._agent_id(tx, agent) if agent else None
            tx.conn.execute(
                "INSERT INTO jobs (id, state, priority, capability, data_class, request, route, seed, client, agent_id, eta_seconds, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    state.value,
                    request.priority.value,
                    request.capability.value,
                    data_class.value,
                    request.model_dump_json(),
                    plan.route.model_dump_json(),
                    seed,
                    client.value,
                    agent_id,
                    plan.eta_seconds,
                    to_iso(tx.now),
                ),
            )
            if plan.requires_approval:
                tx.conn.execute(
                    "INSERT INTO approvals (id, job_id, reason, estimated_cost_eur, host_id, state, created_at) "
                    "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                    (
                        uuid4().hex,
                        job_id,
                        f"Runs on {plan.route.host_id}, which is not local; cost above the approval threshold or unknown.",
                        plan.cost_eur,
                        plan.route.host_id,
                        to_iso(tx.now),
                    ),
                )
            tx.emit(job_id, EventKind.STATE, {"state": state.value})
            tx.wake_workers = state == JobState.QUEUED
        return job_id

    @staticmethod
    def _agent_id(tx: Tx, name: str) -> str:
        row = tx.conn.execute("SELECT id FROM agents WHERE name = ?", (name,)).fetchone()
        if row is not None:
            return row["id"]
        agent_id = uuid4().hex
        # Agent tokens arrive with MCP agent identities; until then the record only names the agent.
        tx.conn.execute("INSERT INTO agents (id, name, token_hash, created_at) VALUES (?, ?, '', ?)", (agent_id, name, to_iso(tx.now)))
        return agent_id

    def _job_outputs(self, job_ids: list[str]) -> dict[str, list[str]]:
        """Output hashes of the given jobs in completion order; only a completed output has a position."""
        rows = self.db.query(
            "SELECT job_id, position, sha256 FROM job_outputs WHERE position IS NOT NULL AND job_id IN (SELECT value FROM json_each(?))",
            (json.dumps(job_ids),),
        )
        outputs: dict[str, list[str]] = {}
        for row in sorted(rows, key=lambda r: (r["job_id"], r["position"])):
            outputs.setdefault(row["job_id"], []).append(row["sha256"])
        return outputs

    @staticmethod
    def _job(row: sqlite3.Row, outputs: list[str]) -> Job:
        return Job(
            id=row["id"],
            state=JobState(row["state"]),
            request=JobRequest.model_validate_json(row["request"]),
            route=Route.model_validate_json(row["route"]) if row["route"] else None,
            progress=Progress.model_validate_json(row["progress"]) if row["progress"] else None,
            eta_seconds=row["eta_seconds"],
            client=ClientKind(row["client"]),
            agent=row["agent_name"],
            outputs=outputs if row["state"] == JobState.SUCCEEDED.value else [],
            error=JobError.model_validate_json(row["error"]) if row["error"] else None,
            created_at=from_iso(row["created_at"]),
            started_at=from_iso(row["started_at"]),
            finished_at=from_iso(row["finished_at"]),
        )

    def list_jobs(self, state: JobState | None, cursor: str | None, limit: int) -> JobList:
        where: list[str] = []
        params: list[object] = []
        if state is not None:
            where.append("jobs.state = ?")
            params.append(state.value)
        if cursor:
            created_at, job_id = _decode_cursor(cursor)
            where.append("(jobs.created_at < ? OR (jobs.created_at = ? AND jobs.id < ?))")
            params += [created_at, created_at, job_id]
        sql = _JOB_SELECT + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY jobs.created_at DESC, jobs.id DESC LIMIT ?"
        rows = self.db.query(sql, [*params, limit + 1])
        next_cursor = _encode_cursor(rows[limit - 1]["created_at"], rows[limit - 1]["id"]) if len(rows) > limit else None
        outputs = self._job_outputs([row["id"] for row in rows[:limit]])
        return JobList(items=[self._job(row, outputs.get(row["id"], [])) for row in rows[:limit]], next_cursor=next_cursor)

    def get_job(self, job_id: str) -> Job:
        row = self.db.query_one(_JOB_SELECT + " WHERE jobs.id = ?", (job_id,))
        if row is None:
            raise NotFound(f"Job {job_id} does not exist.")
        return self._job(row, self._job_outputs([job_id]).get(job_id, []))

    def cancel_job(self, job_id: str) -> Job:
        with self._tx() as tx:
            row = tx.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise NotFound(f"Job {job_id} does not exist.")
            state = JobState(row["state"])
            if state in (JobState.QUEUED, JobState.AWAITING_APPROVAL):
                transition(tx, job_id, JobState.CANCELLED, {"reason": "cancelled"}, finished_at=to_iso(tx.now))
            elif state in (JobState.LEASED, JobState.RUNNING):
                if not row["cancel_requested"]:
                    tx.conn.execute("UPDATE jobs SET cancel_requested = 1 WHERE id = ?", (job_id,))
                    tx.emit(job_id, EventKind.LOG, {"message": "Cancel requested; the worker stops the job at its next check."})
            elif state != JobState.CANCELLED:
                raise Conflict(f"Job {job_id} already {state.value}, so it cannot be cancelled.")
        return self.get_job(job_id)

    def events(self, job_id: str | None, after: str | None) -> AsyncGenerator[JobEvent | SystemEvent, None]:
        return self.event_log.stream(job_id, after)

    # ------------------------------------------------------------------ assets

    def stage_upload(self) -> BlobWriter:
        return self.blobs.writer()

    def add_asset(self, upload: StagedBlob, media_type: str, data_class: DataClass) -> Asset:
        sha256 = self.blobs.place(upload)
        now = self.clock()
        with self.db.transaction() as conn:
            row = record_asset(conn, input_provenance(sha256, media_type, upload.size_bytes, now), data_class, None, now)
        return asset_from_row(row)

    def list_assets(self, cursor: str | None, limit: int) -> AssetList:
        if cursor:
            created_at, sha256 = _decode_cursor(cursor)
            rows = self.db.query(
                "SELECT * FROM assets WHERE created_at < ? OR (created_at = ? AND sha256 < ?) "
                "ORDER BY created_at DESC, sha256 DESC LIMIT ?",
                (created_at, created_at, sha256, limit + 1),
            )
        else:
            rows = self.db.query("SELECT * FROM assets ORDER BY created_at DESC, sha256 DESC LIMIT ?", (limit + 1,))
        next_cursor = _encode_cursor(rows[limit - 1]["created_at"], rows[limit - 1]["sha256"]) if len(rows) > limit else None
        return AssetList(items=[asset_from_row(row) for row in rows[:limit]], next_cursor=next_cursor)

    def get_asset(self, sha256: str) -> Asset:
        row = self.db.query_one("SELECT * FROM assets WHERE sha256 = ?", (sha256,))
        if row is None:
            raise NotFound(f"Asset {sha256} does not exist.")
        return asset_from_row(row)

    def asset_path(self, sha256: str) -> Path:
        if not self.blobs.exists(sha256):
            raise NotFound(f"The file of asset {sha256} is missing from the asset store.")
        return self.blobs.path(sha256)

    # ------------------------------------------------------------------ hosts, approvals, libraries

    def _worker_health(self) -> dict[str, WorkerHealth]:
        """Per host, the health its most recently seen worker reported, among workers registered since this start.

        Rows of earlier starts stay in the database. On the local host the supervisor's view of the process
        wins while the process is starting or down, or runs but has not registered yet.
        """
        cutoff = to_iso(self.clock() - timedelta(seconds=WORKER_STALE_SECONDS))
        rows = self.db.query(
            "SELECT host_id, health, registered_at FROM workers WHERE last_seen_at >= ? AND registered_at >= ? "
            "ORDER BY last_seen_at, registered_at",
            (cutoff, to_iso(self.started_at)),
        )
        latest = {row["host_id"]: row for row in rows}  # ascending order: the last row of a host wins
        health = {host_id: WorkerHealth.model_validate_json(row["health"]) for host_id, row in latest.items()}
        supervisor = self.supervisor
        if supervisor is not None:
            status = supervisor.status()
            local = latest.get(LOCAL_HOST_ID)
            if status.state != "running":
                health[LOCAL_HOST_ID] = WorkerHealth(state="starting" if status.state == "starting" else "failed", detail=status.detail)
            elif local is None or (status.since is not None and from_iso(local["registered_at"]) < status.since):
                health[LOCAL_HOST_ID] = WorkerHealth(state="starting", detail="The worker process runs and has not registered yet.")
        return health

    def list_hosts(self) -> HostList:
        online = {w.host_id for w in self._online_workers()}
        worker_health = self._worker_health()
        last_seen = {
            row["host_id"]: row["seen"] for row in self.db.query("SELECT host_id, MAX(last_seen_at) AS seen FROM workers GROUP BY host_id")
        }
        variants = {row["id"]: row["text_encoder_variant"] for row in self.db.query("SELECT id, text_encoder_variant FROM hosts")}
        items = []
        for host in self._hosts():
            items.append(
                Host(
                    id=host.id,
                    name=host.name,
                    kind=host.kind,
                    allowed_data_classes=sorted(host.allowed_data_classes, key=list(DataClass).index),
                    online=host.kind == HostKind.LOCAL or host.id in online,
                    last_seen_at=from_iso(last_seen.get(host.id)),
                    hardware=self._stored_hardware(host.id),
                    worker_health=worker_health.get(host.id),
                    text_encoder_variant=variants.get(host.id),
                    text_encoder_switch=self._switch if host.id == LOCAL_HOST_ID else None,
                )
            )
        return HostList(items=items)

    def update_host(self, host_id: str, patch: HostPatch) -> None:
        """Check a text encoder variant switch (refusals raise here), then run it in the background.

        The switch installs the variant's files, stores the variant, rewrites WanGP's `text_encoder_quantization`,
        restarts the local worker and then removes the previous variant's files. Only the local host has a model store.
        """
        if self.db.query_one("SELECT 1 FROM hosts WHERE id = ? AND revoked_at IS NULL", (host_id,)) is None:
            raise NotFound(f"Host {host_id} is not enrolled.")
        variant = patch.text_encoder_variant
        if variant is None:
            return
        if host_id != LOCAL_HOST_ID:
            raise Conflict(f"Host {host_id} has no model store of this core yet, so its text encoder cannot be switched.")
        try:
            switch = self.models.prepare_switch(variant)
        except ModelStoreError as exc:
            raise _store_refusal(exc) from None
        self._switch = TextEncoderSwitch(variant=switch.variant)
        self._in_background(f"switch the text encoder to {variant}", lambda: self._run_switch(switch))

    def add_host(self) -> None:
        raise NotImplementedFeature("Adding hosts", "phase 3")

    def list_approvals(self) -> ApprovalList:
        rows = self.db.query(
            "SELECT approvals.*, jobs.data_class FROM approvals JOIN jobs ON jobs.id = approvals.job_id ORDER BY approvals.created_at DESC"
        )
        return ApprovalList(
            items=[
                Approval(
                    id=row["id"],
                    job_id=row["job_id"],
                    reason=row["reason"],
                    estimated_cost_eur=row["estimated_cost_eur"],
                    data_class=DataClass(row["data_class"]),
                    host_id=row["host_id"],
                    state=row["state"],
                    created_at=from_iso(row["created_at"]),
                    decided_at=from_iso(row["decided_at"]),
                )
                for row in rows
            ]
        )

    def decide_approval(self, approval_id: str, approve: bool) -> None:
        raise NotImplementedFeature("Approval decisions", "phase 3")

    def list_recipes(self) -> RecipeList:
        return RecipeList(items=[])

    def list_cast(self) -> CastList:
        return CastList(items=[])

    def list_presets(self) -> PresetList:
        return PresetList(items=[])

    # ------------------------------------------------------------------ worker protocol

    def register_worker(self, registration: WorkerRegistration, enrollment_token: str) -> WorkerRegistered:
        if not hmac.compare_digest(enrollment_token.encode(), self.enrollment_token.encode()):
            raise Unauthorized("The enrollment token is not valid for this core start.")
        host = self.db.query_one("SELECT id FROM hosts WHERE id = ? AND revoked_at IS NULL", (registration.host_id,))
        if host is None:
            raise Invalid(f"Host {registration.host_id} is not enrolled.")
        token = secrets.token_urlsafe(32)
        worker_id = uuid4().hex
        now = to_iso(self.clock())
        self.db.execute(
            "INSERT INTO workers (id, host_id, engine, engine_version, models, health, token_hash, registered_at, last_seen_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                worker_id,
                registration.host_id,
                registration.engine,
                registration.engine_version,
                json.dumps(registration.models),
                registration.health.model_dump_json(),
                _token_hash(token),
                now,
                now,
            ),
        )
        log.info("Worker %s registered for %s on %s", worker_id, registration.engine, registration.host_id)
        if registration.host_id == LOCAL_HOST_ID:
            self._local_worker_registered()
        return WorkerRegistered(worker_id=worker_id, token=token, poll_seconds=WORKER_POLL_SECONDS)

    def authenticate_worker(self, token: str) -> str:
        with self.db.transaction() as conn:
            row = conn.execute("SELECT id FROM workers WHERE token_hash = ?", (_token_hash(token),)).fetchone()
            if row is None:
                raise Unauthorized("Unknown worker token.")
            conn.execute("UPDATE workers SET last_seen_at = ? WHERE id = ?", (to_iso(self.clock()), row["id"]))
        return row["id"]

    def heartbeat(self, worker_id: str, heartbeat: Heartbeat) -> HeartbeatAck:
        with self._tx() as tx:
            tx.conn.execute(
                "UPDATE workers SET health = ?, last_seen_at = ? WHERE id = ?",
                (heartbeat.health.model_dump_json(), to_iso(tx.now), worker_id),
            )
            self.scheduler.extend(tx, worker_id)
            rows = tx.conn.execute(
                "SELECT id FROM jobs WHERE worker_id = ? AND state IN (?, ?) AND cancel_requested = 1", (worker_id, *ACTIVE_STATES)
            ).fetchall()
        return HeartbeatAck(cancel=[row["id"] for row in rows])

    async def lease(self, worker_id: str, request: LeaseRequest, gone: Callable[[], Awaitable[bool]] | None = None) -> Lease | None:
        """Long-poll for a job. `gone` reports whether the requesting worker disconnected; a later check then takes no job.

        A worker retired by `restart_local_worker` gets no job, and its waiting long-poll returns at once.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + request.wait_seconds
        first = True
        while not self._closed and self._leasing:
            wakeup = self.scheduler.wakeup.snapshot()  # before checking, so no enqueue slips between check and wait
            # Only a request's first check releases the worker's abandoned jobs. A later check may belong to a
            # stale long-poll whose connection dropped while the worker's retry already got a job; it must not
            # take that job back. Nor may the long-poll of a worker process that was killed take a new job:
            # the job would wait for its lease to expire while the restarted worker idles.
            if not first and gone is not None and await gone():
                return None
            granted = self._try_lease(worker_id, abandon=first)
            first = False
            remaining = deadline - loop.time()
            if granted is not None or remaining <= 0 or worker_id in self._retired:
                return granted
            try:
                await asyncio.wait_for(wakeup.wait(), remaining)
            except TimeoutError:
                return None
        return None

    def _try_lease(self, worker_id: str, *, abandon: bool = True) -> Lease | None:
        if not self._leasing or worker_id in self._retired:
            return None
        with self._tx() as tx:
            worker = self._worker_record(tx.conn, worker_id)
            row = self.scheduler.lease_next(tx, worker, abandon=abandon)
            if row is None:
                return None
            job = self._worker_job(tx.conn, row)
        if worker.host_id == LOCAL_HOST_ID:
            self.models.mark_used(job.model_id, job.quant)  # after the transaction: lock order
        return Lease(job=job)

    def _worker_job(self, conn: sqlite3.Connection, row: sqlite3.Row) -> WorkerJob:
        request = JobRequest.model_validate_json(row["request"])
        route = Route.model_validate_json(row["route"])
        model = self._require_model(route.model_id)
        binding = self.registry.binding(route.model_id, route.quant, route.engine)
        inputs = []
        for sha256 in request.inputs:
            asset = conn.execute("SELECT media_type FROM assets WHERE sha256 = ?", (sha256,)).fetchone()
            # Path only: the worker resolves it against its own core URL, which may name the loopback
            # address differently (localhost vs 127.0.0.1) than the core's configuration.
            url = f"{API_PREFIX}/worker/inputs/{sha256}"
            inputs.append(WorkerInput(sha256=sha256, media_type=asset["media_type"], url=url))
        return WorkerJob(
            job_id=row["id"],
            lease_id=row["lease_id"],
            lease_expires_at=from_iso(row["lease_expires_at"]),
            capability=request.capability,
            model_id=route.model_id,
            quant=route.quant,
            engine_model=binding.model_type,
            prompt=dialect.translate_prompt(request.prompt, model.prompt_dialect),
            negative_prompt=request.negative_prompt,
            inputs=inputs,
            params=router.effective_params(request, model, row["seed"]),
            seed=row["seed"],
        )

    def worker_input_path(self, worker_id: str, sha256: str) -> Path:
        rows = self.db.query("SELECT request FROM jobs WHERE worker_id = ? AND state IN (?, ?)", (worker_id, *ACTIVE_STATES))
        if any(sha256 in JobRequest.model_validate_json(row["request"]).inputs for row in rows) and self.blobs.exists(sha256):
            return self.blobs.path(sha256)
        raise NotFound(f"Input {sha256} belongs to no job leased to this worker.")

    @staticmethod
    def _active_job(conn: sqlite3.Connection, worker_id: str, job_id: str, lease_id: str | None = None) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise NotFound(f"Job {job_id} does not exist.")
        if row["worker_id"] != worker_id or row["state"] not in ACTIVE_STATES or (lease_id is not None and row["lease_id"] != lease_id):
            raise Conflict(f"Job {job_id} is not leased to this worker under this lease; stop working on it.")
        return row

    @staticmethod
    def _mark_running(tx: Tx, row: sqlite3.Row) -> None:
        if row["state"] == JobState.LEASED.value:
            transition(tx, row["id"], JobState.RUNNING, started_at=row["started_at"] or to_iso(tx.now))

    def worker_events(self, worker_id: str, job_id: str, events: WorkerEvents) -> WorkerEventsAck:
        with self._tx() as tx:
            row = self._active_job(tx.conn, worker_id, job_id, events.lease_id)
            self._mark_running(tx, row)
            for event in events.events:
                data = event.data
                if UPLOAD_RECORD_KEY in data:
                    raise Invalid(f"The event data key {UPLOAD_RECORD_KEY!r} is reserved for the core's upload records.")
                if event.kind == "progress":
                    try:
                        progress = Progress.model_validate(event.data)
                    except ValidationError as exc:
                        raise Invalid(f"A progress event does not match the contract: {exc.errors()[0]['msg']}.") from exc
                    tx.conn.execute("UPDATE jobs SET progress = ? WHERE id = ?", (progress.model_dump_json(), job_id))
                    data = progress.model_dump()
                if event.kind == "preview":
                    tx.broadcast(job_id, EventKind.PREVIEW, data)  # live only; the event log keeps the latest one
                else:
                    tx.emit(job_id, EventKind(event.kind), data)
            self.scheduler.extend(tx, worker_id, job_id)
        return WorkerEventsAck(cancel_requested=bool(row["cancel_requested"]))

    def check_output(self, worker_id: str, job_id: str) -> None:
        with self.db.transaction() as conn:
            self._active_job(conn, worker_id, job_id)

    def store_output(self, worker_id: str, job_id: str, name: str, media_type: str, upload: StagedBlob) -> OutputUploaded:
        self.check_output(worker_id, job_id)  # the job may have ended while the upload ran; then the blob is not placed
        sha256 = self.blobs.place(upload)
        with self._tx() as tx:
            row = self._active_job(tx.conn, worker_id, job_id)
            self._mark_running(tx, row)
            tx.conn.execute(
                "INSERT INTO job_outputs (job_id, name, sha256, media_type, size_bytes, created_at) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (job_id, name) DO UPDATE SET sha256 = excluded.sha256, media_type = excluded.media_type, "
                "size_bytes = excluded.size_bytes, position = NULL, provenance = NULL, created_at = excluded.created_at",
                (job_id, name, sha256, media_type, upload.size_bytes, to_iso(tx.now)),
            )
            tx.emit(job_id, EventKind.LOG, {"message": f"Uploaded output {name}"})
            self.scheduler.extend(tx, worker_id, job_id)
        return OutputUploaded(sha256=sha256, size_bytes=upload.size_bytes)

    @staticmethod
    def _uploaded_outputs(conn: sqlite3.Connection, job_id: str) -> dict[str, _Upload]:
        """Outputs uploaded for a job, by name (the last upload of a name wins)."""
        rows = conn.execute("SELECT name, sha256, size_bytes, media_type FROM job_outputs WHERE job_id = ?", (job_id,))
        return {row["name"]: _Upload(row["sha256"], row["size_bytes"], row["media_type"]) for row in rows}

    def _model_files(self, model: ModelDescriptor | None, route: Route, variant: str) -> list[str]:
        """SHA-256 of the files a job's model ran with, as the model store plans them (`plan_files`).

        On the local host: the transformer files of the routed quantization, then the shared files the descriptor
        `requires`, those of the text encoder `variant` included. Other hosts have no model store of this core yet,
        so only the transformer files are known; so too when the registry entry cannot be planned.
        """
        if model is None:
            return []
        if route.host_id == LOCAL_HOST_ID:
            try:
                return [file.sha256 for file in plan_files(self.registry, route.model_id, route.quant, variant)]
            except ModelStoreError as exc:
                log.warning("Provenance of a %s %s job lists only its transformer files: %s", route.model_id, route.quant, exc.message)
        quant = next((q for q in model.quantizations if q.id == route.quant), None)
        return [file.sha256 for file in quant.files] if quant else []

    def complete_job(self, worker_id: str, job_id: str, completion: JobCompletion) -> None:
        variant = self.models.worker_variant()  # before the transaction: store calls never run inside it
        with self._tx() as tx:
            row = self._active_job(tx.conn, worker_id, job_id, completion.lease_id)
            request = JobRequest.model_validate_json(row["request"])
            route = Route.model_validate_json(row["route"])
            model = self.registry.get(route.model_id)
            files = self._model_files(model, route, variant)
            params = None
            if model is not None:
                effective = router.effective_params(request, model, row["seed"])
                params = provenance_params(effective, request.capability, completion.accepted_extra)
            worker = self._worker_record(tx.conn, worker_id)
            host = tx.conn.execute("SELECT * FROM hosts WHERE id = ?", (route.host_id,)).fetchone()
            agent = tx.conn.execute("SELECT name FROM agents WHERE id = ?", (row["agent_id"],)).fetchone() if row["agent_id"] else None
            names = [output.name for output in completion.outputs]
            if len(set(names)) < len(names):
                twice = sorted({name for name in names if names.count(name) > 1})
                raise Invalid(f"The completion lists output {', '.join(twice)} more than once; list each output name once.")
            uploaded = self._uploaded_outputs(tx.conn, job_id)
            for position, output in enumerate(completion.outputs):
                record = uploaded.get(output.name)
                if record is None or record.sha256 != output.sha256 or not self.blobs.exists(output.sha256):
                    raise Conflict(f"Output {output.name} with this hash was not uploaded for job {job_id}.")
                # The upload's checked Content-Type is the asset's media type; the completion may only repeat it.
                if output.media_type.split(";")[0].strip().lower() != record.media_type:
                    raise Conflict(
                        f"Output {output.name} was uploaded as {record.media_type}, so it cannot be completed as {output.media_type}."
                    )
                provenance = output_provenance(
                    asset=ProvenanceAsset(sha256=output.sha256, media_type=record.media_type, size_bytes=record.size_bytes),
                    job_id=job_id,
                    request=request,
                    route=route,
                    model=model,
                    file_sha256=files,
                    params=params,
                    engine_version=completion.engine_version or worker.engine_version,
                    seed=row["seed"],
                    host=host,
                    gpu=self._gpu_name(route.host_id),
                    client=ClientKind(row["client"]),
                    agent=agent["name"] if agent else None,
                    created_at=tx.now,
                )
                record_asset(tx.conn, provenance, DataClass(row["data_class"]), job_id, tx.now)
                # The asset keeps the provenance of the first job that produced these bytes; this job's own is here.
                tx.conn.execute(
                    "UPDATE job_outputs SET position = ?, provenance = ? WHERE job_id = ? AND name = ?",
                    (position, provenance.model_dump_json(by_alias=True), job_id, output.name),
                )
            transition(
                tx,
                job_id,
                JobState.SUCCEEDED,
                {"outputs": [o.sha256 for o in completion.outputs]},
                finished_at=to_iso(tx.now),
                lease_id=None,
                lease_expires_at=None,
            )
        # After the transaction (lock order): the store's least recently used order, and a restart a model change left pending.
        if route.host_id == LOCAL_HOST_ID:
            self.models.mark_used(route.model_id, route.quant)
        self._restart_if_pending()

    def fail_job(self, worker_id: str, job_id: str, failure: JobFailure) -> None:
        with self._tx() as tx:
            row = self._active_job(tx.conn, worker_id, job_id, failure.lease_id)
            error = failure.error
            finished = {"finished_at": to_iso(tx.now), "lease_id": None, "lease_expires_at": None, "error": error.model_dump_json()}
            if row["cancel_requested"] or error.code == "cancelled":
                transition(tx, job_id, JobState.CANCELLED, {"error": error.model_dump()}, **finished)
            elif error.retryable:
                self.scheduler.release(tx, row, error.code, error)
            else:
                transition(tx, job_id, JobState.FAILED, {"error": error.model_dump()}, **finished)
        self._restart_if_pending()

    # ------------------------------------------------------------------ maintenance

    def expire_leases(self) -> int:
        with self._tx() as tx:
            expired = self.scheduler.expire(tx)
        if expired:
            self._restart_if_pending()
        return expired

    async def run_maintenance(self, interval: float = MAINTENANCE_INTERVAL_S) -> None:
        """Background task of the app: re-queue jobs whose lease expired, and restart the worker for a model change that waited."""
        while not self._closed:
            try:
                expired = self.expire_leases()
            except sqlite3.Error:
                log.exception("Lease expiry failed")
            else:
                if expired:
                    log.info("Released %d job(s) with expired leases", expired)
            try:
                self._restart_if_pending()  # a model change that waited for a job, should no job end report it
            except sqlite3.Error:
                log.exception("Restarting the local worker after a model change failed")
            await asyncio.sleep(interval)


def _wont_run_everywhere(rows: list[CalibrationResult], host_id: str) -> bool:
    host_rows = [r for r in rows if r.host_id == host_id]
    return bool(host_rows) and all(r.status == CalibrationStatus.WONT_RUN for r in host_rows)

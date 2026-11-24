"""The core: the implementation of `CoreService` (see `service.py`).

One `Core` per process start. It owns the database, the asset store, the scheduler and the event
log, and generates the enrollment token that worker processes need to register.
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
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

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
    ModelList,
    ModelQuantState,
    ModelSummary,
    OutputUploaded,
    PresetList,
    Progress,
    ProvenanceAsset,
    RecipeList,
    Route,
    Settings,
    SettingsPatch,
    SystemInfo,
    WorkerEvents,
    WorkerEventsAck,
    WorkerHealth,
    WorkerInput,
    WorkerJob,
    WorkerRegistered,
    WorkerRegistration,
)
from cinewright.config import Config
from cinewright.core import hardware, router
from cinewright.core.assets import BlobStore, BlobWriter, StagedBlob, asset_from_row, input_provenance, output_provenance, record_asset
from cinewright.core.clock import Clock, from_iso, to_iso, utcnow
from cinewright.core.errors import Conflict, Infeasible, Invalid, NotFound, NotImplementedFeature, Unauthorized
from cinewright.core.events import EventLog, Published
from cinewright.core.licenses import WANGP_NAME, WANGP_NOTICE, license_notices
from cinewright.core.registry import Registry
from cinewright.core.scheduler import ACTIVE_STATES, LEASE_SECONDS, Scheduler, Tx, WorkerRecord, transition
from cinewright.db.database import Database

log = logging.getLogger(__name__)

LOCAL_HOST_ID = "local"
LOCAL_HOST_NAME = "Local"
WORKER_STALE_SECONDS = 120.0  # a worker without contact for this long serves no new routes
WORKER_POLL_SECONDS = 1.0
MAINTENANCE_INTERVAL_S = 5.0
# Uploaded outputs are recorded as log events carrying this key; `complete_job` checks against them.
# The 0001 schema has no table for job outputs (see the core agent's contract notes).
UPLOAD_RECORD_KEY = "output"

Detector = Callable[[Path, str], HardwareProfile]

_JOB_SELECT = (
    "SELECT jobs.*, agents.name AS agent_name, "
    "(SELECT data FROM job_events e WHERE e.job_id = jobs.id AND e.kind = 'state' ORDER BY e.seq DESC LIMIT 1) AS last_state "
    "FROM jobs LEFT JOIN agents ON agents.id = jobs.agent_id"
)
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


class _Upload(NamedTuple):
    """An output upload as `store_output` recorded it, including the checked media type."""

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
    ) -> None:
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
        self._closed = False
        self._ensure_local_host()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.event_log.close()
        self.scheduler.wakeup.notify()
        self.db.close()

    # ------------------------------------------------------------------ internals

    @contextmanager
    def _tx(self) -> Iterator[Tx]:
        """Write transaction; its events are published and waiting workers woken after commit."""
        published: list[Published] = []
        now = self.clock()
        with self.db.transaction() as conn:
            tx = Tx(conn, now, lambda job_id, kind, data: published.append(EventLog.append(conn, job_id, kind, data, now)))
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
        return [
            router.WorkerInfo(id=row["id"], host_id=row["host_id"], engine=row["engine"], models=frozenset(json.loads(row["models"])))
            for row in rows
            if WorkerHealth.model_validate_json(row["health"]).state != "failed"
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

    def _wan2gp_info(self) -> EngineInfo:
        online = {w.id for w in self._online_workers() if w.engine == "wan2gp"}
        rows = [
            r
            for r in self.db.query("SELECT id, engine_version FROM workers WHERE engine = 'wan2gp' ORDER BY last_seen_at DESC")
            if r["id"] in online
        ]
        if rows:
            return EngineInfo(id="wan2gp", name=WANGP_NAME, version=rows[0]["engine_version"], state="ready", notice=WANGP_NOTICE)
        return EngineInfo(id="wan2gp", name=WANGP_NAME, version=None, state="not_installed", notice=WANGP_NOTICE)

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
                )
            )
        return states

    def _summary(self, model: ModelDescriptor) -> ModelSummary:
        return ModelSummary(
            id=model.id,
            name=model.name,
            vendor=model.vendor,
            family=model.family,
            capabilities=model.capabilities,
            license=model.license,
            quantizations=self._quant_states(model),
        )

    def list_models(self) -> ModelList:
        return ModelList(items=[self._summary(model) for model in self.registry.all()])

    def get_model(self, model_id: str) -> ModelDetail:
        model = self._require_model(model_id)
        return ModelDetail(
            **self._summary(model).model_dump(),
            limits=model.limits,
            prompt_dialect=model.prompt_dialect,
            defaults=model.defaults,
            source=model.source,
            engines=model.engines,
        )

    def install_model(self, model_id: str, quant: str | None) -> None:
        self._require_model(model_id)
        raise NotImplementedFeature("Model installation", "phase 1")

    def remove_model(self, model_id: str, quant: str | None) -> None:
        self._require_model(model_id)
        raise NotImplementedFeature("Model removal", "phase 1")

    def calibration(self) -> CalibrationReport:
        return CalibrationReport(items=self._calibration_rows())

    def start_calibration(self) -> None:
        raise NotImplementedFeature("Calibration runs", "phase 1")

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
        return self.get_job(job_id)

    @staticmethod
    def _agent_id(tx: Tx, name: str) -> str:
        row = tx.conn.execute("SELECT id FROM agents WHERE name = ?", (name,)).fetchone()
        if row is not None:
            return row["id"]
        agent_id = uuid4().hex
        # Agent tokens arrive with MCP agent identities; until then the record only names the agent.
        tx.conn.execute("INSERT INTO agents (id, name, token_hash, created_at) VALUES (?, ?, '', ?)", (agent_id, name, to_iso(tx.now)))
        return agent_id

    @staticmethod
    def _job(row: sqlite3.Row) -> Job:
        outputs: list[str] = []
        if row["state"] == JobState.SUCCEEDED.value and row["last_state"]:
            outputs = json.loads(row["last_state"]).get("outputs", [])
        return Job(
            id=row["id"],
            state=JobState(row["state"]),
            request=JobRequest.model_validate_json(row["request"]),
            route=Route.model_validate_json(row["route"]) if row["route"] else None,
            progress=Progress.model_validate_json(row["progress"]) if row["progress"] else None,
            eta_seconds=row["eta_seconds"],
            client=ClientKind(row["client"]),
            agent=row["agent_name"],
            outputs=outputs,
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
        return JobList(items=[self._job(row) for row in rows[:limit]], next_cursor=next_cursor)

    def get_job(self, job_id: str) -> Job:
        row = self.db.query_one(_JOB_SELECT + " WHERE jobs.id = ?", (job_id,))
        if row is None:
            raise NotFound(f"Job {job_id} does not exist.")
        return self._job(row)

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

    def events(self, job_id: str | None, after: str | None) -> AsyncIterator[JobEvent]:
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

    def list_hosts(self) -> HostList:
        online = {w.host_id for w in self._online_workers()}
        last_seen = {
            row["host_id"]: row["seen"] for row in self.db.query("SELECT host_id, MAX(last_seen_at) AS seen FROM workers GROUP BY host_id")
        }
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
                )
            )
        return HostList(items=items)

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

    async def lease(self, worker_id: str, request: LeaseRequest) -> Lease | None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + request.wait_seconds
        first = True
        while not self._closed:
            wakeup = self.scheduler.wakeup.snapshot()  # before checking, so no enqueue slips between check and wait
            # Only a request's first check releases the worker's abandoned jobs. A later check may belong to a
            # stale long-poll whose connection dropped while the worker's retry already got a job; it must not
            # take that job back.
            granted = self._try_lease(worker_id, abandon=first)
            first = False
            remaining = deadline - loop.time()
            if granted is not None or remaining <= 0:
                return granted
            try:
                await asyncio.wait_for(wakeup.wait(), remaining)
            except TimeoutError:
                return None
        return None

    def _try_lease(self, worker_id: str, *, abandon: bool = True) -> Lease | None:
        with self._tx() as tx:
            worker = self._worker_record(tx.conn, worker_id)
            row = self.scheduler.lease_next(tx, worker, abandon=abandon)
            if row is None:
                return None
            job = self._worker_job(tx.conn, row)
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
            # Every MVP model reads natural language; dialect translation arrives with the first `tags` model.
            prompt=request.prompt,
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
            output = {"name": name, "sha256": sha256, "media_type": media_type, "size_bytes": upload.size_bytes}
            tx.emit(job_id, EventKind.LOG, {"message": f"Uploaded output {name}", UPLOAD_RECORD_KEY: output})
            self.scheduler.extend(tx, worker_id, job_id)
        return OutputUploaded(sha256=sha256, size_bytes=upload.size_bytes)

    @staticmethod
    def _uploaded_outputs(conn: sqlite3.Connection, job_id: str) -> dict[str, _Upload]:
        """Outputs uploaded for a job, by name, from the core's upload log events (the last upload of a name wins)."""
        uploaded: dict[str, _Upload] = {}
        for row in conn.execute("SELECT data FROM job_events WHERE job_id = ? AND kind = 'log' ORDER BY seq", (job_id,)):
            record = json.loads(row["data"]).get(UPLOAD_RECORD_KEY)
            if isinstance(record, dict) and isinstance(record.get("name"), str) and isinstance(record.get("media_type"), str):
                try:
                    checked = OutputUploaded(sha256=record.get("sha256"), size_bytes=record.get("size_bytes"))
                except ValidationError:
                    continue
                uploaded[record["name"]] = _Upload(checked.sha256, checked.size_bytes, record["media_type"])
        return uploaded

    def complete_job(self, worker_id: str, job_id: str, completion: JobCompletion) -> None:
        with self._tx() as tx:
            row = self._active_job(tx.conn, worker_id, job_id, completion.lease_id)
            request = JobRequest.model_validate_json(row["request"])
            route = Route.model_validate_json(row["route"])
            worker = self._worker_record(tx.conn, worker_id)
            host = tx.conn.execute("SELECT * FROM hosts WHERE id = ?", (route.host_id,)).fetchone()
            agent = tx.conn.execute("SELECT name FROM agents WHERE id = ?", (row["agent_id"],)).fetchone() if row["agent_id"] else None
            uploaded = self._uploaded_outputs(tx.conn, job_id)
            for output in completion.outputs:
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
                    model=self.registry.get(route.model_id),
                    engine_version=completion.engine_version or worker.engine_version,
                    seed=row["seed"],
                    host=host,
                    gpu=self._gpu_name(route.host_id),
                    client=ClientKind(row["client"]),
                    agent=agent["name"] if agent else None,
                    created_at=tx.now,
                )
                record_asset(tx.conn, provenance, DataClass(row["data_class"]), job_id, tx.now)
            transition(
                tx,
                job_id,
                JobState.SUCCEEDED,
                {"outputs": [o.sha256 for o in completion.outputs]},
                finished_at=to_iso(tx.now),
                lease_id=None,
                lease_expires_at=None,
            )

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

    # ------------------------------------------------------------------ maintenance

    def expire_leases(self) -> int:
        with self._tx() as tx:
            return self.scheduler.expire(tx)

    async def run_maintenance(self, interval: float = MAINTENANCE_INTERVAL_S) -> None:
        """Background task of the app: re-queue jobs whose lease expired."""
        while not self._closed:
            try:
                expired = self.expire_leases()
            except sqlite3.Error:
                log.exception("Lease expiry failed")
            else:
                if expired:
                    log.info("Released %d job(s) with expired leases", expired)
            await asyncio.sleep(interval)


def _wont_run_everywhere(rows: list[CalibrationResult], host_id: str) -> bool:
    host_rows = [r for r in rows if r.host_id == host_id]
    return bool(host_rows) and all(r.status == CalibrationStatus.WONT_RUN for r in host_rows)

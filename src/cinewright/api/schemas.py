"""Internal API contract, version 1.

Every client (web UI, MCP server, workers) talks to the core through these models. They are the
single source of the OpenAPI document (`openapi.json`). Changing a model or a route changes the
public surface and is classified as `feat` (see ARCHITECTURE.md).

Conventions
- JSON field names are snake_case. Timestamps are ISO 8601 with offset.
- Hashes are lowercase hex SHA-256 without prefix; the UI shows them as `sha256:…`.
- Unknown or unmeasured values are `null`, never estimates.
- Errors use RFC 9457 problem details (`application/problem+json`, model `Problem`).
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

API_VERSION = "1"
API_PREFIX = "/api/v1"
DEFAULT_PORT = 8737

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", description="Lowercase hex SHA-256")]
ModelId = Annotated[str, Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9.]+)*$", max_length=80)]
TextEncoderVariant = Literal["bf16", "int8"]


class Contract(BaseModel):
    """Base for every contract model: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- enums


class Capability(StrEnum):
    T2I = "t2i"
    I2I = "i2i"
    T2V = "t2v"
    I2V = "i2v"
    V2V = "v2v"
    AUDIO = "audio"
    TTS = "tts"
    LORA_TRAINING = "lora_training"


class DataClass(StrEnum):
    """Routing policy class of a job, asset or cast entry."""

    PUBLIC = "public"
    PRIVATE = "private"
    BIOMETRIC = "biometric"


class Priority(StrEnum):
    INTERACTIVE = "interactive"
    BATCH = "batch"


class JobState(StrEnum):
    AWAITING_APPROVAL = "awaiting_approval"
    QUEUED = "queued"
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_JOB_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED})


class CalibrationStatus(StrEnum):
    RUNS = "runs"
    SLOW = "slow"
    WONT_RUN = "wont_run"


class HostKind(StrEnum):
    LOCAL = "local"
    REMOTE_OWN = "remote_own"
    REMOTE_RENTED = "remote_rented"
    CLOUD = "cloud"


class ClientKind(StrEnum):
    UI = "ui"
    MCP = "mcp"


class EventKind(StrEnum):
    """Kind of a job event, also its SSE event name; `JobEvent` describes the `data` of each kind."""

    STATE = "state"
    PROGRESS = "progress"
    PREVIEW = "preview"
    LOG = "log"


class InstallState(StrEnum):
    ABSENT = "absent"
    DOWNLOADING = "downloading"
    INSTALLED = "installed"
    FAILED = "failed"


# --------------------------------------------------------------------------- errors


class Problem(BaseModel):
    """RFC 9457 problem details. `type` is a URN such as `urn:cinewright:problem:not-implemented`."""

    model_config = ConfigDict(extra="allow")

    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None


PROBLEM_NOT_IMPLEMENTED = "urn:cinewright:problem:not-implemented"
PROBLEM_NOT_FOUND = "urn:cinewright:problem:not-found"
PROBLEM_INFEASIBLE = "urn:cinewright:problem:infeasible"
PROBLEM_CONFLICT = "urn:cinewright:problem:conflict"
PROBLEM_UNAUTHORIZED = "urn:cinewright:problem:unauthorized"
PROBLEM_INVALID = "urn:cinewright:problem:invalid"


# --------------------------------------------------------------------------- system


class SetupError(Contract):
    """Why a setup task (engine provisioning) failed or cannot run."""

    code: str = Field(description="Stable machine code, e.g. `hash_mismatch`, `download_failed`, `install_failed`")
    message: str = Field(description="Cause and consequence in one sentence")
    action: Literal["retry"] | None = Field(default=None, description="The one helpful action the UI offers; null when none helps")


class EngineInfo(Contract):
    id: str = Field(description="Engine id, e.g. `wan2gp`")
    name: str
    version: str | None = Field(description="Pinned upstream commit or version; null when not provisioned")
    state: Literal["not_installed", "provisioning", "ready", "failed"] = Field(description="From engine provisioning")
    notice: str | None = Field(default=None, description="Licence disclosure shown in the UI (WanGP requires one)")
    progress: Progress | None = Field(default=None, description="Provisioning progress while `state` is `provisioning`")
    error: SetupError | None = Field(
        default=None, description="Why provisioning failed (`failed`) or cannot run here (`not_installed`); null otherwise"
    )


class SystemInfo(Contract):
    name: Literal["Cinewright"] = "Cinewright"
    version: str
    api_version: str = API_VERSION
    data_dir: str
    bind: str = Field(description="host:port the API listens on")
    engines: list[EngineInfo]


class Health(Contract):
    status: Literal["ok", "degraded"]
    checks: dict[str, str] = Field(default_factory=dict, description="Check name to short result")


# --------------------------------------------------------------------------- hardware


class GpuInfo(Contract):
    index: int
    name: str
    vram_total_mb: int
    driver_version: str | None = None
    compute_capability: str | None = Field(default=None, description="e.g. `12.0` for Blackwell")
    pcie_link: str | None = Field(default=None, description="e.g. `Gen5 x16`")


class HardwareProfile(Contract):
    host_id: str
    os: str
    cpu: str | None = None
    ram_total_mb: int
    disk_free_mb: int = Field(description="Free space on the drive that holds the data directory")
    gpus: list[GpuInfo]
    cuda_driver_ok: bool | None = Field(description="Driver new enough for the GPU (Blackwell needs CUDA 12.8+); null if no NVIDIA GPU")
    notes: list[str] = Field(default_factory=list, description="Findings worth showing, one sentence each")
    detected_at: datetime


# --------------------------------------------------------------------------- registry and models


class Limits(Contract):
    resolutions: list[str] = Field(default_factory=list, description="`WIDTHxHEIGHT` values the model supports")
    max_frames: int | None = None
    min_frames: int | None = Field(default=None, ge=1, description="Fewest frames of a clip; null for a model without frames (images only)")
    frames_step: int | None = Field(
        default=None,
        ge=1,
        description="A clip has `min_frames + n * frames_step` frames (n >= 0); null for a model without frames",
    )
    fps: list[int] = Field(default_factory=list)
    max_duration_s: float | None = None

    @model_validator(mode="after")
    def _frame_grid(self) -> Self:
        if (self.min_frames is None) != (self.frames_step is None):
            raise ValueError("min_frames and frames_step are set together, or both are null")
        return self


class License(Contract):
    name: str
    url: str | None = None
    commercial_use: bool | None = Field(description="null when the licence needs a case-by-case reading")
    geo_exclusions: list[str] = Field(default_factory=list, description="ISO 3166-1 alpha-2 codes or `EU`")


REGISTRY_FILE_HOSTS = {"huggingface": "huggingface.co", "civitai": "civitai.com"}
# Weights are safetensors or GGUF; companion files are non-executable JSON, text or SentencePiece models.
SHARED_FILE_EXTENSIONS = frozenset({".safetensors", ".gguf", ".json", ".txt", ".model"})
# One path segment. No trailing `.`: Windows drops it, so `enc.` and `enc` would be one folder.
# `_check_path_parts` also refuses Windows device names, which the pattern cannot express.
_PATH_PART = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9_-])?"
# Windows opens a device instead of a file for these names, with any extension (`CON.json`, `nul.tar.gz`).
_WINDOWS_DEVICE_NAMES = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{n}" for n in range(1, 10)), *(f"LPT{n}" for n in range(1, 10))})


def _check_path_parts(path: str) -> str:
    for part in path.split("/"):
        device = part.split(".", 1)[0]
        if device.upper() in _WINDOWS_DEVICE_NAMES:
            raise ValueError(f"{path}: {device} is a reserved device name on Windows")
    return path


def _check_download_url(url: str, provider: str) -> None:
    host = REGISTRY_FILE_HOSTS[provider]
    parsed = urlparse(url)
    # The whole netloc, not just the hostname: no user info, no port, no `\@` trick, no other letter case.
    if parsed.scheme != "https" or parsed.netloc != host:
        raise ValueError(f"{url}: files of a {provider} source download from https://{host}")


class ModelFile(Contract):
    name: str = Field(pattern=rf"^{_PATH_PART}$", description="File name, as the engine looks it up")
    url: str
    sha256: Sha256
    size_bytes: int | None = None

    @field_validator("name")
    @classmethod
    def _no_device_name(cls, name: str) -> str:
        return _check_path_parts(name)


class Quantization(Contract):
    id: str = Field(description="Upstream spelling, e.g. `bf16`, `int8`, `Q6_K`")
    format: Literal["safetensors", "gguf"]
    files: list[ModelFile] = Field(min_length=1)

    @model_validator(mode="after")
    def _weights_match_format(self) -> Self:
        for file in self.files:
            if PurePosixPath(file.name).suffix.lower() != f".{self.format}":
                raise ValueError(f"{file.name}: files of a {self.format} quantization end in .{self.format}")
        return self


class PromptDialect(Contract):
    style: Literal["natural", "tags"] = "natural"
    notes: str | None = None


class EngineBinding(Contract):
    engine: Literal["wan2gp"]
    model_type: str = Field(description="Engine-side model id, e.g. Wan2GP `model_type`")
    quant: str = Field(description="Quantization id this binding serves")


class ModelSource(Contract):
    provider: Literal["huggingface", "civitai"]
    repo: str
    revision: str | None = None


class ModelDescriptor(Contract):
    """One registry entry, stored as YAML under `src/cinewright/registry/models/`."""

    id: ModelId
    name: str
    vendor: str
    family: str
    capabilities: list[Capability] = Field(min_length=1)
    limits: Limits
    license: License
    quantizations: list[Quantization] = Field(min_length=1)
    requires: list[ModelId] = Field(
        default_factory=list, description="Shared components (`registry/shared/<id>.yaml`) every quantization needs"
    )
    prompt_dialect: PromptDialect = Field(default_factory=PromptDialect)
    defaults: dict[str, Any] = Field(default_factory=dict)
    preference: int = Field(
        default=0,
        description="Routing preference among candidates of equal cost, higher first; ranks before calibration and runtime",
    )
    source: ModelSource
    engines: list[EngineBinding] = Field(min_length=1)

    @model_validator(mode="after")
    def _safe_downloads(self) -> Self:
        for quant in self.quantizations:
            for file in quant.files:
                _check_download_url(file.url, self.source.provider)
        return self


class SharedFile(Contract):
    """One engine file shared by several models (text encoder, VAE, upscaler, tokenizer or config file)."""

    name: str = Field(pattern=rf"^{_PATH_PART}$", description="File name, as the engine looks it up")
    subfolder: str = Field(
        default="",
        pattern=rf"^(?:{_PATH_PART}/)*$",
        description="Folder below the checkpoint root the engine expects, with trailing `/`; empty for the root",
    )
    url: str
    sha256: Sha256
    size_bytes: int = Field(gt=0)
    variant: Literal["bf16", "int8"] | None = Field(
        default=None, description="Text encoder quantization this file belongs to; null when every variant needs it"
    )

    @property
    def path(self) -> str:
        """Path relative to the checkpoint root."""
        return self.subfolder + self.name

    @field_validator("name", "subfolder")
    @classmethod
    def _no_device_name(cls, path: str) -> str:
        return _check_path_parts(path)

    @field_validator("name")
    @classmethod
    def _allowed_extension(cls, name: str) -> str:
        if PurePosixPath(name).suffix.lower() not in SHARED_FILE_EXTENSIONS:
            raise ValueError(f"{name}: only {', '.join(sorted(SHARED_FILE_EXTENSIONS))} files are allowed")
        return name


class SharedComponent(Contract):
    """Engine files several models need, from one upstream repository at one revision.

    Stored as YAML under `src/cinewright/registry/shared/`; model descriptors name it in `requires`.
    """

    id: ModelId = Field(description="Equals the file name without `.yaml`")
    source: ModelSource
    files: list[SharedFile] = Field(min_length=1)

    @model_validator(mode="after")
    def _pinned_downloads(self) -> Self:
        revision = self.source.revision
        if revision is None:
            raise ValueError("source.revision is required: hashes are read at a fixed revision")
        if self.source.provider == "huggingface" and not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"source.revision {revision!r} is not a commit")
        host = REGISTRY_FILE_HOSTS[self.source.provider]
        paths: set[str] = set()
        for file in self.files:
            # Windows file systems ignore case, so `Qwen3/x.json` and `qwen3/x.json` are one file.
            if file.path.casefold() in paths:
                raise ValueError(f"{file.path} is listed twice (paths ignore case)")
            paths.add(file.path.casefold())
            _check_download_url(file.url, self.source.provider)
            expected = f"https://{host}/{self.source.repo}/resolve/{revision}/{file.path}"
            if self.source.provider == "huggingface" and file.url != expected:
                raise ValueError(f"{file.url}: expected {expected}")
        return self

    def files_for(self, variant: str | None) -> list[SharedFile]:
        """Files the engine needs with the given text encoder variant; `None` returns every file."""
        return [file for file in self.files if variant is None or file.variant in (None, variant)]


class CalibrationResult(Contract):
    host_id: str
    model_id: ModelId
    quant: str
    resolution: str
    status: CalibrationStatus
    peak_vram_mb: int | None = None
    seconds_per_step: float | None = None
    ram_offload_mb: int | None = None
    detail: str | None = Field(default=None, description="Measurement or reason, e.g. `needs 52 GB VRAM`")
    engine: str
    engine_version: str | None = None
    measured_at: datetime


class CalibrationReport(Contract):
    items: list[CalibrationResult]


class ModelQuantState(Contract):
    quant: str
    install_state: InstallState
    size_bytes: int | None = Field(
        default=None,
        description="Installed: disk space of the quantization, its shared files included. Otherwise: its transformer files",
    )
    calibration: list[CalibrationResult] = Field(default_factory=list)
    error: str | None = Field(default=None, description="Why the last install failed (`failed`), in one sentence; null otherwise")


class ModelSummary(Contract):
    id: ModelId
    name: str
    vendor: str
    family: str
    capabilities: list[Capability]
    license: License
    quantizations: list[ModelQuantState]
    installed_bytes: int = Field(
        default=0, description="Disk space of the model's installed quantizations together, each shared file once; 0 when none is"
    )


class ModelDetail(ModelSummary):
    limits: Limits
    prompt_dialect: PromptDialect
    defaults: dict[str, Any]
    source: ModelSource
    engines: list[EngineBinding]


class ModelStoreUsage(Contract):
    """The model store (`<data dir>/models`) as the Models view shows it."""

    total_bytes: int = Field(description="Disk space of every file in the model store, each shared file once")
    quota_bytes: int | None = Field(description="`settings.model_store_quota_gb` in bytes (1 GB = 1024³ bytes); null = no quota")
    free_bytes: int = Field(description="Free space of the drive that holds the model store, measured for this answer")


class ModelList(Contract):
    items: list[ModelSummary] = Field(description="Registry models, without those whose licence excludes `settings.country`")
    store: ModelStoreUsage | None = Field(default=None, description="Size, quota and free disk space of the model store")


class CapabilityEntry(Contract):
    capability: Capability
    model_id: ModelId
    model_name: str
    quant: str
    limits: Limits
    status: CalibrationStatus | None = Field(description="null = not calibrated on any host")
    hosts: list[str] = Field(default_factory=list, description="Hosts whose policy and calibration allow it")


class CapabilityList(Contract):
    items: list[CapabilityEntry]


# --------------------------------------------------------------------------- jobs


class JobParams(Contract):
    resolution: str | None = Field(default=None, pattern=r"^\d{2,5}x\d{2,5}$")
    frames: int | None = Field(default=None, ge=1)
    fps: int | None = Field(default=None, ge=1, le=120)
    duration_s: float | None = Field(default=None, gt=0)
    steps: int | None = Field(default=None, ge=1, le=200)
    seed: int | None = Field(default=None, ge=0, description="null = random, resolved by the core and recorded")
    guidance: float | None = Field(default=None, ge=0)
    extra: dict[str, Any] = Field(default_factory=dict, description="Model-specific parameters from the registry")


class RecipeRef(Contract):
    id: str
    version: int = Field(ge=1)


class JobRequest(Contract):
    capability: Capability
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str | None = Field(default=None, max_length=4000)
    inputs: list[Sha256] = Field(default_factory=list, description="Input assets (images, clips, audio)")
    params: JobParams = Field(default_factory=JobParams)
    model: ModelId | None = Field(default=None, description="Pin a registry model; null lets the router choose")
    quant: str | None = None
    priority: Priority = Priority.INTERACTIVE
    data_class: DataClass = DataClass.PRIVATE
    budget_eur: float | None = Field(default=None, ge=0)
    deadline: datetime | None = None
    recipe: RecipeRef | None = Field(default=None, description="Phase 2; rejected with 501 until recipes exist")


class Route(Contract):
    model_id: ModelId
    quant: str
    host_id: str
    engine: str
    reasons: list[str] = Field(default_factory=list, description="Router decisions in check order")


class Progress(Contract):
    phase: str
    step: int | None = None
    total_steps: int | None = None
    percent: float = Field(ge=0, le=100)


class JobError(Contract):
    code: str = Field(description="Stable machine code, e.g. `oom`, `engine_failed`, `cancelled`")
    message: str = Field(description="Cause and consequence in one sentence")
    retryable: bool = False


class Job(Contract):
    id: str
    state: JobState
    request: JobRequest
    route: Route | None = None
    progress: Progress | None = None
    eta_seconds: float | None = None
    client: ClientKind
    agent: str | None = Field(default=None, description="Agent name for MCP jobs")
    outputs: list[Sha256] = Field(default_factory=list)
    error: JobError | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class JobList(Contract):
    items: list[Job]
    next_cursor: str | None = None


class JobEvent(Contract):
    """A job event of `GET /events`: the SSE event name is `kind`, the `data:` line is this object.

    Stored events (`state`, `progress`, `log`) have the SSE id `<job_id>:<seq>`. A reconnecting browser resends the
    last one as `Last-Event-ID`, and the stream replays the stored events after it. Previews are not stored in
    `job_events` and never replayed: a `preview` has no `id:` line and repeats the `seq` of the job's latest stored
    event, so `Last-Event-ID` always names a stored event. After a stream opens or reconnects (after its replay), the
    core sends the latest kept preview of each running job the stream covers once; later previews follow live.
    """

    job_id: str
    seq: int = Field(description="Per-job sequence number of the stored events, starting at 1; a `preview` repeats the latest one")
    kind: EventKind
    data: dict[str, Any] = Field(
        description="By `kind`. "
        "`state`: `state` (a JobState), plus where set: `worker_id` (`leased`), `reason` (why the job was queued again, "
        "cancelled or failed after its last attempt), `outputs` (`succeeded`: the output SHA-256 values in completion "
        "order) and `error` (a JobError: `failed`, or `cancelled` while running). "
        "`progress`: a Progress (`phase`, `step`, `total_steps`, `percent`). "
        "`preview`: `media_type` and `data_base64` (the frame as base64), as the worker sent them; the core does not "
        "check them, the shipped worker sends `image/png`, `image/jpeg` or `image/webp` of at most 256 KiB. "
        "`log`: `message`, a sentence for people. Log rows that releases 0.0.1 to 0.0.3 stored for an upload also "
        "carry `output` (`name`, `sha256`, `media_type`, `size_bytes`), and replay still sends them as stored; uploads "
        "are now the job's outputs, and workers may not send `output`."
    )
    created_at: datetime


class SystemEventKind(StrEnum):
    """Kind of a non-job event, also its SSE event name and the `SystemEvent` field that carries its payload."""

    ENGINE = "engine"
    MODEL = "model"


class ModelInstall(Contract):
    """A model quantization's new install state: download progress, the end of an install, or a removal."""

    model_id: ModelId
    quant: str
    install_state: InstallState
    progress: Progress | None = Field(
        default=None, description="While `downloading`: share of the bytes this install downloads (`phase` `download`)"
    )
    downloaded_bytes: int | None = Field(default=None, description="While `downloading`: bytes downloaded so far")
    total_bytes: int | None = Field(default=None, description="While `downloading`: bytes this install downloads")
    error: SetupError | None = Field(default=None, description="Why the install failed (`failed`); null otherwise")


class SystemEvent(Contract):
    """A non-job event of `GET /events`: the SSE event name is `kind`, the `data:` line is this object.

    Only a stream without `job_id` carries them. They have no id, are not stored and never replayed, so a stream
    gets only those published while it is open; reload `GET /system` or `GET /models` whenever the stream opens.
    `engine`: progress or a state change of engine provisioning, with the engine as `GET /system` reports it.
    `model`: progress or a state change of a model install, also a removal or an eviction (`absent`).
    """

    kind: SystemEventKind
    engine: EngineInfo | None = Field(default=None, description="For `engine`: the engine's new state, progress and error")
    model: ModelInstall | None = Field(default=None, description="For `model`: the quantization's new install state and progress")
    created_at: datetime


class Estimate(Contract):
    feasible: bool
    route: Route | None = None
    eta_seconds: float | None = None
    cost_eur: float | None = None
    requires_approval: bool = False
    reasons: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- assets and provenance


class ProvenanceAsset(Contract):
    sha256: Sha256
    media_type: str
    size_bytes: int


class ProvenanceModel(Contract):
    id: ModelId
    name: str
    quant: str
    engine: str
    engine_version: str | None = None
    file_sha256: list[Sha256] = Field(default_factory=list)


class ProvenanceParams(Contract):
    """The requested generation parameters, with the model's registry defaults filled in. The engine may still adjust
    them, for example to a distilled model's fixed steps; a clip's frame count lies on the model's frame grid."""

    resolution: str | None = None
    steps: int | None = None
    frames: int | None = Field(default=None, description="null for images")
    fps: int | None = Field(default=None, description="null for images")
    guidance: float | None = None
    extra: dict[str, Any] | None = Field(
        default=None,
        description="The entries of `params.extra` the worker passed on to the engine as asked; null when it did not report them",
    )


class ProvenanceLora(Contract):
    name: str
    sha256: Sha256 | None = None
    weight: float


class ProvenanceHost(Contract):
    id: str
    name: str
    kind: HostKind
    gpu: str | None = None


class ProvenanceClient(Contract):
    kind: ClientKind
    agent: str | None = None


class Provenance(Contract):
    """Sidecar record of one asset. Field order follows the spec; missing values stay null."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_id: Literal["cinewright.provenance/1"] = Field(default="cinewright.provenance/1", alias="schema")
    asset: ProvenanceAsset
    model: ProvenanceModel | None = None
    recipe: RecipeRef | None = None
    prompt: str | None = None
    negative_prompt: str | None = None
    seed: int | None = None
    params: ProvenanceParams | None = Field(default=None, description="null for uploaded inputs and for records stored by earlier releases")
    loras: list[ProvenanceLora] = Field(default_factory=list)
    host: ProvenanceHost | None = None
    created_at: datetime
    parents: list[Sha256] = Field(default_factory=list)
    job_id: str | None = None
    client: ProvenanceClient | None = None


class Asset(Contract):
    sha256: Sha256
    media_type: str
    size_bytes: int
    data_class: DataClass
    job_id: str | None = None
    created_at: datetime
    provenance: Provenance


class AssetList(Contract):
    items: list[Asset]
    next_cursor: str | None = None


# --------------------------------------------------------------------------- hosts, approvals


class WorkerHealth(Contract):
    state: Literal["starting", "ready", "busy", "failed"]
    vram_used_mb: int | None = None
    vram_total_mb: int | None = None
    detail: str | None = None


class TextEncoderSwitch(Contract):
    """A text encoder variant switch of a host since the core started: running, or failed."""

    variant: TextEncoderVariant = Field(description="The variant the switch installs")
    error: SetupError | None = Field(
        default=None, description="Why the switch failed; the previous variant stays in use. null while the switch runs"
    )


class Host(Contract):
    id: str
    name: str
    kind: HostKind
    allowed_data_classes: list[DataClass]
    online: bool
    last_seen_at: datetime | None = None
    hardware: HardwareProfile | None = None
    worker_health: WorkerHealth | None = Field(
        default=None,
        description="State and VRAM use of the host's worker, as last reported by a worker registered since the core started. "
        "On the local host the core's process supervision overrides it while the worker starts or is down. null = no worker",
    )
    text_encoder_variant: TextEncoderVariant | None = Field(
        default=None,
        description="Text encoder variant of the host's models (WanGP `text_encoder_quantization`); null = not chosen yet, `int8` applies",
    )
    text_encoder_switch: TextEncoderSwitch | None = Field(
        default=None,
        description="The host's last text encoder switch while it runs or after it failed; null when none ran since the core "
        "started or the last one succeeded, which `text_encoder_variant` then shows",
    )


class HostList(Contract):
    items: list[Host]


class HostPatch(Contract):
    text_encoder_variant: TextEncoderVariant | None = Field(
        default=None,
        description="Switch the host's text encoder variant: the core checks free disk space, installs the variant's files, "
        "restarts the worker and removes the previous variant's files. Refused while a job runs on the host",
    )


class Approval(Contract):
    id: str
    job_id: str
    reason: str
    estimated_cost_eur: float | None = None
    data_class: DataClass
    host_id: str | None = None
    state: Literal["pending", "approved", "declined"]
    created_at: datetime
    decided_at: datetime | None = None


class ApprovalList(Contract):
    items: list[Approval]


# --------------------------------------------------------------------------- recipes, cast, presets (phase 2)


class RecipeSummary(Contract):
    id: str
    name: str
    version: int
    description: str | None = None


class RecipeList(Contract):
    items: list[RecipeSummary]


class CastSummary(Contract):
    id: str
    name: str
    data_class: DataClass


class CastList(Contract):
    items: list[CastSummary]


class PresetSummary(Contract):
    id: str
    name: str
    recipe: RecipeRef
    duration_s: float | None = None
    fps: int | None = None


class PresetList(Contract):
    items: list[PresetSummary]


# --------------------------------------------------------------------------- settings, licences


class Settings(Contract):
    update_channel: Literal["stable"] = Field(default="stable", description="Releases from main; see ARCHITECTURE.md")
    country: str = Field(default="DE", pattern=r"^[A-Z]{2}$", description="For licence geo exclusions")
    approval_threshold_eur: float = Field(default=0.0, ge=0, description="Remote and cloud jobs above it need approval")
    budget_eur_per_month: float | None = Field(default=None, ge=0)
    model_store_quota_gb: int | None = Field(default=None, ge=1)
    kwh_price_eur: float | None = Field(default=None, ge=0)


class SettingsPatch(Contract):
    country: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    approval_threshold_eur: float | None = Field(default=None, ge=0)
    budget_eur_per_month: float | None = Field(default=None, ge=0)
    model_store_quota_gb: int | None = Field(default=None, ge=1)
    kwh_price_eur: float | None = Field(default=None, ge=0)


class LicenseNotice(Contract):
    component: str
    license: str
    url: str | None = None
    notice: str | None = None


class LicenseNotices(Contract):
    items: list[LicenseNotice]


# --------------------------------------------------------------------------- worker protocol (core <-> worker, pull model)


class WorkerRegistration(Contract):
    host_id: str
    engine: str
    engine_version: str | None = None
    models: list[str] = Field(description="Engine model ids (e.g. Wan2GP model_type) this worker can serve")
    health: WorkerHealth


class WorkerRegistered(Contract):
    worker_id: str
    token: str = Field(description="Bearer token for all further worker calls; the core stores only its hash")
    poll_seconds: float = Field(description="Suggested wait between lease requests")


class Heartbeat(Contract):
    health: WorkerHealth


class HeartbeatAck(Contract):
    cancel: list[str] = Field(default_factory=list, description="Job ids the worker must cancel")


class LeaseRequest(Contract):
    wait_seconds: float = Field(default=0, ge=0, le=30, description="Long-poll up to this many seconds")


class WorkerInput(Contract):
    sha256: Sha256
    media_type: str
    url: str = Field(description="Path on the core, e.g. `/api/v1/worker/inputs/<sha256>`, resolved against the worker's core URL")


class WorkerJob(Contract):
    job_id: str
    lease_id: str
    lease_expires_at: datetime
    capability: Capability
    model_id: ModelId
    quant: str
    engine_model: str = Field(description="Engine-side model id from the registry binding")
    prompt: str = Field(description="Already translated into the model's prompt dialect")
    negative_prompt: str | None = None
    inputs: list[WorkerInput] = Field(default_factory=list)
    params: JobParams
    seed: int = Field(ge=0, description="Resolved seed; recorded in provenance")


class Lease(Contract):
    job: WorkerJob


class WorkerEvent(Contract):
    kind: Literal["progress", "preview", "log"]
    data: dict[str, Any]


class WorkerEvents(Contract):
    lease_id: str
    events: list[WorkerEvent] = Field(min_length=1, max_length=100)


class WorkerEventsAck(Contract):
    cancel_requested: bool


class OutputUploaded(Contract):
    sha256: Sha256
    size_bytes: int


class CompletedOutput(Contract):
    sha256: Sha256
    media_type: str
    name: str


class JobCompletion(Contract):
    lease_id: str
    outputs: list[CompletedOutput] = Field(min_length=1)
    engine_version: str | None = None
    accepted_extra: list[Annotated[str, Field(max_length=64)]] | None = Field(
        default=None,
        max_length=100,
        description="Keys of the job's `params.extra` the worker passed on to the engine as asked; null when it does not report them",
    )


class JobFailure(Contract):
    lease_id: str
    error: JobError

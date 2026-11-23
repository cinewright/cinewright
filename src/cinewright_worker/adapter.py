"""Engine adapter contract.

Every engine (Wan2GP first, later ComfyUI, diffusers, cloud APIs) is wrapped by one adapter that
runs inside the worker process, in the engine's own Python environment. The worker talks to the
core over the worker protocol (`/api/v1/worker/...`) and to the engine only through this interface.

This package is standard library only and supports Python 3.10+, because it is imported inside
engine environments that Cinewright does not control (Wan2GP recommends Python 3.11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

ENGINE_JOB_STATES = ("queued", "running", "succeeded", "failed", "cancelled")


@dataclass(frozen=True)
class EngineModel:
    """A model the engine can serve, by its engine-side id (e.g. Wan2GP `model_type`)."""

    engine_model: str
    name: str
    available: bool  # all files present locally
    outputs: tuple[str, ...] = ()  # "image", "video", "audio"


@dataclass(frozen=True)
class EngineJob:
    """What the worker hands to `submit`, derived from the core's `WorkerJob`."""

    job_id: str
    engine_model: str
    quant: str
    prompt: str
    negative_prompt: str | None
    seed: int
    params: dict[str, Any]
    input_paths: tuple[str, ...] = ()  # local files the worker downloaded from the core
    output_dir: str = ""


@dataclass
class EngineStatus:
    state: str  # one of ENGINE_JOB_STATES
    phase: str = ""
    step: int | None = None
    total_steps: int | None = None
    percent: float = 0.0
    preview_path: str | None = None
    error_code: str | None = None  # "oom", "engine_failed", "cancelled", ...
    error_message: str | None = None


@dataclass(frozen=True)
class OutputFile:
    path: str
    media_type: str  # "image/png", "video/mp4", "audio/wav", ...
    name: str


@dataclass
class EngineHealth:
    state: str  # "starting", "ready", "busy", "failed"
    vram_used_mb: int | None = None
    vram_total_mb: int | None = None
    detail: str | None = None
    engine_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class EngineAdapter(Protocol):
    """One adapter instance per worker process."""

    name: str

    def capabilities(self) -> list[EngineModel]:
        """Registry-relevant models this engine can serve right now."""

    def prepare(self, engine_model: str, quant: str) -> None:
        """Check that the model's files are present and load it; raise on failure."""

    def submit(self, job: EngineJob) -> str:
        """Start a job and return the engine's job id. Must not block until completion."""

    def status(self, engine_job_id: str) -> EngineStatus:
        """Current state, progress and, if available, a preview frame."""

    def cancel(self, engine_job_id: str) -> None:
        """Request cancellation; `status` reports `cancelled` once the engine stopped."""

    def fetch_outputs(self, engine_job_id: str) -> list[OutputFile]:
        """Result files of a succeeded job."""

    def health(self) -> EngineHealth:
        """Engine state and current VRAM use."""

    def translate_prompt(self, intent: str, engine_model: str) -> str:
        """Translate a user intent into the model's prompt dialect."""

"""Mock engine for tests and the walking skeleton.

Simulates denoising steps in a background thread and writes a small PNG that depends only on the
seed and the requested resolution, so the same job gives the same bytes. Both models write a PNG;
`mock-video` does not produce a real video file. A job fails on purpose when
`params["extra"]["mock_fail"]` names an error code (e.g. `oom`), which lets tests exercise the
failure path end to end.

The core routes a job only to a worker whose engine id and model list match the registry binding
(engine `wan2gp`, a Wan2GP model type). An in-process end-to-end test can therefore stand in for
Wan2GP with `MockAdapter(models=("z_image",), engine="wan2gp")`.
"""

from __future__ import annotations

import dataclasses
import os
import random
import struct
import tempfile
import threading
import uuid
import zlib
from typing import Any

from cinewright_worker.adapter import EngineHealth, EngineJob, EngineModel, EngineStatus, OutputFile
from cinewright_worker.engines import EngineError

DEFAULT_MODELS = ("mock-image", "mock-video")
MOCK_VERSION = "mock-1"
MAX_SIDE = 256  # outputs are scaled down to fit, keeping the aspect ratio
PREVIEW_SIDE = 32
DEFAULT_STEPS = 8
DEFAULT_RESOLUTION = "1024x1024"
TERMINAL_STATES = ("succeeded", "failed", "cancelled")


def encode_png(width: int, height: int, rgb: bytes) -> bytes:
    """Encode 8-bit RGB pixels (row-major, `width * height * 3` bytes) as a PNG file."""
    if len(rgb) != width * height * 3:
        raise ValueError("pixel buffer does not match width x height x 3")
    stride = width * 3
    raw = b"".join(b"\x00" + rgb[row * stride : (row + 1) * stride] for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8 bit, truecolour, no interlace
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def render_png(seed: int, width: int, height: int) -> bytes:
    """A diagonal two-colour gradient whose colours come from `seed`."""
    rng = random.Random(seed)  # noqa: S311 - test pattern, not cryptography
    start = [rng.randrange(256) for _ in range(3)]
    end = [rng.randrange(256) for _ in range(3)]
    span = max(width + height - 2, 1)
    palette = [bytes(round(a + (b - a) * i / span) for a, b in zip(start, end, strict=True)) for i in range(span + 1)]
    rgb = b"".join(palette[x + y] for y in range(height) for x in range(width))
    return encode_png(width, height, rgb)


def fit_resolution(resolution: str | None, max_side: int) -> tuple[int, int]:
    """Parse `WIDTHxHEIGHT` and scale it to fit within `max_side`, keeping the aspect ratio."""
    width, height = (int(part) for part in (resolution or DEFAULT_RESOLUTION).lower().split("x"))
    scale = min(1.0, max_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


@dataclasses.dataclass
class _MockJob:
    job: EngineJob
    total_steps: int
    output_dir: str
    status: EngineStatus
    cancel: threading.Event = dataclasses.field(default_factory=threading.Event)
    outputs: list[OutputFile] = dataclasses.field(default_factory=list)


class MockAdapter:
    """`EngineAdapter` that serves `mock-image` and `mock-video` (configurable) without a GPU."""

    def __init__(self, models: tuple[str, ...] = DEFAULT_MODELS, step_seconds: float = 0.01, engine: str = "mock") -> None:
        self.name = engine
        self._models = tuple(models)
        self._step_seconds = step_seconds
        self._jobs: dict[str, _MockJob] = {}
        self._lock = threading.Lock()

    def capabilities(self) -> list[EngineModel]:
        return [
            EngineModel(engine_model=model, name=f"Mock {model}", available=True, outputs=("video",) if "video" in model else ("image",))
            for model in self._models
        ]

    def prepare(self, engine_model: str, quant: str) -> None:
        if engine_model not in self._models:
            raise EngineError("model_unavailable", f"The mock engine does not serve {engine_model}.")

    def submit(self, job: EngineJob) -> str:
        self.prepare(job.engine_model, job.quant)
        steps = int(job.params.get("steps") or DEFAULT_STEPS)
        engine_job_id = f"mock-{uuid.uuid4().hex}"
        output_dir = job.output_dir or tempfile.mkdtemp(prefix="cinewright-mock-")
        record = _MockJob(job=job, total_steps=steps, output_dir=output_dir, status=EngineStatus(state="queued", total_steps=steps))
        with self._lock:
            self._jobs[engine_job_id] = record
        threading.Thread(target=self._run, args=(record,), daemon=True, name=f"mock-{job.job_id}").start()
        return engine_job_id

    def status(self, engine_job_id: str) -> EngineStatus:
        with self._lock:
            return dataclasses.replace(self._record(engine_job_id).status)

    def cancel(self, engine_job_id: str) -> None:
        self._record(engine_job_id).cancel.set()

    def fetch_outputs(self, engine_job_id: str) -> list[OutputFile]:
        with self._lock:
            record = self._record(engine_job_id)
            if record.status.state != "succeeded":
                raise EngineError("engine_failed", f"Job {engine_job_id} has no outputs (state {record.status.state}).")
            return list(record.outputs)

    def health(self) -> EngineHealth:
        busy = any(record.status.state in ("queued", "running") for record in list(self._jobs.values()))
        return EngineHealth(state="busy" if busy else "ready", engine_version=MOCK_VERSION, detail="Mock engine for tests")

    def translate_prompt(self, intent: str, engine_model: str) -> str:
        return intent

    # ------------------------------------------------------------------ internals

    def _record(self, engine_job_id: str) -> _MockJob:
        try:
            return self._jobs[engine_job_id]
        except KeyError:
            raise EngineError("engine_failed", f"Unknown mock job {engine_job_id}.") from None

    def _update(self, record: _MockJob, **changes: Any) -> None:
        with self._lock:
            record.status = dataclasses.replace(record.status, **changes)

    def _run(self, record: _MockJob) -> None:
        job, total = record.job, record.total_steps
        fail_code = (job.params.get("extra") or {}).get("mock_fail")
        for step in range(1, total + 1):
            if record.cancel.wait(self._step_seconds):
                self._update(record, state="cancelled", phase="cancelled", error_code="cancelled", error_message="Cancelled on request.")
                return
            if fail_code and step > total // 2:
                message = "Simulated out-of-memory error." if fail_code == "oom" else f"Simulated failure ({fail_code})."
                self._update(record, state="failed", phase="failed", error_code=str(fail_code), error_message=message)
                return
            preview = self._write_preview(record, step) if step == max(1, total // 2) else record.status.preview_path
            percent = 100.0 * step / total
            self._update(record, state="running", phase="inference", step=step, total_steps=total, percent=percent, preview_path=preview)
        width, height = fit_resolution(job.params.get("resolution"), MAX_SIDE)
        name = "video.png" if "video" in job.engine_model else "image.png"
        path = os.path.join(record.output_dir, name)
        with open(path, "wb") as out:
            out.write(render_png(job.seed, width, height))
        with self._lock:
            record.outputs = [OutputFile(path=path, media_type="image/png", name=name)]
            record.status = dataclasses.replace(record.status, state="succeeded", phase="done", percent=100.0)

    def _write_preview(self, record: _MockJob, step: int) -> str:
        width, height = fit_resolution(record.job.params.get("resolution"), PREVIEW_SIDE)
        path = os.path.join(record.output_dir, f"preview-{step:04d}.png")
        with open(path, "wb") as out:
            out.write(render_png(record.job.seed, width, height))
        return path

"""Wan2GP adapter over WanGP's in-process Python API (`shared/api.py` of the pinned checkout).

Cinewright uses WanGP (Wan2GP) by deepbeepmeep as its first generation engine. WanGP is
source-available under the WanGP Community License 2.0 and is downloaded from its official
repository during setup; it is not part of Cinewright.
https://github.com/deepbeepmeep/Wan2GP/blob/main/LICENSE.txt

Wan2GP is imported lazily from `root` (the checkout is put on `sys.path`), so this module imports
without it. `init()` runs once per process: WanGP keeps one global runtime. Its config file lives in
Cinewright's data directory, not in the checkout. The adapter never triggers downloads: Wan2GP runs
offline, `prepare()` and `submit()` refuse models whose files are not all present, and Wan2GP is not
started while its ffmpeg binaries are missing (its `init()` would fetch unpinned ones from GitHub).
"""

from __future__ import annotations

import collections
import dataclasses
import importlib
import logging
import os
import re
import shutil
import sys
import threading
from pathlib import Path
from typing import Any

from cinewright_worker.adapter import EngineHealth, EngineJob, EngineModel, EngineStatus, OutputFile
from cinewright_worker.engines import EngineError, media_type_for

log = logging.getLogger(__name__)

# `--offline` makes wgp.py set HF_HUB_OFFLINE. Through shared.api that comes too late for
# huggingface_hub, so `_ensure_session` also sets it before the import (see there).
CLI_ARGS = ("--attention", "sdpa", "--offline")
PREVIEW_MAX_SIDE = 384
RESULT_TIMEOUT = 60.0
_MIB = 1024 * 1024
# Wan2GP rewrites CUDA OOM errors into "... unsufficient VRAM ..." / "... unsufficient RAM ..." (wgp.py).
_OOM = re.compile(r"out of memory|outofmemoryerror|tried to allocate|(?:un|in)sufficient v?ram", re.IGNORECASE)
# Wan2GP's init() downloads these into <root>/ffmpeg_bins from an unpinned GitHub release when one is
# missing (shared/ffmpeg_setup.py). Phase 1 provisioning places pinned copies there.
FFMPEG_DIR = "ffmpeg_bins"
FFMPEG_BINARIES = ("ffmpeg.exe", "ffprobe.exe", "ffplay.exe") if os.name == "nt" else ("ffmpeg", "ffprobe", "ffplay")
# The only Wan2GP settings a job's `params.extra` may set: plain numbers and short enum tokens, plus the
# negative prompt. Nothing path-valued, and no upsampling, post-processing, prompt enhancer, LoRA or
# reference-image settings, which load further models or read files beyond the worker's inputs.
_PLAIN_SETTINGS = frozenset(
    {
        "resolution",
        "num_inference_steps",
        "video_length",
        "duration_seconds",
        "force_fps",
        "seed",
        "guidance_scale",
        "guidance2_scale",
        "guidance3_scale",
        "guidance_phases",
        "model_switch_phase",
        "switch_threshold",
        "switch_threshold2",
        "embedded_guidance_scale",
        "flow_shift",
        "denoising_strength",
        "sample_solver",
    }
)
_TEXT_SETTINGS = frozenset({"negative_prompt"})
_TOKEN = re.compile(r"[A-Za-z0-9._:-]{0,32}")  # "1280x720", "4s", "auto", "euler"; no path separators
_MAX_TEXT_CHARS = 4000


def is_oom(messages: list[str]) -> bool:
    return any(_OOM.search(message or "") for message in messages)


def _accepted(key: str, value: Any) -> bool:
    if key in _TEXT_SETTINGS:
        return isinstance(value, str) and len(value) <= _MAX_TEXT_CHARS
    if key not in _PLAIN_SETTINGS or isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) or (isinstance(value, str) and _TOKEN.fullmatch(value) is not None)


def build_settings(defaults: dict[str, Any], job: EngineJob) -> dict[str, Any]:
    """Wan2GP task settings: the model's pristine defaults overlaid with the job's values."""
    settings = dict(defaults)
    params = job.params
    ignored = []
    for key, value in (params.get("extra") or {}).items():
        if key in settings and _accepted(key, value):
            settings[key] = value
        else:
            ignored.append(str(key)[:64])
    if ignored:
        log.warning("Ignored params.extra keys the Wan2GP adapter does not accept: %r", sorted(ignored)[:20])
    settings["model_type"] = job.engine_model
    settings["prompt"] = job.prompt
    settings["seed"] = job.seed
    if job.negative_prompt is not None:
        settings["negative_prompt"] = job.negative_prompt
    if params.get("resolution"):
        settings["resolution"] = params["resolution"]
    if params.get("steps"):
        settings["num_inference_steps"] = int(params["steps"])
    if params.get("guidance") is not None:
        settings["guidance_scale"] = params["guidance"]
    if params.get("fps"):
        settings["force_fps"] = int(params["fps"])
    if params.get("frames"):
        settings["video_length"] = int(params["frames"])
    elif params.get("duration_s"):
        settings["video_length"] = f"{float(params['duration_s']):g}s"  # Wan2GP converts seconds to a valid frame count
    if params.get("duration_s"):
        settings["duration_seconds"] = params["duration_s"]
    images = [path for path in job.input_paths if media_type_for(path).startswith("image/")]
    if len(images) != len(job.input_paths):
        raise EngineError("unsupported_input", "The Wan2GP adapter accepts image inputs only.")
    if len(images) > 2:
        raise EngineError("unsupported_input", "The Wan2GP adapter accepts at most two images (start and end frame).")
    if images:
        settings["image_start"] = images[0]  # Wan2GP infers the image prompt mode flags from these
    if len(images) > 1:
        settings["image_end"] = images[1]
    return settings


@dataclasses.dataclass
class _Job:
    handle: Any  # shared.api.SessionJob
    output_dir: str
    status: EngineStatus = dataclasses.field(default_factory=lambda: EngineStatus(state="queued"))
    outputs: list[OutputFile] = dataclasses.field(default_factory=list)
    errors: list[str] = dataclasses.field(default_factory=list)
    stream_tail: collections.deque = dataclasses.field(default_factory=lambda: collections.deque(maxlen=20))
    previews: int = 0


class Wan2GPAdapter:
    """`EngineAdapter` for Wan2GP model types such as `ltx2_22B_distilled`, `i2v_2_2`, `t2v_2_2`, `z_image`."""

    name = "wan2gp"

    def __init__(self, root: str, data_dir: str, output_dir: str | None = None, cli_args: tuple[str, ...] = CLI_ARGS) -> None:
        self._root = Path(root).resolve()
        engine_dir = Path(data_dir).resolve() / "engines" / "wan2gp"
        self._config_path = engine_dir / "wgp_config.json"  # WanGP requires exactly this file name
        self._output_dir = Path(output_dir).resolve() if output_dir else engine_dir / "outputs"
        self._cli_args = tuple(cli_args)
        self._session: Any = None
        self._version: str | None = None
        self._init_lock = threading.Lock()
        self._lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}

    # ------------------------------------------------------------------ EngineAdapter

    def capabilities(self) -> list[EngineModel]:
        session = self._ensure_session()
        availability = {record["model_type"]: bool(record.get("available")) for record in session.list_model_availability()}
        models = []
        for definition in session.list_model_defs():
            model_type = str(definition["model_type"])
            metadata = definition.get("metadata") or {}
            models.append(
                EngineModel(
                    engine_model=model_type,
                    name=str(definition.get("name") or model_type),
                    available=availability.get(model_type, False),
                    outputs=tuple(str(kind) for kind in metadata.get("outputs") or metadata.get("main_output") or ()),
                )
            )
        return models

    def prepare(self, engine_model: str, quant: str) -> None:
        """Check that every file of the model is present. Wan2GP loads the model on the first generation.

        `quant` is informational: the Wan2GP `model_type` already names the quantization.
        """
        session = self._ensure_session()
        if session.get_model_def(engine_model) is None:
            raise EngineError("model_unavailable", f"Wan2GP does not know the model type {engine_model}.")
        availability = session.get_model_availability(engine_model)
        if not availability.get("available"):
            raise EngineError(
                "model_unavailable",
                f"The files of Wan2GP model {engine_model} are {availability.get('status', 'missing')}; "
                "install the model before running it (the worker never downloads model files).",
            )

    def submit(self, job: EngineJob) -> str:
        self.prepare(job.engine_model, job.quant)  # also guards against a download started by generation
        session = self._ensure_session()
        settings = build_settings(session.get_default_settings(job.engine_model), job)
        with self._lock:
            if any(record.status.state in ("queued", "running") for record in self._jobs.values()):
                raise EngineError("engine_busy", "Wan2GP runs one job at a time and is busy.")
        try:
            handle = session.submit_task(settings)
        except (ValueError, TypeError) as exc:
            raise EngineError("invalid_request", f"Wan2GP rejected the job settings: {exc}") from exc
        output_dir = job.output_dir or str(self._output_dir)
        record = _Job(handle=handle, output_dir=output_dir)
        with self._lock:
            self._jobs[job.job_id] = record
        threading.Thread(target=self._consume, args=(record,), daemon=True, name=f"wan2gp-{job.job_id}").start()
        return job.job_id

    def status(self, engine_job_id: str) -> EngineStatus:
        with self._lock:
            return dataclasses.replace(self._record(engine_job_id).status)

    def cancel(self, engine_job_id: str) -> None:
        self._record(engine_job_id).handle.cancel()

    def fetch_outputs(self, engine_job_id: str) -> list[OutputFile]:
        with self._lock:
            record = self._record(engine_job_id)
            if record.status.state != "succeeded":
                raise EngineError("engine_failed", f"Job {engine_job_id} has no outputs (state {record.status.state}).")
            return list(record.outputs)

    def health(self) -> EngineHealth:
        if self._session is None:  # CUDA must not be touched before WanGP applied its allocator settings
            return EngineHealth(state="starting", detail="Wan2GP is not initialised yet.")
        used = total = None
        try:
            import torch  # Wan2GP's environment provides it

            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                used, total = (total_bytes - free_bytes) // _MIB, total_bytes // _MIB
        except Exception:  # noqa: BLE001, S110 - VRAM figures are optional
            pass
        with self._lock:
            busy = any(record.status.state in ("queued", "running") for record in self._jobs.values())
        return EngineHealth(state="busy" if busy else "ready", vram_used_mb=used, vram_total_mb=total, engine_version=self._version)

    def translate_prompt(self, intent: str, engine_model: str) -> str:
        """Return the intent unchanged. TODO(phase 1): per-model prompt dialects from the registry."""
        return intent

    # ------------------------------------------------------------------ internals

    def _ensure_session(self) -> Any:
        with self._init_lock:
            if self._session is None:
                bin_dir = self._root / FFMPEG_DIR
                missing = [name for name in FFMPEG_BINARIES if not (bin_dir / name).is_file()]
                if missing:
                    raise EngineError(
                        "engine_unavailable",
                        f"Wan2GP is not started because {', '.join(missing)} {'is' if len(missing) == 1 else 'are'} "
                        f"missing in {bin_dir}, where setup "
                        "places pinned ffmpeg binaries (Wan2GP would otherwise download unpinned ones).",
                    )
                # shared.api imports huggingface_hub, which reads this once at import, before wgp.py
                # handles --offline; setting it here keeps Wan2GP's downloader offline.
                os.environ["HF_HUB_OFFLINE"] = "1"
                if str(self._root) not in sys.path:
                    sys.path.insert(0, str(self._root))
                api = importlib.import_module("shared.api")
                self._config_path.parent.mkdir(parents=True, exist_ok=True)
                self._output_dir.mkdir(parents=True, exist_ok=True)
                self._session = api.init(
                    root=self._root,
                    config_path=self._config_path,
                    output_dir=self._output_dir,
                    cli_args=list(self._cli_args),
                    console_output=False,
                )
                wgp = sys.modules.get("wgp")
                version = getattr(wgp, "WanGP_version", None)
                self._version = str(version) if version is not None else None
            return self._session

    def _record(self, engine_job_id: str) -> _Job:
        try:
            return self._jobs[engine_job_id]
        except KeyError:
            raise EngineError("engine_failed", f"Unknown Wan2GP job {engine_job_id}.") from None

    def _update(self, record: _Job, **changes: Any) -> None:
        with self._lock:
            record.status = dataclasses.replace(record.status, **changes)

    def _consume(self, record: _Job) -> None:
        """Follow the job's event stream until WanGP closes it, then record the final state."""
        try:
            self._finalize(record, self._follow_events(record))
        except Exception as exc:  # noqa: BLE001 - any failure of the API ends the job
            self._update(record, state="failed", phase="failed", error_code="engine_failed", error_message=str(exc))

    def _follow_events(self, record: _Job) -> Any:
        handle = record.handle
        for event in handle.events.iter(timeout=0.2):
            kind, data = event.kind, event.data
            if kind == "started":
                self._update(record, state="running", phase="starting")
            elif kind == "progress":
                self._update(
                    record,
                    state="running",
                    phase=str(data.phase or ""),
                    step=data.current_step,
                    total_steps=data.total_steps,
                    percent=float(data.progress or 0),
                )
            elif kind == "preview" and getattr(data, "image", None) is not None:
                self._save_preview(record, data.image)
            elif kind == "error":
                record.errors.append(str(getattr(data, "message", data)))
            elif kind == "stream":
                record.stream_tail.append(str(getattr(data, "text", "")))
        return handle.result(timeout=RESULT_TIMEOUT)

    def _finalize(self, record: _Job, result: Any) -> None:
        messages = [error.message for error in result.errors] or record.errors
        if result.success and result.generated_files:
            outputs = [self._take_output(path, record.output_dir) for path in result.generated_files]
            with self._lock:
                record.outputs = outputs
                record.status = dataclasses.replace(record.status, state="succeeded", phase="done", percent=100.0)
        elif result.cancelled or (record.handle.cancel_requested and not result.success):
            self._update(record, state="cancelled", phase="cancelled", error_code="cancelled", error_message="Cancelled on request.")
        else:
            code = "oom" if is_oom(messages + list(record.stream_tail)) else "engine_failed"
            message = messages[0] if messages else "Wan2GP finished without an output."
            self._update(record, state="failed", phase="failed", error_code=code, error_message=message)

    def _save_preview(self, record: _Job, image: Any) -> None:
        record.previews += 1
        path = os.path.join(record.output_dir, f"preview-{record.previews:04d}.png")
        partial = path + ".part"
        try:
            thumbnail = image.copy()
            thumbnail.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
            thumbnail.save(partial, format="PNG")
            os.replace(partial, path)
        except Exception:  # noqa: BLE001, S110 - a missing preview is not an error
            return
        self._update(record, preview_path=path)

    @staticmethod
    def _take_output(path: str, output_dir: str) -> OutputFile:
        """Move a generated file into the job's output directory, which the worker removes after upload."""
        source = Path(path)
        target = Path(output_dir) / source.name
        if source.resolve() != target.resolve():
            shutil.move(str(source), str(target))
        return OutputFile(path=str(target), media_type=media_type_for(str(target)), name=target.name)

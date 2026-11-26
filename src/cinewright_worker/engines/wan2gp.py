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

Each job runs the quantization the core routed it to: `submit` names it in the task's `overrides` setting with
Wan2GP's own `set_override` (a checkout without it is not started), because Wan2GP otherwise loads its global
default (`transformer_quantization`, int8). For the same reason the adapter checks files per quantization itself,
from the Wan2GP root as `shared.api` does, instead of asking `shared.api`'s availability, which checks only the
global default. t2i jobs on a video model run in Wan2GP's image mode. i2i jobs are refused: no model type the adapter
runs takes an image to edit in image mode. Of `params.extra` the adapter passes on and reports only the keys Wan2GP's
validation leaves unchanged, as far as the adapter mirrors that validation (`_validated`, for the registry's model types
at the pinned commit).

Engine provisioning (`cinewright.core.provisioning`) writes `PROVISIONING_RECORD` into
`<data dir>/engines/wan2gp/` once the engine is installed. For the checkout it describes, the adapter
reports the record's commit as its engine version and passes the attention mode the record found working.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import importlib
import json
import logging
import os
import re
import shutil
import sys
import threading
from collections.abc import Callable, Collection, Iterator, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from cinewright_worker.adapter import EngineHealth, EngineJob, EngineModel, EngineStatus, OutputFile
from cinewright_worker.engines import EngineError, media_type_for

log = logging.getLogger(__name__)

PREVIEW_MAX_SIDE = 384
RESULT_TIMEOUT = 60.0
_MIB = 1024 * 1024
# Wan2GP rewrites CUDA OOM errors into "... unsufficient VRAM ..." / "... unsufficient RAM ..." (wgp.py).
_OOM = re.compile(r"out of memory|outofmemoryerror|tried to allocate|(?:un|in)sufficient v?ram", re.IGNORECASE)
# Wan2GP's init() downloads these into <root>/ffmpeg_bins from an unpinned GitHub release when one is
# missing (shared/ffmpeg_setup.py). Phase 1 provisioning places pinned copies there.
FFMPEG_DIR = "ffmpeg_bins"
FFMPEG_BINARIES = ("ffmpeg.exe", "ffprobe.exe", "ffplay.exe") if os.name == "nt" else ("ffmpeg", "ffprobe", "ffplay")
DEFAULT_ATTENTION = "sdpa"
# SageAttention modes of wgp.py's --attention; provisioning records one only when its probe found it working.
SAGE_ATTENTION_MODES = ("sage", "sage2")
# Written by engine provisioning into <data dir>/engines/wan2gp/ (cinewright.core.provisioning).
PROVISIONING_RECORD = "provisioning.json"
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
# Image mode renders Wan2GP's minimum frame count at 1 fps and keeps the first frame, whatever these say.
_VIDEO_SETTINGS = frozenset({"video_length", "force_fps", "duration_seconds"})
IMAGE_CAPABILITIES = frozenset({"t2i"})  # jobs that return one still image; i2i is refused (`_input_settings`)
# Wan2GP's `transformer_quantization_choices` at the pinned commit; `submit` reads the live list from wgp.
TRANSFORMER_QUANTIZATIONS = ("int8", "fp8", "bf16")
# Wan2GP's writer of the task setting `overrides` (`set_override`), which the adapter needs to select a quantization.
OVERRIDES_MODULE = "shared.utils.generation_overrides"


def is_oom(messages: list[str]) -> bool:
    return any(_OOM.search(message or "") for message in messages)


def wgp_args(attention: str = DEFAULT_ATTENTION) -> tuple[str, ...]:
    """wgp.py command-line arguments. `--offline` makes wgp.py set HF_HUB_OFFLINE; through shared.api that
    comes too late for huggingface_hub, so `_ensure_session` also sets it before the import (see there)."""
    return ("--attention", attention, "--offline")


def read_provisioning_record(data_dir: str) -> dict[str, Any] | None:
    """The provisioning record in `<data_dir>/engines/wan2gp/`, or None when it is missing or unreadable."""
    path = Path(data_dir) / "engines" / "wan2gp" / PROVISIONING_RECORD
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def attention_mode(record: dict[str, Any] | None) -> str:
    """A SageAttention mode the record found working, else `sdpa`."""
    mode = record.get("attention") if record else None
    return mode if mode in SAGE_ATTENTION_MODES else DEFAULT_ATTENTION


def _accepted(key: str, value: Any) -> bool:
    if key in _TEXT_SETTINGS:
        return isinstance(value, str) and len(value) <= _MAX_TEXT_CHARS
    if key not in _PLAIN_SETTINGS or isinstance(value, bool):
        return False
    if key == "sample_solver" and value == "":  # Wan's handler (fix_settings) replaces an empty solver with its default
        return False
    return isinstance(value, (int, float)) or (isinstance(value, str) and _TOKEN.fullmatch(value) is not None)


def _image_job(job: EngineJob) -> bool:
    return job.capability in IMAGE_CAPABILITIES


def _validated(settings: Mapping[str, Any], model_def: Mapping[str, Any]) -> dict[str, Any]:
    """The values Wan2GP's `validate_settings` (wgp.py at the pinned commit) ends with for the allowlisted settings it
    resets or limits, whatever the task asks for. Mirrored for the registry's model types, so a pin update or a new
    model type must check them again:
    - `denoising_strength` stays only with a control image ("V" and "G" in `video_prompt_type`);
    - `guidance_phases` is capped at the model's `guidance_max_phases` (fixed there with `lock_guidance_phases`);
    - `model_switch_phase` stays only with three phases;
    - LTX-2's distilled pipeline runs 8 steps at guidance 1.0 (`validate_generative_settings` of models/ltx2).
    """
    video_prompt_type = str(settings.get("video_prompt_type") or "")
    values: dict[str, Any] = {}
    if model_def.get("ltx2_pipeline") == "distilled":
        values.update(num_inference_steps=8, guidance_scale=1.0)
    if not ("V" in video_prompt_type and "G" in video_prompt_type):
        values["denoising_strength"] = 1.0
    phases = settings.get("guidance_phases", 1)
    most = model_def.get("guidance_max_phases", 0)
    if model_def.get("lock_guidance_phases"):
        phases = most
    elif isinstance(phases, (int, float)) and not isinstance(phases, bool):
        phases = min(phases, most)
    else:
        phases = None  # Wan2GP cannot compare it with the maximum
    values["guidance_phases"] = phases
    if phases != 3:
        values["model_switch_phase"] = 1
    return values


def _extra_settings(
    defaults: dict[str, Any], job: EngineJob, inputs: dict[str, Any], model_def: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """The allowed `params.extra` values for keys Wan2GP's defaults have, and the names of the keys left out: those the
    adapter does not accept, and those Wan2GP's validation would change for this job (`_validated`).
    `inputs` are the job's input settings (`_input_settings`)."""
    applied: dict[str, Any] = {}
    ignored = []
    for key, value in (job.params.get("extra") or {}).items():
        if key in defaults and _accepted(key, value) and not (_image_job(job) and key in _VIDEO_SETTINGS):
            applied[key] = value
        else:
            ignored.append(str(key)[:64])
    while True:  # leaving one out can change what Wan2GP does with another (guidance_phases, model_switch_phase)
        validated = _validated({**defaults, **applied, **inputs}, model_def)
        changed = [key for key in applied if key in validated and validated[key] != applied[key]]
        if not changed:
            return applied, ignored
        for key in changed:
            del applied[key]
            ignored.append(key)


def _job_settings(job: EngineJob) -> dict[str, Any]:
    """Settings from the job's own fields, which the core filled from the registry defaults; they win over extras."""
    params = job.params
    settings: dict[str, Any] = {"model_type": job.engine_model, "prompt": job.prompt, "seed": job.seed}
    if job.negative_prompt is not None:
        settings["negative_prompt"] = job.negative_prompt
    if params.get("resolution"):
        settings["resolution"] = params["resolution"]
    if params.get("steps"):
        settings["num_inference_steps"] = int(params["steps"])
    if params.get("guidance") is not None:
        settings["guidance_scale"] = params["guidance"]
    if _image_job(job):  # frames, fps and duration of a video model's registry defaults do not apply
        return settings
    if params.get("fps"):
        settings["force_fps"] = int(params["fps"])
    if params.get("frames"):
        settings["video_length"] = int(params["frames"])
    elif params.get("duration_s"):
        settings["video_length"] = f"{float(params['duration_s']):g}s"  # Wan2GP converts seconds to a valid frame count
    if params.get("duration_s"):
        settings["duration_seconds"] = params["duration_s"]
    return settings


def _input_settings(job: EngineJob) -> dict[str, Any]:
    if job.capability == "i2i":
        # Image mode of the registry's models offers no image to edit at the pinned commit. LTX-2.3's only image input
        # there is the Control Image (`image_guide`, "VG"), a raw control input meant for an IC-LoRA: a video model,
        # not an instruction editor, it returned a different picture when tried on the reference PC.
        message = f"The Wan2GP adapter runs no i2i job: Wan2GP's image mode takes no image to edit for {job.engine_model}."
        raise EngineError("unsupported_input", message)
    images = [path for path in job.input_paths if media_type_for(path).startswith("image/")]
    if len(images) != len(job.input_paths):
        raise EngineError("unsupported_input", "The Wan2GP adapter accepts image inputs only.")
    if _image_job(job):
        if images:
            raise EngineError("unsupported_input", "A t2i job takes no input image.")
        return {}
    if len(images) > 2:
        raise EngineError("unsupported_input", "The Wan2GP adapter accepts at most two images (start and end frame).")
    settings = {}
    if images:
        settings["image_start"] = images[0]  # Wan2GP infers the image prompt mode flags from these
    if len(images) > 1:
        settings["image_end"] = images[1]
    return settings


def build_settings(
    defaults: dict[str, Any],
    job: EngineJob,
    *,
    override: Callable[[str, str, Any], str],
    quantizations: Collection[str] = TRANSFORMER_QUANTIZATIONS,
    model_def: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Wan2GP task settings: the model's pristine defaults, overlaid with the accepted `params.extra` values, then
    with the job's own values, its input images, image mode for an image job and the routed quantization.

    `override` is Wan2GP's `set_override`. `quantizations` are Wan2GP's transformer quantizations, which the task's
    `overrides` can select; a GGUF quantization has its own model type and gets no override. `model_def` is Wan2GP's
    definition of the model type (`wgp.get_model_def`), whose limits decide which extras run as asked.
    """
    inputs = _input_settings(job)
    applied, ignored = _extra_settings(defaults, job, inputs, model_def or {})
    if ignored:
        log.warning("Ignored params.extra keys the Wan2GP adapter does not accept or Wan2GP resets: %r", sorted(ignored)[:20])
    settings = {**defaults, **applied, **_job_settings(job), **inputs}
    if _image_job(job) and not defaults.get("image_mode"):
        settings["image_mode"] = 1  # exactly 1, Wan2GP's image generation (2 is inpainting)
    if job.quant in quantizations:
        try:
            settings["overrides"] = override(str(defaults.get("overrides") or ""), "transformer_quantization", job.quant)
        except ValueError as exc:
            raise EngineError("engine_failed", f"Wan2GP's default overrides of {job.engine_model} cannot be read: {exc}") from exc
    return settings


def accepted_extra(defaults: dict[str, Any], job: EngineJob, model_def: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """The `params.extra` keys `build_settings` passes on to Wan2GP, which its validation leaves unchanged (`_validated`)
    and the job's own values do not replace, sorted. Whether a model's code then uses a key is up to Wan2GP."""
    applied, _ = _extra_settings(defaults, job, _input_settings(job), model_def or {})
    return tuple(sorted(set(applied) - set(_job_settings(job))))


class DenoisingStep(NamedTuple):
    """The denoising step a job reached: completed steps of the pass, its steps, its name, and whether it completed."""

    done: int
    total: int | None
    phase: str
    completed: bool = False


def _denoising_step(update: Any, previous: DenoisingStep | None) -> DenoisingStep | None:
    """The denoising step reached once a progress update arrived; `previous` if the update reports none.

    Wan2GP labels the update after the last step of a pass "VAE Decoding" (phase `decoding`, current step = total),
    also between the two passes of LTX-2 distilled, so the pass completed; its name stays that of the step before.
    LTX-2 distilled counts each pass from 0, so the pass name comes from `raw_phase`. Loading (blocks, phases) and the
    tiles of LTX-2's tiled second pass report a unit, and saving has another phase.
    """
    step, total = getattr(update, "current_step", None), getattr(update, "total_steps", None)
    phase = str(getattr(update, "phase", "") or "")
    if step is None or getattr(update, "unit", None) is not None:
        return previous
    if phase == "decoding":
        if not total or step != total:
            return previous
        name = previous.phase if previous is not None and previous.total == total else ""
        return DenoisingStep(int(step), total, name, completed=True)
    if not phase.startswith("inference"):
        return previous
    return DenoisingStep(int(step), total, " ".join(str(getattr(update, "raw_phase", "") or "").split())[:80])


def oom_message(step: DenoisingStep | None, message: str) -> str:
    """Wan2GP's out-of-memory message, led by the denoising step the job had reached."""
    if step is None:
        where = "before the first denoising step"
    elif step.completed:
        where = f"after the last step of {step.phase}" if step.phase else "after the last denoising step"
        where += f" (step {step.done} of {step.total})"
    elif step.done:
        where = f"at step {step.done}" + (f" of {step.total}" if step.total else "") + (f" ({step.phase})" if step.phase else "")
    else:
        where = "before the first step" + (f" of {step.phase}" if step.phase else "")
    return f"Out of memory {where}. Wan2GP: {message}"


def _basenames(names: list[str]) -> str:
    return ", ".join(name.replace("\\", "/").rsplit("/", 1)[-1] for name in names)


def _transformer_files(deps: Any, model_type: str, quantization: str, dtype_policy: str) -> list[str]:
    """The transformer files Wan2GP's `load_models` picks for `quantization`, both experts of a two-expert model."""
    model_def = deps.get_model_def(model_type) or {}
    names = [deps.get_model_filename(model_type, quantization=quantization, dtype_policy=dtype_policy)]
    if "URLs2" in model_def:
        names.append(deps.get_model_filename(model_type, quantization=quantization, dtype_policy=dtype_policy, submodel_no=2))
    return [name for name in names if isinstance(name, str) and name]


@dataclasses.dataclass
class _Job:
    handle: Any  # shared.api.SessionJob
    output_dir: str
    status: EngineStatus = dataclasses.field(default_factory=lambda: EngineStatus(state="queued"))
    image: bool = False  # an image job, which must return images only
    outputs: list[OutputFile] = dataclasses.field(default_factory=list)
    errors: list[str] = dataclasses.field(default_factory=list)
    stream_tail: collections.deque = dataclasses.field(default_factory=lambda: collections.deque(maxlen=20))
    previews: int = 0
    step: DenoisingStep | None = None  # the last denoising step reached (`_denoising_step`)


class Wan2GPAdapter:
    """`EngineAdapter` for Wan2GP model types such as `ltx2_22B_distilled`, `i2v_2_2`, `t2v_2_2`, `z_image`."""

    name = "wan2gp"

    def __init__(self, root: str, data_dir: str, output_dir: str | None = None, cli_args: tuple[str, ...] | None = None) -> None:
        """`cli_args` defaults to `--attention <mode> --offline` with the mode from the provisioning record."""
        self._root = Path(root).resolve()
        engine_dir = Path(data_dir).resolve() / "engines" / "wan2gp"
        self._config_path = engine_dir / "wgp_config.json"  # WanGP requires exactly this file name
        self._output_dir = Path(output_dir).resolve() if output_dir else engine_dir / "outputs"
        self._provisioning = self._own_record(read_provisioning_record(data_dir))
        self._cli_args = tuple(cli_args) if cli_args is not None else wgp_args(attention_mode(self._provisioning))
        self._session: Any = None
        self._version: str | None = None
        self._override: Callable[[str, str, Any], str] | None = None  # Wan2GP's set_override, once it is imported
        self._init_lock = threading.Lock()
        self._lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}

    # ------------------------------------------------------------------ EngineAdapter

    def capabilities(self) -> list[EngineModel]:
        """Wan2GP's model types; one is available when all files of at least one of Wan2GP's transformer
        quantizations are present (the worker has no registry, so these are the quantizations it can ask about)."""
        session = self._ensure_session()
        wgp = self._wgp()
        quantizations = self._quantizations(wgp)
        definitions = session.list_model_defs()
        models = []
        with self._in_root():
            deps = wgp._get_dropdown_deps()
            for definition in definitions:
                model_type = str(definition["model_type"])
                metadata = definition.get("metadata") or {}
                models.append(
                    EngineModel(
                        engine_model=model_type,
                        name=str(definition.get("name") or model_type),
                        available=any(self._files_present(wgp, deps, model_type, quantization) for quantization in quantizations),
                        outputs=tuple(str(kind) for kind in metadata.get("outputs") or metadata.get("main_output") or ()),
                    )
                )
        return models

    def prepare(self, engine_model: str, quant: str) -> None:
        """Check that Wan2GP finds every file the model needs in quantization `quant`, since it would download a missing
        one at generation. Wan2GP loads the model on the first generation.

        A transformer quantization of Wan2GP (bf16, int8, fp8) is checked as `submit` makes Wan2GP load it, with the
        dtype policy `load_models` uses; any other quantization (GGUF) has its own model type, which Wan2GP loads with
        its global default. It refuses a quantization Wan2GP would load from another file than the one the registry
        pins: an int8 that resolves to the bf16 file (no quantized kernels), or a dtype policy that picks Wan 2.2's
        fp16 files, of which the registry pins none.
        """
        self._ensure_session()
        wgp = self._wgp()
        routed = quant in self._quantizations(wgp)
        with self._in_root():
            deps = wgp._get_dropdown_deps()
            if deps.get_model_def(engine_model) is None:
                raise EngineError("model_unavailable", f"Wan2GP does not know the model type {engine_model}.")
            if routed:
                self._check_selectable(deps, engine_model, quant)
            present = self._files_present(wgp, deps, engine_model, quant if routed else deps.transformer_quantization)
        if not present:
            raise EngineError(
                "model_unavailable",
                f"The {quant} files of Wan2GP model {engine_model} are not all installed; "
                "install the model before running it (the worker never downloads model files).",
            )

    def submit(self, job: EngineJob) -> str:
        self.prepare(job.engine_model, job.quant)  # also guards against a download started by generation
        session = self._ensure_session()
        wgp = self._wgp()
        defaults = session.get_default_settings(job.engine_model)
        model_def = wgp.get_model_def(job.engine_model) or {}
        if self._override is None:  # imported with the session
            raise EngineError("engine_unavailable", f"Wan2GP's {OVERRIDES_MODULE}.set_override is not loaded.")
        settings = build_settings(defaults, job, override=self._override, quantizations=self._quantizations(wgp), model_def=model_def)
        with self._lock:
            if any(record.status.state in ("queued", "running") for record in self._jobs.values()):
                raise EngineError("engine_busy", "Wan2GP runs one job at a time and is busy.")
        try:
            handle = session.submit_task(settings)
        except (ValueError, TypeError) as exc:
            raise EngineError("invalid_request", f"Wan2GP rejected the job settings: {exc}") from exc
        output_dir = job.output_dir or str(self._output_dir)
        status = EngineStatus(state="queued", accepted_extra=accepted_extra(defaults, job, model_def))
        record = _Job(handle=handle, output_dir=output_dir, status=status, image=_image_job(job))
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
        """Return the intent unchanged: the core translates the prompt into the model's dialect before it leases
        the job (`cinewright.core.dialect.translate_prompt`), since the worker has no registry."""
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
                try:
                    self._override = importlib.import_module(OVERRIDES_MODULE).set_override
                except (ImportError, AttributeError):
                    raise EngineError(
                        "engine_unavailable",
                        f"Wan2GP is not started because its checkout has no {OVERRIDES_MODULE}.set_override, "
                        "which the adapter needs to run each job in its routed quantization.",
                    ) from None
                self._config_path.parent.mkdir(parents=True, exist_ok=True)
                self._output_dir.mkdir(parents=True, exist_ok=True)
                self._session = api.init(
                    root=self._root,
                    config_path=self._config_path,
                    output_dir=self._output_dir,
                    cli_args=list(self._cli_args),
                    console_output=False,
                )
                commit = self._provisioning.get("commit") if self._provisioning else None
                if isinstance(commit, str) and commit:
                    self._version = commit  # the pinned commit provisioning installed, so provenance names it
                else:  # a checkout set up by hand, e.g. for the live probe
                    version = getattr(sys.modules.get("wgp"), "WanGP_version", None)
                    self._version = str(version) if version is not None else None
            return self._session

    @staticmethod
    def _wgp() -> Any:
        """Wan2GP's main module, which shared.api's `init()` imported."""
        module = sys.modules.get("wgp")
        if module is None:
            raise EngineError("engine_unavailable", "Wan2GP's module wgp is not loaded.")
        return module

    @staticmethod
    def _quantizations(wgp: Any) -> tuple[str, ...]:
        """Wan2GP's transformer quantizations, the values its `overrides` setting accepts."""
        return tuple(str(value) for _label, value in wgp.transformer_quantization_choices)

    @contextlib.contextmanager
    def _in_root(self) -> Iterator[None]:
        """Run Wan2GP's file lookups from its root, as shared.api does: its checkpoint roots may be relative."""
        previous = os.getcwd()
        os.chdir(self._root)
        try:
            yield
        finally:
            os.chdir(previous)

    @staticmethod
    def _files_present(wgp: Any, deps: Any, model_type: str, quantization: str) -> bool:
        """Whether Wan2GP finds every file of `model_type` in `quantization`: its own availability check
        (`model_dropdowns.get_model_download_status`) with the quantization passed in instead of the global one, and
        with the dtype policy `load_models` uses (the module global), not the one in Wan2GP's config."""
        policy = deps.transformer_dtype_policy
        names = _transformer_files(deps, model_type, quantization, policy)
        if not names or any(deps.get_local_model_filename(name) is None for name in names):
            return False
        return bool(wgp.model_dropdowns.has_secondary_model_files_for_status(deps, model_type, quantization, policy))

    @staticmethod
    def _check_selectable(deps: Any, model_type: str, quant: str) -> None:
        """Refuse a transformer quantization that Wan2GP would load from a file the registry does not pin for it."""
        policy = deps.transformer_dtype_policy
        names = _transformer_files(deps, model_type, quant, policy)
        choices = deps.get_model_recursive_prop(model_type, "URLs", return_list=True)
        several = isinstance(choices, list) and len(choices) > 1
        if quant != "bf16" and several and names == _transformer_files(deps, model_type, "bf16", policy):
            raise EngineError(
                "model_unavailable",
                f"Wan2GP cannot select the {quant} file of {model_type} and would load {_basenames(names)} instead "
                f"(its {quant} kernels are not available).",
            )
        if names != _transformer_files(deps, model_type, quant, "bf16"):  # the registry pins bf16-dtype files only
            raise EngineError(
                "model_unavailable",
                f"Wan2GP's transformer dtype policy {policy!r} selects {_basenames(names)} for {model_type} {quant}, "
                "a file Cinewright does not install.",
            )

    def _own_record(self, record: dict[str, Any] | None) -> dict[str, Any] | None:
        """The provisioning record if it describes this adapter's checkout, else None."""
        root = record.get("root") if record else None
        if not isinstance(root, str) or not root:
            return None
        try:
            return record if Path(root).resolve() == self._root else None
        except (OSError, ValueError):
            return None

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
                record.step = _denoising_step(data, record.step)
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
            videos = [output.name for output in outputs if not output.media_type.startswith("image/")]
            if record.image and videos:
                message = f"Wan2GP returned {', '.join(videos)[:200]} for an image job instead of an image."
                self._update(record, state="failed", phase="failed", error_code="engine_failed", error_message=message)
                return
            with self._lock:
                record.outputs = outputs
                record.status = dataclasses.replace(record.status, state="succeeded", phase="done", percent=100.0)
        elif result.cancelled or (record.handle.cancel_requested and not result.success):
            self._update(record, state="cancelled", phase="cancelled", error_code="cancelled", error_message="Cancelled on request.")
        else:
            code = "oom" if is_oom(messages + list(record.stream_tail)) else "engine_failed"
            message = messages[0] if messages else "Wan2GP finished without an output."
            if code == "oom":
                message = oom_message(record.step, message)
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

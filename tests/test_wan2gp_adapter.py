"""Wan2GP adapter against a fake `shared.api` session and a fake `wgp` module (no Wan2GP, no GPU), plus an optional
live probe.

The fakes follow the pinned commit where the adapter depends on it: Wan2GP's default settings carry the keys only the
adapter sets (`overrides`, `image_mode`, `image_guide`, `video_prompt_type`) and every key of the `params.extra`
allowlist, with the values of models/_settings.json and the handlers; a submitted task is validated as validate_task does
for what the adapter depends on (`validate`); `shared.utils.generation_overrides` writes and reads `overrides` as Wan2GP's
does; `wgp` picks file names per quantization and dtype policy as `get_model_filename` does and finds files only
relative to the working directory, through the checkpoint roots `ckpts` and `.`, so a lookup outside the Wan2GP root
finds nothing; its global quantization is int8, and the session's own availability (`get_model_availability`) checks
only that one.

The live probe runs only when CINEWRIGHT_WAN2GP_ROOT (a Wan2GP checkout) and CINEWRIGHT_WAN2GP_PYTHON
(its Python) are set. It imports the adapter in that interpreter and calls capabilities() and
health() with Hugging Face access switched off; it never generates and never loads model weights.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import json
import logging
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_worker_loop import FakeCore, start_loop, wait_closed
from test_worker_loop import make_job as worker_job

from cinewright_worker.adapter import EngineAdapter, EngineJob
from cinewright_worker.engines import EngineError
from cinewright_worker.engines.mock import render_png
from cinewright_worker.engines.wan2gp import (
    FFMPEG_BINARIES,
    FFMPEG_DIR,
    OVERRIDES_MODULE,
    PROVISIONING_RECORD,
    DenoisingStep,
    Wan2GPAdapter,
    accepted_extra,
    is_oom,
    oom_message,
)
from cinewright_worker.engines.wan2gp import build_settings as adapter_build_settings

COMMIT = "38758d3c73366422efcf609abb63cab8385f3552"

# The message wgp.py substitutes for a "CUDA out of memory" exception.
WAN2GP_VRAM_ERROR = (
    "The generation of the video has encountered an error: it is likely that you have unsufficient VRAM and you should "
    "therefore reduce the video resolution or its number of frames."
)
DEFAULTS = {
    "prompt": "default prompt",
    "negative_prompt": "",
    "seed": -1,
    "resolution": "1280x720",
    "num_inference_steps": 8,
    "video_length": 241,
    "duration_seconds": 0,
    "force_fps": "",
    "guidance_scale": 5,
    "guidance2_scale": 5,
    "guidance3_scale": 5,
    "guidance_phases": 1,
    "model_switch_phase": 1,
    "switch_threshold": 0,
    "switch_threshold2": 0,
    "embedded_guidance_scale": 6.0,
    "sample_solver": "",
    "flow_shift": 5.0,
    "denoising_strength": 1.0,
    "output_filename": "",
    "image_start": None,
    "image_end": None,
    # Settings only the adapter sets, with the values of models/_settings.json at the pinned commit.
    "overrides": "",
    "image_mode": 0,
    "image_prompt_type": "",
    "video_prompt_type": "",
    "image_guide": None,
}
# What get_factory_settings changes per model type: Z-Image is an image model, Wan 2.2 I2V starts from an image and
# runs two guidance phases (defaults/i2v_2_2.json), and the LTX-2 handler sets two (update_default_settings).
MODEL_DEFAULTS = {
    "z_image": {"image_mode": 1},
    "i2v_2_2": {"image_prompt_type": "S", "guidance_phases": 2},
    "ltx2_22B_distilled": {"guidance_phases": 2},
    "ltx2_22B_distilled_gguf_q6_k": {"guidance_phases": 2},
}
MODEL_DEFS = [
    {"model_type": "z_image", "name": "Z-Image Turbo 6B", "metadata": {"main_output": ["image"], "outputs": ["image"]}},
    {
        "model_type": "ltx2_22B_distilled",
        "name": "LTX-2 2.3 Distilled 1.0 22B",
        "metadata": {"main_output": ["video"], "outputs": ["video", "audio"]},
    },
    {"model_type": "i2v_2_2", "name": "Wan2.2 Image2video 14B", "metadata": {"main_output": ["video"], "outputs": ["video"]}},
    {
        "model_type": "ltx2_22B_distilled_gguf_q6_k",
        "name": "LTX-2 2.3 Distilled 1.0 22B Q6_K",
        "metadata": {"main_output": ["video"], "outputs": ["video", "audio"]},
    },
]
HF = "https://huggingface.co/DeepBeepMeep"
_LTX_SHARED = {
    "dtype": "bf16",
    "guidance_max_phases": 2,
    "text_encoder_URLs": ["gemma-3-12b-it_bf16.safetensors", "gemma-3-12b-it_quanto_bf16_int8.safetensors"],
    "text_encoder_folder": "gemma-3-12b-it",
    "VAE_URLs": ["ltx-2.3-22b_vae.safetensors"],
}
_WAN_SHARED = {
    "guidance_max_phases": 3,
    "text_encoder_URLs": ["umt5-xxl-enc-bf16.safetensors", "umt5-xxl-enc-quanto_int8.safetensors"],
    "text_encoder_folder": "umt5-xxl",
    "VAE_URLs": ["Wan2.1_VAE.safetensors"],
}
# Model definitions as wgp.get_model_def returns them: the transformer choices of defaults/*.json, the dtype a handler
# pins (LTX-2 and Z-Image), the guidance phases a handler allows and the secondary files
# has_secondary_model_files_for_status checks.
WGP_DEFS: dict[str, dict[str, Any]] = {
    "z_image": {
        "dtype": "bf16",
        "guidance_max_phases": 0,  # Turbo
        "URLs": [
            f"{HF}/Z-Image/resolve/main/ZImageTurbo_bf16.safetensors",
            f"{HF}/Z-Image/resolve/main/ZImageTurbo_quanto_bf16_int8.safetensors",
        ],
        "text_encoder_URLs": ["qwen3_4b_bf16.safetensors", "qwen3_4b_quanto_bf16_int8.safetensors"],
        "text_encoder_folder": "Qwen3",
        "VAE_URLs": ["ZImageTurbo_VAE_bf16.safetensors"],
    },
    "ltx2_22B_distilled": {
        **_LTX_SHARED,
        "URLs": [
            f"{HF}/LTX-2/resolve/main/ltx-2.3-22b-distilled_diffusion_model.safetensors",
            f"{HF}/LTX-2/resolve/main/ltx-2.3-22b-distilled_diffusion_model_quanto_int8.safetensors",
        ],
    },
    "ltx2_22B_distilled_gguf_q6_k": {**_LTX_SHARED, "URLs": [f"{HF}/LTX-2/resolve/main/ltx-2.3-22b-distilled-Q6_K_light.gguf"]},
    "i2v_2_2": {
        **_WAN_SHARED,
        "URLs": [
            f"{HF}/Wan2.2/resolve/main/wan2.2_image2video_14B_high_mbf16.safetensors",
            f"{HF}/Wan2.2/resolve/main/wan2.2_image2video_14B_high_quanto_mbf16_int8.safetensors",
            f"{HF}/Wan2.2/resolve/main/wan2.2_image2video_14B_high_quanto_mfp16_int8.safetensors",
        ],
        "URLs2": [
            f"{HF}/Wan2.2/resolve/main/wan2.2_image2video_14B_low_mbf16.safetensors",
            f"{HF}/Wan2.2/resolve/main/wan2.2_image2video_14B_low_quanto_mbf16_int8.safetensors",
            f"{HF}/Wan2.2/resolve/main/wan2.2_image2video_14B_low_quanto_mfp16_int8.safetensors",
        ],
    },
}
STATUS = {0: "missing", 1: "partial", 2: "available"}  # model_dropdowns.MODEL_FILE_STATUS_*
# The names the task setting `overrides` takes and their defaults, in the order Wan2GP writes them.
OVERRIDE_DEFAULTS = {
    "memory_profile": -1,
    "compilation": "",
    "attention_mode": "",
    "transformer_quantization": "",
    "text_encoder_quantization": "",
}


def parse_overrides(text: str) -> dict[str, Any]:
    """Stand-in for Wan2GP's parse_overrides: Python keyword arguments with literal values and known names only."""
    if not isinstance(text, str):
        raise TypeError("overrides must be a string")
    if not text.strip():
        return {}
    try:
        call = ast.parse(f"f({text})", mode="eval").body
        values = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
    except (SyntaxError, ValueError, AttributeError) as exc:
        raise ValueError(f"malformed overrides {text!r}") from exc
    if call.args or len(values) != len(call.keywords) or not set(values) <= set(OVERRIDE_DEFAULTS):
        raise ValueError(f"malformed or unknown overrides {text!r}")
    return values


def set_override(text: str, name: str, value: Any) -> str:
    """Stand-in for Wan2GP's set_override: `name=repr(value)` pairs joined by ", " in Wan2GP's order, without defaults,
    the other overrides kept."""
    values = {**parse_overrides(text), name: value}
    return ", ".join(f"{key}={values[key]!r}" for key, default in OVERRIDE_DEFAULTS.items() if values.get(key, default) != default)


def overrides_module() -> types.ModuleType:
    module = types.ModuleType(OVERRIDES_MODULE)
    module.OVERRIDE_DEFAULTS = dict(OVERRIDE_DEFAULTS)
    module.parse_overrides = parse_overrides
    module.set_override = set_override
    return module


def build_settings(defaults: dict, job: EngineJob, **kwargs) -> dict:
    """The adapter's build_settings with the stand-in for Wan2GP's set_override."""
    return adapter_build_settings(defaults, job, override=set_override, **kwargs)


def defaults_of(model_type: str) -> dict:
    """Wan2GP's default settings of a model type (shared.api get_default_settings)."""
    return {**DEFAULTS, **MODEL_DEFAULTS.get(model_type, {}), "model_type": model_type}


def basename(name: str) -> str:
    return name.replace("\\", "/").rsplit("/", 1)[-1]


@contextlib.contextmanager
def pushd(path: Path) -> Iterator[None]:
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@dataclasses.dataclass
class FakeDeps:
    """model_dropdowns.DropdownDeps: a snapshot of wgp's globals plus its lookup functions."""

    server_config: dict
    transformer_quantization: str
    transformer_dtype_policy: str
    text_encoder_quantization: str
    get_model_def: Callable
    get_model_recursive_prop: Callable
    get_model_filename: Callable
    get_local_model_filename: Callable
    get_transformer_dtype: Callable


class FakeWgp(types.ModuleType):
    """The parts of Wan2GP's `wgp` module the adapter uses, with the signatures of the pinned commit."""

    def __init__(self) -> None:
        super().__init__("wgp")
        self.WanGP_version = "19.292"
        self.transformer_quantization_choices = [
            ("Scaled Int8 (recommended)", "int8"),
            ("Scaled Fp8", "fp8"),
            ("16-bit (no quantization)", "bf16"),
        ]
        self.server_config: dict[str, Any] = {"transformer_quantization": "int8", "transformer_dtype_policy": ""}
        self.transformer_quantization = "int8"  # module globals, read from server_config at import
        self.transformer_dtype_policy = ""
        self.text_encoder_quantization = "int8"
        self.quantized_kernels = True  # mmgp's quanto qtype aliases; without them int8 resolves to the bf16 file
        self.bf16_supported = True
        self.lookups: list[str] = []  # the working directory of every file lookup
        self.model_dropdowns = SimpleNamespace(
            has_secondary_model_files_for_status=self.has_secondary_model_files_for_status,
            get_model_download_status=self.get_model_download_status,
        )

    def get_model_def(self, model_type: str) -> dict | None:
        return WGP_DEFS.get(model_type)

    def get_model_recursive_prop(self, model_type, prop="URLs", sub_prop_name=None, return_list=True, stack=(), model_def=None):
        model_def = model_def or WGP_DEFS.get(model_type)
        if model_def is None:
            raise Exception(f"Unknown model type '{model_type}'")
        return model_def.get(prop) or []

    def get_transformer_dtype(self, model_type: str, transformer_dtype_policy: str) -> str:
        pinned = (WGP_DEFS.get(model_type) or {}).get("dtype")
        if pinned:
            return pinned
        if transformer_dtype_policy == "":
            return "bf16" if self.bf16_supported else "fp16"
        return "fp16" if transformer_dtype_policy == "fp16" else "bf16"

    def get_model_filename(
        self, model_type, quantization="int8", dtype_policy="", module_type=None, submodel_no=1, URLs=None, stack=(), model_def=None
    ):
        if URLs is None:
            model_def = model_def or WGP_DEFS.get(model_type)
            if model_def is None:
                return ""
            URLs = model_def.get("URLs" if submodel_no <= 1 else f"URLs{submodel_no}", [])
        choices = URLs if isinstance(URLs, list) else [URLs]
        if not choices:
            return ""
        if len(choices) == 1:
            return choices[0]
        order = {"int8": ["int8", "fp8"], "fp8": ["fp8", "int8"]}.get(quantization or "bf16", [])
        tokens = order if self.quantized_kernels else []
        matches = [name for token in tokens for name in choices if token in basename(name).lower()]
        if not matches:
            return choices[0]
        dtype = self.get_transformer_dtype(model_type, dtype_policy)
        return next((name for name in matches if dtype in basename(name)), matches[0])

    def get_local_model_filename(self, model_filename, use_locator=True, extra_paths=None, lora_dir=None):
        self.lookups.append(os.getcwd())
        name = basename(model_filename)
        extras = [] if extra_paths is None else extra_paths if isinstance(extra_paths, list) else [extra_paths]
        for candidate in [os.path.join(extra, name) for extra in extras] + [name]:
            for root in self.server_config.get("checkpoints_paths", ["ckpts", "."]):
                path = os.path.join(root, candidate)
                if os.path.isfile(path):
                    return path
        return None

    def _get_dropdown_deps(self) -> FakeDeps:
        return FakeDeps(
            server_config=self.server_config,
            transformer_quantization=self.transformer_quantization,
            transformer_dtype_policy=self.transformer_dtype_policy,
            text_encoder_quantization=self.text_encoder_quantization,
            get_model_def=self.get_model_def,
            get_model_recursive_prop=self.get_model_recursive_prop,
            get_model_filename=self.get_model_filename,
            get_local_model_filename=self.get_local_model_filename,
            get_transformer_dtype=self.get_transformer_dtype,
        )

    def has_secondary_model_files_for_status(self, deps, model_type, quantization, dtype_policy) -> bool:
        model_def = deps.get_model_def(model_type)
        if model_def is None:
            return True
        encoders = model_def.get("text_encoder_URLs")
        if encoders:  # the text encoder of the global variant, not of `quantization`
            name = deps.get_model_filename(
                model_type=model_type, quantization=deps.text_encoder_quantization, dtype_policy=dtype_policy, URLs=encoders
            )
            if deps.get_local_model_filename(name, extra_paths=model_def.get("text_encoder_folder")) is None:
                return False
        return all(deps.get_local_model_filename(url) is not None for url in model_def.get("VAE_URLs", []))

    def get_model_download_status(self, deps, model_type) -> int:
        """Wan2GP's availability: the files of its configured quantization only."""
        quantization = deps.server_config.get("transformer_quantization", deps.transformer_quantization)
        policy = deps.server_config.get("transformer_dtype_policy", deps.transformer_dtype_policy)
        model_def = deps.get_model_def(model_type)
        names = [deps.get_model_filename(model_type, quantization=quantization, dtype_policy=policy)]
        if "URLs2" in model_def:
            names.append(deps.get_model_filename(model_type, quantization=quantization, dtype_policy=policy, submodel_no=2))
        found = [deps.get_local_model_filename(name) is not None for name in names]
        if all(found):
            return 2 if self.has_secondary_model_files_for_status(deps, model_type, quantization, policy) else 1
        others = model_def.get("URLs", []) + model_def.get("URLs2", [])
        return 1 if any(found) or any(deps.get_local_model_filename(name) is not None for name in others) else 0


def install(
    root: Path, model_type: str, quant: str = "bf16", *, experts: tuple[int, ...] = (1, 2), text_encoder: str | None = "int8"
) -> None:
    """Put the files Wan2GP loads for `model_type` in `quant` (bf16 dtype) into the checkout's `ckpts`, as the model
    store's install leaves them: the transformer (each listed expert of a two-expert model), the text encoder in the
    variant `text_encoder` (None: none) and the VAE."""
    naming, model_def, ckpts = FakeWgp(), WGP_DEFS[model_type], root / "ckpts"
    names = [naming.get_model_filename(model_type, quant, "bf16", submodel_no=no) for no in experts if no == 1 or "URLs2" in model_def]
    if text_encoder is not None:
        encoder = naming.get_model_filename(model_type, text_encoder, "bf16", URLs=model_def["text_encoder_URLs"])
        names.append(f"{model_def['text_encoder_folder']}/{basename(encoder)}")
    for name in names + model_def["VAE_URLs"]:
        path = ckpts / name.removeprefix(f"{HF}/").split("/resolve/main/")[-1]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


class FakeStream:
    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()

    def put(self, kind: str, data: object = None) -> None:
        self._queue.put(SimpleNamespace(kind=kind, data=data, timestamp=time.time()))

    def close(self) -> None:
        self._queue.put(None)

    def iter(self, timeout: float | None = None):
        while True:
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                continue
            if item is None:
                return
            yield item


class FakeJob:
    def __init__(self) -> None:
        self.events = FakeStream()
        self.cancel_requested = False
        self._result = None
        self._done = threading.Event()

    def cancel(self) -> None:
        self.cancel_requested = True

    def finish(self, success: bool, files: list[str] = (), errors: list[str] = (), cancelled: bool = False) -> None:
        self._result = SimpleNamespace(
            success=success, generated_files=list(files), errors=[SimpleNamespace(message=m) for m in errors], cancelled=cancelled
        )
        self._done.set()
        self.events.close()

    def result(self, timeout: float | None = None):
        self._done.wait(timeout)
        return self._result


def progress(phase: str, step: int | None, total: int | None, raw_phase: str | None = None, unit: str | None = None, percent: int = 0):
    """A progress event's data, with the fields of shared.api's ProgressUpdate."""
    fields = ("status", "task_id", "task_index", "task_count", "window_index", "window_count")
    return SimpleNamespace(
        phase=phase, progress=percent, current_step=step, total_steps=total, raw_phase=raw_phase, unit=unit, **dict.fromkeys(fields)
    )


class FakeSession:
    def __init__(self, init_kwargs: dict, wgp: FakeWgp) -> None:
        self.init_kwargs = init_kwargs
        self.wgp = wgp
        self.hf_hub_offline = os.environ.get("HF_HUB_OFFLINE")
        self.submitted: list[tuple[dict, FakeJob]] = []

    def list_model_defs(self) -> list[dict]:
        return [dict(definition) for definition in MODEL_DEFS]

    def list_model_availability(self) -> list[dict]:
        return [self.get_model_availability(d["model_type"]) for d in MODEL_DEFS]

    def get_model_def(self, model_type: str) -> dict | None:
        return next((dict(d) for d in MODEL_DEFS if d["model_type"] == model_type), None)

    def get_model_availability(self, model_type: str) -> dict:
        with pushd(self.init_kwargs["root"]):
            status = self.wgp.model_dropdowns.get_model_download_status(self.wgp._get_dropdown_deps(), model_type)
        return {"model_type": model_type, "status": STATUS[status], "available": status == 2}

    def get_default_settings(self, model_type: str) -> dict:
        return defaults_of(model_type)

    def submit_task(self, settings: dict) -> FakeJob:
        """Record the task as Wan2GP runs it, validated at submission rather than when generation starts."""
        job = FakeJob()
        self.submitted.append((validate(settings, self.wgp), job))
        return job


def validate(settings: dict, wgp: FakeWgp, model_def: dict | None = None) -> dict:
    """The settings as Wan2GP's validate_task leaves them, for what the adapter depends on (validate_settings at the
    pinned commit): `overrides` must parse, with a transformer quantization among Wan2GP's choices; without "V" in
    `video_prompt_type` the control image is dropped; `denoising_strength` is 1.0 unless "V" and "G" are set;
    `guidance_phases` is capped at the model's maximum; `model_switch_phase` is 1 unless there are three phases.
    `model_def` stands in for Wan2GP's definition of the task's model type."""
    settings = dict(settings)
    quantization = parse_overrides(settings.get("overrides", "")).get("transformer_quantization", "")
    if quantization and quantization not in [value for _label, value in wgp.transformer_quantization_choices]:
        raise ValueError(f"Unknown transformer quantization {quantization!r}")
    model_def = model_def or wgp.get_model_def(settings["model_type"]) or {}
    video_prompt_type = str(settings.get("video_prompt_type") or "")
    if "V" not in video_prompt_type:
        settings["image_guide"] = None
    if "V" not in video_prompt_type or "G" not in video_prompt_type:
        settings["denoising_strength"] = 1.0
    most = model_def.get("guidance_max_phases", 0)
    settings["guidance_phases"] = most if model_def.get("lock_guidance_phases") else min(settings["guidance_phases"], most)
    if settings["guidance_phases"] != 3:
        settings["model_switch_phase"] = 1
    return settings


class FakeImage:
    def __init__(self) -> None:
        self.size = (1600, 400)

    def copy(self) -> FakeImage:
        return self

    def thumbnail(self, size: tuple[int, int]) -> None:
        self.size = (min(self.size[0], size[0]), min(self.size[1], size[1]))

    def save(self, path: str, format: str) -> None:  # noqa: A002 - PIL's signature
        assert format == "PNG"
        Path(path).write_bytes(render_png(1, 8, 2))


def fake_wan2gp(monkeypatch: pytest.MonkeyPatch) -> list[FakeSession]:
    """Put the fake shared.api and wgp modules in place; returns the sessions shared.api's `init()` creates."""
    sessions: list[FakeSession] = []
    api = types.ModuleType("shared.api")
    wgp = FakeWgp()

    def init(**kwargs) -> FakeSession:
        sessions.append(FakeSession(kwargs, wgp))
        return sessions[-1]

    api.init = init
    monkeypatch.setitem(sys.modules, "shared", types.ModuleType("shared"))
    monkeypatch.setitem(sys.modules, "shared.api", api)
    monkeypatch.setitem(sys.modules, "shared.utils", types.ModuleType("shared.utils"))
    monkeypatch.setitem(sys.modules, OVERRIDES_MODULE, overrides_module())
    monkeypatch.setitem(sys.modules, "wgp", wgp)
    monkeypatch.setitem(sys.modules, "torch", None)  # no torch in the core venv: VRAM figures stay null
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")  # restored afterwards; the adapter sets it to 1
    return sessions


@pytest.fixture
def fake_api(monkeypatch):
    return fake_wan2gp(monkeypatch)


@pytest.fixture
def wgp(fake_api) -> FakeWgp:
    return sys.modules["wgp"]


def provision_ffmpeg(root: Path, names: tuple[str, ...] = FFMPEG_BINARIES) -> None:
    (root / FFMPEG_DIR).mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / FFMPEG_DIR / name).write_bytes(b"")


@pytest.fixture
def adapter(tmp_path) -> Wan2GPAdapter:
    """An adapter on a checkout with Z-Image and Wan 2.2 I2V installed in bf16 (not in Wan2GP's default int8)."""
    root = tmp_path / "wan2gp"
    provision_ffmpeg(root)
    install(root, "z_image")
    install(root, "i2v_2_2")
    return Wan2GPAdapter(root=str(root), data_dir=str(tmp_path / "data"))


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(f"network access attempted: {args!r}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def make_job(
    output_dir: Path,
    engine_model: str = "i2v_2_2",
    inputs: tuple[str, ...] = (),
    *,
    capability: str = "i2v",
    quant: str = "bf16",
    **params,
) -> EngineJob:
    return EngineJob(
        job_id="job-1",
        engine_model=engine_model,
        quant=quant,
        prompt="A lighthouse at dusk",
        negative_prompt="blurry",
        seed=1234,
        params=params,
        input_paths=inputs,
        output_dir=str(output_dir),
        capability=capability,
    )


def wait_state(adapter: Wan2GPAdapter, engine_job_id: str, states: tuple[str, ...], timeout: float = 5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = adapter.status(engine_job_id)
        if status.state in states:
            return status
        time.sleep(0.01)
    raise AssertionError(f"status never reached {states}: {adapter.status(engine_job_id)}")


# ---------------------------------------------------------------------- tests


def test_import_and_construction_need_no_wan2gp(adapter):
    assert isinstance(adapter, EngineAdapter)
    assert "shared.api" not in sys.modules or not hasattr(sys.modules["shared.api"], "WanGPSession")
    assert adapter.health().state == "starting"


def test_init_runs_once_with_cinewright_paths(fake_api, adapter, tmp_path):
    adapter.capabilities()
    adapter.prepare("z_image", "bf16")
    assert len(fake_api) == 1
    kwargs = fake_api[0].init_kwargs
    assert kwargs["root"] == (tmp_path / "wan2gp").resolve()
    assert kwargs["config_path"] == (tmp_path / "data" / "engines" / "wan2gp" / "wgp_config.json").resolve()
    assert kwargs["output_dir"] == (tmp_path / "data" / "engines" / "wan2gp" / "outputs").resolve()
    assert kwargs["cli_args"] == ["--attention", "sdpa", "--offline"]
    assert kwargs["console_output"] is False
    assert fake_api[0].hf_hub_offline == "1"  # set before shared.api (and huggingface_hub) is imported
    assert str((tmp_path / "wan2gp").resolve()) in sys.path


@pytest.mark.parametrize("present", [(), FFMPEG_BINARIES[:2]])
def test_missing_ffmpeg_refuses_to_start_wan2gp(fake_api, no_network, tmp_path, present):
    root = tmp_path / "wan2gp"
    if present:
        provision_ffmpeg(root, present)
    adapter = Wan2GPAdapter(root=str(root), data_dir=str(tmp_path / "data"))
    for call in (adapter.capabilities, lambda: adapter.prepare("z_image", "bf16")):
        with pytest.raises(EngineError) as raised:
            call()
        assert raised.value.code == "engine_unavailable"
        assert FFMPEG_BINARIES[2] in raised.value.message and FFMPEG_DIR in raised.value.message
    assert fake_api == []  # shared.api's init (which would download ffmpeg) never ran
    assert str(root.resolve()) not in sys.path
    assert adapter.health().state == "starting"


def test_capabilities_and_health(fake_api, adapter):
    models = {m.engine_model: m for m in adapter.capabilities()}
    assert set(models) == {"z_image", "ltx2_22B_distilled", "i2v_2_2", "ltx2_22B_distilled_gguf_q6_k"}
    assert models["ltx2_22B_distilled"].name == "LTX-2 2.3 Distilled 1.0 22B"
    assert models["ltx2_22B_distilled"].outputs == ("video", "audio")
    assert models["ltx2_22B_distilled"].available is False
    # Only bf16 is installed, not Wan2GP's default int8, whose files shared.api's availability checks.
    assert models["z_image"].available is True and models["i2v_2_2"].available is True
    assert fake_api[0].get_model_availability("z_image")["status"] == "partial"
    health = adapter.health()
    assert (health.state, health.engine_version, health.vram_total_mb) == ("ready", "19.292", None)


def write_record(data_dir: Path, checkout: Path, **fields) -> None:
    """The record engine provisioning writes once the engine is installed (cinewright.core.provisioning)."""
    engine_dir = data_dir / "engines" / "wan2gp"
    engine_dir.mkdir(parents=True, exist_ok=True)
    record = {"engine": "wan2gp", "commit": COMMIT, "root": str(checkout), "attention": "sdpa", **fields}
    (engine_dir / PROVISIONING_RECORD).write_text(json.dumps(record), encoding="utf-8")


def test_engine_version_is_the_provisioned_commit(fake_api, tmp_path):
    root = tmp_path / "wan2gp"
    provision_ffmpeg(root)
    write_record(tmp_path / "data", root)
    adapter = Wan2GPAdapter(root=str(root), data_dir=str(tmp_path / "data"))
    adapter.capabilities()
    assert adapter.health().engine_version == COMMIT  # not WanGP_version, so job provenance names the commit


@pytest.mark.parametrize(
    ("fields", "attention", "version"),
    [
        (None, "sdpa", "19.292"),  # no provisioning record
        ({}, "sdpa", COMMIT),  # the probe found SageAttention not working
        ({"attention": "sage2"}, "sage2", COMMIT),
        ({"attention": "sage"}, "sage", COMMIT),
        ({"attention": "sage2 --share"}, "sdpa", COMMIT),  # only known modes become a command-line argument
        ({"attention": None}, "sdpa", COMMIT),
        ({"attention": "sage2", "root": "elsewhere"}, "sdpa", "19.292"),  # a record of another checkout
    ],
)
def test_attention_mode_comes_from_the_provisioning_record(fake_api, tmp_path, fields, attention, version):
    root = tmp_path / "wan2gp"
    provision_ffmpeg(root)
    if fields is not None:
        write_record(tmp_path / "data", root, **fields)
    adapter = Wan2GPAdapter(root=str(root), data_dir=str(tmp_path / "data"))
    adapter.capabilities()
    assert fake_api[0].init_kwargs["cli_args"] == ["--attention", attention, "--offline"]
    assert adapter.health().engine_version == version


def test_an_unreadable_provisioning_record_keeps_sdpa(fake_api, tmp_path):
    root = tmp_path / "wan2gp"
    provision_ffmpeg(root)
    record = tmp_path / "data" / "engines" / "wan2gp" / PROVISIONING_RECORD
    record.parent.mkdir(parents=True)
    record.write_text("{not json", encoding="utf-8")
    adapter = Wan2GPAdapter(root=str(root), data_dir=str(tmp_path / "data"))
    adapter.capabilities()
    assert fake_api[0].init_kwargs["cli_args"] == ["--attention", "sdpa", "--offline"]


def test_missing_or_unknown_models_are_refused(fake_api, adapter, tmp_path):
    for model in ("ltx2_22B_distilled", "no_such_model"):
        with pytest.raises(EngineError) as raised:
            adapter.prepare(model, "bf16")
        assert raised.value.code == "model_unavailable"
    with pytest.raises(EngineError):
        adapter.submit(make_job(tmp_path, engine_model="ltx2_22B_distilled"))
    assert fake_api[0].submitted == []


def test_settings_mapping(tmp_path):
    start, end = str(tmp_path / "start.png"), str(tmp_path / "end.jpg")
    job = make_job(
        tmp_path,
        inputs=(start, end),
        resolution="832x480",
        steps=20,
        frames=81,
        fps=16,
        guidance=3.5,
        duration_s=5.0,
        extra={"flow_shift": 7.0, "image_start": "C:/secret.png", "output_filename": "x", "unknown_key": 1, "_api": {}},
    )
    settings = build_settings({**DEFAULTS, "model_type": "i2v_2_2"}, job)
    assert settings["model_type"] == "i2v_2_2"
    assert (settings["prompt"], settings["negative_prompt"], settings["seed"]) == ("A lighthouse at dusk", "blurry", 1234)
    assert (settings["resolution"], settings["num_inference_steps"], settings["guidance_scale"]) == ("832x480", 20, 3.5)
    assert (settings["video_length"], settings["force_fps"], settings["duration_seconds"]) == (81, 16, 5.0)
    assert (settings["image_start"], settings["image_end"]) == (start, end)
    assert settings["flow_shift"] == 7.0
    assert settings["output_filename"] == "" and "unknown_key" not in settings and "_api" not in settings


# Keys and values from the review's end-to-end probe plus other non-plain settings of Wan2GP's defaults.
FORBIDDEN_EXTRA = {
    "spatial_upsampling": "flashvsr",
    "temporal_upsampling": "rife2",
    "postprocess_audio": "mmaudio",
    "postprocessing_reference_images": ["C:/Users/victim/secret.png"],
    "spatial_postprocessing_parameters": {"model": "x"},
    "prompt_enhancer": "TI",
    "activated_loras": ["C:/loras/evil.safetensors"],
    "loras_multipliers": "1.0",
    "image_refs": ["C:/secret.png"],
    "video_source": "C:/secret.mp4",
    "output_filename": "../../escape",
    "model_type": "other_model",
}


def test_extra_settings_are_an_allowlist(tmp_path, caplog):
    defaults = {
        **DEFAULTS,
        **{key: "" for key in FORBIDDEN_EXTRA},
        "sample_solver": "",
        "guidance2_scale": 5,
        "embedded_guidance_scale": 6.0,
    }
    plain = {"flow_shift": 3.0, "sample_solver": "euler", "guidance2_scale": 4, "embedded_guidance_scale": 5.5, "negative_prompt": "noise"}
    bad_values = {"resolution": "C:/secret.png", "force_fps": ["24"], "num_inference_steps": True, "guidance_scale": {"x": 1}}
    caplog.set_level(logging.WARNING, logger="cinewright_worker")
    settings = build_settings(defaults, make_job(tmp_path, extra={**FORBIDDEN_EXTRA, **plain, **bad_values, "not_a_setting": 1}))
    assert {key: settings[key] for key in FORBIDDEN_EXTRA} == {
        **{key: "" for key in FORBIDDEN_EXTRA},
        "model_type": "i2v_2_2",
    }
    assert {key: settings[key] for key in ("flow_shift", "sample_solver", "guidance2_scale", "embedded_guidance_scale")} == {
        "flow_shift": 3.0,
        "sample_solver": "euler",
        "guidance2_scale": 4,
        "embedded_guidance_scale": 5.5,
    }
    assert settings["negative_prompt"] == "blurry"  # the job's own negative prompt wins over extra
    assert (settings["resolution"], settings["force_fps"], settings["num_inference_steps"], settings["guidance_scale"]) == (
        DEFAULTS["resolution"],
        DEFAULTS["force_fps"],
        DEFAULTS["num_inference_steps"],
        DEFAULTS["guidance_scale"],
    )
    assert "not_a_setting" not in settings
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "spatial_upsampling" in logged and "not_a_setting" in logged and "resolution" in logged
    assert "secret" not in logged  # key names only, never values
    no_negative = dataclasses.replace(make_job(tmp_path), negative_prompt=None, params={"extra": {"negative_prompt": "noise"}})
    assert build_settings(dict(DEFAULTS), no_negative)["negative_prompt"] == "noise"


def test_settings_defaults_and_duration(tmp_path):
    settings = build_settings({**DEFAULTS}, make_job(tmp_path, engine_model="ltx2_22B_distilled", duration_s=4))
    assert settings["video_length"] == "4s"  # Wan2GP turns seconds into a valid frame count
    assert (settings["resolution"], settings["num_inference_steps"], settings["image_start"]) == ("1280x720", 8, None)
    with pytest.raises(EngineError) as raised:
        build_settings({**DEFAULTS}, make_job(tmp_path, inputs=(str(tmp_path / "clip.mp4"),)))
    assert raised.value.code == "unsupported_input"


def test_events_drive_status_and_outputs_move_to_the_job(fake_api, adapter, tmp_path):
    job_dir, engine_out = tmp_path / "job", tmp_path / "wan2gp-out"
    job_dir.mkdir()
    engine_out.mkdir()
    engine_job_id = adapter.submit(make_job(job_dir, steps=8))
    settings, handle = fake_api[0].submitted[0]
    assert settings["num_inference_steps"] == 8

    handle.events.put("started", {"tasks": 1})
    handle.events.put("progress", SimpleNamespace(phase="inference", progress=54, current_step=4, total_steps=8))
    handle.events.put("preview", SimpleNamespace(image=FakeImage()))
    handle.events.put("stream", SimpleNamespace(stream="stdout", text="Denoising"))
    status = wait_state(adapter, engine_job_id, ("running",))
    deadline = time.monotonic() + 5
    while adapter.status(engine_job_id).preview_path is None and time.monotonic() < deadline:
        time.sleep(0.01)
    status = adapter.status(engine_job_id)
    assert (status.phase, status.step, status.total_steps, status.percent) == ("inference", 4, 8, 54.0)
    assert Path(status.preview_path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert Path(status.preview_path).parent == job_dir
    assert adapter.health().state == "busy"

    produced = engine_out / "clip_001.mp4"
    produced.write_bytes(b"mp4 bytes")
    handle.finish(success=True, files=[str(produced)])
    assert wait_state(adapter, engine_job_id, ("succeeded", "failed")).state == "succeeded"
    outputs = adapter.fetch_outputs(engine_job_id)
    assert [(Path(o.path).parent, o.name, o.media_type) for o in outputs] == [(job_dir, "clip_001.mp4", "video/mp4")]
    assert not produced.exists()


def test_oom_is_detected(fake_api, adapter, tmp_path):
    engine_job_id = adapter.submit(make_job(tmp_path))
    handle = fake_api[0].submitted[0][1]
    handle.events.put("error", SimpleNamespace(message=WAN2GP_VRAM_ERROR))
    handle.finish(success=False, errors=[WAN2GP_VRAM_ERROR])
    status = wait_state(adapter, engine_job_id, ("failed", "succeeded"))
    assert (status.state, status.error_code) == ("failed", "oom")
    assert status.error_message == f"Out of memory before the first denoising step. Wan2GP: {WAN2GP_VRAM_ERROR}"


def test_other_failures_are_engine_failed(fake_api, adapter, tmp_path):
    engine_job_id = adapter.submit(make_job(tmp_path))
    fake_api[0].submitted[0][1].finish(success=False, errors=["Task 1 failed validation"])
    status = wait_state(adapter, engine_job_id, ("failed", "succeeded"))
    assert (status.error_code, status.error_message) == ("engine_failed", "Task 1 failed validation")
    assert is_oom(["CUDA out of memory. Tried to allocate 2.00 GiB"]) and not is_oom(["Loading model"])


def test_cancel(fake_api, adapter, tmp_path):
    engine_job_id = adapter.submit(make_job(tmp_path))
    handle = fake_api[0].submitted[0][1]
    adapter.cancel(engine_job_id)
    assert handle.cancel_requested
    handle.finish(success=False, errors=["Generation was cancelled"], cancelled=True)
    assert wait_state(adapter, engine_job_id, ("cancelled", "failed")).state == "cancelled"


def test_one_job_at_a_time(fake_api, adapter, tmp_path):
    adapter.submit(make_job(tmp_path))
    with pytest.raises(EngineError) as raised:
        adapter.submit(make_job(tmp_path))
    assert raised.value.code == "engine_busy"


# ---------------------------------------------------------------------- the routed quantization


def checkout(tmp_path: Path, *installs: tuple[str, str]) -> Wan2GPAdapter:
    """An adapter on a checkout with exactly these (model type, quantization) installs."""
    root = tmp_path / "wan2gp"
    provision_ffmpeg(root)
    for model_type, quant in installs:
        install(root, model_type, quant)
    return Wan2GPAdapter(root=str(root), data_dir=str(tmp_path / "data"))


@pytest.mark.parametrize("quant", ["int8", "fp8", "bf16", "Q6_K"])
def test_the_task_overrides_come_from_wan2gps_set_override(tmp_path, quant):
    calls = []

    def override(text: str, name: str, value: Any) -> str:
        calls.append((text, name, value))
        return "written by set_override"

    defaults = {**DEFAULTS, "overrides": "attention_mode='sage2'"}
    settings = adapter_build_settings(defaults, make_job(tmp_path, quant=quant), override=override)
    if quant == "Q6_K":  # a GGUF quantization is its own model type
        assert (calls, settings["overrides"]) == ([], "attention_mode='sage2'")
    else:  # the other overrides of the defaults go in, so set_override keeps them
        assert calls == [("attention_mode='sage2'", "transformer_quantization", quant)]
        assert settings["overrides"] == "written by set_override"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", "transformer_quantization='int8'"),
        ("attention_mode='veda'", "attention_mode='veda', transformer_quantization='int8'"),
        ("transformer_quantization='bf16', memory_profile=4", "memory_profile=4, transformer_quantization='int8'"),
    ],
)
def test_the_stand_in_set_override_writes_what_wan2gp_writes(text, expected):
    """The tests' set_override follows Wan2GP's (shared/utils/generation_overrides.py at the pinned commit)."""
    assert set_override(text, "transformer_quantization", "int8") == expected


def test_unreadable_default_overrides_fail_the_engine_not_the_request(tmp_path):
    with pytest.raises(EngineError) as raised:  # Wan2GP's set_override refuses its own defaults: the engine is broken
        build_settings({**DEFAULTS, "overrides": "attention_mode="}, make_job(tmp_path))
    assert raised.value.code == "engine_failed"


def test_the_checkouts_own_set_override_writes_the_overrides(fake_api, adapter, tmp_path, monkeypatch):
    calls = []
    module = types.ModuleType(OVERRIDES_MODULE)
    module.set_override = lambda text, name, value: calls.append((text, name, value)) or f"{name}={value!r}"
    monkeypatch.setitem(sys.modules, OVERRIDES_MODULE, module)
    adapter.submit(make_job(tmp_path))
    assert calls == [("", "transformer_quantization", "bf16")]
    assert fake_api[0].submitted[0][0]["overrides"] == "transformer_quantization='bf16'"


def test_a_checkout_without_set_override_is_not_started(fake_api, adapter, monkeypatch):
    monkeypatch.delitem(sys.modules, OVERRIDES_MODULE)
    for call in (adapter.capabilities, lambda: adapter.prepare("z_image", "bf16")):
        with pytest.raises(EngineError) as raised:
            call()
        assert raised.value.code == "engine_unavailable"
        assert OVERRIDES_MODULE in raised.value.message
    assert fake_api == []  # shared.api's init never ran
    assert adapter.health().state == "starting"


def test_an_int8_job_reaches_wan2gp_with_its_quantization(fake_api, wgp, tmp_path):
    adapter = checkout(tmp_path, ("ltx2_22B_distilled", "int8"))
    wgp.transformer_quantization = wgp.server_config["transformer_quantization"] = "bf16"  # Wan2GP's own default
    image = tmp_path / "start.png"
    image.write_bytes(render_png(3, 4, 4))
    extra = {"denoising_strength": 0.7, "guidance_phases": 1}
    job = make_job(tmp_path / "job", "ltx2_22B_distilled", (str(image),), quant="int8", steps=8, extra=extra)

    engine_job_id = adapter.submit(job)

    settings, _ = fake_api[0].submitted[0]  # as Wan2GP runs it
    assert settings["overrides"] == "transformer_quantization='int8'"
    assert (settings["image_start"], settings["image_mode"], settings["guidance_phases"]) == (str(image), 0, 1)
    # Wan2GP runs denoising_strength only with a control image, so the adapter neither passes nor reports it.
    assert settings["denoising_strength"] == 1.0
    assert adapter.status(engine_job_id).accepted_extra == ("guidance_phases",)
    assert fake_api[0].get_model_availability("ltx2_22B_distilled")["available"] is False  # bf16, its default, is absent


def test_prepare_checks_the_routed_quantization_from_the_wan2gp_root(fake_api, wgp, adapter, tmp_path):
    cwd = os.getcwd()
    adapter.prepare("z_image", "bf16")  # Wan2GP's global quantization is int8, whose files are absent
    for quant in ("int8", "fp8"):  # fp8 resolves to the int8 file
        with pytest.raises(EngineError) as raised:
            adapter.prepare("z_image", quant)
        assert raised.value.code == "model_unavailable"
        assert f"{quant} files of Wan2GP model z_image" in raised.value.message
    assert fake_api[0].get_model_availability("z_image")["available"] is False
    assert wgp.lookups and {Path(path).resolve() for path in wgp.lookups} == {(tmp_path / "wan2gp").resolve()}
    assert os.getcwd() == cwd


@pytest.mark.parametrize("experts", [(1,), (2,)])
def test_a_two_expert_model_needs_both_experts(fake_api, tmp_path, experts):
    adapter = checkout(tmp_path)
    install(tmp_path / "wan2gp", "i2v_2_2", "int8", experts=experts)
    with pytest.raises(EngineError) as raised:
        adapter.prepare("i2v_2_2", "int8")
    assert raised.value.code == "model_unavailable"
    assert {model.engine_model: model.available for model in adapter.capabilities()}["i2v_2_2"] is False
    install(tmp_path / "wan2gp", "i2v_2_2", "int8")
    adapter.prepare("i2v_2_2", "int8")
    assert {model.engine_model: model.available for model in adapter.capabilities()}["i2v_2_2"] is True


def test_the_text_encoder_must_be_installed_in_the_variant_wan2gp_loads(fake_api, wgp, tmp_path):
    adapter = checkout(tmp_path)
    install(tmp_path / "wan2gp", "z_image", "bf16", text_encoder="bf16")
    with pytest.raises(EngineError) as raised:  # Wan2GP loads the int8 text encoder
        adapter.prepare("z_image", "bf16")
    assert raised.value.code == "model_unavailable"
    wgp.text_encoder_quantization = "bf16"  # the host's variant, which the model store writes into wgp_config.json
    adapter.prepare("z_image", "bf16")


def test_prepare_refuses_an_int8_that_wan2gp_would_load_from_the_bf16_file(fake_api, wgp, tmp_path):
    adapter = checkout(tmp_path, ("z_image", "int8"), ("z_image", "bf16"))
    adapter.prepare("z_image", "int8")
    wgp.quantized_kernels = False  # without quanto's int8 kernels Wan2GP falls back to the first file, bf16
    with pytest.raises(EngineError) as raised:
        adapter.prepare("z_image", "int8")
    assert raised.value.code == "model_unavailable"
    assert "cannot select the int8 file of z_image and would load ZImageTurbo_bf16.safetensors" in raised.value.message


def test_prepare_checks_files_with_the_dtype_policy_wan2gp_loads_with(fake_api, wgp, tmp_path):
    adapter = checkout(tmp_path, ("i2v_2_2", "int8"))  # the mbf16 int8 files the registry pins
    wgp.server_config["transformer_dtype_policy"] = "fp16"  # Wan2GP's availability would check the mfp16 files ...
    adapter.prepare("i2v_2_2", "int8")  # ... but load_models uses the module global, still automatic (bf16 here)
    wgp.transformer_dtype_policy = "fp16"  # --fp16: load_models picks the mfp16 files, which the registry does not pin
    with pytest.raises(EngineError) as raised:
        adapter.prepare("i2v_2_2", "int8")
    assert raised.value.code == "model_unavailable"
    assert "wan2.2_image2video_14B_high_quanto_mfp16_int8.safetensors" in raised.value.message
    wgp.transformer_dtype_policy, wgp.bf16_supported = "", False  # the automatic policy on a GPU without bf16
    with pytest.raises(EngineError):
        adapter.prepare("i2v_2_2", "int8")


def test_a_gguf_model_type_runs_without_an_override(fake_api, wgp, tmp_path):
    adapter = checkout(tmp_path, ("ltx2_22B_distilled_gguf_q6_k", "Q6_K"))
    adapter.submit(make_job(tmp_path / "job", "ltx2_22B_distilled_gguf_q6_k", capability="t2v", quant="Q6_K"))
    assert fake_api[0].submitted[0][0]["overrides"] == ""  # Wan2GP loads the type's one file


# ---------------------------------------------------------------------- image mode


def test_a_t2i_job_on_a_video_model_runs_in_image_mode_and_returns_a_jpeg(fake_api, tmp_path):
    adapter = checkout(tmp_path, ("ltx2_22B_distilled", "int8"))
    params = {"resolution": "1280x704", "steps": 8, "frames": 121, "fps": 24, "duration_s": 5.0}
    job = make_job(tmp_path / "job", "ltx2_22B_distilled", capability="t2i", quant="int8", extra={"video_length": 17}, **params)
    (tmp_path / "job").mkdir()

    engine_job_id = adapter.submit(job)

    settings, handle = fake_api[0].submitted[0]
    assert settings["image_mode"] == 1
    # Not the job's frames, fps and duration: Wan2GP renders its minimum frame count at 1 fps and keeps the first.
    assert (settings["video_length"], settings["force_fps"], settings["duration_seconds"]) == (241, "", 0)
    assert (settings["resolution"], settings["num_inference_steps"]) == ("1280x704", 8)
    assert settings["overrides"] == "transformer_quantization='int8'"
    assert adapter.status(engine_job_id).accepted_extra == ()
    produced = tmp_path / "engine" / "2026-11-25-12h00m00s_seed1234_A lighthouse at dusk.jpg"  # image_output_codec jpeg_95
    produced.parent.mkdir()
    produced.write_bytes(b"jpeg bytes")
    handle.finish(success=True, files=[str(produced)])
    assert wait_state(adapter, engine_job_id, ("succeeded", "failed")).state == "succeeded"
    assert [(o.name, o.media_type) for o in adapter.fetch_outputs(engine_job_id)] == [(produced.name, "image/jpeg")]


@pytest.mark.parametrize(("default", "expected"), [(0, 1), (1, 1), (2, 2)])
def test_image_mode_is_set_only_where_wan2gp_defaults_to_video(tmp_path, default, expected):
    settings = build_settings({**DEFAULTS, "image_mode": default}, make_job(tmp_path, "z_image", capability="t2i"))
    assert settings["image_mode"] == expected
    assert build_settings({**DEFAULTS, "image_mode": default}, make_job(tmp_path))["image_mode"] == default  # i2v


def test_an_i2i_job_hands_its_image_to_wan2gp_as_the_control_image(fake_api, tmp_path, monkeypatch):
    adapter = checkout(tmp_path, ("ltx2_22B_distilled", "bf16"))
    (tmp_path / "in.png").write_bytes(render_png(5, 4, 4))
    monkeypatch.chdir(tmp_path)  # a relative input path; Wan2GP generates from its own root
    job = make_job(tmp_path / "job", "ltx2_22B_distilled", ("in.png",), capability="i2i", extra={"denoising_strength": 0.6})

    engine_job_id = adapter.submit(job)

    settings, _ = fake_api[0].submitted[0]  # the fake drops image_guide unless "V" enables it, as Wan2GP does
    assert (settings["image_mode"], settings["image_guide"], settings["video_prompt_type"]) == (1, str(tmp_path / "in.png"), "VG")
    assert settings["image_start"] is None and "S" not in settings["image_prompt_type"]
    assert settings["denoising_strength"] == 0.6
    assert adapter.status(engine_job_id).accepted_extra == ("denoising_strength",)


@pytest.mark.parametrize(("capability", "inputs"), [("i2i", ()), ("i2i", ("a.png", "b.png")), ("t2i", ("a.png",))])
def test_image_jobs_take_the_inputs_their_capability_names(tmp_path, capability, inputs):
    job = make_job(tmp_path, "ltx2_22B_distilled", tuple(str(tmp_path / name) for name in inputs), capability=capability)
    with pytest.raises(EngineError) as raised:
        build_settings(dict(DEFAULTS), job)
    assert raised.value.code == "unsupported_input"


def test_an_image_job_that_returns_a_video_fails(fake_api, adapter, tmp_path):
    engine_job_id = adapter.submit(make_job(tmp_path / "job", "z_image", capability="t2i"))
    produced = tmp_path / "engine" / "clip.mp4"
    produced.parent.mkdir()
    produced.write_bytes(b"mp4 bytes")
    (tmp_path / "job").mkdir()
    fake_api[0].submitted[0][1].finish(success=True, files=[str(produced)])
    status = wait_state(adapter, engine_job_id, ("failed", "succeeded"))
    assert (status.state, status.error_code) == ("failed", "engine_failed")
    assert "clip.mp4 for an image job" in status.error_message


# ---------------------------------------------------------------------- params.extra


ADAPTER_SETTINGS = {
    "overrides": "transformer_quantization='bf16'",
    "image_mode": 2,
    "image_guide": "C:/Users/victim/secret.png",
    "video_prompt_type": "PVG",
    "image_prompt_type": "S",
}


def test_extra_cannot_set_the_settings_only_the_adapter_sets(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="cinewright_worker")
    image_job = make_job(tmp_path, "ltx2_22B_distilled", capability="t2i", quant="int8", extra={**ADAPTER_SETTINGS, "flow_shift": 3.0})
    settings = build_settings(dict(DEFAULTS), image_job)
    assert {key: settings[key] for key in ADAPTER_SETTINGS} == {
        "overrides": "transformer_quantization='int8'",
        "image_mode": 1,
        "image_guide": None,
        "video_prompt_type": "",
        "image_prompt_type": "",
    }
    assert accepted_extra(dict(DEFAULTS), image_job) == ("flow_shift",)
    video_job = make_job(tmp_path, "ltx2_22B_distilled_gguf_q6_k", capability="t2v", quant="Q6_K", extra=ADAPTER_SETTINGS)
    settings = build_settings(dict(DEFAULTS), video_job)  # neither an override nor image mode of its own
    assert {key: settings[key] for key in ADAPTER_SETTINGS} == {key: DEFAULTS[key] for key in ADAPTER_SETTINGS}
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert all(key in logged for key in ADAPTER_SETTINGS) and "secret" not in logged


def test_accepted_extra_lists_the_extras_wan2gp_runs_as_asked(tmp_path):
    extra = {
        "flow_shift": 7.0,
        "denoising_strength": 0.6,  # Wan2GP resets it without a control image
        "video_length": 17,
        "guidance_phases": 3,  # Wan 2.2 allows three
        "model_switch_phase": 2,  # counts only with three phases
        "num_inference_steps": 30,  # the job's steps win
        "negative_prompt": "noise",  # the job's negative prompt wins
        "seed": 5,  # the job's resolved seed wins
        "sample_solver": "C:/x",  # not a plain token
        "not_a_setting": 1,  # not in Wan2GP's defaults
    }
    model_def = WGP_DEFS["i2v_2_2"]
    video = make_job(tmp_path, steps=8, extra=extra)
    image = dataclasses.replace(video, capability="t2i")  # image mode forces the frame count
    no_negative = dataclasses.replace(video, negative_prompt=None)
    for job, expected in (
        (video, ("flow_shift", "guidance_phases", "model_switch_phase", "video_length")),
        (image, ("flow_shift", "guidance_phases", "model_switch_phase")),
        (no_negative, ("flow_shift", "guidance_phases", "model_switch_phase", "negative_prompt", "video_length")),
    ):
        assert accepted_extra(defaults_of("i2v_2_2"), job, model_def) == expected
        ran = validate(build_settings(defaults_of("i2v_2_2"), job, model_def=model_def), FakeWgp())  # as Wan2GP runs it
        assert {key for key, value in extra.items() if ran.get(key) == value} == set(expected)


LOCKED = {**WGP_DEFS["ltx2_22B_distilled"], "lock_guidance_phases": True}


@pytest.mark.parametrize(
    ("model_type", "model_def", "extra", "expected"),
    [
        ("ltx2_22B_distilled", None, {"guidance_phases": 1}, ("guidance_phases",)),
        ("ltx2_22B_distilled", None, {"guidance_phases": 3, "model_switch_phase": 2}, ()),  # LTX-2 allows two phases
        ("ltx2_22B_distilled", LOCKED, {"guidance_phases": 1}, ()),  # a locked model always runs its maximum
        ("z_image", None, {"guidance_phases": 1}, ()),  # Z-Image Turbo has no guidance phases
        ("i2v_2_2", None, {"guidance_phases": 5, "model_switch_phase": 2}, ()),  # so the job keeps its default two
        ("i2v_2_2", None, {"guidance_phases": "3"}, ()),  # Wan2GP compares a number with the maximum
        ("i2v_2_2", None, {"denoising_strength": 1.0}, ("denoising_strength",)),  # what Wan2GP runs anyway
    ],
)
def test_extras_wan2gp_would_change_are_left_out(tmp_path, model_type, model_def, extra, expected):
    model_def = model_def or WGP_DEFS[model_type]
    job = make_job(tmp_path, model_type, capability="t2v", extra=extra)
    assert accepted_extra(defaults_of(model_type), job, model_def) == expected
    ran = validate(build_settings(defaults_of(model_type), job, model_def=model_def), FakeWgp(), model_def)
    assert {key for key in extra if ran[key] == extra[key]} == set(expected)


# ---------------------------------------------------------------------- out of memory and prompts


PASS_1 = [
    progress("loading_model", None, None, "Loading - transformer"),
    progress("inference", 0, 8, "Denoising First Phase"),
    progress("inference", 7, 8, "Denoising First Phase"),
    progress("decoding", 8, 8, "VAE Decoding"),  # Wan2GP's label for the update after the last step of a pass
]
PASS_2 = [
    progress("inference", 0, 3, "Denoising Second Phase"),
    progress("inference", 2, 3, "Denoising Second Phase"),
    progress("inference", 5, 9, "Denoising Second Phase", unit="tiles"),  # tiled fusion counts tiles, not steps
    progress("inference", 12, 40, "Saving File ...", unit="blocks"),  # a model reload under a stale phase
]


@pytest.mark.parametrize(
    ("updates", "where"),
    [
        ([], "before the first denoising step"),
        (PASS_1[:3], "at step 7 of 8 (Denoising First Phase)"),
        (PASS_1, "after the last step of Denoising First Phase (step 8 of 8)"),  # LTX-2 distilled: before its second pass
        (PASS_1 + PASS_2[:1], "before the first step of Denoising Second Phase"),
        (PASS_1 + PASS_2, "at step 2 of 3 (Denoising Second Phase)"),
        (PASS_1 + PASS_2 + [progress("decoding", 3, 3, "VAE Decoding")], "after the last step of Denoising Second Phase (step 3 of 3)"),
    ],
)
def test_an_oom_names_the_denoising_step_it_reached(fake_api, adapter, tmp_path, updates, where):
    engine_job_id = adapter.submit(make_job(tmp_path))
    handle = fake_api[0].submitted[0][1]
    for update in updates:
        handle.events.put("progress", update)
    handle.finish(success=False, errors=[WAN2GP_VRAM_ERROR])
    status = wait_state(adapter, engine_job_id, ("failed", "succeeded"))
    assert status.error_code == "oom"
    assert status.error_message == f"Out of memory {where}. Wan2GP: {WAN2GP_VRAM_ERROR}"


def test_oom_messages_without_a_pass_name():
    assert oom_message(DenoisingStep(4, 8, ""), "x") == "Out of memory at step 4 of 8. Wan2GP: x"
    completed = DenoisingStep(8, 8, "", completed=True)
    assert oom_message(completed, "x") == "Out of memory after the last denoising step (step 8 of 8). Wan2GP: x"


def test_the_prompt_reaches_wan2gp_as_the_core_translated_it(fake_api, adapter, tmp_path):
    assert adapter.translate_prompt("A lighthouse at dusk", "z_image") == "A lighthouse at dusk"
    adapter.submit(make_job(tmp_path, "z_image", capability="t2i"))
    assert fake_api[0].submitted[0][0]["prompt"] == "A lighthouse at dusk"


# ---------------------------------------------------------------------- with the worker loop


def test_the_worker_uploads_an_image_jobs_jpeg_and_reports_the_extras_passed_on(fake_api, tmp_path):
    adapter = checkout(tmp_path, ("ltx2_22B_distilled", "int8"))
    core = FakeCore()  # validates every request against the core's contract and upload media types
    work = tmp_path / "work"
    work.mkdir()
    try:
        # Wan2GP resets denoising_strength without a control image; image mode renders its own frame count.
        extra = {"flow_shift": 4.0, "denoising_strength": 0.5, "video_length": 17}
        params = {"steps": 8, "frames": 121, "fps": 24, "extra": extra}
        job = worker_job(engine_model="ltx2_22B_distilled", params=params).model_copy(update={"quant": "int8"})
        assert job.capability == "t2i"
        core.queue.append(job)
        _, thread, result = start_loop(core, work, adapter=adapter, once=True)
        deadline = time.monotonic() + 10
        while not (fake_api and fake_api[0].submitted):
            assert time.monotonic() < deadline, "the worker did not submit the job"
            time.sleep(0.01)
        settings, handle = fake_api[0].submitted[0]
        assert (settings["image_mode"], settings["overrides"]) == (1, "transformer_quantization='int8'")
        produced = tmp_path / "engine" / "image.jpg"
        produced.parent.mkdir()
        produced.write_bytes(b"jpeg bytes")
        handle.finish(success=True, files=[str(produced)])
        wait_closed(core, thread)
    finally:
        core.shutdown()
    assert result == [0] and core.violations == [] and core.failed == []
    completion = core.completed[0]
    assert [(output.name, output.media_type) for output in completion.outputs] == [("image.jpg", "image/jpeg")]
    assert core.uploads["image.jpg"] == ("image/jpeg", b"jpeg bytes")
    assert completion.accepted_extra == ["flow_shift"]


# ---------------------------------------------------------------------- live probe (opt-in)

LIVE_ROOT = os.environ.get("CINEWRIGHT_WAN2GP_ROOT")
LIVE_PYTHON = os.environ.get("CINEWRIGHT_WAN2GP_PYTHON")
SENTINEL = "@@CINEWRIGHT_PROBE@@"
PROBE_SCRIPT = """
import json, sys, time
sys.path.insert(0, sys.argv[1])
import cinewright_worker.__main__, cinewright_worker.client, cinewright_worker.loop, cinewright_worker.engines.mock
from cinewright_worker.engines import EngineError
from cinewright_worker.engines.wan2gp import Wan2GPAdapter
adapter = Wan2GPAdapter(root=sys.argv[2], data_dir=sys.argv[3])
before = adapter.health().state
started = time.monotonic()
try:
    adapter.capabilities()
except EngineError as exc:  # e.g. engine_unavailable: the checkout has no provisioned ffmpeg_bins
    print(sys.argv[4] + json.dumps({"error": exc.code, "message": exc.message}))
    raise SystemExit(0)
first = time.monotonic() - started
started = time.monotonic()
models = adapter.capabilities()
listing = time.monotonic() - started
health = adapter.health()
print(sys.argv[4] + json.dumps({
    "python": sys.version.split()[0],
    "health_before_init": before,
    "health": {"state": health.state, "engine_version": health.engine_version,
               "vram_used_mb": health.vram_used_mb, "vram_total_mb": health.vram_total_mb},
    "model_count": len(models),
    "available": sorted(m.engine_model for m in models if m.available),
    "mvp": {m.engine_model: [m.name, m.available, list(m.outputs)] for m in models
            if m.engine_model in ("ltx2_22B_distilled", "ltx2_22B_distilled_gguf_q6_k", "i2v_2_2", "t2v_2_2", "z_image")},
    "seconds_init_and_capabilities": round(first, 1),
    "seconds_capabilities": round(listing, 2),
}))
"""


@pytest.mark.skipif(not (LIVE_ROOT and LIVE_PYTHON), reason="set CINEWRIGHT_WAN2GP_ROOT and CINEWRIGHT_WAN2GP_PYTHON")
def test_wan2gp_live_probe(tmp_path):
    src = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    completed = subprocess.run(  # noqa: S603 - interpreter and checkout are named by the developer
        [LIVE_PYTHON, "-I", "-c", PROBE_SCRIPT, str(src), LIVE_ROOT, str(tmp_path / "data"), SENTINEL],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith(SENTINEL)]
    assert completed.returncode == 0 and lines, completed.stderr[-4000:] + completed.stdout[-2000:]
    result = json.loads(lines[-1].removeprefix(SENTINEL))
    print(json.dumps(result, indent=1))
    assert "error" not in result, f"{result.get('error')}: {result.get('message')}"
    assert result["health_before_init"] == "starting"
    assert result["health"]["state"] == "ready"
    assert result["health"]["engine_version"]
    assert {"ltx2_22B_distilled", "i2v_2_2", "t2v_2_2", "z_image"} <= set(result["mvp"])
    assert (tmp_path / "data" / "engines" / "wan2gp" / "wgp_config.json").exists()

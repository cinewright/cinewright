"""Wan2GP adapter against a fake `shared.api` module (no Wan2GP, no GPU), plus an optional live probe.

The live probe runs only when CINEWRIGHT_WAN2GP_ROOT (a Wan2GP checkout) and CINEWRIGHT_WAN2GP_PYTHON
(its Python) are set. It imports the adapter in that interpreter and calls capabilities() and
health() with Hugging Face access switched off; it never generates and never loads model weights.
"""

from __future__ import annotations

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
from pathlib import Path
from types import SimpleNamespace

import pytest

from cinewright_worker.adapter import EngineAdapter, EngineJob
from cinewright_worker.engines import EngineError
from cinewright_worker.engines.mock import render_png
from cinewright_worker.engines.wan2gp import FFMPEG_BINARIES, FFMPEG_DIR, Wan2GPAdapter, build_settings, is_oom

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
    "flow_shift": 5.0,
    "output_filename": "",
    "image_start": None,
    "image_end": None,
}
MODEL_DEFS = [
    {"model_type": "z_image", "name": "Z-Image Turbo 6B", "metadata": {"main_output": ["image"], "outputs": ["image"]}},
    {
        "model_type": "ltx2_22B_distilled",
        "name": "LTX-2 2.3 Distilled 1.0 22B",
        "metadata": {"main_output": ["video"], "outputs": ["video", "audio"]},
    },
    {"model_type": "i2v_2_2", "name": "Wan2.2 Image2video 14B", "metadata": {"main_output": ["video"], "outputs": ["video"]}},
]


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


class FakeSession:
    def __init__(self, init_kwargs: dict) -> None:
        self.init_kwargs = init_kwargs
        self.hf_hub_offline = os.environ.get("HF_HUB_OFFLINE")
        self.available = {"z_image": True, "ltx2_22B_distilled": False, "i2v_2_2": True}
        self.submitted: list[tuple[dict, FakeJob]] = []

    def list_model_defs(self) -> list[dict]:
        return [dict(definition) for definition in MODEL_DEFS]

    def list_model_availability(self) -> list[dict]:
        return [self.get_model_availability(d["model_type"]) for d in MODEL_DEFS]

    def get_model_def(self, model_type: str) -> dict | None:
        return next((dict(d) for d in MODEL_DEFS if d["model_type"] == model_type), None)

    def get_model_availability(self, model_type: str) -> dict:
        available = self.available[model_type]
        return {"model_type": model_type, "status": "available" if available else "missing", "available": available}

    def get_default_settings(self, model_type: str) -> dict:
        return {**DEFAULTS, "model_type": model_type}

    def submit_task(self, settings: dict) -> FakeJob:
        job = FakeJob()
        self.submitted.append((settings, job))
        return job


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


@pytest.fixture
def fake_api(monkeypatch):
    sessions: list[FakeSession] = []
    api = types.ModuleType("shared.api")

    def init(**kwargs) -> FakeSession:
        sessions.append(FakeSession(kwargs))
        return sessions[-1]

    api.init = init
    monkeypatch.setitem(sys.modules, "shared", types.ModuleType("shared"))
    monkeypatch.setitem(sys.modules, "shared.api", api)
    monkeypatch.setitem(sys.modules, "wgp", SimpleNamespace(WanGP_version="19.292"))
    monkeypatch.setitem(sys.modules, "torch", None)  # no torch in the core venv: VRAM figures stay null
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")  # restored afterwards; the adapter sets it to 1
    return sessions


def provision_ffmpeg(root: Path, names: tuple[str, ...] = FFMPEG_BINARIES) -> None:
    (root / FFMPEG_DIR).mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / FFMPEG_DIR / name).write_bytes(b"")


@pytest.fixture
def adapter(tmp_path) -> Wan2GPAdapter:
    provision_ffmpeg(tmp_path / "wan2gp")
    return Wan2GPAdapter(root=str(tmp_path / "wan2gp"), data_dir=str(tmp_path / "data"))


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(f"network access attempted: {args!r}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def make_job(output_dir: Path, engine_model: str = "i2v_2_2", inputs: tuple[str, ...] = (), **params) -> EngineJob:
    return EngineJob(
        job_id="job-1",
        engine_model=engine_model,
        quant="bf16",
        prompt="A lighthouse at dusk",
        negative_prompt="blurry",
        seed=1234,
        params=params,
        input_paths=inputs,
        output_dir=str(output_dir),
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
    assert set(models) == {"z_image", "ltx2_22B_distilled", "i2v_2_2"}
    assert models["ltx2_22B_distilled"].name == "LTX-2 2.3 Distilled 1.0 22B"
    assert models["ltx2_22B_distilled"].outputs == ("video", "audio")
    assert models["ltx2_22B_distilled"].available is False
    assert models["z_image"].available is True
    health = adapter.health()
    assert (health.state, health.engine_version, health.vram_total_mb) == ("ready", "19.292", None)


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
    assert (status.state, status.error_code, status.error_message) == ("failed", "oom", WAN2GP_VRAM_ERROR)


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

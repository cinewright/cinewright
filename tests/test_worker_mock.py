"""Mock engine adapter: determinism, PNG validity, cancel and failure paths."""

from __future__ import annotations

import io
import struct
import time
import zlib

import pytest
from PIL import Image

from cinewright_worker.adapter import EngineAdapter, EngineJob
from cinewright_worker.engines import EngineError
from cinewright_worker.engines.mock import MockAdapter, encode_png, fit_resolution, render_png


def make_job(tmp_path, **params) -> EngineJob:
    return EngineJob(
        job_id="job-1",
        engine_model=params.pop("engine_model", "mock-image"),
        quant="bf16",
        prompt="A lighthouse at dusk",
        negative_prompt=None,
        seed=params.pop("seed", 7),
        params=params,
        output_dir=str(tmp_path),
    )


def wait_terminal(adapter: MockAdapter, engine_job_id: str, timeout: float = 5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = adapter.status(engine_job_id)
        if status.state in ("succeeded", "failed", "cancelled"):
            return status
        time.sleep(0.005)
    raise AssertionError("mock job did not finish")


def png_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    chunks, offset = [], 8
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        tag, body = data[offset + 4 : offset + 8], data[offset + 8 : offset + 8 + length]
        (crc,) = struct.unpack(">I", data[offset + 8 + length : offset + 12 + length])
        assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF
        chunks.append((tag, body))
        offset += 12 + length
    return chunks


def test_adapter_implements_the_protocol():
    adapter = MockAdapter()
    assert isinstance(adapter, EngineAdapter)
    assert [(m.engine_model, m.available, m.outputs) for m in adapter.capabilities()] == [
        ("mock-image", True, ("image",)),
        ("mock-video", True, ("video",)),
    ]
    stand_in = MockAdapter(models=("z_image",), engine="wan2gp")
    assert (stand_in.name, stand_in.capabilities()[0].engine_model) == ("wan2gp", "z_image")
    assert adapter.health().state == "ready"
    assert adapter.translate_prompt("a cat", "mock-image") == "a cat"


def test_png_is_valid():
    data = render_png(3, 40, 20)
    chunks = png_chunks(data)
    assert [tag for tag, _ in chunks] == [b"IHDR", b"IDAT", b"IEND"]
    width, height, depth, colour, _, _, _ = struct.unpack(">IIBBBBB", chunks[0][1])
    assert (width, height, depth, colour) == (40, 20, 8, 2)
    assert len(zlib.decompress(chunks[1][1])) == height * (1 + width * 3)
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (40, 20) and image.mode == "RGB"
    with pytest.raises(ValueError):
        encode_png(2, 2, b"\x00" * 3)


def test_rendering_is_deterministic_by_seed():
    assert render_png(42, 64, 32) == render_png(42, 64, 32)
    assert render_png(42, 64, 32) != render_png(43, 64, 32)
    assert fit_resolution("1280x720", 256) == (256, 144)
    assert fit_resolution("100x50", 256) == (100, 50)


def test_job_succeeds_with_progress_and_preview(tmp_path):
    adapter = MockAdapter(step_seconds=0.001)
    engine_job_id = adapter.submit(make_job(tmp_path, steps=4, resolution="512x256"))
    status = wait_terminal(adapter, engine_job_id)
    assert (status.state, status.step, status.total_steps, status.percent) == ("succeeded", 4, 4, 100.0)
    assert status.preview_path and status.preview_path.endswith(".png")
    outputs = adapter.fetch_outputs(engine_job_id)
    assert [(o.name, o.media_type) for o in outputs] == [("image.png", "image/png")]
    with open(outputs[0].path, "rb") as fh:
        assert fh.read() == render_png(7, 256, 128)


def test_same_job_gives_same_bytes(tmp_path):
    adapter = MockAdapter(step_seconds=0)
    results = []
    for run in ("a", "b"):
        (tmp_path / run).mkdir()
        engine_job_id = adapter.submit(make_job(tmp_path / run, engine_model="mock-video", seed=11))
        wait_terminal(adapter, engine_job_id)
        with open(adapter.fetch_outputs(engine_job_id)[0].path, "rb") as fh:
            results.append(fh.read())
    assert results[0] == results[1]


def test_cancel(tmp_path):
    adapter = MockAdapter(step_seconds=0.05)
    engine_job_id = adapter.submit(make_job(tmp_path, steps=200))
    adapter.cancel(engine_job_id)
    status = wait_terminal(adapter, engine_job_id)
    assert (status.state, status.error_code) == ("cancelled", "cancelled")
    with pytest.raises(EngineError):
        adapter.fetch_outputs(engine_job_id)


def test_simulated_failure(tmp_path):
    adapter = MockAdapter(step_seconds=0)
    engine_job_id = adapter.submit(make_job(tmp_path, steps=4, extra={"mock_fail": "oom"}))
    status = wait_terminal(adapter, engine_job_id)
    assert (status.state, status.error_code) == ("failed", "oom")


def test_unknown_model_is_refused():
    with pytest.raises(EngineError) as raised:
        MockAdapter().prepare("ltx2_22B_distilled", "bf16")
    assert raised.value.code == "model_unavailable"

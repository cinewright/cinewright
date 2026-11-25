"""Shared fixtures: a temp data directory, a test registry, a fake clock and hardware, a core and a client.

Every test runs with `CINEWRIGHT_HOME` pointing at a temp directory. API tests use the client's
base URL `http://127.0.0.1:8737`, because the core refuses any other `Host`. The router routes only
installed quantizations, so the test core has `test-image` `bf16` installed (`mark_installed`), the
quantization the test workers serve.
"""

from __future__ import annotations

import asyncio
import base64
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from cinewright.api.schemas import (
    ClientKind,
    GpuInfo,
    HardwareProfile,
    JobCompletion,
    JobRequest,
    LeaseRequest,
    ModelDescriptor,
    Problem,
    WorkerEvents,
    WorkerRegistration,
)
from cinewright.app import create_app
from cinewright.config import Config
from cinewright.core.assets import StagedBlob
from cinewright.core.clock import to_iso
from cinewright.core.core import Core
from cinewright.core.registry import Registry

BASE_URL = "http://127.0.0.1:8737"
API = "/api/v1"
PNG = b"\x89PNG\r\n\x1a\n" + b"test-input-image"

TEST_IMAGE: dict[str, Any] = {
    "id": "test-image",
    "name": "Test Image 1B",
    "vendor": "Test",
    "family": "test",
    "capabilities": ["t2i", "i2i"],
    "limits": {"resolutions": ["512x512", "1024x1024"]},
    "license": {"name": "Apache-2.0", "commercial_use": True},
    "quantizations": [
        {
            "id": "bf16",
            "format": "safetensors",
            "files": [
                {
                    "name": "test-image-bf16.safetensors",
                    "url": "https://huggingface.co/test/test-image/bf16",
                    "sha256": "a" * 64,
                    "size_bytes": 1000,
                }
            ],
        },
        {
            "id": "Q8_0",
            "format": "gguf",
            "files": [
                {"name": "test-image-q8.gguf", "url": "https://huggingface.co/test/test-image/q8", "sha256": "b" * 64, "size_bytes": 500}
            ],
        },
    ],
    "defaults": {"steps": 4, "resolution": "512x512"},
    "source": {"provider": "huggingface", "repo": "test/test-image"},
    "engines": [
        {"engine": "wan2gp", "model_type": "test_image", "quant": "bf16"},
        {"engine": "wan2gp", "model_type": "test_image_q8", "quant": "Q8_0"},
    ],
}

TEST_VIDEO: dict[str, Any] = {
    "id": "test-video",
    "name": "Test Video 2B",
    "vendor": "Test",
    "family": "test",
    "capabilities": ["t2v"],
    "limits": {"resolutions": ["832x480"], "max_frames": 81, "fps": [16]},
    "license": {"name": "Test Community License", "commercial_use": None, "geo_exclusions": ["EU"]},
    "quantizations": [
        {
            "id": "bf16",
            "format": "safetensors",
            "files": [{"name": "v.safetensors", "url": "https://huggingface.co/test/test-video/v", "sha256": "c" * 64}],
        }
    ],
    "defaults": {"steps": 10, "frames": 81, "fps": 16, "resolution": "832x480"},
    "source": {"provider": "huggingface", "repo": "test/test-video"},
    "engines": [{"engine": "wan2gp", "model_type": "test_video", "quant": "bf16"}],
}

TEST_MODELS = [ModelDescriptor.model_validate(TEST_IMAGE), ModelDescriptor.model_validate(TEST_VIDEO)]
TEST_INSTALLED = (("test-image", "bf16"),)  # installed in the test core: the quantization of `test_image`


class FakeClock:
    """Starts at a fixed moment and moves 1 ms per reading, so timestamps are unique and ordered."""

    def __init__(self) -> None:
        self.now = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.now += timedelta(milliseconds=1)
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def fake_detect(data_dir: Path, host_id: str = "local") -> HardwareProfile:
    gpu = GpuInfo(
        index=0,
        name="NVIDIA GeForce RTX 5080",
        vram_total_mb=16303,
        driver_version="621.14",
        compute_capability="12.0",
        pcie_link="Gen5 x16",
    )
    return HardwareProfile(
        host_id=host_id,
        os="TestOS 1",
        cpu="Test CPU",
        ram_total_mb=65536,
        disk_free_mb=100_000,
        gpus=[gpu],
        cuda_driver_ok=True,
        detected_at=datetime(2026, 11, 22, tzinfo=UTC),
    )


def assert_problem(response: Any, status: int, problem_type: str | None = None) -> Problem:
    assert response.status_code == status, response.text
    assert response.headers["content-type"].startswith("application/problem+json")
    problem = Problem.model_validate(response.json())
    assert problem.status == status
    if problem_type is not None:
        assert problem.type == problem_type
    return problem


def mark_installed(core: Core, *keys: tuple[str, str]) -> None:
    """Record (model id, quantization) pairs as installed in the core's model store, as a completed install leaves
    them, without files. The router routes only installed quantizations."""
    now = to_iso(core.clock())
    with core.db.transaction() as conn:
        conn.executemany(
            "INSERT INTO installed_models (model_id, quant, state, installed_at) VALUES (?, ?, 'installed', ?) "
            "ON CONFLICT (model_id, quant) DO UPDATE SET state = 'installed', error = NULL",
            [(model_id, quant, now) for model_id, quant in keys],
        )


def stage(core: Core, data: bytes) -> StagedBlob:
    """Bytes as the API layer hands an upload to the core: a complete temporary file in the asset store."""
    writer = core.stage_upload()
    writer.write(data)
    return writer.finish()


def leased_job(core: Core) -> tuple[str, str, str]:
    """Register a worker on the core, submit a t2i job and lease it to that worker: (worker id, job id, lease id).

    Call it outside an event loop; the lease runs in its own.
    """
    registration = WorkerRegistration(host_id="local", engine="wan2gp", models=["test_image"], health={"state": "ready"})
    worker_id = core.register_worker(registration, core.enrollment_token).worker_id
    job_id = core.submit_job(JobRequest.model_validate({"capability": "t2i", "prompt": "a lighthouse"}), ClientKind.UI, None).id
    granted = asyncio.run(core.lease(worker_id, LeaseRequest(wait_seconds=0)))
    assert granted is not None and granted.job.job_id == job_id
    return worker_id, job_id, granted.job.lease_id


def report(core: Core, worker_id: str, job_id: str, lease_id: str, *events: dict[str, Any]) -> None:
    """Post worker events for a leased job, as the worker's events call does."""
    core.worker_events(worker_id, job_id, WorkerEvents(lease_id=lease_id, events=list(events)))


def finish_job(core: Core, worker_id: str, job_id: str, lease_id: str) -> None:
    sha256 = core.store_output(worker_id, job_id, "image.png", "image/png", stage(core, PNG)).sha256
    outputs = [{"sha256": sha256, "media_type": "image/png", "name": "image.png"}]
    core.complete_job(worker_id, job_id, JobCompletion(lease_id=lease_id, outputs=outputs))


def preview_event(frame: int) -> dict[str, Any]:
    return {"kind": "preview", "data": {"media_type": "image/png", "data_base64": base64.b64encode(f"frame {frame}".encode()).decode()}}


def register_worker(client: TestClient, models: tuple[str, ...] = ("test_image",), engine: str = "wan2gp") -> dict[str, str]:
    """Register a worker over HTTP with the enrollment token; returns its auth header."""
    response = client.post(
        f"{API}/worker/register",
        json={"host_id": "local", "engine": engine, "engine_version": "test-1", "models": list(models), "health": {"state": "ready"}},
        headers={"Authorization": f"Bearer {client.app.state.enrollment_token}"},
    )
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "home"
    monkeypatch.setenv("CINEWRIGHT_HOME", str(path))
    monkeypatch.delenv("CINEWRIGHT_HOST", raising=False)
    monkeypatch.delenv("CINEWRIGHT_PORT", raising=False)
    return path


@pytest.fixture
def registry_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "registry"
    directory.mkdir()
    for model in (TEST_IMAGE, TEST_VIDEO):
        (directory / f"{model['id']}.yaml").write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    return directory


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def config(home: Path) -> Config:
    return Config(home=home)


def no_network(request: httpx.Request) -> httpx.Response:
    """The download transport of test cores: model installs never reach the network."""
    return httpx.Response(404)


@pytest.fixture
def core(config: Config, registry_dir: Path, clock: FakeClock) -> Core:
    instance = Core(
        config, registry=Registry.load(registry_dir), detect=fake_detect, clock=clock, download_transport=httpx.MockTransport(no_network)
    )
    mark_installed(instance, *TEST_INSTALLED)
    yield instance
    instance.close()


@pytest.fixture
def client(core: Core) -> TestClient:
    with TestClient(create_app(core), base_url=BASE_URL) as test_client:
        yield test_client

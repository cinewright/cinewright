"""The core side of the generation path (#5): routing to installed quantizations, the prompt dialect and provenance.

The core runs on the bundled registry (the four MVP models) without a network; a worker registered on the core stands
in for the local Wan2GP worker, and installs are rows of `installed_models` (`mark_installed`), without files.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from conftest import API, BASE_URL, PNG, fake_detect, mark_installed, no_network, stage
from fastapi.testclient import TestClient

from cinewright.api.schemas import (
    ClientKind,
    DataClass,
    JobCompletion,
    JobRequest,
    LeaseRequest,
    PromptDialect,
    Provenance,
    WorkerJob,
    WorkerRegistration,
)
from cinewright.app import create_app
from cinewright.core import dialect
from cinewright.core.core import Core
from cinewright.core.registry import Registry

LTX, LTX_TYPE = "ltx-2.3-distilled-22b", "ltx2_22B_distilled"
PROMPT = "A lighthouse on a cliff at dusk, waves breaking below"


@pytest.fixture
def mvp(config, clock) -> Core:
    instance = Core(config, registry=Registry.load(), detect=fake_detect, clock=clock, download_transport=httpx.MockTransport(no_network))
    yield instance
    instance.close()


def register(core: Core, *models: str) -> str:
    registration = WorkerRegistration(
        host_id="local", engine="wan2gp", engine_version="38758d3", models=list(models), health={"state": "ready"}
    )
    return core.register_worker(registration, core.enrollment_token).worker_id


def submit(core: Core, **fields: Any) -> str:
    request = JobRequest.model_validate({"prompt": PROMPT, **fields})
    return core.submit_job(request, ClientKind.UI, None).id


def lease(core: Core, worker_id: str) -> WorkerJob:
    granted = asyncio.run(core.lease(worker_id, LeaseRequest(wait_seconds=0)))
    assert granted is not None
    return granted.job


def complete(core: Core, worker_id: str, job: WorkerJob, accepted_extra: list[str] | None = None) -> Provenance:
    """Upload a stand-in output, complete the job and return its provenance as the asset records it."""
    sha256 = core.store_output(worker_id, job.job_id, "out.png", "image/png", stage(core, PNG + job.job_id.encode())).sha256
    outputs = [{"sha256": sha256, "media_type": "image/png", "name": "out.png"}]
    core.complete_job(worker_id, job.job_id, JobCompletion(lease_id=job.lease_id, outputs=outputs, accepted_extra=accepted_extra))
    return core.get_asset(sha256).provenance


def expected_files(registry: Registry, model_id: str, quant: str, variant: str) -> list[str]:
    """The transformer files of the quantization, then the shared files of every required component with `variant`."""
    model = registry.get(model_id)
    transformer = next(q for q in model.quantizations if q.id == quant).files
    shared = [file for component in registry.requirements(model_id) for file in component.files if file.variant in (None, variant)]
    return [file.sha256 for file in transformer] + [file.sha256 for file in shared]


def test_an_i2v_job_routes_to_the_installed_int8_and_records_what_it_ran_with(mvp):
    mark_installed(mvp, (LTX, "int8"))
    worker_id = register(mvp, LTX_TYPE)
    image = mvp.add_asset(stage(mvp, PNG), "image/png", DataClass.PRIVATE).sha256
    extra = {"denoising_strength": 0.7, "flow_shift": 5}

    job_id = submit(mvp, capability="i2v", model=LTX, inputs=[image], params={"extra": extra})
    assert mvp.get_job(job_id).route.quant == "int8"  # although bf16 comes first in the registry
    job = lease(mvp, worker_id)
    assert (job.quant, job.engine_model) == ("int8", LTX_TYPE)

    # The worker applied one of the two extras and names a key the job does not have, which is not recorded.
    provenance = complete(mvp, worker_id, job, accepted_extra=["denoising_strength", "guidance_phases"])

    assert provenance.model.quant == "int8"
    assert provenance.model.file_sha256 == expected_files(mvp.registry, LTX, "int8", "int8")
    int8_transformer = mvp.registry.get(LTX).quantizations[1].files[0].sha256
    assert provenance.model.file_sha256[0] == int8_transformer
    assert provenance.params.model_dump() == {
        "resolution": "1280x704",
        "steps": 8,
        "frames": 241,
        "fps": 24,
        "guidance": None,
        "extra": {"denoising_strength": 0.7},
    }


def test_provenance_lists_the_shared_files_of_the_hosts_text_encoder_variant(mvp):
    mark_installed(mvp, ("z-image-turbo", "bf16"))
    mvp.db.execute("UPDATE hosts SET text_encoder_variant = 'bf16' WHERE id = 'local'")
    worker_id = register(mvp, "z_image")

    submit(mvp, capability="t2i", model="z-image-turbo")
    provenance = complete(mvp, worker_id, lease(mvp, worker_id))

    files = provenance.model.file_sha256
    assert files == expected_files(mvp.registry, "z-image-turbo", "bf16", "bf16")
    encoders = {file.variant: file.sha256 for c in mvp.registry.requirements("z-image-turbo") for file in c.files if file.variant}
    assert encoders["bf16"] in files and encoders["int8"] not in files


def test_a_job_leased_before_the_worker_restarted_after_a_switch_records_the_previous_variant(mvp, monkeypatch):
    mark_installed(mvp, ("z-image-turbo", "int8"))
    worker_id = register(mvp, "z_image")
    submit(mvp, capability="t2i", model="z-image-turbo")
    job = lease(mvp, worker_id)
    # The switch to bf16 ended while the job ran; the worker restarts only once the job ended (`worker_variant`).
    mvp.db.execute("UPDATE hosts SET text_encoder_variant = 'bf16' WHERE id = 'local'")
    monkeypatch.setattr(mvp.models, "_previous_variant", "int8")

    provenance = complete(mvp, worker_id, job)

    assert provenance.model.file_sha256 == expected_files(mvp.registry, "z-image-turbo", "int8", "int8")


def test_an_image_job_on_a_video_model_records_no_frames_or_fps(mvp):
    mark_installed(mvp, (LTX, "int8"))
    worker_id = register(mvp, LTX_TYPE)

    submit(mvp, capability="t2i", model=LTX, params={"steps": 6})
    provenance = complete(mvp, worker_id, lease(mvp, worker_id), accepted_extra=[])

    params = provenance.params
    assert (params.resolution, params.steps, params.frames, params.fps, params.extra) == ("1280x704", 6, None, None, {})


def test_the_prompt_is_translated_once_per_lease_and_natural_prompts_pass_unchanged(mvp, monkeypatch):
    mark_installed(mvp, ("z-image-turbo", "int8"))
    worker_id = register(mvp, "z_image")
    calls: list[tuple[str, str]] = []
    original = dialect.translate_prompt

    def counted(prompt, prompt_dialect):
        calls.append((prompt, prompt_dialect.style))
        return original(prompt, prompt_dialect)

    monkeypatch.setattr(dialect, "translate_prompt", counted)
    submit(mvp, capability="t2i", model="z-image-turbo")
    assert calls == []  # not at submission: the core translates when it hands the job to a worker

    job = lease(mvp, worker_id)

    assert job.prompt == PROMPT
    assert calls == [(PROMPT, "natural")]


def test_tags_prompts_pass_unchanged_until_the_first_tags_model():
    assert dialect.translate_prompt("a lighthouse, dusk", PromptDialect(style="tags")) == "a lighthouse, dusk"


def test_an_estimate_pinned_to_a_quantization_that_is_not_installed_names_it(mvp):
    mark_installed(mvp, (LTX, "int8"))
    register(mvp, LTX_TYPE)

    estimate = mvp.estimate(JobRequest.model_validate({"capability": "t2v", "prompt": PROMPT, "model": LTX, "quant": "bf16"}))

    assert not estimate.feasible
    assert estimate.reasons == [f"feasibility: {LTX} bf16 is not installed on local"]


def test_an_asset_with_provenance_stored_in_the_0_0_1_format_still_loads(mvp):
    sha256 = "e" * 64
    # A record as release 0.0.1 stored it: no `params`, every field it knew.
    record = {
        "schema": "cinewright.provenance/1",
        "asset": {"sha256": sha256, "media_type": "image/png", "size_bytes": 1234},
        "model": {
            "id": "z-image-turbo",
            "name": "Z-Image Turbo 6B",
            "quant": "bf16",
            "engine": "wan2gp",
            "engine_version": "mock-1",
            "file_sha256": ["1ef8d2c0d15c5727c3e824d93e259669cdf292e9eb4f9deb9a8c56f7245a2fa0"],
        },
        "recipe": None,
        "prompt": "a lighthouse",
        "negative_prompt": None,
        "seed": 1234,
        "loras": [],
        "host": {"id": "local", "name": "Local", "kind": "local", "gpu": None},
        "created_at": "2026-11-22T12:00:00+00:00",
        "parents": [],
        "job_id": "0123456789abcdef0123456789abcdef",
        "client": {"kind": "ui", "agent": None},
    }
    mvp.db.execute(
        "INSERT INTO assets (sha256, media_type, size_bytes, data_class, job_id, provenance, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (sha256, "image/png", 1234, "private", None, json.dumps(record), "2026-11-22T12:00:00+00:00"),
    )

    with TestClient(create_app(mvp), base_url=BASE_URL) as client:
        response = client.get(f"{API}/assets/{sha256}")
        assert response.status_code == 200, response.text
        provenance = response.json()["provenance"]
        assert provenance["params"] is None and provenance["model"]["quant"] == "bf16" and provenance["seed"] == 1234
        assert client.get(f"{API}/assets/{sha256}/provenance").status_code == 200
        assert any(item["sha256"] == sha256 for item in client.get(f"{API}/assets").json()["items"])

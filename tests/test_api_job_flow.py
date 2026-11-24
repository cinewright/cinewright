"""The walking skeleton over HTTP: the test acts as a worker against a real core."""

from __future__ import annotations

import asyncio
import hashlib

from conftest import API, PNG, assert_problem, register_worker, stage

from cinewright.api.schemas import Asset, DataClass, Job, JobEvent, JobState, Lease, Provenance
from cinewright.core.assets import BlobWriter

OUTPUT = b"\x89PNG\r\n\x1a\n" + b"generated-output"
CONTENT_CSP = "sandbox; default-src 'none'"


def read_sse(client, params=None, headers=None) -> list[JobEvent]:
    with client.stream("GET", f"{API}/events", params=params, headers=headers) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        text = "".join(response.iter_text())
    events = []
    for block in text.split("\n\n"):
        lines = block.splitlines()
        data = [line.removeprefix("data: ") for line in lines if line.startswith("data: ")]
        if data:
            event = JobEvent.model_validate_json(data[0])
            assert f"id: {event.job_id}:{event.seq}" in lines
            assert f"event: {event.kind}" in lines
            events.append(event)
    return events


def submit(client, **fields) -> Job:
    response = client.post(f"{API}/jobs", json={"capability": "t2i", "prompt": "a lighthouse at dusk", **fields})
    assert response.status_code == 202, response.text
    return Job.model_validate(response.json())


def lease(client, auth) -> Lease:
    response = client.post(f"{API}/worker/lease", json={"wait_seconds": 0}, headers=auth)
    assert response.status_code == 200, response.text
    return Lease.model_validate(response.json())


def run_job(client, auth, output: bytes = OUTPUT) -> tuple[Lease, str]:
    granted = lease(client, auth)
    job_id, lease_id = granted.job.job_id, granted.job.lease_id
    put = client.put(f"{API}/worker/jobs/{job_id}/outputs/image.png", content=output, headers={**auth, "Content-Type": "image/png"})
    assert put.status_code == 201, put.text
    sha = put.json()["sha256"]
    done = client.post(
        f"{API}/worker/jobs/{job_id}/complete",
        json={
            "lease_id": lease_id,
            "outputs": [{"sha256": sha, "media_type": "image/png", "name": "image.png"}],
            "engine_version": "38758d3",
        },
        headers=auth,
    )
    assert done.status_code == 204, done.text
    return granted, sha


def test_full_job_flow_as_a_worker(client):
    upload = client.post(f"{API}/assets", params={"data_class": "public"}, content=PNG, headers={"Content-Type": "image/png"})
    assert upload.status_code == 201, upload.text
    input_sha = Asset.model_validate(upload.json()).sha256

    auth = register_worker(client)
    job = submit(client, capability="i2i", inputs=[input_sha])
    assert job.state == JobState.QUEUED
    assert job.route.model_id == "test-image"

    granted = lease(client, auth).job
    assert (granted.job_id, granted.engine_model, granted.quant, granted.params.steps) == (job.id, "test_image", "bf16", 4)
    assert granted.params.seed == granted.seed
    assert granted.inputs[0].url == f"{API}/worker/inputs/{input_sha}"  # path only; the worker resolves it against its core URL
    download = client.get(granted.inputs[0].url, headers=auth)
    assert download.status_code == 200 and download.content == PNG

    events = client.post(
        f"{API}/worker/jobs/{job.id}/events",
        json={
            "lease_id": granted.lease_id,
            "events": [
                {"kind": "progress", "data": {"phase": "denoise", "step": 1, "total_steps": 4, "percent": 25}},
                {"kind": "log", "data": {"message": "model loaded"}},
            ],
        },
        headers=auth,
    )
    assert events.status_code == 200 and events.json() == {"cancel_requested": False}
    running = Job.model_validate(client.get(f"{API}/jobs/{job.id}").json())
    assert running.state == JobState.RUNNING
    assert running.progress.percent == 25 and running.started_at is not None

    put = client.put(f"{API}/worker/jobs/{job.id}/outputs/image.png", content=OUTPUT, headers={**auth, "Content-Type": "image/png"})
    assert put.status_code == 201
    assert put.json() == {"sha256": hashlib.sha256(OUTPUT).hexdigest(), "size_bytes": len(OUTPUT)}
    output_sha = put.json()["sha256"]
    done = client.post(
        f"{API}/worker/jobs/{job.id}/complete",
        json={"lease_id": granted.lease_id, "outputs": [{"sha256": output_sha, "media_type": "image/png", "name": "image.png"}]},
        headers=auth,
    )
    assert done.status_code == 204, done.text

    finished = Job.model_validate(client.get(f"{API}/jobs/{job.id}").json())
    assert finished.state == JobState.SUCCEEDED and finished.outputs == [output_sha] and finished.finished_at is not None

    asset = Asset.model_validate(client.get(f"{API}/assets/{output_sha}").json())
    assert (asset.job_id, asset.data_class, asset.media_type) == (job.id, "private", "image/png")  # stricter of request and input
    provenance = asset.provenance
    assert provenance.model.id == "test-image" and provenance.model.quant == "bf16" and provenance.model.engine == "wan2gp"
    assert provenance.model.engine_version == "test-1"  # from the worker registration
    assert provenance.model.file_sha256 == ["a" * 64]
    assert (provenance.prompt, provenance.seed, provenance.parents, provenance.job_id) == (
        "a lighthouse at dusk",
        granted.seed,
        [input_sha],
        job.id,
    )
    assert (provenance.host.id, provenance.host.kind, provenance.host.gpu) == ("local", "local", "NVIDIA GeForce RTX 5080")
    assert provenance.client.kind == "ui"

    sidecar = client.get(f"{API}/assets/{output_sha}/provenance")
    assert sidecar.json()["schema"] == "cinewright.provenance/1"
    assert Provenance.model_validate(sidecar.json()) == provenance
    assert "attachment" in sidecar.headers["content-disposition"]
    content = client.get(f"{API}/assets/{output_sha}/content")
    assert content.content == OUTPUT and content.headers["content-type"] == "image/png"
    assert content.headers["content-security-policy"] == CONTENT_CSP
    assert "content-disposition" not in content.headers

    history = read_sse(client, params={"job_id": job.id})
    assert [e.seq for e in history] == list(range(1, len(history) + 1))
    assert [e.data.get("state") for e in history if e.kind == "state"] == ["queued", "leased", "running", "succeeded"]
    assert [e.kind for e in history] == ["state", "state", "state", "progress", "log", "log", "state"]
    assert history[-1].data["outputs"] == [output_sha]
    replayed = read_sse(client, params={"job_id": job.id}, headers={"Last-Event-ID": f"{job.id}:4"})
    assert [e.seq for e in replayed] == [5, 6, 7]


def test_identical_outputs_of_two_jobs_are_one_asset(client):
    auth = register_worker(client)
    first, second = submit(client, params={"seed": 1}), submit(client, params={"seed": 1})
    _, sha_a = run_job(client, auth)
    _, sha_b = run_job(client, auth)
    assert sha_a == sha_b
    assert client.get(f"{API}/jobs/{second.id}").json()["outputs"] == [sha_a]
    assert client.get(f"{API}/assets/{sha_a}").json()["job_id"] == first.id
    assert len(client.get(f"{API}/assets").json()["items"]) == 1


def test_worker_protocol_refusals(client):
    auth = register_worker(client)
    assert_problem(client.post(f"{API}/worker/lease", json={}, headers={"Authorization": "Bearer nope"}), 401)
    assert_problem(client.post(f"{API}/worker/lease", json={}), 401)
    bad_enrollment = client.post(
        f"{API}/worker/register",
        json={"host_id": "local", "engine": "wan2gp", "models": [], "health": {"state": "ready"}},
        headers={"Authorization": "Bearer wrong"},
    )
    assert_problem(bad_enrollment, 401)

    job = submit(client)
    granted = lease(client, auth).job
    wrong_lease = client.post(
        f"{API}/worker/jobs/{job.id}/events", json={"lease_id": "other", "events": [{"kind": "log", "data": {}}]}, headers=auth
    )
    assert_problem(wrong_lease, 409, "urn:cinewright:problem:conflict")
    other = register_worker(client)
    assert_problem(
        client.put(f"{API}/worker/jobs/{job.id}/outputs/x.png", content=OUTPUT, headers={**other, "Content-Type": "image/png"}), 409
    )
    assert_problem(client.get(f"{API}/worker/inputs/{'0' * 64}", headers=auth), 404)
    bad_progress = client.post(
        f"{API}/worker/jobs/{job.id}/events",
        json={"lease_id": granted.lease_id, "events": [{"kind": "progress", "data": {"percent": 500}}]},
        headers=auth,
    )
    assert_problem(bad_progress, 422)
    forged = client.post(
        f"{API}/worker/jobs/{job.id}/events",
        json={"lease_id": granted.lease_id, "events": [{"kind": "log", "data": {"output": {"name": "image.png"}}}]},
        headers=auth,
    )
    assert_problem(forged, 422)
    not_uploaded = client.post(
        f"{API}/worker/jobs/{job.id}/complete",
        json={"lease_id": granted.lease_id, "outputs": [{"sha256": "f" * 64, "media_type": "image/png", "name": "image.png"}]},
        headers=auth,
    )
    assert_problem(not_uploaded, 409)


def test_failure_and_retry(client):
    auth = register_worker(client)
    job = submit(client)
    granted = lease(client, auth).job
    retry = client.post(
        f"{API}/worker/jobs/{job.id}/fail",
        json={"lease_id": granted.lease_id, "error": {"code": "oom", "message": "Out of memory at step 3", "retryable": True}},
        headers=auth,
    )
    assert retry.status_code == 204
    assert client.get(f"{API}/jobs/{job.id}").json()["state"] == "queued"
    granted = lease(client, auth).job
    final = client.post(
        f"{API}/worker/jobs/{job.id}/fail",
        json={"lease_id": granted.lease_id, "error": {"code": "engine_failed", "message": "The engine crashed at step 2"}},
        headers=auth,
    )
    assert final.status_code == 204
    failed = Job.model_validate(client.get(f"{API}/jobs/{job.id}").json())
    assert failed.state == JobState.FAILED and failed.error.code == "engine_failed"
    assert_problem(client.post(f"{API}/jobs/{job.id}/cancel"), 409)


def test_cancel_queued_and_running_jobs(client):
    auth = register_worker(client)
    queued = submit(client)
    assert client.post(f"{API}/jobs/{queued.id}/cancel").json()["state"] == "cancelled"
    assert client.post(f"{API}/jobs/{queued.id}/cancel").json()["state"] == "cancelled"

    running = submit(client)
    granted = lease(client, auth).job
    assert client.post(f"{API}/jobs/{running.id}/cancel").json()["state"] == "leased"
    ack = client.post(
        f"{API}/worker/jobs/{running.id}/events", json={"lease_id": granted.lease_id, "events": [{"kind": "log", "data": {}}]}, headers=auth
    )
    assert ack.json() == {"cancel_requested": True}
    assert client.post(f"{API}/worker/heartbeat", json={"health": {"state": "busy"}}, headers=auth).json() == {"cancel": [running.id]}
    stopped = client.post(
        f"{API}/worker/jobs/{running.id}/fail",
        json={"lease_id": granted.lease_id, "error": {"code": "cancelled", "message": "Cancelled at step 2"}},
        headers=auth,
    )
    assert stopped.status_code == 204
    assert client.get(f"{API}/jobs/{running.id}").json()["state"] == "cancelled"


def test_job_list_pagination_and_filter(client):
    register_worker(client)
    ids = [submit(client).id for _ in range(3)]
    client.post(f"{API}/jobs/{ids[0]}/cancel")
    page = client.get(f"{API}/jobs", params={"limit": 2}).json()
    rest = client.get(f"{API}/jobs", params={"limit": 2, "cursor": page["next_cursor"]}).json()
    assert [j["id"] for j in page["items"] + rest["items"]] == ids[::-1]
    assert rest["next_cursor"] is None
    assert [j["id"] for j in client.get(f"{API}/jobs", params={"state": "cancelled"}).json()["items"]] == [ids[0]]
    assert_problem(client.get(f"{API}/jobs", params={"cursor": "bogus"}), 422)


def test_completion_cannot_change_the_uploaded_media_type(client):
    auth = register_worker(client)
    job = submit(client)
    granted = lease(client, auth).job
    html = b"<script>fetch('/api/v1/settings')</script>"
    put = client.put(
        f"{API}/worker/jobs/{job.id}/outputs/page.bin", content=html, headers={**auth, "Content-Type": "application/octet-stream"}
    )
    assert put.status_code == 201, put.text
    sha = put.json()["sha256"]

    def complete(media_type: str):
        body = {"lease_id": granted.lease_id, "outputs": [{"sha256": sha, "media_type": media_type, "name": "page.bin"}]}
        return client.post(f"{API}/worker/jobs/{job.id}/complete", json=body, headers=auth)

    assert "application/octet-stream" in assert_problem(complete("text/html"), 409, "urn:cinewright:problem:conflict").detail
    assert_problem(client.get(f"{API}/assets/{sha}"), 404)  # nothing was recorded
    assert complete("application/octet-stream").status_code == 204
    asset = Asset.model_validate(client.get(f"{API}/assets/{sha}").json())
    assert asset.media_type == asset.provenance.asset.media_type == "application/octet-stream"


def test_asset_content_outside_the_allow_list_is_only_a_download(client):
    core = client.app.state.core
    # As an older or buggy core stored it.
    sha = core.add_asset(stage(core, b"<script>alert(1)</script>"), "text/html", DataClass.PRIVATE).sha256
    content = client.get(f"{API}/assets/{sha}/content")
    assert content.status_code == 200 and content.content == b"<script>alert(1)</script>"
    assert content.headers["content-type"] == "application/octet-stream"
    assert content.headers["content-disposition"] == f'attachment; filename="{sha[:12]}.bin"'
    assert content.headers["content-security-policy"] == CONTENT_CSP


def test_mcp_clients_are_recorded_with_their_agent(client):
    auth = register_worker(client)
    mcp = {"X-Cinewright-Client": "mcp", "X-Cinewright-Agent": "Claude Code 2.1"}
    response = client.post(f"{API}/jobs", json={"capability": "t2i", "prompt": "a lighthouse at dusk"}, headers=mcp)
    assert response.status_code == 202, response.text
    job = Job.model_validate(response.json())
    assert (job.client, job.agent) == ("mcp", "Claude Code 2.1")
    _, sha = run_job(client, auth)
    provenance = Asset.model_validate(client.get(f"{API}/assets/{sha}").json()).provenance
    assert provenance.job_id == job.id
    assert (provenance.client.kind, provenance.client.agent) == ("mcp", "Claude Code 2.1")


def test_client_headers_are_validated_and_agent_needs_mcp(client):
    register_worker(client)
    body = {"capability": "t2i", "prompt": "a lighthouse"}
    plain = Job.model_validate(client.post(f"{API}/jobs", json=body).json())
    assert (plain.client, plain.agent) == ("ui", None)
    ui = Job.model_validate(client.post(f"{API}/jobs", json=body, headers={"X-Cinewright-Agent": "spoofed"}).json())
    assert (ui.client, ui.agent) == ("ui", None)
    for headers in (
        {"X-Cinewright-Client": "web"},
        {"X-Cinewright-Client": "mcp", "X-Cinewright-Agent": "x" * 65},
        {"X-Cinewright-Client": "mcp", "X-Cinewright-Agent": "agent<script>"},
        {"X-Cinewright-Client": "mcp", "X-Cinewright-Agent": ""},
    ):
        assert_problem(client.post(f"{API}/jobs", json=body, headers=headers), 422)


def test_uploads_are_stored_off_the_event_loop(client, monkeypatch):
    core = client.app.state.core
    calls: list[tuple[str, bool]] = []

    def checked(name, method):
        def wrapper(*args):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                calls.append((name, False))
            else:
                calls.append((name, True))
            return method(*args)

        return wrapper

    for name in ("stage_upload", "add_asset", "check_output", "store_output"):
        monkeypatch.setattr(core, name, checked(name, getattr(core, name)))
    for name in ("write", "finish"):  # the streamed chunk writes, hashing and fsync
        monkeypatch.setattr(BlobWriter, name, checked(name, getattr(BlobWriter, name)))
    assert client.post(f"{API}/assets", content=PNG, headers={"Content-Type": "image/png"}).status_code == 201
    auth = register_worker(client)
    submit(client)
    run_job(client, auth)
    on_loop = [name for name, loop in calls if loop]
    assert on_loop == []
    assert [name for name, _ in calls] == [
        *("stage_upload", "write", "finish", "add_asset"),
        *("check_output", "stage_upload", "write", "finish", "store_output", "check_output"),  # store_output checks again
    ]

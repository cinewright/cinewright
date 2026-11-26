"""Deleting assets over HTTP (#109): record, file and job output records go together; jobs and provenance stay."""

from __future__ import annotations

import sys

import pytest
from conftest import API, PNG, assert_problem, register_worker, stage

from cinewright.api.schemas import Asset, ClientKind, DataClass, Job, JobRequest, JobState, Lease, WorkerRegistration
from cinewright.core import core as core_module
from cinewright.core.assets import BlobStore
from cinewright.core.errors import Invalid

OUTPUT = b"\x89PNG\r\n\x1a\n" + b"generated-output"
CONFLICT = "urn:cinewright:problem:conflict"
NOT_FOUND = "urn:cinewright:problem:not-found"


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    monkeypatch.setattr(core_module, "ASSET_DELETE_RETRY_S", 0)


def upload(client, data: bytes = PNG) -> str:
    response = client.post(f"{API}/assets", content=data, headers={"Content-Type": "image/png"})
    assert response.status_code == 201, response.text
    return Asset.model_validate(response.json()).sha256


def submit(client, **fields) -> Job:
    response = client.post(f"{API}/jobs", json={"capability": "t2i", "prompt": "a lighthouse at dusk", **fields})
    assert response.status_code == 202, response.text
    return Job.model_validate(response.json())


def lease(client, auth) -> Lease:
    response = client.post(f"{API}/worker/lease", json={"wait_seconds": 0}, headers=auth)
    assert response.status_code == 200, response.text
    return Lease.model_validate(response.json())


def put_output(client, auth, job_id: str, data: bytes = OUTPUT) -> str:
    put = client.put(f"{API}/worker/jobs/{job_id}/outputs/image.png", content=data, headers={**auth, "Content-Type": "image/png"})
    assert put.status_code == 201, put.text
    return put.json()["sha256"]


def complete(client, auth, granted: Lease, sha256: str) -> None:
    outputs = [{"sha256": sha256, "media_type": "image/png", "name": "image.png"}]
    done = client.post(
        f"{API}/worker/jobs/{granted.job.job_id}/complete", json={"lease_id": granted.job.lease_id, "outputs": outputs}, headers=auth
    )
    assert done.status_code == 204, done.text


def run_job(client, auth, data: bytes = OUTPUT, **fields) -> tuple[str, str]:
    """Submit a job, lease it and complete it with `data` as its output: (job id, output hash)."""
    job = submit(client, **fields)
    granted = lease(client, auth)
    assert granted.job.job_id == job.id
    sha256 = put_output(client, auth, job.id, data)
    complete(client, auth, granted, sha256)
    return job.id, sha256


def delete(client, sha256: str):
    return client.delete(f"{API}/assets/{sha256}")


def output_rows(client, sha256: str) -> list[str]:
    return [
        row["job_id"]
        for row in client.app.state.core.db.query("SELECT job_id FROM job_outputs WHERE sha256 = ? ORDER BY job_id", (sha256,))
    ]


def test_delete_removes_the_record_the_file_and_the_job_output_record(client):
    core = client.app.state.core
    auth = register_worker(client)
    job_id, sha256 = run_job(client, auth)
    events_before = core.db.query_one("SELECT COUNT(*) AS n FROM job_events WHERE job_id = ?", (job_id,))["n"]
    assert core.blobs.exists(sha256) and output_rows(client, sha256) == [job_id]

    response = delete(client, sha256)
    assert response.status_code == 204, response.text
    assert response.content == b""

    assert_problem(client.get(f"{API}/assets/{sha256}"), 404, NOT_FOUND)
    assert_problem(client.get(f"{API}/assets/{sha256}/content"), 404, NOT_FOUND)
    assert sha256 not in [item["sha256"] for item in client.get(f"{API}/assets").json()["items"]]
    assert not core.blobs.exists(sha256)
    assert output_rows(client, sha256) == []
    # The job keeps its history but no longer lists the output.
    job = Job.model_validate(client.get(f"{API}/jobs/{job_id}").json())
    assert job.state == JobState.SUCCEEDED and job.outputs == []
    assert core.db.query_one("SELECT COUNT(*) AS n FROM job_events WHERE job_id = ?", (job_id,))["n"] == events_before

    # Deleting is final: a second delete finds nothing.
    assert_problem(delete(client, sha256), 404, NOT_FOUND)


def test_the_same_bytes_can_be_uploaded_again_after_a_delete(client):
    sha256 = upload(client)
    assert delete(client, sha256).status_code == 204
    assert upload(client) == sha256
    assert client.app.state.core.blobs.path(sha256).read_bytes() == PNG
    assert client.get(f"{API}/assets/{sha256}/content").content == PNG


def test_delete_of_an_unknown_asset_is_404_and_of_a_malformed_id_422(client):
    problem = assert_problem(delete(client, "0" * 64), 404, NOT_FOUND)
    assert problem.detail == f"Asset {'0' * 64} does not exist."
    assert_problem(delete(client, "A" * 64), 422)


def test_delete_is_refused_while_a_job_that_has_not_ended_uses_the_asset_as_input(client):
    core = client.app.state.core
    input_sha = upload(client)
    auth = register_worker(client)
    job = submit(client, capability="i2i", inputs=[input_sha])
    assert job.state == JobState.QUEUED

    def refused(state_word: str) -> None:
        problem = assert_problem(delete(client, input_sha), 409, CONFLICT)
        assert problem.detail == (
            f"This asset is an input of job {job.id}, which is still {state_word}, so it was not deleted. "
            "Delete it once that job has ended."
        )
        assert client.get(f"{API}/assets/{input_sha}").status_code == 200
        assert core.blobs.exists(input_sha)

    refused("queued")
    # No test route needs an approval, so the job is set to wait for one.
    core.db.execute("UPDATE jobs SET state = ? WHERE id = ?", (JobState.AWAITING_APPROVAL.value, job.id))
    refused("waiting for approval")
    core.db.execute("UPDATE jobs SET state = ? WHERE id = ?", (JobState.QUEUED.value, job.id))
    granted = lease(client, auth)
    refused("starting")  # leased: the worker downloads its input next
    events = {"lease_id": granted.job.lease_id, "events": [{"kind": "log", "data": {"message": "loading"}}]}
    assert client.post(f"{API}/worker/jobs/{job.id}/events", json=events, headers=auth).status_code == 200
    refused("running")

    complete(client, auth, granted, put_output(client, auth, job.id))
    assert delete(client, input_sha).status_code == 204
    assert not core.blobs.exists(input_sha)


def test_a_derived_asset_still_loads_after_its_parent_was_deleted(client):
    core = client.app.state.core
    input_sha = upload(client)
    auth = register_worker(client)
    _, output_sha = run_job(client, auth, capability="i2i", inputs=[input_sha])
    assert core.db.query("SELECT parent_sha256 FROM asset_parents WHERE asset_sha256 = ?", (output_sha,))[0][0] == input_sha

    assert delete(client, input_sha).status_code == 204

    derived = client.get(f"{API}/assets/{output_sha}")
    assert derived.status_code == 200, derived.text
    assert Asset.model_validate(derived.json()).provenance.parents == [input_sha]  # the provenance stays as it was
    sidecar = client.get(f"{API}/assets/{output_sha}/provenance")
    assert sidecar.status_code == 200 and sidecar.json()["parents"] == [input_sha]
    assert client.get(f"{API}/assets/{output_sha}/content").content == OUTPUT
    assert core.db.query("SELECT * FROM asset_parents WHERE parent_sha256 = ?", (input_sha,)) == []

    # The derived asset can be deleted in turn; its own link rows go with it.
    assert delete(client, output_sha).status_code == 204
    assert core.db.query("SELECT * FROM asset_parents") == []


def test_a_running_job_keeps_its_upload_of_the_same_bytes(client):
    core = client.app.state.core
    auth = register_worker(client)
    first_job, sha256 = run_job(client, auth)
    second = submit(client)
    granted = lease(client, auth)
    assert put_output(client, auth, second.id) == sha256  # identical bytes, not completed yet

    assert delete(client, sha256).status_code == 204
    assert_problem(client.get(f"{API}/assets/{sha256}"), 404)
    assert output_rows(client, sha256) == [second.id]  # the finished job's record went, the running job's stays
    assert core.blobs.exists(sha256)

    complete(client, auth, granted, sha256)
    asset = Asset.model_validate(client.get(f"{API}/assets/{sha256}").json())
    assert asset.job_id == second.id and asset.provenance.job_id == second.id
    assert Job.model_validate(client.get(f"{API}/jobs/{second.id}").json()).outputs == [sha256]
    assert Job.model_validate(client.get(f"{API}/jobs/{first_job}").json()).outputs == []


def fail(client, auth, granted: Lease, code: str = "engine_failed", *, retryable: bool = False) -> None:
    failure = {"lease_id": granted.job.lease_id, "error": {"code": code, "message": "The engine stopped at step 2", "retryable": retryable}}
    failed = client.post(f"{API}/worker/jobs/{granted.job.job_id}/fail", json=failure, headers=auth)
    assert failed.status_code == 204, failed.text


def deleted_while_a_second_job_holds_it(client, auth) -> tuple[str, Lease]:
    """Run a job, let a second one upload the same bytes, then delete the asset: (hash, the second job's lease)."""
    _, sha256 = run_job(client, auth)
    submit(client)
    granted = lease(client, auth)
    assert put_output(client, auth, granted.job.job_id) == sha256
    assert delete(client, sha256).status_code == 204
    assert output_rows(client, sha256) == [granted.job.job_id] and client.app.state.core.blobs.exists(sha256)
    return sha256, granted


def assert_gone(client, sha256: str) -> None:
    assert output_rows(client, sha256) == []
    assert not client.app.state.core.blobs.exists(sha256)
    assert_problem(client.get(f"{API}/assets/{sha256}"), 404, NOT_FOUND)
    assert_problem(delete(client, sha256), 404, NOT_FOUND)


def test_a_job_that_fails_after_a_delete_drops_its_upload_of_the_same_bytes(client):
    auth = register_worker(client)
    sha256, granted = deleted_while_a_second_job_holds_it(client, auth)

    fail(client, auth, granted)
    assert client.get(f"{API}/jobs/{granted.job.job_id}").json()["state"] == "failed"
    assert_gone(client, sha256)


def test_a_job_cancelled_after_a_delete_drops_its_upload_of_the_same_bytes(client):
    auth = register_worker(client)
    sha256, granted = deleted_while_a_second_job_holds_it(client, auth)

    assert client.post(f"{API}/jobs/{granted.job.job_id}/cancel").json()["state"] == "running"  # the upload started it
    fail(client, auth, granted, "cancelled")
    assert client.get(f"{API}/jobs/{granted.job.job_id}").json()["state"] == "cancelled"
    assert_gone(client, sha256)


def test_a_job_whose_lease_expires_for_the_last_time_drops_its_upload(client, clock, monkeypatch):
    core = client.app.state.core
    monkeypatch.setattr(core.scheduler, "max_attempts", 1)
    auth = register_worker(client)
    sha256, granted = deleted_while_a_second_job_holds_it(client, auth)

    clock.advance(core.scheduler.lease_seconds + 1)
    assert core.expire_leases() == 1
    assert client.get(f"{API}/jobs/{granted.job.job_id}").json()["state"] == "failed"
    assert_gone(client, sha256)


def test_a_job_that_is_queued_again_keeps_its_upload(client):
    auth = register_worker(client)
    sha256, granted = deleted_while_a_second_job_holds_it(client, auth)

    fail(client, auth, granted, "oom", retryable=True)
    assert client.get(f"{API}/jobs/{granted.job.job_id}").json()["state"] == "queued"
    assert output_rows(client, sha256) == [granted.job.job_id]
    assert client.app.state.core.blobs.exists(sha256)


def test_a_failed_job_leaves_the_file_of_an_asset_with_the_same_bytes(client):
    core = client.app.state.core
    auth = register_worker(client)
    first_job, sha256 = run_job(client, auth)
    submit(client)
    granted = lease(client, auth)
    assert put_output(client, auth, granted.job.job_id) == sha256

    fail(client, auth, granted)
    assert client.get(f"{API}/assets/{sha256}").status_code == 200
    assert client.get(f"{API}/assets/{sha256}/content").content == OUTPUT
    assert output_rows(client, sha256) == sorted([first_job, granted.job.job_id])

    # The failed job's record no longer keeps the file once the asset is deleted.
    assert delete(client, sha256).status_code == 204
    assert output_rows(client, sha256) == [] and not core.blobs.exists(sha256)


def test_a_job_fails_even_when_the_os_refuses_to_delete_its_upload(client, monkeypatch, caplog):
    core = client.app.state.core
    auth = register_worker(client)
    sha256, granted = deleted_while_a_second_job_holds_it(client, auth)

    def refuse(self, digest: str) -> None:
        raise PermissionError(32, "in use")

    monkeypatch.setattr(BlobStore, "remove", refuse)
    fail(client, auth, granted)
    assert client.get(f"{API}/jobs/{granted.job.job_id}").json()["state"] == "failed"
    assert core.blobs.exists(sha256)
    assert f"Did not delete the file {sha256}" in caplog.text


def test_a_file_the_os_refuses_to_delete_keeps_the_asset_and_its_records(client, monkeypatch):
    core = client.app.state.core
    input_sha = upload(client)
    auth = register_worker(client)
    job_id, output_sha = run_job(client, auth, capability="i2i", inputs=[input_sha])
    attempts: list[str] = []
    remove = BlobStore.remove

    def refuse(self, sha256: str) -> None:
        attempts.append(sha256)
        raise PermissionError(32, "The process cannot access the file because it is being used by another process")

    monkeypatch.setattr(BlobStore, "remove", refuse)
    for sha256 in (input_sha, output_sha):
        problem = assert_problem(delete(client, sha256), 409, CONFLICT)
        assert problem.detail == (
            "The file of this asset is still open, for example in a player in another tab, so it was not deleted. Close it and try again."
        )
    assert attempts == [input_sha] * core_module.ASSET_DELETE_ATTEMPTS + [output_sha] * core_module.ASSET_DELETE_ATTEMPTS
    # Everything was rolled back.
    assert client.get(f"{API}/assets/{input_sha}").status_code == 200
    assert client.get(f"{API}/assets/{output_sha}").status_code == 200
    assert output_rows(client, output_sha) == [job_id]
    assert len(core.db.query("SELECT * FROM asset_parents WHERE parent_sha256 = ?", (input_sha,))) == 1
    assert Job.model_validate(client.get(f"{API}/jobs/{job_id}").json()).outputs == [output_sha]

    monkeypatch.setattr(BlobStore, "remove", remove)
    assert delete(client, output_sha).status_code == 204
    assert not core.blobs.exists(output_sha)


def test_a_file_that_is_released_while_the_delete_retries_is_deleted(client, monkeypatch):
    core = client.app.state.core
    sha256 = upload(client)
    remove = BlobStore.remove
    refusals = 3

    def released_later(self, digest: str) -> None:
        nonlocal refusals
        if refusals:
            refusals -= 1
            raise PermissionError(32, "in use")
        remove(self, digest)

    monkeypatch.setattr(BlobStore, "remove", released_later)
    assert delete(client, sha256).status_code == 204
    assert refusals == 0 and not core.blobs.exists(sha256)
    assert_problem(client.get(f"{API}/assets/{sha256}"), 404)


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows refuses to delete a file another handle holds open")
def test_a_file_held_open_on_windows_is_refused_until_it_is_closed(client):
    core = client.app.state.core
    sha256 = upload(client)
    with core.blobs.path(sha256).open("rb"):  # as a response that still streams the file to a player
        assert_problem(delete(client, sha256), 409, CONFLICT)
        assert client.get(f"{API}/assets/{sha256}").status_code == 200
    assert delete(client, sha256).status_code == 204
    assert not core.blobs.exists(sha256)


def test_a_job_whose_input_was_deleted_after_routing_is_refused(core, monkeypatch):
    sha256 = core.add_asset(stage(core, PNG), "image/png", DataClass.PRIVATE).sha256
    registration = WorkerRegistration(host_id="local", engine="wan2gp", models=["test_image"], health={"state": "ready"})
    core.register_worker(registration, core.enrollment_token)
    request = JobRequest.model_validate({"capability": "i2i", "prompt": "a lighthouse", "inputs": [sha256]})
    planned = core._plan(request)
    assert planned[0].route is not None

    # The deletion lands between the route check and the insert.
    core.delete_asset(sha256)
    monkeypatch.setattr(core, "_plan", lambda _request: planned)
    with pytest.raises(Invalid, match=f"Input asset {sha256} does not exist"):
        core.submit_job(request, ClientKind.UI, None)
    assert core.db.query("SELECT * FROM jobs") == []

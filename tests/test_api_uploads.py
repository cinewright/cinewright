"""Upload checks over HTTP for both upload routes: media types, size limits, empty bodies, the job check, cleanup.

The size-limit tests lower `MAX_UPLOAD_BYTES`, so they never send 2 GiB. Streaming through a real server
(memory, responsiveness, interrupted uploads, upload slots) is covered in `test_upload_streaming.py`.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator

import pytest
from conftest import API, assert_problem, register_worker

from cinewright.api import deps

LIMIT = 4096
ROUTES = ["asset", "output"]


def upload_target(client, route: str) -> tuple[str, str, dict[str, str]]:
    """Method, path and auth headers of an upload; for an output, of a job leased to a fresh worker."""
    if route == "asset":
        return "POST", f"{API}/assets", {}
    auth = register_worker(client)
    submitted = client.post(f"{API}/jobs", json={"capability": "t2i", "prompt": "a lighthouse at dusk"})
    assert submitted.status_code == 202, submitted.text
    leased = client.post(f"{API}/worker/lease", json={"wait_seconds": 0}, headers=auth)
    assert leased.status_code == 200, leased.text
    return "PUT", f"{API}/worker/jobs/{leased.json()['job']['job_id']}/outputs/clip.mp4", auth


def temporary_files(client) -> list:
    return list(client.app.state.core.paths.assets_dir.rglob(".upload-*"))


def chunked(data: bytes, size: int = 1000) -> Iterator[bytes]:
    """A body without Content-Length, so only the limit while streaming can refuse it."""
    for start in range(0, len(data), size):
        yield data[start : start + size]


@pytest.fixture
def staged_uploads(client, monkeypatch) -> list[str]:
    """Records every temporary upload file the core opens, so a test can tell that nothing was written."""
    core = client.app.state.core
    opened: list[str] = []
    original = core.stage_upload

    def stage_upload():
        writer = original()
        opened.append(writer.path.name)
        return writer

    monkeypatch.setattr(core, "stage_upload", stage_upload)
    return opened


@pytest.mark.parametrize("route", ROUTES)
def test_an_upload_at_the_limit_is_stored_and_one_byte_more_is_refused(client, monkeypatch, route):
    monkeypatch.setattr(deps, "MAX_UPLOAD_BYTES", LIMIT)
    method, path, auth = upload_target(client, route)
    headers = {**auth, "Content-Type": "video/mp4"}

    data = os.urandom(LIMIT)
    stored = client.request(method, path, content=chunked(data), headers=headers)
    assert stored.status_code == 201, stored.text
    sha256 = hashlib.sha256(data).hexdigest()
    assert (stored.json()["sha256"], stored.json()["size_bytes"]) == (sha256, LIMIT)
    assert client.app.state.core.blobs.path(sha256).read_bytes() == data

    oversized = os.urandom(LIMIT + 1)
    assert "exceeds" in assert_problem(client.request(method, path, content=chunked(oversized), headers=headers), 422).detail
    assert not client.app.state.core.blobs.exists(hashlib.sha256(oversized).hexdigest())
    assert not temporary_files(client)


@pytest.mark.parametrize("route", ROUTES)
def test_a_declared_size_above_the_limit_is_refused_before_reading(client, monkeypatch, staged_uploads, route):
    monkeypatch.setattr(deps, "MAX_UPLOAD_BYTES", LIMIT)
    method, path, auth = upload_target(client, route)
    response = client.request(method, path, content=b"x" * (LIMIT + 1), headers={**auth, "Content-Type": "video/mp4"})
    assert "exceeds" in assert_problem(response, 422).detail
    assert staged_uploads == []


@pytest.mark.parametrize("route", ROUTES)
def test_an_empty_upload_is_refused_and_leaves_no_file(client, route):
    method, path, auth = upload_target(client, route)
    response = client.request(method, path, content=b"", headers={**auth, "Content-Type": "image/png"})
    assert assert_problem(response, 422).detail == "Empty upload."
    assert not temporary_files(client)


@pytest.mark.parametrize("route", ROUTES)
def test_an_unlisted_media_type_is_refused_before_reading(client, staged_uploads, route):
    method, path, auth = upload_target(client, route)
    response = client.request(method, path, content=b"<html></html>", headers={**auth, "Content-Type": "text/html"})
    assert "text/html" in assert_problem(response, 422).detail
    assert staged_uploads == []


def test_an_output_for_a_job_of_another_worker_writes_nothing_to_disk(client, staged_uploads):
    _, path, _ = upload_target(client, "output")
    other = register_worker(client)
    response = client.put(path, content=b"\x00" * 100, headers={**other, "Content-Type": "video/mp4"})
    assert_problem(response, 409, "urn:cinewright:problem:conflict")
    assert staged_uploads == []
    assert not temporary_files(client)
